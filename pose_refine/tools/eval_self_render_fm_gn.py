#!/usr/bin/env python3
"""Self-render FM-GN upper-bound: render DCFF features at GT pose as reference,
at perturbed pose as query — eliminates student↔DCFF gap (cos ~1.0).

Answers: what can FM-GN achieve with perfect features?
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import add_pose_noise
from feature_extract.train_impl import (
    JointRADIOQueryDataset,
    MapFeatureRenderer,
    TeacherFeatureStore,
    build_all_records,
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    resolve_query_feature_dims,
    safe_torch_load,
    split_records,
)
from pose_refine import apply_pose_delta
from pose_refine.utils.geometry_solver import feature_metric_solve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_bucket(samples: list[dict], bucket_label: str) -> dict:
    init_trans = np.array([s["init_trans_mm"] for s in samples])
    final_trans = np.array([s["final_trans_mm"] for s in samples])
    gains = np.array([s["trans_gain_mm"] for s in samples])
    init_rot = np.array([s["init_rot_deg"] for s in samples])
    final_rot = np.array([s["final_rot_deg"] for s in samples])

    return {
        "bucket": bucket_label,
        "count": len(samples),
        "init_trans_mean_mm": float(np.mean(init_trans)),
        "init_trans_med_mm": float(np.median(init_trans)),
        "final_trans_mean_mm": float(np.mean(final_trans)),
        "final_trans_med_mm": float(np.median(final_trans)),
        "trans_gain_mean_mm": float(np.mean(gains)),
        "trans_gain_med_mm": float(np.median(gains)),
        "trans_gain_pos_frac": float((gains > 0).mean()),
        "init_rot_mean_deg": float(np.mean(init_rot)),
        "init_rot_med_deg": float(np.median(init_rot)),
        "final_rot_mean_deg": float(np.mean(final_rot)),
        "final_rot_med_deg": float(np.median(final_rot)),
    }


@torch.no_grad()
def evaluate_self_render(args: argparse.Namespace) -> dict:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    set_seed(int(args.seed))

    cfg = load_config(args.config)
    cfg["training"]["batch_size"] = 1
    cfg["training"]["num_workers"] = 0

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=False,
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    all_records = build_all_records(
        cfg["dataset"], teacher_store,
        allow_synthetic=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    _train_records, val_records = split_records(all_records, cfg["dataset"])

    dataset = JointRADIOQueryDataset(
        val_records, teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    map_renderer = MapFeatureRenderer(
        cfg, feature_hw=tuple(cfg["dataset"]["feature_hw"]), device=device,
        logger=type("_Logger", (), {"info": staticmethod(lambda *a, **k: None)})(),
    )
    checkpoint = safe_torch_load(args.checkpoint)
    map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
    map_renderer.set_train_mode(False)

    noise_buckets = args.noise_buckets if args.noise_buckets else [
        [0.1, 2.0], [0.25, 5.0], [0.5, 10.0], [1.0, 20.0], [2.0, 30.0],
    ]
    max_samples = int(args.max_samples) if args.max_samples > 0 else None

    all_per_sample = []
    bucket_summaries = []

    for noise_deg, noise_m in noise_buckets:
        per_sample = []
        bucket_count = 0
        for batch in tqdm(loader, desc=f"self-render [{noise_deg}°,{noise_m}m]", leave=False):
            if max_samples and bucket_count >= max_samples:
                break
            bucket_count += 1
            batch = move_batch_to_device(batch, device)
            sample_name = batch["sample_name"][0]
            normalized = map_renderer._normalize_name(sample_name)

            pose_gt = map_renderer.name_to_pose[normalized].to(device)
            intr = map_renderer.name_to_intr[normalized]
            intrinsics = {"fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"]}

            # Render reference features at GT pose
            _fr, ref_fine, _coarse, ref_mask, _alpha, _rgb, ref_depth, _pos = map_renderer._render_pose(
                sample_name, pose_gt.unsqueeze(0) if pose_gt.ndim == 2 else pose_gt,
                require_grad=False,
            )

            # Generate noisy pose and render query features
            np.random.seed(int(args.seed) + hash(normalized) % (2**31))
            noisy_pose_np = add_pose_noise(
                pose_gt.squeeze(0).cpu().numpy(),
                rot_deg=float(noise_deg), trans_m=float(noise_m),
            )
            pose_noisy = torch.from_numpy(noisy_pose_np).float().to(device)

            _fr2, query_fine, _coarse2, query_mask, _alpha2, _rgb2, query_depth, _pos2 = map_renderer._render_pose(
                sample_name, pose_noisy.unsqueeze(0) if pose_noisy.ndim == 2 else pose_noisy,
                require_grad=False,
            )

            # Compute init error
            init_rot, init_trans = _pose_error(pose_noisy, pose_gt)

            # Run FM-GN: ref_fine = reference (GT), query_fine = current (noisy)
            pose_cur = pose_noisy.unsqueeze(0) if pose_noisy.ndim == 2 else pose_noisy

            for _ in range(int(args.outer_iters)):
                # Re-render at current pose
                _fr_i, cur_fine, _coarse_i, cur_mask, _alpha_i, _rgb_i, cur_depth, _pos_i = map_renderer._render_pose(
                    sample_name, pose_cur,
                    require_grad=False,
                )

                fm_ref = ref_fine.float()
                fm_query = cur_fine.float()
                fm_depth = cur_depth.float()
                valid_mask = cur_mask.bool()

                if fm_depth.ndim == 4 and fm_depth.shape[1] == 1:
                    fm_depth = fm_depth[:, 0]
                elif fm_depth.ndim == 3:
                    pass
                else:
                    fm_depth = fm_depth.squeeze(1)

                # Match resolutions: downsample ref to query size
                if fm_query.shape[-2:] != fm_ref.shape[-2:]:
                    qH, qW = fm_query.shape[-2:]
                    fm_ref = F.interpolate(fm_ref, (qH, qW), mode="bilinear", align_corners=False)
                    fm_depth = F.interpolate(fm_depth.unsqueeze(1), (qH, qW), mode="nearest").squeeze(1)
                    if valid_mask.shape[-2:] != (qH, qW):
                        valid_mask = F.interpolate(valid_mask.float(), (qH, qW), mode="nearest").bool()

                fm_ref = F.normalize(fm_ref, dim=1)
                fm_query = F.normalize(fm_query, dim=1)

                delta_xi, _residual = feature_metric_solve(
                    fm_query, fm_ref, fm_depth, intrinsics,
                    damping=float(args.damping),
                    valid_mask=valid_mask,
                )
                pose_cur = apply_pose_delta(pose_cur.float(), delta_xi.float(), scale=float(args.update_scale))

            final_rot, final_trans = _pose_error(pose_cur.squeeze(0), pose_gt)

            per_sample.append({
                "sample": str(normalized),
                "noise_deg": noise_deg,
                "noise_m": noise_m,
                "init_rot_deg": float(init_rot),
                "init_trans_mm": float(init_trans * 1000.0),
                "final_rot_deg": float(final_rot),
                "final_trans_mm": float(final_trans * 1000.0),
                "trans_gain_mm": float((init_trans - final_trans) * 1000.0),
            })

        label = f"{noise_deg}deg_{noise_m}m"
        bucket_summaries.append(summarize_bucket(per_sample, label))
        all_per_sample.extend(per_sample)

    init_all = np.array([s["init_trans_mm"] for s in all_per_sample])
    final_all = np.array([s["final_trans_mm"] for s in all_per_sample])
    gains_all = np.array([s["trans_gain_mm"] for s in all_per_sample])

    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "total_samples_per_bucket": len(all_per_sample) // len(noise_buckets),
        "buckets": bucket_summaries,
        "overall": {
            "init_trans_med_mm": float(np.median(init_all)),
            "final_trans_med_mm": float(np.median(final_all)),
            "trans_gain_med_mm": float(np.median(gains_all)),
            "trans_gain_pos_frac": float((gains_all > 0).mean()),
        },
    }


def _pose_error(pose_est: torch.Tensor, pose_gt: torch.Tensor) -> tuple[float, float]:
    """Return (rot_deg, trans_m) for a single pose pair."""
    R_est, t_est = pose_est[..., :3, :3], pose_est[..., :3, 3]
    R_gt, t_gt = pose_gt[..., :3, :3], pose_gt[..., :3, 3]
    if R_est.ndim == 3:
        R_est, t_est = R_est[0], t_est[0]
    if R_gt.ndim == 3:
        R_gt, t_gt = R_gt[0], t_gt[0]
    rot_err = torch.acos(torch.clamp((torch.trace(R_est @ R_gt.T) - 1) / 2, -1, 1))
    trans_err = torch.norm(t_est - t_gt)
    return float(rot_err.item() * 180 / math.pi), float(trans_err.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-render FM-GN upper-bound evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise_buckets", type=json.loads, default=None)
    parser.add_argument("--outer_iters", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--update_scale", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per bucket (0=all)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate_self_render(args)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2))
    if args.output:
        output = Path(args.output)
    else:
        output = Path("/root/ICLPose/result/feature_extract/cpr_baseline_v4/eval_self_render_fm_gn.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
