#!/usr/bin/env python3
"""Per-noise-bucket evaluation of RadioQueryStudent + correlation/WLS refinement.

Based on eval_joint_corr_wls.py but evaluates across multiple noise buckets
and reports per-bucket init→final gain, matching the expert's evaluation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
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
    augment_feature_with_scene_coord,
    build_all_records,
    build_radio_query_student,
    compute_w2c_flow,
    local_correlation_feature_preprocess,
    load_config,
    move_batch_to_device,
    normalize_scene_coord_map,
    pose_error_tensors,
    resolve_query_feature_dims,
    scene_coord_center_scale,
    safe_torch_load,
    shifted_local_correlation,
    split_records,
)
from pose_refine import apply_pose_delta, compute_image_jacobian, diff_pose_solve
from pose_refine.utils.geometry_solver import feature_metric_solve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    radius = int(radius)
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx.reshape(1, -1, 1, 1), yy.reshape(1, -1, 1, 1)


def decode_local_correlation(
    corr: torch.Tensor,
    *,
    radius: int,
    temperature: float = 0.05,
    flow_decode_mode: str = "softargmax",
    confidence_mode: str = "max",
    confidence_margin_temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    radius = int(radius)
    flow_decode = str(flow_decode_mode or "softargmax").lower()
    conf_mode = str(confidence_mode or "max").lower()

    B, C, H, W = corr.shape
    expected = (2 * radius + 1) ** 2
    if C != expected:
        raise ValueError(f"corr has {C} channels, expected {expected} for radius={radius}")

    corr_f = corr.float()
    dx, dy = local_offsets(radius, corr.device, corr_f.dtype)
    probs = torch.softmax(corr_f / max(float(temperature), 1e-6), dim=1)
    soft_flow = torch.cat(
        [(probs * dx).sum(dim=1, keepdim=True), (probs * dy).sum(dim=1, keepdim=True)], dim=1,
    )
    hard_idx = corr_f.argmax(dim=1, keepdim=True)
    hard_flow = torch.cat(
        [dx.expand(B, -1, H, W).gather(1, hard_idx), dy.expand(B, -1, H, W).gather(1, hard_idx)], dim=1,
    )
    if flow_decode == "argmax":
        flow = hard_flow
    elif flow_decode == "argmax_st":
        flow = hard_flow + (soft_flow - soft_flow.detach())
    else:
        flow = soft_flow

    top2 = torch.topk(corr_f, k=2, dim=1).values
    peak_margin = top2[:, 0:1] - top2[:, 1:2]
    if conf_mode == "margin":
        confidence = torch.sigmoid(peak_margin / max(float(confidence_margin_temperature), 1e-6))
    else:
        confidence = probs.max(dim=1, keepdim=True).values

    metrics = {"peak_margin": peak_margin.mean().detach(), "confidence": confidence.mean().detach()}
    return flow.to(dtype=corr.dtype), confidence.to(dtype=corr.dtype), metrics


def apply_confidence_filter(
    confidence: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.0,
    top_fraction: float = 0.0,
    power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    conf = confidence.float().clamp(min=0.0)
    valid = torch.ones_like(conf)
    if valid_mask is not None:
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != conf.shape[-2:]:
            valid = F.interpolate(valid, size=conf.shape[-2:], mode="nearest")
        valid = (valid > 0.0).float()

    if abs(float(power) - 1.0) > 1e-6:
        conf = conf.clamp(max=1.0).pow(float(power))
    if float(threshold) > 0.0:
        conf = conf * (conf >= float(threshold)).float()

    frac = float(top_fraction)
    if 0.0 < frac < 1.0:
        filtered = torch.zeros_like(conf)
        for b in range(conf.shape[0]):
            valid_b = valid[b : b + 1] > 0.0
            values = conf[b : b + 1][valid_b]
            if values.numel() == 0:
                continue
            keep_count = max(1, int(math.ceil(values.numel() * frac)))
            kth = torch.topk(values, k=keep_count, largest=True).values[-1]
            filtered[b : b + 1] = conf[b : b + 1] * (valid_b & (conf[b : b + 1] >= kth)).float()
        conf = filtered

    conf = conf * valid
    coverage = ((conf > 0.0).float() * valid).sum() / valid.sum().clamp(min=1.0)
    return conf, coverage.detach()


def compute_query_map_flow(
    rendered_feat: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    flow_decode_mode: str,
    confidence_mode: str,
    confidence_margin_temperature: float,
    matcher: torch.nn.Module | None = None,
    flow_head: torch.nn.Module | None = None,
    depth: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    feature_preprocess: str = "none",
    highpass_kernel: int = 3,
    highpass_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    query = query_feat.float()
    if query.shape[-2:] != rendered_feat.shape[-2:]:
        query = F.interpolate(query, size=rendered_feat.shape[-2:], mode="bilinear", align_corners=False)
    rendered_corr = local_correlation_feature_preprocess(
        rendered_feat.float(), mode=feature_preprocess, highpass_kernel=highpass_kernel, highpass_scale=highpass_scale,
    )
    query_corr = local_correlation_feature_preprocess(
        query, mode=feature_preprocess, highpass_kernel=highpass_kernel, highpass_scale=highpass_scale,
    )
    rendered_n = F.normalize(rendered_corr, dim=1)
    query_n = F.normalize(query_corr, dim=1)
    corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
    if matcher is not None:
        corr = matcher(corr, depth=depth, valid_mask=valid_mask).float()
    if flow_head is not None:
        pred = flow_head(corr, depth=depth, valid_mask=valid_mask)
        flow = pred["flow"].float()
        confidence = pred.get("confidence")
        if confidence is None:
            confidence = torch.ones(flow.shape[0], 1, flow.shape[2], flow.shape[3], device=flow.device)
        return flow, confidence.float(), {"peak_margin": corr.max(dim=1, keepdim=True).values.mean().detach()}
    flow, confidence, metrics = decode_local_correlation(
        corr, radius=radius, temperature=temperature, flow_decode_mode=flow_decode_mode,
        confidence_mode=confidence_mode, confidence_margin_temperature=confidence_margin_temperature,
    )
    return flow.float(), confidence.float(), metrics


def maybe_project_local_corr_features(
    model: torch.nn.Module, model_cfg: dict, rendered_feat: torch.Tensor, query_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projector = getattr(model, "local_corr_projector", None)
    if projector is None:
        return rendered_feat, query_feat
    rendered_projected = projector(rendered_feat)
    if bool(model_cfg.get("local_corr_projector_apply_to_query", True)):
        query_feat = projector(query_feat)
    return rendered_projected, query_feat


def stable_sample_seed(base_seed: int, sample_name: str) -> int:
    payload = f"{int(base_seed)}:{sample_name}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def make_initial_pose(
    pose_gt: torch.Tensor,
    noise_deg: float,
    noise_m: float,
    device: torch.device,
    *,
    sample_names: list[str] | None = None,
    base_seed: int = 0,
) -> torch.Tensor:
    poses = []
    np_state = np.random.get_state()
    try:
        for idx, pose in enumerate(pose_gt.detach().cpu()):
            if sample_names is not None:
                np.random.seed(stable_sample_seed(int(base_seed), str(sample_names[idx])))
            noisy = add_pose_noise(pose.numpy(), rot_deg=float(noise_deg), trans_m=float(noise_m))
            poses.append(torch.from_numpy(noisy).float())
    finally:
        np.random.set_state(np_state)
    return torch.stack(poses, dim=0).to(device)


@torch.no_grad()
def evaluate_on_bucket(
    model: torch.nn.Module,
    map_renderer: MapFeatureRenderer,
    loader: DataLoader,
    device: torch.device,
    bucket: tuple[float, float],
    args: argparse.Namespace,
    map_cfg: dict,
    model_cfg: dict,
    query_fine_key: str,
) -> dict:
    """Evaluate one noise bucket and return per-sample metrics."""
    noise_deg, noise_m = float(bucket[0]), float(bucket[1])
    radius = int(args.radius if args.radius is not None else map_cfg.get("query_corr_radius", 4))
    temperature = float(args.temperature if args.temperature is not None else map_cfg.get("query_corr_temperature", 0.05))
    flow_decode_mode = str(args.flow_decode_mode or map_cfg.get("query_corr_flow_decode_mode", "softargmax"))
    feature_preprocess = str(map_cfg.get("query_corr_feature_preprocess", "none"))
    highpass_kernel = int(map_cfg.get("query_corr_highpass_kernel", 3))
    highpass_scale = float(map_cfg.get("query_corr_highpass_scale", 1.0))
    query_corr_scene_coord_weight = float(map_cfg.get("query_corr_scene_coord_weight", 0.0))
    scene_center, scene_scale = scene_coord_center_scale(map_cfg, device, dtype=torch.float32)
    use_projector = not args.bypass_projector
    use_matcher = not args.bypass_matcher

    per_sample = []

    for batch in tqdm(loader, desc=f"bucket [{noise_deg}°,{noise_m}m]", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["rgb"])
        query_fine = outputs[query_fine_key].float()
        query_coarse = outputs.get("coarse")
        if query_coarse is not None:
            query_coarse = query_coarse.float()

        pose_gt, intrinsics, normalized_names = [], [], []
        for sample_name in batch["sample_name"]:
            normalized = map_renderer._normalize_name(sample_name)
            normalized_names.append(normalized)
            pose_gt.append(map_renderer.name_to_pose[normalized].to(device))
            intr = map_renderer.name_to_intr[normalized]
            intrinsics.append([intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
        pose_gt = torch.stack(pose_gt, dim=0).float()
        intrinsics_t = torch.tensor(intrinsics, device=device, dtype=torch.float32)

        pose_cur = make_initial_pose(
            pose_gt, noise_deg, noise_m, device,
            sample_names=normalized_names, base_seed=int(args.seed),
        )

        _init_rot_loss, init_rot, init_trans = pose_error_tensors(pose_cur, pose_gt)

        is_corr_mode = (args.refine_mode == "corr_wls")
        is_coarse_to_fine = (args.refine_mode == "coarse_to_fine_fm")
        coarse_iters = int(args.coarse_iters) if is_coarse_to_fine else 0

        for iter_idx in range(int(args.outer_iters)):
            rendered_list, depth_list, valid_list, position_list = [], [], [], []
            rendered_coarse_list = []
            for sample_name, pose in zip(batch["sample_name"], pose_cur):
                _fine_raw, fine, _coarse, mask, _alpha, _rgb, depth, position = map_renderer._render_pose(
                    sample_name, pose, require_grad=False,
                )
                rendered_list.append(fine.squeeze(0))
                depth_list.append(depth.squeeze(0))
                valid_list.append(mask.squeeze(0))
                position_list.append(position.squeeze(0))
                if is_coarse_to_fine and _coarse is not None:
                    rendered_coarse_list.append(_coarse.squeeze(0))
            rendered = torch.stack(rendered_list, dim=0).float()
            depth = torch.stack(depth_list, dim=0).float()
            valid_mask = torch.stack(valid_list, dim=0).float()
            position = torch.stack(position_list, dim=0).float()
            if depth.ndim == 4 and depth.shape[1] == 1:
                depth_s = depth[:, 0]
            elif depth.ndim == 3:
                depth_s = depth
            else:
                depth_s = depth.squeeze(1)

            if is_corr_mode:
                corr_rendered, corr_query = rendered, query_fine
                if query_corr_scene_coord_weight > 0.0 and "scene_coord" in outputs:
                    rendered_scene = normalize_scene_coord_map(
                        position, scene_center, scene_scale, target_hw=rendered.shape[-2:],
                    )
                    corr_rendered = augment_feature_with_scene_coord(rendered, rendered_scene, query_corr_scene_coord_weight)
                    corr_query = augment_feature_with_scene_coord(query_fine, outputs["scene_coord"], query_corr_scene_coord_weight)
                elif use_projector and getattr(model, "local_corr_projector", None) is not None:
                    corr_rendered, corr_query = maybe_project_local_corr_features(model, model_cfg, corr_rendered, corr_query)

                flow, confidence, _corr_metrics = compute_query_map_flow(
                    corr_rendered, corr_query, radius=radius, temperature=temperature,
                    flow_decode_mode=flow_decode_mode, confidence_mode=args.confidence_mode,
                    confidence_margin_temperature=args.confidence_margin_temperature,
                    matcher=getattr(model, "local_matcher", None) if use_matcher else None,
                    flow_head=getattr(model, "local_flow_head", None) if not args.bypass_matcher else None,
                    depth=depth, valid_mask=valid_mask,
                    feature_preprocess=feature_preprocess, highpass_kernel=highpass_kernel, highpass_scale=highpass_scale,
                )
                confidence, _conf_cov = apply_confidence_filter(
                    confidence, valid_mask=valid_mask,
                    threshold=float(args.confidence_threshold),
                    top_fraction=float(args.confidence_top_fraction),
                    power=float(args.confidence_power),
                )

                Ju, Jv, depth_valid = compute_image_jacobian(depth_s.float(), intrinsics_t)
                delta_xi = diff_pose_solve(
                    flow.float(), confidence.float(), Ju, Jv, depth_valid,
                    damping=float(args.damping),
                    irls_iters=int(args.irls_iters),
                    pixel_stride=int(args.pixel_stride),
                    robust_kernel=str(args.robust_kernel),
                )
            else:
                # Feature-metric GN: directly minimize feature residual
                if is_coarse_to_fine and iter_idx < coarse_iters:
                    # Coarse stage: use coarse features for basin expansion
                    fm_query = query_coarse.float() if query_coarse is not None else query_fine.float()
                    fm_rendered = (torch.stack(rendered_coarse_list, dim=0).float()
                                   if rendered_coarse_list else rendered.float())
                else:
                    fm_query = query_fine.float()
                    fm_rendered = rendered.float()
                fm_depth = depth_s.float()

                # Build intrinsics dict from tensor if needed
                if isinstance(intrinsics_t, dict):
                    fm_intrinsics = intrinsics_t
                else:
                    # intrinsics_t is (B,4) tensor [fx, fy, cx, cy]
                    intr = intrinsics_t[0] if intrinsics_t.ndim == 2 else intrinsics_t
                    fm_intrinsics = {"fx": intr[0].item(), "fy": intr[1].item(),
                                     "cx": intr[2].item(), "cy": intr[3].item()}

                # Match resolutions: downsample ref to query size for clean gradients
                if fm_query.shape[-2:] != fm_rendered.shape[-2:]:
                    qH, qW = fm_query.shape[-2:]
                    rH, rW = fm_rendered.shape[-2:]
                    scale_h = qH / rH
                    scale_w = qW / rW

                    fm_rendered = F.interpolate(fm_rendered, (qH, qW), mode="bilinear", align_corners=False)
                    fm_depth = F.interpolate(fm_depth.unsqueeze(1), (qH, qW), mode="nearest").squeeze(1)
                    if valid_mask.shape[-2:] != (qH, qW):
                        valid_mask = F.interpolate(valid_mask.float(), (qH, qW), mode="nearest").bool()

                    # Scale intrinsics to query resolution
                    fm_intrinsics = {
                        k: v * scale_w if k in ('fx', 'cx') else v * scale_h if k in ('fy', 'cy') else v
                        for k, v in fm_intrinsics.items()
                    }
                if use_projector and getattr(model, "local_corr_projector", None) is not None:
                    fm_rendered, fm_query = maybe_project_local_corr_features(model, model_cfg, fm_rendered, fm_query)
                fm_rendered = F.normalize(fm_rendered, dim=1)
                fm_query = F.normalize(fm_query, dim=1)
                delta_xi, _residual = feature_metric_solve(
                    fm_query, fm_rendered, fm_depth, fm_intrinsics,
                    damping=float(args.damping),
                    valid_mask=valid_mask,
                    rot_damping_multiplier=float(args.rot_damping_multiplier),
                    solve_translation_only=bool(args.solve_translation_only),
                )

            pose_cur = apply_pose_delta(pose_cur.float(), delta_xi.float(), scale=float(args.update_scale))

        _final_rot_loss, final_rot, final_trans = pose_error_tensors(pose_cur, pose_gt)

        for idx, sample_name in enumerate(normalized_names):
            per_sample.append({
                "sample": str(sample_name),
                "noise_deg": noise_deg,
                "noise_m": noise_m,
                "init_rot_deg": float(init_rot[idx].item()),
                "init_trans_mm": float((init_trans[idx] * 1000.0).item()),
                "final_rot_deg": float(final_rot[idx].item()),
                "final_trans_mm": float((final_trans[idx] * 1000.0).item()),
                "trans_gain_mm": float(((init_trans[idx] - final_trans[idx]) * 1000.0).item()),
            })

    return per_sample


def summarize_bucket(samples: list[dict], bucket_label: str) -> dict:
    """Compute summary statistics for a bucket."""
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
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    set_seed(int(args.seed))

    cfg = load_config(args.config)
    cfg["training"]["batch_size"] = int(args.batch_size)
    cfg["training"]["num_workers"] = int(args.num_workers)
    checkpoint = safe_torch_load(args.checkpoint)
    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    all_records = build_all_records(
        cfg["dataset"], teacher_store,
        allow_synthetic=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    _train_records, val_records = split_records(all_records, cfg["dataset"])
    records = val_records

    dataset = JointRADIOQueryDataset(
        records, teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
    )
    loader = DataLoader(
        dataset, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=device.type == "cuda",
    )

    model = build_radio_query_student(cfg, fine_feature_dim=fine_dim, coarse_feature_dim=coarse_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    map_renderer = MapFeatureRenderer(
        cfg, feature_hw=tuple(cfg["dataset"]["feature_hw"]), device=device,
        logger=type("_Logger", (), {"info": staticmethod(lambda *a, **k: None)})(),
    )
    map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
    map_renderer.set_train_mode(False)

    query_fine_key = str(cfg.get("map_supervision", {}).get("query_fine_key", "fine"))
    map_cfg = cfg.get("map_supervision", {})
    model_cfg = cfg.get("model", {})

    noise_buckets = args.noise_buckets if args.noise_buckets else [[0.1, 0.05], [0.25, 0.10], [0.5, 0.20], [1.0, 0.50], [2.0, 1.0]]

    all_per_sample = []
    bucket_summaries = []

    for bucket in noise_buckets:
        samples = evaluate_on_bucket(model, map_renderer, loader, device, bucket, args, map_cfg, model_cfg, query_fine_key)
        all_per_sample.extend(samples)
        label = f"{bucket[0]}deg_{bucket[1]}m"
        summary = summarize_bucket(samples, label)
        bucket_summaries.append(summary)

    # Overall summary
    init_all = np.array([s["init_trans_mm"] for s in all_per_sample])
    final_all = np.array([s["final_trans_mm"] for s in all_per_sample])
    gains_all = np.array([s["trans_gain_mm"] for s in all_per_sample])

    result = {
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
    if args.save_per_sample:
        result["per_sample"] = all_per_sample
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-bucket Corr-WLS evaluation for CPR checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise_buckets", type=json.loads, default=None,
                        help='JSON list of [rot_deg, trans_m] pairs, e.g. \'[[0.1,0.05],[1.0,0.5]]\'')
    parser.add_argument("--outer_iters", type=int, default=3)
    parser.add_argument("--refine_mode", choices=["corr_wls", "feature_metric", "coarse_to_fine_fm"], default="corr_wls",
                        help="Refinement: corr_wls, feature_metric, or coarse_to_fine_fm (coarse FM-GN then fine FM-GN)")
    parser.add_argument("--coarse_iters", type=int, default=3, help="Coarse-stage FM-GN iterations for coarse_to_fine_fm")
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--bypass_projector", action="store_true", help="Skip local_corr_projector even if model has one")
    parser.add_argument("--bypass_matcher", action="store_true", help="Skip local_matcher even if model has one")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--flow_decode_mode", default=None)
    parser.add_argument("--confidence_mode", choices=["max", "margin"], default="margin")
    parser.add_argument("--confidence_margin_temperature", type=float, default=0.05)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--confidence_top_fraction", type=float, default=0.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--solve_translation_only", action="store_true",
                        help="Only solve for translation (DoF 0,1,2), keep rotation fixed.")
    parser.add_argument("--rot_damping_multiplier", type=float, default=1.0,
                        help="Multiply damping for rotation DoF (indices 3,4,5). "
                             ">1 reduces rotation drift at cost of slower rot convergence.")
    parser.add_argument("--update_scale", type=float, default=0.5)
    parser.add_argument("--irls_iters", type=int, default=2)
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--robust_kernel", choices=["huber", "gm", "gnc_gm"], default="huber")
    parser.add_argument("--save_per_sample", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
    else:
        stem = Path(args.checkpoint).parents[1].name
        output = Path("/root/ICLPose/result/pose_refine/eval_per_bucket") / f"{stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
