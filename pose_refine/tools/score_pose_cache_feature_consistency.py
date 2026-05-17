#!/usr/bin/env python3
"""Score pose-cache entries by query/render DCFF feature consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import RadioLocDataset, read_colmap_cameras  # noqa: E402
from feature_field import build_dcff_runtime, intrinsics_to_K, render_feature_bundle_batch  # noqa: E402


def cache_stem_for_image_name(image_name: str) -> str:
    path = Path(str(image_name).replace("\\", "/"))
    if path.parent == Path(".") or not path.parent.name:
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def scale_intrinsics(base_intr: Dict[str, float], orig_hw: tuple[int, int], target_hw: tuple[int, int]) -> Dict[str, float]:
    orig_h, orig_w = orig_hw
    h, w = target_hw
    return {
        "fx": float(base_intr["fx"] * w / orig_w),
        "fy": float(base_intr["fy"] * h / orig_h),
        "cx": float(base_intr["cx"] * w / orig_w),
        "cy": float(base_intr["cy"] * h / orig_h),
    }


def compute_feature_consistency_metrics(
    query_features: torch.Tensor,
    rendered_features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Return lower-is-better residual and cosine over valid query/render pixels."""
    if query_features.ndim != 4 or rendered_features.ndim != 4:
        raise ValueError("query_features and rendered_features must be BCHW tensors")
    if query_features.shape[0] != rendered_features.shape[0]:
        raise ValueError("query_features and rendered_features must have the same batch size")
    if query_features.shape[1] != rendered_features.shape[1]:
        raise ValueError("query_features and rendered_features must have the same channel count")
    if query_features.shape[-2:] != rendered_features.shape[-2:]:
        rendered_features = F.interpolate(
            rendered_features,
            size=query_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    query = F.normalize(query_features.float(), dim=1, eps=1e-6)
    rendered = F.normalize(rendered_features.float(), dim=1, eps=1e-6)
    cosine = (query * rendered).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    if valid_mask is None:
        valid = torch.ones_like(cosine, dtype=torch.bool)
    else:
        valid = valid_mask
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != cosine.shape[-2:]:
            valid = F.interpolate(valid.float(), size=cosine.shape[-2:], mode="nearest").bool()
        valid = valid.to(device=cosine.device, dtype=torch.bool)
        if valid.shape[1] != 1:
            valid = valid.any(dim=1, keepdim=True)

    total = int(valid.numel())
    num_valid = int(valid.sum().item())
    valid_frac = float(num_valid / max(total, 1))
    if num_valid == 0:
        return {
            "residual_mean": float("inf"),
            "cosine_mean": 0.0,
            "valid_frac": 0.0,
        }

    cosine_valid = cosine[valid]
    cosine_mean = float(cosine_valid.mean().item())
    residual_mean = float((1.0 - cosine_valid).mean().item())
    return {
        "residual_mean": residual_mean,
        "cosine_mean": cosine_mean,
        "valid_frac": valid_frac,
    }


def _dataset_stem_index(dataset: RadioLocDataset) -> Dict[str, int]:
    return {
        cache_stem_for_image_name(sample["image_name"]): idx
        for idx, sample in enumerate(dataset.samples)
    }


def _copy_npz_payload(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


@torch.no_grad()
def score_pose_cache_feature_consistency(
    *,
    config_path: str,
    pose_cache_path: str,
    output_path: str,
    gpu: int = 0,
    max_samples: int = 0,
    split: str = "test",
    split_key: str = "test_split",
    batch_size: int = 1,
) -> Dict[str, float | int | str]:
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() and int(gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    runtime = build_dcff_runtime(config, device, printer=print)
    render_h = int(runtime.render_height)
    render_w = int(runtime.render_width)
    ds_cfg = config["dataset"]

    dataset = RadioLocDataset(
        feature_dir=ds_cfg["feature_dir"],
        colmap_dir=ds_cfg["colmap_dir"],
        source_dir=ds_cfg.get("source_dir"),
        split=split,
        split_file=ds_cfg.get(split_key),
        noise_rot_deg=0.0,
        noise_trans_m=0.0,
        coarse_hw=tuple(ds_cfg.get("coarse_hw", [render_h, render_w])),
        fine_hw=tuple(ds_cfg.get("fine_hw", [render_h, render_w])),
        cache_in_memory=False,
        normalize_features=ds_cfg.get("normalize_features", False),
    )
    cameras = read_colmap_cameras(str(Path(ds_cfg["colmap_dir"]) / "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    orig_hw = (int(first_cam.height), int(first_cam.width))
    render_intr = scale_intrinsics(dataset.intrinsics, orig_hw, (render_h, render_w))
    K = intrinsics_to_K(render_intr, device)

    payload = _copy_npz_payload(pose_cache_path)
    stems = [str(v) for v in payload["query_image_stems"]]
    pose_inits = np.asarray(payload["pose_inits"], dtype=np.float32)
    stem_to_idx = _dataset_stem_index(dataset)
    limit = len(stems) if max_samples <= 0 else min(int(max_samples), len(stems))

    residuals = np.full((len(stems),), np.inf, dtype=np.float32)
    cosines = np.zeros((len(stems),), dtype=np.float32)
    valid_fracs = np.zeros((len(stems),), dtype=np.float32)
    scored = np.zeros((len(stems),), dtype=bool)
    skipped_missing = 0

    for start in tqdm(range(0, limit, int(batch_size)), desc="feature-score", leave=False):
        batch_indices = list(range(start, min(start + int(batch_size), limit)))
        query_feats = []
        poses = []
        out_indices = []
        for qi in batch_indices:
            ds_idx = stem_to_idx.get(stems[qi])
            if ds_idx is None:
                skipped_missing += 1
                continue
            sample = dataset[ds_idx]
            query = sample["query_fine"].unsqueeze(0).float()
            if query.shape[-2:] != (render_h, render_w):
                query = F.interpolate(query, size=(render_h, render_w), mode="bilinear", align_corners=False)
            query_feats.append(query.squeeze(0))
            poses.append(torch.from_numpy(pose_inits[qi]).float())
            out_indices.append(qi)
        if not out_indices:
            continue

        query_batch = torch.stack(query_feats, dim=0).to(device)
        pose_batch = torch.stack(poses, dim=0).to(device)
        bundle = render_feature_bundle_batch(
            runtime.gaussians,
            runtime.renderer,
            runtime.refiner,
            pose_batch,
            K,
            render_h,
            render_w,
            render_coarse=False,
        )
        rendered = bundle["fine_features"].float()
        valid = None
        alpha = bundle.get("alpha")
        if alpha is not None:
            valid = alpha.float() > 0.01
        for local_idx, qi in enumerate(out_indices):
            valid_i = valid[local_idx : local_idx + 1] if valid is not None else None
            metrics = compute_feature_consistency_metrics(
                query_batch[local_idx : local_idx + 1],
                rendered[local_idx : local_idx + 1],
                valid_i,
            )
            residuals[qi] = np.float32(metrics["residual_mean"])
            cosines[qi] = np.float32(metrics["cosine_mean"])
            valid_fracs[qi] = np.float32(metrics["valid_frac"])
            scored[qi] = True

    payload.update(
        {
            "feature_residual_mean": residuals,
            "feature_cosine_mean": cosines,
            "feature_valid_frac": valid_fracs,
            "feature_score_valid": scored,
            "feature_score_source": np.asarray("dcff_query_render_cosine_residual"),
        }
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)

    finite = np.isfinite(residuals) & scored
    summary: Dict[str, float | int | str] = {
        "config": str(config_path),
        "pose_cache": str(pose_cache_path),
        "output": str(output_path),
        "num_entries": int(len(stems)),
        "num_requested": int(limit),
        "num_scored": int(scored.sum()),
        "num_skipped_missing": int(skipped_missing),
        "feature_residual_mean": float(np.mean(residuals[finite])) if finite.any() else float("inf"),
        "feature_residual_median": float(np.median(residuals[finite])) if finite.any() else float("inf"),
        "feature_cosine_mean": float(np.mean(cosines[finite])) if finite.any() else 0.0,
        "feature_valid_frac_mean": float(np.mean(valid_fracs[finite])) if finite.any() else 0.0,
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pose_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--split_key", choices=("train_split", "test_split"), default="test_split")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = score_pose_cache_feature_consistency(
        config_path=args.config,
        pose_cache_path=args.pose_cache,
        output_path=args.output,
        gpu=args.gpu,
        max_samples=args.max_samples,
        split=args.split,
        split_key=args.split_key,
        batch_size=args.batch_size,
    )
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
