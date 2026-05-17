#!/usr/bin/env python3
"""Score LoFTR correspondences by query/render DCFF feature consistency."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def cache_stem_for_image_name(image_name: str) -> str:
    path = Path(str(image_name).replace("\\", "/"))
    if path.parent == Path(".") or not path.parent.name:
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def _as_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def binary_auc_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    lower_score_is_positive: bool = True,
) -> float:
    """Return binary AUC with tie-aware average ranks.

    ``labels=True`` denotes the positive class.  For feature residuals, lower
    scores should indicate inliers, so ``lower_score_is_positive`` defaults to
    ``True``.
    """
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0:
        return float("nan")
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranking_scores = -scores if bool(lower_score_is_positive) else scores
    order = np.argsort(ranking_scores, kind="mergesort")
    sorted_scores = ranking_scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks_unsorted = np.empty_like(ranks)
    ranks_unsorted[order] = ranks
    pos_rank_sum = float(ranks_unsorted[labels].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) * 0.5) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def summarize_labeled_residuals(residuals: np.ndarray, inlier_mask: np.ndarray) -> Dict[str, float | int | None]:
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    count = min(residuals.size, inlier_mask.size)
    residuals = residuals[:count]
    inlier_mask = inlier_mask[:count]
    finite = np.isfinite(residuals)
    residuals = residuals[finite]
    inlier_mask = inlier_mask[finite]
    inlier_res = residuals[inlier_mask]
    outlier_res = residuals[~inlier_mask]
    auc = binary_auc_from_scores(inlier_mask, residuals, lower_score_is_positive=True)
    return {
        "num_points": int(residuals.size),
        "num_inliers": int(inlier_res.size),
        "num_outliers": int(outlier_res.size),
        "inlier_residual_mean": _as_float(np.mean(inlier_res)) if inlier_res.size else None,
        "inlier_residual_median": _as_float(np.median(inlier_res)) if inlier_res.size else None,
        "outlier_residual_mean": _as_float(np.mean(outlier_res)) if outlier_res.size else None,
        "outlier_residual_median": _as_float(np.median(outlier_res)) if outlier_res.size else None,
        "inlier_lower_residual_auc": _as_float(auc),
    }


def _scale_xy_to_feature_hw(
    xy: torch.Tensor,
    *,
    source_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    if source_hw is None:
        return xy
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if source_h <= 1 or source_w <= 1 or target_h <= 1 or target_w <= 1:
        raise ValueError("source_hw and target_hw must be > 1 for coordinate scaling")
    scaled = xy.clone()
    scaled[..., 0] = scaled[..., 0] * float(target_w - 1) / float(source_w - 1)
    scaled[..., 1] = scaled[..., 1] * float(target_h - 1) / float(source_h - 1)
    return scaled


def sample_feature_vectors_at_xy(
    feature: torch.Tensor,
    xy: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    source_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample BCHW features at pixel coordinates.

    Coordinates are interpreted in ``source_hw`` if provided and are scaled to
    the feature grid before sampling.  Returned vectors have shape ``(B,N,C)``.
    """
    if feature.ndim != 4:
        raise ValueError("feature must have shape (B,C,H,W)")
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy must have shape (B,N,2)")
    if xy.shape[0] != feature.shape[0]:
        raise ValueError("xy and feature batch dimensions must match")
    bsz, channels, height, width = feature.shape
    xy = xy.to(device=feature.device, dtype=feature.dtype)
    xy_feat = _scale_xy_to_feature_hw(xy, source_hw=source_hw, target_hw=(height, width))
    inside = (
        torch.isfinite(xy_feat).all(dim=-1)
        & (xy_feat[..., 0] >= 0.0)
        & (xy_feat[..., 0] <= max(width - 1, 0))
        & (xy_feat[..., 1] >= 0.0)
        & (xy_feat[..., 1] <= max(height - 1, 0))
    )
    if valid is not None:
        inside = inside & valid.to(device=feature.device).bool()
    norm_x = 2.0 * xy_feat[..., 0] / float(max(width - 1, 1)) - 1.0
    norm_y = 2.0 * xy_feat[..., 1] / float(max(height - 1, 1)) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(bsz, -1, 1, 2)
    sampled = F.grid_sample(
        feature.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    vectors = sampled.squeeze(-1).transpose(1, 2).contiguous()
    return vectors[:, :, :channels], inside


def _load_npz_payload(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _pose_cache_by_stem(path: str | Path) -> Dict[str, np.ndarray]:
    payload = _load_npz_payload(path)
    stems = [str(stem) for stem in payload["query_image_stems"]]
    poses = np.asarray(payload["pose_inits"], dtype=np.float32)
    return {stem: poses[idx] for idx, stem in enumerate(stems)}


def _corr_files(corr_dir: str | Path) -> list[Path]:
    return sorted(Path(corr_dir).glob("*.npz"))


def _dataset_stem_index(dataset) -> Dict[str, int]:
    return {
        cache_stem_for_image_name(sample["image_name"]): idx
        for idx, sample in enumerate(dataset.samples)
    }


def _scale_intrinsics(base_intr: Dict[str, float], orig_hw: tuple[int, int], target_hw: tuple[int, int]) -> Dict[str, float]:
    orig_h, orig_w = orig_hw
    h, w = target_hw
    return {
        "fx": float(base_intr["fx"] * w / orig_w),
        "fy": float(base_intr["fy"] * h / orig_h),
        "cx": float(base_intr["cx"] * w / orig_w),
        "cy": float(base_intr["cy"] * h / orig_h),
    }


@torch.no_grad()
def score_loftr_correspondence_feature_consistency(
    *,
    config_path: str,
    pose_cache_path: str,
    corr_dir: str,
    summary_json: str,
    output_dir: str | None = None,
    gpu: int = 0,
    max_samples: int = 0,
    split: str = "test",
    split_key: str = "test_split",
    batch_size: int = 1,
) -> Dict[str, object]:
    from data.radio_loc_dataset import RadioLocDataset, read_colmap_cameras
    from feature_field import build_dcff_runtime, intrinsics_to_K, render_feature_bundle_batch

    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() and int(gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
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
    render_intr = _scale_intrinsics(dataset.intrinsics, orig_hw, (render_h, render_w))
    K = intrinsics_to_K(render_intr, device)

    pose_by_stem = _pose_cache_by_stem(pose_cache_path)
    stem_to_idx = _dataset_stem_index(dataset)
    files = _corr_files(corr_dir)
    if int(max_samples) > 0:
        files = files[: int(max_samples)]
    out_dir = Path(output_dir) if output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    per_query: list[Dict[str, object]] = []
    all_residuals: list[np.ndarray] = []
    all_inliers: list[np.ndarray] = []
    skipped_missing = 0

    for start in tqdm(range(0, len(files), int(batch_size)), desc="corr-feature-score", leave=False):
        batch_files = files[start : start + int(batch_size)]
        query_feats = []
        pose_tensors = []
        corr_payloads = []
        active_files = []
        for path in batch_files:
            stem = path.stem
            ds_idx = stem_to_idx.get(stem)
            pose = pose_by_stem.get(stem)
            if ds_idx is None or pose is None:
                skipped_missing += 1
                continue
            payload = _load_npz_payload(path)
            if "query_xy" not in payload or "map_xy" not in payload:
                skipped_missing += 1
                continue
            sample = dataset[ds_idx]
            query = sample["query_fine"].unsqueeze(0).float()
            if query.shape[-2:] != (render_h, render_w):
                query = F.interpolate(query, size=(render_h, render_w), mode="bilinear", align_corners=False)
            query_feats.append(query.squeeze(0))
            pose_tensors.append(torch.from_numpy(np.asarray(pose, dtype=np.float32)))
            corr_payloads.append(payload)
            active_files.append(path)
        if not active_files:
            continue
        query_batch = torch.stack(query_feats, dim=0).to(device)
        pose_batch = torch.stack(pose_tensors, dim=0).to(device)
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
        render_batch = bundle["fine_features"].float()

        for local_idx, (path, payload) in enumerate(zip(active_files, corr_payloads)):
            query_xy_np = np.asarray(payload["query_xy"], dtype=np.float32)
            map_xy_np = np.asarray(payload["map_xy"], dtype=np.float32)
            count = min(len(query_xy_np), len(map_xy_np))
            if count == 0:
                summary = summarize_labeled_residuals(np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool))
                summary.update({"stem": path.stem})
                per_query.append(summary)
                continue
            inlier_np = np.asarray(payload.get("pnp_inlier_mask", np.ones((count,), dtype=bool)), dtype=bool)[:count]
            valid_np = np.ones((count,), dtype=bool)
            query_hw = tuple(int(v) for v in np.asarray(payload.get("query_hw", [render_h, render_w])).reshape(-1)[:2])
            map_hw = tuple(int(v) for v in np.asarray(payload.get("map_hw", [render_h, render_w])).reshape(-1)[:2])
            query_xy = torch.from_numpy(query_xy_np[:count]).view(1, count, 2).to(device)
            map_xy = torch.from_numpy(map_xy_np[:count]).view(1, count, 2).to(device)
            valid = torch.from_numpy(valid_np).view(1, count).to(device)
            q_vec, q_in = sample_feature_vectors_at_xy(
                query_batch[local_idx : local_idx + 1],
                query_xy,
                valid,
                source_hw=query_hw,
            )
            r_vec, r_in = sample_feature_vectors_at_xy(
                render_batch[local_idx : local_idx + 1],
                map_xy,
                valid,
                source_hw=map_hw,
            )
            valid_mask = (q_in & r_in)[0].detach().cpu().numpy().astype(bool)
            q_norm = F.normalize(q_vec.float(), dim=-1, eps=1.0e-6)
            r_norm = F.normalize(r_vec.float(), dim=-1, eps=1.0e-6)
            cosine = (q_norm * r_norm).sum(dim=-1).clamp(-1.0, 1.0)[0].detach().cpu().numpy()
            residual = (1.0 - cosine).astype(np.float32)
            residual_valid = residual[valid_mask]
            inlier_valid = inlier_np[valid_mask]
            summary = summarize_labeled_residuals(residual_valid, inlier_valid)
            summary.update(
                {
                    "stem": path.stem,
                    "feature_valid_points": int(valid_mask.sum()),
                    "feature_valid_frac": float(valid_mask.mean()) if valid_mask.size else 0.0,
                }
            )
            per_query.append(summary)
            all_residuals.append(residual_valid)
            all_inliers.append(inlier_valid)
            if out_dir is not None:
                np.savez_compressed(
                    out_dir / f"{path.stem}.npz",
                    query_xy=query_xy_np[:count],
                    map_xy=map_xy_np[:count],
                    pnp_inlier_mask=inlier_np,
                    feature_valid_mask=valid_mask,
                    feature_cosine=cosine.astype(np.float32),
                    feature_residual=residual.astype(np.float32),
                )

    if all_residuals:
        residual_cat = np.concatenate(all_residuals, axis=0)
        inlier_cat = np.concatenate(all_inliers, axis=0)
    else:
        residual_cat = np.zeros((0,), dtype=np.float32)
        inlier_cat = np.zeros((0,), dtype=bool)
    global_summary = summarize_labeled_residuals(residual_cat, inlier_cat)
    query_aucs = [
        float(row["inlier_lower_residual_auc"])
        for row in per_query
        if row.get("inlier_lower_residual_auc") is not None
    ]
    summary: Dict[str, object] = {
        "config": str(config_path),
        "pose_cache": str(pose_cache_path),
        "corr_dir": str(corr_dir),
        "output_dir": str(output_dir) if output_dir else None,
        "num_corr_files": int(len(files)),
        "num_scored_queries": int(len(per_query)),
        "num_skipped_missing": int(skipped_missing),
        "global": global_summary,
        "query_auc_mean": float(np.mean(query_aucs)) if query_aucs else None,
        "query_auc_median": float(np.median(query_aucs)) if query_aucs else None,
        "per_query": per_query,
    }
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pose_cache", required=True)
    parser.add_argument("--corr_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--split_key", choices=("train_split", "test_split"), default="test_split")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = score_loftr_correspondence_feature_consistency(
        config_path=args.config,
        pose_cache_path=args.pose_cache,
        corr_dir=args.corr_dir,
        summary_json=args.summary_json,
        output_dir=args.output_dir,
        gpu=args.gpu,
        max_samples=args.max_samples,
        split=args.split,
        split_key=args.split_key,
        batch_size=args.batch_size,
    )
    printable = dict(summary)
    printable["per_query"] = printable.get("per_query", [])[:5]
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
