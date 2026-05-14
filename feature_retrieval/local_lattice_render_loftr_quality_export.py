#!/usr/bin/env python3
"""Export local-lattice render-LoFTR/PnP quality caches for CPR training.

This cache is different from retrieval-init caches: ``pose_init`` is the
external initial pose T0, and ``pose_init_candidates`` are local hypotheses
around T0.  The render-LoFTR/PnP fields are teacher quality targets for those
hypotheses, not the poses used at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import list_colmap_split_samples, save_retrieval_init_entries  # noqa: E402
from feature_extract.train_impl import build_local_pose_lattice_candidates, pose_error_tensors  # noqa: E402
from feature_retrieval.pairwise_pnp_init_export import PNP_CANDIDATE_QUALITY_FIELDS  # noqa: E402
from feature_retrieval.render_loftr_pnp_init_export import (  # noqa: E402
    attach_pnp_quality_stats,
    _camera_center_from_w2c,
)
from pose_refine import apply_pose_delta  # noqa: E402
from pose_refine.evaluate_pipeline import _render_rgbd_for_loftr  # noqa: E402
from pose_refine.sparse_init import LoFTRInitializer, _compute_loftr_resolution, _scale_intrinsics  # noqa: E402


def parse_float_csv(value: str | Sequence[float] | float | int | None) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = value.replace(";", ",").split(",")
    elif isinstance(value, (int, float)):
        chunks = [value]
    else:
        chunks = list(value)
    return [float(v) for v in chunks if str(v).strip()]


def _axis_direction(index: int) -> np.ndarray:
    axes = np.eye(3, dtype=np.float32)
    axis = axes[(index // 2) % 3].copy()
    if index % 2:
        axis *= -1.0
    return axis


def deterministic_initial_delta(
    *,
    query_index: int,
    trans_cm: float,
    rot_deg: float,
    direction_mode: str = "axis",
) -> np.ndarray:
    """Return a deterministic 6D perturbation used to form T0 from GT."""
    mode = str(direction_mode or "axis").lower()
    if mode in {"axis", "axes", "cycle"}:
        trans_dir = _axis_direction(int(query_index))
        rot_dir = _axis_direction(int(query_index) + 2)
    elif mode in {"random", "sphere"}:
        rng = np.random.default_rng(int(query_index) + 1337)
        trans_dir = rng.normal(size=3).astype(np.float32)
        rot_dir = rng.normal(size=3).astype(np.float32)
        trans_dir /= max(float(np.linalg.norm(trans_dir)), 1.0e-6)
        rot_dir /= max(float(np.linalg.norm(rot_dir)), 1.0e-6)
    else:
        raise ValueError("direction_mode must be axis or random")
    delta = np.zeros((6,), dtype=np.float32)
    delta[:3] = trans_dir * (float(trans_cm) * 0.01)
    delta[3:] = rot_dir * math.radians(float(rot_deg))
    return delta


def apply_delta_np(pose_w2c: np.ndarray, delta: np.ndarray) -> np.ndarray:
    pose_t = torch.from_numpy(np.asarray(pose_w2c, dtype=np.float32))[None]
    delta_t = torch.from_numpy(np.asarray(delta, dtype=np.float32))[None]
    return apply_pose_delta(pose_t.float(), delta_t.float())[0].detach().cpu().numpy().astype(np.float32)


def build_direction_hard_candidates(
    init_pose_w2c: np.ndarray,
    init_delta: np.ndarray,
    *,
    fractions: Sequence[float],
    include_identity: bool = True,
) -> np.ndarray:
    """Build same-magnitude toward/opposite hard candidates around T0.

    ``init_delta`` is the perturbation from GT to T0.  ``-init_delta`` is the
    approximate correction direction from T0 toward GT; ``+init_delta`` is the
    same-magnitude opposite-direction negative.
    """
    rows: List[np.ndarray] = []
    if include_identity:
        rows.append(np.zeros((6,), dtype=np.float32))
    init_delta = np.asarray(init_delta, dtype=np.float32)
    for frac in fractions:
        f = float(frac)
        joint = init_delta * f
        trans_only = np.zeros((6,), dtype=np.float32)
        trans_only[:3] = init_delta[:3] * f
        rot_only = np.zeros((6,), dtype=np.float32)
        rot_only[3:] = init_delta[3:] * f
        for delta in (-joint, joint, -trans_only, trans_only, -rot_only, rot_only):
            rows.append(delta.astype(np.float32))
    poses = [apply_delta_np(init_pose_w2c, delta) for delta in rows]
    return np.stack(poses, axis=0).astype(np.float32)


def merge_candidate_banks(
    banks: Sequence[np.ndarray],
    *,
    max_candidates: int,
    pad_pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge and de-duplicate candidate pose banks, padding to a fixed size."""
    rows: List[np.ndarray] = []
    seen = set()
    for bank in banks:
        arr = np.asarray(bank, dtype=np.float32).reshape(-1, 4, 4)
        for pose in arr:
            key = tuple(np.round(pose.reshape(-1), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(pose.astype(np.float32))
            if len(rows) >= int(max_candidates):
                break
        if len(rows) >= int(max_candidates):
            break
    if not rows:
        rows.append(np.asarray(pad_pose, dtype=np.float32))
    valid_count = min(len(rows), int(max_candidates))
    while len(rows) < int(max_candidates):
        rows.append(np.asarray(pad_pose, dtype=np.float32))
    valid = np.zeros((int(max_candidates),), dtype=bool)
    valid[:valid_count] = True
    return np.stack(rows[: int(max_candidates)], axis=0).astype(np.float32), valid


def build_local_lattice_candidate_bank(
    gt_pose_w2c: np.ndarray,
    *,
    query_index: int,
    center_trans_cm: float,
    center_rot_deg: float,
    lattice_trans_cm: Sequence[float],
    lattice_rot_deg: Sequence[float],
    topk: int,
    direction_fractions: Sequence[float],
    include_identity: bool = True,
    append_gt_candidate: bool = False,
    direction_mode: str = "axis",
    lattice_direction_mode: str = "axis",
    combine_trans_rot: bool = True,
    limit_strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(init_pose, candidates, valid_mask, init_delta)``."""
    init_delta = deterministic_initial_delta(
        query_index=int(query_index),
        trans_cm=float(center_trans_cm),
        rot_deg=float(center_rot_deg),
        direction_mode=direction_mode,
    )
    init_pose = apply_delta_np(np.asarray(gt_pose_w2c, dtype=np.float32), init_delta)
    directed = build_direction_hard_candidates(
        init_pose,
        init_delta,
        fractions=direction_fractions,
        include_identity=include_identity,
    )
    lattice = build_local_pose_lattice_candidates(
        torch.from_numpy(init_pose)[None].float(),
        trans_cm=list(lattice_trans_cm),
        rot_deg=list(lattice_rot_deg),
        include_identity=include_identity,
        max_candidates=max(int(topk), 1),
        limit_strategy=limit_strategy,
        combine_trans_rot=combine_trans_rot,
        direction_mode=lattice_direction_mode,
    )[0].detach().cpu().numpy().astype(np.float32)
    banks: List[np.ndarray] = [directed, lattice]
    if append_gt_candidate:
        banks.insert(0, np.asarray(gt_pose_w2c, dtype=np.float32)[None])
    candidates, valid = merge_candidate_banks(banks, max_candidates=int(topk), pad_pose=init_pose)
    return init_pose.astype(np.float32), candidates, valid, init_delta.astype(np.float32)


def _load_rgb(path: str) -> np.ndarray | None:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pose_errors_for_candidates(candidates: np.ndarray, gt_pose_w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cand_t = torch.from_numpy(np.asarray(candidates, dtype=np.float32))
    gt_t = torch.from_numpy(np.asarray(gt_pose_w2c, dtype=np.float32))[None].expand(cand_t.shape[0], -1, -1)
    _rot_cos, rot_deg, trans_m = pose_error_tensors(cand_t.float(), gt_t.float())
    return trans_m.detach().cpu().numpy().astype(np.float32), rot_deg.detach().cpu().numpy().astype(np.float32)


def _candidate_dict_from_loftr_result(
    result,
    *,
    candidate_pose_w2c: np.ndarray,
    candidate_index: int,
    pose_error_trans_m: float,
    pose_error_rot_deg: float,
) -> Dict:
    success = bool(getattr(result, "success", False))
    extra = getattr(result, "extra", {}) or {}
    quality = extra.get("pnp_quality", {}) if success else {}
    fallback_score = 1.0 / (1.0 + float(pose_error_trans_m) + 0.01 * float(pose_error_rot_deg))
    return {
        "pose_w2c": np.asarray(candidate_pose_w2c, dtype=np.float32),
        "retrieval_frame_id": int(candidate_index),
        "retrieval_image_name": f"local_lattice_candidate_{int(candidate_index):03d}",
        "retrieval_score": float(fallback_score),
        "pnp_success": success,
        "num_inliers": int(getattr(result, "num_inliers", 0)),
        "num_matches": int(getattr(result, "num_confident_matches", 0)),
        "candidate_pose_error_trans_m": float(pose_error_trans_m),
        "candidate_pose_error_rot_deg": float(pose_error_rot_deg),
        **quality,
        "failure_reason": "" if success else str(getattr(result, "failure_reason", "")),
    }


def _quality_arrays(candidate_results: Sequence[Dict]) -> Dict[str, np.ndarray]:
    arrays = {}
    for entry_key, candidate_key, default in PNP_CANDIDATE_QUALITY_FIELDS:
        arrays[entry_key] = np.asarray(
            [
                float(c.get(candidate_key, default))
                if candidate_key != "pnp_success"
                else float(bool(c.get(candidate_key, False)))
                for c in candidate_results
            ],
            dtype=np.float32,
        )
    return arrays


def export_local_lattice_render_loftr_quality(
    *,
    colmap_dir: str,
    query_split: str,
    images_root: str,
    ply_path: str,
    save_path: str,
    topk: int = 16,
    center_trans_cm: float = 25.0,
    center_rot_deg: float = 5.0,
    lattice_trans_cm: Sequence[float] = (0.0, 5.0, 10.0, 25.0),
    lattice_rot_deg: Sequence[float] = (0.0, 1.0, 2.0, 5.0),
    direction_fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    source_name: str = "local_lattice_render_loftr_quality",
    gpu: int = 0,
    loftr_long_edge: int = 640,
    loftr_conf: float = 0.3,
    reproj_threshold: float = 8.0,
    pnp_iters: int = 10000,
    render_long_edge: int = 640,
    query_start: int = 0,
    max_queries: int = 0,
    include_identity: bool = True,
    append_gt_candidate: bool = False,
    direction_mode: str = "axis",
    lattice_direction_mode: str = "axis",
    combine_trans_rot: bool = True,
    use_magsac: bool = False,
) -> Dict:
    device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    samples = list_colmap_split_samples(colmap_dir, query_split)
    start = max(0, int(query_start))
    samples = samples[start:]
    if max_queries and int(max_queries) > 0:
        samples = samples[: int(max_queries)]

    cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    intrinsics = camera_params_to_intrinsics(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))
    scale = float(render_long_edge) / float(max(orig_hw))
    render_hw = (
        max(8, int(round(orig_hw[0] * scale / 8.0)) * 8),
        max(8, int(round(orig_hw[1] * scale / 8.0)) * 8),
    )
    loftr_hw = _compute_loftr_resolution(orig_hw, int(loftr_long_edge))
    intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)

    from feature_gaussian import HybridGaussianModel

    gaussians_depth = HybridGaussianModel(sh_degree=3, latent_dim=0)
    gaussians_depth.load_ply(ply_path, freeze_geometry=True)
    gaussians_depth.active_sh_degree = 3
    gaussians_depth = gaussians_depth.to(device) if hasattr(gaussians_depth, "to") else gaussians_depth

    loftr = LoFTRInitializer(
        device=device,
        loftr_long_edge=int(loftr_long_edge),
        confidence_threshold=float(loftr_conf),
        reproj_threshold=float(reproj_threshold),
        pnp_iters=int(pnp_iters),
        use_magsac=bool(use_magsac),
    )

    entries: List[Dict] = []
    failure_counts: Dict[str, int] = {}
    success_counts: List[int] = []
    inlier_counts: List[int] = []
    trans_errors: List[float] = []
    rot_errors: List[float] = []

    with torch.no_grad():
        for local_idx, sample in enumerate(samples):
            global_idx = start + local_idx
            query_rgb = _load_rgb(os.path.join(images_root, sample["image_name"]))
            if query_rgb is None:
                raise FileNotFoundError(os.path.join(images_root, sample["image_name"]))
            init_pose, candidates, valid_mask, init_delta = build_local_lattice_candidate_bank(
                sample["pose_w2c"],
                query_index=global_idx,
                center_trans_cm=float(center_trans_cm),
                center_rot_deg=float(center_rot_deg),
                lattice_trans_cm=lattice_trans_cm,
                lattice_rot_deg=lattice_rot_deg,
                topk=int(topk),
                direction_fractions=direction_fractions,
                include_identity=bool(include_identity),
                append_gt_candidate=bool(append_gt_candidate),
                direction_mode=direction_mode,
                lattice_direction_mode=lattice_direction_mode,
                combine_trans_rot=bool(combine_trans_rot),
            )
            cand_trans_err, cand_rot_err = _pose_errors_for_candidates(candidates, sample["pose_w2c"])
            candidate_results: List[Dict] = []
            for cand_idx, cand_pose in enumerate(candidates):
                if not bool(valid_mask[cand_idx]):
                    class _Invalid:
                        success = False
                        pose_w2c = None
                        num_inliers = 0
                        num_confident_matches = 0
                        failure_reason = "padded_invalid_candidate"

                    result = _Invalid()
                else:
                    try:
                        rendered_rgb, rendered_depth = _render_rgbd_for_loftr(
                            gaussians_depth,
                            cand_pose.astype(np.float32),
                            intrinsics,
                            render_hw,
                            orig_hw,
                            device,
                        )
                        result = loftr.estimate_pose(
                            query_rgb,
                            rendered_rgb,
                            rendered_depth,
                            cand_pose.astype(np.float32),
                            intrinsics,
                            orig_hw,
                        )
                    except Exception as exc:
                        class _Failed:
                            success = False
                            pose_w2c = None
                            num_inliers = 0
                            num_confident_matches = 0
                            failure_reason = f"exception:{type(exc).__name__}:{exc}"

                        result = _Failed()
                if bool(getattr(result, "success", False)) and getattr(result, "pose_w2c", None) is not None:
                    attach_pnp_quality_stats(result, result.pose_w2c, intr_loftr)
                    inlier_counts.append(int(getattr(result, "num_inliers", 0)))
                else:
                    reason = str(getattr(result, "failure_reason", "") or "unknown")
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                candidate_results.append(
                    _candidate_dict_from_loftr_result(
                        result,
                        candidate_pose_w2c=cand_pose,
                        candidate_index=cand_idx,
                        pose_error_trans_m=float(cand_trans_err[cand_idx]),
                        pose_error_rot_deg=float(cand_rot_err[cand_idx]),
                    )
                )
            success_count = sum(1 for c in candidate_results if c.get("pnp_success"))
            success_counts.append(success_count)
            best_idx = max(
                range(len(candidate_results)),
                key=lambda i: (
                    1 if candidate_results[i].get("pnp_success") else 0,
                    int(candidate_results[i].get("num_inliers", 0)),
                    -float(cand_trans_err[i]),
                ),
            )
            trans_errors.append(float(cand_trans_err[best_idx]))
            rot_errors.append(float(cand_rot_err[best_idx]))
            quality_arrays = _quality_arrays(candidate_results)
            entries.append(
                {
                    "query_img_id": int(sample["img_id"]),
                    "query_image_name": str(sample["image_name"]),
                    "query_image_stem": str(sample["image_stem"]),
                    "pose_init": init_pose.astype(np.float32),
                    "init_source": source_name,
                    "retrieval_frame_id": int(best_idx),
                    "retrieval_image_name": str(candidate_results[best_idx]["retrieval_image_name"]),
                    "retrieval_score": float(candidate_results[best_idx]["retrieval_score"]),
                    "pose_init_candidates": candidates.astype(np.float32),
                    "candidate_valid_mask": valid_mask.astype(bool),
                    "retrieval_frame_ids_candidates": np.arange(len(candidate_results), dtype=np.int64),
                    "retrieval_image_names_candidates": np.asarray(
                        [c["retrieval_image_name"] for c in candidate_results]
                    ),
                    "retrieval_scores_candidates": np.asarray(
                        [float(c["retrieval_score"]) for c in candidate_results],
                        dtype=np.float32,
                    ),
                    **quality_arrays,
                }
            )
            if (local_idx + 1) % 2 == 0 or local_idx == 0:
                print(
                    f"[{local_idx + 1}/{len(samples)}] local lattice LoFTR successes: "
                    f"{sum(success_counts)}/{max(1, len(success_counts) * int(topk))}"
                )

    stats = {
        "method_requested": "local_lattice_render_loftr_quality",
        "method_used": source_name,
        "num_query_samples": int(len(entries)),
        "retrieval_topk_requested": int(topk),
        "local_lattice_topk": int(topk),
        "center_trans_cm": float(center_trans_cm),
        "center_rot_deg": float(center_rot_deg),
        "lattice_trans_cm": [float(v) for v in lattice_trans_cm],
        "lattice_rot_deg": [float(v) for v in lattice_rot_deg],
        "direction_fractions": [float(v) for v in direction_fractions],
        "append_gt_candidate": bool(append_gt_candidate),
        "query_start": int(query_start),
        "max_queries": int(max_queries),
        "images_root": str(images_root),
        "ply_path": str(ply_path),
        "orig_hw": list(orig_hw),
        "render_hw": list(render_hw),
        "loftr_hw": list(loftr_hw),
        "loftr_long_edge": int(loftr_long_edge),
        "loftr_conf": float(loftr_conf),
        "reproj_threshold": float(reproj_threshold),
        "pnp_iters": int(pnp_iters),
        "candidate_success_rate": float(sum(success_counts) / max(1, len(success_counts) * int(topk))),
        "queries_with_success_rate": float(sum(1 for v in success_counts if v > 0) / max(1, len(success_counts))),
        "mean_success_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "best_candidate_trans_m_median": float(np.median(trans_errors)) if trans_errors else None,
        "best_candidate_rot_deg_median": float(np.median(rot_errors)) if rot_errors else None,
        "failure_counts": failure_counts,
        "counts_by_source": {source_name: int(len(entries))},
    }
    save_retrieval_init_entries(entries, stats, save_path)
    del loftr, gaussians_depth
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--center_trans_cm", type=float, default=25.0)
    parser.add_argument("--center_rot_deg", type=float, default=5.0)
    parser.add_argument("--lattice_trans_cm", default="0,5,10,25")
    parser.add_argument("--lattice_rot_deg", default="0,1,2,5")
    parser.add_argument("--direction_fractions", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--source_name", default="local_lattice_render_loftr_quality")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loftr_long_edge", type=int, default=640)
    parser.add_argument("--loftr_conf", type=float, default=0.3)
    parser.add_argument("--reproj_threshold", type=float, default=8.0)
    parser.add_argument("--pnp_iters", type=int, default=10000)
    parser.add_argument("--render_long_edge", type=int, default=640)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--no_identity", action="store_true")
    parser.add_argument("--append_gt_candidate", action="store_true")
    parser.add_argument("--direction_mode", default="axis")
    parser.add_argument("--lattice_direction_mode", default="axis")
    parser.add_argument("--no_combine_trans_rot", action="store_true")
    parser.add_argument("--use_magsac", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_local_lattice_render_loftr_quality(
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        images_root=args.images_root,
        ply_path=args.ply_path,
        save_path=args.save_path,
        topk=args.topk,
        center_trans_cm=args.center_trans_cm,
        center_rot_deg=args.center_rot_deg,
        lattice_trans_cm=parse_float_csv(args.lattice_trans_cm),
        lattice_rot_deg=parse_float_csv(args.lattice_rot_deg),
        direction_fractions=parse_float_csv(args.direction_fractions),
        source_name=args.source_name,
        gpu=args.gpu,
        loftr_long_edge=args.loftr_long_edge,
        loftr_conf=args.loftr_conf,
        reproj_threshold=args.reproj_threshold,
        pnp_iters=args.pnp_iters,
        render_long_edge=args.render_long_edge,
        query_start=args.query_start,
        max_queries=args.max_queries,
        include_identity=not bool(args.no_identity),
        append_gt_candidate=bool(args.append_gt_candidate),
        direction_mode=args.direction_mode,
        lattice_direction_mode=args.lattice_direction_mode,
        combine_trans_rot=not bool(args.no_combine_trans_rot),
        use_magsac=bool(args.use_magsac),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
