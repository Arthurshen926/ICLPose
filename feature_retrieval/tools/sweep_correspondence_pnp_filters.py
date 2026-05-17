#!/usr/bin/env python3
"""Sweep confidence/model filters before re-solving PnP from correspondence stores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_dataset import camera_params_to_intrinsics, read_colmap_cameras  # noqa: E402
from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    load_retrieval_init_entries,
)
from feature_retrieval.tools.score_correspondence_reliability import (  # noqa: E402
    LogisticReliabilityModel,
)


def cache_stem_for_image_name(image_name: str) -> str:
    path = Path(str(image_name).replace("\\", "/"))
    if path.parent == Path(".") or not path.parent.name:
        return path.stem
    return f"{path.parent.name}_{path.stem}"


def select_top_fraction_mask(scores: np.ndarray, *, keep_frac: float, min_points: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.zeros((0,), dtype=bool)
    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    keep = int(math.ceil(float(keep_frac) * scores.size))
    keep = max(int(min_points), keep)
    keep = min(scores.size, keep)
    order = np.argsort(-finite_scores, kind="mergesort")
    mask = np.zeros((scores.size,), dtype=bool)
    mask[order[:keep]] = True
    return mask


def camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    return (-pose[:3, :3].T @ pose[:3, 3]).astype(np.float64)


def pose_error(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred_w2c, dtype=np.float64)
    gt = np.asarray(gt_w2c, dtype=np.float64)
    r_rel = pred[:3, :3].T @ gt[:3, :3]
    cos_angle = np.clip((float(np.trace(r_rel)) - 1.0) * 0.5, -1.0, 1.0)
    rot = float(np.degrees(np.arccos(cos_angle)))
    trans = float(np.linalg.norm(camera_center_from_w2c(pred) - camera_center_from_w2c(gt)))
    return rot, trans


def summarize_pose_errors(rot_deg: np.ndarray, trans_m: np.ndarray, *, total: int) -> Dict[str, float | int | None]:
    rot = np.asarray(rot_deg, dtype=np.float64).reshape(-1)
    trans = np.asarray(trans_m, dtype=np.float64).reshape(-1)
    finite = np.isfinite(rot) & np.isfinite(trans)
    rot = rot[finite]
    trans = trans[finite]
    total = max(int(total), 0)
    denom = float(max(total, 1))
    return {
        "num_total": int(total),
        "num_success": int(len(trans)),
        "success_frac": float(len(trans) / denom),
        "rot_mean": float(np.mean(rot)) if len(rot) else None,
        "rot_median": float(np.median(rot)) if len(rot) else None,
        "trans_mean": float(np.mean(trans)) if len(trans) else None,
        "trans_median": float(np.median(trans)) if len(trans) else None,
        "joint_1deg_50mm": float(np.sum((rot < 1.0) & (trans < 0.05)) / denom),
        "joint_1deg_100mm": float(np.sum((rot < 1.0) & (trans < 0.10)) / denom),
        "joint_2deg_100mm": float(np.sum((rot < 2.0) & (trans < 0.10)) / denom),
        "joint_5deg_250mm": float(np.sum((rot < 5.0) & (trans < 0.25)) / denom),
        "joint_5deg_1000mm": float(np.sum((rot < 5.0) & (trans < 1.0)) / denom),
    }


def model_from_reliability_json_payload(payload: Mapping[str, object]) -> Tuple[LogisticReliabilityModel, List[str]]:
    model_payload = payload["model"]  # type: ignore[index]
    if not isinstance(model_payload, Mapping):
        raise ValueError("Reliability JSON missing model object")
    feature_names = [str(v) for v in model_payload.get("feature_names", [])]
    model = LogisticReliabilityModel(
        weights=np.asarray(model_payload["weights"], dtype=np.float64),
        bias=float(model_payload["bias"]),
        feature_mean=np.asarray(model_payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(model_payload["feature_std"], dtype=np.float64),
    )
    return model, feature_names


def load_model_from_reliability_json(path: str | None) -> Tuple[LogisticReliabilityModel | None, List[str]]:
    if not path:
        return None, []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return model_from_reliability_json_payload(payload)


def _normalized_xy(xy: np.ndarray, hw: Sequence[int]) -> np.ndarray:
    h = int(hw[0]) if len(hw) > 0 else 1
    w = int(hw[1]) if len(hw) > 1 else 1
    out = np.zeros_like(xy, dtype=np.float64)
    out[:, 0] = xy[:, 0] / float(max(w - 1, 1))
    out[:, 1] = xy[:, 1] / float(max(h - 1, 1))
    return out


def _features_for_payload(payload: Mapping[str, np.ndarray], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    query_xy = np.asarray(payload["query_xy"], dtype=np.float64)
    map_xy = np.asarray(payload["map_xy"], dtype=np.float64)
    confidence = np.asarray(payload["confidence"], dtype=np.float64).reshape(-1)
    pts3d = np.asarray(payload["pts3d_world"], dtype=np.float64)
    count = min(len(query_xy), len(map_xy), len(confidence), len(pts3d))
    query_xy = query_xy[:count]
    map_xy = map_xy[:count]
    confidence = confidence[:count]
    pts3d = pts3d[:count]
    query_hw = np.asarray(payload.get("query_hw", [1, 1]), dtype=np.int64).reshape(-1)
    map_hw = np.asarray(payload.get("map_hw", query_hw), dtype=np.int64).reshape(-1)
    q_norm = _normalized_xy(query_xy, query_hw)
    m_norm = _normalized_xy(map_xy, map_hw)
    feature_values = {
        "confidence": confidence,
        "query_x_norm": q_norm[:, 0],
        "query_y_norm": q_norm[:, 1],
        "map_x_norm": m_norm[:, 0],
        "map_y_norm": m_norm[:, 1],
        "delta_x_norm": q_norm[:, 0] - m_norm[:, 0],
        "delta_y_norm": q_norm[:, 1] - m_norm[:, 1],
    }
    columns = [feature_values[name] for name in feature_names]
    features = np.stack(columns, axis=1).astype(np.float32)
    valid = np.isfinite(features).all(axis=1) & np.isfinite(query_xy).all(axis=1) & np.isfinite(pts3d).all(axis=1)
    return features[valid], valid


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scale_intrinsics(base_intr: Mapping[str, float], orig_hw: Tuple[int, int], target_hw: Sequence[int]) -> Dict[str, float]:
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    h, w = int(target_hw[0]), int(target_hw[1])
    return {
        "fx": float(base_intr["fx"] * w / orig_w),
        "fy": float(base_intr["fy"] * h / orig_h),
        "cx": float(base_intr["cx"] * w / orig_w),
        "cy": float(base_intr["cy"] * h / orig_h),
    }


def _intrinsics_matrix(intr: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_pnp_ransac(
    query_xy: np.ndarray,
    pts3d_world: np.ndarray,
    intrinsics: Mapping[str, float],
    *,
    reproj_threshold: float,
    iterations: int,
) -> Tuple[bool, np.ndarray | None, int]:
    points = np.asarray(pts3d_world, dtype=np.float64).reshape(-1, 3)
    xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    count = min(len(points), len(xy))
    points = points[:count]
    xy = xy[:count]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(xy).all(axis=1)
    points = points[finite]
    xy = xy[finite]
    if len(points) < 6:
        return False, None, 0
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points,
        xy,
        _intrinsics_matrix(intrinsics),
        None,
        iterationsCount=int(iterations),
        reprojectionError=float(reproj_threshold),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not bool(success) or rvec is None or tvec is None:
        return False, None, 0
    if inliers is not None and len(inliers) >= 6:
        inlier_idx = inliers.reshape(-1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                points[inlier_idx],
                xy[inlier_idx],
                _intrinsics_matrix(intrinsics),
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return True, pose.astype(np.float32), int(0 if inliers is None else len(inliers))


def _gt_by_stem(colmap_dir: str, query_split: str) -> Dict[str, np.ndarray]:
    samples = list_colmap_split_samples(colmap_dir, query_split)
    return {
        cache_stem_for_image_name(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32)
        for sample in samples
    }


def _init_pose_by_stem(init_cache: str) -> Dict[str, np.ndarray]:
    entries, _ = load_retrieval_init_entries(init_cache)
    result = {}
    for entry in entries:
        stem = str(entry.get("query_image_stem") or cache_stem_for_image_name(entry["query_image_name"]))
        result[stem] = np.asarray(entry["pose_init"], dtype=np.float32)
    return result


def _payload_scores(
    payload: Mapping[str, np.ndarray],
    *,
    score_source: str,
    model: LogisticReliabilityModel | None,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_xy = np.asarray(payload["query_xy"], dtype=np.float64)
    pts3d = np.asarray(payload["pts3d_world"], dtype=np.float64)
    confidence = np.asarray(payload["confidence"], dtype=np.float64).reshape(-1)
    count = min(len(query_xy), len(pts3d), len(confidence))
    query_xy = query_xy[:count]
    pts3d = pts3d[:count]
    confidence = confidence[:count]
    if score_source == "confidence":
        scores = confidence
        valid = np.isfinite(query_xy).all(axis=1) & np.isfinite(pts3d).all(axis=1) & np.isfinite(scores)
        return query_xy[valid], pts3d[valid], scores[valid]
    if score_source == "model":
        if model is None:
            raise ValueError("score_source=model requires --model_json")
        features, valid = _features_for_payload(payload, feature_names)
        return query_xy[: len(valid)][valid], pts3d[: len(valid)][valid], model.predict_proba(features)
    raise ValueError(f"Unsupported score_source: {score_source}")


def evaluate_filter_sweep(
    *,
    corr_dir: str,
    init_cache: str,
    colmap_dir: str,
    query_split: str,
    output_json: str,
    score_sources: Sequence[str],
    keep_fracs: Sequence[float],
    model_json: str | None = None,
    min_points: int = 32,
    reproj_threshold: float = 8.0,
    pnp_iters: int = 10000,
) -> Dict[str, object]:
    cameras = read_colmap_cameras(str(Path(colmap_dir) / "cameras.bin"))
    first_cam = next(iter(cameras.values()))
    base_intr = camera_params_to_intrinsics(first_cam)
    orig_hw = (int(first_cam.height), int(first_cam.width))
    gt_by_stem = _gt_by_stem(colmap_dir, query_split)
    init_by_stem = _init_pose_by_stem(init_cache)
    model, feature_names = load_model_from_reliability_json(model_json)
    files = sorted(Path(corr_dir).glob("*.npz"))
    total = len(files)

    init_rot: List[float] = []
    init_trans: List[float] = []
    for path in files:
        gt = gt_by_stem.get(path.stem)
        init = init_by_stem.get(path.stem)
        if gt is None or init is None:
            continue
        rot, trans = pose_error(init, gt)
        init_rot.append(rot)
        init_trans.append(trans)

    rows: List[Dict[str, object]] = []
    for source in score_sources:
        for keep_frac in keep_fracs:
            rot_values: List[float] = []
            trans_values: List[float] = []
            pnp_inliers: List[int] = []
            num_attempted = 0
            num_missing = 0
            for path in files:
                gt = gt_by_stem.get(path.stem)
                if gt is None:
                    num_missing += 1
                    continue
                payload = _load_npz(path)
                query_hw = np.asarray(payload.get("query_hw", [first_cam.height, first_cam.width]), dtype=np.int64).reshape(-1)
                intr = _scale_intrinsics(base_intr, orig_hw, query_hw[:2])
                query_xy, pts3d, scores = _payload_scores(
                    payload,
                    score_source=source,
                    model=model,
                    feature_names=feature_names,
                )
                mask = select_top_fraction_mask(scores, keep_frac=float(keep_frac), min_points=int(min_points))
                if int(mask.sum()) < int(min_points):
                    continue
                num_attempted += 1
                success, pose, num_inliers = solve_pnp_ransac(
                    query_xy[mask],
                    pts3d[mask],
                    intr,
                    reproj_threshold=float(reproj_threshold),
                    iterations=int(pnp_iters),
                )
                if success and pose is not None:
                    rot, trans = pose_error(pose, gt)
                    rot_values.append(rot)
                    trans_values.append(trans)
                    pnp_inliers.append(num_inliers)
            summary = summarize_pose_errors(np.asarray(rot_values), np.asarray(trans_values), total=total)
            summary.update(
                {
                    "score_source": source,
                    "keep_frac": float(keep_frac),
                    "min_points": int(min_points),
                    "num_attempted": int(num_attempted),
                    "num_missing_gt": int(num_missing),
                    "pnp_inliers_mean": float(np.mean(pnp_inliers)) if pnp_inliers else None,
                    "pnp_inliers_median": float(np.median(pnp_inliers)) if pnp_inliers else None,
                }
            )
            rows.append(summary)

    result = {
        "corr_dir": str(corr_dir),
        "init_cache": str(init_cache),
        "colmap_dir": str(colmap_dir),
        "query_split": str(query_split),
        "model_json": str(model_json) if model_json else None,
        "num_corr_files": int(total),
        "init": summarize_pose_errors(np.asarray(init_rot), np.asarray(init_trans), total=total),
        "sweeps": rows,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corr_dir", required=True)
    parser.add_argument("--init_cache", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--score_sources", nargs="+", choices=("confidence", "model"), default=["confidence"])
    parser.add_argument("--model_json", default=None)
    parser.add_argument("--keep_fracs", nargs="+", type=float, default=[1.0, 0.9, 0.75, 0.5, 0.25])
    parser.add_argument("--min_points", type=int, default=32)
    parser.add_argument("--reproj_threshold", type=float, default=8.0)
    parser.add_argument("--pnp_iters", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_filter_sweep(
        corr_dir=args.corr_dir,
        init_cache=args.init_cache,
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        output_json=args.output_json,
        score_sources=args.score_sources,
        keep_fracs=args.keep_fracs,
        model_json=args.model_json,
        min_points=args.min_points,
        reproj_threshold=args.reproj_threshold,
        pnp_iters=args.pnp_iters,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
