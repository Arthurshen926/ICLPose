#!/usr/bin/env python3
"""Append an externally refined pose as an extra candidate in an init cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS,
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)
from feature_retrieval.tools.gate_refined_init_cache import _load_refined_by_stem  # noqa: E402


_OPTIONAL_DEFAULTS = {
    "retrieval_original_scores_candidates": 0.0,
    "retrieval_pnp_success_candidates": 0.0,
    "retrieval_pnp_num_inliers_candidates": 0.0,
    "retrieval_pnp_num_matches_candidates": 0.0,
    "retrieval_pnp_reproj_rmse_candidates": float("inf"),
    "retrieval_pnp_reproj_median_candidates": float("inf"),
    "retrieval_pnp_inlier_ratio_candidates": 0.0,
    "retrieval_pnp_inlier_conf_mean_candidates": 0.0,
}


def _copy_entry(entry: Dict) -> Dict:
    copied: Dict = {}
    for key, value in entry.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _candidate_quality_array(entry: Dict, key: str, num_candidates: int) -> np.ndarray:
    if key in entry:
        value = np.asarray(entry[key], dtype=np.float32).reshape(-1)
        if value.shape[0] != num_candidates:
            raise ValueError(f"{key} length {value.shape[0]} does not match {num_candidates} candidates")
        return value.copy()
    return np.full((num_candidates,), float(_OPTIONAL_DEFAULTS[key]), dtype=np.float32)


def _refined_quality_value(key: str, quality: Dict) -> float:
    if key == "retrieval_pnp_success_candidates":
        return 1.0 if bool(quality["success"]) else 0.0
    if key == "retrieval_pnp_num_inliers_candidates":
        return float(quality["num_inliers"])
    if key == "retrieval_pnp_num_matches_candidates":
        return float(quality["num_raw_matches"])
    if key == "retrieval_pnp_inlier_ratio_candidates":
        raw = int(quality["num_raw_matches"])
        return float(quality["num_inliers"]) / float(raw) if raw > 0 else 0.0
    return float(_OPTIONAL_DEFAULTS[key])


def append_refined_init_candidate(
    base_cache_path: str,
    refined_cache_path: str,
    save_path: str,
    *,
    candidate_name: str = "external_refined_pose",
    max_base_candidates: int | None = None,
) -> Tuple[List[Dict], Dict]:
    """Append a refined pose candidate while preserving the original active init."""
    base_entries, base_stats = load_retrieval_init_entries(str(base_cache_path))
    refined_by_stem = _load_refined_by_stem(str(refined_cache_path))

    missing = [entry["query_image_stem"] for entry in base_entries if entry["query_image_stem"] not in refined_by_stem]
    if missing:
        preview = ", ".join(str(v) for v in missing[:5])
        raise ValueError(f"missing refined entries for {len(missing)} base cache rows: {preview}")

    exported_entries: List[Dict] = []
    appended_valid = 0
    if max_base_candidates is not None and int(max_base_candidates) < 1:
        raise ValueError("max_base_candidates must be >= 1 when provided")
    for base_entry in base_entries:
        exported = _copy_entry(base_entry)
        stem = str(base_entry["query_image_stem"])
        quality = refined_by_stem[stem]
        refined_pose = np.asarray(quality["entry"]["pose_init"], dtype=np.float32)
        was_success = bool(quality["success"])
        appended_valid += int(was_success)

        poses = np.asarray(exported["pose_init_candidates"], dtype=np.float32)
        original_num_candidates = int(poses.shape[0])
        num_candidates = original_num_candidates
        if max_base_candidates is not None:
            num_candidates = min(original_num_candidates, int(max_base_candidates))
            poses = poses[:num_candidates]
            exported["candidate_valid_mask"] = np.asarray(exported["candidate_valid_mask"], dtype=bool).reshape(-1)[
                :num_candidates
            ]
            exported["retrieval_frame_ids_candidates"] = np.asarray(
                exported["retrieval_frame_ids_candidates"], dtype=np.int64
            ).reshape(-1)[:num_candidates]
            exported["retrieval_image_names_candidates"] = list(exported["retrieval_image_names_candidates"])[
                :num_candidates
            ]
            exported["retrieval_scores_candidates"] = np.asarray(
                exported["retrieval_scores_candidates"], dtype=np.float32
            ).reshape(-1)[:num_candidates]
            for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
                if key in exported:
                    exported[key] = np.asarray(exported[key], dtype=np.float32).reshape(-1)[:num_candidates]
        exported["pose_init_candidates"] = np.concatenate([poses, refined_pose[None]], axis=0).astype(np.float32)
        exported["candidate_valid_mask"] = np.concatenate(
            [np.asarray(exported["candidate_valid_mask"], dtype=bool).reshape(-1), np.array([was_success], dtype=bool)]
        )
        exported["retrieval_frame_ids_candidates"] = np.concatenate(
            [
                np.asarray(exported["retrieval_frame_ids_candidates"], dtype=np.int64).reshape(-1),
                np.array([-1], dtype=np.int64),
            ]
        )
        exported["retrieval_image_names_candidates"] = list(exported["retrieval_image_names_candidates"]) + [
            str(candidate_name)
        ]
        exported["retrieval_scores_candidates"] = np.concatenate(
            [
                np.asarray(exported["retrieval_scores_candidates"], dtype=np.float32).reshape(-1),
                np.array([0.0], dtype=np.float32),
            ]
        )
        for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
            values = _candidate_quality_array(exported, key, num_candidates)
            appended = np.array([_refined_quality_value(key, quality)], dtype=np.float32)
            exported[key] = np.concatenate([values, appended], axis=0).astype(np.float32)
        exported_entries.append(exported)

    exported_stats = dict(base_stats or {})
    exported_stats.update(
        {
            "selection_source": "append_refined_candidate",
            "base_cache": str(base_cache_path),
            "refined_cache": str(refined_cache_path),
            "num_entries": len(exported_entries),
            "num_appended_valid": int(appended_valid),
            "candidate_name": str(candidate_name),
            "max_base_candidates": None if max_base_candidates is None else int(max_base_candidates),
        }
    )
    save_retrieval_init_entries(exported_entries, exported_stats, str(save_path))
    return exported_entries, exported_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_cache", required=True, help="Original retrieval/init cache .npz")
    parser.add_argument("--refined_cache", required=True, help="External refined pose cache .npz")
    parser.add_argument("--save_path", required=True, help="Path for the appended output cache .npz")
    parser.add_argument("--candidate_name", default="external_refined_pose")
    parser.add_argument("--max_base_candidates", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        _entries, stats = append_refined_init_candidate(
            args.base_cache,
            args.refined_cache,
            args.save_path,
            candidate_name=args.candidate_name,
            max_base_candidates=args.max_base_candidates,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Exported {stats['num_entries']} init entries with appended refined candidate "
        f"to {args.save_path} (valid_appended={stats['num_appended_valid']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
