#!/usr/bin/env python3
"""Gate an externally refined pose cache against the original init cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)


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


def _quality_array(data, key: str, length: int, default, dtype) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=dtype).reshape(-1)
    return np.full((length,), default, dtype=dtype)


def _load_refined_by_stem(refined_cache_path: str) -> Dict[str, Dict]:
    refined_entries, _stats = load_retrieval_init_entries(str(refined_cache_path))
    data = np.load(refined_cache_path, allow_pickle=True)
    n = len(refined_entries)
    successes = _quality_array(data, "refine_success", n, True, bool)
    inliers = _quality_array(data, "refine_num_inliers", n, 0, np.int32)
    raw_matches = _quality_array(data, "refine_num_raw_matches", n, 0, np.int32)

    refined_by_stem: Dict[str, Dict] = {}
    for idx, entry in enumerate(refined_entries):
        stem = str(entry["query_image_stem"])
        refined_by_stem[stem] = {
            "entry": entry,
            "success": bool(successes[idx]),
            "num_inliers": int(inliers[idx]),
            "num_raw_matches": int(raw_matches[idx]),
        }
    return refined_by_stem


def _load_score_by_stem(scored_cache_path: str, *, score_key: str) -> Dict[str, float]:
    scored_entries, _stats = load_retrieval_init_entries(str(scored_cache_path))
    data = np.load(scored_cache_path, allow_pickle=True)
    if score_key not in data.files:
        raise ValueError(f"score key '{score_key}' is missing from {scored_cache_path}")
    scores = np.asarray(data[score_key], dtype=np.float32).reshape(-1)
    if len(scores) != len(scored_entries):
        raise ValueError(
            f"score key '{score_key}' length mismatch in {scored_cache_path}: "
            f"{len(scores)} scores for {len(scored_entries)} entries"
        )
    return {
        str(entry["query_image_stem"]): float(scores[idx])
        for idx, entry in enumerate(scored_entries)
    }


def _passes_gate(
    quality: Dict,
    *,
    min_inliers: int,
    min_raw_matches: int,
    min_inlier_ratio: float,
) -> bool:
    raw_matches = int(quality["num_raw_matches"])
    ratio = float(quality["num_inliers"]) / float(raw_matches) if raw_matches > 0 else 0.0
    return (
        bool(quality["success"])
        and int(quality["num_inliers"]) >= int(min_inliers)
        and raw_matches >= int(min_raw_matches)
        and ratio >= float(min_inlier_ratio)
    )


def _save_entries_with_refine_quality(
    entries: Sequence[Dict],
    stats: Dict,
    save_path: str,
    *,
    refine_success: Sequence[bool],
    refine_num_inliers: Sequence[int],
    refine_num_raw_matches: Sequence[int],
    refine_score_selected: Sequence[float] | None = None,
    refine_score_reference: Sequence[float] | None = None,
    refine_score_candidate: Sequence[float] | None = None,
) -> None:
    save_retrieval_init_entries(entries, stats, str(save_path))
    with np.load(save_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload.update(
        {
            "refine_success": np.asarray(refine_success, dtype=bool),
            "refine_num_inliers": np.asarray(refine_num_inliers, dtype=np.int32),
            "refine_num_raw_matches": np.asarray(refine_num_raw_matches, dtype=np.int32),
        }
    )
    if refine_score_selected is not None:
        payload["refine_score_selected"] = np.asarray(refine_score_selected, dtype=np.float32)
    if refine_score_reference is not None:
        payload["refine_score_reference"] = np.asarray(refine_score_reference, dtype=np.float32)
    if refine_score_candidate is not None:
        payload["refine_score_candidate"] = np.asarray(refine_score_candidate, dtype=np.float32)
    np.savez(save_path, **payload)


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    return -(pose[:3, :3].T @ pose[:3, 3])


def _relative_pose_step(reference_pose: np.ndarray, candidate_pose: np.ndarray) -> Tuple[float, float]:
    reference = np.asarray(reference_pose, dtype=np.float64)
    candidate = np.asarray(candidate_pose, dtype=np.float64)
    trans_m = float(
        np.linalg.norm(_camera_center_from_w2c(reference) - _camera_center_from_w2c(candidate))
    )
    r_rel = reference[:3, :3].T @ candidate[:3, :3]
    cos_angle = float(np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    return trans_m, rot_deg


def _passes_pose_step_gate(
    reference_pose: np.ndarray,
    candidate_pose: np.ndarray,
    *,
    min_step_m: float,
    max_step_m: float | None,
    min_rot_deg: float,
    max_rot_deg: float | None,
) -> Tuple[bool, float, float]:
    trans_m, rot_deg = _relative_pose_step(reference_pose, candidate_pose)
    if trans_m < float(min_step_m):
        return False, trans_m, rot_deg
    if max_step_m is not None and trans_m > float(max_step_m):
        return False, trans_m, rot_deg
    if rot_deg < float(min_rot_deg):
        return False, trans_m, rot_deg
    if max_rot_deg is not None and rot_deg > float(max_rot_deg):
        return False, trans_m, rot_deg
    return True, trans_m, rot_deg


def gate_refined_init_cache(
    base_cache_path: str,
    refined_cache_path: str,
    save_path: str,
    *,
    min_inliers: int = 0,
    min_raw_matches: int = 0,
    min_inlier_ratio: float = 0.0,
    source_name: str = "gated_refined_pose",
) -> Tuple[List[Dict], Dict]:
    """Export a retrieval-init-compatible cache with quality-gated refined poses.

    The output preserves all top-K candidate arrays from ``base_cache_path``.
    Only the active ``pose_init`` is replaced by the refined pose when the
    external correspondence quality passes the gate.
    """
    base_entries, base_stats = load_retrieval_init_entries(str(base_cache_path))
    refined_by_stem = _load_refined_by_stem(str(refined_cache_path))

    missing = [entry["query_image_stem"] for entry in base_entries if entry["query_image_stem"] not in refined_by_stem]
    if missing:
        preview = ", ".join(str(v) for v in missing[:5])
        raise ValueError(f"missing refined entries for {len(missing)} base cache rows: {preview}")

    exported_entries: List[Dict] = []
    accepted = 0
    refine_success: List[bool] = []
    refine_num_inliers: List[int] = []
    refine_num_raw_matches: List[int] = []
    for base_entry in base_entries:
        stem = str(base_entry["query_image_stem"])
        quality = refined_by_stem[stem]
        exported = _copy_entry(base_entry)
        if _passes_gate(
            quality,
            min_inliers=min_inliers,
            min_raw_matches=min_raw_matches,
            min_inlier_ratio=min_inlier_ratio,
        ):
            refined_entry = quality["entry"]
            exported["pose_init"] = np.asarray(refined_entry["pose_init"], dtype=np.float32).copy()
            exported["init_source"] = str(source_name)
            accepted += 1
            refine_success.append(bool(quality["success"]))
            refine_num_inliers.append(int(quality["num_inliers"]))
            refine_num_raw_matches.append(int(quality["num_raw_matches"]))
        else:
            refine_success.append(False)
            refine_num_inliers.append(0)
            refine_num_raw_matches.append(0)
        exported_entries.append(exported)

    exported_stats = dict(base_stats or {})
    exported_stats.update(
        {
            "selection_source": "quality_gated_refined_pose",
            "base_cache": str(base_cache_path),
            "refined_cache": str(refined_cache_path),
            "num_entries": len(exported_entries),
            "num_refined_accepted": int(accepted),
            "num_refined_rejected": int(len(exported_entries) - accepted),
            "min_inliers": int(min_inliers),
            "min_raw_matches": int(min_raw_matches),
            "min_inlier_ratio": float(min_inlier_ratio),
            "source_name": str(source_name),
        }
    )
    _save_entries_with_refine_quality(
        exported_entries,
        exported_stats,
        str(save_path),
        refine_success=refine_success,
        refine_num_inliers=refine_num_inliers,
        refine_num_raw_matches=refine_num_raw_matches,
    )
    return exported_entries, exported_stats


def gate_refined_init_cache_by_pose_step(
    base_cache_path: str,
    reference_refined_cache_path: str,
    candidate_refined_cache_path: str,
    save_path: str,
    *,
    step_candidate_refined_cache_path: str | None = None,
    min_step_m: float = 0.0,
    max_step_m: float | None = None,
    min_rot_deg: float = 0.0,
    max_rot_deg: float | None = None,
    min_inliers: int = 0,
    min_raw_matches: int = 0,
    min_inlier_ratio: float = 0.0,
    reference_source_name: str = "reference_refined_pose",
    candidate_source_name: str = "pose_step_gated_refined_pose",
) -> Tuple[List[Dict], Dict]:
    """Choose between two refined pose caches using only their relative step.

    The output preserves the candidate-bank metadata from ``base_cache_path``.
    The active pose defaults to the reference refined pose when it passes the
    quality gate; it switches to the candidate refined pose only when the
    candidate quality and relative pose-step gates pass.
    """
    base_entries, base_stats = load_retrieval_init_entries(str(base_cache_path))
    reference_by_stem = _load_refined_by_stem(str(reference_refined_cache_path))
    candidate_by_stem = _load_refined_by_stem(str(candidate_refined_cache_path))
    step_candidate_by_stem = (
        _load_refined_by_stem(str(step_candidate_refined_cache_path))
        if step_candidate_refined_cache_path is not None
        else candidate_by_stem
    )

    missing = [
        entry["query_image_stem"]
        for entry in base_entries
        if entry["query_image_stem"] not in reference_by_stem
        or entry["query_image_stem"] not in candidate_by_stem
        or entry["query_image_stem"] not in step_candidate_by_stem
    ]
    if missing:
        preview = ", ".join(str(v) for v in missing[:5])
        raise ValueError(f"missing refined entries for {len(missing)} base cache rows: {preview}")

    exported_entries: List[Dict] = []
    num_reference = 0
    num_candidate = 0
    num_base = 0
    accepted_steps_m: List[float] = []
    accepted_steps_rot_deg: List[float] = []
    refine_success: List[bool] = []
    refine_num_inliers: List[int] = []
    refine_num_raw_matches: List[int] = []

    for base_entry in base_entries:
        stem = str(base_entry["query_image_stem"])
        reference = reference_by_stem[stem]
        candidate = candidate_by_stem[stem]
        step_candidate = step_candidate_by_stem[stem]
        reference_ok = _passes_gate(
            reference,
            min_inliers=min_inliers,
            min_raw_matches=min_raw_matches,
            min_inlier_ratio=min_inlier_ratio,
        )
        candidate_ok = _passes_gate(
            candidate,
            min_inliers=min_inliers,
            min_raw_matches=min_raw_matches,
            min_inlier_ratio=min_inlier_ratio,
        )

        exported = _copy_entry(base_entry)
        selected_pose = None
        selected_source = None
        selected_quality = None
        selected_kind = "base"
        if reference_ok:
            selected_pose = reference["entry"]["pose_init"]
            selected_source = reference_source_name
            selected_quality = reference
            selected_kind = "reference"
            if candidate_ok:
                step_ok, step_m, step_rot_deg = _passes_pose_step_gate(
                    reference["entry"]["pose_init"],
                    step_candidate["entry"]["pose_init"],
                    min_step_m=min_step_m,
                    max_step_m=max_step_m,
                    min_rot_deg=min_rot_deg,
                    max_rot_deg=max_rot_deg,
                )
                if step_ok:
                    selected_pose = candidate["entry"]["pose_init"]
                    selected_source = candidate_source_name
                    selected_quality = candidate
                    selected_kind = "candidate"
                    accepted_steps_m.append(step_m)
                    accepted_steps_rot_deg.append(step_rot_deg)

        if selected_pose is not None:
            exported["pose_init"] = np.asarray(selected_pose, dtype=np.float32).copy()
            exported["init_source"] = str(selected_source)
        if selected_quality is not None:
            refine_success.append(bool(selected_quality["success"]))
            refine_num_inliers.append(int(selected_quality["num_inliers"]))
            refine_num_raw_matches.append(int(selected_quality["num_raw_matches"]))
        else:
            refine_success.append(False)
            refine_num_inliers.append(0)
            refine_num_raw_matches.append(0)

        if selected_kind == "candidate":
            num_candidate += 1
        elif selected_kind == "reference":
            num_reference += 1
        else:
            num_base += 1
        exported_entries.append(exported)

    exported_stats = dict(base_stats or {})
    exported_stats.update(
        {
            "selection_source": "pose_step_gated_refined_pose",
            "base_cache": str(base_cache_path),
            "reference_refined_cache": str(reference_refined_cache_path),
            "candidate_refined_cache": str(candidate_refined_cache_path),
            "step_candidate_refined_cache": str(step_candidate_refined_cache_path)
            if step_candidate_refined_cache_path is not None
            else str(candidate_refined_cache_path),
            "num_entries": len(exported_entries),
            "num_reference_selected": int(num_reference),
            "num_candidate_selected": int(num_candidate),
            "num_base_fallback": int(num_base),
            "min_step_m": float(min_step_m),
            "max_step_m": None if max_step_m is None else float(max_step_m),
            "min_rot_deg": float(min_rot_deg),
            "max_rot_deg": None if max_rot_deg is None else float(max_rot_deg),
            "min_inliers": int(min_inliers),
            "min_raw_matches": int(min_raw_matches),
            "min_inlier_ratio": float(min_inlier_ratio),
            "reference_source_name": str(reference_source_name),
            "candidate_source_name": str(candidate_source_name),
            "accepted_step_mean_m": float(np.mean(accepted_steps_m)) if accepted_steps_m else 0.0,
            "accepted_step_mean_rot_deg": float(np.mean(accepted_steps_rot_deg)) if accepted_steps_rot_deg else 0.0,
        }
    )
    _save_entries_with_refine_quality(
        exported_entries,
        exported_stats,
        str(save_path),
        refine_success=refine_success,
        refine_num_inliers=refine_num_inliers,
        refine_num_raw_matches=refine_num_raw_matches,
    )
    return exported_entries, exported_stats


def gate_refined_init_cache_by_score_delta(
    base_cache_path: str,
    reference_refined_cache_path: str,
    candidate_refined_cache_path: str,
    save_path: str,
    *,
    step_candidate_refined_cache_path: str | None = None,
    score_key: str,
    min_score_improvement: float = 0.0,
    max_candidate_score: float | None = None,
    min_step_m: float = 0.0,
    max_step_m: float | None = None,
    min_rot_deg: float = 0.0,
    max_rot_deg: float | None = None,
    min_inliers: int = 0,
    min_raw_matches: int = 0,
    min_inlier_ratio: float = 0.0,
    reference_source_name: str = "reference_refined_pose",
    candidate_source_name: str = "score_delta_gated_refined_pose",
) -> Tuple[List[Dict], Dict]:
    """Choose between refined caches using a lower-is-better scalar score.

    This gate is intended for deployable feature-consistency scores such as
    query/render POFD residuals. It preserves the base cache candidate bank and
    only swaps the active pose when the candidate quality gate passes and its
    score improves over the reference by ``min_score_improvement``.
    """
    base_entries, base_stats = load_retrieval_init_entries(str(base_cache_path))
    reference_by_stem = _load_refined_by_stem(str(reference_refined_cache_path))
    candidate_by_stem = _load_refined_by_stem(str(candidate_refined_cache_path))
    step_candidate_by_stem = (
        _load_refined_by_stem(str(step_candidate_refined_cache_path))
        if step_candidate_refined_cache_path is not None
        else candidate_by_stem
    )
    reference_scores = _load_score_by_stem(str(reference_refined_cache_path), score_key=score_key)
    candidate_scores = _load_score_by_stem(str(candidate_refined_cache_path), score_key=score_key)

    missing = [
        entry["query_image_stem"]
        for entry in base_entries
        if entry["query_image_stem"] not in reference_by_stem
        or entry["query_image_stem"] not in candidate_by_stem
        or entry["query_image_stem"] not in step_candidate_by_stem
        or entry["query_image_stem"] not in reference_scores
        or entry["query_image_stem"] not in candidate_scores
    ]
    if missing:
        preview = ", ".join(str(v) for v in missing[:5])
        raise ValueError(f"missing scored refined entries for {len(missing)} base cache rows: {preview}")

    exported_entries: List[Dict] = []
    num_reference = 0
    num_candidate = 0
    num_base = 0
    accepted_improvements: List[float] = []
    accepted_steps_m: List[float] = []
    accepted_steps_rot_deg: List[float] = []
    refine_success: List[bool] = []
    refine_num_inliers: List[int] = []
    refine_num_raw_matches: List[int] = []
    selected_scores: List[float] = []
    saved_reference_scores: List[float] = []
    saved_candidate_scores: List[float] = []

    for base_entry in base_entries:
        stem = str(base_entry["query_image_stem"])
        reference = reference_by_stem[stem]
        candidate = candidate_by_stem[stem]
        step_candidate = step_candidate_by_stem[stem]
        reference_score = float(reference_scores[stem])
        candidate_score = float(candidate_scores[stem])
        saved_reference_scores.append(reference_score)
        saved_candidate_scores.append(candidate_score)

        reference_ok = _passes_gate(
            reference,
            min_inliers=min_inliers,
            min_raw_matches=min_raw_matches,
            min_inlier_ratio=min_inlier_ratio,
        )
        candidate_ok = _passes_gate(
            candidate,
            min_inliers=min_inliers,
            min_raw_matches=min_raw_matches,
            min_inlier_ratio=min_inlier_ratio,
        )

        exported = _copy_entry(base_entry)
        selected_pose = None
        selected_source = None
        selected_quality = None
        selected_score = float("inf")
        selected_kind = "base"

        if reference_ok and np.isfinite(reference_score):
            selected_pose = reference["entry"]["pose_init"]
            selected_source = reference_source_name
            selected_quality = reference
            selected_score = reference_score
            selected_kind = "reference"

            improvement = reference_score - candidate_score
            candidate_score_ok = np.isfinite(candidate_score) and improvement >= float(min_score_improvement)
            if max_candidate_score is not None:
                candidate_score_ok = candidate_score_ok and candidate_score <= float(max_candidate_score)
            step_ok, step_m, step_rot_deg = _passes_pose_step_gate(
                reference["entry"]["pose_init"],
                step_candidate["entry"]["pose_init"],
                min_step_m=min_step_m,
                max_step_m=max_step_m,
                min_rot_deg=min_rot_deg,
                max_rot_deg=max_rot_deg,
            )
            if candidate_ok and candidate_score_ok and step_ok:
                selected_pose = candidate["entry"]["pose_init"]
                selected_source = candidate_source_name
                selected_quality = candidate
                selected_score = candidate_score
                selected_kind = "candidate"
                accepted_improvements.append(float(improvement))
                accepted_steps_m.append(step_m)
                accepted_steps_rot_deg.append(step_rot_deg)

        if selected_pose is not None:
            exported["pose_init"] = np.asarray(selected_pose, dtype=np.float32).copy()
            exported["init_source"] = str(selected_source)

        if selected_quality is not None:
            refine_success.append(bool(selected_quality["success"]))
            refine_num_inliers.append(int(selected_quality["num_inliers"]))
            refine_num_raw_matches.append(int(selected_quality["num_raw_matches"]))
        else:
            refine_success.append(False)
            refine_num_inliers.append(0)
            refine_num_raw_matches.append(0)
        selected_scores.append(selected_score)

        if selected_kind == "candidate":
            num_candidate += 1
        elif selected_kind == "reference":
            num_reference += 1
        else:
            num_base += 1
        exported_entries.append(exported)

    exported_stats = dict(base_stats or {})
    exported_stats.update(
        {
            "selection_source": "score_delta_gated_refined_pose",
            "base_cache": str(base_cache_path),
            "reference_refined_cache": str(reference_refined_cache_path),
            "candidate_refined_cache": str(candidate_refined_cache_path),
            "step_candidate_refined_cache": str(step_candidate_refined_cache_path)
            if step_candidate_refined_cache_path is not None
            else str(candidate_refined_cache_path),
            "num_entries": len(exported_entries),
            "num_reference_selected": int(num_reference),
            "num_candidate_selected": int(num_candidate),
            "num_base_fallback": int(num_base),
            "score_key": str(score_key),
            "min_score_improvement": float(min_score_improvement),
            "max_candidate_score": None if max_candidate_score is None else float(max_candidate_score),
            "min_step_m": float(min_step_m),
            "max_step_m": None if max_step_m is None else float(max_step_m),
            "min_rot_deg": float(min_rot_deg),
            "max_rot_deg": None if max_rot_deg is None else float(max_rot_deg),
            "min_inliers": int(min_inliers),
            "min_raw_matches": int(min_raw_matches),
            "min_inlier_ratio": float(min_inlier_ratio),
            "reference_source_name": str(reference_source_name),
            "candidate_source_name": str(candidate_source_name),
            "mean_candidate_score_improvement": float(np.mean(accepted_improvements))
            if accepted_improvements
            else 0.0,
            "accepted_step_mean_m": float(np.mean(accepted_steps_m)) if accepted_steps_m else 0.0,
            "accepted_step_mean_rot_deg": float(np.mean(accepted_steps_rot_deg))
            if accepted_steps_rot_deg
            else 0.0,
        }
    )
    _save_entries_with_refine_quality(
        exported_entries,
        exported_stats,
        str(save_path),
        refine_success=refine_success,
        refine_num_inliers=refine_num_inliers,
        refine_num_raw_matches=refine_num_raw_matches,
        refine_score_selected=selected_scores,
        refine_score_reference=saved_reference_scores,
        refine_score_candidate=saved_candidate_scores,
    )
    return exported_entries, exported_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_cache", required=True, help="Original retrieval/init cache .npz")
    parser.add_argument(
        "--refined_cache",
        required=True,
        help="External refined pose cache .npz. In pose-step mode this is the reference cache.",
    )
    parser.add_argument(
        "--candidate_refined_cache",
        default=None,
        help="Optional candidate refined cache. When set, export a pose-step-gated cache.",
    )
    parser.add_argument(
        "--step_candidate_refined_cache",
        default=None,
        help="Optional cache used only to measure the reference->candidate step.",
    )
    parser.add_argument("--save_path", required=True, help="Path for the gated output cache .npz")
    parser.add_argument("--min_inliers", type=int, default=0)
    parser.add_argument("--min_raw_matches", type=int, default=0)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.0)
    parser.add_argument("--source_name", default="gated_refined_pose")
    parser.add_argument("--min_step_m", type=float, default=0.0)
    parser.add_argument("--max_step_m", type=float, default=None)
    parser.add_argument("--min_rot_deg", type=float, default=0.0)
    parser.add_argument("--max_rot_deg", type=float, default=None)
    parser.add_argument("--reference_source_name", default="reference_refined_pose")
    parser.add_argument("--candidate_source_name", default="pose_step_gated_refined_pose")
    parser.add_argument(
        "--score_key",
        default=None,
        help="When set with --candidate_refined_cache, use lower-is-better score-delta gating instead of pose-step gating.",
    )
    parser.add_argument("--min_score_improvement", type=float, default=0.0)
    parser.add_argument("--max_candidate_score", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.candidate_refined_cache is not None and args.score_key is not None:
            _entries, stats = gate_refined_init_cache_by_score_delta(
                args.base_cache,
                args.refined_cache,
                args.candidate_refined_cache,
                args.save_path,
                step_candidate_refined_cache_path=args.step_candidate_refined_cache,
                score_key=args.score_key,
                min_score_improvement=args.min_score_improvement,
                max_candidate_score=args.max_candidate_score,
                min_step_m=args.min_step_m,
                max_step_m=args.max_step_m,
                min_rot_deg=args.min_rot_deg,
                max_rot_deg=args.max_rot_deg,
                min_inliers=args.min_inliers,
                min_raw_matches=args.min_raw_matches,
                min_inlier_ratio=args.min_inlier_ratio,
                reference_source_name=args.reference_source_name,
                candidate_source_name=args.candidate_source_name,
            )
        elif args.candidate_refined_cache is not None:
            _entries, stats = gate_refined_init_cache_by_pose_step(
                args.base_cache,
                args.refined_cache,
                args.candidate_refined_cache,
                args.save_path,
                step_candidate_refined_cache_path=args.step_candidate_refined_cache,
                min_step_m=args.min_step_m,
                max_step_m=args.max_step_m,
                min_rot_deg=args.min_rot_deg,
                max_rot_deg=args.max_rot_deg,
                min_inliers=args.min_inliers,
                min_raw_matches=args.min_raw_matches,
                min_inlier_ratio=args.min_inlier_ratio,
                reference_source_name=args.reference_source_name,
                candidate_source_name=args.candidate_source_name,
            )
        else:
            _entries, stats = gate_refined_init_cache(
                args.base_cache,
                args.refined_cache,
                args.save_path,
                min_inliers=args.min_inliers,
                min_raw_matches=args.min_raw_matches,
                min_inlier_ratio=args.min_inlier_ratio,
                source_name=args.source_name,
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.candidate_refined_cache is not None and args.score_key is not None:
        print(
            f"Exported {stats['num_entries']} score-delta-gated init entries to {args.save_path} "
            f"(reference={stats['num_reference_selected']}, "
            f"candidate={stats['num_candidate_selected']}, "
            f"base={stats['num_base_fallback']})"
        )
    elif args.candidate_refined_cache is not None:
        print(
            f"Exported {stats['num_entries']} pose-step-gated init entries to {args.save_path} "
            f"(reference={stats['num_reference_selected']}, "
            f"candidate={stats['num_candidate_selected']}, "
            f"base={stats['num_base_fallback']})"
        )
    else:
        print(
            f"Exported {stats['num_entries']} gated init entries to {args.save_path} "
            f"(accepted_refined={stats['num_refined_accepted']}, "
            f"rejected={stats['num_refined_rejected']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
