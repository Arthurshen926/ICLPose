#!/usr/bin/env python3
"""Utilities for the localization-oriented feature reconstruction mainline.

The project mainline treats initialization as a fixed evaluation condition:
build or inspect one deterministic real-init cache, then compare feature-map
and refinement components under that same cache.  This module intentionally
does not train another initializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.radio_loc_retrieval_dataset import (
    list_colmap_split_samples,
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)


def sha256_file(path: str) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    return (-R.T @ t).astype(np.float64)


def pose_error(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> Tuple[float, float]:
    """Return rotation error in degrees and translation error in millimeters."""
    pred = np.asarray(pred_w2c, dtype=np.float64)
    gt = np.asarray(gt_w2c, dtype=np.float64)
    R_rel = pred[:3, :3].T @ gt[:3, :3]
    trace = float(np.trace(R_rel))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    trans_mm = float(np.linalg.norm(camera_center_from_w2c(pred) - camera_center_from_w2c(gt)) * 1000.0)
    return rot_deg, trans_mm


def summarize_pose_metrics(rot_errs: Sequence[float], trans_errs: Sequence[float]) -> Dict[str, float]:
    """Paper-facing pose metrics used by fixed-init and ablation reports."""
    rot = np.asarray(rot_errs, dtype=np.float64)
    trans = np.asarray(trans_errs, dtype=np.float64)
    if rot.size == 0 or trans.size == 0:
        raise ValueError("Cannot summarize empty pose error arrays")
    metrics = {
        "rot_mean": float(np.nanmean(rot)),
        "rot_median": float(np.nanmedian(rot)),
        "trans_mean": float(np.nanmean(trans)),
        "trans_median": float(np.nanmedian(trans)),
        "pct_1deg": float(np.mean(rot < 1.0) * 100.0),
        "pct_5deg": float(np.mean(rot < 5.0) * 100.0),
        "pct_10deg": float(np.mean(rot < 10.0) * 100.0),
        "pct_50mm": float(np.mean(trans < 50.0) * 100.0),
        "pct_100mm": float(np.mean(trans < 100.0) * 100.0),
        "pct_250mm": float(np.mean(trans < 250.0) * 100.0),
        "pct_1000mm": float(np.mean(trans < 1000.0) * 100.0),
        "pct_2000mm": float(np.mean(trans < 2000.0) * 100.0),
        "joint_1deg_50mm": float(np.mean((rot < 1.0) & (trans < 50.0)) * 100.0),
        "joint_1deg_100mm": float(np.mean((rot < 1.0) & (trans < 100.0)) * 100.0),
        "joint_2deg_100mm": float(np.mean((rot < 2.0) & (trans < 100.0)) * 100.0),
        "joint_5deg_250mm": float(np.mean((rot < 5.0) & (trans < 250.0)) * 100.0),
        "joint_5deg_1000mm": float(np.mean((rot < 5.0) & (trans < 1000.0)) * 100.0),
        "joint_10deg_2000mm": float(np.mean((rot < 10.0) & (trans < 2000.0)) * 100.0),
    }
    metrics["recall_1deg_50mm"] = metrics["joint_1deg_50mm"]
    metrics["recall_1deg_100mm"] = metrics["joint_1deg_100mm"]
    metrics["recall_2deg_100mm"] = metrics["joint_2deg_100mm"]
    metrics["recall_5deg_250mm"] = metrics["joint_5deg_250mm"]
    return metrics


def _as_bool_mask(entry: Mapping, length: int) -> np.ndarray:
    if "candidate_valid_mask" not in entry:
        return np.ones((length,), dtype=bool)
    mask = np.asarray(entry["candidate_valid_mask"], dtype=bool).reshape(-1)
    if len(mask) < length:
        padded = np.zeros((length,), dtype=bool)
        padded[: len(mask)] = mask
        return padded
    return mask[:length]


def _entry_candidates(entry: Mapping, *, source: Optional[str] = None) -> List[Dict]:
    poses = np.asarray(entry.get("pose_init_candidates", np.asarray(entry["pose_init"])[None]), dtype=np.float32)
    if poses.ndim == 2:
        poses = poses[None]
    valid = _as_bool_mask(entry, len(poses))
    frame_ids = np.asarray(
        entry.get("retrieval_frame_ids_candidates", [entry.get("retrieval_frame_id", -1)] * len(poses)),
        dtype=np.int64,
    )
    names = list(entry.get("retrieval_image_names_candidates", [entry.get("retrieval_image_name", "")] * len(poses)))
    scores = np.asarray(
        entry.get("retrieval_scores_candidates", [entry.get("retrieval_score", float("nan"))] * len(poses)),
        dtype=np.float32,
    )
    candidates: List[Dict] = []
    for idx in range(len(poses)):
        if not bool(valid[idx]):
            continue
        candidates.append(
            {
                "pose_w2c": poses[idx].astype(np.float32),
                "retrieval_frame_id": int(frame_ids[idx]) if idx < len(frame_ids) else -1,
                "retrieval_image_name": str(names[idx]) if idx < len(names) else "",
                "retrieval_score": float(scores[idx]) if idx < len(scores) else float("nan"),
                "source": str(source or entry.get("init_source", "")),
            }
        )
    if not candidates:
        candidates.append(
            {
                "pose_w2c": np.asarray(entry["pose_init"], dtype=np.float32),
                "retrieval_frame_id": int(entry.get("retrieval_frame_id", -1)),
                "retrieval_image_name": str(entry.get("retrieval_image_name", "")),
                "retrieval_score": float(entry.get("retrieval_score", float("nan"))),
                "source": str(source or entry.get("init_source", "")),
            }
        )
    return candidates


def _by_query_name(entries: Optional[Sequence[Mapping]]) -> Dict[str, Mapping]:
    if not entries:
        return {}
    return {str(entry["query_image_name"]): entry for entry in entries}


def _ordered_query_names(*entry_groups: Optional[Sequence[Mapping]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for entries in entry_groups:
        if not entries:
            continue
        for entry in entries:
            name = str(entry["query_image_name"])
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def _is_external_success(entry: Optional[Mapping], *, external_min_score: Optional[float]) -> bool:
    if entry is None:
        return False
    source = str(entry.get("init_source", ""))
    source_lower = source.lower()
    if source_lower.startswith("fallback_") or source_lower.endswith("_fallback"):
        return False
    if "retrieval_fallback" in source_lower:
        return False
    if external_min_score is None:
        return True
    score = float(entry.get("retrieval_score", float("nan")))
    if not math.isfinite(score):
        return True
    return score >= float(external_min_score)


def _pad_or_clip_candidates(candidates: List[Dict], topk: int) -> Tuple[List[Dict], np.ndarray]:
    if not candidates:
        raise ValueError("Cannot build an init entry without candidates")
    topk = max(1, int(topk))
    real = candidates[:topk]
    valid = np.zeros((topk,), dtype=bool)
    valid[: len(real)] = True
    pad_source = real[-1]
    while len(real) < topk:
        real.append(dict(pad_source))
    return real, valid


def build_hybrid_init_entries(
    *,
    external_entries: Sequence[Mapping],
    learned_entries: Optional[Sequence[Mapping]] = None,
    raw_fallback_entries: Optional[Sequence[Mapping]] = None,
    topk: int = 10,
    external_min_score: Optional[float] = 1.0,
    source_name: str = "fixed_hybrid_external_first",
) -> Tuple[List[Dict], Dict]:
    """Merge init caches into one fixed external-first protocol.

    External PnP/render-LoFTR candidates win when their top-1 source is not a
    fallback and the score passes ``external_min_score``.  Learned and raw
    candidates are retained only as deterministic fallbacks/extra hypotheses.
    """
    external_by_name = _by_query_name(external_entries)
    learned_by_name = _by_query_name(learned_entries)
    raw_by_name = _by_query_name(raw_fallback_entries)
    query_names = _ordered_query_names(external_entries, learned_entries, raw_fallback_entries)
    if not query_names:
        raise ValueError("No entries were provided to build a hybrid init cache")

    entries: List[Dict] = []
    counts_by_selected_source: Dict[str, int] = {}
    num_external_success = 0
    num_learned_fallback = 0
    num_raw_fallback = 0

    for query_name in query_names:
        external = external_by_name.get(query_name)
        learned = learned_by_name.get(query_name)
        raw = raw_by_name.get(query_name)
        external_success = _is_external_success(external, external_min_score=external_min_score)

        ordered_candidates: List[Dict] = []
        if external_success and external is not None:
            ordered_candidates.extend(_entry_candidates(external))
            if learned is not None:
                ordered_candidates.extend(_entry_candidates(learned))
            if raw is not None:
                ordered_candidates.extend(_entry_candidates(raw))
            source_entry = external
            num_external_success += 1
        else:
            if learned is not None:
                ordered_candidates.extend(_entry_candidates(learned))
                source_entry = learned
                num_learned_fallback += 1
            elif raw is not None:
                ordered_candidates.extend(_entry_candidates(raw))
                source_entry = raw
                num_raw_fallback += 1
            elif external is not None:
                ordered_candidates.extend(_entry_candidates(external))
                source_entry = external
                num_raw_fallback += 1
            else:
                raise ValueError(f"No usable init candidates for query {query_name}")
            if external is not None:
                ordered_candidates.extend(_entry_candidates(external))
            if raw is not None and source_entry is not raw:
                ordered_candidates.extend(_entry_candidates(raw))

        candidates, valid_mask = _pad_or_clip_candidates(ordered_candidates, topk=topk)
        best = candidates[0]
        selected_source = str(best["source"] or source_entry.get("init_source", source_name))
        counts_by_selected_source[selected_source] = counts_by_selected_source.get(selected_source, 0) + 1
        query_stem = str(source_entry.get("query_image_stem", Path(query_name).with_suffix("").as_posix().replace("/", "_")))
        entries.append(
            {
                "query_img_id": int(source_entry.get("query_img_id", -1)),
                "query_image_name": query_name,
                "query_image_stem": query_stem,
                "pose_init": np.asarray(best["pose_w2c"], dtype=np.float32),
                "init_source": selected_source,
                "retrieval_frame_id": int(best["retrieval_frame_id"]),
                "retrieval_image_name": str(best["retrieval_image_name"]),
                "retrieval_score": float(best["retrieval_score"]),
                "pose_init_candidates": np.stack([c["pose_w2c"] for c in candidates], axis=0).astype(np.float32),
                "candidate_valid_mask": valid_mask.astype(bool),
                "retrieval_frame_ids_candidates": np.asarray(
                    [int(c["retrieval_frame_id"]) for c in candidates], dtype=np.int64
                ),
                "retrieval_image_names_candidates": np.asarray(
                    [str(c["retrieval_image_name"]) for c in candidates]
                ),
                "retrieval_scores_candidates": np.asarray(
                    [float(c["retrieval_score"]) for c in candidates], dtype=np.float32
                ),
            }
        )

    stats = {
        "method_requested": "fixed_hybrid_external_first",
        "method_used": source_name,
        "retrieval_topk_requested": int(max(1, topk)),
        "external_min_score": None if external_min_score is None else float(external_min_score),
        "num_query_samples": int(len(entries)),
        "num_external_entries": int(len(external_by_name)),
        "num_learned_entries": int(len(learned_by_name)),
        "num_raw_fallback_entries": int(len(raw_by_name)),
        "num_selected_external_success": int(num_external_success),
        "num_selected_learned_fallback": int(num_learned_fallback),
        "num_selected_raw_or_external_fallback": int(num_raw_fallback),
        "counts_by_selected_source": counts_by_selected_source,
        "counts_by_source": counts_by_selected_source,
    }
    return entries, stats


def write_hybrid_init_cache(
    *,
    external_init_path: str,
    save_path: str,
    learned_init_path: Optional[str] = None,
    raw_fallback_init_path: Optional[str] = None,
    topk: int = 10,
    external_min_score: Optional[float] = 1.0,
    source_name: str = "fixed_hybrid_external_first",
) -> Dict:
    external_entries, external_stats = load_retrieval_init_entries(external_init_path)
    learned_entries: Optional[List[Dict]] = None
    learned_stats: Dict = {}
    raw_entries: Optional[List[Dict]] = None
    raw_stats: Dict = {}
    if learned_init_path:
        learned_entries, learned_stats = load_retrieval_init_entries(learned_init_path)
    if raw_fallback_init_path:
        raw_entries, raw_stats = load_retrieval_init_entries(raw_fallback_init_path)
    entries, stats = build_hybrid_init_entries(
        external_entries=external_entries,
        learned_entries=learned_entries,
        raw_fallback_entries=raw_entries,
        topk=topk,
        external_min_score=external_min_score,
        source_name=source_name,
    )
    stats.update(
        {
            "external_init_path": str(external_init_path),
            "external_init_sha256": sha256_file(external_init_path),
            "external_init_stats": external_stats,
            "learned_init_path": str(learned_init_path) if learned_init_path else None,
            "learned_init_sha256": sha256_file(learned_init_path) if learned_init_path else None,
            "learned_init_stats": learned_stats,
            "raw_fallback_init_path": str(raw_fallback_init_path) if raw_fallback_init_path else None,
            "raw_fallback_init_sha256": sha256_file(raw_fallback_init_path) if raw_fallback_init_path else None,
            "raw_fallback_init_stats": raw_stats,
        }
    )
    save_retrieval_init_entries(entries, stats, save_path)
    stats["fixed_init_cache_path"] = str(save_path)
    stats["fixed_init_cache_sha256"] = sha256_file(save_path)
    return stats


def summarize_init_entries(
    entries: Sequence[Mapping],
    *,
    gt_poses_by_name: Mapping[str, np.ndarray],
    cache_path: Optional[str] = None,
) -> Dict:
    rot_errs: List[float] = []
    trans_errs: List[float] = []
    missing_gt: List[str] = []
    for entry in entries:
        name = str(entry["query_image_name"])
        if name not in gt_poses_by_name:
            missing_gt.append(name)
            continue
        rot, trans = pose_error(np.asarray(entry["pose_init"], dtype=np.float32), gt_poses_by_name[name])
        rot_errs.append(rot)
        trans_errs.append(trans)
    if not rot_errs:
        raise ValueError("No init entries matched the provided ground-truth poses")
    summary = {
        "protocol": "fixed_init_cache",
        "num_samples": int(len(rot_errs)),
        "num_entries": int(len(entries)),
        "num_missing_gt": int(len(missing_gt)),
        "missing_gt": missing_gt[:20],
        "init_metrics": summarize_pose_metrics(rot_errs, trans_errs),
    }
    if cache_path:
        summary["init_cache_path"] = str(cache_path)
        summary["init_cache_sha256"] = sha256_file(cache_path)
    return summary


def summarize_init_cache(
    *,
    init_cache_path: str,
    gt_poses_by_name: Optional[Mapping[str, np.ndarray]] = None,
    colmap_dir: Optional[str] = None,
    query_split: Optional[str] = None,
) -> Dict:
    entries, stats = load_retrieval_init_entries(init_cache_path)
    if gt_poses_by_name is None:
        if not colmap_dir or not query_split:
            raise ValueError("Provide either gt_poses_by_name or both colmap_dir and query_split")
        samples = list_colmap_split_samples(colmap_dir, query_split)
        gt_poses_by_name = {str(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32) for sample in samples}
    summary = summarize_init_entries(entries, gt_poses_by_name=gt_poses_by_name, cache_path=init_cache_path)
    summary["dataset_init_stats"] = stats
    return summary


def _metric_block(summary: Mapping, key: str) -> Mapping:
    if key == "init":
        return summary.get("init_metrics") or summary.get("full_pipeline_metrics", {}).get("init") or {}
    if key == "final":
        return summary.get("final_metrics") or summary.get("full_pipeline_metrics", {}).get("final") or {}
    return {}


def _metric(metrics: Mapping, *names: str) -> Optional[float]:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return float(metrics[name])
    return None


def _summary_cache_hash(summary: Mapping) -> Optional[str]:
    if summary.get("init_cache_sha256"):
        return str(summary["init_cache_sha256"])
    stats = summary.get("dataset_init_stats") or {}
    if isinstance(stats, Mapping) and stats.get("fixed_init_cache_sha256"):
        return str(stats["fixed_init_cache_sha256"])
    path = summary.get("init_pose_cache")
    if path and os.path.isfile(str(path)):
        return sha256_file(str(path))
    return None


def _summary_cache_path(summary: Mapping) -> Optional[str]:
    path = summary.get("init_pose_cache") or summary.get("init_cache_path")
    if path:
        return str(path)
    stats = summary.get("dataset_init_stats") or {}
    if isinstance(stats, Mapping) and stats.get("fixed_init_cache_path"):
        return str(stats["fixed_init_cache_path"])
    return None


def _feature_metric(summary: Mapping, name: str) -> Optional[float]:
    containers = [
        summary.get("feature_metrics"),
        summary.get("localization_feature_metrics"),
        summary.get("intermediate_metrics"),
        summary.get("final_metrics"),
        summary,
    ]
    aliases = {
        "feature_reconstruction_cosine": [
            "feature_reconstruction_cosine",
            "reconstruction_cosine",
            "map_feature_reconstruction_cosine",
            "loc_feature_cosine",
        ],
        "flow_epe": ["flow_epe", "flow_epe_px", "val_flow_epe"],
        "peak_acc": ["peak_acc", "peak_accuracy", "matching_peak_acc"],
        "peak_pos": ["peak_pos", "matching_peak_pos", "map_query_corr_peak_pos"],
        "peak_neg": ["peak_neg", "matching_peak_neg", "map_query_corr_peak_neg"],
        "wls_gain_mm": ["wls_gain_mm", "wls_pose_gain_mm", "geometry_wls_gain_mm"],
        "featuremetric_gain_mm": [
            "featuremetric_gain_mm",
            "feature_metric_gain_mm",
            "featuremetric_pose_gain_mm",
            "map_feature_metric_trans_gain_mm",
        ],
    }
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        value = _metric(container, *aliases.get(name, [name]))
        if value is not None:
            return value
    return None


def build_ablation_report(
    run_summaries: Sequence[Mapping],
    *,
    fixed_init_sha256: Optional[str] = None,
    fixed_init_path: Optional[str] = None,
) -> Dict:
    """Build an ablation table and enforce that all runs use one init cache."""
    if not run_summaries:
        raise ValueError("At least one run summary is required")

    observed_hashes = []
    observed_paths = []
    for summary in run_summaries:
        cache_hash = _summary_cache_hash(summary)
        cache_path = _summary_cache_path(summary)
        if cache_hash:
            observed_hashes.append(cache_hash)
        if cache_path:
            observed_paths.append(cache_path)

    expected_hash = fixed_init_sha256 or (observed_hashes[0] if observed_hashes else None)
    expected_path = fixed_init_path or (observed_paths[0] if observed_paths else None)
    for cache_hash in observed_hashes:
        if expected_hash and cache_hash != expected_hash:
            raise ValueError("All ablation runs must use the same fixed init cache SHA256")
    for cache_path in observed_paths:
        if expected_path and not expected_hash and cache_path != expected_path:
            raise ValueError("All ablation runs must use the same fixed init cache path")

    rows: List[Dict] = []
    for idx, summary in enumerate(run_summaries):
        init = _metric_block(summary, "init")
        final = _metric_block(summary, "final")
        init_rot = _metric(init, "rot_median", "rot_median_deg")
        init_trans = _metric(init, "trans_median", "trans_median_mm")
        final_rot = _metric(final, "rot_median", "rot_median_deg")
        final_trans = _metric(final, "trans_median", "trans_median_mm")
        row = {
            "name": str(summary.get("name") or summary.get("run_name") or f"run_{idx:02d}"),
            "component": str(summary.get("component") or summary.get("ablation_component") or ""),
            "init_cache_path": _summary_cache_path(summary),
            "init_cache_sha256": _summary_cache_hash(summary),
            "init_rot_median_deg": init_rot,
            "init_trans_median_mm": init_trans,
            "final_rot_median_deg": final_rot,
            "final_trans_median_mm": final_trans,
            "rot_median_gain_deg": None if init_rot is None or final_rot is None else float(init_rot - final_rot),
            "trans_median_gain_mm": None if init_trans is None or final_trans is None else float(init_trans - final_trans),
            "init_recall_1deg_50mm": _metric(init, "joint_1deg_50mm", "recall_1deg_50mm"),
            "init_recall_1deg_100mm": _metric(init, "joint_1deg_100mm", "recall_1deg_100mm"),
            "final_recall_1deg_50mm": _metric(final, "joint_1deg_50mm", "recall_1deg_50mm"),
            "final_recall_1deg_100mm": _metric(final, "joint_1deg_100mm", "recall_1deg_100mm"),
            "final_recall_2deg_100mm": _metric(final, "joint_2deg_100mm", "recall_2deg_100mm"),
            "final_recall_5deg_250mm": _metric(final, "joint_5deg_250mm", "recall_5deg_250mm"),
            "flow_epe": _feature_metric(summary, "flow_epe"),
            "peak_acc": _feature_metric(summary, "peak_acc"),
            "peak_pos": _feature_metric(summary, "peak_pos"),
            "peak_neg": _feature_metric(summary, "peak_neg"),
            "wls_gain_mm": _feature_metric(summary, "wls_gain_mm"),
            "featuremetric_gain_mm": _feature_metric(summary, "featuremetric_gain_mm"),
            "feature_reconstruction_cosine": _feature_metric(summary, "feature_reconstruction_cosine"),
        }
        peak_ok = (
            row["peak_pos"] is None
            or row["peak_neg"] is None
            or row["peak_pos"] > row["peak_neg"]
        )
        geometry_gain_ok = (
            row["wls_gain_mm"] is None
            or row["wls_gain_mm"] > 0.0
        ) and (
            row["featuremetric_gain_mm"] is None
            or row["featuremetric_gain_mm"] > 0.0
        )
        pose_gain_positive = bool(
            (row["trans_median_gain_mm"] is not None and row["trans_median_gain_mm"] > 0.0)
            or (row["rot_median_gain_deg"] is not None and row["rot_median_gain_deg"] > 0.0)
        )
        row["core_claim_supported"] = bool(pose_gain_positive and peak_ok and geometry_gain_ok)
        rows.append(row)

    return {
        "protocol": "localization_feature_reconstruction_ablation",
        "fixed_init_cache": {
            "path": expected_path,
            "sha256": expected_hash,
        },
        "num_runs": int(len(rows)),
        "runs": rows,
    }


def load_run_summaries(paths: Iterable[str]) -> List[Dict]:
    summaries: List[Dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary.setdefault("summary_path", str(path))
        summary.setdefault("name", Path(path).parent.name or Path(path).stem)
        summaries.append(summary)
    return summaries


def ablation_report_to_markdown(report: Mapping) -> str:
    headers = [
        "run",
        "component",
        "init med",
        "final med",
        "gain",
        "R@1deg/50mm",
        "R@1deg/100mm",
        "flow EPE",
        "peak acc",
        "WLS gain",
        "recon cos",
    ]
    lines = [
        "# Localization Feature Reconstruction Ablation",
        "",
        f"- fixed init cache: `{report.get('fixed_init_cache', {}).get('path')}`",
        f"- fixed init sha256: `{report.get('fixed_init_cache', {}).get('sha256')}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in report.get("runs", []):
        init_med = _fmt_pair(row.get("init_rot_median_deg"), row.get("init_trans_median_mm"))
        final_med = _fmt_pair(row.get("final_rot_median_deg"), row.get("final_trans_median_mm"))
        gain = _fmt_pair(row.get("rot_median_gain_deg"), row.get("trans_median_gain_mm"))
        values = [
            str(row.get("name", "")),
            str(row.get("component", "")),
            init_med,
            final_med,
            gain,
            _fmt_scalar(row.get("final_recall_1deg_50mm")),
            _fmt_scalar(row.get("final_recall_1deg_100mm")),
            _fmt_scalar(row.get("flow_epe")),
            _fmt_scalar(row.get("peak_acc")),
            _fmt_scalar(row.get("wls_gain_mm")),
            _fmt_scalar(row.get("feature_reconstruction_cosine")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _fmt_scalar(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _fmt_pair(rot, trans) -> str:
    if rot is None or trans is None:
        return ""
    return f"{float(rot):.3f}deg / {float(trans):.1f}mm"


def _load_gt_json(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "poses" in payload:
        payload = payload["poses"]
    return {str(name): np.asarray(pose, dtype=np.float32) for name, pose in payload.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-init localization mainline utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-hybrid-init", help="Build an external-first fixed init cache")
    build.add_argument("--external_init", required=True)
    build.add_argument("--learned_init", default=None)
    build.add_argument("--raw_fallback_init", default=None)
    build.add_argument("--save_path", required=True)
    build.add_argument("--topk", type=int, default=10)
    build.add_argument("--external_min_score", type=float, default=1.0)
    build.add_argument("--source_name", default="fixed_hybrid_external_first")

    inspect = sub.add_parser("inspect-init", help="Report hash and init metrics for a fixed cache")
    inspect.add_argument("--init_cache", required=True)
    inspect.add_argument("--gt_json", default=None)
    inspect.add_argument("--colmap_dir", default=None)
    inspect.add_argument("--query_split", default=None)
    inspect.add_argument("--output_json", default=None)

    report = sub.add_parser("report-ablation", help="Build the final localization ablation report")
    report.add_argument("--summary_json", nargs="+", required=True)
    report.add_argument("--fixed_init_sha256", default=None)
    report.add_argument("--fixed_init_path", default=None)
    report.add_argument("--output_json", default=None)
    report.add_argument("--output_md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-hybrid-init":
        stats = write_hybrid_init_cache(
            external_init_path=args.external_init,
            learned_init_path=args.learned_init,
            raw_fallback_init_path=args.raw_fallback_init,
            save_path=args.save_path,
            topk=args.topk,
            external_min_score=args.external_min_score,
            source_name=args.source_name,
        )
        print(json.dumps(stats, indent=2))
        return

    if args.command == "inspect-init":
        gt_by_name = _load_gt_json(args.gt_json) if args.gt_json else None
        summary = summarize_init_cache(
            init_cache_path=args.init_cache,
            gt_poses_by_name=gt_by_name,
            colmap_dir=args.colmap_dir,
            query_split=args.query_split,
        )
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "report-ablation":
        summaries = load_run_summaries(args.summary_json)
        report = build_ablation_report(
            summaries,
            fixed_init_sha256=args.fixed_init_sha256,
            fixed_init_path=args.fixed_init_path,
        )
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        if args.output_md:
            Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_md, "w", encoding="utf-8") as handle:
                handle.write(ablation_report_to_markdown(report))
        print(json.dumps(report, indent=2))
        return


if __name__ == "__main__":
    main()
