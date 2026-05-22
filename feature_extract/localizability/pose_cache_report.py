"""Pose-cache comparison utilities for solver handoff localization reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_retrieval.localization_mainline import pose_error, sha256_file, summarize_pose_metrics


def query_names_from_cache(cache_path: str | Path) -> list[str]:
    """Return query image names in the stored cache order."""
    entries, _stats = load_retrieval_init_entries(str(cache_path))
    return [str(entry["query_image_name"]) for entry in entries]


def _refine_success_by_name(cache_path: str | Path) -> dict[str, bool] | None:
    data = np.load(str(cache_path), allow_pickle=True)
    if "refine_success" not in data.files or "query_image_names" not in data.files:
        return None
    names = [str(name) for name in data["query_image_names"]]
    success = np.asarray(data["refine_success"], dtype=bool).reshape(-1)
    return {name: bool(success[idx]) for idx, name in enumerate(names[: len(success)])}


def summarize_pose_cache_for_queries(
    *,
    cache_path: str | Path,
    label: str,
    gt_poses_by_name: Mapping[str, np.ndarray],
    query_names: Sequence[str] | None = None,
) -> dict:
    """Summarize one pose cache on either all entries or an explicit query subset."""
    entries, stats = load_retrieval_init_entries(str(cache_path))
    by_name = {str(entry["query_image_name"]): entry for entry in entries}
    ordered_names = [str(name) for name in query_names] if query_names is not None else list(by_name.keys())

    matched_names: list[str] = []
    missing_names: list[str] = []
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    sources: dict[str, int] = {}
    for name in ordered_names:
        entry = by_name.get(name)
        gt_pose = gt_poses_by_name.get(name)
        if entry is None or gt_pose is None:
            missing_names.append(name)
            continue
        rot, trans = pose_error(np.asarray(entry["pose_init"], dtype=np.float32), gt_pose)
        rot_errs.append(rot)
        trans_errs.append(trans)
        matched_names.append(name)
        source = str(entry.get("init_source", ""))
        sources[source] = sources.get(source, 0) + 1

    if not matched_names:
        raise ValueError(f"No query names from {cache_path} matched the provided ground truth")

    metrics = summarize_pose_metrics(rot_errs, trans_errs)
    refine_success = _refine_success_by_name(cache_path)
    solver_success_frac = None
    if refine_success is not None:
        solver_success_frac = float(np.mean([refine_success.get(name, False) for name in matched_names]))

    return {
        "label": str(label),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(str(cache_path)),
        "num_samples": int(len(matched_names)),
        "num_cache_entries": int(len(entries)),
        "num_missing": int(len(missing_names)),
        "missing_names": missing_names[:20],
        "metrics": metrics,
        "solver_success_frac": solver_success_frac,
        "sources": sources,
        "dataset_init_stats": stats,
    }


def build_pose_cache_comparison(
    *,
    caches: Sequence[tuple[str, str | Path]],
    gt_poses_by_name: Mapping[str, np.ndarray],
    query_names: Sequence[str] | None = None,
    protocol: str = "pose_cache_solver_handoff_comparison",
) -> dict:
    """Build a comparable real-pose report for multiple init/refined caches."""
    return {
        "protocol": str(protocol),
        "num_caches": int(len(caches)),
        "query_subset_size": None if query_names is None else int(len(query_names)),
        "rows": [
            summarize_pose_cache_for_queries(
                cache_path=cache_path,
                label=label,
                gt_poses_by_name=gt_poses_by_name,
                query_names=query_names,
            )
            for label, cache_path in caches
        ],
    }


def pose_cache_comparison_to_markdown(report: Mapping) -> str:
    """Render a compact markdown table for a cache comparison report."""
    lines = [
        "# Solver Handoff Localization Report",
        "",
        f"- protocol: `{report.get('protocol')}`",
        f"- query subset size: `{report.get('query_subset_size')}`",
        "",
        "| method | n | rot med deg | trans med mm | trans mean mm | R@1deg/50mm | R@1deg/100mm | R@5deg/250mm | solver success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows", []):
        metrics = row.get("metrics", {})
        success = row.get("solver_success_frac")
        success_text = "" if success is None else f"{float(success) * 100.0:.1f}%"
        lines.append(
            "| {label} | {n:d} | {rot:.3f} | {trans_med:.1f} | {trans_mean:.1f} | "
            "{r50:.1f} | {r100:.1f} | {r250:.1f} | {success} |".format(
                label=str(row.get("label", "")),
                n=int(row.get("num_samples", 0)),
                rot=float(metrics.get("rot_median", float("nan"))),
                trans_med=float(metrics.get("trans_median", float("nan"))),
                trans_mean=float(metrics.get("trans_mean", float("nan"))),
                r50=float(metrics.get("joint_1deg_50mm", float("nan"))),
                r100=float(metrics.get("joint_1deg_100mm", float("nan"))),
                r250=float(metrics.get("joint_5deg_250mm", float("nan"))),
                success=success_text,
            )
        )
    return "\n".join(lines) + "\n"
