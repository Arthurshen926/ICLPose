"""Pose-cache comparison utilities for solver handoff localization reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_retrieval.localization_mainline import pose_error, sha256_file, summarize_pose_metrics


def query_names_from_cache(cache_path: str | Path) -> list[str]:
    """Return query image names in the stored cache order."""
    entries, _stats = load_retrieval_init_entries(str(cache_path))
    return [str(entry["query_image_name"]) for entry in entries]


def query_names_from_candidate_table(table_path: str | Path) -> list[str]:
    """Return unique sample names from a candidate-table JSONL in file order."""
    names: list[str] = []
    seen: set[str] = set()
    with Path(table_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["sample_name"])
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


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


def _safe_quantiles(values: Sequence[float], quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in quantiles}
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in quantiles}


def summarize_pose_candidate_cache_geometry(
    *,
    cache_path: str | Path,
    label: str,
    gt_poses_by_name: Mapping[str, np.ndarray],
    query_names: Sequence[str] | None = None,
    rot_cost_weight: float = 0.1,
    basin_trans_m: float = 0.25,
    basin_rot_deg: float = 10.0,
    topk: Sequence[int] = (1, 5, 10, 20),
) -> dict:
    """Summarize pose-candidate geometry without using learned scorer outputs."""

    data = np.load(str(cache_path), allow_pickle=True)
    if "pose_init_candidates" not in data.files or "query_image_names" not in data.files:
        raise ValueError(f"{cache_path} is missing pose_init_candidates/query_image_names")
    names = [str(name) for name in data["query_image_names"]]
    candidates = np.asarray(data["pose_init_candidates"], dtype=np.float32)
    if candidates.ndim != 4 or candidates.shape[-2:] != (4, 4):
        raise ValueError("pose_init_candidates must have shape (N,K,4,4)")
    valid = (
        np.asarray(data["candidate_valid_mask"], dtype=bool)
        if "candidate_valid_mask" in data.files
        else np.ones(candidates.shape[:2], dtype=bool)
    )
    if valid.shape != candidates.shape[:2]:
        raise ValueError("candidate_valid_mask must have shape (N,K)")
    ordered_names = [str(name) for name in query_names] if query_names is not None else names
    index_by_name = {name: idx for idx, name in enumerate(names)}

    topk = tuple(sorted({int(k) for k in topk if int(k) > 0}))
    matched_names: list[str] = []
    missing_names: list[str] = []
    oracle_trans: list[float] = []
    oracle_rot: list[float] = []
    oracle_cost: list[float] = []
    order_top1_trans: list[float] = []
    order_top1_rot: list[float] = []
    order_top1_cost: list[float] = []
    oracle_rank_by_order: list[int] = []
    basin_hits_by_k = {int(k): [] for k in topk}
    oracle_in_topk = {int(k): [] for k in topk}
    valid_counts: list[int] = []
    metadata_fields = sorted(
        key
        for key in data.files
        if key.startswith("retrieval_") or key in {"candidate_permutation", "init_sources"}
    )

    for name in ordered_names:
        idx = index_by_name.get(name)
        gt_pose = gt_poses_by_name.get(name)
        if idx is None or gt_pose is None:
            missing_names.append(name)
            continue
        candidate_poses = candidates[idx]
        candidate_valid = valid[idx].astype(bool)
        valid_counts.append(int(candidate_valid.sum()))
        trans = np.full(candidate_poses.shape[0], np.inf, dtype=np.float64)
        rot = np.full(candidate_poses.shape[0], np.inf, dtype=np.float64)
        cost = np.full(candidate_poses.shape[0], np.inf, dtype=np.float64)
        for cand_idx, pose in enumerate(candidate_poses):
            if not candidate_valid[cand_idx]:
                continue
            rot_err, trans_err_mm = pose_error(pose, np.asarray(gt_pose, dtype=np.float32))
            trans_err = float(trans_err_mm) / 1000.0
            rot[cand_idx] = float(rot_err)
            trans[cand_idx] = trans_err
            cost[cand_idx] = trans_err + float(rot_cost_weight) * float(np.deg2rad(rot_err))
        if not np.isfinite(cost).any():
            missing_names.append(name)
            continue
        best_idx = int(np.nanargmin(cost))
        first_valid = int(np.flatnonzero(candidate_valid)[0])
        matched_names.append(name)
        oracle_rank_by_order.append(best_idx + 1)
        oracle_trans.append(float(trans[best_idx]))
        oracle_rot.append(float(rot[best_idx]))
        oracle_cost.append(float(cost[best_idx]))
        order_top1_trans.append(float(trans[first_valid]))
        order_top1_rot.append(float(rot[first_valid]))
        order_top1_cost.append(float(cost[first_valid]))
        basin = (trans <= float(basin_trans_m)) & (rot <= float(basin_rot_deg)) & candidate_valid
        for k in topk:
            upto = min(int(k), candidate_poses.shape[0])
            basin_hits_by_k[k].append(bool(basin[:upto].any()))
            oracle_in_topk[k].append(bool(best_idx < upto))

    if not matched_names:
        raise ValueError(f"No query names from {cache_path} matched the provided ground truth")

    return {
        "label": str(label),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(str(cache_path)),
        "num_samples": int(len(matched_names)),
        "num_cache_entries": int(len(names)),
        "num_missing": int(len(missing_names)),
        "missing_names": missing_names[:20],
        "num_candidates": int(candidates.shape[1]),
        "valid_fraction": float(np.mean(valid)) if valid.size else 0.0,
        "valid_candidates_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "rot_cost_weight": float(rot_cost_weight),
        "basin_trans_m": float(basin_trans_m),
        "basin_rot_deg": float(basin_rot_deg),
        "order_top1": {
            "trans_m_quantiles": _safe_quantiles(order_top1_trans),
            "rot_deg_quantiles": _safe_quantiles(order_top1_rot),
            "cost_m_mean": float(np.mean(order_top1_cost)),
        },
        "oracle": {
            "trans_m_quantiles": _safe_quantiles(oracle_trans),
            "rot_deg_quantiles": _safe_quantiles(oracle_rot),
            "cost_m_mean": float(np.mean(oracle_cost)),
            "cost_m_quantiles": _safe_quantiles(oracle_cost),
            "rank_by_cache_order_quantiles": _safe_quantiles(oracle_rank_by_order),
        },
        "basin_recall_by_order": {
            f"@{k}": float(np.mean(basin_hits_by_k[k])) for k in topk
        },
        "oracle_in_topk_by_order": {
            f"@{k}": float(np.mean(oracle_in_topk[k])) for k in topk
        },
        "metadata_fields": metadata_fields,
    }


def pose_candidate_cache_diagnostics_to_markdown(report: Mapping) -> str:
    """Render candidate-cache diagnostics as a compact markdown table."""

    lines = [
        "# Pose Candidate Cache Diagnostics",
        "",
        f"- protocol: `{report.get('protocol')}`",
        f"- query subset size: `{report.get('query_subset_size')}`",
        "",
        "| cache | n | K | valid | top1 trans q50 mm | oracle trans q50 mm | oracle cost mean m | basin@1 | basin@5 | basin@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows", []):
        top1 = row.get("order_top1", {})
        oracle = row.get("oracle", {})
        basin = row.get("basin_recall_by_order", {})
        top1_trans = top1.get("trans_m_quantiles", {}).get("q50", float("nan"))
        oracle_trans = oracle.get("trans_m_quantiles", {}).get("q50", float("nan"))
        lines.append(
            "| {label} | {n:d} | {k:d} | {valid:.3f} | {top1_mm:.1f} | {oracle_mm:.1f} | "
            "{oracle_cost:.4f} | {b1:.3f} | {b5:.3f} | {b10:.3f} |".format(
                label=str(row.get("label", "")),
                n=int(row.get("num_samples", 0)),
                k=int(row.get("num_candidates", 0)),
                valid=float(row.get("valid_fraction", float("nan"))),
                top1_mm=float(top1_trans) * 1000.0,
                oracle_mm=float(oracle_trans) * 1000.0,
                oracle_cost=float(oracle.get("cost_m_mean", float("nan"))),
                b1=float(basin.get("@1", float("nan"))),
                b5=float(basin.get("@5", float("nan"))),
                b10=float(basin.get("@10", float("nan"))),
            )
        )
    return "\n".join(lines) + "\n"


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
