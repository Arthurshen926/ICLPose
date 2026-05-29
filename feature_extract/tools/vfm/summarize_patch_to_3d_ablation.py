"""Summarize patch-to-3D VFM matching ablation summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


CORE_FIELDS = (
    "label",
    "landmark_bank_name",
    "query_count",
    "submap_mode",
    "submap_top_n",
    "match_mode",
    "top_k",
    "mutual_top_k",
    "ratio_threshold",
    "min_similarity_margin",
    "min_quality_weighted_similarity",
    "landmark_quality_enabled",
    "mean_match_count",
    "mean_patch_at_1",
    "mean_patch_at_5",
    "mean_gt_precision_stride",
    "mean_pnp_inlier_patch_at_1",
    "mean_pnp_inlier_patch_at_5",
    "mean_pnp_inlier_gt_precision_stride",
    "mean_pnp_inlier_count",
    "mean_pnp_inlier_convex_hull_area_frac",
    "visible_landmark_recall_mean",
    "visible_landmark_recall_median",
    "visible_landmark_recall_p25",
    "visible_landmark_recall_p75",
    "mean_gt_visible_bank_tracks",
    "mean_submap_gt_visible_tracks",
    "mean_positives_per_token",
    "mean_zero_positive_token_ratio",
    "reference_top1_median_translation_error_m",
    "reference_top1_median_rotation_error_deg",
    "reference_top1_success_25cm_10deg",
    "reference_oracle_median_translation_error_m",
    "reference_oracle_median_rotation_error_deg",
    "reference_oracle_success_25cm_10deg",
    "pnp_solve_rate",
    "success_25cm_10deg",
    "success_50cm_10deg",
    "success_1m_10deg",
    "median_translation_error_m",
    "median_rotation_error_deg",
    "summary_path",
)


def _label_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_summary.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _lower_is_better(metric: str) -> bool:
    return metric.startswith("median_") or metric.endswith("_error_m") or metric.endswith("_error_deg")


def _row_from_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text())
    config = dict(summary.get("matching_config") or {})
    quality = dict(config.get("landmark_quality") or {})
    submap = dict(summary.get("submap") or {})
    positives = dict(summary.get("positive_set_summary") or {})
    visible_recall = dict(summary.get("visible_landmark_recall") or {})
    reference_prior = dict(summary.get("reference_prior") or {})
    reference_top1 = dict(reference_prior.get("top1") or {})
    reference_oracle = dict(reference_prior.get("oracle") or {})
    inputs = dict(summary.get("inputs") or {})
    landmark_bank = Path(str(inputs.get("landmark_bank", ""))).name
    patch_at_5 = summary.get("mean_patch_at_5")
    inlier_patch_at_5 = summary.get("mean_pnp_inlier_patch_at_5")
    return {
        "label": _label_from_path(path),
        "landmark_bank_name": landmark_bank,
        "query_count": summary.get("query_count"),
        "submap_mode": submap.get("mode"),
        "submap_top_n": submap.get("top_n"),
        "match_mode": config.get("match_mode"),
        "top_k": config.get("top_k"),
        "mutual_top_k": config.get("mutual_top_k"),
        "ratio_threshold": config.get("ratio_threshold"),
        "min_similarity_margin": config.get("min_similarity_margin"),
        "min_quality_weighted_similarity": config.get("min_quality_weighted_similarity"),
        "landmark_quality_enabled": quality.get("enabled", False),
        "mean_match_count": summary.get("mean_match_count"),
        "mean_patch_at_1": summary.get("mean_patch_at_1"),
        "mean_patch_at_5": patch_at_5,
        "mean_gt_precision_stride": summary.get("mean_gt_precision_stride"),
        "mean_pnp_inlier_patch_at_1": summary.get("mean_pnp_inlier_patch_at_1"),
        "mean_pnp_inlier_patch_at_5": inlier_patch_at_5,
        "mean_pnp_inlier_gt_precision_stride": summary.get("mean_pnp_inlier_gt_precision_stride"),
        "mean_pnp_inlier_count": summary.get("mean_pnp_inlier_count"),
        "mean_pnp_inlier_convex_hull_area_frac": summary.get("mean_pnp_inlier_convex_hull_area_frac"),
        "visible_landmark_recall_mean": visible_recall.get("mean"),
        "visible_landmark_recall_median": visible_recall.get("median"),
        "visible_landmark_recall_p25": visible_recall.get("p25"),
        "visible_landmark_recall_p75": visible_recall.get("p75"),
        "mean_gt_visible_bank_tracks": visible_recall.get("mean_gt_visible_bank_tracks"),
        "mean_submap_gt_visible_tracks": visible_recall.get("mean_submap_gt_visible_tracks"),
        "mean_positives_per_token": positives.get("mean_positives_per_token"),
        "mean_zero_positive_token_ratio": positives.get("mean_zero_positive_token_ratio"),
        "reference_top1_median_translation_error_m": reference_top1.get("median_translation_error_m"),
        "reference_top1_median_rotation_error_deg": reference_top1.get("median_rotation_error_deg"),
        "reference_top1_success_25cm_10deg": reference_top1.get("success_25cm_10deg"),
        "reference_oracle_median_translation_error_m": reference_oracle.get("median_translation_error_m"),
        "reference_oracle_median_rotation_error_deg": reference_oracle.get("median_rotation_error_deg"),
        "reference_oracle_success_25cm_10deg": reference_oracle.get("success_25cm_10deg"),
        "pnp_solve_rate": summary.get("pnp_solve_rate"),
        "success_25cm_10deg": summary.get("success_25cm_10deg"),
        "success_50cm_10deg": summary.get("success_50cm_10deg"),
        "success_1m_10deg": summary.get("success_1m_10deg"),
        "median_translation_error_m": summary.get("median_translation_error_m"),
        "median_rotation_error_deg": summary.get("median_rotation_error_deg"),
        "summary_path": str(path),
    }


def _sort_value(row: dict[str, object], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return float("inf")
    return float(value)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CORE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CORE_FIELDS})


def _write_markdown(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "submap_mode",
        "submap_top_n",
        "match_mode",
        "mean_patch_at_1",
        "mean_patch_at_5",
        "mean_pnp_inlier_patch_at_1",
        "visible_landmark_recall_median",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "reference_top1_median_translation_error_m",
        "reference_oracle_median_translation_error_m",
        "median_translation_error_m",
        "median_rotation_error_deg",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize patch-to-3D ablation summaries")
    parser.add_argument("--summary_json", nargs="+", required=True)
    parser.add_argument("--sort_metric", default="median_translation_error_m")
    parser.add_argument("--higher_is_better", action="store_true")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows = [_row_from_summary(Path(path)) for path in args.summary_json]
    reverse = bool(args.higher_is_better or not _lower_is_better(args.sort_metric))
    rows = sorted(rows, key=lambda row: _sort_value(row, args.sort_metric), reverse=reverse)
    report = {
        "stage": "patch_to_3d_vfm_ablation_summary",
        "sort_metric": args.sort_metric,
        "higher_is_better": reverse,
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        _write_csv(rows, Path(args.output_csv))
    if args.output_md:
        _write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
