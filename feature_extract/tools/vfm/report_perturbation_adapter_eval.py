"""Aggregate perturbation-aware render/match/pose evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


_METRIC_KEYS = (
    "query_count",
    "ok_count",
    "solve_rate",
    "median_translation_error_m",
    "median_rotation_error_deg",
    "success_10cm_5deg",
    "success_25cm_10deg",
    "success_50cm_10deg",
    "match_gt_5px",
    "match_gt_16px",
    "match_gt_32px",
    "pnp_inlier_gt_16px",
    "pnp_inlier_gt_32px",
    "median_match_reprojection_error_px",
    "median_pnp_inlier_reprojection_error_px",
    "mean_match_count",
    "mean_pnp_inlier_count",
)


_METRIC_ALIASES = {
    "ok_count": ("ok_count", "query_count"),
    "solve_rate": ("solve_rate", "pnp_solve_rate"),
    "match_gt_5px": ("match_gt_5px", "mean_gt_precision_5px"),
    "match_gt_16px": ("match_gt_16px", "mean_gt_precision_16px"),
    "match_gt_32px": ("match_gt_32px", "mean_gt_precision_32px"),
    "pnp_inlier_gt_16px": ("pnp_inlier_gt_16px", "mean_pnp_inlier_gt_precision_16px"),
    "pnp_inlier_gt_32px": ("pnp_inlier_gt_32px", "mean_pnp_inlier_gt_precision_32px"),
    "median_match_reprojection_error_px": (
        "median_match_reprojection_error_px",
        "median_gt_reprojection_error_px",
    ),
    "median_pnp_inlier_reprojection_error_px": (
        "median_pnp_inlier_reprojection_error_px",
        "median_pnp_inlier_gt_reprojection_error_px",
    ),
    "mean_pnp_inlier_count": ("mean_pnp_inlier_count", "mean_pnp_inliers"),
}


def _load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def _row_from_summary(path: Path) -> dict[str, object]:
    summary = _load_summary(path)
    inputs = dict(summary.get("inputs", {}) or {})
    config = dict(summary.get("config", {}) or {})
    metrics = dict(summary.get("metrics", {}) or {})
    row: dict[str, object] = {
        "summary": str(path),
        "name": str(path.parent.name),
        "stage": str(summary.get("stage", "")),
        "render_pose_mode": str(config.get("render_pose_mode", "")),
        "render_pose_world_offset": str(config.get("render_pose_world_offset", "")),
        "pose_update_iterations": config.get("pose_update_iterations", ""),
        "pose_update_selection": str(config.get("pose_update_selection", "")),
        "match_mode": str(config.get("match_mode", "")),
        "renderer": str(config.get("renderer", "")),
        "matcha_joint_checkpoint": str(inputs.get("matcha_joint_checkpoint", "")),
        "matcha_adapter_checkpoint": str(inputs.get("matcha_adapter_checkpoint", "")),
        "selector_checkpoint": str(inputs.get("selector_checkpoint", "")),
        "candidate_bank": str(inputs.get("candidate_bank", "")),
    }
    for key in _METRIC_KEYS:
        aliases = _METRIC_ALIASES.get(key, (key,))
        row[key] = next((metrics[name] for name in aliases if name in metrics), "")
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = [_row_from_summary(Path(path)) for path in args.summaries]
    _write_csv(Path(args.output_csv), rows)
    if str(args.output_json):
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"row_count": len(rows), "output_csv": str(args.output_csv)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
