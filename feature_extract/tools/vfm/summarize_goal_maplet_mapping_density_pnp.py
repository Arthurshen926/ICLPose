"""Seal map-density and pose-PnP comparisons from immutable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


THRESHOLDS = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 10.0), (2.0, 45.0))


def _parse_group(value: str) -> tuple[str, list[Path]]:
    label, separator, paths = value.partition("=")
    if not separator or not label or not paths:
        raise ValueError("group must be LABEL=PATH[,PATH]")
    return label, [Path(path) for path in paths.split(",")]


def _summarize_reports(paths: list[Path]) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    rows, route_metrics, report_hashes = [], {}, []
    phase_pass = True
    for path in paths:
        report = json.loads(path.read_text())
        local = list(report["rows"])
        rows.extend(local)
        report_hashes.append(file_sha256(path))
        phase_pass &= bool(report.get("strict_runtime_phase_separation_eligible"))
        translation = np.asarray([row.get("translation_error_m", np.inf) for row in local], np.float64)
        rotation = np.asarray([row.get("rotation_error_deg", np.inf) for row in local], np.float64)
        usable = np.isfinite(translation) & np.isfinite(rotation)
        route_metrics[path.name] = {
            "query_count": len(local), "usable_count": int(np.sum(usable)),
            "median_translation_m": float(np.median(translation[usable])),
            "median_rotation_deg": float(np.median(rotation[usable])),
            "threshold_hit_counts": {
                f"{t:g}m_{r:g}deg": int(np.sum((translation <= t) & (rotation <= r)))
                for t, r in THRESHOLDS
            },
        }
    by_name = {str(row["name"]): row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("PnP report query names are duplicated")
    translation = np.asarray([row.get("translation_error_m", np.inf) for row in rows], np.float64)
    rotation = np.asarray([row.get("rotation_error_deg", np.inf) for row in rows], np.float64)
    usable = np.isfinite(translation) & np.isfinite(rotation)
    summary = {
        "report_file_sha256_in_order": report_hashes,
        "query_count": len(rows), "usable_count": int(np.sum(usable)),
        "median_translation_m": float(np.median(translation[usable])),
        "median_rotation_deg": float(np.median(rotation[usable])),
        "threshold_hit_counts": {
            f"{t:g}m_{r:g}deg": int(np.sum((translation <= t) & (rotation <= r)))
            for t, r in THRESHOLDS
        },
        "threshold_recall_over_all_queries": {
            f"{t:g}m_{r:g}deg": float(np.mean((translation <= t) & (rotation <= r)))
            for t, r in THRESHOLDS
        },
        "strict_runtime_phase_separation_all_reports": phase_pass,
        "routes": route_metrics,
    }
    return summary, by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_report", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--pnp_group", action="append", required=True, help="LABEL=PATH[,PATH]")
    parser.add_argument("--paired", action="append", default=[], help="OLD:NEW")
    parser.add_argument("--uses_query_labels_for_configuration_selection", action="store_true")
    parser.add_argument(
        "--evaluation_role", default="historical_validation_not_pristine_blind_test",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite density comparison")
    maps = {}
    for value in args.map_report:
        label, paths = _parse_group(value)
        if len(paths) != 1:
            raise ValueError("map report accepts one path")
        report = json.loads(paths[0].read_text())
        maps[label] = {**report, "report_file_sha256": file_sha256(paths[0])}
    groups, rows = {}, {}
    for value in args.pnp_group:
        label, paths = _parse_group(value)
        groups[label], rows[label] = _summarize_reports(paths)
    paired = {}
    for value in args.paired:
        old, separator, new = value.partition(":")
        if not separator or old not in rows or new not in rows:
            raise ValueError("paired comparison references an unknown group")
        names = sorted(set(rows[old]) & set(rows[new]))
        if len(names) != len(rows[old]) or len(names) != len(rows[new]):
            raise ValueError("paired PnP query inventories differ")
        threshold = {}
        for translation_limit, rotation_limit in THRESHOLDS:
            old_hit = np.asarray([
                rows[old][name].get("translation_error_m", np.inf) <= translation_limit
                and rows[old][name].get("rotation_error_deg", np.inf) <= rotation_limit
                for name in names
            ])
            new_hit = np.asarray([
                rows[new][name].get("translation_error_m", np.inf) <= translation_limit
                and rows[new][name].get("rotation_error_deg", np.inf) <= rotation_limit
                for name in names
            ])
            threshold[f"{translation_limit:g}m_{rotation_limit:g}deg"] = {
                "old_hits": int(np.sum(old_hit)), "new_hits": int(np.sum(new_hit)),
                "gained": int(np.sum(new_hit & ~old_hit)), "lost": int(np.sum(old_hit & ~new_hit)),
            }
        paired[f"{old}_to_{new}"] = {"query_count": len(names), "thresholds": threshold}
    payload = {
        "artifact_type": "goal_maplet_mapping_density_plane_pnp_comparison_v1",
        "maps": maps, "pnp_groups": groups, "paired": paired,
        "uses_query_labels_for_configuration_selection": bool(
            args.uses_query_labels_for_configuration_selection
        ),
        "evaluation_role": str(args.evaluation_role),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "content_sha256": payload["content_sha256"],
        "output_file_sha256": file_sha256(args.output),
        "pnp_groups": groups,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
