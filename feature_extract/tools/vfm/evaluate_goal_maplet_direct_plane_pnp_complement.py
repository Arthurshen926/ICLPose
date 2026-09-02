"""Evaluate a frozen PnP confidence rule on queries outside an earlier pilot.

This is deliberately a post-label aggregation step.  It never changes a pose,
correspondence, ranking, or confidence threshold; it only partitions already
frozen PnP rows by the exact pilot names and reports selective localization
metrics on the complement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _selected_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    names = [str(name) for name in payload.get("selected_names_in_order", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("pilot manifest has empty or duplicated selected names")
    return names


def _summarize(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    usable = [row for row in rows if bool(row["usable"])]
    raw_good_2m45 = [
        row for row in usable
        if float(row["translation_error_m"]) <= 2.0 and float(row["rotation_error_deg"]) <= 45.0
    ]
    raw_good_1m10 = [
        row for row in usable
        if float(row["translation_error_m"]) <= 1.0 and float(row["rotation_error_deg"]) <= 10.0
    ]
    accepted = [
        row for row in usable
        if int(row["pnp_inlier_count"]) / max(int(row["candidate_correspondence_count"]), 1) >= threshold
    ]
    accepted_good_2m45 = [
        row for row in accepted
        if float(row["translation_error_m"]) <= 2.0 and float(row["rotation_error_deg"]) <= 45.0
    ]
    accepted_good_1m10 = [
        row for row in accepted
        if float(row["translation_error_m"]) <= 1.0 and float(row["rotation_error_deg"]) <= 10.0
    ]
    count = len(rows)
    translation = np.asarray([float(row["translation_error_m"]) for row in usable], np.float64)
    rotation = np.asarray([float(row["rotation_error_deg"]) for row in usable], np.float64)
    return {
        "query_count": count,
        "usable_count": len(usable),
        "raw_recall_2m45": len(raw_good_2m45) / count if count else 0.0,
        "raw_recall_1m10": len(raw_good_1m10) / count if count else 0.0,
        "median_translation_m_among_usable": float(np.median(translation)) if len(translation) else None,
        "median_rotation_deg_among_usable": float(np.median(rotation)) if len(rotation) else None,
        "confidence_threshold": threshold,
        "accepted_count": len(accepted),
        "accepted_fraction": len(accepted) / count if count else 0.0,
        "accepted_precision_2m45": len(accepted_good_2m45) / len(accepted) if accepted else 0.0,
        "accepted_precision_1m10": len(accepted_good_1m10) / len(accepted) if accepted else 0.0,
        "selective_system_recall_2m45": len(accepted_good_2m45) / count if count else 0.0,
        "selective_system_recall_1m10": len(accepted_good_1m10) / count if count else 0.0,
        "rejected_good_2m45_count": len(raw_good_2m45) - len(accepted_good_2m45),
        "rejected_bad_2m45_count": (len(usable) - len(raw_good_2m45)) - (len(accepted) - len(accepted_good_2m45)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_pnp_report", type=Path, action="append", required=True)
    parser.add_argument("--pilot_manifest", type=Path, action="append", required=True)
    parser.add_argument("--confidence_config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite complement evaluation")
    confidence = json.loads(args.confidence_config.read_text())
    if (
        confidence.get("artifact_type") != "goal_maplet_direct_plane_pnp_confidence_config_v1"
        or confidence.get("eligible_use") != "future unseen official-test complement or new scene only"
    ):
        raise ValueError("confidence config is not eligible for a future complement")
    threshold = float(confidence["threshold"])

    rows_by_route: dict[str, list[dict[str, object]]] = {}
    pilots_by_route: dict[str, set[str]] = {}
    route_summaries: dict[str, object] = {}
    excluded: list[str] = []
    report_hashes: list[str] = []
    manifest_hashes: list[str] = []
    for pilot_path in args.pilot_manifest:
        names = _selected_names(pilot_path)
        route = names[0].split("__", 1)[0]
        if any(name.split("__", 1)[0] != route for name in names) or route in pilots_by_route:
            raise ValueError("pilot manifests must each contain one unique route")
        pilots_by_route[route] = set(names)
        manifest_hashes.append(file_sha256(pilot_path))
    for report_path in args.full_pnp_report:
        report = json.loads(report_path.read_text())
        if (
            report.get("artifact_type") != "goal_maplet_radio_plane_2d3d_pnp_control_v2"
            or report.get("strict_runtime_phase_separation_eligible") is not True
            or report.get("official_test_map_disjoint_declared") is not True
            or report.get("plane_retrieval") != "direct query-region RADIO to finite-plane observation descriptors"
        ):
            raise ValueError("full PnP report is not a strict direct-plane official-test control")
        rows = list(report["rows"])
        names = [str(row["name"]) for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("full PnP report contains duplicate query names")
        route = names[0].split("__", 1)[0] if names else ""
        if not route or any(name.split("__", 1)[0] != route for name in names):
            raise ValueError("each full or sharded report must contain one route")
        rows_by_route.setdefault(route, []).extend(rows)
        report_hashes.append(file_sha256(report_path))

    if set(rows_by_route) != set(pilots_by_route):
        raise ValueError("full-report routes and pilot-manifest routes differ")
    all_rows: list[dict[str, object]] = []
    for route in sorted(rows_by_route):
        rows = rows_by_route[route]
        names = [str(row["name"]) for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("sharded full PnP reports overlap")
        pilot = pilots_by_route[route]
        if not pilot < set(names):
            raise ValueError("pilot names must be a strict subset of the full route")
        complement = [row for row in rows if str(row["name"]) not in pilot]
        route_summaries[route] = _summarize(complement, threshold)
        all_rows.extend(complement)
        excluded.extend(sorted(pilot))

    names = [str(row["name"]) for row in all_rows]
    if len(names) != len(set(names)) or set(names) & set(excluded):
        raise ValueError("complement partition is not globally disjoint")
    payload = {
        "artifact_type": "goal_maplet_direct_plane_pnp_unseen_complement_evaluation_v1",
        "evaluation_role": "post-pilot complement validation; not a pristine method-selection test",
        "poses_rankings_and_confidence_frozen_before_this_postlabel_aggregation": True,
        "pilot_queries_excluded": True,
        "pilot_query_count": len(excluded),
        "complement_query_count": len(all_rows),
        "confidence_config_file_sha256": file_sha256(args.confidence_config),
        "confidence_threshold": threshold,
        "full_pnp_report_file_sha256_in_order": report_hashes,
        "pilot_manifest_file_sha256_in_order": manifest_hashes,
        "route_summaries": route_summaries,
        "aggregate": _summarize(all_rows, threshold),
        "production_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
