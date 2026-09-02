"""Summarize frozen sparse-occlusion carrier and candidate-sharing controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


THRESHOLDS = ((0.10, 1.0), (0.25, 2.0), (0.50, 5.0), (1.0, 10.0), (2.0, 45.0))


def _reports(paths: list[Path]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    reports = [json.loads(path.read_text()) for path in paths]
    rows = {}
    for report in reports:
        if report.get("artifact_type") != "goal_maplet_radio_plane_2d3d_pnp_control_v2":
            raise ValueError("PnP report contract differs")
        for row in report.get("rows", []):
            if row["name"] in rows:
                raise ValueError("PnP report routes overlap")
            rows[row["name"]] = row
    return rows, reports


def _metrics(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    usable = [row for row in rows.values() if row.get("usable")]
    translation = np.asarray([row["translation_error_m"] for row in usable], np.float64)
    rotation = np.asarray([row["rotation_error_deg"] for row in usable], np.float64)
    return {
        "query_count": len(rows), "usable_count": len(usable),
        "median_translation_m": float(np.median(translation)) if len(translation) else None,
        "median_rotation_deg": float(np.median(rotation)) if len(rotation) else None,
        "threshold_hits": {
            f"{distance:g}m_{angle:g}deg": int(np.sum((translation <= distance) & (rotation <= angle)))
            for distance, angle in THRESHOLDS
        },
        "median_candidate_correspondence_count": float(np.median([
            row["candidate_correspondence_count"] for row in rows.values()
        ])),
        "median_pnp_inlier_count": float(np.median([
            row["pnp_inlier_count"] for row in rows.values()
        ])),
        "median_inlier_query_hull_fraction": float(np.median([
            row["inlier_query_hull_fraction"] for row in rows.values()
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True)
    parser.add_argument("--a1", type=Path, nargs=2, required=True)
    parser.add_argument("--a2", type=Path, nargs=2, required=True)
    parser.add_argument("--a4", type=Path, nargs=2)
    parser.add_argument("--budget6_baseline", type=Path, nargs=2)
    parser.add_argument("--a5", type=Path, nargs=2)
    parser.add_argument("--a6", type=Path, nargs=2)
    parser.add_argument("--a7_evaluation", type=Path)
    parser.add_argument("--base_query_plane_dir", type=Path, nargs=2, required=True)
    parser.add_argument("--carrier_query_plane_dir", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-occlusion comparison")
    baseline, baseline_reports = _reports(args.baseline)
    a1, a1_reports = _reports(args.a1)
    a2, a2_reports = _reports(args.a2)
    a4 = a4_reports = None
    if args.a4 is not None:
        a4, a4_reports = _reports(args.a4)
    if (args.budget6_baseline is None) != (args.a5 is None):
        raise ValueError("budget6 baseline and A5 must be supplied together")
    budget6 = budget6_reports = a5 = a5_reports = None
    if args.budget6_baseline is not None:
        budget6, budget6_reports = _reports(args.budget6_baseline)
        a5, a5_reports = _reports(args.a5)
    a6 = a6_reports = None
    if args.a6 is not None:
        a6, a6_reports = _reports(args.a6)
    inventories = [set(baseline), set(a1), set(a2)]
    if a4 is not None:
        inventories.append(set(a4))
    if budget6 is not None:
        inventories.extend((set(budget6), set(a5)))
    if a6 is not None:
        inventories.append(set(a6))
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError("A0/A1/A2/A4 query inventories differ")
    for report in a1_reports + a2_reports:
        if (
            report.get("sparse_occlusion_carrier") is not True
            or report.get("strict_runtime_phase_separation_eligible") is not True
            or report.get("query_moge3_role") != "plane_segmentation_and_sparse_foreground_carrier_only"
        ):
            raise ValueError("carrier PnP phase/geometry contract differs")
    for report in a4_reports or []:
        if (
            report.get("sparse_occlusion_carrier") is not False
            or report.get("sparse_occlusion_candidate_sharing") is not True
            or report.get("strict_runtime_phase_separation_eligible") is not True
            or report.get("query_moge3_role")
            != "plane_segmentation_and_sparse_foreground_candidate_sharing_only"
        ):
            raise ValueError("candidate-sharing PnP phase/geometry contract differs")
    for report in budget6_reports or []:
        if (
            report.get("topk_planes") != 6
            or report.get("sparse_occlusion_candidate_sharing") is not False
            or report.get("strict_runtime_phase_separation_eligible") is not True
        ):
            raise ValueError("budget6 baseline PnP contract differs")
    for report in a5_reports or []:
        if (
            report.get("topk_planes") != 6
            or report.get("sparse_occlusion_candidate_sharing") is not True
            or report.get("strict_runtime_phase_separation_eligible") is not True
        ):
            raise ValueError("A5 supplement PnP contract differs")
    for report in a6_reports or []:
        if (
            report.get("topk_planes") != 6
            or report.get("sparse_occlusion_candidate_sharing") is not True
            or report.get("strict_runtime_phase_separation_eligible") is not True
        ):
            raise ValueError("A6 multi-island PnP contract differs")
    carrier_rows: dict[str, dict[str, object]] = {}
    bridge_foreground = []
    bridge_corridor = []
    support_exact = True
    for base_dir, carrier_dir in zip(args.base_query_plane_dir, args.carrier_query_plane_dir):
        base_paths = sorted(base_dir.glob("*.npz")); carrier_paths = sorted(carrier_dir.glob("*.npz"))
        if [path.name for path in base_paths] != [path.name for path in carrier_paths]:
            raise ValueError("base/carrier query-plane inventories differ")
        for base_path, carrier_path in zip(base_paths, carrier_paths):
            base_planes, _ = QueryPlaneRegions.load_npz(base_path)
            carrier_planes, metadata = QueryPlaneRegions.load_npz(carrier_path)
            diagnostic = metadata.get("carrier_diagnostics", {})
            exact = bool(np.array_equal(base_planes.labels >= 0, carrier_planes.labels >= 0))
            support_exact &= exact and diagnostic.get("hidden_pixel_count_added") == 0
            for edge in diagnostic.get("accepted_edges", []):
                for path in edge.get("bridge_paths", []):
                    if path.get("accepted"):
                        bridge_foreground.append(float(path["foreground_fraction"]))
                        bridge_corridor.append(int(path["corridor_pixels"]))
            carrier_rows[base_path.name] = {
                "base_plane_count": int(len(base_planes.normals_camera)),
                "carrier_plane_count": int(len(carrier_planes.normals_camera)),
                "merged_component_count": int(diagnostic.get("merged_component_count", 0)),
                "maximum_carrier_component_count": int(diagnostic.get("maximum_carrier_component_count", 0)),
                "observed_support_bit_exact": exact,
            }
    if set(carrier_rows) != set(baseline):
        raise ValueError("carrier audit and PnP inventories differ")
    per_query = []
    for name in sorted(baseline):
        row = {"name": name, **carrier_rows[name]}
        sources = [("a0", baseline), ("a1", a1), ("a2", a2)]
        if a4 is not None:
            sources.append(("a4", a4))
        if budget6 is not None:
            sources.extend((("budget6", budget6), ("a5", a5)))
        if a6 is not None:
            sources.append(("a6", a6))
        for label, source in sources:
            item = source[name]
            row[label] = {
                "usable": bool(item.get("usable")),
                "translation_error_m": item.get("translation_error_m"),
                "rotation_error_deg": item.get("rotation_error_deg"),
                "candidate_correspondence_count": int(item["candidate_correspondence_count"]),
                "pnp_inlier_count": int(item["pnp_inlier_count"]),
                "inlier_query_hull_fraction": float(item["inlier_query_hull_fraction"]),
            }
        per_query.append(row)
    affected = [row for row in per_query if row["merged_component_count"] > 0]
    def delta_summary(left: str, right: str) -> dict[str, object]:
        comparable = [row for row in per_query if row[left]["usable"] and row[right]["usable"]]
        delta = np.asarray([
            row[right]["translation_error_m"] - row[left]["translation_error_m"]
            for row in comparable
        ], np.float64)
        return {
            "comparable_count": len(comparable),
            "translation_improved_count": int(np.sum(delta < -1e-12)),
            "translation_worsened_count": int(np.sum(delta > 1e-12)),
            "translation_unchanged_count": int(np.sum(np.abs(delta) <= 1e-12)),
            "median_translation_delta_m": float(np.median(delta)) if len(delta) else None,
        }
    branch_names = ["a0", "a1", "a2"] + (["a4"] if a4 is not None else [])
    if a6 is not None:
        branch_names.append("a6")
    oracle_hits = {}
    for distance, angle in THRESHOLDS:
        oracle_hits[f"{distance:g}m_{angle:g}deg"] = int(sum(
            any(
                row[branch]["usable"]
                and row[branch]["translation_error_m"] <= distance
                and row[branch]["rotation_error_deg"] <= angle
                for branch in branch_names
            )
            for row in per_query
        ))
    payload = {
        "artifact_type": "goal_maplet_sparse_occlusion_carrier_pnp_comparison_v1",
        "status": "HISTORICAL_HELD_MECHANISM_CONTROL_NOT_BLIND_PROMOTION",
        "uses_pose_or_ground_truth": True,
        "carrier_frozen_before_pose_evaluation": True,
        "carrier_observed_support_only": bool(support_exact),
        "hidden_pixel_count_added": 0,
        "input_file_sha256": {
            "baseline": [file_sha256(path) for path in args.baseline],
            "a1": [file_sha256(path) for path in args.a1],
            "a2": [file_sha256(path) for path in args.a2],
            "a4": None if args.a4 is None else [file_sha256(path) for path in args.a4],
            "budget6_baseline": (
                None if args.budget6_baseline is None
                else [file_sha256(path) for path in args.budget6_baseline]
            ),
            "a5": None if args.a5 is None else [file_sha256(path) for path in args.a5],
            "a6": None if args.a6 is None else [file_sha256(path) for path in args.a6],
            "a7_evaluation": (
                None if args.a7_evaluation is None else file_sha256(args.a7_evaluation)
            ),
            "base_query_plane_manifests": [file_sha256(path / "manifest.json") for path in args.base_query_plane_dir],
            "carrier_query_plane_manifests": [file_sha256(path / "manifest.json") for path in args.carrier_query_plane_dir],
        },
        "label_free_carrier_audit": {
            "query_count": len(per_query),
            "affected_query_count": len(affected),
            "merged_component_count": int(sum(row["merged_component_count"] for row in per_query)),
            "maximum_merged_components_per_query": int(max((row["merged_component_count"] for row in per_query), default=0)),
            "maximum_carrier_component_count": int(max((row["maximum_carrier_component_count"] for row in per_query), default=0)),
            "accepted_bridge_path_count": len(bridge_foreground),
            "bridge_foreground_fraction_median": float(np.median(bridge_foreground)) if bridge_foreground else None,
            "bridge_foreground_fraction_minimum": float(np.min(bridge_foreground)) if bridge_foreground else None,
            "bridge_corridor_pixels_median": float(np.median(bridge_corridor)) if bridge_corridor else None,
            "bridge_corridor_pixels_maximum": int(np.max(bridge_corridor)) if bridge_corridor else None,
        },
        "metrics": {
            "a0": _metrics(baseline), "a1": _metrics(a1), "a2": _metrics(a2),
            **({"a4": _metrics(a4)} if a4 is not None else {}),
            **({"budget6": _metrics(budget6), "a5": _metrics(a5)} if budget6 is not None else {}),
            **({"a6": _metrics(a6)} if a6 is not None else {}),
        },
        "paired_deltas": {
            "a0_to_a1": delta_summary("a0", "a1"),
            "a1_to_a2": delta_summary("a1", "a2"),
            **({"a0_to_a4": delta_summary("a0", "a4")} if a4 is not None else {}),
            **({"budget6_to_a5": delta_summary("budget6", "a5")} if budget6 is not None else {}),
            **({"budget6_to_a6": delta_summary("budget6", "a6")} if a6 is not None else {}),
        },
        "a7_cross_island_selection_evaluation": (
            None if args.a7_evaluation is None
            else json.loads(args.a7_evaluation.read_text()).get("summary")
        ),
        "postlabel_branch_union_oracle_threshold_hits": oracle_hits,
        "per_query": per_query,
        "claim_boundary": "OldHospital has already been inspected; use only as mechanism evidence and freeze a branch before a new unseen query window.",
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
