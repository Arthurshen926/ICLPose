"""Select frozen Top5/Top10 plane-PnP poses by MoGe3 normal agreement.

The two candidate poses and both render-consistency reports must already exist.
This phase never opens query pose labels.  It uses no source image or source-view
identity at runtime: the score is the fraction of valid query pixels whose MoGe3
normal agrees with the rendered physical-map normal within 20 degrees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_direct_plane_pnp_by_inlier_ratio import _merge
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


REPORT_TYPE = "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1"


def _load_reports(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("artifact_type") != REPORT_TYPE
            or report.get("query_pose_or_ground_truth_read") is not False
            or report.get("query_depth_changes_frozen_pose", False) is not False
            or report.get("query_depth_or_scale_used_by_pose_solver") is not False
        ):
            raise ValueError("render-consistency report is not a frozen label-free diagnostic")
        local_rows = report.get("rows")
        if not isinstance(local_rows, list) or len(local_rows) != int(report.get("query_count", -1)):
            raise ValueError("render-consistency row inventory differs")
        rows.extend(local_rows)
        reports.append(report)
    return rows, reports


def _shared_lineage(reports: list[dict[str, object]]) -> dict[str, object]:
    keys = (
        "physical_map_content_sha256",
        "physical_map_file_sha256",
        "query_camera_inventory_content_sha256",
        "query_camera_inventory_file_sha256",
        "moge3_manifest_content_sha256_in_order",
        "moge3_manifest_file_sha256_in_order",
    )
    result: dict[str, object] = {}
    for key in keys:
        values = [json.dumps(report.get(key), sort_keys=True) for report in reports]
        if len(set(values)) != 1:
            raise ValueError(f"Top5/Top10 render lineage differs: {key}")
        result[key] = reports[0].get(key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--top5_render_report", type=Path, nargs="+", required=True)
    parser.add_argument("--top10_render_report", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = {
        len(args.top5_pose_inventory), len(args.top10_pose_inventory),
        len(args.top5_render_report), len(args.top10_render_report),
    }
    if len(counts) != 1:
        raise ValueError("Top5/Top10 pose/report shard counts differ")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite selected PnP inventory")

    top5, meta5 = _merge(args.top5_pose_inventory)
    top10, meta10 = _merge(args.top10_pose_inventory)
    rows5, reports5 = _load_reports(args.top5_render_report)
    rows10, reports10 = _load_reports(args.top10_render_report)
    names = top5["names"].astype(str)
    if not np.array_equal(names, top10["names"].astype(str)) or len(set(names.tolist())) != len(names):
        raise ValueError("Top5/Top10 pose names differ")
    if [str(row.get("name")) for row in rows5] != names.tolist():
        raise ValueError("Top5 report order differs from pose inventory")
    if [str(row.get("name")) for row in rows10] != names.tolist():
        raise ValueError("Top10 report order differs from pose inventory")
    for paths, reports in ((args.top5_pose_inventory, reports5), (args.top10_pose_inventory, reports10)):
        for path, report in zip(paths, reports):
            if file_sha256(path) != report.get("frozen_pose_inventory_file_sha256"):
                raise ValueError("render report does not bind the supplied frozen pose file")
    shared = _shared_lineage(reports5 + reports10)
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256")) for meta in meta5 + meta10
    }
    if not camera_hashes or "None" in camera_hashes:
        raise ValueError("pose camera lineage is missing")

    usable5 = np.asarray(top5["usable"], bool) & np.asarray(
        [bool(row.get("usable")) for row in rows5], bool,
    )
    usable10 = np.asarray(top10["usable"], bool) & np.asarray(
        [bool(row.get("usable")) for row in rows10], bool,
    )
    score5 = np.asarray([float(row.get("normal_within_20deg", -1.0)) for row in rows5])
    score10 = np.asarray([float(row.get("normal_within_20deg", -1.0)) for row in rows10])
    score5 = np.where(usable5 & np.isfinite(score5), score5, -1.0)
    score10 = np.where(usable10 & np.isfinite(score10), score10, -1.0)
    choose10 = score10 > score5  # deterministic tie goes to the smaller Top5 branch.
    arrays = {
        "names": names,
        "pose_w2c": np.where(choose10[:, None, None], top10["pose_w2c"], top5["pose_w2c"]),
        "usable": np.where(choose10, usable10, usable5),
        "selected_branch": np.where(choose10, 10, 5).astype(np.int8),
        "selected_inlier_ratio": np.where(choose10, score10, score5).astype(np.float64),
        "selected_candidate_correspondence_count": np.where(
            choose10, top10["candidate_correspondence_count"], top5["candidate_correspondence_count"],
        ).astype(np.int64),
        "selected_pnp_inlier_count": np.where(
            choose10, top10["pnp_inlier_count"], top5["pnp_inlier_count"],
        ).astype(np.int64),
        "top5_normal_within_20deg": score5.astype(np.float64),
        "top10_normal_within_20deg": score10.astype(np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_top5_top10_moge3_normal_selected_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": "maximum_moge3_rendered_map_normal_within_20deg_tie_top5",
        "selection_rule_frozen_on": "seq10_development_route_then_replayed_unchanged_on_seq13",
        "selected_confidence_semantics": "moge3_rendered_physical_map_normal_within_20deg",
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_used_for_frozen_candidate_selection": True,
        "source_rgb_stored_in_runtime_map": False,
        "source_view_identity_used_for_selection": False,
        "query_camera_only_inventory_file_sha256_sorted": sorted(camera_hashes),
        "top5_pose_file_sha256_in_order": [file_sha256(path) for path in args.top5_pose_inventory],
        "top10_pose_file_sha256_in_order": [file_sha256(path) for path in args.top10_pose_inventory],
        "top5_render_file_sha256_in_order": [file_sha256(path) for path in args.top5_render_report],
        "top10_render_file_sha256_in_order": [file_sha256(path) for path in args.top10_render_report],
        "selected_top10_count": int(np.sum(choose10)),
        "shared_render_lineage": shared,
        "strict_runtime_phase_separation_eligible": True,
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
