"""Select RADIO or masked-SIFT plane PnP by pose-nearby global RADIO context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_view_context import (
    _candidate_context_score,
    _load_global_view_field,
    _query_global_context,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _records
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _select_sift(
    radio_score: float,
    sift_score: float,
    radio_usable: bool,
    sift_usable: bool,
) -> bool:
    if not sift_usable:
        return False
    if not radio_usable:
        return True
    return float(sift_score) > float(radio_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_selected_inventory", type=Path, required=True)
    parser.add_argument("--sift_candidate_inventory", type=Path, required=True)
    parser.add_argument("--sift_report", type=Path, required=True)
    parser.add_argument("--global_view_field", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite global-context SIFT/RADIO selection")

    with np.load(args.radio_selected_inventory, allow_pickle=False) as data:
        radio_meta = json.loads(str(data["metadata_json"].item()))
        radio = {key: np.asarray(data[key]) for key in (
            "names", "pose_w2c", "usable", "selected_inlier_ratio",
            "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
        )}
    if (
        radio_meta.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_top5_top10_inlier_selected_v1"
        or radio_meta.get("query_pose_or_ground_truth_read") is not False
    ):
        raise ValueError("RADIO selected pose inventory differs")

    with np.load(args.sift_candidate_inventory, allow_pickle=False) as data:
        sift_meta = json.loads(str(data["metadata_json"].item()))
        sift = {key: np.asarray(data[key]) for key in (
            "names", "candidate_offsets", "candidate_pose_w2c", "candidate_inlier_count",
            "candidate_origin",
        )}
    if (
        sift_meta.get("artifact_type")
        != "goal_maplet_masked_sift_plane_pnp_candidate_inventory_v1"
        or sift_meta.get("pose_or_ground_truth_opened") is not False
        or arrays_sha256(sift) != sift_meta.get("arrays_sha256")
        or not np.array_equal(radio["names"], sift["names"])
    ):
        raise ValueError("SIFT candidate inventory differs")
    report = json.loads(args.sift_report.read_text())
    if (
        report.get("candidate_inventory_file_sha256")
        != file_sha256(args.sift_candidate_inventory)
        or [row["name"] for row in report["pose_free_rows"]]
        != sift["names"].astype(str).tolist()
    ):
        raise ValueError("SIFT report does not bind candidate inventory")

    field, field_meta = _load_global_view_field(args.global_view_field)
    records = _records(args.radio_manifest)
    centers = np.asarray(field["camera_centers_world"], np.float64)
    forwards = np.asarray(field["camera_forwards_world"], np.float64)
    descriptors = np.asarray(field["global_radio_descriptors"], np.float32)

    poses, usable, branch = [], [], []
    ratio, correspondence_count, inlier_count = [], [], []
    baseline_ratio, sift_inliers, sift_candidates = [], [], []
    radio_scores, sift_scores, radio_views, sift_views = [], [], [], []
    for index, name in enumerate(sift["names"].astype(str).tolist()):
        lo, hi = map(int, sift["candidate_offsets"][index:index + 2])
        sift_usable = lo < hi
        radio_usable = bool(radio["usable"][index])
        query = _query_global_context(name, records)
        if radio_usable:
            radio_score, radio_count = _candidate_context_score(
                radio["pose_w2c"][index], query, centers, forwards, descriptors,
            )
        else:
            radio_score, radio_count = -1.0, 0
        if sift_usable:
            sift_score, sift_count = _candidate_context_score(
                sift["candidate_pose_w2c"][lo], query, centers, forwards, descriptors,
            )
        else:
            sift_score, sift_count = -1.0, 0
        use_sift = _select_sift(radio_score, sift_score, radio_usable, sift_usable)
        diagnostic = report["pose_free_rows"][index]
        best_sift_inliers = 0 if not sift_usable else int(sift["candidate_inlier_count"][lo])
        if use_sift:
            poses.append(sift["candidate_pose_w2c"][lo]); usable.append(True); branch.append(40)
            correspondence_count.append(int(diagnostic["correspondence_count"]))
            inlier_count.append(best_sift_inliers)
            ratio.append(
                best_sift_inliers / max(int(diagnostic["unique_query_keypoint_count"]), 1)
            )
        else:
            poses.append(radio["pose_w2c"][index]); usable.append(radio_usable); branch.append(5)
            correspondence_count.append(int(radio["selected_candidate_correspondence_count"][index]))
            inlier_count.append(int(radio["selected_pnp_inlier_count"][index]))
            ratio.append(float(radio["selected_inlier_ratio"][index]))
        baseline_ratio.append(float(radio["selected_inlier_ratio"][index]))
        sift_inliers.append(best_sift_inliers); sift_candidates.append(hi - lo)
        radio_scores.append(radio_score); sift_scores.append(sift_score)
        radio_views.append(radio_count); sift_views.append(sift_count)

    arrays = {
        "names": sift["names"], "pose_w2c": np.asarray(poses, np.float64),
        "usable": np.asarray(usable, bool), "selected_branch": np.asarray(branch, np.int16),
        "selected_inlier_ratio": np.asarray(ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(correspondence_count, np.int64),
        "selected_pnp_inlier_count": np.asarray(inlier_count, np.int64),
        "baseline_inlier_ratio": np.asarray(baseline_ratio, np.float64),
        "sift_inlier_count": np.asarray(sift_inliers, np.int64),
        "sift_candidate_count": np.asarray(sift_candidates, np.int64),
        "radio_global_context_score": np.asarray(radio_scores, np.float64),
        "sift_global_context_score": np.asarray(sift_scores, np.float64),
        "radio_eligible_mapping_view_count": np.asarray(radio_views, np.int64),
        "sift_eligible_mapping_view_count": np.asarray(sift_views, np.int64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_radio_sift_global_context_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(sift["names"])),
        "selection_rule": "maximum_mean_top4_pose_nearby_mapping_view_global_RADIO_tie_RADIO",
        "maximum_distance_m": 10.0, "maximum_direction_degrees": 45.0,
        "top_views": 4, "selection_rule_frozen_on": "existing_Top5Top10_context_ablation",
        "query_pose_or_ground_truth_opened": False, "query_depth_or_scale_used": False,
        "radio_inventory_file_sha256": file_sha256(args.radio_selected_inventory),
        "sift_candidate_inventory_file_sha256": file_sha256(args.sift_candidate_inventory),
        "sift_report_file_sha256": file_sha256(args.sift_report),
        "global_view_field_file_sha256": file_sha256(args.global_view_field),
        "global_view_field_content_sha256": field_meta.get("content_sha256"),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "selected_sift_count": int(np.sum(np.asarray(branch) == 40)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
