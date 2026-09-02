"""Select RADIO or masked-SIFT plane PnP by pose-nearby spatial RADIO matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_pnp_pose_conditioned_spatial_radio import (
    _compact,
    _load_global_view_field,
    _nearby_rows,
    _projection,
    _spatial_match_score,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _radio, _records
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _select_sift(
    radio_score: tuple[int, float],
    sift_score: tuple[int, float],
    radio_usable: bool,
    sift_usable: bool,
) -> bool:
    if not sift_usable:
        return False
    if not radio_usable:
        return True
    return tuple(sift_score) > tuple(radio_score)


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
        raise FileExistsError("refusing to overwrite spatial-context SIFT/RADIO selection")

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
    view_names = field["names"].astype(str)
    centers = np.asarray(field["camera_centers_world"], np.float64)
    forwards = np.asarray(field["camera_forwards_world"], np.float64)
    records = _records(args.radio_manifest)
    projection = _projection()
    feature_cache: dict[str, np.ndarray] = {}

    def compact(name: str) -> np.ndarray:
        if name not in feature_cache:
            feature_cache[name] = _compact(_radio(name, records), projection)
        return feature_cache[name]

    def score(pose: np.ndarray, query: np.ndarray) -> tuple[tuple[int, float], int]:
        nearby = _nearby_rows(pose, centers, forwards, top_views=4)
        values = [
            _spatial_match_score(query, compact(str(view_names[row])))
            for row in nearby.tolist()
        ]
        return max(values, default=(0, 0.0)), int(len(nearby))

    poses, usable, branch = [], [], []
    ratio, correspondence_count, inlier_count = [], [], []
    baseline_ratio, sift_inliers, sift_candidates = [], [], []
    radio_spatial_inliers, sift_spatial_inliers = [], []
    radio_spatial_cosine, sift_spatial_cosine = [], []
    radio_views, sift_views = [], []
    for index, name in enumerate(sift["names"].astype(str).tolist()):
        lo, hi = map(int, sift["candidate_offsets"][index:index + 2])
        sift_usable = lo < hi
        radio_usable = bool(radio["usable"][index])
        query = compact(name)
        radio_score, radio_count = (
            score(radio["pose_w2c"][index], query) if radio_usable else ((0, 0.0), 0)
        )
        sift_score, sift_count = (
            score(sift["candidate_pose_w2c"][lo], query) if sift_usable else ((0, 0.0), 0)
        )
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
        radio_spatial_inliers.append(radio_score[0]); sift_spatial_inliers.append(sift_score[0])
        radio_spatial_cosine.append(radio_score[1]); sift_spatial_cosine.append(sift_score[1])
        radio_views.append(radio_count); sift_views.append(sift_count)
        if (index + 1) % 10 == 0:
            print(json.dumps({"completed": index + 1, "total": len(sift["names"])}))

    arrays = {
        "names": sift["names"], "pose_w2c": np.asarray(poses, np.float64),
        "usable": np.asarray(usable, bool), "selected_branch": np.asarray(branch, np.int16),
        "selected_inlier_ratio": np.asarray(ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(correspondence_count, np.int64),
        "selected_pnp_inlier_count": np.asarray(inlier_count, np.int64),
        "baseline_inlier_ratio": np.asarray(baseline_ratio, np.float64),
        "sift_inlier_count": np.asarray(sift_inliers, np.int64),
        "sift_candidate_count": np.asarray(sift_candidates, np.int64),
        "radio_spatial_inlier_count": np.asarray(radio_spatial_inliers, np.int64),
        "sift_spatial_inlier_count": np.asarray(sift_spatial_inliers, np.int64),
        "radio_spatial_mean_cosine": np.asarray(radio_spatial_cosine, np.float64),
        "sift_spatial_mean_cosine": np.asarray(sift_spatial_cosine, np.float64),
        "radio_nearby_mapping_view_count": np.asarray(radio_views, np.int64),
        "sift_nearby_mapping_view_count": np.asarray(sift_views, np.int64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_radio_sift_spatial_context_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(sift["names"])),
        "selection_rule": "maximum_pose_nearby_spatial_RADIO_homography_inliers_tie_cosine_then_RADIO",
        "nearby_view_gate": "distance<=10m_and_forward_angle<=45deg_then_closest4",
        "spatial_pool": "2x2_RADIO_tokens_to_18x32",
        "projection": "fixed_seed_1280_to_128_Rademacher_JL",
        "selection_rule_frozen_on": "existing_Top5Top10_spatial_context_ablation",
        "query_pose_or_ground_truth_opened": False, "query_depth_or_scale_used": False,
        "radio_inventory_file_sha256": file_sha256(args.radio_selected_inventory),
        "sift_candidate_inventory_file_sha256": file_sha256(args.sift_candidate_inventory),
        "sift_report_file_sha256": file_sha256(args.sift_report),
        "global_view_field_file_sha256": file_sha256(args.global_view_field),
        "global_view_field_content_sha256": field_meta.get("content_sha256"),
        "radio_manifest_file_sha256_in_order": [file_sha256(path) for path in args.radio_manifest],
        "selected_sift_count": int(np.sum(np.asarray(branch) == 40)),
        "cached_image_count": int(len(feature_cache)),
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
