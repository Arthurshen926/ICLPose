"""Select masked-SIFT plane PnP only with sufficient support, else RADIO PnP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


SIFT_MINIMUM_INLIERS = 16


def _use_sift(candidate_count: int, inlier_count: int) -> bool:
    return int(candidate_count) > 0 and int(inlier_count) >= SIFT_MINIMUM_INLIERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_selected_inventory", type=Path, required=True)
    parser.add_argument("--sift_candidate_inventory", type=Path, required=True)
    parser.add_argument("--sift_report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite SIFT/RADIO hybrid selection")
    with np.load(args.radio_selected_inventory, allow_pickle=False) as data:
        radio_meta = json.loads(str(data["metadata_json"].item()))
        radio = {key: np.asarray(data[key]) for key in (
            "names", "pose_w2c", "usable", "selected_inlier_ratio",
            "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
        )}
    if (
        radio_meta.get("artifact_type") != "goal_maplet_direct_plane_pnp_top5_top10_inlier_selected_v1"
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
        sift_meta.get("artifact_type") != "goal_maplet_masked_sift_plane_pnp_candidate_inventory_v1"
        or sift_meta.get("pose_or_ground_truth_opened") is not False
        or arrays_sha256(sift) != sift_meta.get("arrays_sha256")
        or not np.array_equal(radio["names"], sift["names"])
    ):
        raise ValueError("SIFT candidate inventory differs")
    report = json.loads(args.sift_report.read_text())
    if (
        report.get("candidate_inventory_file_sha256") != file_sha256(args.sift_candidate_inventory)
        or [row["name"] for row in report["pose_free_rows"]]
        != sift["names"].astype(str).tolist()
    ):
        raise ValueError("SIFT report does not bind candidate inventory")

    poses, usable, branch, ratio, correspondence_count, inlier_count = [], [], [], [], [], []
    baseline_ratio, sift_inliers, sift_candidates = [], [], []
    for index in range(len(sift["names"])):
        lo, hi = map(int, sift["candidate_offsets"][index:index + 2])
        best_inliers = 0 if lo == hi else int(sift["candidate_inlier_count"][lo])
        use_sift = _use_sift(hi - lo, best_inliers)
        diagnostic = report["pose_free_rows"][index]
        if use_sift:
            poses.append(sift["candidate_pose_w2c"][lo]); usable.append(True); branch.append(40)
            correspondence_count.append(int(diagnostic["correspondence_count"]))
            inlier_count.append(best_inliers)
            ratio.append(best_inliers / max(int(diagnostic["unique_query_keypoint_count"]), 1))
        else:
            poses.append(radio["pose_w2c"][index]); usable.append(bool(radio["usable"][index])); branch.append(5)
            correspondence_count.append(int(radio["selected_candidate_correspondence_count"][index]))
            inlier_count.append(int(radio["selected_pnp_inlier_count"][index]))
            ratio.append(float(radio["selected_inlier_ratio"][index]))
        baseline_ratio.append(float(radio["selected_inlier_ratio"][index]))
        sift_inliers.append(best_inliers); sift_candidates.append(hi - lo)
    arrays = {
        "names": sift["names"], "pose_w2c": np.asarray(poses, np.float64),
        "usable": np.asarray(usable, bool), "selected_branch": np.asarray(branch, np.int16),
        "selected_inlier_ratio": np.asarray(ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(correspondence_count, np.int64),
        "selected_pnp_inlier_count": np.asarray(inlier_count, np.int64),
        "baseline_inlier_ratio": np.asarray(baseline_ratio, np.float64),
        "sift_inlier_count": np.asarray(sift_inliers, np.int64),
        "sift_candidate_count": np.asarray(sift_candidates, np.int64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_radio_sift_support_hybrid_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(sift["names"])),
        "selection_rule": "masked_SIFT_if_best_unique_keypoint_inliers_ge_16_else_Top5Top10_RADIO",
        "sift_minimum_inlier_count": SIFT_MINIMUM_INLIERS,
        "selection_rule_fitted_on": "seq10_development_smoke15",
        "query_pose_or_ground_truth_opened": False, "query_depth_or_scale_used": False,
        "radio_inventory_file_sha256": file_sha256(args.radio_selected_inventory),
        "sift_candidate_inventory_file_sha256": file_sha256(args.sift_candidate_inventory),
        "sift_report_file_sha256": file_sha256(args.sift_report),
        "selected_sift_count": int(np.sum(np.asarray(branch) == 40)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
