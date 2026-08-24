"""Phase-2 raw coverage for a frozen factorized pose-free proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    factorized_raw_coverage,
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_pose_free_factorized_raw_coverage_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--position_seed_budgets", default="1,2,4")
    parser.add_argument("--orientation_budgets", default="4,8,16,32,64")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite factorized coverage")
    position_budgets = tuple(sorted({
        int(value) for value in str(args.position_seed_budgets).split(",")
    }))
    orientation_budgets = tuple(sorted({
        int(value) for value in str(args.orientation_budgets).split(",")
    }))
    if not position_budgets or not orientation_budgets:
        raise ValueError("factorized coverage budgets are empty")

    proposal_path = Path(args.proposal).resolve()
    direct_path = Path(args.direct_dataset).resolve()
    arrays, metadata = load_factorized_pose_free_proposal(proposal_path)
    pool_path = Path(str(metadata["candidate_pool"])).resolve()
    if (
        not pool_path.is_file()
        or file_sha256(pool_path) != metadata.get("candidate_pool_file_sha256")
    ):
        raise ValueError("factorized proposal pose-free pool bytes differ")
    direct, direct_metadata = load_pose_candidate_dataset(
        direct_path, require_rendered_targets=False,
    )
    if (
        direct_metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA
        or direct_metadata.get("candidate_pool_file_sha256")
        != metadata.get("candidate_pool_file_sha256")
        or direct_metadata.get("candidate_pool_content_sha256")
        != metadata.get("candidate_pool_content_sha256")
        or direct_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or direct_metadata.get(
            "nonanchor_candidates_preserve_pose_free_pool_exact_order"
        ) is not True
        or direct_metadata.get(
            "gt_anchor_does_not_change_nonanchor_candidate_membership"
        ) is not True
        or direct_metadata.get(
            "pose_free_pool_internal_duplicates_rejected_before_gt_join"
        ) is not True
    ):
        raise ValueError("factorized coverage direct labels violate the post-freeze join")
    image_ids = np.asarray(arrays["image_ids"])
    if not np.array_equal(image_ids, np.asarray(direct["image_ids"])):
        raise ValueError("factorized proposal/direct query identities differ")

    target_poses: list[np.ndarray] = []
    for row, (image_id, contributor) in enumerate(zip(
        image_ids.tolist(), np.asarray(direct["contributor_paths"]).tolist(),
    )):
        path = Path(str(contributor))
        if file_sha256(path) != str(direct["contributor_file_sha256"][row]):
            raise ValueError("factorized Phase-2 contributor bytes differ")
        with np.load(path, allow_pickle=False) as data:
            target_pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        if target_pose.shape != (4, 4) or np.any(~np.isfinite(target_pose)):
            raise ValueError("factorized Phase-2 target pose is invalid")
        target_poses.append(target_pose)
    target = np.stack(target_poses)
    if not np.array_equal(
        target,
        np.asarray(direct["candidate_poses_w2c"], dtype=np.float64)[:, 0],
    ):
        raise ValueError("factorized Phase-2 GT differs from direct diagnostic anchor")

    coverage = factorized_raw_coverage(
        arrays["position_centers_world"],
        arrays["orientation_rotations_w2c"], arrays["orientation_valid"], target,
        position_seed_budgets=position_budgets,
        orientation_budgets=orientation_budgets,
    )
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(metadata["content_sha256"]),
        "proposal_factor_arrays_sha256": arrays_sha256(arrays),
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": str(direct_metadata["content_sha256"]),
        "candidate_pool": str(pool_path),
        "candidate_pool_file_sha256": str(metadata["candidate_pool_file_sha256"]),
        "candidate_pool_content_sha256": str(
            metadata["candidate_pool_content_sha256"]
        ),
        "query_route": str(metadata["query_route"]),
        "position_seed_budgets": list(position_budgets),
        "orientation_budgets": list(orientation_budgets),
        "coverage": coverage,
        "metric_semantics": (
            "factorized_position_orientation_implicit_cartesian_support_upper_bound_"
            "not_ranked_not_searched_not_localization_success_v1"
        ),
        "raw_coverage_is_implicit_factor_support_upper_bound_only": True,
        "position_lattice_occupancy_checked": False,
        "position_collision_free_space_certified": False,
        "raw_coverage_does_not_certify_collision_free_or_reachable_camera_centers": True,
        "score_before_label_separation": {
            "proposal_frozen_before_direct_dataset_or_gt_opened": True,
            "query_gt_opened_only_by_this_phase2_evaluator": True,
            "direct_nonanchor_error_labels_consumed": False,
            "direct_diagnostic_anchor_used_only_to_audit_gt_convention": True,
        },
        "cartesian_pose_product_materialized": False,
        "generation_configuration_status": metadata[
            "generation_configuration_status"
        ],
        "strict_mapper_fit_val_only_atlas_excluding_seq10_12_14": bool(
            metadata.get("strict_mapper_fit_val_only_atlas_excluding_seq10_12_14")
        ),
        "contains_seq10_calibration_route_in_atlas": bool(
            metadata.get("contains_seq10_calibration_route_in_atlas")
        ),
        "production_eligible": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    full_rows = [
        row for row in coverage["rows"]
        if int(row["position_seed_budget"]) == max(position_budgets)
    ]
    print(json.dumps({
        "output": str(output.resolve()),
        "query_route": report["query_route"],
        "query_count": coverage["query_count"],
        "position_only": coverage["full_factor_position_only"],
        "orientation_only": coverage["full_factor_orientation_only"],
        "full_position_seed_budget_joint_by_orientation_budget": full_rows,
        "cartesian_pose_product_materialized": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
