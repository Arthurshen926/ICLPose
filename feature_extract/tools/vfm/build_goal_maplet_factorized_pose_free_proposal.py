"""Freeze a query-pose-free factorized position/orientation proposal artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    SCHEMA,
    SEMANTICS,
    build_factorized_pose_free_proposal_arrays,
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    load_pose_free_candidate_pool,
)


def _atomic_save(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--position_seed_count", type=int, default=4)
    parser.add_argument("--orientation_budget", type=int, default=64)
    parser.add_argument("--position_step_m", type=float, default=2.0)
    parser.add_argument("--position_xz_half_extent_m", type=float, default=10.0)
    parser.add_argument("--position_y_half_extent_m", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite factorized proposal outputs")

    pool_path = Path(args.candidate_pool).resolve()
    pool = load_pose_free_candidate_pool(pool_path)
    atlas_path = Path(str(pool.get("atlas", ""))).resolve()
    if not atlas_path.is_file() or pool.get("atlas_file_sha256") != file_sha256(atlas_path):
        raise ValueError("factorized proposal pool atlas bytes differ")
    arrays = build_factorized_pose_free_proposal_arrays(
        pool,
        position_seed_count=int(args.position_seed_count),
        orientation_budget=int(args.orientation_budget),
        step_m=float(args.position_step_m),
        xz_half_extent_m=float(args.position_xz_half_extent_m),
        y_half_extent_m=float(args.position_y_half_extent_m),
    )
    valid_orientation = np.asarray(arrays["orientation_valid"], dtype=bool)
    position_count = int(np.prod(np.asarray(arrays["position_centers_world"]).shape[1:3]))
    atlas_routes = {
        str(value)
        for value in pool["route_disjoint_atlas_audit"].get(
            "allowed_trajectories", ()
        )
    }
    strict_retrieval_audits = pool.get("strict_retrieval_promotion_audits", [])
    strict_v4_retrieval = bool(
        pool.get("strict_retrieval_promotion_required") is True
        and isinstance(strict_retrieval_audits, list)
        and strict_retrieval_audits
        and all(
            isinstance(value, dict)
            and value.get("promotion_eligible") is True
            and value.get("control_only") is False
            and value.get("query_split_disjoint") is True
            and value.get("validity_calibration_fit_trajectories") == ["seq10"]
            for value in strict_retrieval_audits
        )
    )
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_count": int(np.asarray(arrays["image_ids"]).size),
        "query_route": str(pool.get("query_route", "")),
        "candidate_pool": str(pool_path),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_pool_content_sha256": str(pool["content_sha256"]),
        "candidate_pool_atlas": str(atlas_path),
        "candidate_pool_atlas_file_sha256": file_sha256(atlas_path),
        "candidate_pool_atlas_content_sha256": str(pool["atlas_content_sha256"]),
        "route_disjoint_atlas_audit": pool["route_disjoint_atlas_audit"],
        "strict_retrieval_promotion_required": bool(
            pool.get("strict_retrieval_promotion_required", False)
        ),
        "strict_retrieval_promotion_audits": strict_retrieval_audits,
        "strict_v4_seq10_calibrated_retrieval_confirmed": strict_v4_retrieval,
        "strict_mapper_fit_val_only_atlas_excluding_seq10_12_14": bool(
            not ({"seq10", "seq12", "seq14"} & atlas_routes)
        ),
        "contains_seq10_calibration_route_in_atlas": bool("seq10" in atlas_routes),
        "position_seed_count": int(args.position_seed_count),
        "position_seed_semantics": "strict_pose_free_pool_prefix_v1",
        "position_step_m": float(args.position_step_m),
        "position_xz_half_extent_m": float(args.position_xz_half_extent_m),
        "position_y_half_extent_m": float(args.position_y_half_extent_m),
        "positions_per_seed": int(np.asarray(arrays["position_offsets_camera"]).shape[0]),
        "stored_position_count_per_query": position_count,
        "orientation_source_semantics": (
            "stable_unique_w2c_rotations_from_same_pose_free_pool_prefix_v1"
        ),
        "orientation_source_prefix_budget": int(args.orientation_budget),
        "minimum_stored_orientation_count_per_query": int(
            np.min(np.sum(valid_orientation, axis=1))
        ),
        "maximum_stored_orientation_count_per_query": int(
            np.max(np.sum(valid_orientation, axis=1))
        ),
        "implicit_cartesian_pose_count_per_query_maximum": int(
            position_count * int(args.orientation_budget)
        ),
        "cartesian_pose_product_materialized": False,
        "position_lattice_occupancy_checked": False,
        "position_collision_free_space_certified": False,
        "position_lattice_may_include_occupied_or_physically_unreachable_centers": True,
        "raw_coverage_is_implicit_factor_support_upper_bound_only": True,
        "proposal_semantics": "coarse_factor_support_not_a_final_q_pose_v1",
        "candidate_pool_numeric_scores_consumed": False,
        "candidate_pool_prefix_order_consumed": True,
        "query_pose_member_opened_during_generation": False,
        "query_ground_truth_member_opened_during_generation": False,
        "direct_label_dataset_opened_during_generation": False,
        "generation_configuration_status": (
            "seq12_seq14_posthoc_development_control_contains_seq10_calibration_"
            "route_and_uses_nonpreregistered_lattice_constants"
            if "seq10" in atlas_routes
            else "strict_fit_val_only_atlas_but_seq12_seq14_posthoc_development_"
            "lattice_constants_are_not_final_test_preregistered"
        ),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "output_is_final_pose": False,
    }
    _atomic_save(output, arrays, metadata)
    # Re-open through the strict reader before publishing the sidecar.  This
    # validates exact NPZ membership as well as geometry and padding contracts.
    load_factorized_pose_free_proposal(output)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "content_sha256": metadata["content_sha256"],
        "query_count": metadata["query_count"],
        "stored_position_count_per_query": position_count,
        "orientation_count_range": [
            metadata["minimum_stored_orientation_count_per_query"],
            metadata["maximum_stored_orientation_count_per_query"],
        ],
        "cartesian_pose_product_materialized": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
