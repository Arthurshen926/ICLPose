"""Score one frozen factorized query with the pose-free parent layout guide.

The executable has no query-pose, ground-truth, correspondence, PnP, or RGB
input.  Camera calibration is passed explicitly and is bound by a canonical
intrinsics-only hash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    camera_intrinsics_content_sha256,
    load_pose_free_candidate_pool,
)
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    NORMAL_CONTRACT,
    ParentLayoutCamera,
    SCORE_SEMANTICS,
    TOKEN_FOOTPRINT_PHASE,
    score_parent_support_layout_guide,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)


SCHEMA = "goal_maplet_factorized_parent_support_layout_coarse_guide_v1"
HISTORICAL_SEQ12_FITTED_VALIDITY_SHA256 = (
    "091751f0dd21cc892bb19be716089688abcfeb741bec1596af130f8066a2cc59"
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
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--query_index", type=int, required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--camera_model_id", type=int, required=True)
    parser.add_argument("--camera_width", type=int, required=True)
    parser.add_argument("--camera_height", type=int, required=True)
    parser.add_argument("--camera_params", nargs="+", type=float, required=True)
    parser.add_argument("--maximum_query_parents", type=int, default=64)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent layout guide outputs")

    proposal_path = Path(args.proposal).resolve()
    retrieval_path = Path(args.retrieval).resolve()
    physical_path = Path(args.physical_map).resolve()
    factors, proposal_metadata = load_factorized_pose_free_proposal(proposal_path)
    query_index = int(args.query_index)
    image_ids = np.asarray(factors["image_ids"])
    if query_index < 0 or query_index >= image_ids.size:
        raise IndexError("parent layout guide query index is outside the proposal")
    image_id = str(image_ids[query_index])
    retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    if retrieval.image_id != image_id:
        raise ValueError("parent layout guide retrieval/proposal image identity differs")
    if retrieval.metadata.get("physical_map_file_sha256") != file_sha256(physical_path):
        raise ValueError("parent layout guide physical map file lineage differs")

    pool_path = Path(str(proposal_metadata["candidate_pool"])).resolve()
    pool = load_pose_free_candidate_pool(pool_path)
    row_by_image = {str(row["image_id"]): row for row in pool["rows"]}
    pool_row = row_by_image.get(image_id)
    if pool_row is None:
        raise ValueError("parent layout guide image is absent from the frozen pool")
    declared_retrieval = Path(str(pool_row["retrieval_artifact"])).resolve()
    if declared_retrieval != retrieval_path:
        raise ValueError("parent layout guide retrieval is not the frozen pool input")
    if str(pool_row["retrieval_content_sha256"]) != retrieval.content_sha256:
        raise ValueError("parent layout guide retrieval content differs from the frozen pool")

    camera = ParentLayoutCamera(
        int(args.camera_model_id), int(args.camera_width), int(args.camera_height),
        tuple(float(value) for value in args.camera_params),
    )
    start = time.perf_counter()
    result = score_parent_support_layout_guide(
        np.asarray(factors["position_centers_world"])[query_index],
        np.asarray(factors["orientation_rotations_w2c"])[query_index],
        np.asarray(factors["orientation_valid"])[query_index],
        np.asarray(factors["orientation_source_candidate_ranks"])[query_index],
        retrieval, physical, camera,
        maximum_query_parents=int(args.maximum_query_parents),
        topk=int(args.topk), candidate_batch_size=int(args.candidate_batch_size),
    )
    elapsed = time.perf_counter() - start
    validity_sha = str(retrieval.metadata.get("validity_calibration_file_sha256", ""))
    strict_audits = pool.get("strict_retrieval_promotion_audits", [])
    strict_v4_retrieval = bool(
        pool.get("strict_retrieval_promotion_required") is True
        and proposal_metadata.get(
            "strict_v4_seq10_calibrated_retrieval_confirmed"
        ) is True
        and isinstance(strict_audits, list) and strict_audits
        and all(
            isinstance(value, dict)
            and value.get("promotion_eligible") is True
            and value.get("control_only") is False
            and value.get("query_split_disjoint") is True
            and value.get("validity_calibration_fit_trajectories") == ["seq10"]
            and value.get("query_route") == str(pool.get("query_route", ""))
            for value in strict_audits
        )
    )
    if strict_v4_retrieval:
        protocol_status = "strict_v4_seq10_calibrated_route_disjoint_retrieval"
    elif validity_sha == HISTORICAL_SEQ12_FITTED_VALIDITY_SHA256:
        protocol_status = (
            "historical_v3_retrieval_control_seq12_fitted_validity_calibration_"
            "not_v4_seq10_calibrated"
        )
    else:
        protocol_status = "unverified_retrieval_calibration_lineage_control"
    arrays = {
        "image_id": np.asarray(image_id),
        "selected_query_parent_ids": np.asarray(result.selected_query_parent_ids),
        "selected_query_parent_probability_mass": np.asarray(
            result.selected_query_parent_probability_mass,
        ),
        "top_scores": np.asarray(result.top_scores),
        "top_position_factor_indices": np.asarray(
            result.top_position_factor_indices,
        ),
        "top_position_seed_indices": np.asarray(result.top_position_seed_indices),
        "top_position_offset_indices": np.asarray(result.top_position_offset_indices),
        "top_orientation_factor_indices": np.asarray(
            result.top_orientation_factor_indices,
        ),
        "top_orientation_source_candidate_ranks": np.asarray(
            result.top_orientation_source_candidate_ranks,
        ),
        "top_visible_parent_counts": np.asarray(result.top_visible_parent_counts),
        "top_front_facing_parent_counts": np.asarray(
            result.top_front_facing_parent_counts,
        ),
        "top_positive_depth_parent_counts": np.asarray(
            result.top_positive_depth_parent_counts,
        ),
        "top_center_in_image_parent_counts": np.asarray(
            result.top_center_in_image_parent_counts,
        ),
        "top_projected_token_footprint_mass": np.asarray(
            result.top_projected_token_footprint_mass,
        ),
        "top_sqrt_overlap_mass": np.asarray(result.top_sqrt_overlap_mass),
    }
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "content_sha256": arrays_sha256(arrays),
        "image_id": image_id,
        "query_index": query_index,
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(proposal_metadata["content_sha256"]),
        "retrieval": str(retrieval_path),
        "retrieval_file_sha256": file_sha256(retrieval_path),
        "retrieval_content_sha256": retrieval.content_sha256,
        "retrieval_validity_calibration_file_sha256": validity_sha,
        "protocol_status": protocol_status,
        "strict_v4_seq10_calibrated_retrieval_confirmed": strict_v4_retrieval,
        "seq12_result_is_calibration_leaky_posthoc_upper_bound": bool(
            image_id.startswith("seq12/")
            and validity_sha == HISTORICAL_SEQ12_FITTED_VALIDITY_SHA256
        ),
        "physical_map": str(physical_path),
        "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_content_sha256": physical.content_sha256,
        "camera_model_id": camera.model_id,
        "camera_width": camera.width,
        "camera_height": camera.height,
        "camera_params": list(camera.params),
        "camera_intrinsics_only_content_sha256": camera_intrinsics_content_sha256(
            image_id, camera.model_id, camera.width, camera.height, camera.params,
        ),
        "score_semantics": SCORE_SEMANTICS,
        "normal_contract": NORMAL_CONTRACT,
        "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
        "complete_query_parent_probability_mass": (
            result.complete_query_parent_probability_mass
        ),
        "selected_query_parent_probability_mass_total": (
            result.selected_query_parent_probability_mass_total
        ),
        "selected_query_parent_count": int(result.selected_query_parent_ids.size),
        "total_factor_pair_count": result.total_factor_pair_count,
        "scored_orientation_count": result.scored_orientation_count,
        "returned_topk": int(result.top_scores.size),
        "candidate_batch_size": int(args.candidate_batch_size),
        "elapsed_seconds": float(elapsed),
        "query_denominator_fixed_across_candidates": True,
        "projected_parent_center_rectangle_depth_unsigned_incidence_explicit": True,
        "uses_signed_mapping_camera_oriented_normal": False,
        "projected_support_is_binary_intersected_radio_token_footprint": True,
        "top_factor_pairs_are_distinct": True,
        "cartesian_pose_product_materialized": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "uses_mapping_rgb": False,
        "uses_gaussian_rendering": False,
        "is_coarse_parent_support_guide": True,
        "output_is_final_q_pose": False,
        "occlusion_modeled": False,
        "collision_free_space_certified": False,
        "production_eligible": False,
    }
    _atomic_save(output, arrays, metadata)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
        "top_score": float(result.top_scores[0]),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "image_id": image_id,
        "total_factor_pair_count": result.total_factor_pair_count,
        "top_score": float(result.top_scores[0]),
        "top_position_factor_index": int(result.top_position_factor_indices[0]),
        "top_orientation_factor_index": int(result.top_orientation_factor_indices[0]),
        "elapsed_seconds": float(elapsed),
        "output_is_final_q_pose": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
