"""Phase-2 pose coverage of a frozen parent-support coarse-guide ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    camera_centers_from_w2c,
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    NORMAL_CONTRACT,
    SCORE_SEMANTICS,
    TOKEN_FOOTPRINT_PHASE,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_factorized_parent_support_layout_guide_phase2_coverage_v1"
SCORE_SCHEMA = "goal_maplet_factorized_parent_support_layout_coarse_guide_v1"
SCORE_ARRAY_NAMES = {
    "image_id", "selected_query_parent_ids",
    "selected_query_parent_probability_mass", "top_scores",
    "top_position_factor_indices", "top_position_seed_indices",
    "top_position_offset_indices", "top_orientation_factor_indices",
    "top_orientation_source_candidate_ranks", "top_visible_parent_counts",
    "top_front_facing_parent_counts", "top_positive_depth_parent_counts",
    "top_center_in_image_parent_counts", "top_projected_token_footprint_mass",
    "top_sqrt_overlap_mass",
}
THRESHOLDS = {
    "region_2m_45deg": (2.0, 45.0),
    "loose_1m_10deg": (1.0, 10.0),
    "strict_0_5m_5deg": (0.5, 5.0),
}


def _load_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with zipfile.ZipFile(path, mode="r") as archive:
        members = [value.filename for value in archive.infolist()]
    expected = {*(f"{name}.npy" for name in SCORE_ARRAY_NAMES), "metadata_json.npy"}
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("parent layout score NPZ members differ")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {*SCORE_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("parent layout score arrays differ")
        arrays = {name: np.asarray(data[name]) for name in SCORE_ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if (
        metadata.get("artifact_type") != SCORE_SCHEMA
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("score_semantics") != SCORE_SEMANTICS
        or metadata.get("normal_contract") != NORMAL_CONTRACT
        or metadata.get("token_footprint_phase") != TOKEN_FOOTPRINT_PHASE
        or metadata.get("uses_signed_mapping_camera_oriented_normal") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("uses_pnp") is not False
        or metadata.get("uses_point_correspondences") is not False
        or metadata.get("output_is_final_q_pose") is not False
    ):
        raise ValueError("parent layout score violates the strict score-before-label contract")
    return arrays, metadata


def _rotation_error_degrees(rotation: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(rotation, dtype=np.float64) @ target[:3, :3].T
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )
    return np.degrees(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--prefix_budgets", default="1,4,8,16,32,64,128,256,512")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent layout Phase-2 coverage")
    budgets = tuple(sorted({
        int(value) for value in str(args.prefix_budgets).split(",")
    }))
    if not budgets or min(budgets) <= 0:
        raise ValueError("parent layout Phase-2 prefix budgets are invalid")

    score_path = Path(args.score).resolve()
    direct_path = Path(args.direct_dataset).resolve()
    score, score_metadata = _load_score(score_path)
    proposal_path = Path(str(score_metadata["proposal"])).resolve()
    if file_sha256(proposal_path) != score_metadata.get("proposal_file_sha256"):
        raise ValueError("parent layout Phase-2 proposal bytes differ")
    factors, factor_metadata = load_factorized_pose_free_proposal(proposal_path)
    direct, direct_metadata = load_pose_candidate_dataset(
        direct_path, require_rendered_targets=False,
    )
    if (
        direct_metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA
        or direct_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or direct_metadata.get(
            "nonanchor_candidates_preserve_pose_free_pool_exact_order"
        ) is not True
        or direct_metadata.get(
            "gt_anchor_does_not_change_nonanchor_candidate_membership"
        ) is not True
    ):
        raise ValueError("parent layout Phase-2 direct label contract differs")
    query_index = int(score_metadata["query_index"])
    image_id = str(np.asarray(score["image_id"]).item())
    if (
        query_index < 0 or query_index >= np.asarray(factors["image_ids"]).size
        or str(factors["image_ids"][query_index]) != image_id
        or str(direct["image_ids"][query_index]) != image_id
    ):
        raise ValueError("parent layout Phase-2 query identity differs")
    target = np.asarray(direct["candidate_poses_w2c"], dtype=np.float64)[
        query_index, 0,
    ]
    target_center = camera_centers_from_w2c(target)
    position_all = np.asarray(
        factors["position_centers_world"], dtype=np.float64,
    )[query_index].reshape(-1, 3)
    rotation_all = np.asarray(
        factors["orientation_rotations_w2c"], dtype=np.float64,
    )[query_index]
    valid_all = np.asarray(factors["orientation_valid"], dtype=bool)[query_index]
    position_indices = np.asarray(
        score["top_position_factor_indices"], dtype=np.int64,
    )
    orientation_indices = np.asarray(
        score["top_orientation_factor_indices"], dtype=np.int64,
    )
    if (
        position_indices.shape != orientation_indices.shape
        or np.any((position_indices < 0) | (position_indices >= position_all.shape[0]))
        or np.any((orientation_indices < 0) | (orientation_indices >= rotation_all.shape[0]))
        or np.any(~valid_all[orientation_indices])
    ):
        raise ValueError("parent layout Phase-2 factor indices differ")
    translation_error = np.linalg.norm(
        position_all[position_indices] - target_center[None], axis=1,
    )
    rotation_error = _rotation_error_degrees(
        rotation_all[orientation_indices], target,
    )
    raw_translation = np.linalg.norm(position_all - target_center[None], axis=1)
    raw_rotation = _rotation_error_degrees(rotation_all[valid_all], target)
    raw_support = {}
    for name, (translation_limit, rotation_limit) in THRESHOLDS.items():
        raw_support[name] = bool(
            np.any(raw_translation <= translation_limit)
            and np.any(raw_rotation <= rotation_limit)
        )

    rows = []
    for requested in budgets:
        budget = min(int(requested), int(position_indices.size))
        hit = {}
        for name, (translation_limit, rotation_limit) in THRESHOLDS.items():
            hit[name] = bool(np.any(
                (translation_error[:budget] <= translation_limit)
                & (rotation_error[:budget] <= rotation_limit)
            ))
        rows.append({
            "requested_prefix_budget": int(requested),
            "effective_prefix_budget": budget,
            "compression_from_all_factor_pairs": float(
                int(score_metadata["total_factor_pair_count"]) / budget
            ),
            "distinct_position_factor_count": int(
                np.unique(position_indices[:budget]).size
            ),
            "distinct_orientation_factor_count": int(
                np.unique(orientation_indices[:budget]).size
            ),
            "minimum_translation_error_m": float(np.min(translation_error[:budget])),
            "minimum_rotation_error_deg": float(np.min(rotation_error[:budget])),
            "joint_support": hit,
        })
    first_hit_rank = {}
    for name, (translation_limit, rotation_limit) in THRESHOLDS.items():
        matches = np.flatnonzero(
            (translation_error <= translation_limit)
            & (rotation_error <= rotation_limit)
        )
        first_hit_rank[name] = None if matches.size == 0 else int(matches[0] + 1)
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "score": str(score_path),
        "score_file_sha256": file_sha256(score_path),
        "score_content_sha256": str(score_metadata["content_sha256"]),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(factor_metadata["content_sha256"]),
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": str(direct_metadata["content_sha256"]),
        "image_id": image_id,
        "query_index": query_index,
        "total_factor_pair_count": int(score_metadata["total_factor_pair_count"]),
        "returned_rank_count": int(position_indices.size),
        "raw_implicit_factor_support": raw_support,
        "raw_minimum_translation_error_m": float(np.min(raw_translation)),
        "raw_minimum_rotation_error_deg": float(np.min(raw_rotation)),
        "first_hit_rank": first_hit_rank,
        "prefix_rows": rows,
        "score_before_label_separation": {
            "score_frozen_before_this_phase2_direct_dataset_open": True,
            "score_artifact_contains_no_gt_or_query_pose": True,
            "query_gt_consumed_only_by_this_phase2_evaluator": True,
        },
        "normal_contract": NORMAL_CONTRACT,
        "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
        "score_protocol_status": str(score_metadata.get("protocol_status", "")),
        "strict_v4_seq10_calibrated_retrieval_confirmed": bool(
            score_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed", False)
        ),
        "seq12_result_is_calibration_leaky_posthoc_upper_bound": bool(
            score_metadata.get("seq12_result_is_calibration_leaky_posthoc_upper_bound", False)
        ),
        "metric_semantics": (
            "coarse_guide_candidate_survival_upper_bound_before_exact_footprint_"
            "scoring_not_localization_success_v1"
        ),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "image_id": image_id,
        "raw_implicit_factor_support": raw_support,
        "first_hit_rank": first_hit_rank,
        "prefix_rows": rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
