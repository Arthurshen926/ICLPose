"""Phase-2 basin survival for a sealed parent-layout score run.

All rankings are loaded and hash-validated from a Phase-1 run before this
command opens the direct diagnostic labels.  The evaluator never changes a
rank or selects a hyperparameter from labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_factorized_parent_layout_guide import (
    THRESHOLDS,
    _load_score,
    _rotation_error_degrees,
)
from feature_extract.tools.vfm.seal_goal_maplet_factorized_parent_layout_guide_run import (
    SCHEMA as PHASE1_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    camera_centers_from_w2c,
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    adapt_ranked_factor_pairs_for_exact_scorer,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_factorized_parent_support_layout_guide_phase2_run_v1"


def _validated_phase1(path: Path) -> dict[str, object]:
    report = json.loads(Path(path).read_text())
    content = str(report.get("content_sha256", ""))
    unhashed = dict(report)
    unhashed.pop("content_sha256", None)
    contract = report.get("score_before_label_contract")
    if (
        report.get("artifact_type") != PHASE1_SCHEMA
        or content != canonical_json_sha256(unhashed)
        or report.get("strict_v4_required") is not True
        or report.get("strict_v4_seq10_calibrated_retrieval_confirmed") is not True
        or not isinstance(contract, dict)
        or contract.get("label_or_direct_dataset_argument_accepted") is not False
        or contract.get("query_pose_or_gt_argument_accepted") is not False
        or contract.get("contributor_argument_accepted") is not False
        or contract.get("contributor_file_bytes_hashed") is not False
        or contract.get("contributor_pose_member_opened") is not False
        or contract.get("camera_binding_is_intrinsics_and_image_id_only") is not True
        or contract.get("ranked_scores_are_frozen_by_file_and_content_hash") is not True
    ):
        raise ValueError("parent layout Phase-1 run violates score-before-label")
    rows = report.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) != int(report.get("query_count", -1))
    ):
        raise ValueError("parent layout Phase-1 row inventory differs")
    return report


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("cannot summarize an empty/non-finite metric")
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def evaluate_parent_layout_score_run(
    *,
    phase1_path: Path,
    direct_path: Path,
    prefix_budgets: tuple[int, ...],
) -> dict[str, object]:
    phase1_path = Path(phase1_path).resolve()
    direct_path = Path(direct_path).resolve()
    phase1 = _validated_phase1(phase1_path)
    rows = list(phase1["rows"])

    # Freeze and validate every score byte before opening the first label.
    frozen_scores: list[tuple[dict[str, np.ndarray], dict[str, object]]] = []
    for expected_index, row in enumerate(rows):
        score_path = Path(str(row.get("score", ""))).resolve()
        if (
            int(row.get("query_index", -1)) != expected_index
            or row.get("score_file_sha256") != file_sha256(score_path)
        ):
            raise ValueError("parent layout Phase-1 frozen score bytes differ")
        score, metadata = _load_score(score_path)
        if (
            metadata.get("content_sha256") != row.get("score_content_sha256")
            or metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed") is not True
            or int(metadata.get("query_index", -1)) != expected_index
            or str(np.asarray(score["image_id"]).item()) != row.get("image_id")
        ):
            raise ValueError("parent layout Phase-1 frozen score content differs")
        frozen_scores.append((score, metadata))

    proposal_path = Path(str(phase1["proposal"])).resolve()
    if (
        phase1.get("proposal_file_sha256") != file_sha256(proposal_path)
        or len(frozen_scores) != int(phase1["query_count"])
    ):
        raise ValueError("parent layout Phase-1 proposal bytes differ")
    factors, proposal_metadata = load_factorized_pose_free_proposal(proposal_path)
    if (
        proposal_metadata.get("content_sha256") != phase1.get("proposal_content_sha256")
        or proposal_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed") is not True
    ):
        raise ValueError("parent layout Phase-1 proposal lineage differs")

    # Phase 2 begins here.  The direct artifact contains the diagnostic GT
    # anchor and is deliberately not accepted by the Phase-1 sealing API.
    direct, direct_metadata = load_pose_candidate_dataset(
        direct_path, require_rendered_targets=False,
    )
    if (
        direct_metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA
        or direct_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or direct_metadata.get("nonanchor_candidates_preserve_pose_free_pool_exact_order")
        is not True
        or direct_metadata.get("gt_anchor_does_not_change_nonanchor_candidate_membership")
        is not True
        or direct_metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
        or direct_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed") is not True
        or direct_metadata.get("candidate_pool_content_sha256")
        != phase1.get("candidate_pool_content_sha256")
    ):
        raise ValueError("parent layout Phase-2 direct label contract differs")
    image_ids = [str(value) for value in np.asarray(factors["image_ids"]).tolist()]
    if image_ids != [str(value) for value in np.asarray(direct["image_ids"]).tolist()]:
        raise ValueError("parent layout Phase-2 query order differs")

    budgets = tuple(sorted(set(int(value) for value in prefix_budgets)))
    if not budgets or min(budgets) <= 0 or max(budgets) > int(phase1["returned_topk_per_query"]):
        raise ValueError("parent layout Phase-2 prefix budgets differ")
    raw_joint = {name: 0 for name in THRESHOLDS}
    raw_pool_joint = {name: 0 for name in THRESHOLDS}
    raw_position_2m = 0
    prefix = {
        budget: {
            "position_region_2m_count": 0,
            "joint_support_count": {name: 0 for name in THRESHOLDS},
            "distinct_position": [],
            "distinct_orientation": [],
        }
        for budget in budgets
    }
    first_hit_values: dict[str, list[int]] = {name: [] for name in THRESHOLDS}
    query_rows: list[dict[str, object]] = []
    elapsed_values: list[float] = []

    for query_index, ((score, metadata), image_id) in enumerate(
        zip(frozen_scores, image_ids)
    ):
        target = np.asarray(direct["candidate_poses_w2c"], dtype=np.float64)[
            query_index, 0,
        ]
        target_center = camera_centers_from_w2c(target)
        positions = np.asarray(
            factors["position_centers_world"], dtype=np.float64,
        )[query_index].reshape(-1, 3)
        rotations = np.asarray(
            factors["orientation_rotations_w2c"], dtype=np.float64,
        )[query_index]
        orientation_valid = np.asarray(
            factors["orientation_valid"], dtype=bool,
        )[query_index]
        orientation_source = np.asarray(
            factors["orientation_source_candidate_ranks"], dtype=np.int64,
        )[query_index]
        position_index = np.asarray(
            score["top_position_factor_indices"], dtype=np.int64,
        )
        orientation_index = np.asarray(
            score["top_orientation_factor_indices"], dtype=np.int64,
        )
        translation_error = np.linalg.norm(
            positions[position_index] - target_center[None], axis=1,
        )
        rotation_error = _rotation_error_degrees(
            rotations[orientation_index], target,
        )
        raw_translation = np.linalg.norm(
            positions - target_center[None], axis=1,
        )
        raw_rotation = _rotation_error_degrees(rotations[orientation_valid], target)
        pool_valid = np.asarray(direct["candidate_valid"], dtype=bool)[query_index, 1:]
        pool_translation = np.asarray(
            direct["translation_m"], dtype=np.float64,
        )[query_index, 1:][pool_valid]
        pool_rotation = np.asarray(
            direct["rotation_deg"], dtype=np.float64,
        )[query_index, 1:][pool_valid]
        position_region = bool(np.any(raw_translation <= 2.0))
        raw_position_2m += int(position_region)
        raw_query: dict[str, bool] = {}
        raw_pool_query: dict[str, bool] = {}
        first_query: dict[str, int | None] = {}
        for name, (translation_limit, rotation_limit) in THRESHOLDS.items():
            # Position and orientation are independent factors: this is the
            # support of the implicit Cartesian product, not a rendered pose.
            supported = bool(
                np.any(raw_translation <= translation_limit)
                and np.any(raw_rotation <= rotation_limit)
            )
            pool_supported = bool(np.any(
                (pool_translation <= translation_limit)
                & (pool_rotation <= rotation_limit)
            ))
            raw_joint[name] += int(supported)
            raw_pool_joint[name] += int(pool_supported)
            raw_query[name] = supported
            raw_pool_query[name] = pool_supported
            matches = np.flatnonzero(
                (translation_error <= translation_limit)
                & (rotation_error <= rotation_limit)
            )
            first_rank = None if matches.size == 0 else int(matches[0] + 1)
            first_query[name] = first_rank
            if first_rank is not None:
                first_hit_values[name].append(first_rank)
        for budget in budgets:
            prefix[budget]["position_region_2m_count"] += int(
                np.any(translation_error[:budget] <= 2.0)
            )
            prefix[budget]["distinct_position"].append(int(
                np.unique(position_index[:budget]).size
            ))
            prefix[budget]["distinct_orientation"].append(int(
                np.unique(orientation_index[:budget]).size
            ))
            for name, (translation_limit, rotation_limit) in THRESHOLDS.items():
                prefix[budget]["joint_support_count"][name] += int(np.any(
                    (translation_error[:budget] <= translation_limit)
                    & (rotation_error[:budget] <= rotation_limit)
                ))

        # This also gates the exact-scorer handoff geometry/provenance.  It
        # materializes only the already selected Top-K pairs.
        adapted = adapt_ranked_factor_pairs_for_exact_scorer(
            np.asarray(factors["position_centers_world"])[query_index],
            rotations,
            orientation_valid,
            orientation_source,
            position_index,
            orientation_index,
            np.asarray(score["top_orientation_source_candidate_ranks"]),
            np.asarray(score["top_scores"]),
            maximum_pairs=max(budgets),
        )
        if adapted["candidate_poses_w2c"].shape != (max(budgets), 4, 4):
            raise AssertionError("exact-scorer adapter returned a different budget")
        elapsed_values.append(float(metadata["elapsed_seconds"]))
        query_rows.append({
            "query_index": query_index,
            "image_id": image_id,
            "raw_position_region_2m": position_region,
            "raw_implicit_joint_support": raw_query,
            "raw_pose_free_pool_joint_support": raw_pool_query,
            "first_guide_hit_rank": first_query,
            "score_file_sha256": str(rows[query_index]["score_file_sha256"]),
        })

    query_count = len(image_ids)
    prefix_rows = []
    for budget in budgets:
        counts = dict(prefix[budget]["joint_support_count"])
        prefix_rows.append({
            "prefix_budget": budget,
            "compression_from_all_implicit_factor_pairs": float(
                int(phase1["total_factor_pair_count_per_query"]) / budget
            ),
            "position_region_2m_count": int(prefix[budget]["position_region_2m_count"]),
            "position_region_2m_rate": float(
                prefix[budget]["position_region_2m_count"] / query_count
            ),
            "position_region_2m_conditional_retention": (
                float(prefix[budget]["position_region_2m_count"] / raw_position_2m)
                if raw_position_2m else None
            ),
            "joint_support_count": counts,
            "joint_support_rate": {
                name: float(counts[name] / query_count) for name in THRESHOLDS
            },
            "conditional_retention_of_raw_implicit_support": {
                name: (
                    float(counts[name] / raw_joint[name])
                    if raw_joint[name] else None
                )
                for name in THRESHOLDS
            },
            "distinct_position_factor_count": _summary(
                prefix[budget]["distinct_position"]
            ),
            "distinct_orientation_factor_count": _summary(
                prefix[budget]["distinct_orientation"]
            ),
        })
    first_hit = {}
    for name, values in first_hit_values.items():
        first_hit[name] = {
            "found_within_returned_topk_count": len(values),
            "missing_within_returned_topk_count": query_count - len(values),
            "rank_summary_when_found": _summary(values) if values else None,
        }
    direct_mtime = int(direct_path.stat().st_mtime_ns)
    latest_score_mtime = int(phase1["latest_score_file_mtime_ns"])
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_count": query_count,
        "query_route": str(phase1["query_route"]),
        "phase1_score_run": str(phase1_path),
        "phase1_score_run_file_sha256": file_sha256(phase1_path),
        "phase1_score_run_content_sha256": str(phase1["content_sha256"]),
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": str(direct_metadata["content_sha256"]),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(proposal_metadata["content_sha256"]),
        "strict_v4_seq10_calibrated_retrieval_confirmed": True,
        "phase_separation_audit": {
            "all_score_bytes_hash_frozen_before_direct_labels_opened_by_evaluator": True,
            "latest_score_file_mtime_ns": latest_score_mtime,
            "direct_dataset_file_mtime_ns": direct_mtime,
            "direct_dataset_created_after_all_scores": bool(
                direct_mtime > latest_score_mtime
            ),
            "phase1_api_accepts_no_label_direct_contributor_or_query_pose_input": True,
            "phase2_does_not_change_ranking_or_configuration": True,
        },
        "fixed_development_configuration": {
            "position_seed_count": int(proposal_metadata["position_seed_count"]),
            "position_step_m": float(proposal_metadata["position_step_m"]),
            "position_xz_half_extent_m": float(
                proposal_metadata["position_xz_half_extent_m"]
            ),
            "position_y_half_extent_m": float(
                proposal_metadata["position_y_half_extent_m"]
            ),
            "orientation_source_prefix_budget": int(
                proposal_metadata["orientation_source_prefix_budget"]
            ),
            "maximum_query_parents": 32,
            "guide_returned_topk": int(phase1["returned_topk_per_query"]),
            "posthoc_development_constants_not_final_test_preregistered": True,
            "configuration_tuned_on_this_phase2_labels": False,
        },
        "raw_pose_free_pool_joint_support_count": raw_pool_joint,
        "raw_position_factor_region_2m_count": raw_position_2m,
        "raw_implicit_cartesian_joint_support_count": raw_joint,
        "prefix_rows": prefix_rows,
        "first_hit_rank": first_hit,
        "runtime_seconds": {
            **_summary(elapsed_values),
            "sum": float(np.sum(elapsed_values)),
            "parallel_runner_wall_seconds": phase1.get(
                "parallel_runner_wall_seconds"
            ),
        },
        "exact_scorer_adapter_contract": {
            "adapter": (
                "adapt_ranked_factor_pairs_for_exact_scorer(position_centers_world,"
                " orientation_rotations_w2c, orientation_valid, orientation_source_"
                "candidate_ranks, ranked_position_indices, ranked_orientation_indices,"
                " ranked_orientation_source_ranks, ranked_scores, maximum_pairs=K)"
            ),
            "output_pose_semantics": "w2c_R_and_t_equal_minus_R_times_camera_center_v1",
            "materializes_only_selected_factor_pairs": True,
            "preserves_guide_order_and_source_rank_provenance": True,
            "accepts_query_pose_or_ground_truth": False,
        },
        "metric_semantics": (
            "coarse_guide_candidate_survival_before_exact_footprint_scoring_not_"
            "final_localization_success_v1"
        ),
        "position_lattice_collision_or_free_space_certified": False,
        "raw_implicit_support_is_an_upper_bound": True,
        "query_rows": query_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_score_run", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--prefix_budgets", default="1,4,8,16,32,64")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent layout Phase-2 run")
    budgets = tuple(int(value) for value in str(args.prefix_budgets).split(","))
    report = evaluate_parent_layout_score_run(
        phase1_path=Path(args.phase1_score_run),
        direct_path=Path(args.direct_dataset),
        prefix_budgets=budgets,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "query_count": report["query_count"],
        "raw_position_factor_region_2m_count": report[
            "raw_position_factor_region_2m_count"
        ],
        "raw_implicit_cartesian_joint_support_count": report[
            "raw_implicit_cartesian_joint_support_count"
        ],
        "prefix_rows": report["prefix_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
