"""Phase-2 raw coverage of a frozen two-branch factor-domain union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_factorized_parent_layout_guide import (
    _rotation_error_degrees,
)
from feature_extract.vfm.localization_goal_maplet.factorized_branch_union import (
    BRANCH_NAMES,
    load_factorized_branch_union,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    camera_centers_from_w2c,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_factorized_two_branch_domain_union_raw_coverage_v1"
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)


def _stats(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": int(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": int(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union_proposal", required=True)
    parser.add_argument("--baseline_direct", required=True)
    parser.add_argument("--layout_direct", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite branch-union raw coverage")
    union_path = Path(args.union_proposal).resolve()
    arrays, metadata = load_factorized_branch_union(union_path)
    baseline_path = Path(args.baseline_direct).resolve()
    layout_path = Path(args.layout_direct).resolve()
    baseline, baseline_metadata = load_pose_candidate_dataset(
        baseline_path, require_rendered_targets=False,
    )
    layout, layout_metadata = load_pose_candidate_dataset(
        layout_path, require_rendered_targets=False,
    )
    image_ids = np.asarray(arrays["image_ids"])
    if (
        not np.array_equal(image_ids, baseline["image_ids"])
        or not np.array_equal(image_ids, layout["image_ids"])
        or baseline_metadata.get("candidate_pool_file_sha256")
        != metadata.get("baseline_candidate_pool_file_sha256")
        or baseline_metadata.get("candidate_pool_content_sha256")
        != metadata.get("baseline_candidate_pool_content_sha256")
        or layout_metadata.get("candidate_pool_file_sha256")
        != metadata.get("layout_candidate_pool_file_sha256")
        or layout_metadata.get("candidate_pool_content_sha256")
        != metadata.get("layout_candidate_pool_content_sha256")
        or baseline_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
        or layout_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
    ):
        raise ValueError("branch-union Phase-2 direct label lineage differs")
    baseline_target = np.asarray(
        baseline["candidate_poses_w2c"], dtype=np.float64,
    )[:, 0]
    layout_target = np.asarray(
        layout["candidate_poses_w2c"], dtype=np.float64,
    )[:, 0]
    if not np.array_equal(baseline_target, layout_target):
        raise ValueError("branch-union Phase-2 target anchors differ")

    branch_hits = {
        branch: {name: 0 for name, *_ in THRESHOLDS}
        for branch in BRANCH_NAMES
    }
    union_hits = {name: 0 for name, *_ in THRESHOLDS}
    baseline_only = {name: 0 for name, *_ in THRESHOLDS}
    layout_only = {name: 0 for name, *_ in THRESHOLDS}
    query_rows = []
    unique_seed_counts: list[int] = []
    unique_position_counts: list[int] = []
    unique_orientation_counts: list[int] = []
    implicit_pair_counts: list[int] = []
    for query in range(image_ids.size):
        target = baseline_target[query]
        target_center = camera_centers_from_w2c(target)
        branch_support: dict[str, dict[str, bool]] = {}
        best: dict[str, dict[str, float]] = {}
        for branch_index, branch in enumerate(BRANCH_NAMES):
            seed_indices = np.asarray(
                arrays["branch_position_seed_to_unique"][query, branch_index],
                dtype=np.int64,
            )
            positions = np.asarray(
                arrays["unique_position_centers_world"], dtype=np.float64,
            )[query, seed_indices].reshape(-1, 3)
            orientation_valid = np.asarray(
                arrays["branch_orientation_valid"], dtype=bool,
            )[query, branch_index]
            orientation_indices = np.asarray(
                arrays["branch_orientation_to_unique"], dtype=np.int64,
            )[query, branch_index, orientation_valid]
            rotations = np.asarray(
                arrays["unique_orientation_rotations_w2c"], dtype=np.float64,
            )[query, orientation_indices]
            translation_error = np.linalg.norm(
                positions - target_center[None], axis=1,
            )
            rotation_error = _rotation_error_degrees(rotations, target)
            branch_support[branch] = {}
            best[branch] = {
                "translation_m": float(np.min(translation_error)),
                "rotation_deg": float(np.min(rotation_error)),
            }
            for name, translation_limit, rotation_limit in THRESHOLDS:
                hit = bool(
                    np.any(translation_error <= translation_limit)
                    and np.any(rotation_error <= rotation_limit)
                )
                branch_support[branch][name] = hit
                branch_hits[branch][name] += int(hit)
        union_support = {}
        for name, *_ in THRESHOLDS:
            baseline_hit = branch_support["baseline"][name]
            layout_hit = branch_support["layout"][name]
            hit = baseline_hit or layout_hit
            union_support[name] = hit
            union_hits[name] += int(hit)
            baseline_only[name] += int(baseline_hit and not layout_hit)
            layout_only[name] += int(layout_hit and not baseline_hit)
            if hit < baseline_hit or hit < layout_hit:
                raise AssertionError("branch-domain OR violated mathematical monotonicity")
        valid_seed = np.asarray(
            arrays["unique_position_seed_valid"], dtype=bool,
        )[query]
        positions = np.asarray(
            arrays["unique_position_centers_world"], dtype=np.float64,
        )[query, valid_seed].reshape(-1, 3)
        unique_seed_counts.append(int(np.sum(valid_seed)))
        unique_position_counts.append(int(np.unique(
            positions.round(10), axis=0,
        ).shape[0]))
        unique_orientation_counts.append(int(np.sum(
            arrays["unique_orientation_valid"][query]
        )))
        implicit_pair_counts.append(int(
            arrays["implicit_lattice_pose_pair_count_by_query"][query]
        ))
        query_rows.append({
            "query_index": query,
            "image_id": str(image_ids[query]),
            "branch_support": branch_support,
            "union_support": union_support,
            "best_independent_factor_errors": best,
        })
    for branch in BRANCH_NAMES:
        for name, *_ in THRESHOLDS:
            if union_hits[name] < branch_hits[branch][name]:
                raise AssertionError("aggregate branch-domain union coverage decreased")
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_route": str(metadata["query_route"]),
        "query_count": int(image_ids.size),
        "union_proposal": str(union_path),
        "union_proposal_file_sha256": file_sha256(union_path),
        "union_proposal_content_sha256": str(metadata["content_sha256"]),
        "baseline_direct": str(baseline_path),
        "baseline_direct_file_sha256": file_sha256(baseline_path),
        "baseline_direct_content_sha256": str(baseline_metadata["content_sha256"]),
        "layout_direct": str(layout_path),
        "layout_direct_file_sha256": file_sha256(layout_path),
        "layout_direct_content_sha256": str(layout_metadata["content_sha256"]),
        "branch_raw_support_count": branch_hits,
        "union_raw_support_count": union_hits,
        "branch_exclusive_support_count": {
            "baseline_only": baseline_only,
            "layout_only": layout_only,
        },
        "unique_position_seed_count": _stats(unique_seed_counts),
        "unique_position_factor_count": _stats(unique_position_counts),
        "unique_orientation_factor_count": _stats(unique_orientation_counts),
        "implicit_lattice_pose_pair_count": _stats(implicit_pair_counts),
        "domain_semantics": (
            "(P_baseline_x_O_baseline)_union_(P_layout_x_O_layout)_no_cross_v1"
        ),
        "mathematically_non_decreasing_against_each_branch": True,
        "cross_branch_cartesian_products_included": False,
        "cartesian_pose_product_materialized": False,
        "score_before_label_separation": {
            "union_proposal_builder_accepts_no_direct_label_or_query_pose_input": True,
            "source_branch_proposals_frozen_before_label_join": True,
            "target_anchor_opened_only_by_this_phase2_evaluator": True,
            "held_labels_used_to_select_union_factors": False,
        },
        "raw_support_is_implicit_factor_upper_bound_not_localization_success": True,
        "position_collision_free_space_certified": False,
        "parent_layout_guide_run": False,
        "query_rows": query_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "query_route": report["query_route"],
        "branch_raw_support_count": branch_hits,
        "union_raw_support_count": union_hits,
        "unique_position_seed_count": report["unique_position_seed_count"],
        "unique_orientation_factor_count": report[
            "unique_orientation_factor_count"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
