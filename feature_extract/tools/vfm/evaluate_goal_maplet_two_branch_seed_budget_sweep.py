"""Phase-2 seed-budget/miss decomposition for frozen branch-local factors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    camera_centers_from_w2c,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.two_branch_seed_budget import (
    BRANCH_NAMES,
    POSITION_SEED_BUDGETS,
    load_two_branch_seed_budget,
)


SCHEMA = "goal_maplet_two_branch_position_seed_budget_raw_coverage_v1"
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)
SELECTION_THRESHOLD = 0.95


def _stats(value: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "minimum": int(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": int(np.max(array)),
    }


def _rotation_error_degrees(rotation: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(rotation, dtype=np.float64) @ target[:3, :3].T
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )
    return np.degrees(np.arccos(cosine))


def _select_minimum_budget(
    budget_rows: list[dict[str, object]], *, query_count: int,
    threshold: float = SELECTION_THRESHOLD,
) -> tuple[int | None, int]:
    required_hits = int(math.ceil(float(threshold) * int(query_count) - 1.0e-12))
    qualifying = [
        int(row["position_seed_budget_per_branch"])
        for row in budget_rows
        if int(row["union"]["region_2m_45deg"]["joint_hits"])
        >= required_hits
    ]
    return (min(qualifying) if qualifying else None), required_hits


def _load_direct_target(
    path: Path, *, expected_pool_file: str, expected_pool_content: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata = load_pose_candidate_dataset(
        path, require_rendered_targets=False,
    )
    if (
        metadata.get("artifact_type") != DIRECT_POSE_CANDIDATE_DATASET_SCHEMA
        or metadata.get("candidate_pool_file_sha256") != expected_pool_file
        or metadata.get("candidate_pool_content_sha256") != expected_pool_content
        or metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or metadata.get(
            "nonanchor_candidates_preserve_pose_free_pool_exact_order"
        ) is not True
        or metadata.get(
            "gt_anchor_does_not_change_nonanchor_candidate_membership"
        ) is not True
    ):
        raise ValueError("seed-sweep direct dataset violates post-freeze lineage")
    target_rows = []
    for row, contributor in enumerate(arrays["contributor_paths"].tolist()):
        contributor_path = Path(str(contributor))
        if file_sha256(contributor_path) != str(
            arrays["contributor_file_sha256"][row]
        ):
            raise ValueError("seed-sweep contributor bytes differ")
        with np.load(contributor_path, allow_pickle=False) as data:
            target_rows.append(np.asarray(data["pose_w2c"], dtype=np.float64))
    target = np.stack(target_rows)
    if not np.array_equal(
        target, np.asarray(arrays["candidate_poses_w2c"], dtype=np.float64)[:, 0],
    ):
        raise ValueError("seed-sweep direct diagnostic anchor differs from GT")
    return np.asarray(arrays["image_ids"]), target, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--baseline_direct", required=True)
    parser.add_argument("--layout_direct", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development_selection", action="store_true")
    mode.add_argument("--frozen_selection")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite seed-sweep Phase-2 report")
    started = time.perf_counter()
    proposal_path = Path(args.proposal).resolve()
    arrays, metadata = load_two_branch_seed_budget(proposal_path)
    query_route = str(metadata["query_route"])
    if bool(args.development_selection) != (query_route == "seq10"):
        raise ValueError("development seed-budget selection is restricted to seq10")
    frozen_selection = None
    evaluated_budgets = list(POSITION_SEED_BUDGETS)
    if args.frozen_selection:
        frozen_path = Path(args.frozen_selection).resolve()
        frozen_selection = json.loads(frozen_path.read_text())
        selection = frozen_selection.get("seed_budget_selection", {})
        if (
            frozen_selection.get("artifact_type") != SCHEMA
            or frozen_selection.get("query_route") != "seq10"
            or frozen_selection.get("content_sha256")
            != canonical_json_sha256({
                key: value for key, value in frozen_selection.items()
                if key != "content_sha256"
            })
            or selection.get("decision") != "GO"
            or selection.get("threshold") != SELECTION_THRESHOLD
            or int(selection.get("selected_position_seed_budget", -1))
            not in POSITION_SEED_BUDGETS
        ):
            raise ValueError("held seed sweep lacks a valid frozen seq10 selection")
        evaluated_budgets = [int(selection["selected_position_seed_budget"])]
    direct_paths = [Path(args.baseline_direct).resolve(), Path(args.layout_direct).resolve()]
    direct = [
        _load_direct_target(
            direct_paths[branch],
            expected_pool_file=str(metadata[f"{name}_pool_file_sha256"]),
            expected_pool_content=str(metadata[f"{name}_pool_content_sha256"]),
        )
        for branch, name in enumerate(BRANCH_NAMES)
    ]
    if (
        not np.array_equal(direct[0][0], arrays["image_ids"])
        or not np.array_equal(direct[1][0], arrays["image_ids"])
        or not np.array_equal(direct[0][1], direct[1][1])
    ):
        raise ValueError("seed-sweep branch direct targets differ")
    target = direct[0][1]
    query_count = int(target.shape[0])
    target_center = camera_centers_from_w2c(target)
    branch_translation = []
    branch_rotation = []
    for branch in range(2):
        branch_translation.append(np.linalg.norm(
            arrays["branch_position_centers_world"][:, branch]
            - target_center[:, None, None, :], axis=3,
        ))
        rotation_rows = []
        for query in range(query_count):
            valid = arrays["branch_orientation_valid"][query, branch]
            error = np.full((64,), np.inf, dtype=np.float64)
            error[valid] = _rotation_error_degrees(
                arrays["branch_orientation_rotations_w2c"][query, branch, valid],
                target[query],
            )
            rotation_rows.append(error)
        branch_rotation.append(np.stack(rotation_rows))

    budget_rows = []
    per_query_rows = []
    query_evidence: dict[int, dict[str, object]] = {}
    for budget in evaluated_budgets:
        budget_index = POSITION_SEED_BUDGETS.index(budget)
        branch_best_translation = [
            np.min(value[:, :budget], axis=(1, 2)) for value in branch_translation
        ]
        branch_best_rotation = [np.min(value, axis=1) for value in branch_rotation]
        branch_metrics = {}
        union_metrics = {}
        per_threshold_masks = {}
        for threshold_name, translation_limit, rotation_limit in THRESHOLDS:
            branch_position_hit = [
                value <= translation_limit for value in branch_best_translation
            ]
            branch_orientation_hit = [
                value <= rotation_limit for value in branch_best_rotation
            ]
            branch_joint_hit = [
                branch_position_hit[index] & branch_orientation_hit[index]
                for index in range(2)
            ]
            for branch, name in enumerate(BRANCH_NAMES):
                branch_metrics.setdefault(name, {})[threshold_name] = {
                    "position_only_hits": int(np.sum(branch_position_hit[branch])),
                    "orientation_only_hits": int(np.sum(branch_orientation_hit[branch])),
                    "joint_hits": int(np.sum(branch_joint_hit[branch])),
                }
            union_position = branch_position_hit[0] | branch_position_hit[1]
            union_orientation = branch_orientation_hit[0] | branch_orientation_hit[1]
            union_joint = branch_joint_hit[0] | branch_joint_hit[1]
            cross_branch_false = union_position & union_orientation & ~union_joint
            union_metrics[threshold_name] = {
                "position_only_hits": int(np.sum(union_position)),
                "orientation_only_hits": int(np.sum(union_orientation)),
                "joint_hits": int(np.sum(union_joint)),
                "joint_rate": float(np.mean(union_joint)),
                "position_misses": int(np.sum(~union_position)),
                "orientation_misses": int(np.sum(~union_orientation)),
                "joint_misses": int(np.sum(~union_joint)),
                "cross_branch_only_false_support_if_illegally_crossed": int(
                    np.sum(cross_branch_false)
                ),
            }
            per_threshold_masks[threshold_name] = (
                branch_position_hit, branch_orientation_hit, branch_joint_hit,
                union_position, union_orientation, union_joint,
            )
        budget_rows.append({
            "position_seed_budget_per_branch": budget,
            "orientation_budget_per_branch": 64,
            "branch": branch_metrics,
            "union": union_metrics,
            "unique_position_seed_count": _stats(
                arrays["unique_position_seed_count_by_budget"][:, budget_index]
            ),
            "unique_position_factor_count": _stats(
                arrays["unique_position_factor_count_by_budget"][:, budget_index]
            ),
            "unique_orientation_factor_count": _stats(
                arrays["unique_orientation_factor_count"]
            ),
            "implicit_lattice_pose_pair_count": _stats(
                arrays["implicit_lattice_pose_pair_count_by_budget"][:, budget_index]
            ),
        })
        query_evidence[budget] = {
            "best_translation": branch_best_translation,
            "best_rotation": branch_best_rotation,
            "threshold_masks": per_threshold_masks,
        }
    selected, required_hits = _select_minimum_budget(
        budget_rows, query_count=query_count,
    )
    seed_budget_selection = None
    if bool(args.development_selection):
        seed_budget_selection = {
            "selection_route": "seq10",
            "threshold_metric": "union_region_2m_45deg_absolute_query_rate",
            "threshold": SELECTION_THRESHOLD,
            "required_hits": required_hits,
            "query_count": query_count,
            "candidate_budgets_in_preregistered_order": list(POSITION_SEED_BUDGETS),
            "selected_position_seed_budget": selected,
            "decision": "GO" if selected is not None else "KILL",
            "held_routes_opened_for_selection": False,
        }
    selected_for_rows = (
        seed_budget_selection.get("selected_position_seed_budget")
        if seed_budget_selection is not None
        else evaluated_budgets[0]
    )
    if selected_for_rows is None:
        selected_for_rows = max(evaluated_budgets)
    evidence = query_evidence[int(selected_for_rows)]
    for query in range(query_count):
        threshold_rows = {}
        for name, masks in evidence["threshold_masks"].items():
            bp, bo, bj, up, uo, uj = masks
            threshold_rows[name] = {
                "baseline": {
                    "position": bool(bp[0][query]), "orientation": bool(bo[0][query]),
                    "joint": bool(bj[0][query]),
                },
                "layout": {
                    "position": bool(bp[1][query]), "orientation": bool(bo[1][query]),
                    "joint": bool(bj[1][query]),
                },
                "union": {
                    "position": bool(up[query]), "orientation": bool(uo[query]),
                    "joint": bool(uj[query]),
                },
            }
        per_query_rows.append({
            "query_index": query,
            "image_id": str(arrays["image_ids"][query]),
            "reported_position_seed_budget_per_branch": int(selected_for_rows),
            "best_translation_m": {
                name: float(evidence["best_translation"][branch][query])
                for branch, name in enumerate(BRANCH_NAMES)
            },
            "best_rotation_deg": {
                name: float(evidence["best_rotation"][branch][query])
                for branch, name in enumerate(BRANCH_NAMES)
            },
            "threshold_support": threshold_rows,
        })
    phase2_seconds = float(time.perf_counter() - started)
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_route": query_route,
        "query_count": query_count,
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": str(metadata["content_sha256"]),
        "baseline_direct": str(direct_paths[0]),
        "baseline_direct_file_sha256": file_sha256(direct_paths[0]),
        "baseline_direct_content_sha256": str(direct[0][2]["content_sha256"]),
        "layout_direct": str(direct_paths[1]),
        "layout_direct_file_sha256": file_sha256(direct_paths[1]),
        "layout_direct_content_sha256": str(direct[1][2]["content_sha256"]),
        "evaluated_position_seed_budgets": evaluated_budgets,
        "orientation_budget_per_branch": 64,
        "budget_rows": budget_rows,
        "seed_budget_selection": seed_budget_selection,
        "frozen_seq10_selection": (
            {
                "path": str(Path(args.frozen_selection).resolve()),
                "file_sha256": file_sha256(Path(args.frozen_selection).resolve()),
                "content_sha256": str(frozen_selection["content_sha256"]),
            }
            if args.frozen_selection else None
        ),
        "domain_semantics": (
            "(P_baseline_x_O_baseline)_union_(P_layout_x_O_layout)_no_cross_v1"
        ),
        "cross_branch_cartesian_products_included": False,
        "phase2_elapsed_seconds": phase2_seconds,
        "score_before_label_separation": {
            "proposal_builder_accepts_no_direct_or_query_pose_input": True,
            "proposal_frozen_before_this_phase2_label_join": True,
            "target_pose_opened_only_by_this_phase2_evaluator": True,
            "held_labels_used_for_budget_selection": False,
        },
        "raw_support_is_implicit_factor_upper_bound_not_localization_success": True,
        "position_collision_free_space_certified": False,
        "parent_layout_guide_run": False,
        "query_rows_at_selected_or_max_budget": per_query_rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "query_route": query_route,
        "query_count": query_count,
        "budget_rows": budget_rows,
        "seed_budget_selection": seed_budget_selection,
        "phase2_elapsed_seconds": phase2_seconds,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
