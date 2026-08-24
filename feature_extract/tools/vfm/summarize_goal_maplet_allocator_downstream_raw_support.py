"""Seal baseline/f50/union pose-pool and factor-support comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_allocator_downstream_raw_support_comparison_v1"
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)


def _branch(
    coverage_path: Path, direct_path: Path, *, expected_route: str, branch: str,
) -> dict[str, object]:
    coverage_path = Path(coverage_path).resolve()
    direct_path = Path(direct_path).resolve()
    coverage = json.loads(coverage_path.read_text())
    if (
        coverage.get("artifact_type")
        != "goal_maplet_pose_free_factorized_raw_coverage_v1"
        or coverage.get("query_route") != expected_route
        or coverage.get("raw_coverage_is_implicit_factor_support_upper_bound_only")
        is not True
        or coverage.get("score_before_label_separation", {}).get(
            "proposal_frozen_before_direct_dataset_or_gt_opened"
        ) is not True
    ):
        raise ValueError("factorized raw coverage contract differs")
    unhashed = dict(coverage)
    content = str(unhashed.pop("content_sha256", ""))
    if content != canonical_json_sha256(unhashed):
        raise ValueError("factorized raw coverage content hash differs")
    proposal_path = Path(str(coverage["proposal"])).resolve()
    factors, proposal_metadata = load_factorized_pose_free_proposal(proposal_path)
    if (
        file_sha256(proposal_path) != coverage.get("proposal_file_sha256")
        or proposal_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
        or int(proposal_metadata.get("position_seed_count", -1)) != 4
        or int(proposal_metadata.get("orientation_source_prefix_budget", -1)) != 64
    ):
        raise ValueError("factorized proposal strict/budget contract differs")
    direct, direct_metadata = load_pose_candidate_dataset(
        direct_path, require_rendered_targets=False,
    )
    if (
        file_sha256(direct_path) != coverage.get("direct_dataset_file_sha256")
        or direct_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
        or int(direct_metadata.get("candidate_pool_maximum_modes", -1)) != 64
        or not np.array_equal(factors["image_ids"], direct["image_ids"])
    ):
        raise ValueError("direct pool label join strict/budget contract differs")
    valid = np.asarray(direct["candidate_valid"], dtype=bool)[:, 1:]
    translation = np.asarray(direct["translation_m"], dtype=np.float64)[:, 1:]
    rotation = np.asarray(direct["rotation_deg"], dtype=np.float64)[:, 1:]
    pool_hits = {
        name: int(np.sum(np.any(
            valid & (translation <= translation_limit)
            & (rotation <= rotation_limit), axis=1,
        )))
        for name, translation_limit, rotation_limit in THRESHOLDS
    }
    rows = coverage.get("coverage", {}).get("rows", [])
    full = [
        value for value in rows
        if int(value.get("position_seed_budget", -1)) == 4
        and int(value.get("orientation_budget", -1)) == 64
    ]
    if len(full) != 1:
        raise ValueError("factorized raw coverage lacks the fixed 4x64 row")
    factor_hits = {name: int(full[0][name]["hits"]) for name, *_ in THRESHOLDS}
    pool_path = Path(str(coverage["candidate_pool"])).resolve()
    pool = json.loads(pool_path.read_text())
    retrieval_runs = list(pool.get("retrieval_runs", ()))
    if (
        file_sha256(pool_path) != coverage.get("candidate_pool_file_sha256")
        or pool.get("strict_retrieval_promotion_required") is not True
        or int(pool.get("maximum_modes", -1)) != 64
        or not retrieval_runs
    ):
        raise ValueError("pose-free pool strict/budget contract differs")
    retrieval_lineage = []
    for binding in retrieval_runs:
        run_path = Path(str(binding["path"])).resolve()
        if file_sha256(run_path) != binding.get("file_sha256"):
            raise ValueError("pose-free pool retrieval summary bytes differ")
        run = json.loads(run_path.read_text())
        if (
            run.get("promotion_eligible") is not True
            or run.get("control_only") is not False
            or run.get("query_split_audit", {}).get("disjoint") is not True
        ):
            raise ValueError("pose-free pool retrieval summary is not strict")
        if branch == "f50" and (
            run.get("hierarchical_child_allocator_tuning_route") != "seq10"
            or run.get("query_split_audit", {}).get(
                "allocator_tuning_query_disjoint"
            ) is not True
        ):
            raise ValueError("f50 allocator lineage differs")
        if branch == "union" and (
            run.get("union_uses_held_labels") is not False
            or run.get("union_uses_query_pose") is not False
            or run.get("final_pose_pool_budget") != 64
            or run.get("final_factorized_orientation_budget") != 64
        ):
            raise ValueError("union score-before-label/budget lineage differs")
        if branch == "layout" and (
            run.get("layout_child_allocator_tuning_route") != "seq10"
            or run.get("query_split_audit", {}).get(
                "allocator_tuning_query_disjoint"
            ) is not True
            or len(str(run.get("layout_child_allocator_config_file_sha256", "")))
            != 64
            or len(str(run.get("layout_child_allocator_config_content_sha256", "")))
            != 64
        ):
            raise ValueError("layout allocator lineage differs")
        retrieval_lineage.append({
            "path": str(run_path), "file_sha256": file_sha256(run_path),
            "scene_child_selection_semantics": run.get(
                "scene_child_selection_semantics"
            ),
        })
    return {
        "branch": branch,
        "query_count": int(np.asarray(factors["image_ids"]).size),
        "pool_k64_joint_support_count": pool_hits,
        "factorized_4x64_raw_joint_support_count": factor_hits,
        "candidate_pool": str(pool_path),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_pool_content_sha256": str(pool["content_sha256"]),
        "factorized_proposal": str(proposal_path),
        "factorized_proposal_file_sha256": file_sha256(proposal_path),
        "factorized_proposal_content_sha256": str(
            proposal_metadata["content_sha256"]
        ),
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": file_sha256(direct_path),
        "factorized_coverage": str(coverage_path),
        "factorized_coverage_file_sha256": file_sha256(coverage_path),
        "factorized_coverage_content_sha256": content,
        "retrieval_lineage": retrieval_lineage,
    }


def _delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    fields = (
        "pool_k64_joint_support_count", "factorized_4x64_raw_joint_support_count",
    )
    delta = {
        field: {
            name: int(candidate[field][name]) - int(baseline[field][name])
            for name, *_ in THRESHOLDS
        }
        for field in fields
    }
    flat = [value for field in fields for value in delta[field].values()]
    return {
        "delta_from_baseline": delta,
        "pareto_non_degrading": all(value >= 0 for value in flat),
        "strict_pareto_improvement": all(value >= 0 for value in flat)
        and any(value > 0 for value in flat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for route in ("seq12", "seq14"):
        for branch in ("baseline", "f50", "union"):
            parser.add_argument(f"--{route}_{branch}_coverage", required=True)
            parser.add_argument(f"--{route}_{branch}_direct", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite allocator downstream comparison")
    routes: dict[str, object] = {}
    for route in ("seq12", "seq14"):
        branch_rows = {
            branch: _branch(
                Path(getattr(args, f"{route}_{branch}_coverage")),
                Path(getattr(args, f"{route}_{branch}_direct")),
                expected_route=route, branch=branch,
            )
            for branch in ("baseline", "f50", "union")
        }
        if len({value["query_count"] for value in branch_rows.values()}) != 1:
            raise ValueError("allocator comparison query counts differ")
        routes[route] = {
            "branches": branch_rows,
            "f50_comparison": _delta(branch_rows["f50"], branch_rows["baseline"]),
            "union_comparison": _delta(branch_rows["union"], branch_rows["baseline"]),
        }
    robust_gate = {
        branch: all(
            bool(routes[route][f"{branch}_comparison"]["strict_pareto_improvement"])
            for route in routes
        )
        for branch in ("f50", "union")
    }
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "routes": routes,
        "pre_registered_robust_raw_improvement_gate": {
            "semantics": (
                "strict_non_degradation_on_all_pool_and_factor_basin_counts_"
                "on_each_held_route_and_at_least_one_strict_gain_v1"
            ),
            "f50_passed": robust_gate["f50"],
            "union_passed": robust_gate["union"],
        },
        "parent_layout_guide_rerun_permitted": bool(any(robust_gate.values())),
        "decision": (
            "stop_guide_keep_f50_as_auxiliary_branch_do_not_replace_baseline_pose_pool"
        ),
        "held_labels_used_to_select_union_members": False,
        "union_scene_child_input_may_exceed_64": True,
        "final_pose_pool_budget": 64,
        "final_position_seed_budget": 4,
        "final_orientation_budget": 64,
        "raw_factor_support_is_implicit_cartesian_upper_bound": True,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "gate": report["pre_registered_robust_raw_improvement_gate"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
