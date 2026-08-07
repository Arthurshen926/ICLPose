"""Evaluate frozen-pool sparse relation inference and configuration-diverse modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import rotation_angle_deg


MODE = "actual_parent_actual_child"


def _rank_percentile(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(-value, kind="stable")
    output = np.ones(value.shape, dtype=np.float64) * 0.5
    if value.size > 1:
        position = np.linspace(1.0, 0.0, value.size)
        sorted_value = value[order]
        start = 0
        while start < value.size:
            end = start + 1
            while end < value.size and sorted_value[end] == sorted_value[start]:
                end += 1
            output[order[start:end]] = float(np.mean(position[start:end]))
            start = end
    return output


def _metrics(translation: np.ndarray, rotation: np.ndarray, oracle: np.ndarray) -> dict:
    t, r, o = (np.asarray(value, dtype=np.float64) for value in (translation, rotation, oracle))
    success = (t <= 0.5) & (r <= 5.0)
    return {
        "query_count": int(t.size),
        "translation_median_m": float(np.median(t)),
        "translation_p90_m": float(np.percentile(t, 90.0)),
        "translation_p95_m": float(np.percentile(t, 95.0)),
        "rotation_median_deg": float(np.median(r)),
        "rotation_p90_deg": float(np.percentile(r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean(success)),
        "top1_1m_10deg_fraction": float(np.mean((t <= 1.0) & (r <= 10.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (r > 10.0))),
        "median_selection_regret_m": float(np.median(t - o)),
        "p90_selection_regret_m": float(np.percentile(t - o, 90.0)),
    }


def _configuration_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.int64).reshape(-1)
    b = np.asarray(right, dtype=np.int64).reshape(-1)
    union = (a >= 0) | (b >= 0)
    return float(np.mean(a[union] != b[union])) if np.any(union) else 0.0


def _diverse_topn(
    order: np.ndarray,
    assignments: np.ndarray,
    poses: np.ndarray,
    count: int,
) -> np.ndarray:
    chosen: list[int] = []
    for candidate in np.asarray(order, dtype=np.int64).tolist():
        keep = True
        for selected in chosen:
            pose_a, pose_b = poses[candidate], poses[selected]
            center_a = -pose_a[:3, :3].T @ pose_a[:3, 3]
            center_b = -pose_b[:3, :3].T @ pose_b[:3, 3]
            different = (
                _configuration_distance(assignments[candidate], assignments[selected]) >= 0.25
                or float(np.linalg.norm(center_a - center_b)) >= 0.5
                or rotation_angle_deg(pose_a[:3, :3], pose_b[:3, :3]) >= 5.0
            )
            if not different:
                keep = False
                break
        if keep:
            chosen.append(int(candidate))
            if len(chosen) >= int(count):
                break
    return np.asarray(chosen, dtype=np.int64)


def _risk_coverage(rows: list[dict], policy: str) -> dict:
    ordered = sorted(rows, key=lambda row: (-float(row[policy]["confidence"]), row["image_id"]))
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(np.ceil(coverage * len(ordered))))
        selected = ordered[:count]
        result[f"{coverage:.1f}"] = _metrics(
            [row[policy]["translation_m"] for row in selected],
            [row[policy]["rotation_deg"] for row in selected],
            [row["oracle"]["translation_m"] for row in selected],
        )
    return result


def _aggregate_endpoint_coverage(rows: list[dict]) -> dict:
    adaptive = bool(rows) and "truth_child_adaptive_budget" in rows[0]["endpoint_coverage"]
    names = (
        (
            "truth_child_full_posterior", "truth_child_adaptive_budget",
            "truth_primitive_top8_given_adaptive_child",
            "truth_primitive_adaptive_budget", "endpoint_pose_valid",
        ) if adaptive else (
            "truth_child_top16", "truth_child_parent_quota_top2",
            "truth_child_family_top8", "truth_primitive_top8",
            "truth_primitive_relation_top2", "endpoint_pose_valid",
        )
    )
    denominator = int(sum(int(row["endpoint_coverage"]["observable_group_count"]) for row in rows))
    result = {"observable_group_count": denominator}
    previous = denominator
    for name in names:
        count = int(sum(int(row["endpoint_coverage"][name]["count"]) for row in rows))
        result[name] = {
            "count": count, "fraction": float(count / max(denominator, 1)),
            "conditional_survival": float(count / max(previous, 1)),
        }
        previous = count
    pair_count = int(sum(
        int(row["endpoint_coverage"]["pair_both_endpoints_valid"]["count"]) for row in rows
    ))
    edge_count = int(sum(
        int(row["endpoint_coverage"]["pair_both_endpoints_valid"]["edge_count"]) for row in rows
    ))
    result["pair_both_endpoints_valid"] = {
        "count": pair_count, "edge_count": edge_count,
        "fraction": float(pair_count / max(edge_count, 1)),
    }
    return result


def _optional_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite mode-relation audit")
    payload = json.loads(Path(args.candidate_pool).read_text())
    contract = dict(payload.get("configuration_evidence_contract", {}))
    if contract.get("mode_relation_inference") not in (
        "mass_conserving_exact_sum_product_fit_tree_pairwise_heldout_v2",
        "mass_adaptive_hierarchical_exact_sum_product_fit_tree_v3",
    ):
        raise ValueError("candidate pool lacks mass-conserving relation evidence v2")
    rows = []
    policy_names = (
        "proposal", "g13_equal_rank", "g14_research_baseline", "relation_v2_node_only",
        "relation_v2_node_fit", "relation_v2_heldout_only", "relation_v2_joint",
        "g13_relation_v2_node_fit_equal_rank",
    )
    topn = {name: {3: [], 5: [], "diverse3": []} for name in policy_names}
    for row in payload["rows"]:
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        base = diagnostics["configuration_evidence_v4"]
        relation = diagnostics["mode_relation_evidence_v2"]
        assignment = np.asarray(
            diagnostics["mode_relation_assignments_v2"]["selected_child_rows"], dtype=np.int64,
        )
        poses = np.asarray([item["pose_w2c"] for item in details], dtype=np.float64)
        translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
        rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
        utility = translation / 0.5 + rotation / 5.0
        oracle = int(np.argmin(utility))
        legacy = np.asarray(base["configuration_legacy_valid_factor_mass_mean"], dtype=np.float64)
        llr_median = np.asarray(base["configuration_pose_llr_fixed_median"], dtype=np.float64)
        g13 = _rank_percentile(legacy) + _rank_percentile(llr_median)
        node = np.asarray(relation["relation_node_log_evidence_mean"], dtype=np.float64)
        fit = np.asarray(relation["relation_fit_incremental_llr_mean"], dtype=np.float64)
        verify = np.asarray(relation["relation_verify_predictive_llr_mean"], dtype=np.float64)
        verify_valid = np.asarray(relation["relation_verify_valid_fraction"], dtype=np.float64)
        old_relation = diagnostics.get("mode_relation_evidence_v1", {})
        old_verify = np.asarray(
            old_relation.get("relation_verify_llr_median", np.zeros_like(verify)), dtype=np.float64,
        )
        score = {
            "proposal": np.asarray(diagnostics["proposal_scores"], dtype=np.float64),
            "g13_equal_rank": g13,
            "g14_research_baseline": _rank_percentile(g13) + _rank_percentile(old_verify),
            "relation_v2_node_only": node,
            "relation_v2_node_fit": node + fit,
            "relation_v2_heldout_only": verify,
            # All three terms are mean log likelihood-ratio evidence on their
            # own fixed denominators; no promotion threshold or Dev-fitted
            # override is introduced.
            "relation_v2_joint": node + fit + verify,
            # Preserve the independently useful G13 local/configuration
            # statistic while injecting only relation evidence observed on the
            # fit tree.  This is a fixed scale-free fusion, not a threshold or
            # a fitted component weight.
            "g13_relation_v2_node_fit_equal_rank": (
                _rank_percentile(g13) + _rank_percentile(node + fit)
            ),
        }
        covered = bool(np.any((translation <= 0.5) & (rotation <= 5.0)))
        result = {
            "image_id": row["image_id"], "trajectory": row["image_id"].split("/", 1)[0],
            "proposal_covered": covered,
            "oracle": {
                "rank": oracle + 1, "translation_m": float(translation[oracle]),
                "rotation_deg": float(rotation[oracle]),
            },
            "relation_v2_query_diagnostics": diagnostics[
                "mode_relation_assignments_v2"
            ].get("query_diagnostics", {}),
            "endpoint_coverage": diagnostics["mode_relation_endpoint_coverage_v2"],
        }
        gt_evidence = diagnostics["mode_relation_gt_pose_evidence_v2"]
        candidate_non_null = np.asarray(
            relation["relation_posterior_non_null_mass_mean"], dtype=np.float64,
        )
        phase_mask = (
            (translation > 0.5) & (translation <= 3.0) & (rotation <= 5.0)
        )
        result["probability_semantics"] = {
            "gt_pose_non_null_mass": float(
                gt_evidence["relation_posterior_non_null_mass_mean"]
            ),
            "near_miss_phase_non_null_mass_median": float(np.median(
                candidate_non_null[phase_mask]
            )) if np.any(phase_mask) else None,
            "oracle_candidate_non_null_mass": float(candidate_non_null[oracle]),
        }
        for name, value in score.items():
            order = np.argsort(-value, kind="stable")
            selected = int(order[0])
            margin = float(value[order[0]] - value[order[1]]) if order.size > 1 else 0.0
            confidence = margin * max(float(verify_valid[selected]), 0.05)
            result[name] = {
                "rank": selected + 1, "translation_m": float(translation[selected]),
                "rotation_deg": float(rotation[selected]), "confidence": confidence,
                "score_margin": margin, "verify_valid_fraction": float(verify_valid[selected]),
                "ranking_failure": bool(
                    covered and not (translation[selected] <= 0.5 and rotation[selected] <= 5.0)
                ),
            }
            for count in (3, 5):
                keep = order[: min(count, order.size)]
                topn[name][count].append(bool(np.any(
                    (translation[keep] <= 0.5) & (rotation[keep] <= 5.0)
                )))
            diverse = _diverse_topn(order, assignment, poses, 3)
            topn[name]["diverse3"].append(bool(np.any(
                (translation[diverse] <= 0.5) & (rotation[diverse] <= 5.0)
            )))
            if name in ("relation_v2_joint", "g13_relation_v2_node_fit_equal_rank"):
                keep = order[: min(5, order.size)]
                non_null_mass = np.asarray(
                    relation["relation_posterior_non_null_mass_mean"], dtype=np.float64,
                )
                entropy = np.asarray(relation["relation_posterior_entropy_mean"], dtype=np.float64)
                result[name].update({
                    "poses_topk": [
                        {"candidate_index": int(index), "pose_w2c": poses[index].tolist()}
                        for index in keep.tolist()
                    ],
                    "configuration_assignments_topk": assignment[keep].tolist(),
                    "endpoint_valid_mass": float(non_null_mass[selected]),
                    "ambiguity_entropy": float(entropy[selected]),
                    "abstain_probability": float(np.clip(1.0 - non_null_mass[selected], 0.0, 1.0)),
                    "relation_verify_evidence": float(verify[selected]),
                })
        rows.append(result)
    summary = {}
    trajectory_summary = {}
    for name in policy_names:
        summary[name] = _metrics(
            [row[name]["translation_m"] for row in rows],
            [row[name]["rotation_deg"] for row in rows],
            [row["oracle"]["translation_m"] for row in rows],
        )
        summary[name].update({
            "top3_0p5m_5deg_recall": float(np.mean(topn[name][3])),
            "top5_0p5m_5deg_recall": float(np.mean(topn[name][5])),
            "configuration_diverse_top3_0p5m_5deg_recall": float(np.mean(topn[name]["diverse3"])),
            "ranking_failure_fraction": float(np.mean([row[name]["ranking_failure"] for row in rows])),
            "proposal_coverage_failure_fraction": float(np.mean([not row["proposal_covered"] for row in rows])),
        })
        trajectory_summary[name] = {}
        for trajectory in sorted({row["trajectory"] for row in rows}):
            selected = [row for row in rows if row["trajectory"] == trajectory]
            trajectory_summary[name][trajectory] = _metrics(
                [row[name]["translation_m"] for row in selected],
                [row[name]["rotation_deg"] for row in selected],
                [row["oracle"]["translation_m"] for row in selected],
            )
    report = {
        "stage": "evaluate_goal_maplet_mode_relation_v2",
        "protocol": "frozen_top32_mass_conserving_fit_tree_exact_pair_verify_multimode",
        "query_count": len(rows), "candidate_pool": str(args.candidate_pool),
        "configuration_evidence_contract": contract,
        "decision_status": {
            "production_promoted": False,
            "untouched_test_opened": False,
            "continuous_refiner_opened": False,
            "relation_guided_proposals_opened": False,
            "g13_relation_v2_node_fit_equal_rank": "posthoc_research_diagnostic_only",
            "g16_mass_adaptive_endpoint": (
                "frozen_pool_research_candidate"
                if contract.get("mode_relation_inference")
                == "mass_adaptive_hierarchical_exact_sum_product_fit_tree_v3"
                else "not_present"
            ),
        },
        "summary": summary, "trajectory_summary": trajectory_summary,
        "risk_coverage": {name: _risk_coverage(rows, name) for name in policy_names},
        "endpoint_coverage_ladder": _aggregate_endpoint_coverage(rows),
        "endpoint_coverage_ladder_by_trajectory": {
            trajectory: _aggregate_endpoint_coverage([
                row for row in rows if row["trajectory"] == trajectory
            ])
            for trajectory in sorted({row["trajectory"] for row in rows})
        },
        "probability_semantics_audit": {
            "maximum_prior_mass_residual": float(max(
                max(row["ranking_diagnostics"][MODE]["mode_relation_evidence_v2"][
                    "relation_prior_mass_residual_max"
                ]) for row in payload["rows"]
            )),
            "complete_link_relation_nodes_mean": float(np.mean([
                row["relation_v2_query_diagnostics"].get("complete_link_relation_node_count", 0)
                for row in rows
            ])),
            "legacy_connected_relation_nodes_mean": float(np.mean([
                row["relation_v2_query_diagnostics"].get("legacy_connected_relation_node_count", 0)
                for row in rows
            ])),
            "chain_restored_relation_nodes_total": int(np.sum([
                row["relation_v2_query_diagnostics"].get("chain_restored_relation_node_count", 0)
                for row in rows
            ])),
            "inference_relation_nodes_mean": float(np.mean([
                row["relation_v2_query_diagnostics"].get("inference_relation_node_count", 0)
                for row in rows
            ])),
            "endpoint_state_count_mean": float(np.mean([
                row["relation_v2_query_diagnostics"].get("endpoint_state_count_mean", 0.0)
                for row in rows
            ])),
            "expanded_child_count_mean": float(np.mean([
                row["relation_v2_query_diagnostics"].get("expanded_child_count_mean", 0.0)
                for row in rows
            ])),
            "maxsum_non_null_query_fraction": float(np.mean([
                row["relation_v2_joint"]["endpoint_valid_mass"] > 0.0
                and any(value >= 0 for value in row["relation_v2_joint"][
                    "configuration_assignments_topk"
                ][0])
                for row in rows
            ])),
            "gt_pose_non_null_mass_median": float(np.median([
                row["probability_semantics"]["gt_pose_non_null_mass"] for row in rows
            ])),
            "oracle_candidate_non_null_mass_median": float(np.median([
                row["probability_semantics"]["oracle_candidate_non_null_mass"] for row in rows
            ])),
            "near_miss_phase_non_null_mass_median": _optional_median([
                row["probability_semantics"]["near_miss_phase_non_null_mass_median"]
                for row in rows
                if row["probability_semantics"]["near_miss_phase_non_null_mass_median"] is not None
            ]),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
