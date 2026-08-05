"""Evaluate fixed-denominator and group-aware latent Goal-Maplet policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODE = "actual_parent_actual_child"

POLICIES = {
    "legacy_valid_factor_mass": ("configuration_legacy_valid_factor_mass_mean", 1.0),
    "fixed_unweighted_valid_mass": ("configuration_fixed_valid_factor_mass_unweighted_mean", 1.0),
    "fixed_decorrelated_valid_mass": ("configuration_fixed_valid_factor_mass_mean", 1.0),
    "latent_uncap_valid_mass": ("configuration_latent_uncap_valid_mass_mean", 1.0),
    "latent_cap_valid_mass": ("configuration_latent_cap_valid_mass_mean", 1.0),
    "latent_typed_consistency": ("configuration_latent_typed_consistency_mass_mean", 1.0),
}

LIKELIHOOD_POLICIES = {
    "pose_llr_fixed_mean": ("configuration_pose_llr_fixed_mean", 1.0),
    "pose_llr_fixed_median": ("configuration_pose_llr_fixed_median", 1.0),
    "pose_llr_log_bayes_mean": ("configuration_pose_llr_log_bayes_mean", 1.0),
    "soft_valid_probability": ("configuration_soft_valid_probability_mean", 1.0),
    "soft_cap_valid_probability": ("configuration_soft_cap_valid_probability_mean", 1.0),
}

FUSION_POLICIES = (
    "legacy_llr_median_equal_rank_fusion",
    "decorrelated_llr_mean_equal_rank_fusion",
)


def _rank_percentile(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(-value, kind="stable")
    output = np.ones(value.shape, dtype=np.float64)
    if value.size > 1:
        output[order] = np.linspace(1.0, 0.0, value.size)
    return output


def _metrics(translation: np.ndarray, rotation: np.ndarray, oracle: np.ndarray) -> dict:
    t = np.asarray(translation, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    o = np.asarray(oracle, dtype=np.float64)
    return {
        "query_count": int(t.size),
        "translation_median_m": float(np.median(t)),
        "translation_p90_m": float(np.percentile(t, 90.0)),
        "translation_p95_m": float(np.percentile(t, 95.0)),
        "rotation_median_deg": float(np.median(r)),
        "rotation_p90_deg": float(np.percentile(r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((t <= 0.5) & (r <= 5.0))),
        "top1_1m_10deg_fraction": float(np.mean((t <= 1.0) & (r <= 10.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (r > 10.0))),
        "median_selection_regret_m": float(np.median(t - o)),
        "p90_selection_regret_m": float(np.percentile(t - o, 90.0)),
    }


def _risk_coverage(rows: list[dict], policy: str) -> dict:
    ordered = sorted(rows, key=lambda row: (-float(row[policy]["confidence"]), row["image_id"]))
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(np.ceil(float(coverage) * len(ordered))))
        selected = ordered[:count]
        result[f"{coverage:.1f}"] = _metrics(
            np.asarray([row[policy]["translation_m"] for row in selected]),
            np.asarray([row[policy]["rotation_deg"] for row in selected]),
            np.asarray([row["oracle"]["translation_m"] for row in selected]),
        )
    return result


def _subset_metrics(rows: list[dict], policy: str, key: str, *, high: bool) -> dict:
    value = np.asarray([row[policy][key] for row in rows], dtype=np.float64)
    threshold = float(np.quantile(value, 2.0 / 3.0 if high else 1.0 / 3.0))
    selected = [
        row for row in rows
        if (float(row[policy][key]) >= threshold if high else float(row[policy][key]) <= threshold)
    ]
    return {
        "proxy_threshold": threshold,
        **_metrics(
            np.asarray([row[policy]["translation_m"] for row in selected]),
            np.asarray([row[policy]["rotation_deg"] for row in selected]),
            np.asarray([row["oracle"]["translation_m"] for row in selected]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite latent configuration audit")
    payload = json.loads(Path(args.candidate_pool).read_text())
    contract = dict(payload.get("configuration_evidence_contract", {}))
    evidence_version = str(contract.get("evidence_version"))
    if not (
        evidence_version in ("v3", "v4")
        and bool(contract.get("fixed_group_denominator"))
        and bool(contract.get("one_mode_per_group"))
    ):
        raise ValueError("candidate pool is not fixed-denominator latent evidence v3")

    rows = []
    policies = dict(POLICIES)
    if evidence_version == "v4":
        policies.update(LIKELIHOOD_POLICIES)
    evaluated_policy_names = list(policies)
    if evidence_version == "v4":
        evaluated_policy_names.extend(FUSION_POLICIES)
    topn_success = {name: {3: [], 5: []} for name in evaluated_policy_names}
    for row in payload["rows"]:
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        evidence = diagnostics[f"configuration_evidence_{evidence_version}"]
        translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
        rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
        utility = translation / 0.5 + rotation / 5.0
        oracle = int(np.argmin(utility))
        proposal_score = np.asarray(diagnostics["proposal_scores"], dtype=np.float64)
        policy_score = {
            "proposal": proposal_score,
            **{
                name: float(direction) * np.asarray(evidence[feature], dtype=np.float64)
                for name, (feature, direction) in policies.items()
            },
        }
        if evidence_version == "v4":
            policy_score["legacy_llr_median_equal_rank_fusion"] = (
                _rank_percentile(np.asarray(
                    evidence["configuration_legacy_valid_factor_mass_mean"], dtype=np.float64,
                ))
                + _rank_percentile(np.asarray(
                    evidence["configuration_pose_llr_fixed_median"], dtype=np.float64,
                ))
            )
            policy_score["decorrelated_llr_mean_equal_rank_fusion"] = (
                _rank_percentile(np.asarray(
                    evidence["configuration_fixed_valid_factor_mass_mean"], dtype=np.float64,
                ))
                + _rank_percentile(np.asarray(
                    evidence["configuration_pose_llr_fixed_mean"], dtype=np.float64,
                ))
            )
        result = {
            "image_id": row["image_id"],
            "oracle": {
                "rank": oracle + 1,
                "translation_m": float(translation[oracle]),
                "rotation_deg": float(rotation[oracle]),
            },
        }
        for name, score in policy_score.items():
            order = np.argsort(-score, kind="stable")
            selected = int(order[0])
            margin = float(score[order[0]] - score[order[1]]) if order.size > 1 else 0.0
            assigned = np.asarray(
                evidence["configuration_latent_cap_assigned_group_fraction"], dtype=np.float64
            )
            incompatible = np.asarray(
                evidence["configuration_latent_pose_incompatible_mass_mean"], dtype=np.float64
            )
            confidence = margin * max(float(assigned[selected]), 0.05) * max(
                1.0 - float(incompatible[selected]), 0.05
            )
            result[name] = {
                "rank": selected + 1,
                "translation_m": float(translation[selected]),
                "rotation_deg": float(rotation[selected]),
                "confidence": confidence,
                "score_margin": margin,
                "eligible_group_fraction": float(np.asarray(
                    evidence["configuration_fixed_eligible_group_fraction"]
                )[selected]),
                "assigned_group_fraction": float(assigned[selected]),
                "primitive_collision_fraction": float(np.asarray(
                    evidence["configuration_latent_uncap_primitive_collision_fraction"]
                )[selected]),
                "assignment_margin": float(np.asarray(
                    evidence["configuration_latent_assignment_margin_mean"]
                )[selected]),
            }
            if name in evaluated_policy_names:
                for topn in (3, 5):
                    keep = order[: min(topn, order.size)]
                    topn_success[name][topn].append(bool(np.any(
                        (translation[keep] <= 0.5) & (rotation[keep] <= 5.0)
                    )))
        rows.append(result)

    policy_names = ["proposal", *evaluated_policy_names]
    summary = {}
    for name in policy_names:
        summary[name] = _metrics(
            np.asarray([row[name]["translation_m"] for row in rows]),
            np.asarray([row[name]["rotation_deg"] for row in rows]),
            np.asarray([row["oracle"]["translation_m"] for row in rows]),
        )
        if name in topn_success:
            summary[name]["top3_0p5m_5deg_recall"] = float(np.mean(topn_success[name][3]))
            summary[name]["top5_0p5m_5deg_recall"] = float(np.mean(topn_success[name][5]))
    report = {
        "stage": f"evaluate_goal_maplet_latent_configuration_{evidence_version}",
        "protocol": "predefined_fixed_denominator_group_aware_likelihood_policies",
        "query_count": len(rows),
        "candidate_pool": str(args.candidate_pool),
        "configuration_evidence_contract": contract,
        "summary": summary,
        "risk_coverage": {name: _risk_coverage(rows, name) for name in evaluated_policy_names},
        "runtime_ambiguity_subsets": {
            name: {
                "high_primitive_collision_proxy": _subset_metrics(
                    rows, name, "primitive_collision_fraction", high=True,
                ),
                "low_assignment_margin_proxy": _subset_metrics(
                    rows, name, "assignment_margin", high=False,
                ),
                "low_eligible_coverage_proxy": _subset_metrics(
                    rows, name, "eligible_group_fraction", high=False,
                ),
            }
            for name in evaluated_policy_names
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
