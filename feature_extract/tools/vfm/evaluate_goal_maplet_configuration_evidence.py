"""Audit predefined configuration-factor policies without fitting on Dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODE = "actual_parent_actual_child"


POLICIES = {
    "maximum_valid_factor_mass": ("configuration_valid_factor_mass_mean", 1.0),
    "maximum_valid_factor_log_mass": ("configuration_valid_factor_log_mean", 1.0),
    "minimum_pose_incompatible_mass": ("configuration_pose_incompatible_mass_mean", -1.0),
    "minimum_reprojection_p90": ("configuration_normalized_reprojection_p90", -1.0),
    "maximum_assigned_group_fraction": ("configuration_assigned_group_fraction", 1.0),
}


def _summary(rows: list[dict], policy: str) -> dict:
    translation = np.asarray([row[policy]["translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row[policy]["rotation_deg"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle"]["translation_m"] for row in rows], dtype=np.float64)
    utility_regret = np.asarray([row[policy]["utility_regret"] for row in rows], dtype=np.float64)
    return {
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((translation > 2.0) | (rotation > 10.0))),
        "median_selection_regret_m": float(np.median(translation - oracle)),
        "p90_selection_regret_m": float(np.percentile(translation - oracle, 90.0)),
        "median_normalized_utility_regret": float(np.median(utility_regret)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite configuration evidence audit")
    payload = json.loads(Path(args.candidate_pool).read_text())
    rows = []
    pair_concordance = {name: [] for name in POLICIES}
    for row in payload["rows"]:
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        evidence = diagnostics["configuration_evidence_v2"]
        utility = np.asarray([
            item["translation_m"] / 0.5 + item["rotation_deg"] / 5.0 for item in details
        ], dtype=np.float64)
        oracle = int(np.argmin(utility))
        proposal = int(np.argmax(np.asarray(diagnostics["proposal_scores"], dtype=np.float64)))
        result = {
            "image_id": row["image_id"],
            "oracle": {"rank": oracle + 1, **{
                key: float(details[oracle][key]) for key in ("translation_m", "rotation_deg")
            }},
        }
        for name, index in (("proposal", proposal),):
            result[name] = {
                "rank": index + 1, "translation_m": float(details[index]["translation_m"]),
                "rotation_deg": float(details[index]["rotation_deg"]),
                "utility_regret": float(utility[index] - utility[oracle]),
            }
        left, right = np.triu_indices(len(details), 1)
        utility_preference = utility[left] < utility[right]
        for name, (feature_name, direction) in POLICIES.items():
            score = float(direction) * np.asarray(evidence[feature_name], dtype=np.float64)
            index = int(np.argmax(score))
            result[name] = {
                "rank": index + 1, "translation_m": float(details[index]["translation_m"]),
                "rotation_deg": float(details[index]["rotation_deg"]),
                "utility_regret": float(utility[index] - utility[oracle]),
            }
            if left.size:
                predicted = score[left] > score[right]
                non_tie = score[left] != score[right]
                if np.any(non_tie):
                    pair_concordance[name].append(float(np.mean(predicted[non_tie] == utility_preference[non_tie])))
        rows.append(result)
    policies = ["proposal", *POLICIES]
    report = {
        "stage": "evaluate_goal_maplet_configuration_evidence_v2",
        "protocol": "predefined_unfitted_factor_policies",
        "query_count": len(rows), "candidate_pool": str(args.candidate_pool),
        "configuration_evidence_contract": payload.get("configuration_evidence_contract"),
        "summary": {name: _summary(rows, name) for name in policies},
        "within_query_pair_concordance": {
            name: float(np.mean(values)) if values else None for name, values in pair_concordance.items()
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
