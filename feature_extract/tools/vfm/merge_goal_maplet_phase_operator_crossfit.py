"""Merge disjoint feature-cross-fit phase-operator fold evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_goal_maplet_phase_operator_crossfit import (
    _candidate_coverage_summary,
    _finite_distribution,
    _summary,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _operator_decision(folds: list[dict[str, object]]) -> dict[str, object]:
    def noninferior(candidate: str, baseline: str) -> bool:
        for fold in folds:
            metrics = fold["metrics"]
            selected = metrics[candidate]
            reference = metrics[baseline]
            if (
                float(selected["strict_0.5m_5deg"])
                < float(reference["strict_0.5m_5deg"])
                or float(selected["within_1m_10deg"])
                < float(reference["within_1m_10deg"])
                or float(selected["catastrophic_rate"])
                > float(reference["catastrophic_rate"])
            ):
                return False
        return True

    directional = noninferior("directional", "jacobian")
    jacobian = noninferior("jacobian", "directional")
    if directional and not jacobian:
        selected = "directional"
        reason = "directional is foldwise non-inferior on strict, 1m and catastrophe"
    elif jacobian and not directional:
        selected = "jacobian"
        reason = "jacobian is foldwise non-inferior on strict, 1m and catastrophe"
    elif directional and jacobian:
        selected = "directional"
        reason = "success/tail gates tie; retain the simpler incumbent"
    else:
        selected = "directional"
        reason = (
            "no consistent winner across folds; retain the simpler parameter-free "
            "directional baseline rather than tune a mixture"
        )
    return {
        "selected_phase_operator": selected,
        "directional_foldwise_noninferior": directional,
        "jacobian_foldwise_noninferior": jacobian,
        "selection_reason": reason,
        "phase_combination_training_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged cross-fit evaluation")
    paths = [Path(value) for value in args.inputs]
    folds = [json.loads(path.read_text()) for path in paths]
    if any(
        fold.get("stage")
        != "goal_maplet_phase_operator_outer_feature_crossfit_evaluation"
        for fold in folds
    ):
        raise ValueError("input is not a phase-operator cross-fit fold")
    if any(not bool(fold.get("feature_pipeline_outer_crossfit")) for fold in folds):
        raise ValueError("input fold is not feature-pipeline cross-fitted")
    heldout = [
        trajectory
        for fold in folds
        for trajectory in fold.get("heldout_trajectories", ())
    ]
    if len(heldout) != len(set(heldout)):
        raise ValueError("cross-fit held trajectories overlap")
    query_results = [row for fold in folds for row in fold.get("query_results", ())]
    image_ids = [str(row["image_id"]) for row in query_results]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("cross-fit query rows overlap")
    method_rows = {name: [] for name in ("directional", "jacobian")}
    per_trajectory = defaultdict(lambda: {name: [] for name in method_rows})
    complement = {
        "both_right": 0,
        "directional_right_jacobian_wrong": 0,
        "jacobian_right_directional_wrong": 0,
        "both_wrong": 0,
        "same_top1_pose": 0,
        "query_count": len(query_results),
    }
    kendall = []
    pearson = []
    for row in query_results:
        values = {name: dict(row[name]) for name in method_rows}
        for name, item in values.items():
            method_rows[name].append(item)
            per_trajectory[str(row["trajectory_id"])][name].append(item)
        left, right = bool(values["directional"]["strict"]), bool(
            values["jacobian"]["strict"]
        )
        if left and right:
            complement["both_right"] += 1
        elif left:
            complement["directional_right_jacobian_wrong"] += 1
        elif right:
            complement["jacobian_right_directional_wrong"] += 1
        else:
            complement["both_wrong"] += 1
        complement["same_top1_pose"] += int(bool(row["same_top1_pose"]))
        if row.get("operator_score_kendall") is not None:
            kendall.append(float(row["operator_score_kendall"]))
        if row.get("operator_score_pearson") is not None:
            pearson.append(float(row["operator_score_pearson"]))
    pairwise = {}
    bands = {}
    for name in method_rows:
        pairwise[name] = {"wins": 0, "ties": 0, "total": 0}
        band_names = sorted(
            folds[0]["oracle_vs_negative_distance_band_concordance"][name]
        )
        bands[name] = {
            band: {"wins": 0, "ties": 0, "total": 0} for band in band_names
        }
        for fold in folds:
            counts = fold["pose_quality_pairwise_concordance"][name]
            for key in pairwise[name]:
                pairwise[name][key] += int(counts[key])
            for band in band_names:
                counts = fold["oracle_vs_negative_distance_band_concordance"][name][band]
                for key in bands[name][band]:
                    bands[name][band][key] += int(counts[key])
        total = pairwise[name]["total"]
        pairwise[name]["concordance"] = (
            (pairwise[name]["wins"] + 0.5 * pairwise[name]["ties"]) / total
            if total else None
        )
        for counts in bands[name].values():
            total = counts["total"]
            counts["concordance"] = (
                (counts["wins"] + 0.5 * counts["ties"]) / total if total else None
            )
    exclusive = complement["directional_right_jacobian_wrong"] + complement[
        "jacobian_right_directional_wrong"
    ]
    result = {
        "stage": "goal_maplet_phase_operator_outer_feature_crossfit_merged",
        "fold_count": len(folds),
        "heldout_trajectories": sorted(heldout),
        "query_count": len(query_results),
        "geometry_protocols": sorted({str(fold["geometry_protocol"]) for fold in folds}),
        "geometry_was_outer_crossfit": all(
            bool(fold.get("geometry_was_outer_crossfit")) for fold in folds
        ),
        "feature_pipeline_outer_crossfit": True,
        "ranking_and_null_separated": True,
        "null_or_abstention_evaluated": False,
        "operator_parameters_refit": False,
        "production_promotion_allowed": False,
        "metrics": {name: _summary(rows) for name, rows in method_rows.items()},
        "per_trajectory": {
            trajectory: {name: _summary(rows) for name, rows in methods.items()}
            for trajectory, methods in sorted(per_trajectory.items())
        },
        "top1_complementarity": complement,
        "exclusive_success_fraction": exclusive / max(len(query_results), 1),
        "operator_score_correlation_per_query": {
            "kendall": _finite_distribution(kendall),
            "pearson": _finite_distribution(pearson),
        },
        "pose_quality_pairwise_concordance": pairwise,
        "oracle_vs_negative_distance_band_concordance": bands,
        "candidate_coverage_and_ranking_headroom": _candidate_coverage_summary(
            query_results
        ),
        "fold_inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "query_results": sorted(query_results, key=lambda row: str(row["image_id"])),
        "decision": _operator_decision(folds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": result["query_count"],
        "metrics": result["metrics"],
        "top1_complementarity": result["top1_complementarity"],
        "decision": result["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
