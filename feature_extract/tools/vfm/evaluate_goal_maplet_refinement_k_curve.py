"""Derive frozen refinement success--compute curves from a single Top-Kmax run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _parse_k(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise ValueError("refinement K values must be positive")
    return result


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [float("nan"), float("nan")]
    p = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count**2)) / denominator
    return [float(center - radius), float(center + radius)]


def _prefix_selection(
    row: dict[str, object],
    *,
    refine_k: int,
    validation_enabled: bool,
    require_cross_splat_winner_consistency: bool,
) -> tuple[dict[str, object], str, int]:
    refinements = list(row.get("candidate_refinements", []))
    if not refinements:
        return row, "gate_passthrough", 0
    eligible = [
        value
        for value in refinements
        if int(value["common_initial_rank"]) <= int(refine_k)
        or int(value.get("union_candidate_index", -1)) == 0
    ]
    baseline = [
        value
        for value in eligible
        if int(value.get("union_candidate_index", -1)) == 0
    ]
    if len(baseline) != 1:
        raise ValueError("prefix has no unique deployed baseline candidate")
    baseline_value = baseline[0]
    selected = max(
        eligible,
        key=lambda value: (
            float(value["final_score"]),
            -int(value["common_initial_rank"]),
        ),
    )
    decision = "maximum_primary_discretization_score"
    if validation_enabled:
        if any("validation_score" not in value for value in eligible):
            raise ValueError("validation selection requires every validation score")
        primary = max(
            eligible,
            key=lambda value: (
                float(value["final_score"]),
                -int(value["common_initial_rank"]),
            ),
        )
        validation = max(
            eligible,
            key=lambda value: (
                float(value["validation_score"]),
                -int(value["common_initial_rank"]),
            ),
        )
        threshold = row.get("baseline_null_threshold")
        baseline_is_null = (
            threshold is not None
            and float(baseline_value["final_score"]) < float(threshold)
        )
        if threshold is not None and not baseline_is_null:
            selected = baseline_value
            decision = "baseline_explained_query_control_limit"
        elif (
            require_cross_splat_winner_consistency
            and int(primary["union_candidate_index"])
            != int(validation["union_candidate_index"])
        ):
            selected = baseline_value
            decision = "cross_discretization_disagreement_baseline_fallback"
        else:
            selected = primary
            decision = (
                "cross_discretization_consistent_expert_override"
                if int(primary["union_candidate_index"])
                == int(validation["union_candidate_index"])
                and int(primary["union_candidate_index"]) != 0
                else "cross_discretization_consistent_baseline"
            )
    return selected, decision, len(eligible)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    translation = np.asarray([row["translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row["rotation_deg"] for row in rows], dtype=np.float64)
    strict = (translation <= 0.5) & (rotation <= 5.0)
    loose = (translation <= 1.0) & (rotation <= 10.0)
    catastrophic = (translation > 2.0) | (rotation > 20.0)
    count = len(rows)
    return {
        "query_count": count,
        "strict_count": int(np.sum(strict)),
        "strict_rate": float(np.mean(strict)),
        "strict_wilson95": _wilson(int(np.sum(strict)), count),
        "loose_count": int(np.sum(loose)),
        "loose_rate": float(np.mean(loose)),
        "loose_wilson95": _wilson(int(np.sum(loose)), count),
        "catastrophic_count": int(np.sum(catastrophic)),
        "catastrophic_rate": float(np.mean(catastrophic)),
        "catastrophic_wilson95": _wilson(int(np.sum(catastrophic)), count),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.quantile(translation, 0.9)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.quantile(rotation, 0.9)),
        "mean_refined_candidate_count": float(
            np.mean([row["refined_candidate_count"] for row in rows])
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement_reports", required=True, nargs="+")
    parser.add_argument("--k_values", default="1,2,4,8,16,32")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite K curve: {output}")
    paths = [Path(value) for value in args.refinement_reports]
    reports = [json.loads(path.read_text()) for path in paths]
    for report in reports:
        if str(report.get("refinement_candidate_policy", "score_topk")) != "score_topk":
            raise ValueError("prefix K curve requires a score_topk Kmax run")
        if float(report.get("refinement_score_margin", -1.0)) >= 0.0:
            raise ValueError("prefix K curve requires disabled adaptive score margin")
    k_values = _parse_k(str(args.k_values))
    max_available = min(int(report["refine_topk"]) for report in reports)
    if max(k_values) > max_available:
        raise ValueError("requested K exceeds source Kmax")
    curve = []
    for refine_k in k_values:
        selected_rows = []
        for report in reports:
            validation_enabled = int(report.get("validation_splat_radius_tokens", -1)) >= 0
            for row in report.get("rows", []):
                selected, decision, refined_count = _prefix_selection(
                    row,
                    refine_k=refine_k,
                    validation_enabled=validation_enabled,
                    require_cross_splat_winner_consistency=bool(
                        report.get("require_cross_splat_winner_consistency", False)
                    ),
                )
                selected_rows.append(
                    {
                        "image_id": str(row["image_id"]),
                        "translation_m": float(
                            selected.get("final_translation_m", row["final_translation_m"])
                        ),
                        "rotation_deg": float(
                            selected.get("final_rotation_deg", row["final_rotation_deg"])
                        ),
                        "selected_common_initial_rank": int(
                            selected.get("common_initial_rank", 1)
                        ),
                        "selected_union_candidate_index": int(
                            selected.get("union_candidate_index", 0)
                        ),
                        "selection_decision": decision,
                        "refined_candidate_count": refined_count,
                    }
                )
        curve.append(
            {
                "refine_k": int(refine_k),
                "summary": _aggregate(selected_rows),
                "rows": selected_rows,
            }
        )
    payload = {
        "artifact_type": "goal_maplet_refinement_success_compute_curve_v1",
        "source_reports": [str(path) for path in paths],
        "source_report_sha256": [file_sha256(path) for path in paths],
        "prefix_replay_uses_ground_truth_for_selection": False,
        "k_values": k_values,
        "curve": curve,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "curve"}, indent=2))


if __name__ == "__main__":
    main()
