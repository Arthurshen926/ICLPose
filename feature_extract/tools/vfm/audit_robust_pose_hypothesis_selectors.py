"""Audit immutable-likelihood hypothesis selectors with target-only metrics.

The selected hypothesis is determined only by fields already computed on the
fixed held-out candidate denominator. Ground-truth pose errors are read only
after selection and are never used to rank hypotheses within a query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SELECTOR_FIELDS: Mapping[str, str] = {
    "mean": "fixed_posterior_log_likelihood_mean",
    "median": "fixed_posterior_log_likelihood_median",
    "trimmed_mean_10": "fixed_posterior_log_likelihood_trimmed_mean_10",
    "worst_quartile_mean": (
        "fixed_posterior_log_likelihood_worst_quartile_mean"
    ),
    "lcb95": "fixed_posterior_log_likelihood_lcb95",
    "spatial_mom_2x2": "fixed_posterior_spatial_median_of_means_2x2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def select_hypothesis_by_immutable_statistic(
    query_row: Mapping[str, object],
    field_name: str,
) -> dict[str, object] | None:
    """Select by one immutable field, breaking exact ties by stable index."""

    audit = query_row.get("hypothesis_information_audit_with_TARGET_ONLY_errors")
    if not isinstance(audit, list):
        raise ValueError("query row lacks the hypothesis information audit")
    eligible: list[dict[str, object]] = []
    for raw in audit:
        if not isinstance(raw, dict) or not bool(
            raw.get("verified_for_final_selection", False)
        ):
            continue
        score = _finite_float(raw.get(field_name))
        translation = _finite_float(raw.get("translation_m_TARGET_ONLY"))
        rotation = _finite_float(raw.get("rotation_deg_TARGET_ONLY"))
        if score is None or translation is None or rotation is None:
            continue
        eligible.append(raw)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item[field_name]),
            -int(item["hypothesis_index"]),
        ),
    )


def _selected_query_row(
    query_row: Mapping[str, object],
    *,
    selector_name: str,
    field_name: str,
) -> dict[str, object] | None:
    selected = select_hypothesis_by_immutable_statistic(query_row, field_name)
    if selected is None:
        return None
    audit = query_row["hypothesis_information_audit_with_TARGET_ONLY_errors"]
    assert isinstance(audit, list)
    eligible = [
        item
        for item in audit
        if isinstance(item, dict)
        and bool(item.get("verified_for_final_selection", False))
        and _finite_float(item.get(field_name)) is not None
        and _finite_float(item.get("translation_m_TARGET_ONLY")) is not None
        and _finite_float(item.get("rotation_deg_TARGET_ONLY")) is not None
    ]
    selected_translation = float(selected["translation_m_TARGET_ONLY"])
    translation_rank = 1 + sum(
        float(item["translation_m_TARGET_ONLY"]) < selected_translation - 1e-12
        for item in eligible
    )
    ordered_scores = sorted(
        (float(item[field_name]) for item in eligible), reverse=True
    )
    score_gap = (
        None
        if len(ordered_scores) < 2
        else float(ordered_scores[0] - ordered_scores[1])
    )
    oracle = min(
        eligible,
        key=lambda item: (
            float(item["translation_m_TARGET_ONLY"]),
            float(item["rotation_deg_TARGET_ONLY"]),
        ),
    )
    return {
        "query_id": str(query_row["query_id"]),
        "selector": str(selector_name),
        "selector_field": str(field_name),
        "selected_hypothesis_index": int(selected["hypothesis_index"]),
        "selected_score": float(selected[field_name]),
        "top1_minus_top2_score_gap": score_gap,
        "translation_m_TARGET_ONLY": selected_translation,
        "rotation_deg_TARGET_ONLY": float(selected["rotation_deg_TARGET_ONLY"]),
        "translation_rank_TARGET_ONLY": int(translation_rank),
        "verified_hypothesis_count": int(len(eligible)),
        "oracle_translation_m_TARGET_ONLY": float(
            oracle["translation_m_TARGET_ONLY"]
        ),
        "oracle_rotation_deg_TARGET_ONLY": float(
            oracle["rotation_deg_TARGET_ONLY"]
        ),
        "selection_regret_m_TARGET_ONLY": float(
            selected_translation - float(oracle["translation_m_TARGET_ONLY"])
        ),
    }


def summarize_selected_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"query_count": 0}
    translation = np.asarray(
        [float(row["translation_m_TARGET_ONLY"]) for row in rows],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [float(row["rotation_deg_TARGET_ONLY"]) for row in rows],
        dtype=np.float64,
    )
    regret = np.asarray(
        [float(row["selection_regret_m_TARGET_ONLY"]) for row in rows],
        dtype=np.float64,
    )
    rank = np.asarray(
        [float(row["translation_rank_TARGET_ONLY"]) for row in rows],
        dtype=np.float64,
    )
    count = int(len(rows))
    return {
        "query_count": count,
        "median_translation_m_TARGET_ONLY": float(np.median(translation)),
        "p90_translation_m_TARGET_ONLY": float(np.percentile(translation, 90)),
        "median_rotation_deg_TARGET_ONLY": float(np.median(rotation)),
        "p90_rotation_deg_TARGET_ONLY": float(np.percentile(rotation, 90)),
        "recall_3cm_5deg_TARGET_ONLY": float(
            np.mean((translation <= 0.03) & (rotation <= 5.0))
        ),
        "recall_5cm_5deg_TARGET_ONLY": float(
            np.mean((translation <= 0.05) & (rotation <= 5.0))
        ),
        "recall_10cm_5deg_TARGET_ONLY": float(
            np.mean((translation <= 0.10) & (rotation <= 5.0))
        ),
        "recall_25cm_2deg_TARGET_ONLY": float(
            np.mean((translation <= 0.25) & (rotation <= 2.0))
        ),
        "catastrophic_gt1m_count_TARGET_ONLY": int(np.sum(translation > 1.0)),
        "median_selection_regret_m_TARGET_ONLY": float(np.median(regret)),
        "p90_selection_regret_m_TARGET_ONLY": float(np.percentile(regret, 90)),
        "median_selected_translation_rank_TARGET_ONLY": float(np.median(rank)),
    }


def _comparison_to_mean(
    selected_by_policy: Mapping[str, Sequence[Mapping[str, object]]],
    selector_name: str,
) -> dict[str, object]:
    mean_by_query = {
        str(row["query_id"]): row for row in selected_by_policy.get("mean", ())
    }
    rows = selected_by_policy.get(selector_name, ())
    deltas: list[float] = []
    agreement = 0
    wins = 0
    losses = 0
    for row in rows:
        baseline = mean_by_query.get(str(row["query_id"]))
        if baseline is None:
            continue
        delta = float(row["translation_m_TARGET_ONLY"]) - float(
            baseline["translation_m_TARGET_ONLY"]
        )
        deltas.append(delta)
        wins += int(delta < -1e-12)
        losses += int(delta > 1e-12)
        agreement += int(
            int(row["selected_hypothesis_index"])
            == int(baseline["selected_hypothesis_index"])
        )
    return {
        "paired_query_count": int(len(deltas)),
        "selection_agreement_with_mean_count": int(agreement),
        "win_count_vs_mean_TARGET_ONLY": int(wins),
        "loss_count_vs_mean_TARGET_ONLY": int(losses),
        "median_translation_delta_vs_mean_m_TARGET_ONLY": (
            None if not deltas else float(np.median(np.asarray(deltas)))
        ),
        "mean_translation_delta_vs_mean_m_TARGET_ONLY": (
            None if not deltas else float(np.mean(np.asarray(deltas)))
        ),
    }


def audit_pose_rows(
    payload: Mapping[str, object],
    *,
    splits: Sequence[str],
    evaluation_label: str | None = None,
    variant: str = "verified",
) -> dict[str, object]:
    output: dict[str, object] = {}
    for split in splits:
        split_payload = payload.get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(f"pose rows lack split {split}")
        labels = list(split_payload)
        label = str(evaluation_label) if evaluation_label else (
            labels[0] if len(labels) == 1 else ""
        )
        if not label or label not in split_payload:
            raise ValueError(
                f"split {split} requires one explicit evaluation label; found {labels}"
            )
        variants = split_payload[label]
        if not isinstance(variants, dict) or variant not in variants:
            raise ValueError(f"split {split} lacks variant {variant}")
        query_rows = variants[variant]
        if not isinstance(query_rows, list):
            raise ValueError("pose-row variant must be a list")

        selected_by_policy: dict[str, list[dict[str, object]]] = {}
        for selector_name, field_name in SELECTOR_FIELDS.items():
            selected_by_policy[selector_name] = [
                selected
                for row in query_rows
                if isinstance(row, dict)
                and (
                    selected := _selected_query_row(
                        row,
                        selector_name=selector_name,
                        field_name=field_name,
                    )
                )
                is not None
            ]
        output[split] = {
            "evaluation_label": label,
            "variant": str(variant),
            "runtime_promotion_gate_applied": False,
            "selector_uses_target_pose": False,
            "target_pose_used_for_reporting_only": True,
            "selectors": {
                name: {
                    "field": SELECTOR_FIELDS[name],
                    "summary": summarize_selected_rows(rows),
                    "comparison_to_mean": _comparison_to_mean(
                        selected_by_policy, name
                    ),
                    "query_rows_TARGET_ONLY": rows,
                }
                for name, rows in selected_by_policy.items()
            },
        }
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_rows", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--splits", default="validation,late_development"
    )
    parser.add_argument("--evaluation_label", default="")
    parser.add_argument("--variant", default="verified")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.pose_rows)
    payload = json.loads(input_path.read_text())
    output = {
        "format": "robust_pose_hypothesis_selector_audit_v1",
        "input_pose_rows": str(input_path),
        "input_pose_rows_sha256": _sha256(input_path),
        "selection_contract": (
            "argmax_one_immutable_per_group_likelihood_statistic_then_"
            "target_only_pose_reporting"
        ),
        "audit": audit_pose_rows(
            payload,
            splits=tuple(
                item.strip()
                for item in str(args.splits).split(",")
                if item.strip()
            ),
            evaluation_label=(
                None if not str(args.evaluation_label) else str(args.evaluation_label)
            ),
            variant=str(args.variant),
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
