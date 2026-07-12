"""Fit an abstaining margin for fixed-posterior marginal pose evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.pairwise_pose_promotion import _is_beneficial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginal_evidence_rows", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_precision", type=float, default=0.8)
    args = parser.parse_args()
    evidence_rows = json.loads(Path(args.marginal_evidence_rows).read_text())
    evidence_by_query: dict[str, dict[str, float]] = {}
    for row in evidence_rows:
        evidence_by_query.setdefault(str(row["query_id"]), {})[str(row["strategy"])] = float(
            row["marginal_log_likelihood_per_token"]
        )
    pose_by_strategy = {
        strategy: {
            str(row["query_id"]): row
            for row in json.loads((Path(args.pose_dir) / f"{strategy}.json").read_text())
        }
        for strategy in (
            "frozen_L97_replay",
            "candidate_pgeometry_argmax_DIAGNOSTIC_ONLY",
            "candidate_prior_assignment",
        )
    }
    records: list[tuple[float, bool, str, str]] = []
    for query_id, scores in evidence_by_query.items():
        baseline_score = scores["frozen_L97_replay"]
        optional_strategy = max(
            (strategy for strategy in scores if strategy != "frozen_L97_replay"),
            key=lambda strategy: (scores[strategy], strategy),
        )
        beneficial = _is_beneficial(
            pose_by_strategy["frozen_L97_replay"][query_id],
            pose_by_strategy[optional_strategy][query_id],
        )
        records.append(
            (
                float(scores[optional_strategy] - baseline_score),
                bool(beneficial),
                query_id,
                optional_strategy,
            )
        )
    deltas = np.asarray([record[0] for record in records], dtype=np.float64)
    candidate_margins = np.unique(
        np.concatenate(
            [np.asarray([0.0]), np.quantile(deltas, np.linspace(0.0, 1.0, 201))]
        )
    )
    best = None
    for margin in candidate_margins.tolist():
        selected = [record for record in records if record[0] >= float(margin)]
        beneficial_count = sum(record[1] for record in selected)
        precision = None if not selected else float(beneficial_count / len(selected))
        if precision is None or precision < float(args.target_precision):
            continue
        key = (beneficial_count, -len(selected), -float(margin))
        if best is None or key > best[0]:
            best = (
                key,
                float(margin),
                {
                    "promotion_count": int(len(selected)),
                    "beneficial_count_TARGET_ONLY": int(beneficial_count),
                    "promotion_precision_TARGET_ONLY": precision,
                },
            )
    if best is None:
        margin = float("inf")
        metrics = {
            "promotion_count": 0,
            "beneficial_count_TARGET_ONLY": 0,
            "promotion_precision_TARGET_ONLY": None,
        }
    else:
        _key, margin, metrics = best
    report = {
        "stage": "marginal_pose_promotion_margin_fit",
        "promotion_margin": margin,
        "target_precision": float(args.target_precision),
        "train_metrics": metrics,
        "protocol": {
            "candidate_posterior_fixed": True,
            "pose_identity_reselection": False,
            "GT_pose_errors_target_only": True,
        },
        "inputs": {
            "marginal_evidence_rows_sha256": file_sha256_short(Path(args.marginal_evidence_rows)),
            "pose_summary_sha256": file_sha256_short(Path(args.pose_dir) / "summary.json"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
