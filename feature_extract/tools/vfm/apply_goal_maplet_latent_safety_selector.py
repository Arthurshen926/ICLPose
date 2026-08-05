"""Apply a frozen Goal-Maplet safety residual to a disjoint pose pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from feature_extract.vfm.localization_goal_maplet.latent_selector import (
    MODE,
    LatentSafetySelectorArtifact,
    latent_selector_features,
)


def _summary(rows: list[dict], policy: str) -> dict:
    translation = np.asarray([row[policy]["translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row[policy]["rotation_deg"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle"]["translation_m"] for row in rows], dtype=np.float64)
    return {
        "query_count": len(rows),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "translation_p95_m": float(np.percentile(translation, 95.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "top1_1m_10deg_fraction": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean(
            (translation > 2.0) | (rotation > 10.0)
        )),
        "median_selection_regret_m": float(np.median(translation - oracle)),
        "p90_selection_regret_m": float(np.percentile(translation - oracle, 90.0)),
    }


def _risk_coverage(rows: list[dict], policy: str) -> dict:
    ordered = sorted(rows, key=lambda row: (-row[policy]["confidence"], row["image_id"]))
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(np.ceil(coverage * len(ordered))))
        result[f"{coverage:.1f}"] = _summary(ordered[:count], policy)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite latent safety selection")
    payload = json.loads(Path(args.candidate_pool).read_text())
    artifact = LatentSafetySelectorArtifact.load(Path(args.selector))
    for key in (
        "physical_map_sha256", "canonical_field_sha256", "typed_graph_sha256",
        "field_feature_contract_sha256",
    ):
        if artifact.metadata.get(key) != payload.get(key):
            raise ValueError(f"latent safety selector lineage differs: {key}")
    base_feature = str(artifact.metadata["base_feature"])
    rows = []
    basin_labels, catastrophic_labels = [], []
    basin_probabilities, catastrophic_probabilities, head_weights = [], [], []
    for row in payload["rows"]:
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        evidence = diagnostics["configuration_evidence_v3"]
        base = np.asarray(evidence[base_feature], dtype=np.float64)
        features = latent_selector_features(row)
        residual, basin, catastrophic = artifact.score_candidates(
            features, base, safety_veto=False,
        )
        safety, _, _ = artifact.score_candidates(features, base, safety_veto=True)
        proposal = np.asarray(diagnostics["proposal_scores"], dtype=np.float64)
        translation = np.asarray([item["translation_m"] for item in details], dtype=np.float64)
        rotation = np.asarray([item["rotation_deg"] for item in details], dtype=np.float64)
        utility = translation / 0.5 + rotation / 5.0
        oracle = int(np.argmin(utility))
        basin_labels.extend(((translation <= 0.5) & (rotation <= 5.0)).tolist())
        catastrophic_labels.extend(((translation > 2.0) | (rotation > 10.0)).tolist())
        basin_probabilities.extend(basin.tolist())
        catastrophic_probabilities.extend(catastrophic.tolist())
        head_weights.extend([1.0 / max(translation.size, 1)] * translation.size)
        current = {
            "image_id": row["image_id"],
            "oracle": {
                "translation_m": float(translation[oracle]),
                "rotation_deg": float(rotation[oracle]),
            },
        }
        for name, score in (
            ("proposal", proposal), ("base", base),
            ("residual", residual), ("safety_residual", safety),
        ):
            order = np.argsort(-score, kind="stable")
            selected = int(order[0])
            score_margin = float(score[order[0]] - score[order[1]]) if order.size > 1 else 0.0
            current[name] = {
                "translation_m": float(translation[selected]),
                "rotation_deg": float(rotation[selected]),
                "score_margin": score_margin,
                "basin_probability": float(basin[selected]),
                "catastrophic_probability": float(catastrophic[selected]),
                "confidence": float(
                    max(1.0 - catastrophic[selected], 0.0)
                    * max(basin[selected], 0.05)
                    * max(score_margin, 1e-6)
                ),
            }
        rows.append(current)
    policies = ("proposal", "base", "residual", "safety_residual")
    report = {
        "stage": "apply_goal_maplet_latent_safety_selector_v1",
        "query_count": len(rows),
        "candidate_pool": str(args.candidate_pool),
        "selector": str(args.selector),
        "selector_metadata": dict(artifact.metadata),
        "head_diagnostics": {
            "basin_auprc": float(average_precision_score(
                basin_labels, basin_probabilities, sample_weight=head_weights,
            )),
            "catastrophic_auprc": float(average_precision_score(
                catastrophic_labels, catastrophic_probabilities, sample_weight=head_weights,
            )),
            "basin_prevalence": float(np.average(basin_labels, weights=head_weights)),
            "catastrophic_prevalence": float(np.average(
                catastrophic_labels, weights=head_weights,
            )),
        },
        "summary": {name: _summary(rows, name) for name in policies},
        "risk_coverage": {
            name: _risk_coverage(rows, name) for name in ("base", "residual", "safety_residual")
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
