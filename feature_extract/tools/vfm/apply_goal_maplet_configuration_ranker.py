"""Apply a frozen runtime-only configuration ranker to a candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.configuration_ranker import (
    EXACT_FEATURE_NAMES,
    ConfigurationRankerArtifact,
    candidate_runtime_features,
)


MODE = "actual_parent_actual_child"


def _summary(rows: list[dict], prefix: str) -> dict[str, float]:
    translation = np.asarray([row[f"{prefix}_translation_m"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row[f"{prefix}_rotation_deg"] for row in rows], dtype=np.float64)
    oracle_translation = np.asarray([row["oracle_translation_m"] for row in rows], dtype=np.float64)
    return {
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((translation > 2.0) | (rotation > 10.0))),
        "oracle_translation_median_m": float(np.median(oracle_translation)),
        "oracle_translation_p90_m": float(np.percentile(oracle_translation, 90.0)),
        "median_selection_regret_m": float(np.median(translation - oracle_translation)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite ranked candidate report")
    payload = json.loads(Path(args.candidate_pool).read_text())
    ranker = ConfigurationRankerArtifact.load(Path(args.ranker))
    for key in ("physical_map_sha256", "canonical_field_sha256", "typed_graph_sha256", "field_feature_contract_sha256"):
        expected = ranker.metadata.get(key)
        if expected and payload.get(key) != expected:
            raise ValueError(f"ranker and candidate pool differ: {key}")
    rows = []
    include_exact = tuple(ranker.metadata.get("feature_names", ())) == EXACT_FEATURE_NAMES
    for row in payload["rows"]:
        feature = candidate_runtime_features(row, mode_name=MODE, include_exact=include_exact)
        if feature.shape[0] == 0:
            raise ValueError(f"query has no configuration candidates: {row['image_id']}")
        candidate_indices = np.arange(feature.shape[0], dtype=np.int64)
        if include_exact:
            keep = np.asarray(
                row["ranking_diagnostics"][MODE]["cascade_exact_evaluated"], dtype=bool
            )
            candidate_indices = candidate_indices[keep]
            feature = feature[keep]
        probability = ranker.predict_probability(feature)
        selected = int(candidate_indices[int(np.argmax(probability))])
        details = row["mode_details"][MODE]
        diagnostics = row["ranking_diagnostics"][MODE]
        proposal = int(np.argmax(np.asarray(diagnostics["proposal_scores"], dtype=np.float64)))
        cheap = int(np.argmax(np.asarray(diagnostics["identity_scores"], dtype=np.float64)))
        utility = np.asarray([d["translation_m"] / 0.5 + d["rotation_deg"] / 5.0 for d in details])
        oracle = int(np.argmin(utility))
        rows.append({
            "image_id": row["image_id"],
            "selected_rank": selected + 1,
            "selected_probability": float(probability[selected]),
            "selected_translation_m": float(details[selected]["translation_m"]),
            "selected_rotation_deg": float(details[selected]["rotation_deg"]),
            "selected_0p5m_5deg": bool(details[selected]["translation_m"] <= 0.5 and details[selected]["rotation_deg"] <= 5.0),
            "proposal_rank": proposal + 1,
            "proposal_translation_m": float(details[proposal]["translation_m"]),
            "proposal_rotation_deg": float(details[proposal]["rotation_deg"]),
            "cheap_identity_rank": cheap + 1,
            "cheap_identity_translation_m": float(details[cheap]["translation_m"]),
            "cheap_identity_rotation_deg": float(details[cheap]["rotation_deg"]),
            "oracle_rank": oracle + 1,
            "oracle_translation_m": float(details[oracle]["translation_m"]),
            "oracle_rotation_deg": float(details[oracle]["rotation_deg"]),
            "mode_probabilities": probability.tolist(),
            "evaluated_candidate_indices": candidate_indices.tolist(),
        })
    result = {
        "stage": "apply_goal_maplet_configuration_ranker",
        "query_count": len(rows),
        "candidate_pool": str(args.candidate_pool),
        "ranker": str(args.ranker),
        "ranker_metadata": dict(ranker.metadata),
        "summary": {
            "learned": _summary(rows, "selected"),
            "proposal": _summary(rows, "proposal"),
            "cheap_identity": _summary(rows, "cheap_identity"),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
