"""Apply a frozen query-set pairwise configuration ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.configuration_ranker import (
    BASE_CONFIGURATION_FEATURE_NAMES,
    EXACT_FEATURE_NAMES,
    EXACT_CONFIGURATION_FEATURE_NAMES,
    ConfigurationPairwiseRankerArtifact,
    candidate_runtime_features,
)
from feature_extract.vfm.localization_goal_maplet.configuration_evidence import (
    FEATURE_NAMES as CONFIGURATION_EVIDENCE_FEATURE_NAMES,
)


MODE = "actual_parent_actual_child"


def _summary(rows: list[dict], prefix: str) -> dict:
    t = np.asarray([row[f"{prefix}_translation_m"] for row in rows], dtype=np.float64)
    r = np.asarray([row[f"{prefix}_rotation_deg"] for row in rows], dtype=np.float64)
    oracle_t = np.asarray([row["oracle_translation_m"] for row in rows], dtype=np.float64)
    utility_regret = np.asarray([row[f"{prefix}_utility_regret"] for row in rows], dtype=np.float64)
    return {
        "translation_median_m": float(np.median(t)), "translation_p90_m": float(np.percentile(t, 90.0)),
        "rotation_median_deg": float(np.median(r)), "rotation_p90_deg": float(np.percentile(r, 90.0)),
        "top1_0p5m_5deg_fraction": float(np.mean((t <= 0.5) & (r <= 5.0))),
        "catastrophic_2m_or_10deg_fraction": float(np.mean((t > 2.0) | (r > 10.0))),
        "oracle_translation_median_m": float(np.median(oracle_t)),
        "median_selection_regret_m": float(np.median(t - oracle_t)),
        "p90_selection_regret_m": float(np.percentile(t - oracle_t, 90.0)),
        "median_normalized_utility_regret": float(np.median(utility_regret)),
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
        raise FileExistsError("refusing to overwrite pairwise ranking report")
    payload = json.loads(Path(args.candidate_pool).read_text())
    ranker = ConfigurationPairwiseRankerArtifact.load(Path(args.ranker))
    for key in ("physical_map_sha256", "canonical_field_sha256", "typed_graph_sha256", "field_feature_contract_sha256"):
        if ranker.metadata.get(key) and ranker.metadata.get(key) != payload.get(key):
            raise ValueError(f"ranker and candidate pool differ: {key}")
    names = tuple(ranker.metadata.get("feature_names", ()))
    include_exact = names in (EXACT_FEATURE_NAMES, EXACT_CONFIGURATION_FEATURE_NAMES)
    include_configuration = names in (BASE_CONFIGURATION_FEATURE_NAMES, EXACT_CONFIGURATION_FEATURE_NAMES)
    if include_configuration:
        contract = dict(payload.get("configuration_evidence_contract", {}))
        if tuple(contract.get("feature_names", ())) != CONFIGURATION_EVIDENCE_FEATURE_NAMES:
            raise ValueError("candidate pool configuration evidence contract differs")
    rows = []
    for row in payload["rows"]:
        feature = candidate_runtime_features(
            row, mode_name=MODE, include_exact=include_exact,
            include_configuration=include_configuration,
        )
        indices = np.arange(feature.shape[0], dtype=np.int64)
        if include_exact:
            keep = np.asarray(row["ranking_diagnostics"][MODE]["cascade_exact_evaluated"], dtype=bool)
            feature, indices = feature[keep], indices[keep]
        score = ranker.score_candidates(feature)
        selected = int(indices[int(np.argmax(score))])
        diagnostics = row["ranking_diagnostics"][MODE]
        proposal = int(np.argmax(np.asarray(diagnostics["proposal_scores"], dtype=np.float64)))
        cheap = int(np.argmax(np.asarray(diagnostics["identity_scores"], dtype=np.float64)))
        details = row["mode_details"][MODE]
        utility = np.asarray([item["translation_m"] / 0.5 + item["rotation_deg"] / 5.0 for item in details])
        oracle = int(np.argmin(utility))
        result = {"image_id": row["image_id"], "candidate_scores": score.tolist(), "evaluated_candidate_indices": indices.tolist()}
        for name, index in (("selected", selected), ("proposal", proposal), ("cheap_identity", cheap)):
            result.update({
                f"{name}_rank": index + 1,
                f"{name}_translation_m": float(details[index]["translation_m"]),
                f"{name}_rotation_deg": float(details[index]["rotation_deg"]),
                f"{name}_utility_regret": float(utility[index] - utility[oracle]),
            })
        result.update({
            "oracle_rank": oracle + 1, "oracle_translation_m": float(details[oracle]["translation_m"]),
            "oracle_rotation_deg": float(details[oracle]["rotation_deg"]),
        })
        rows.append(result)
    report = {
        "stage": "apply_goal_maplet_configuration_pairwise_ranker_v2",
        "query_count": len(rows), "candidate_pool": str(args.candidate_pool), "ranker": str(args.ranker),
        "ranker_metadata": dict(ranker.metadata),
        "summary": {name: _summary(rows, name) for name in ("selected", "proposal", "cheap_identity")},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
