"""Merge disjoint Goal-Maplet pose-mode shards with lineage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged pose report")
    shards = [json.loads(Path(path).read_text()) for path in args.inputs]
    for key in (
        "stage", "physical_map_sha256", "canonical_field_sha256",
        "validity_calibration_sha256", "proposal_method", "maximum_modes",
        "typed_graph_sha256", "render_identity_rerank", "identity_render_mode",
        "field_feature_contract_sha256", "proposal_seed_policy",
        "cascade_contract",
        "configuration_evidence_contract",
    ):
        values = {json.dumps(item.get(key), sort_keys=True) for item in shards}
        if len(values) != 1:
            raise ValueError(f"pose shard lineage/config differs: {key}")
    rows = [row for shard in shards for row in shard.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("pose shards contain duplicate queries")
    rows.sort(key=lambda row: str(row["image_id"]))
    mode_names = sorted({name for row in rows for name in row.get("modes", {})})
    summary = {}
    for mode in mode_names:
        selected = [row["modes"][mode] for row in rows if mode in row.get("modes", {})]
        metrics = sorted({key for item in selected for key in item})
        summary[mode] = {}
        for metric in metrics:
            values = [item.get(metric) for item in selected if item.get(metric) is not None]
            if not values:
                summary[mode][metric] = None
            elif isinstance(values[0], bool):
                summary[mode][metric] = float(np.mean(values))
            elif metric == "mode_count":
                summary[mode][metric] = float(np.mean(values))
            else:
                array = np.asarray(values, dtype=np.float64)
                summary[mode][metric] = {
                    "median": float(np.median(array)),
                    "p90": float(np.percentile(array, 90.0)),
                    "p95": float(np.percentile(array, 95.0)),
                }
    result = {
        **{key: shards[0].get(key) for key in (
            "stage", "physical_map_sha256", "canonical_field_sha256",
            "validity_calibration_sha256", "proposal_method", "maximum_modes",
            "typed_graph_sha256", "render_identity_rerank", "identity_render_mode",
            "field_feature_contract_sha256", "proposal_seed_policy",
            "cascade_contract", "detector_radio_refine_topn", "alike_detector_only",
            "configuration_evidence_contract",
        )},
        "query_count": len(rows),
        "shard_count": len(shards),
        "source_shards": [str(value) for value in args.inputs],
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
