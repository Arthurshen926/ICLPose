"""Merge disjoint Goal-Maplet PFIR shards with lineage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROW_METRICS = (
    "support_count",
    "support_effective_count",
    "weighted_recall_at_1",
    "weighted_recall_at_5",
    "weighted_recall_at_20",
    "weighted_recall_at_64",
    "multi_positive_ap",
    "ndcg",
    "mrr",
    "null_ece",
    "null_brier",
    "top1_expected_maplet_center_distance_m",
    "candidate_entropy",
    "mean_best_cosine",
    "mean_predicted_null",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged PFIR output")
    reports = [json.loads(Path(path).read_text()) for path in args.inputs]
    invariant_keys = (
        "stage",
        "pooling",
        "support_mode",
        "physical_map_sha256",
        "canonical_field_sha256",
        "validity_calibration_sha256",
        "null_similarity_center",
        "null_similarity_scale",
        "stored_feature_type_count",
        "stored_downstream_embedding_count",
    )
    for key in invariant_keys:
        values = [report.get(key) for report in reports]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"PFIR shard lineage/config mismatch for {key}: {values}")
    rows = [row for report in reports for row in report.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("PFIR shards contain duplicate image IDs")
    result = {
        **{key: reports[0].get(key) for key in invariant_keys},
        "query_count": len(rows),
        "shard_count": len(reports),
        "source_shards": [str(Path(path)) for path in args.inputs],
        **{
            key: (
                float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
                if any(row.get(key) is not None for row in rows)
                else None
            )
            for key in ROW_METRICS
        },
        **{
            f"whole_image_coverage_at_{k}": (
                float(np.mean([float(row["scene"][f"coverage_at_{k}"]) for row in rows])) if rows else 0.0
            )
            for k in (1, 5, 20, 64)
        },
        "pose_sufficient_at_64_fraction": (
            float(np.mean([bool(row["scene"]["pose_sufficient_at_64"]) for row in rows])) if rows else 0.0
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
