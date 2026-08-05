"""Merge disjoint child-local likelihood shards with query-balanced metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged child-local report")
    shards = [json.loads(Path(path).read_text()) for path in args.inputs]
    for key in (
        "stage", "protocol", "physical_map_sha256", "canonical_field_sha256",
        "field_feature_contract_sha256", "child_eligibility_sha256",
        "temperature", "maximum_modes", "joint_pose_trials", "child_local_mode_ranker",
        "child_local_mode_ranker_sha256",
    ):
        values = {json.dumps(shard.get(key), sort_keys=True) for shard in shards}
        if len(values) != 1:
            raise ValueError(f"child-local shard lineage/config differs: {key}")
    rows = sorted(
        [row for shard in shards for row in shard.get("rows", [])],
        key=lambda row: str(row["image_id"]),
    )
    if len({str(row["image_id"]) for row in rows}) != len(rows):
        raise ValueError("child-local shards contain duplicate queries")
    direct_names = sorted({
        name for row in rows for name in row.get("direct_error_median_m", {})
    })
    pose_names = sorted({name for row in rows for name in row.get("pose", {})})
    query_balanced_direct = {
        name: _metrics([
            float(row["direct_error_median_m"][name])
            for row in rows if name in row.get("direct_error_median_m", {})
        ])
        for name in direct_names
    }
    pose_error = {}
    for name in pose_names:
        selected = [
            row["pose"][name] for row in rows
            if row.get("pose", {}).get(name, {}).get("success", False)
        ]
        pose_error[name] = {
            "success_fraction": float(len(selected) / max(len(rows), 1)),
            "translation_m": _metrics([float(value["translation_m"]) for value in selected]),
            "rotation_deg": _metrics([float(value["rotation_deg"]) for value in selected]),
        }
    result = {
        **{key: shards[0].get(key) for key in (
            "stage", "protocol", "physical_map_sha256", "canonical_field_sha256",
            "field_feature_contract_sha256", "child_eligibility_sha256",
            "temperature", "maximum_modes", "joint_pose_trials", "child_local_mode_ranker",
            "child_local_mode_ranker_sha256",
        )},
        "query_count": len(rows),
        "shard_count": len(shards),
        "source_shards": [str(value) for value in args.inputs],
        "eligible_group_count": int(sum(int(row.get("eligible_group_count", 0)) for row in rows)),
        "query_balanced_direct_point_error_m": query_balanced_direct,
        "pose_error": pose_error,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
