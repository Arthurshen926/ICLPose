"""Merge disjoint GPU shards of G19-C phase audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.audit_goal_maplet_phase_basin import _sequence_metrics
from feature_extract.tools.vfm.audit_goal_maplet_phase_correctness import _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged phase audit")
    shards = [json.loads(Path(path).read_text()) for path in args.inputs]
    stages = {str(shard.get("stage")) for shard in shards}
    if len(stages) != 1:
        raise ValueError("phase audit shard stages differ")
    for key in ("physical_map_sha256", "canonical_field_sha256"):
        if len({str(shard.get(key)) for shard in shards}) != 1:
            raise ValueError(f"phase audit shard lineage differs: {key}")
    result = {key: value for key, value in shards[0].items() if key not in {"summary", "rows", "ground_truth", "query_count"}}
    rows = [row for shard in shards for row in shard.get("rows", [])]
    result["query_count"] = int(sum(int(shard.get("query_count", 0)) for shard in shards))
    result["rows"] = rows
    stage = next(iter(stages))
    if stage == "goal_maplet_phase_basin_g19_c":
        result["ground_truth"] = [row for shard in shards for row in shard.get("ground_truth", [])]
        result["summary"] = _sequence_metrics(rows, list(result["translation_magnitudes"]), list(result["rotation_magnitudes_deg"]))
    elif stage == "goal_maplet_phase_correctness_g19_c":
        result["summary"] = _summary(rows)
    else:
        raise ValueError(f"unsupported phase audit stage: {stage}")
    result["merged_shards"] = [str(Path(path)) for path in args.inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "ground_truth"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
