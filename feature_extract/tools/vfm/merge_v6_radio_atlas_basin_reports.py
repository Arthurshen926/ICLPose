"""Merge disjoint shards from evaluate_v6_radio_atlas_basin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_v6_radio_atlas_basin import _aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    reports = [
        json.loads(Path(value).read_text(encoding="utf-8"))
        for value in args.inputs
    ]
    if not reports:
        raise ValueError("no shard reports")
    reference = reports[0]
    for report in reports[1:]:
        if report["artifact_sha256"] != reference["artifact_sha256"]:
            raise ValueError("shard artifacts differ")
        first = dict(reference["configuration"])
        second = dict(report["configuration"])
        first.pop("query_shard_index", None)
        second.pop("query_shard_index", None)
        if first != second:
            raise ValueError("shard configurations differ")
    rows = sorted(
        [row for report in reports for row in report.get("rows", [])],
        key=lambda value: str(value["image_id"]),
    )
    image_ids = [str(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("query shards overlap")
    merged = dict(reference)
    merged["stage"] = "v6_radio_atlas_oracle_identity_local_basin_merged"
    merged["rows"] = rows
    merged["summary"] = _aggregate(rows)
    configuration = dict(reference["configuration"])
    configuration["query_shard_index"] = "merged"
    configuration["merged_shard_count"] = len(reports)
    merged["configuration"] = configuration
    protocol = dict(reference["protocol"])
    protocol["strict_test"] = sorted(
        {
            str(value)
            for report in reports
            for value in report["protocol"].get("strict_test", [])
        }
    )
    merged["protocol"] = protocol
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(merged["summary"], indent=2))


if __name__ == "__main__":
    main()
