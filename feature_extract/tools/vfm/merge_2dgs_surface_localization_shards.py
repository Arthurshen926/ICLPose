"""Merge complete map-only 2DGS localization shards in query-list order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_jsonl", nargs="+", required=True)
    parser.add_argument("--shard_summary", nargs="+", required=True)
    parser.add_argument(
        "--replacement_jsonl",
        nargs="*",
        default=(),
        help="Successful single-query reruns that replace failed shard rows.",
    )
    parser.add_argument("--expected_query_list", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    return parser.parse_args(argv)


def _query_ids(path: Path) -> list[str]:
    values = [
        line.strip().split()[0]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError("expected query list contains duplicate IDs")
    return values


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result_paths = [Path(value) for value in args.shard_jsonl]
    summary_paths = [Path(value) for value in args.shard_summary]
    if len(result_paths) != len(summary_paths) or not result_paths:
        raise ValueError("one summary is required for every result shard")
    expected_ids = _query_ids(Path(args.expected_query_list))
    records: dict[str, dict[str, object]] = {}
    reference_config = None
    reference_contract = None
    reference_artifacts = None
    shard_records: list[dict[str, object]] = []
    for result_path, summary_path in zip(result_paths, summary_paths):
        summary = dict(json.loads(summary_path.read_text()))
        if summary.get("stage") != "localize_2dgs_surface_queries_map_only":
            raise ValueError(f"unsupported shard stage: {summary_path}")
        config = summary.get("config")
        contract = summary.get("production_contract")
        artifacts = dict(summary.get("artifacts") or {})
        artifacts.pop("query_manifest", None)
        for current, reference, name in (
            (config, reference_config, "config"),
            (contract, reference_contract, "production contract"),
            (artifacts, reference_artifacts, "map artifacts"),
        ):
            if reference is not None and _canonical(current) != reference:
                raise ValueError(f"shards have inconsistent {name}")
        if reference_config is None:
            reference_config = _canonical(config)
            reference_contract = _canonical(contract)
            reference_artifacts = _canonical(artifacts)
        shard_count = 0
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            record = dict(json.loads(line))
            image_id = str(record["image_id"])
            if image_id in records:
                raise ValueError(f"duplicate query across shards: {image_id}")
            records[image_id] = record
            shard_count += 1
        if shard_count != int(summary.get("query_count", -1)):
            raise ValueError(f"shard result count differs from summary: {result_path}")
        shard_records.append(
            {
                "results": str(result_path),
                "results_sha256": file_sha256_short(result_path),
                "summary": str(summary_path),
                "summary_sha256": file_sha256_short(summary_path),
                "query_count": int(shard_count),
            }
        )
    replacement_records: list[dict[str, object]] = []
    for replacement_path_value in args.replacement_jsonl:
        replacement_path = Path(replacement_path_value)
        replacement_count = 0
        for line in replacement_path.read_text().splitlines():
            if not line.strip():
                continue
            replacement = dict(json.loads(line))
            image_id = str(replacement["image_id"])
            previous = records.get(image_id)
            if previous is None:
                raise ValueError(
                    f"replacement query is absent from shards: {image_id}"
                )
            if bool(previous.get("success")):
                raise ValueError(
                    f"refusing to replace a successful shard row: {image_id}"
                )
            if (
                not bool(replacement.get("success"))
                or replacement.get("pose_w2c") is None
            ):
                raise ValueError(
                    f"replacement row is not successful: {image_id}"
                )
            records[image_id] = replacement
            replacement_count += 1
        replacement_records.append(
            {
                "path": str(replacement_path),
                "sha256": file_sha256_short(replacement_path),
                "query_count": int(replacement_count),
            }
        )
    missing = sorted(set(expected_ids) - set(records))
    extra = sorted(set(records) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"shard coverage differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(records[image_id], sort_keys=True) + "\n"
            for image_id in expected_ids
        )
    )
    success_count = sum(
        int(
            bool(records[image_id].get("success"))
            and records[image_id].get("pose_w2c") is not None
        )
        for image_id in expected_ids
    )
    output = {
        "stage": "merge_2dgs_surface_localization_shards",
        "query_count": len(expected_ids),
        "success_count": int(success_count),
        "success_rate": float(success_count / max(len(expected_ids), 1)),
        "expected_query_list": str(args.expected_query_list),
        "expected_query_list_sha256": file_sha256_short(
            Path(args.expected_query_list)
        ),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": file_sha256_short(output_path),
        "shards": shard_records,
        "failure_replacements": replacement_records,
        "config": json.loads(reference_config),
        "production_contract": json.loads(reference_contract),
        "map_artifacts": json.loads(reference_artifacts),
    }
    summary_output = Path(args.output_summary)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
