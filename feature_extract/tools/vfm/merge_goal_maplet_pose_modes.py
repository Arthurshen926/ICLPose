"""Merge disjoint Goal-Maplet pose-mode shards with lineage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _risk_summary
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


_SHARD_LOCAL_FIELDS = {"query_count", "rows", "run_manifest", "summary", "risk_summary"}


_SOURCE_STATE_FIELDS = (
    "git_commit_sha", "git_tracked_worktree_dirty",
    "git_tracked_status_sha256", "git_tracked_diff_sha256",
    "untracked_source_file_count", "untracked_source_files_sha256",
    "repository_state_capture",
)


def _merged_run_manifest(
    shards: list[dict[str, object]],
    rows: list[dict[str, object]],
    input_paths: list[Path],
) -> dict[str, object]:
    manifests = [value.get("run_manifest") for value in shards]
    if all(value is None for value in manifests):
        return {
            "schema": "goal_maplet_legacy_merged_run_manifest_v1",
            "source_run_manifests": manifests,
            "source_shard_sha256": [file_sha256(path) for path in input_paths],
        }
    if any(not isinstance(value, dict) for value in manifests):
        raise ValueError("pose shards mix missing and present run manifests")
    source = [dict(value) for value in manifests]
    if any(value.get("schema") != "goal_maplet_run_manifest_v1" for value in source):
        raise ValueError("pose shard run-manifest schema differs")
    for field in _SOURCE_STATE_FIELDS:
        if len({json.dumps(value.get(field), sort_keys=True) for value in source}) != 1:
            raise ValueError(f"pose shard process-start source state differs: {field}")
    if len({
        json.dumps(value.get("numeric_contract"), sort_keys=True) for value in source
    }) != 1:
        raise ValueError("pose shard numeric contracts differ")
    configurations = []
    for value in source:
        configuration = dict(value.get("configuration", {}))
        configuration.pop("shard_index", None)
        configuration.pop("shard_count", None)
        configurations.append(configuration)
    if any(value != configurations[0] for value in configurations[1:]):
        raise ValueError("pose shard inference configurations differ")
    image_ids = [str(row["image_id"]) for row in rows]
    candidate_counts = {}
    for value in source:
        for image_id, counts in value.get("candidate_counts", {}).items():
            if image_id in candidate_counts:
                raise ValueError(f"pose shard candidate counts overlap: {image_id}")
            candidate_counts[str(image_id)] = counts
    return {
        "schema": "goal_maplet_run_manifest_v1",
        **{field: source[0][field] for field in _SOURCE_STATE_FIELDS},
        "merge_semantics": "disjoint_query_shards_with_identical_start_source_state",
        "argv": [value.get("argv", []) for value in source],
        "configuration": configurations[0],
        "configuration_sha256": canonical_json_sha256(configurations[0]),
        "input_artifacts": {
            "source_input_artifacts": [value.get("input_artifacts", {}) for value in source],
            "source_shard_sha256": [file_sha256(path) for path in input_paths],
        },
        "query_ids": image_ids,
        "query_list_sha256": canonical_json_sha256(image_ids),
        "device": source[0].get("device"),
        "numeric_contract": source[0]["numeric_contract"],
        "candidate_counts": candidate_counts,
        "source_run_manifests": source,
        "source_shard_sha256": [file_sha256(path) for path in input_paths],
    }


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
    # Every non-query field is configuration or lineage.  Enumerating a short
    # allow-list silently stopped protecting new fields such as mapping-view
    # graph and geometry-head hashes, so compare the complete top-level
    # contract and exclude only fields that must differ between shards.
    contract_keys = sorted(set().union(*(set(item) for item in shards)) - _SHARD_LOCAL_FIELDS)
    for key in contract_keys:
        values = {json.dumps(item.get(key), sort_keys=True) for item in shards}
        if len(values) != 1:
            raise ValueError(f"pose shard lineage/config differs: {key}")
    rows = [row for shard in shards for row in shard.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("pose shards contain duplicate queries")
    rows.sort(key=lambda row: str(row["image_id"]))
    input_paths = [Path(value) for value in args.inputs]
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
        **{key: shards[0].get(key) for key in contract_keys},
        "query_count": len(rows),
        "shard_count": len(shards),
        "source_shards": [str(value) for value in args.inputs],
        "source_shard_sha256": [file_sha256(path) for path in input_paths],
        "run_manifest": _merged_run_manifest(shards, rows, input_paths),
        "summary": summary,
        "risk_summary": _risk_summary(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
