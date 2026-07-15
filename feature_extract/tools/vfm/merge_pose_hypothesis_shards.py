"""Merge deterministic query shards from pose-hypothesis evaluation.

The merger is reporting-only. It refuses incomplete shards, configuration or
artifact drift, duplicate queries, and split coverage gaps before recomputing
full-split pose summaries from per-query rows.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_pose_hypothesis_verification import (
    SELECTED_POSE_ARTIFACT_FORMAT,
    _canonical_manifest_sha256,
    _load_npz,
    _pose_summary,
    _query_execution_shard,
    _write_selected_pose_artifact,
)
from feature_extract.vfm.artifacts import file_sha256_short


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_shard(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    summary_path = path / "summary.json"
    rows_path = path / "pose_rows.json"
    if not summary_path.exists() or not rows_path.exists():
        raise ValueError(f"shard is missing summary/rows: {path}")
    summary = json.loads(summary_path.read_text())
    rows = json.loads(rows_path.read_text())
    if summary.get("stage") != "heldout_multi_hypothesis_pose_verification":
        raise ValueError(f"unsupported shard stage: {path}")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "pose_rows_sha256"
    ) != file_sha256_short(rows_path):
        raise ValueError(f"shard pose rows are stale: {path}")
    return summary, rows


def _load_selected_pose_rows(
    shard_path: Path,
    summary: Mapping[str, object],
) -> list[dict[str, object]] | None:
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"shard outputs are missing: {shard_path}")
    artifact_value = outputs.get("selected_pose_artifact")
    artifact_hash = outputs.get("selected_pose_artifact_sha256")
    if artifact_value is None and artifact_hash is None:
        return None
    if not isinstance(artifact_value, str) or not artifact_value or not artifact_hash:
        raise ValueError(f"shard selected pose output is incomplete: {shard_path}")
    artifact_path = Path(artifact_value)
    if not artifact_path.exists() or file_sha256_short(artifact_path) != artifact_hash:
        raise ValueError(f"shard selected pose artifact is stale: {shard_path}")
    payload = _load_npz(artifact_path)
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "success",
        "poses_w2c",
        "match_counts",
        "inlier_counts",
        "metadata_json",
    }
    if set(payload) != required:
        raise ValueError(f"shard selected pose fields differ: {shard_path}")
    metadata = json.loads(str(payload["metadata_json"].item()))
    source_manifest = metadata.get("source_manifest")
    expected_manifest = {
        "stage": summary.get("stage"),
        "inputs": summary.get("inputs"),
        "protocol": summary.get("protocol"),
        "execution": summary.get("execution"),
        "config": summary.get("config"),
    }
    if (
        metadata.get("format") != SELECTED_POSE_ARTIFACT_FORMAT
        or not isinstance(source_manifest, dict)
        or metadata.get("source_manifest_sha256")
        != _canonical_manifest_sha256(source_manifest)
        or _canonical_json(source_manifest) != _canonical_json(expected_manifest)
    ):
        raise ValueError(f"shard selected pose manifest is stale: {shard_path}")
    query_ids = payload["query_ids"].astype(str).reshape(-1)
    split_names = payload["split_names"].astype(str).reshape(-1)
    labels = payload["evaluation_labels"].astype(str).reshape(-1)
    success = payload["success"].astype(bool).reshape(-1)
    poses = payload["poses_w2c"].astype("float64")
    match_counts = payload["match_counts"].astype("int64").reshape(-1)
    inlier_counts = payload["inlier_counts"].astype("int64").reshape(-1)
    row_count = len(query_ids)
    if (
        any(
            len(values) != row_count
            for values in (
                split_names,
                labels,
                success,
                match_counts,
                inlier_counts,
            )
        )
        or poses.shape != (row_count, 4, 4)
        or int(metadata.get("row_count", -1)) != row_count
    ):
        raise ValueError(f"shard selected pose dimensions differ: {shard_path}")
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        row_success = bool(success[index])
        pose = poses[index]
        if row_success and not bool(np.all(np.isfinite(pose))):
            raise ValueError(f"successful shard pose is non-finite: {shard_path}")
        if not row_success and bool(np.any(np.isfinite(pose))):
            raise ValueError(f"failed shard pose carries finite values: {shard_path}")
        rows.append(
            {
                "query_id": str(query_ids[index]),
                "split_name": str(split_names[index]),
                "evaluation_label": str(labels[index]),
                "success": row_success,
                "pose_w2c": pose if row_success else None,
                "match_count": int(match_counts[index]),
                "inlier_count": int(inlier_counts[index]),
            }
        )
    return rows


def _merge_selected_pose_rows(
    shard_rows: Sequence[list[dict[str, object]] | None],
    *,
    split: Mapping[str, Sequence[str]],
) -> list[dict[str, object]] | None:
    present = [rows for rows in shard_rows if rows is not None]
    if not present:
        return None
    if len(present) != len(shard_rows):
        raise ValueError("only some shards exported selected pose artifacts")
    rows_by_identity: dict[tuple[str, str, str], dict[str, object]] = {}
    for rows in present:
        for row in rows:
            identity = (
                str(row["evaluation_label"]),
                str(row["split_name"]),
                str(row["query_id"]),
            )
            if identity in rows_by_identity:
                raise ValueError(f"duplicate selected pose identity: {identity}")
            rows_by_identity[identity] = row
    split_order = tuple(name for name in ("train", "validation", "test") if name in split)
    labels = sorted({identity[0] for identity in rows_by_identity})
    merged: list[dict[str, object]] = []
    for label in labels:
        for split_name in split_order:
            available = {
                query_id
                for candidate_label, candidate_split, query_id in rows_by_identity
                if candidate_label == label and candidate_split == split_name
            }
            if not available:
                continue
            expected = tuple(str(value) for value in split[split_name])
            if available != set(expected):
                raise ValueError(
                    "selected pose query coverage mismatch for "
                    f"{label}/{split_name}"
                )
            merged.extend(
                rows_by_identity[(label, split_name, query_id)]
                for query_id in expected
            )
    if len(merged) != len(rows_by_identity):
        unknown = sorted(
            {
                identity[1]
                for identity in rows_by_identity
                if identity[1] not in split_order
            }
        )
        raise ValueError(f"selected pose artifacts contain unknown splits: {unknown}")
    return merged


def _merge_row_sections(
    shards: Sequence[dict[str, object]],
    *,
    split: Mapping[str, Sequence[str]],
) -> tuple[dict[str, object], dict[str, object]]:
    section_to_split = {
        "train_oof_geometry": "train",
        "validation": "validation",
        "late_development": "test",
    }
    merged: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for section, split_name in section_to_split.items():
        section_payloads = [dict(rows.get(section, {})) for rows in shards]
        policy_sets = [set(payload) for payload in section_payloads]
        nonempty_policy_sets = [value for value in policy_sets if value]
        if not nonempty_policy_sets:
            merged[section] = {}
            metrics[section] = {}
            continue
        if any(value != nonempty_policy_sets[0] for value in policy_sets):
            raise ValueError(f"shards expose different policies in {section}")
        expected_ids = [str(value) for value in split[split_name]]
        expected_set = set(expected_ids)
        merged_section: dict[str, object] = {}
        metric_section: dict[str, object] = {}
        for policy in sorted(nonempty_policy_sets[0]):
            backend_sets = [set(dict(payload[policy])) for payload in section_payloads]
            if any(value != backend_sets[0] for value in backend_sets):
                raise ValueError(
                    f"shards expose different backends for {section}/{policy}"
                )
            merged_policy: dict[str, object] = {}
            metric_policy: dict[str, object] = {}
            for backend in sorted(backend_sets[0]):
                rows_by_id: dict[str, dict[str, object]] = {}
                for payload in section_payloads:
                    for row in dict(payload[policy])[backend]:
                        query_id = str(row["query_id"])
                        if query_id in rows_by_id:
                            raise ValueError(
                                f"duplicate query {query_id} in {section}/{policy}/{backend}"
                            )
                        rows_by_id[query_id] = dict(row)
                if set(rows_by_id) != expected_set:
                    missing = sorted(expected_set - set(rows_by_id))[:5]
                    extra = sorted(set(rows_by_id) - expected_set)[:5]
                    raise ValueError(
                        f"query coverage mismatch in {section}/{policy}/{backend}: "
                        f"missing={missing}, extra={extra}"
                    )
                ordered_rows = [rows_by_id[query_id] for query_id in expected_ids]
                merged_policy[backend] = ordered_rows
                metric_policy[backend] = _pose_summary(ordered_rows)
            merged_section[policy] = merged_policy
            metric_section[policy] = metric_policy
        merged[section] = merged_section
        metrics[section] = metric_section
    return merged, metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    shard_dirs = tuple(Path(value) for value in args.shard_dirs)
    loaded = [_load_shard(path) for path in shard_dirs]
    summaries = [item[0] for item in loaded]
    rows = [item[1] for item in loaded]
    first = summaries[0]
    invariant = {
        "inputs": first.get("inputs"),
        "protocol": first.get("protocol"),
        "config": first.get("config"),
    }
    for summary in summaries[1:]:
        candidate = {
            "inputs": summary.get("inputs"),
            "protocol": summary.get("protocol"),
            "config": summary.get("config"),
        }
        if _canonical_json(candidate) != _canonical_json(invariant):
            raise ValueError("shard inference inputs/configuration differ")

    executions = [dict(summary.get("execution", {})) for summary in summaries]
    shard_counts = {int(item.get("query_shard_count", -1)) for item in executions}
    if len(shard_counts) != 1:
        raise ValueError("shard counts differ")
    shard_count = shard_counts.pop()
    shard_indices = [int(item.get("query_shard_index", -1)) for item in executions]
    if shard_count <= 1 or sorted(shard_indices) != list(range(shard_count)):
        raise ValueError("all nontrivial shard indices must be present exactly once")
    if len(shard_dirs) != shard_count:
        raise ValueError("shard directory count differs from query_shard_count")

    split_path = Path(str(dict(first["inputs"])["split_json"]))
    split = json.loads(split_path.read_text())
    for execution in executions:
        index = int(execution["query_shard_index"])
        manifest = dict(execution.get("query_ids", {}))
        for split_name in ("train", "validation", "test"):
            expected = _query_execution_shard(
                split[split_name], shard_count=shard_count, shard_index=index
            )
            if tuple(str(value) for value in manifest.get(split_name, ())) != expected:
                raise ValueError("shard query manifest differs from deterministic policy")

    merged_rows, metrics = _merge_row_sections(rows, split=split)
    selected_pose_rows = _merge_selected_pose_rows(
        [
            _load_selected_pose_rows(path, summary)
            for path, summary in zip(shard_dirs, summaries)
        ],
        split=split,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(json.dumps(merged_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "heldout_multi_hypothesis_pose_verification_shard_merge",
        "inputs": deepcopy(invariant["inputs"]),
        "protocol": deepcopy(invariant["protocol"]),
        "config": deepcopy(invariant["config"]),
        "execution": {
            "query_shard_count": shard_count,
            "query_shard_policy": "split_order_position_modulo_v1",
            "complete_shard_set": True,
            "source_shards": [
                {
                    "path": str(path),
                    "summary_sha256": file_sha256_short(path / "summary.json"),
                    "pose_rows_sha256": file_sha256_short(path / "pose_rows.json"),
                    "selected_pose_artifact_sha256": dict(
                        summary.get("outputs", {})
                    ).get("selected_pose_artifact_sha256"),
                    "query_shard_index": int(execution["query_shard_index"]),
                }
                for path, execution, summary in sorted(
                    zip(shard_dirs, executions, summaries),
                    key=lambda item: int(item[1]["query_shard_index"]),
                )
            ],
        },
        "metrics": metrics,
        "outputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "selected_pose_artifact": None,
            "selected_pose_artifact_sha256": None,
            "summary": str(output_dir / "summary.json"),
        },
    }
    if selected_pose_rows is not None:
        selected_pose_path = output_dir / SELECTED_POSE_ARTIFACT_FORMAT
        selected_pose_path = selected_pose_path.with_suffix(".npz")
        source_manifest = {
            "stage": summary["stage"],
            "inputs": summary["inputs"],
            "protocol": summary["protocol"],
            "execution": summary["execution"],
            "config": summary["config"],
        }
        _write_selected_pose_artifact(
            selected_pose_path,
            selected_pose_rows,
            source_manifest=source_manifest,
        )
        summary["outputs"]["selected_pose_artifact"] = str(selected_pose_path)
        summary["outputs"]["selected_pose_artifact_sha256"] = file_sha256_short(
            selected_pose_path
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
