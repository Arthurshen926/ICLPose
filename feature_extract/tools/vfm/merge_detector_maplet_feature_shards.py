"""Merge strict inference-only detector-maplet feature shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


_REQUIRED_INFERENCE_FIELDS = {
    "selected_rows",
    "selected_columns",
    "features",
    "valid_edges",
}
_OPTIONAL_LEGACY_FIELDS = {"selected_from_pose_keep"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_shards", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_row_count", type=int, default=0)
    return parser.parse_args(argv)


def _load(path: Path) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files or "feature_names" not in payload.files:
            raise ValueError(f"{path}: feature shard has no schema metadata")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key not in {"metadata_json", "feature_names"}
        }
        names = tuple(np.asarray(payload["feature_names"]).astype(str).tolist())
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not _REQUIRED_INFERENCE_FIELDS.issubset(arrays) or (
        set(arrays) - _REQUIRED_INFERENCE_FIELDS - _OPTIONAL_LEGACY_FIELDS
    ):
        raise ValueError(
            f"{path}: inference-only feature fields differ: {sorted(arrays)}"
        )
    row_count = len(arrays["selected_rows"])
    if any(value.shape[0] != row_count for value in arrays.values()):
        raise ValueError(f"{path}: feature shard arrays are not row-aligned")
    if metadata.get("format") != "detector_maplet_geometry_features_v1":
        raise ValueError(f"{path}: unsupported detector-maplet feature format")
    if metadata.get("supervision_mode") != "none_inference_only":
        raise ValueError(f"{path}: feature shard is not inference-only")
    if not names or arrays["features"].shape[2] != len(names):
        raise ValueError(f"{path}: feature names and tensor dimension differ")
    return arrays, names, metadata


def _compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    output = dict(metadata)
    output.pop("query_shard_index", None)
    return output


def merge_feature_shards(
    paths: Sequence[Path],
    *,
    expected_row_count: int = 0,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, object]]:
    if not paths:
        raise ValueError("at least one feature shard is required")
    loaded = [_load(path) for path in paths]
    names = loaded[0][1]
    if any(item[1] != names for item in loaded[1:]):
        raise ValueError("feature shard schemas differ")
    compatibility = _compatibility(loaded[0][2])
    if any(_compatibility(item[2]) != compatibility for item in loaded[1:]):
        raise ValueError("feature shard manifests differ")
    shard_count = int(compatibility.get("query_shard_count", -1))
    shard_indices = [int(item[2].get("query_shard_index", -1)) for item in loaded]
    if shard_count != len(paths) or sorted(shard_indices) != list(range(shard_count)):
        raise ValueError("feature shard coverage is incomplete")
    arrays = {
        key: np.concatenate([item[0][key] for item in loaded], axis=0)
        for key in _REQUIRED_INFERENCE_FIELDS
    }
    legacy_pose_keep_present = any(
        "selected_from_pose_keep" in item[0] for item in loaded
    )
    if legacy_pose_keep_present and not all(
        "selected_from_pose_keep" in item[0] for item in loaded
    ):
        raise ValueError("feature shards mix legacy and target-free field contracts")
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    if len(np.unique(selected_rows)) != len(selected_rows):
        raise ValueError("feature shards overlap selected detector rows")
    order = np.argsort(selected_rows, kind="mergesort")
    arrays = {key: value[order] for key, value in arrays.items()}
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    if int(expected_row_count) > 0 and not np.array_equal(
        selected_rows, np.arange(int(expected_row_count), dtype=np.int64)
    ):
        raise ValueError("merged feature shards do not cover every expected row")
    if legacy_pose_keep_present and int(expected_row_count) <= 0:
        raise ValueError(
            "legacy pose-derived audit fields may only be removed after exact full-row coverage"
        )
    metadata = dict(compatibility)
    metadata.pop("query_shard_count", None)
    metadata["merged_query_shard_count"] = shard_count
    metadata["selected_row_count"] = int(len(selected_rows))
    if legacy_pose_keep_present:
        metadata["source_query_point_selection"] = metadata.get(
            "query_point_selection"
        )
        metadata["query_point_selection"] = (
            "full_proposal_row_coverage_target_free_sanitized_v1"
        )
        metadata["legacy_pose_keep_audit_field_removed"] = True
    return arrays, names, metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = tuple(
        Path(value.strip())
        for value in str(args.feature_shards).split(",")
        if value.strip()
    )
    arrays, names, metadata = merge_feature_shards(
        paths, expected_row_count=int(args.expected_row_count)
    )
    metadata["source_feature_shards"] = [str(path) for path in paths]
    metadata["source_feature_shard_sha256"] = [
        file_sha256_short(path) for path in paths
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **arrays,
        feature_names=np.asarray(names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "merge_detector_maplet_feature_shards",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "row_count": int(len(arrays["selected_rows"])),
        "feature_count": int(len(names)),
        "shard_count": int(len(paths)),
        "supervision_loaded": False,
    }
    (output.parent / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
