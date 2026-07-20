"""Strictly merge contiguous frozen-layout structured S1b feature shards.

The structured exporter is intentionally CPU-heavy because each candidate is
compared with real support-image context.  Sharding is safe only when the
merged artifact restores the original frozen layout exactly; this tool rejects
missing, overlapping, reordered, diagnostic, or manifest-mismatched shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
)


ARTIFACT_FORMAT = "structured_multiscale_candidate_probe_features_v2"
COST_VOLUME_ARTIFACT_FORMAT = "cost_volume_multiscale_candidate_probe_features_v1"
WIDE_FULL_CORRELATION_ARTIFACT_FORMAT = (
    "wide_full_correlation_multiscale_candidate_probe_features_v1"
)
_SUPPORTED_SCHEMAS = {
    ARTIFACT_FORMAT: STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    COST_VOLUME_ARTIFACT_FORMAT: COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_ARTIFACT_FORMAT: (
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    ),
}
FROZEN_LAYOUT_FORMAT = "multiscale_candidate_probe_features_v1"
HELDOUT_SOURCE_ROW_SELECTION = "heldout_detector_merit_after_target_free_fit_rows_v1"
_ROW_FIELDS = (
    "layout_positions",
    "source_row_indices",
    "query_ids",
    "split_names",
    "xy",
    "candidate_track_ids",
    "candidate_canonical_rows",
    "candidate_features",
    "candidate_view_valid",
    "candidate_support_image_ids",
    "candidate_support_coverage_counts",
)
_SHARD_METADATA_FIELDS = {
    "layout_shard_count",
    "layout_shard_index",
    "layout_position_count",
    "is_complete_frozen_layout",
    "query_row_count",
    "query_image_count",
    "split_row_counts",
    "diagnostic_max_rows",
    "frozen_source_rows_sha256",
    "frozen_candidate_tracks_sha256",
    "frozen_support_view_ids_sha256",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_shards", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, object]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} has no metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _load_shard(path: Path) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError(f"{path}: shard unexpectedly contains labels")
        missing = (set(_ROW_FIELDS) | {"feature_names", "metadata_json"}) - set(data.files)
        if missing:
            raise ValueError(f"{path}: shard lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in _ROW_FIELDS}
        names = tuple(np.asarray(data["feature_names"]).astype(str).tolist())
        metadata = _metadata(data, context=str(path))
    expected_names = _SUPPORTED_SCHEMAS.get(str(metadata.get("format")))
    if expected_names is None:
        raise ValueError(f"{path}: unsupported frozen feature format")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError(f"{path}: frozen feature shard is not target-free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)) or bool(
        metadata.get("whole_image_summary_or_global_used", True)
    ):
        raise ValueError(f"{path}: frozen feature shard violates no-retrieval protocol")
    if int(metadata.get("diagnostic_max_rows", 0)) != 0:
        raise ValueError(f"{path}: diagnostic frozen feature shard cannot be merged")
    if bool(metadata.get("is_complete_frozen_layout", True)):
        raise ValueError(f"{path}: expected a proper layout shard, got a complete artifact")
    row_count = len(arrays["layout_positions"])
    if row_count == 0 or any(values.shape[0] != row_count for values in arrays.values()):
        raise ValueError(f"{path}: frozen feature shard row arrays are not aligned")
    positions = np.asarray(arrays["layout_positions"], dtype=np.int64).reshape(-1)
    if np.any(positions < 0) or not np.all(positions[1:] > positions[:-1]):
        raise ValueError(f"{path}: layout positions are not strictly increasing")
    if int(metadata.get("layout_position_count", -1)) != row_count:
        raise ValueError(f"{path}: layout-position count differs from artifact rows")
    if names != expected_names:
        raise ValueError(f"{path}: unsupported frozen feature schema")
    features = np.asarray(arrays["candidate_features"], dtype=np.float32)
    views = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    if features.ndim != 4 or views.shape != features.shape[:3] or features.shape[3] != len(names):
        raise ValueError(f"{path}: frozen feature tensor is invalid")
    valid_values = features[views]
    if np.any(np.isinf(valid_values)) or np.any(~np.isfinite(valid_values[:, :3])):
        raise ValueError(f"{path}: frozen feature anchors are invalid")
    return arrays, names, metadata


def _compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _SHARD_METADATA_FIELDS
    }


def _load_and_validate_frozen_layout_protocol(
    metadata: Mapping[str, object],
    *,
    merged_arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Bind merged evidence to the exact held-out layout it claims to extend."""

    source = Path(str(metadata.get("frozen_layout_features", "")))
    expected_sha256 = str(metadata.get("frozen_layout_features_sha256", ""))
    if not source.exists() or not expected_sha256 or file_sha256_short(source) != expected_sha256:
        raise ValueError("structured shards reference a missing or stale frozen layout")
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_view_valid",
        "candidate_support_image_ids",
        "candidate_support_coverage_counts",
        "feature_names",
        "metadata_json",
    }
    with np.load(source, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"frozen S1 layout lacks {sorted(missing)}")
        frozen_metadata = _metadata(data, context="frozen S1 layout")
        frozen = {
            key: np.asarray(data[key])
            for key in (
                "source_row_indices",
                "query_ids",
                "split_names",
                "xy",
                "candidate_track_ids",
                "candidate_canonical_rows",
                "candidate_view_valid",
                "candidate_support_image_ids",
                "candidate_support_coverage_counts",
            )
        }
    if frozen_metadata.get("format") != FROZEN_LAYOUT_FORMAT:
        raise ValueError("structured shards reference an unsupported frozen layout")
    if frozen_metadata.get("contains_ground_truth") is not False or frozen_metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError("structured shards reference a non-target-free frozen layout")
    if bool(frozen_metadata.get("image_retrieval_or_submap_used", True)) or bool(
        frozen_metadata.get("whole_image_summary_or_global_used", True)
    ):
        raise ValueError("structured shards reference a retrieval/global frozen layout")
    if frozen_metadata.get("source_row_selection") != HELDOUT_SOURCE_ROW_SELECTION:
        raise ValueError("structured frozen layout is not held out from hypothesis fit rows")
    if int(frozen_metadata.get("verification_point_count", 0)) <= 0 or not str(
        frozen_metadata.get("support_view_selection", "")
    ):
        raise ValueError("structured frozen layout lacks held-out/support-view protocol")
    for key, value in frozen.items():
        if not np.array_equal(np.asarray(merged_arrays[key]), value):
            raise ValueError(f"merged structured rows differ from frozen layout field {key}")
    return {
        "source_row_selection": frozen_metadata["source_row_selection"],
        "verification_point_count": int(frozen_metadata["verification_point_count"]),
        "support_view_selection": frozen_metadata["support_view_selection"],
        "support_view_count": int(frozen_metadata.get("support_view_count", 0)),
        "candidate_fit_rows_sha256": frozen_metadata.get("candidate_fit_rows_sha256"),
        "detector_log_merit_weight": frozen_metadata.get("detector_log_merit_weight"),
        "frozen_layout_feature_definition": frozen_metadata.get("feature_definition"),
    }


def merge_structured_multiscale_candidate_probe_shards(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, object]]:
    if not paths:
        raise ValueError("at least one structured feature shard is required")
    loaded = [_load_shard(Path(path)) for path in paths]
    names = loaded[0][1]
    if any(item[1] != names for item in loaded[1:]):
        raise ValueError("structured shard schemas differ")
    compatibility = _compatibility(loaded[0][2])
    if any(_compatibility(item[2]) != compatibility for item in loaded[1:]):
        raise ValueError("structured shard manifests differ")
    shard_count = int(loaded[0][2].get("layout_shard_count", -1))
    shard_indices = [int(item[2].get("layout_shard_index", -1)) for item in loaded]
    if shard_count <= 1 or shard_count != len(loaded) or sorted(shard_indices) != list(
        range(shard_count)
    ):
        raise ValueError("structured shard coverage is incomplete")
    full_count = int(compatibility.get("full_frozen_layout_row_count", -1))
    if full_count <= 0:
        raise ValueError("structured shard has no full frozen-layout count")
    arrays = {
        key: np.concatenate([item[0][key] for item in loaded], axis=0)
        for key in _ROW_FIELDS
    }
    order = np.argsort(np.asarray(arrays["layout_positions"], dtype=np.int64), kind="stable")
    arrays = {key: values[order] for key, values in arrays.items()}
    positions = np.asarray(arrays.pop("layout_positions"), dtype=np.int64)
    if not np.array_equal(positions, np.arange(full_count, dtype=np.int64)):
        raise ValueError("structured shards do not cover every frozen layout position once")
    frozen_protocol = _load_and_validate_frozen_layout_protocol(
        compatibility, merged_arrays=arrays
    )
    metadata = dict(compatibility)
    metadata.update(
        {
            "query_row_count": int(full_count),
            "query_image_count": int(
                len(set(np.asarray(arrays["query_ids"]).astype(str).tolist()))
            ),
            "split_row_counts": {
                split: int(np.sum(np.asarray(arrays["split_names"]).astype(str) == split))
                for split in ("train", "validation", "test")
            },
            "diagnostic_max_rows": 0,
            "is_complete_frozen_layout": True,
            "merged_layout_shard_count": int(shard_count),
            "merged_layout_shard_indices": sorted(shard_indices),
            "frozen_source_rows_sha256": _array_sha256_short(arrays["source_row_indices"]),
            "frozen_candidate_tracks_sha256": _array_sha256_short(
                arrays["candidate_track_ids"]
            ),
            "frozen_support_view_ids_sha256": _array_sha256_short(
                arrays["candidate_support_image_ids"]
            ),
            **frozen_protocol,
        }
    )
    for full_key, emitted_key in (
        ("full_frozen_source_rows_sha256", "frozen_source_rows_sha256"),
        ("full_frozen_candidate_tracks_sha256", "frozen_candidate_tracks_sha256"),
        ("full_frozen_support_view_ids_sha256", "frozen_support_view_ids_sha256"),
    ):
        if str(metadata.get(full_key)) != str(metadata[emitted_key]):
            raise ValueError(f"merged structured {emitted_key} differs from frozen layout")
    return arrays, names, metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = tuple(
        Path(value.strip())
        for value in str(args.feature_shards).split(",")
        if value.strip()
    )
    output = Path(args.output)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite structured merged artifact")
    arrays, names, metadata = merge_structured_multiscale_candidate_probe_shards(paths)
    metadata["source_feature_shards"] = [str(path) for path in paths]
    metadata["source_feature_shard_sha256"] = [file_sha256_short(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            feature_names=np.asarray(names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    temporary.replace(output)
    summary = {
        "stage": "merge_frozen_structured_multiscale_candidate_probe_shards",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "row_count": int(len(arrays["source_row_indices"])),
        "feature_count": int(len(names)),
        "shard_count": int(metadata["merged_layout_shard_count"]),
        "contains_ground_truth": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
