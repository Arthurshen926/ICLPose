"""Export a fixed-maplet-support8 soft global RADIO-context probe artifact.

Unlike the two-view S1f artifact, this diagnostic retains every deterministic
coverage-ranked support image attached to a fixed landmark candidate.  It
still compares only query/support image pairs already present in the maplet
index: no image search, candidate reselection, pose, residual, or target is
available to the exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
    _descriptor_indices,
    _load_frozen_layout,
    _load_radio_final_global_descriptors,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT,
    GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES,
    GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT,
    fixed_candidate_support_global_context_cosine,
)


ARTIFACT_FORMAT = GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT
GLOBAL_CONTEXT_USAGE = GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT[ARTIFACT_FORMAT]
MAPLET_FORMAT = "local_maplet_support_index_npz"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _load_maplet_support_index(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "anchor_track_ids",
        "support_image_ids",
        "support_image_indices",
        "support_coverage_counts",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"maplet support index lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="maplet support index")
    if metadata.get("format") != MAPLET_FORMAT:
        raise ValueError("maplet support index format is unsupported")
    tracks = np.asarray(arrays["anchor_track_ids"], dtype=np.int64).reshape(-1)
    image_ids = np.asarray(arrays["support_image_ids"]).astype(str).reshape(-1)
    indices = np.asarray(arrays["support_image_indices"], dtype=np.int64)
    coverage = np.asarray(arrays["support_coverage_counts"], dtype=np.int32)
    if (
        len(tracks) == 0
        or len(set(tracks.tolist())) != len(tracks)
        or len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or indices.ndim != 2
        or indices.shape[0] != len(tracks)
        or coverage.shape != indices.shape
        or indices.shape[1] <= 0
        or np.any((indices < -1) | (indices >= len(image_ids)))
        or np.any(coverage < 0)
        or np.any((indices < 0) & (coverage != 0))
    ):
        raise ValueError("maplet support index arrays are invalid")
    if metadata.get("max_support_views") not in (None, int(indices.shape[1])):
        raise ValueError("maplet support-view count differs from its manifest")
    return {
        "anchor_track_ids": tracks,
        "support_image_ids": image_ids,
        "support_image_indices": indices,
        "support_coverage_counts": coverage,
    }, metadata


def _fixed_support8_layout(
    *, layout: Mapping[str, np.ndarray], maplet: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical = np.asarray(layout["candidate_canonical_rows"], dtype=np.int64)
    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    if canonical.shape != tracks.shape:
        raise ValueError("frozen candidate rows and tracks are incompatible")
    valid_candidate = canonical >= 0
    if np.any(canonical[valid_candidate] >= len(maplet["anchor_track_ids"])):
        raise ValueError("frozen candidate canonical row is absent from maplet support index")
    safe_canonical = np.maximum(canonical, 0)
    expected_tracks = np.asarray(maplet["anchor_track_ids"], dtype=np.int64)[safe_canonical]
    if np.any(expected_tracks[valid_candidate] != tracks[valid_candidate]):
        raise ValueError("frozen candidate tracks differ from maplet support index")
    source_indices = np.asarray(maplet["support_image_indices"], dtype=np.int64)[safe_canonical]
    coverage = np.asarray(maplet["support_coverage_counts"], dtype=np.int32)[safe_canonical]
    view_valid = valid_candidate[..., None] & (source_indices >= 0)
    image_ids = np.asarray(maplet["support_image_ids"]).astype(str)
    support_ids = np.full(source_indices.shape, "", dtype=image_ids.dtype)
    support_ids[view_valid] = image_ids[source_indices[view_valid]]
    coverage = np.where(view_valid, coverage, 0).astype(np.int32)
    if np.any(valid_candidate & ~np.any(view_valid, axis=2)):
        raise ValueError("a fixed candidate has no maplet support view")
    if np.any(support_ids[view_valid] == ""):
        raise RuntimeError("valid maplet support view has no image id")
    return support_ids, view_valid, coverage


def build_global_context_support8_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    maplet_support_index: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite support8 global-context probe outputs")
    layout, layout_metadata = _load_frozen_layout(Path(frozen_layout_features))
    maplet, maplet_metadata = _load_maplet_support_index(Path(maplet_support_index))
    radio, radio_metadata = _load_radio_final_global_descriptors(Path(radio_final_context_cache))
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposals_sha256 lineage")
    support_ids, view_valid, coverage = _fixed_support8_layout(layout=layout, maplet=maplet)
    query_indices, support_indices = _descriptor_indices(
        image_ids=radio["image_ids"],
        query_ids=layout["query_ids"],
        support_ids=support_ids,
        view_valid=view_valid,
    )
    query_global = radio["global_descriptors"][query_indices]
    support_global = radio["global_descriptors"][np.maximum(support_indices, 0)]
    global_cosine = fixed_candidate_support_global_context_cosine(
        query_global, support_global, view_valid
    )
    output_features = global_cosine[..., None].astype(np.float32, copy=False)
    if np.any(~np.isfinite(output_features[view_valid])) or np.any(
        output_features[~view_valid] != 0.0
    ):
        raise RuntimeError("support8 global-context candidate features are invalid")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": "per_view_fixed_maplet_support8_radio_final_global_cosine_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": True,
        "soft_global_context_factor": True,
        "global_context_usage": GLOBAL_CONTEXT_USAGE,
        "global_context_hard_retrieval_or_candidate_reselection": False,
        "global_context_descriptor_source": "radio_final_global_descriptors_pca64",
        "global_context_normalization": radio_metadata.get("normalization"),
        "global_context_pca_fit_scope": radio_metadata.get("pca_fit_scope"),
        "global_context_cache_sha256": file_sha256_short(radio_final_context_cache),
        "global_context_cache_source_manifest_sha256": radio_metadata.get(
            "source_image_manifest_sha256"
        ),
        "global_context_image_count": int(len(radio["image_ids"])),
        "global_context_query_image_count": int(len(set(layout["query_ids"].tolist()))),
        "global_context_fixed_support_image_count": int(len(set(support_ids[view_valid].tolist()))),
        "global_context_support_view_count": int(view_valid.shape[2]),
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": "fixed_maplet_coverage_rank_top8_v1",
        "maplet_support_index": str(maplet_support_index),
        "maplet_support_index_sha256": file_sha256_short(maplet_support_index),
        "maplet_support_index_format": maplet_metadata.get("format"),
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "proposals_sha256": proposals_sha256,
        "frozen_layout_features_sha256": _array_sha256_short(layout["candidate_features"]),
        "full_frozen_source_rows_sha256": _array_sha256_short(layout["source_row_indices"]),
        "is_complete_frozen_layout": True,
        "diagnostic_max_queries": 0,
        "diagnostic_max_rows": 0,
        "render": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_row_indices=layout["source_row_indices"],
        query_ids=layout["query_ids"],
        split_names=layout["split_names"],
        xy=layout["xy"],
        candidate_track_ids=layout["candidate_track_ids"],
        candidate_canonical_rows=layout["candidate_canonical_rows"],
        candidate_features=output_features,
        candidate_view_valid=view_valid,
        candidate_support_image_ids=support_ids,
        candidate_support_coverage_counts=coverage,
        feature_names=np.asarray(GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_fixed_maplet_support8_soft_global_context_probe_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES)),
        "support_view_count": int(view_valid.shape[2]),
        "valid_candidate_view_count": int(np.sum(view_valid)),
        "protocol": {
            "image_retrieval_or_submap_used": False,
            "hard_image_retrieval_or_candidate_reselection": False,
            "soft_global_context_factor": True,
            "pose_or_ground_truth_used": False,
            "render": False,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_global_context_support8_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        maplet_support_index=Path(args.maplet_support_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
