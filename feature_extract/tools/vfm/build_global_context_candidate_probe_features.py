"""Export a fixed-candidate soft global RADIO-context S1 probe artifact.

The artifact augments only each existing candidate/support-view pair with the
cosine between its query image and that fixed support image's RADIO-final
global descriptor.  It never searches images, changes the global top-L track
pool, reselects support views, or reads pose/target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES,
    GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT,
    fixed_candidate_support_global_context_cosine,
)


ARTIFACT_FORMAT = GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT
FEATURE_DEFINITION = (
    "per_view_fixed_candidate_support_image_radio_final_global_cosine_v1"
)
GLOBAL_CONTEXT_USAGE = GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT[ARTIFACT_FORMAT]
RADIO_FINAL_CONTEXT_FORMAT = "radio_final_context_pca_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _load_frozen_layout(
    path: Path,
    *,
    required_feature_names: Sequence[str] = GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
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
        "feature_names",
    }
    with np.load(path, allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError("frozen layout unexpectedly contains labels")
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"frozen layout lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="frozen layout")
    if metadata.get("contains_ground_truth") is not False or metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError("frozen layout is not target-free")
    if bool(metadata.get("image_retrieval_or_submap_used", True)) or bool(
        metadata.get("whole_image_summary_or_global_used", True)
    ):
        raise ValueError("frozen layout violates the local no-retrieval source protocol")
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    canonical = np.asarray(arrays["candidate_canonical_rows"], dtype=np.int64)
    features = np.asarray(arrays["candidate_features"], dtype=np.float32)
    view_valid = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    support_ids = np.asarray(arrays["candidate_support_image_ids"]).astype(str)
    coverage = np.asarray(arrays["candidate_support_coverage_counts"], dtype=np.int32)
    names = tuple(str(value) for value in arrays["feature_names"].tolist())
    if (
        len(rows) == 0
        or np.unique(rows).size != len(rows)
        or not (query_ids.shape == split_names.shape == (len(rows),))
        or xy.shape != (len(rows), 2)
        or tracks.ndim != 2
        or canonical.shape != tracks.shape
        or features.ndim != 4
        or view_valid.shape != features.shape[:3]
        or features.shape[:2] != tracks.shape
        or support_ids.shape != view_valid.shape
        or coverage.shape != view_valid.shape
        or features.shape[0] != len(rows)
    ):
        raise ValueError("frozen layout arrays are not aligned")
    required_names = tuple(str(value) for value in required_feature_names)
    if not required_names or len(set(required_names)) != len(required_names):
        raise ValueError("frozen layout required features are invalid")
    if len(set(names)) != len(names) or not set(required_names).issubset(names):
        raise ValueError("frozen layout lacks required anchor features")
    valid_candidates = tracks >= 0
    if np.any(valid_candidates & ~np.any(view_valid, axis=2)):
        raise ValueError("a valid candidate has no fixed support view")
    if np.any(~np.isfinite(features[view_valid])):
        raise ValueError("frozen-layout support features are non-finite")
    return {
        "source_row_indices": rows,
        "query_ids": query_ids,
        "split_names": split_names,
        "xy": xy,
        "candidate_track_ids": tracks,
        "candidate_canonical_rows": canonical,
        "candidate_features": features,
        "candidate_view_valid": view_valid,
        "candidate_support_image_ids": support_ids,
        "candidate_support_coverage_counts": coverage,
        "feature_names": np.asarray(names, dtype=np.str_),
    }, metadata


def _load_radio_final_global_descriptors(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {"image_ids", "global_descriptors"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"RADIO-final context cache lacks {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        metadata = _metadata(data, context="RADIO-final context cache")
    if metadata.get("format") != RADIO_FINAL_CONTEXT_FORMAT:
        raise ValueError("RADIO-final context cache format is unsupported")
    if metadata.get("pca_fit_scope") != "mapping_train_images_only":
        raise ValueError("RADIO-final global PCA was not fit on mapping-train images only")
    if metadata.get("pose_or_ground_truth_used") is not False or bool(
        metadata.get("image_retrieval_or_submap_used", True)
    ):
        raise ValueError("RADIO-final context cache violates the target-free protocol")
    image_ids = np.asarray(arrays["image_ids"]).astype(str).reshape(-1)
    descriptors = np.asarray(arrays["global_descriptors"], dtype=np.float32)
    if (
        len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or descriptors.ndim != 2
        or descriptors.shape[0] != len(image_ids)
        or descriptors.shape[1] == 0
        or np.any(~np.isfinite(descriptors))
    ):
        raise ValueError("RADIO-final global descriptor cache is invalid")
    norms = np.linalg.norm(descriptors, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("RADIO-final global descriptor cache has zero-norm rows")
    return {"image_ids": image_ids, "global_descriptors": descriptors}, metadata


def _descriptor_indices(
    *,
    image_ids: np.ndarray,
    query_ids: np.ndarray,
    support_ids: np.ndarray,
    view_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
    try:
        query_indices = np.asarray([lookup[str(image_id)] for image_id in query_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"query image has no RADIO-final global descriptor: {error.args[0]}") from error
    support_indices = np.full(support_ids.shape, -1, dtype=np.int64)
    for image_id in np.unique(support_ids[view_valid]).tolist():
        index = lookup.get(str(image_id))
        if index is None:
            raise ValueError(
                f"fixed candidate support image has no RADIO-final global descriptor: {image_id}"
            )
        support_indices[support_ids == str(image_id)] = int(index)
    if np.any(support_indices[view_valid] < 0):
        raise RuntimeError("valid support images were not indexed")
    return query_indices, support_indices


def build_global_context_candidate_probe_features(
    *,
    frozen_layout_features: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite global-context probe outputs")
    layout, layout_metadata = _load_frozen_layout(Path(frozen_layout_features))
    radio, radio_metadata = _load_radio_final_global_descriptors(Path(radio_final_context_cache))
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposals_sha256 lineage")
    view_valid = np.asarray(layout["candidate_view_valid"], dtype=bool)
    query_indices, support_indices = _descriptor_indices(
        image_ids=radio["image_ids"],
        query_ids=layout["query_ids"],
        support_ids=layout["candidate_support_image_ids"],
        view_valid=view_valid,
    )
    query_global = radio["global_descriptors"][query_indices]
    safe_support_indices = np.maximum(support_indices, 0)
    support_global = radio["global_descriptors"][safe_support_indices]
    global_cosine = fixed_candidate_support_global_context_cosine(
        query_global, support_global, view_valid
    )
    source_feature_names = tuple(str(value) for value in layout["feature_names"].tolist())
    source_features = np.asarray(layout["candidate_features"], dtype=np.float32)
    output_features = np.zeros(
        (*view_valid.shape, len(GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES)), dtype=np.float32
    )
    for output_index, feature_name in enumerate(
        GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    ):
        source_index = source_feature_names.index(feature_name)
        output_features[..., output_index] = np.where(
            view_valid, source_features[..., source_index], 0.0
        )
    output_features[..., -1] = global_cosine
    if np.any(~np.isfinite(output_features[view_valid])):
        raise RuntimeError("global-context candidate features are non-finite")
    if np.any(output_features[~view_valid] != 0.0):
        raise RuntimeError("invalid support views must have zero global-context features")
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "feature_definition": FEATURE_DEFINITION,
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
        "global_context_fixed_support_image_count": int(
            len(set(layout["candidate_support_image_ids"][view_valid].tolist()))
        ),
        "candidate_set": "frozen_global_top20_tracks",
        "support_view_selection": layout_metadata.get("support_view_selection"),
        "source_frozen_layout": str(frozen_layout_features),
        "source_frozen_layout_sha256": file_sha256_short(frozen_layout_features),
        "proposals_sha256": proposals_sha256,
        "source_candidate_artifact_sha256": layout_metadata.get("candidate_artifact_sha256"),
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
        candidate_support_image_ids=layout["candidate_support_image_ids"],
        candidate_support_coverage_counts=layout["candidate_support_coverage_counts"],
        feature_names=np.asarray(GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_fixed_candidate_soft_global_context_probe_features",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": ARTIFACT_FORMAT,
        "row_count": int(len(layout["source_row_indices"])),
        "feature_count": int(len(GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES)),
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
    summary = build_global_context_candidate_probe_features(
        frozen_layout_features=Path(args.frozen_layout_features),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
