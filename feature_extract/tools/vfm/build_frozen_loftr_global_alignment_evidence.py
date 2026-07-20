"""Export frozen candidate-specific global LoFTR alignment evidence.

This reads an already-complete query-to-all-mapping-image LoFTR cache.  It
fits one deterministic support-to-query homography for each support image
needed by the immutable maplet views, then evaluates the existing SfM support
observation anchors under that image-level model.  It never accesses labels,
poses, 3-D projections, image retrieval, or a candidate/support reselection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_frozen_loftr_candidate_anchor_evidence import (
    _aggregate_view_features,
    _expected_input_sha,
    _validate_pair_cache_lineage,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _fixed_candidate_views,
    _load_maplet_support_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.frozen_loftr_global_alignment import (
    FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT,
    LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES,
    FrozenLoFTRHomographyConfig,
    frozen_loftr_global_alignment_view_features,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    load_frozen_loftr_pair_cache,
)
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    load_frozen_appearance_probe_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)


_LOFTR_GLOBAL_ALIGNMENT_EXPORTER_SOURCE_SHA256 = file_sha256_short(Path(__file__))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifact", required=True)
    parser.add_argument("--loftr-pair-cache", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _source_size_from_cache_metadata(metadata: Mapping[str, Any]) -> tuple[int, int]:
    """Require one identical original pixel grid for query and mapping images."""

    source = metadata.get("image_source_contract")
    if not isinstance(source, Mapping):
        raise ValueError("LoFTR cache has no source-image contract")
    dimension_keys: list[str] = []
    for name in ("query", "mapping_support"):
        item = source.get(name)
        values = item.get("source_image_dimensions") if isinstance(item, Mapping) else None
        if not isinstance(values, Mapping) or len(values) != 1:
            raise ValueError("LoFTR global alignment requires one declared source image size")
        key, count = next(iter(values.items()))
        if not isinstance(key, str) or int(count) <= 0:
            raise ValueError("LoFTR source image dimension declaration is invalid")
        dimension_keys.append(key)
    if dimension_keys[0] != dimension_keys[1]:
        raise ValueError("query and mapping source image grids differ")
    key = dimension_keys[0]
    if "x" not in key:
        raise ValueError("LoFTR source image dimension declaration is invalid")
    width_text, height_text = key.lower().split("x", 1)
    try:
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise ValueError("LoFTR source image dimension declaration is malformed") from error
    if width <= 0 or height <= 0:
        raise ValueError("LoFTR source image dimensions must be positive")
    return width, height


def build_frozen_loftr_global_alignment_evidence(
    *,
    appearance_artifact: Path,
    loftr_pair_cache: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    image_root: Path,
    loftr_checkpoint: Path,
    output: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Attach image-pair global alignment features to fixed candidate views."""

    output_path = Path(output)
    summary_path = Path(summary_json)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite frozen LoFTR global alignment evidence")
    features = load_frozen_appearance_probe_features([Path(appearance_artifact)])
    if len(features.query_ids) != 192 or len(set(features.query_ids.tolist())) != 1:
        raise ValueError("global alignment evidence requires one complete 192-row frozen query")
    query_id = str(features.query_ids[0])
    split_names = set(features.split_names.tolist())
    if len(split_names) != 1:
        raise ValueError("frozen appearance artifact has inconsistent split identity")
    appearance_metadata = dict(features.metadata)
    _expected_input_sha(appearance_metadata, "maplet_support_index", Path(maplet_support_index))
    _expected_input_sha(appearance_metadata, "support_geometry_index", Path(support_geometry_index))
    cache = load_frozen_loftr_pair_cache(Path(loftr_pair_cache))
    _validate_pair_cache_lineage(
        cache=cache,
        query_id=query_id,
        maplet_support_index=Path(maplet_support_index),
        image_root=Path(image_root),
        loftr_checkpoint=Path(loftr_checkpoint),
    )
    source_size = _source_size_from_cache_metadata(cache.metadata)
    maplet_tracks, maplet_image_ids, maplet_image_indices, maplet_coverage, _maplet_metadata = (
        _load_maplet_support_fields(Path(maplet_support_index))
    )
    views = _fixed_candidate_views(
        candidate_track_ids=features.candidate_track_ids,
        candidate_probabilities=features.candidate_probabilities,
        maplet_track_ids=maplet_tracks,
        support_image_ids=maplet_image_ids,
        support_image_indices=maplet_image_indices,
        support_coverage_counts=maplet_coverage,
    )
    if not np.array_equal(views.weights, features.candidate_view_weights):
        raise ValueError("frozen appearance candidate view weights differ from maplet fixed mixture")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    cache_image_ids = np.asarray([query_id, *cache.support_image_ids.tolist()], dtype=np.str_)
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.full((len(features.query_ids),), query_id),
        query_xy=features.xy,
        candidate_track_ids=features.candidate_track_ids,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        cache_image_ids=cache_image_ids,
        support_geometry=geometry,
    )
    config = FrozenLoFTRHomographyConfig()
    started = time.monotonic()
    view_features, view_usable, pair_match_counts, view_model_valid = (
        frozen_loftr_global_alignment_view_features(
            cache=cache,
            query_xy=features.xy,
            candidate_support_xy=runtime.support_xy,
            candidate_support_image_ids=views.support_image_ids,
            candidate_view_valid=views.valid,
            source_size=source_size,
            config=config,
        )
    )
    candidate_mean, _candidate_maximum, usable_mass = _aggregate_view_features(
        view_features=view_features,
        view_usable=view_usable,
        view_weights=views.weights,
    )
    if (
        view_features.shape
        != (*features.candidate_view_weights.shape, len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES))
        or view_usable.shape != features.candidate_view_weights.shape
        or view_model_valid.shape != features.candidate_view_weights.shape
        or np.any(~np.isfinite(candidate_mean[usable_mass > 0.0]))
        or np.any(usable_mass < 0.0)
        or np.any(usable_mass > 1.0001)
    ):
        raise RuntimeError("frozen LoFTR global alignment outputs are invalid")
    strict_contract = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "support_view_descriptor_averaging": False,
        "all_mapping_images_pair_cached": True,
        "pair_cache_image_level_selection": False,
        "global_alignment_pose_free": True,
        "global_alignment_model_per_candidate": False,
        "homography_fits_all_cached_pair_matches": True,
        "missing_pair_or_model": "explicit_nan_and_zero_usable_weight_mass_v1",
    }
    metadata: dict[str, Any] = {
        "format": FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT,
        "version": 1,
        "query_id": query_id,
        "split_name": next(iter(split_names)),
        "row_count": int(len(features.query_ids)),
        "contains_target_fields": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "strict_frozen_loftr_global_alignment_contract": strict_contract,
        "feature_field": "candidate_view_features",
        "feature_names": list(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES),
        "fixed_candidate_top_k": int(features.candidate_track_ids.shape[1]),
        "global_alignment_config": config.metadata(),
        "source_image_size": {"width": int(source_size[0]), "height": int(source_size[1])},
        "inputs": {
            "appearance_artifact": {
                "path": str(Path(appearance_artifact)),
                "sha256": file_sha256_short(Path(appearance_artifact)),
            },
            "loftr_pair_cache": {
                "path": str(Path(loftr_pair_cache)),
                "sha256": file_sha256_short(Path(loftr_pair_cache)),
            },
            "maplet_support_index": {
                "path": str(Path(maplet_support_index)),
                "sha256": file_sha256_short(Path(maplet_support_index)),
            },
            "support_geometry_index": {
                "path": str(Path(support_geometry_index)),
                "sha256": file_sha256_short(Path(support_geometry_index)),
            },
            "loftr_checkpoint": {
                "path": str(Path(loftr_checkpoint)),
                "sha256": file_sha256_short(Path(loftr_checkpoint)),
            },
        },
        "pair_cache_contract": cache.metadata["strict_global_pair_contract"],
        "support_geometry_metadata": geometry_metadata,
        "appearance_source_contract": appearance_metadata.get(
            "strict_frozen_appearance_contract"
        ),
        "runtime": {
            "elapsed_seconds": float(time.monotonic() - started),
            "source_size": [int(source_size[0]), int(source_size[1])],
        },
        "implementation": {
            "source_path": str(Path(__file__)),
            "source_sha256": _LOFTR_GLOBAL_ALIGNMENT_EXPORTER_SOURCE_SHA256,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            verification_query_ids=features.query_ids,
            split_names=features.split_names,
            verification_source_row_indices=features.source_row_indices,
            verification_xy=features.xy,
            candidate_track_ids=features.candidate_track_ids,
            candidate_probabilities=features.candidate_probabilities,
            null_probabilities=features.null_probabilities,
            candidate_view_weights=features.candidate_view_weights,
            feature_names=np.asarray(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES, dtype=np.str_),
            candidate_view_usable=view_usable,
            candidate_view_features=view_features,
            candidate_view_pair_match_counts=pair_match_counts,
            candidate_view_homography_model_valid=view_model_valid,
            candidate_usable_view_weight_mass=usable_mass,
            candidate_coverage_weighted_features=candidate_mean,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    valid_pair_counts = pair_match_counts[view_model_valid]
    summary = {
        "stage": "build_frozen_loftr_global_alignment_evidence",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": next(iter(split_names)),
        "row_count": int(len(features.query_ids)),
        "feature_names": list(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES),
        "view_coverage": {
            "fixed_view_count": int(np.sum(views.valid)),
            "homography_model_valid_view_count": int(np.sum(view_model_valid)),
            "anchor_usable_view_count": int(np.sum(view_usable)),
            "model_valid_rate_over_fixed_views": float(
                np.mean(view_model_valid[views.valid]) if np.any(views.valid) else 0.0
            ),
            "anchor_usable_rate_over_fixed_views": float(
                np.mean(view_usable[views.valid]) if np.any(views.valid) else 0.0
            ),
            "median_pair_match_count_for_valid_model": None
            if len(valid_pair_counts) == 0
            else float(np.median(valid_pair_counts)),
        },
        "elapsed_seconds": metadata["runtime"]["elapsed_seconds"],
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "full_mapping_pair_cache": True,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_or_ground_truth_used": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_frozen_loftr_global_alignment_evidence(
        appearance_artifact=Path(args.appearance_artifact),
        loftr_pair_cache=Path(args.loftr_pair_cache),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        image_root=Path(args.image_root),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
