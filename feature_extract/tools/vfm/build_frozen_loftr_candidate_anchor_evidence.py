"""Export fixed-candidate LoFTR anchor evidence from a full mapping-pair cache.

The input LoFTR cache must have paired the query with every mapping image in
the frozen maplet manifest.  This exporter does not run image retrieval or
LoFTR again: it only attaches cache correspondences to the exact top-20
candidate tracks, their immutable support-view weights, and their SfM
observation anchors already present in a target-free S0 appearance shard.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _fixed_candidate_views,
    _load_maplet_support_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
    LOFTR_ANCHOR_FEATURE_NAMES,
    frozen_loftr_candidate_view_features,
)
from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    build_frozen_loftr_colmap_coordinate_contract,
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
from feature_extract.vfm.measurement_v1.rgb_data_contract import image_root_manifest


# See the pair-cache builder: provenance must identify the code loaded by a
# long-running worker, even if the source path is edited before it finishes.
_LOFTR_ANCHOR_EXPORTER_SOURCE_SHA256 = file_sha256_short(Path(__file__))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifact", required=True)
    parser.add_argument("--loftr-pair-cache", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--match-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _expected_input_sha(metadata: Mapping[str, Any], name: str, path: Path) -> None:
    inputs = metadata.get("inputs")
    expected = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(expected, Mapping) or str(expected.get("sha256", "")) != file_sha256_short(
        Path(path)
    ):
        raise ValueError(f"frozen appearance artifact has stale or mismatched {name} lineage")


def _validate_pair_cache_lineage(
    *,
    cache: Any,
    query_id: str,
    maplet_support_index: Path,
    image_root: Path,
    loftr_checkpoint: Path,
) -> None:
    metadata = dict(cache.metadata)
    if cache.query_id != str(query_id):
        raise ValueError("LoFTR pair cache query differs from frozen appearance artifact")
    maplet = metadata.get("maplet_support_index")
    matcher = metadata.get("matcher")
    source = metadata.get("image_source_contract")
    if (
        not isinstance(maplet, Mapping)
        or str(maplet.get("sha256", "")) != file_sha256_short(Path(maplet_support_index))
        or not isinstance(matcher, Mapping)
        or str(matcher.get("checkpoint_sha256", ""))
        != file_sha256_short(Path(loftr_checkpoint))
        or not isinstance(source, Mapping)
        or not isinstance(source.get("query"), Mapping)
        or not isinstance(source.get("mapping_support"), Mapping)
    ):
        raise ValueError("LoFTR pair cache lineage is incompatible with current inputs")
    current_query = image_root_manifest(Path(image_root), [str(query_id)])
    current_support = image_root_manifest(Path(image_root), cache.support_image_ids.tolist())
    if dict(source["query"]) != current_query or dict(source["mapping_support"]) != current_support:
        raise ValueError("LoFTR pair cache is stale for the current real-image source manifest")


def _aggregate_view_features(
    *, view_features: np.ndarray, view_usable: np.ndarray, view_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(view_features, dtype=np.float32)
    usable = np.asarray(view_usable, dtype=bool)
    weights = np.asarray(view_weights, dtype=np.float32)
    if (
        features.ndim != 4
        or usable.shape != features.shape[:3]
        or weights.shape != usable.shape
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(features[usable]))
    ):
        raise ValueError("LoFTR view aggregation inputs are invalid")
    masked_weights = weights * usable.astype(np.float32)
    mass = masked_weights.sum(axis=2)
    weighted = np.where(usable[..., None], features, 0.0)
    mean = (weighted * weights[..., None]).sum(axis=2) / mass[..., None].clip(1e-12)
    maximum = np.where(
        usable[..., None], features, np.full_like(features, -np.inf)
    ).max(axis=2)
    mean = np.where(mass[..., None] > 0.0, mean, np.full_like(mean, np.nan))
    maximum = np.where(mass[..., None] > 0.0, maximum, np.full_like(maximum, np.nan))
    return mean.astype(np.float32), maximum.astype(np.float32), mass.astype(np.float32)


def build_frozen_loftr_candidate_anchor_evidence(
    *,
    appearance_artifact: Path,
    loftr_pair_cache: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    colmap_model_dir: Path,
    image_root: Path,
    loftr_checkpoint: Path,
    match_chunk_size: int,
    device: str,
    output: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Attach full-bank cached LoFTR correspondences to frozen candidate views."""

    output_path = Path(output)
    summary_path = Path(summary_json)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite frozen LoFTR candidate evidence")
    if int(match_chunk_size) <= 0:
        raise ValueError("LoFTR match chunk size must be positive")
    device_value = torch.device(str(device))
    if device_value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LoFTR candidate evidence requested unavailable CUDA")
    features = load_frozen_appearance_probe_features([Path(appearance_artifact)])
    if len(features.query_ids) != 192 or len(set(features.query_ids.tolist())) != 1:
        raise ValueError("LoFTR candidate evidence requires one complete 192-row frozen query")
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
    coordinate_contract = build_frozen_loftr_colmap_coordinate_contract(
        cache_metadata=cache.metadata,
        query_id=query_id,
        support_image_ids=cache.support_image_ids.tolist(),
        colmap_model_dir=Path(colmap_model_dir),
    )
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
    query_xy_source = coordinate_contract.model_to_source(
        features.xy,
        image_ids=np.repeat(np.asarray([query_id]), len(features.xy)),
    )
    support_xy_source = np.full_like(runtime.support_xy, np.nan, dtype=np.float32)
    flat_valid = views.valid.reshape(-1)
    if np.any(flat_valid):
        support_xy_source.reshape(-1, 2)[flat_valid] = coordinate_contract.model_to_source(
            runtime.support_xy.reshape(-1, 2)[flat_valid],
            image_ids=views.support_image_ids.reshape(-1)[flat_valid],
        )
    started = time.monotonic()
    view_features, view_usable, pair_match_counts = frozen_loftr_candidate_view_features(
        cache=cache,
        query_xy=query_xy_source,
        candidate_support_xy=support_xy_source,
        candidate_support_image_ids=views.support_image_ids,
        candidate_view_valid=views.valid,
        chunk_size=int(match_chunk_size),
        device=device_value,
    )
    candidate_mean, candidate_maximum, usable_mass = _aggregate_view_features(
        view_features=view_features,
        view_usable=view_usable,
        view_weights=views.weights,
    )
    if (
        view_features.shape
        != (*features.candidate_view_weights.shape, len(LOFTR_ANCHOR_FEATURE_NAMES))
        or view_usable.shape != features.candidate_view_weights.shape
        or np.any(~np.isfinite(candidate_mean[usable_mass > 0.0]))
        or np.any(usable_mass < 0.0)
        or np.any(usable_mass > 1.0001)
    ):
        raise RuntimeError("frozen LoFTR candidate evidence outputs are invalid")
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
        "anchor_evidence_pose_free": True,
        "source_to_colmap_coordinate_contract": True,
        "legacy_coordinate_ambiguous_evidence_rejected": True,
        "missing_pair_matches": "explicit_nan_and_zero_usable_weight_mass_v1",
    }
    metadata: dict[str, Any] = {
        "format": FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
        "version": 2,
        "query_id": query_id,
        "split_name": next(iter(split_names)),
        "row_count": int(len(features.query_ids)),
        "contains_target_fields": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "strict_frozen_loftr_anchor_contract": strict_contract,
        "feature_field": "candidate_view_features",
        "feature_names": list(LOFTR_ANCHOR_FEATURE_NAMES),
        "fixed_candidate_top_k": int(features.candidate_track_ids.shape[1]),
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
            "colmap_model": coordinate_contract.metadata()["colmap_model"],
            "loftr_checkpoint": {
                "path": str(Path(loftr_checkpoint)),
                "sha256": file_sha256_short(Path(loftr_checkpoint)),
            },
        },
        "pair_cache_contract": cache.metadata["strict_global_pair_contract"],
        "coordinate_contract": coordinate_contract.metadata(),
        "support_geometry_metadata": geometry_metadata,
        "appearance_source_contract": appearance_metadata.get(
            "strict_frozen_appearance_contract"
        ),
        "runtime": {
            "device": str(device_value),
            "match_chunk_size": int(match_chunk_size),
            "elapsed_seconds": float(time.monotonic() - started),
        },
        "implementation": {
            "source_path": str(Path(__file__)),
            "source_sha256": _LOFTR_ANCHOR_EXPORTER_SOURCE_SHA256,
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
            verification_xy_loftr_source=query_xy_source,
            candidate_track_ids=features.candidate_track_ids,
            candidate_probabilities=features.candidate_probabilities,
            null_probabilities=features.null_probabilities,
            candidate_view_weights=features.candidate_view_weights,
            feature_names=np.asarray(LOFTR_ANCHOR_FEATURE_NAMES, dtype=np.str_),
            candidate_view_usable=view_usable,
            candidate_view_features=view_features,
            candidate_view_pair_match_counts=pair_match_counts,
            candidate_usable_view_weight_mass=usable_mass,
            candidate_coverage_weighted_features=candidate_mean,
            candidate_max_view_features=candidate_maximum,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    valid_pair_counts = pair_match_counts[view_usable]
    summary = {
        "stage": "build_frozen_loftr_candidate_anchor_evidence",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": next(iter(split_names)),
        "row_count": int(len(features.query_ids)),
        "feature_names": list(LOFTR_ANCHOR_FEATURE_NAMES),
        "view_coverage": {
            "valid_view_count": int(np.sum(view_usable)),
            "valid_view_rate_over_fixed_views": float(
                np.mean(view_usable[views.valid]) if np.any(views.valid) else 0.0
            ),
            "median_pair_match_count": None
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
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_frozen_loftr_candidate_anchor_evidence(
        appearance_artifact=Path(args.appearance_artifact),
        loftr_pair_cache=Path(args.loftr_pair_cache),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        colmap_model_dir=Path(args.colmap_model_dir),
        image_root=Path(args.image_root),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        match_chunk_size=int(args.match_chunk_size),
        device=str(args.device),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
