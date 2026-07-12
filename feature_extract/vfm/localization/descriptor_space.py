"""Descriptor-space manifests for query/landmark compatibility checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from feature_extract.vfm.landmark_feature_aggregation import TRACK_PROTOTYPE_BUILDER_VERSION


DESCRIPTOR_SPACE_VERSION = 3
PROTOTYPE_BUILDER_VERSION = TRACK_PROTOTYPE_BUILDER_VERSION
TRACK_IMAGE_OBSERVATION_SELECTION_V1: dict[str, Any] = {
    "version": "track_image_observation_selection_v1",
    "deduplicate_track_images": True,
    "key": ["track_id", "image_id"],
    "tie_break": ["reprojection_error", "point2d_idx", "xy_x", "xy_y"],
}
NO_VIEW_CLUSTERING_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "none",
    "max_prototypes_per_track": 1,
}
RADIO_PREPROCESSING_V1: dict[str, Any] = {
    "implementation": "feature_extract.tools.vfm.extract_tokens.RadioTokenExtractor.v1",
    "color_space": "RGB",
    "input_dtype": "float32",
    "input_range": "0_1",
    "resize_policy": "radio_nearest_supported_resolution",
    "resize_mode": "bilinear",
    "resize_align_corners": False,
}
OBSERVATION_SAMPLING_CONVENTION = "pixel_endpoint_to_token_endpoint_v1"

_PROJECTION_SPACE_KEYS = (
    "version",
    "checkpoint_sha256",
    "mapper_mode",
    "feature_key",
    "projection_source",
    "output_branch",
    "radio_model",
    "radio_model_version",
    "radio_intermediate_layer_index",
    "preprocessing",
    "feature_grid_stride",
    "input_channels",
    "sampling_mode",
    "sampling_convention",
    "sampling_align_corners",
    "descriptor_dimension",
    "normalization_mode",
)

_PROJECTED_OBSERVATION_REQUIRED_KEYS = (
    *_PROJECTION_SPACE_KEYS,
    "aggregation_method",
    "l2_normalize_observations",
    "prototype_builder_version",
    "prototype_builder_config",
    "view_clustering",
    "maplet_bank_version",
    "image_manifest_hash",
    "sfm_track_hash",
    "observation_selection",
    "projection_space_id",
    "descriptor_space_id",
)


def _canonical_hash(payload: Mapping[str, Any], *, length: int = 16) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def canonical_descriptor_space_id(manifest: Mapping[str, Any]) -> str:
    """Return a stable short id for a complete descriptor-bank manifest."""

    payload = {
        str(key): value
        for key, value in dict(manifest).items()
        if str(key) not in {"descriptor_space_id", "projection_space_id"}
    }
    return _canonical_hash(payload)


def canonical_projection_space_id(manifest: Mapping[str, Any]) -> str:
    """Hash only fields that must be identical for query and bank observations."""

    payload = {key: dict(manifest).get(key) for key in _PROJECTION_SPACE_KEYS}
    return _canonical_hash(payload)


def token_feature_source_config(token_manifest: Any, feature_key: str) -> dict[str, Any]:
    """Infer and validate one immutable feature-source description from a token manifest."""

    specs: set[tuple[str, str, int, int]] = set()
    records = tuple(getattr(token_manifest, "records", ()))
    if not records:
        raise ValueError("token manifest is empty")
    for record in records:
        matches = [layer for layer in record.layers if str(layer.name) == str(feature_key)]
        if len(matches) != 1:
            raise ValueError(
                f"feature_key={feature_key!r} must occur exactly once in every token record; "
                f"image_id={record.image_id!r} has {len(matches)}"
            )
        layer = matches[0]
        specs.add((str(layer.model), str(layer.layer), int(layer.channels), int(layer.stride)))
    if len(specs) != 1:
        raise ValueError(f"token manifest mixes feature source specifications for {feature_key!r}: {sorted(specs)!r}")
    model_version, layer_name, channels, stride = next(iter(specs))
    radio_model = "C-RADIO" if "radio" in model_version.lower() else model_version
    intermediate_layer: int | str | None = None if layer_name == "final" else layer_name
    return {
        "radio_model": radio_model,
        "radio_model_version": model_version,
        "radio_intermediate_layer_index": intermediate_layer,
        "preprocessing": dict(RADIO_PREPROCESSING_V1),
        "feature_grid_stride": int(stride),
        "input_channels": int(channels),
    }


def validate_projected_observation_descriptor_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject incomplete, stale, or internally inconsistent production manifests."""

    values = dict(manifest)
    errors: list[str] = []
    if int(values.get("version", -1)) != int(DESCRIPTOR_SPACE_VERSION):
        errors.append(f"version: expected {DESCRIPTOR_SPACE_VERSION}, got {values.get('version')!r}")
    for key in _PROJECTED_OBSERVATION_REQUIRED_KEYS:
        if key not in values or values.get(key) == "":
            errors.append(f"{key}: missing")
    if values.get("projection_source") != "projected_observation_full_map":
        errors.append(
            "projection_source: production landmark banks require 'projected_observation_full_map', "
            f"got {values.get('projection_source')!r}"
        )
    if values.get("observation_selection") != TRACK_IMAGE_OBSERVATION_SELECTION_V1:
        errors.append(
            "observation_selection: production landmark banks require deterministic track/image deduplication"
        )
    expected_projection_id = canonical_projection_space_id(values)
    if values.get("projection_space_id") != expected_projection_id:
        errors.append(
            f"projection_space_id: expected recomputed {expected_projection_id!r}, "
            f"got {values.get('projection_space_id')!r}"
        )
    expected_descriptor_id = canonical_descriptor_space_id(values)
    if values.get("descriptor_space_id") != expected_descriptor_id:
        errors.append(
            f"descriptor_space_id: expected recomputed {expected_descriptor_id!r}, "
            f"got {values.get('descriptor_space_id')!r}"
        )
    if errors:
        raise ValueError("invalid projected-observation descriptor manifest: " + "; ".join(errors))


def descriptor_space_manifest(
    *,
    checkpoint_sha256: str,
    mapper_mode: str,
    feature_key: str,
    projection_source: str,
    aggregation_method: str,
    l2_normalize_observations: bool,
    image_manifest_hash: str,
    sfm_track_hash: str,
    descriptor_dimension: int,
    normalization_mode: str,
    output_branch: str = "coarse_descriptors",
    radio_model: str = "C-RADIO",
    radio_model_version: str = "c-radio_v4-h",
    radio_intermediate_layer_index: int | str | None = None,
    preprocessing: Mapping[str, Any] | None = None,
    feature_grid_stride: int = 16,
    input_channels: int = 1280,
    sampling_mode: str = "bilinear",
    sampling_convention: str = OBSERVATION_SAMPLING_CONVENTION,
    sampling_align_corners: bool = True,
    prototype_builder_version: str = PROTOTYPE_BUILDER_VERSION,
    prototype_builder_config: Mapping[str, Any] | None = None,
    view_clustering: Mapping[str, Any] | None = None,
    maplet_bank_version: str = "none",
    observation_selection: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": int(DESCRIPTOR_SPACE_VERSION),
        "checkpoint_sha256": str(checkpoint_sha256),
        "mapper_mode": str(mapper_mode),
        "feature_key": str(feature_key),
        "projection_source": str(projection_source),
        "output_branch": str(output_branch),
        "radio_model": str(radio_model),
        "radio_model_version": str(radio_model_version),
        "radio_intermediate_layer_index": radio_intermediate_layer_index,
        "preprocessing": dict(preprocessing or RADIO_PREPROCESSING_V1),
        "feature_grid_stride": int(feature_grid_stride),
        "input_channels": int(input_channels),
        "sampling_mode": str(sampling_mode),
        "sampling_convention": str(sampling_convention),
        "sampling_align_corners": bool(sampling_align_corners),
        "aggregation_method": str(aggregation_method),
        "l2_normalize_observations": bool(l2_normalize_observations),
        "prototype_builder_version": str(prototype_builder_version),
        "prototype_builder_config": dict(prototype_builder_config or {}),
        "view_clustering": dict(view_clustering or NO_VIEW_CLUSTERING_CONFIG),
        "maplet_bank_version": str(maplet_bank_version),
        "image_manifest_hash": str(image_manifest_hash),
        "sfm_track_hash": str(sfm_track_hash),
        "observation_selection": dict(observation_selection),
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
    manifest["projection_space_id"] = canonical_projection_space_id(manifest)
    manifest["descriptor_space_id"] = canonical_descriptor_space_id(manifest)
    return manifest


def post_aggregate_1x1_descriptor_space_manifest(
    *,
    checkpoint_sha256: str,
    feature_key: str,
    raw_landmark_bank_sha256: str,
    sfm_track_hash: str,
    descriptor_dimension: int,
    normalization_mode: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": int(DESCRIPTOR_SPACE_VERSION),
        "checkpoint_sha256": str(checkpoint_sha256),
        "mapper_mode": "post_aggregate_1x1_projection_baseline",
        "feature_key": str(feature_key),
        "projection_source": "post_aggregate_1x1_projection_baseline",
        "output_branch": "coarse_descriptors",
        "radio_model": "unknown_diagnostic_input",
        "radio_model_version": "unknown_diagnostic_input",
        "radio_intermediate_layer_index": None,
        "preprocessing": {},
        "feature_grid_stride": 1,
        "input_channels": int(descriptor_dimension),
        "sampling_mode": "none",
        "sampling_convention": "post_aggregate_1x1_diagnostic",
        "sampling_align_corners": False,
        "raw_landmark_bank_sha256": str(raw_landmark_bank_sha256),
        "sfm_track_hash": str(sfm_track_hash),
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
    manifest["projection_space_id"] = canonical_projection_space_id(manifest)
    manifest["descriptor_space_id"] = canonical_descriptor_space_id(manifest)
    return manifest


def raw_descriptor_space_manifest(
    *,
    feature_key: str,
    descriptor_dimension: int,
    normalization_mode: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": int(DESCRIPTOR_SPACE_VERSION),
        "checkpoint_sha256": "",
        "mapper_mode": "raw_feature",
        "feature_key": str(feature_key),
        "projection_source": "raw_query_to_raw_aggregated_landmark",
        "output_branch": "raw_feature",
        "radio_model": "unknown_raw_input",
        "radio_model_version": "unknown_raw_input",
        "radio_intermediate_layer_index": None,
        "preprocessing": {},
        "feature_grid_stride": 1,
        "input_channels": int(descriptor_dimension),
        "sampling_mode": "unknown",
        "sampling_convention": "raw_diagnostic",
        "sampling_align_corners": False,
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
    manifest["projection_space_id"] = canonical_projection_space_id(manifest)
    manifest["descriptor_space_id"] = canonical_descriptor_space_id(manifest)
    return manifest
