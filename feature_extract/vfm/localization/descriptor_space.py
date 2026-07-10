"""Descriptor-space manifests for query/landmark compatibility checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


DESCRIPTOR_SPACE_VERSION = 1


def canonical_descriptor_space_id(manifest: Mapping[str, Any]) -> str:
    """Return a stable short id for a descriptor-space manifest."""

    payload = {str(key): value for key, value in dict(manifest).items() if str(key) != "descriptor_space_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": int(DESCRIPTOR_SPACE_VERSION),
        "checkpoint_sha256": str(checkpoint_sha256),
        "mapper_mode": str(mapper_mode),
        "feature_key": str(feature_key),
        "projection_source": str(projection_source),
        "aggregation_method": str(aggregation_method),
        "l2_normalize_observations": bool(l2_normalize_observations),
        "image_manifest_hash": str(image_manifest_hash),
        "sfm_track_hash": str(sfm_track_hash),
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
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
        "raw_landmark_bank_sha256": str(raw_landmark_bank_sha256),
        "sfm_track_hash": str(sfm_track_hash),
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
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
        "descriptor_dimension": int(descriptor_dimension),
        "normalization_mode": str(normalization_mode),
    }
    manifest["descriptor_space_id"] = canonical_descriptor_space_id(manifest)
    return manifest
