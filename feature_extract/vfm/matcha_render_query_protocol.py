"""Protocol metadata helpers for render-query RADIO-MATCHA manifests."""

from __future__ import annotations

from typing import Mapping, Sequence


PAIR_SOURCE_2DGS_SYNTHETIC = "2dgs_synthetic"
PAIR_SOURCE_REAL_GT_RENDER = "real_gt_render"
PAIR_SOURCE_REAL_PERTURBED_RENDER = "real_perturbed_render"
PAIR_SOURCE_REAL_REFERENCE_RENDER = "real_reference_render"

RENDER_QUERY_PAIR_SOURCES = frozenset(
    {
        PAIR_SOURCE_2DGS_SYNTHETIC,
        PAIR_SOURCE_REAL_GT_RENDER,
        PAIR_SOURCE_REAL_PERTURBED_RENDER,
        PAIR_SOURCE_REAL_REFERENCE_RENDER,
    }
)

REAL_RENDER_QUERY_PAIR_SOURCES = frozenset(
    {
        PAIR_SOURCE_REAL_GT_RENDER,
        PAIR_SOURCE_REAL_PERTURBED_RENDER,
        PAIR_SOURCE_REAL_REFERENCE_RENDER,
    }
)

_REQUIRED_BASE_METADATA_FIELDS = (
    "pair_source",
    "source_query_manifest",
)

_REQUIRED_REAL_METADATA_FIELDS = (
    "query_pose_file",
    "train_query_count",
    "validation_query_count",
    "train_validation_query_overlap_count",
    "pair_type_counts",
)


def _as_query_id_set(query_ids: Sequence[str]) -> set[str]:
    return {str(query_id) for query_id in query_ids if str(query_id)}


def build_render_query_metadata(
    *,
    pair_source: str,
    source_query_manifest: str,
    query_pose_file: str,
    train_query_ids: Sequence[str],
    validation_query_ids: Sequence[str],
    pair_type_counts: Mapping[str, int],
    candidate_bank: str = "",
    pose_bin_policy: str = "",
) -> dict[str, object]:
    """Build validated metadata for real/synthetic render-query pair manifests."""

    train_ids = _as_query_id_set(train_query_ids)
    validation_ids = _as_query_id_set(validation_query_ids)
    metadata: dict[str, object] = {
        "pair_source": str(pair_source),
        "source_query_manifest": str(source_query_manifest),
        "query_pose_file": str(query_pose_file),
        "candidate_bank": str(candidate_bank),
        "pose_bin_policy": str(pose_bin_policy),
        "train_query_count": int(len(train_ids)),
        "validation_query_count": int(len(validation_ids)),
        "train_validation_query_overlap_count": int(len(train_ids & validation_ids)),
        "pair_type_counts": {str(key): int(value) for key, value in dict(pair_type_counts).items()},
    }
    validate_render_query_metadata(metadata)
    return metadata


def validate_render_query_metadata(metadata: Mapping[str, object]) -> None:
    """Reject ambiguous render-query metadata before training or reporting."""

    missing_base = [key for key in _REQUIRED_BASE_METADATA_FIELDS if key not in metadata]
    if missing_base:
        raise ValueError(f"render-query metadata is missing required field(s): {', '.join(missing_base)}")
    pair_source = str(metadata.get("pair_source", ""))
    if pair_source not in RENDER_QUERY_PAIR_SOURCES:
        raise ValueError(f"unsupported render-query pair_source: {pair_source}")
    if pair_source not in REAL_RENDER_QUERY_PAIR_SOURCES:
        if not str(metadata.get("source_query_manifest", "")):
            raise ValueError("render-query metadata field source_query_manifest must be non-empty")
        return
    missing_real = [key for key in _REQUIRED_REAL_METADATA_FIELDS if key not in metadata]
    if missing_real:
        raise ValueError(f"render-query metadata is missing required field(s): {', '.join(missing_real)}")
    if pair_source == PAIR_SOURCE_REAL_REFERENCE_RENDER and not str(metadata.get("candidate_bank", "")):
        raise ValueError("candidate_bank is required for real_reference_render manifests")
    for key in ("source_query_manifest", "query_pose_file"):
        if not str(metadata.get(key, "")):
            raise ValueError(f"render-query metadata field {key} must be non-empty")
    pair_type_counts = metadata.get("pair_type_counts")
    if not isinstance(pair_type_counts, Mapping) or not pair_type_counts:
        raise ValueError("render-query metadata field pair_type_counts must be a non-empty mapping")
    for key in ("train_query_count", "validation_query_count", "train_validation_query_overlap_count"):
        try:
            value = int(metadata.get(key, -1))
        except (TypeError, ValueError):
            raise ValueError(f"render-query metadata field {key} must be an integer") from None
        if value < 0:
            raise ValueError(f"render-query metadata field {key} must be non-negative")
