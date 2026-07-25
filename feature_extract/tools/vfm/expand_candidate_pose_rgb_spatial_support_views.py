"""Expand fixed support coverage for an already frozen target-free P1 layout.

This utility is intentionally narrower than rebuilding a P1 layout.  It keeps
the selected query points, top-L candidates, priors, and first support slots
from a reference layout byte-identical, then appends further coverage-ranked
SfM support observations from the complete frozen source.  It never reads
poses, residuals, identities, or train targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_layout import (
    _array_sha256_short,
    _load_frozen_layout,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    _truncate_resolved_rgb_support_views,
    load_candidate_pose_rgb_spatial_layout,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-layout", required=True)
    parser.add_argument("--frozen-layout", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--support-views-per-candidate", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


_REFERENCE_METADATA_REQUIRED = {
    "contains_ground_truth": False,
    "contains_target_errors": False,
    "pose_or_ground_truth_used": False,
    "image_retrieval_or_submap_used": False,
    "render": False,
    "candidate_reselection": False,
}


def _require_target_free_reference(layout: CandidatePoseRGBSpatialLayout) -> None:
    if not isinstance(layout, CandidatePoseRGBSpatialLayout):
        raise ValueError("support expansion requires a validated P1 layout")
    metadata = layout.metadata
    if any(metadata.get(key) is not value for key, value in _REFERENCE_METADATA_REQUIRED.items()):
        raise ValueError("support expansion reference layout is not target-free")
    if int(layout.candidate_count) <= 0 or int(layout.support_view_count) <= 0:
        raise ValueError("support expansion reference layout is empty")


def _lookup_source_rows(
    *, source_row_indices: np.ndarray, requested_source_ids: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_row_indices, dtype=np.int64).reshape(-1)
    requested = np.asarray(requested_source_ids, dtype=np.int64).reshape(-1)
    if len(source) == 0 or len(np.unique(source)) != len(source):
        raise ValueError("complete frozen layout source IDs are invalid")
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    positions = np.searchsorted(sorted_source, requested)
    clipped = np.clip(positions, 0, len(sorted_source) - 1)
    if np.any(positions >= len(sorted_source)) or np.any(sorted_source[clipped] != requested):
        raise ValueError("reference layout point is absent from the complete frozen layout")
    return order[positions]


def _validate_reference_matches_complete_source(
    *, reference: CandidatePoseRGBSpatialLayout, source: Mapping[str, np.ndarray], rows: np.ndarray
) -> None:
    expected = {
        "query_ids": np.asarray(reference.query_ids).astype(str),
        "split_names": np.asarray(reference.split_names).astype(str),
        "xy": np.asarray(reference.xy, dtype=np.float32),
        "candidate_track_ids": np.asarray(reference.candidate_track_ids, dtype=np.int64),
        "candidate_canonical_rows": np.asarray(reference.candidate_bank_rows, dtype=np.int64),
    }
    for field, values in expected.items():
        observed = np.asarray(source[field])[rows]
        if field == "xy":
            equal = np.allclose(observed, values, rtol=0.0, atol=1e-4)
        else:
            equal = np.array_equal(observed, values)
        if not equal:
            raise ValueError(f"reference layout differs from complete frozen layout: {field}")


def _validate_reference_support_prefix(
    *,
    reference: CandidatePoseRGBSpatialLayout,
    support_image_ids: np.ndarray,
    support_xy: np.ndarray,
    support_view_valid: np.ndarray,
    support_coverage_counts: np.ndarray,
) -> None:
    """Require the old support slots to remain an ordered prefix exactly."""

    prefix = int(reference.support_view_count)
    candidate_shape = (int(reference.row_count), int(reference.candidate_count))
    values = {
        "support_image_ids": np.asarray(support_image_ids),
        "support_xy": np.asarray(support_xy),
        "support_view_valid": np.asarray(support_view_valid),
        "support_coverage_counts": np.asarray(support_coverage_counts),
    }
    if any(value.shape[:2] != candidate_shape or value.shape[2] < prefix for value in values.values()):
        raise ValueError("expanded support layout has an invalid candidate/view shape")
    for field, value in values.items():
        expected = np.asarray(getattr(reference, field))
        observed = value[:, :, :prefix]
        if not np.array_equal(observed, expected):
            raise ValueError(f"expanded support layout changes ordered {field} prefix")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def expand_candidate_pose_rgb_spatial_support_views(
    *,
    reference_layout: Path,
    frozen_layout: Path,
    support_geometry_index: Path,
    output: Path,
    summary_json: Path,
    support_views_per_candidate: int,
    force: bool,
) -> dict[str, Any]:
    """Append fixed coverage-ranked support observations to one frozen P1 subset."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite expanded support-layout outputs")
    if int(support_views_per_candidate) <= 0:
        raise ValueError("expanded support-view count must be positive")
    reference = load_candidate_pose_rgb_spatial_layout(Path(reference_layout))
    _require_target_free_reference(reference)
    if int(support_views_per_candidate) <= int(reference.support_view_count):
        raise ValueError("expanded support-view count must exceed the reference count")
    source, source_metadata = _load_frozen_layout(Path(frozen_layout))
    source_rows = _lookup_source_rows(
        source_row_indices=np.asarray(source["source_row_indices"], dtype=np.int64),
        requested_source_ids=np.asarray(reference.source_point_ids, dtype=np.int64),
    )
    _validate_reference_matches_complete_source(
        reference=reference,
        source=source,
        rows=source_rows,
    )
    source_ids = np.asarray(source["candidate_support_image_ids"])[source_rows].astype(str)
    source_valid = np.asarray(source["candidate_view_valid"], dtype=bool)[source_rows]
    source_coverage = np.asarray(source["candidate_support_coverage_counts"], dtype=np.int32)[
        source_rows
    ]
    if int(support_views_per_candidate) > int(source_ids.shape[2]):
        raise ValueError("complete frozen layout lacks the requested support-view count")
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(Path(support_geometry_index))
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support expansion requires real SfM support observation xy")
    cache_image_ids = np.unique(
        np.concatenate((np.asarray(reference.query_ids).astype(str), source_ids[source_valid]))
    )
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.asarray(reference.query_ids).astype(str),
        query_xy=np.asarray(reference.xy, dtype=np.float32),
        candidate_track_ids=np.asarray(reference.candidate_track_ids, dtype=np.int64),
        candidate_support_image_ids=source_ids,
        candidate_view_valid=source_valid,
        cache_image_ids=cache_image_ids,
        support_geometry=geometry,
    )
    selected_ids, selected_xy, selected_valid, selected_weights, selected_coverage = (
        _truncate_resolved_rgb_support_views(
            query_ids=np.asarray(reference.query_ids).astype(str),
            support_image_ids=source_ids,
            support_xy=runtime.support_xy,
            support_view_valid=runtime.view_valid,
            support_coverage_counts=source_coverage,
            support_views_per_candidate=int(support_views_per_candidate),
        )
    )
    _validate_reference_support_prefix(
        reference=reference,
        support_image_ids=selected_ids,
        support_xy=selected_xy,
        support_view_valid=selected_valid,
        support_coverage_counts=selected_coverage,
    )
    metadata = {
        **dict(reference.metadata),
        "support_views_per_candidate": int(support_views_per_candidate),
        "support_view_weight_semantics": "selected_maplet_coverage_normalized_v1",
        "support_view_expansion": {
            "format": "frozen_target_free_support_view_prefix_expansion_v1",
            "reference_layout": str(Path(reference_layout).resolve()),
            "reference_layout_sha256": file_sha256_short(Path(reference_layout)),
            "complete_frozen_layout": str(Path(frozen_layout).resolve()),
            "complete_frozen_layout_sha256": file_sha256_short(Path(frozen_layout)),
            "support_geometry_index": str(Path(support_geometry_index).resolve()),
            "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
            "reference_support_view_count": int(reference.support_view_count),
            "expanded_support_view_count": int(support_views_per_candidate),
            "reference_support_prefix_verified": True,
            "source_target_arrays_read": False,
        },
        "point_candidate_support_sha256": _array_sha256_short(
            np.asarray(reference.source_point_ids, dtype=np.int64),
            np.asarray(reference.candidate_track_ids, dtype=np.int64),
            selected_ids,
            selected_xy,
        ),
    }
    expanded = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray(reference.source_point_ids, dtype=np.int64),
        query_ids=np.asarray(reference.query_ids).astype(str),
        split_names=np.asarray(reference.split_names).astype(str),
        xy=np.asarray(reference.xy, dtype=np.float32),
        point_sources=np.asarray(reference.point_sources).astype(str),
        candidate_track_ids=np.asarray(reference.candidate_track_ids, dtype=np.int64),
        candidate_bank_rows=np.asarray(reference.candidate_bank_rows, dtype=np.int64),
        candidate_coarse_similarities=np.asarray(reference.candidate_coarse_similarities, dtype=np.float32),
        candidate_prior_probabilities=np.asarray(reference.candidate_prior_probabilities, dtype=np.float32),
        null_probabilities=np.asarray(reference.null_probabilities, dtype=np.float32),
        support_image_ids=selected_ids,
        support_xy=selected_xy,
        support_view_valid=selected_valid,
        support_view_weights=selected_weights,
        support_coverage_counts=selected_coverage,
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_layout(expanded, output)
    summary = {
        "stage": "expand_candidate_pose_rgb_spatial_support_views",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "reference_layout": str(Path(reference_layout)),
        "reference_layout_sha256": file_sha256_short(Path(reference_layout)),
        "row_count": int(expanded.row_count),
        "candidate_top_k": int(expanded.candidate_count),
        "reference_support_view_count": int(reference.support_view_count),
        "support_views_per_candidate": int(expanded.support_view_count),
        "valid_support_view_count": int(np.count_nonzero(expanded.support_view_valid)),
        "protocol": {
            "source_target_arrays_read": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "candidate_reselection": False,
            "ordered_reference_support_prefix_verified": True,
        },
    }
    _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = expand_candidate_pose_rgb_spatial_support_views(
        reference_layout=Path(args.reference_layout),
        frozen_layout=Path(args.frozen_layout),
        support_geometry_index=Path(args.support_geometry_index),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        support_views_per_candidate=int(args.support_views_per_candidate),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
