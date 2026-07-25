"""Build a strict raw-RGB spatial-evidence layout from frozen P1 candidates.

The output is target-free.  It binds every current verification point and its
fixed global top-L tracks to real SfM support observations before RGB evidence
is trained or scored.  Ground-truth poses and residuals are deliberately not
read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
    CandidatePoseRGBSpatialLayout,
    _truncate_resolved_rgb_support_views,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    load_mixed_verification_points,
)


_FROZEN_LAYOUT_FORMAT = "mixed_multiscale_verification_frozen_candidate_layout_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--frozen_layout", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--support_views_per_candidate", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _metadata(payload: Any, *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        raise ValueError(f"{context} lacks metadata_json")
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata is invalid")
    return metadata


def _load_frozen_layout(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"frozen RGB layout source lacks {sorted(missing)}")
        metadata = _metadata(payload, context="frozen RGB layout source")
        arrays = {field: np.asarray(payload[field]).copy() for field in required}
    if (
        metadata.get("format") != _FROZEN_LAYOUT_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("is_complete_frozen_layout") is not True
    ):
        raise ValueError("frozen RGB layout source is not target-free")
    return arrays, metadata


def _array_sha256_short(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()[:16]


def _validate_layout_matches_points(
    *, layout: dict[str, np.ndarray], layout_metadata: dict[str, Any], points: Any, points_path: Path
) -> None:
    points_sha = file_sha256_short(Path(points_path))
    expected_pairs = (
        ("source_row_indices", np.asarray(points.source_point_ids, dtype=np.int64)),
        ("query_ids", np.asarray(points.query_ids).astype(str)),
        ("split_names", np.asarray(points.split_names).astype(str)),
        ("candidate_track_ids", np.asarray(points.candidate_track_ids, dtype=np.int64)),
        ("candidate_canonical_rows", np.asarray(points.candidate_bank_rows, dtype=np.int64)),
    )
    if str(layout_metadata.get("verification_points_sha256", "")) != points_sha:
        raise ValueError("frozen RGB layout source references a different verification-point artifact")
    for field, expected in expected_pairs:
        actual = np.asarray(layout[field])
        if not np.array_equal(actual, expected):
            raise ValueError(f"frozen RGB layout source differs from verification points: {field}")
    if not np.allclose(
        np.asarray(layout["xy"], dtype=np.float32),
        np.asarray(points.xy, dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("frozen RGB layout source differs from verification points: xy")
    if str(layout_metadata.get("projected_landmark_bank_sha256", "")) != str(
        points.metadata.get("projected_landmark_bank_sha256", "")
    ):
        raise ValueError("frozen RGB layout source has different landmark-bank lineage")


def build_candidate_pose_rgb_spatial_layout(
    *,
    verification_points: Path,
    frozen_layout: Path,
    support_geometry_index: Path,
    output: Path,
    summary_json: Path,
    support_views_per_candidate: int,
    force: bool,
) -> dict[str, Any]:
    """Write one immutable target-free P1 RGB support layout."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite RGB spatial layout outputs")
    if int(support_views_per_candidate) <= 0:
        raise ValueError("support views per candidate must be positive")
    points = load_mixed_verification_points(Path(verification_points))
    if (
        points.metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
        or points.metadata.get("contains_ground_truth") is not False
        or points.metadata.get("contains_target_errors") is not False
        or points.metadata.get("pose_or_ground_truth_used") is not False
        or points.metadata.get("image_retrieval_or_submap_used") is not False
        or points.metadata.get("render") is not False
    ):
        raise ValueError("verification points are not target-free")
    source_layout, source_metadata = _load_frozen_layout(Path(frozen_layout))
    _validate_layout_matches_points(
        layout=source_layout,
        layout_metadata=source_metadata,
        points=points,
        points_path=Path(verification_points),
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("RGB spatial layout requires real SfM support observation xy")
    valid_views = np.asarray(source_layout["candidate_view_valid"], dtype=bool)
    support_ids = np.asarray(source_layout["candidate_support_image_ids"]).astype(str)
    cache_image_ids = np.unique(
        np.concatenate(
            (
                np.asarray(points.query_ids).astype(str),
                support_ids[valid_views],
            )
        )
    )
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.asarray(points.query_ids).astype(str),
        query_xy=np.asarray(points.xy, dtype=np.float32),
        candidate_track_ids=np.asarray(points.candidate_track_ids, dtype=np.int64),
        candidate_support_image_ids=support_ids,
        candidate_view_valid=valid_views,
        cache_image_ids=cache_image_ids,
        support_geometry=geometry,
    )
    (
        selected_ids,
        selected_xy,
        selected_valid,
        selected_weights,
        selected_coverage,
    ) = _truncate_resolved_rgb_support_views(
        query_ids=np.asarray(points.query_ids).astype(str),
        support_image_ids=support_ids,
        support_xy=runtime.support_xy,
        support_view_valid=runtime.view_valid,
        support_coverage_counts=np.asarray(
            source_layout["candidate_support_coverage_counts"], dtype=np.int32
        ),
        support_views_per_candidate=int(support_views_per_candidate),
    )
    exported_splits = sorted(set(np.asarray(points.split_names).astype(str).tolist()))
    metadata: dict[str, Any] = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_set": "fixed_full_global_faiss_top_l_tracks",
        "candidate_reselection": False,
        "candidate_top_k": int(points.candidate_track_ids.shape[1]),
        "candidate_prior_semantics": points.metadata.get("candidate_prior_semantics"),
        "candidate_prior_pose_calibrated": points.metadata.get(
            "candidate_prior_pose_calibrated"
        ),
        "support_view_selection": "fixed_maplet_coverage_rank_excluding_query_image_v1",
        "support_view_weight_semantics": "selected_maplet_coverage_normalized_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "support_views_per_candidate": int(support_views_per_candidate),
        "verification_points": str(Path(verification_points).resolve()),
        "verification_points_sha256": file_sha256_short(Path(verification_points)),
        "frozen_layout": str(Path(frozen_layout).resolve()),
        "frozen_layout_sha256": file_sha256_short(Path(frozen_layout)),
        "maplet_support_index": source_metadata.get("maplet_support_index"),
        "maplet_support_index_sha256": source_metadata.get("maplet_support_index_sha256"),
        "support_geometry_index": str(Path(support_geometry_index).resolve()),
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "projected_landmark_bank_sha256": points.metadata.get(
            "projected_landmark_bank_sha256"
        ),
        "projection_space_id": points.metadata.get("projection_space_id"),
        "descriptor_space_id": points.metadata.get("descriptor_space_id"),
        "matcha_joint_checkpoint_sha256": points.metadata.get(
            "matcha_joint_checkpoint_sha256"
        ),
        "exported_splits": exported_splits,
        "test_source_points_materialized": "test" in exported_splits,
        "source_target_arrays_read": False,
        "source_target_arrays_excluded": ["pose", "residual", "registered_track_identity"],
        "point_candidate_support_sha256": _array_sha256_short(
            np.asarray(points.source_point_ids, dtype=np.int64),
            np.asarray(points.candidate_track_ids, dtype=np.int64),
            selected_ids,
            selected_xy,
        ),
    }
    layout = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray(points.source_point_ids, dtype=np.int64),
        query_ids=np.asarray(points.query_ids).astype(str),
        split_names=np.asarray(points.split_names).astype(str),
        xy=np.asarray(points.xy, dtype=np.float32),
        point_sources=np.asarray(points.point_sources).astype(str),
        candidate_track_ids=np.asarray(points.candidate_track_ids, dtype=np.int64),
        candidate_bank_rows=np.asarray(points.candidate_bank_rows, dtype=np.int64),
        candidate_coarse_similarities=np.asarray(
            points.candidate_coarse_similarities, dtype=np.float32
        ),
        candidate_prior_probabilities=np.asarray(
            points.candidate_prior_probabilities, dtype=np.float32
        ),
        null_probabilities=np.asarray(points.null_probabilities, dtype=np.float32),
        support_image_ids=selected_ids,
        support_xy=selected_xy,
        support_view_valid=selected_valid,
        support_view_weights=selected_weights,
        support_coverage_counts=selected_coverage,
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_layout(layout, output)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_layout",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "row_count": layout.row_count,
        "candidate_top_k": layout.candidate_count,
        "support_views_per_candidate": layout.support_view_count,
        "valid_support_view_count": int(np.count_nonzero(layout.support_view_valid)),
        "candidate_without_rgb_support_count": int(
            np.count_nonzero(
                (layout.candidate_track_ids >= 0)
                & ~np.any(layout.support_view_valid, axis=2)
            )
        ),
        "point_count_by_source": {
            source: int(np.count_nonzero(layout.point_sources == source))
            for source in sorted(set(layout.point_sources.tolist()))
        },
        "protocol": {
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "candidate_reselection": False,
            "test_source_points_materialized": "test" in exported_splits,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_layout(
        verification_points=Path(args.verification_points),
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
