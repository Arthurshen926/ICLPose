"""Build train-only spatial targets for a frozen candidate RGB layout.

The input layout is target-free and is also the only layout available at
runtime.  This builder joins train-only correct/coherent-wrong poses after
that layout is frozen.  The correct-pose projection defines local spatial
support; the support geometry index remains a layout-lineage input and is not
misused as a query-observation label source.  Its output is only for training;
no scorer is allowed to load it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    project_simple_radial_offsets,
    save_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.query_observation_identity import (
    RegisteredQueryObservationTargets,
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    validate_serialized_grouped_hypothesis_semantic_lineage,
)
TRAIN_PAIR_FORMAT = "candidate_pose_llr_train_pairs_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--spatial-search-radius-px", type=float, default=12.0)
    parser.add_argument(
        "--spatial-supervision-mode",
        choices=("geometry_projected", "registered_exact_identity"),
        default="geometry_projected",
        help=(
            "Use legacy geometry-only local support or train-only exact SfM track "
            "identity with explicit candidate dustbins."
        ),
    )
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def spatial_supervision_from_correct_projections(
    *,
    projection_offsets_xy: np.ndarray,
    projection_valid: np.ndarray,
    spatial_search_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a correct-pose projection into local density/dustbin supervision.

    The support index intentionally excludes query images, so it cannot define
    a query observation target.  A frozen candidate receives a non-dustbin
    local target only when its correct-pose projection is visible and lies in
    the spatial branch's fixed square support.  All other candidate edges are
    trained as dustbin.
    """

    offsets = np.asarray(projection_offsets_xy, dtype=np.float32)
    valid = np.asarray(projection_valid, dtype=bool)
    radius = float(spatial_search_radius_px)
    if (
        offsets.ndim != 3
        or offsets.shape[2] != 2
        or valid.shape != offsets.shape[:2]
        or not np.isfinite(offsets).all()
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError("correct-projection spatial supervision inputs are invalid")
    non_dustbin = valid & (np.max(np.abs(offsets), axis=2) <= radius)
    return offsets, non_dustbin, ~non_dustbin


def spatial_supervision_from_registered_identity(
    *,
    projection_offsets_xy: np.ndarray,
    projection_valid: np.ndarray,
    candidate_track_ids: np.ndarray,
    registered_targets: RegisteredQueryObservationTargets,
    spatial_search_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build exact-track local targets without treating nearby wrong tracks as positives.

    A frozen top-L candidate that merely projects near the query anchor under
    the correct camera pose is not necessarily the physical point observed at
    that anchor.  For the candidate-specific likelihood, only the registered
    SfM track at a train anchor is a local-offset positive.  Other frozen
    candidates become dustbin examples; anchors without a registered
    observation remain entirely unsupervised rather than being mislabeled as
    null.  The builder is train-only and this identity information never
    enters the runtime layout or scorer.
    """

    offsets = np.asarray(projection_offsets_xy, dtype=np.float32)
    valid = np.asarray(projection_valid, dtype=bool)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    radius = float(spatial_search_radius_px)
    if (
        offsets.ndim != 3
        or offsets.shape[2] != 2
        or valid.shape != offsets.shape[:2]
        or tracks.shape != offsets.shape[:2]
        or len(registered_targets.track_ids) != len(offsets)
        or not np.isfinite(offsets).all()
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError("registered-identity spatial supervision inputs are invalid")
    labels = registered_candidate_identity_labels(tracks, registered_targets)
    if np.any(np.sum(labels, axis=1) > 1):
        raise ValueError("registered identity labels select duplicate frozen candidates")
    local = valid & (np.max(np.abs(offsets), axis=2) <= radius)
    observed = labels & local
    has_exact_candidate = np.any(labels, axis=1)
    has_local_exact_candidate = np.any(observed, axis=1)
    # An exact candidate whose train-pose projection cannot reach the fixed
    # local support signals a coordinate/projection contract problem.  Do not
    # turn that row into a false all-dustbin target; exclude it and report it.
    projection_consistent = ~has_exact_candidate | has_local_exact_candidate
    supervised_rows = np.asarray(registered_targets.supervised, dtype=bool) & projection_consistent
    supervised = np.broadcast_to(supervised_rows[:, None], offsets.shape[:2]).copy()
    dustbin = supervised & ~observed
    if np.any(observed & dustbin) or np.any((observed | dustbin) & ~supervised):
        raise RuntimeError("registered-identity spatial supervision is inconsistent")
    identity_summary = summarize_registered_candidate_identity(labels, registered_targets)
    audit: dict[str, object] = {
        "registered_identity_target_coverage": identity_summary,
        "registered_supervised_row_count": int(np.count_nonzero(registered_targets.supervised)),
        "projection_consistent_registered_row_count": int(
            np.count_nonzero(np.asarray(registered_targets.supervised, dtype=bool) & projection_consistent)
        ),
        "projection_inconsistent_exact_candidate_row_count": int(
            np.count_nonzero(np.asarray(registered_targets.supervised, dtype=bool) & ~projection_consistent)
        ),
        "exact_identity_observed_candidate_count": int(np.count_nonzero(observed)),
        "explicit_null_supervised_row_count": int(
            np.count_nonzero(np.asarray(registered_targets.supervised, dtype=bool) & ~has_exact_candidate)
        ),
        "supervised_candidate_edge_count": int(np.count_nonzero(supervised)),
        "dustbin_candidate_edge_count": int(np.count_nonzero(dustbin)),
    }
    return offsets, observed, dustbin, supervised, audit


def _load_train_pairs(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    required = {
        "query_ids",
        "split_names",
        "correct_poses_w2c",
        "coherent_wrong_poses_w2c",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"RGB spatial train pairs lack {sorted(missing)}")
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in required
            if name != "metadata_json"
        }
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    correct = np.asarray(arrays["correct_poses_w2c"], dtype=np.float64)
    wrong = np.asarray(arrays["coherent_wrong_poses_w2c"], dtype=np.float64)
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != TRAIN_PAIR_FORMAT
        or metadata.get("training_only_target_artifact") is not True
        or metadata.get("contains_validation_or_test_targets") is not False
        or metadata.get("pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer")
        is not True
        or len(query_ids) == 0
        or splits.shape != query_ids.shape
        or np.any(query_ids == "")
        or np.any(splits != "train")
        or correct.shape != (len(query_ids), 4, 4)
        or wrong.shape != correct.shape
        or not np.isfinite(correct).all()
        or not np.isfinite(wrong).all()
    ):
        raise ValueError("RGB spatial train pairs violate the train-only contract")
    try:
        validate_serialized_grouped_hypothesis_semantic_lineage(
            metadata.get("hypothesis_semantic_lineage")
        )
    except ValueError as error:
        raise ValueError("RGB spatial train pairs have invalid hypothesis lineage") from error
    return {
        "query_ids": query_ids,
        "correct_poses_w2c": correct,
        "coherent_wrong_poses_w2c": wrong,
    }, dict(metadata)


def _load_bank_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"track_ids", "xyz", "metadata_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"projected landmark bank lacks {sorted(missing)}")
        tracks = np.asarray(payload["track_ids"], dtype=np.int64).reshape(-1)
        xyz = np.asarray(payload["xyz"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "landmark_map_index_npz"
        or len(tracks) == 0
        or len(np.unique(tracks)) != len(tracks)
        or xyz.shape != (len(tracks), 3)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("projected landmark bank geometry is invalid")
    return tracks, xyz, metadata


def _candidate_xyz_for_layout(
    *, layout: CandidatePoseRGBSpatialLayout, bank_tracks: np.ndarray, bank_xyz: np.ndarray
) -> np.ndarray:
    rows = np.asarray(layout.candidate_bank_rows, dtype=np.int64)
    tracks = np.asarray(layout.candidate_track_ids, dtype=np.int64)
    valid = tracks >= 0
    if (
        rows.shape != tracks.shape
        or np.any(valid & ((rows < 0) | (rows >= len(bank_tracks))))
        or np.any(~valid & (rows != -1))
        or np.any(bank_tracks[rows[valid]] != tracks[valid])
    ):
        raise ValueError("RGB spatial layout candidate bank rows are stale")
    output = np.zeros((*tracks.shape, 3), dtype=np.float32)
    output[valid] = bank_xyz[rows[valid]]
    return output


def _query_camera_parameters(
    *, colmap_model_dir: Path, query_ids: Sequence[str]
) -> dict[str, tuple[float, float, float, float, int, int]]:
    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    camera_ids = read_colmap_image_camera_ids_binary(Path(colmap_model_dir) / "images.bin")
    output: dict[str, tuple[float, float, float, float, int, int]] = {}
    for query_id in sorted(set(str(value) for value in query_ids)):
        camera_id = camera_ids.get(query_id)
        camera = None if camera_id is None else cameras.get(int(camera_id))
        if camera is None or int(camera.model_id) != 2 or len(camera.params) != 4:
            raise ValueError("RGB spatial targets require SIMPLE_RADIAL query cameras")
        focal, principal_x, principal_y, radial_k = map(float, camera.params)
        output[query_id] = (
            focal,
            principal_x,
            principal_y,
            radial_k,
            int(camera.width),
            int(camera.height),
        )
    return output


def build_candidate_pose_rgb_spatial_training_targets(
    *,
    rgb_spatial_layout: Path,
    train_pairs: Path,
    support_geometry_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    spatial_search_radius_px: float,
    spatial_supervision_mode: str,
    registered_identity_radius_px: float,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, object]:
    """Materialize all train-only supervision without contaminating runtime P1."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite RGB spatial training targets")
    supervision_mode = str(spatial_supervision_mode)
    identity_radius = float(registered_identity_radius_px)
    if supervision_mode not in {"geometry_projected", "registered_exact_identity"}:
        raise ValueError("RGB spatial supervision mode is invalid")
    if not np.isfinite(identity_radius) or identity_radius <= 0.0:
        raise ValueError("registered identity radius must be finite and positive")
    layout_path = Path(rgb_spatial_layout)
    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    if set(np.asarray(layout.split_names).astype(str).tolist()) - {"train", "validation"}:
        raise ValueError("RGB spatial target layout may contain only train/validation runtime rows")
    train_rows = np.flatnonzero(np.asarray(layout.split_names).astype(str) == "train")
    if len(train_rows) == 0:
        raise ValueError("RGB spatial target layout has no train rows")
    pair_arrays, pair_metadata = _load_train_pairs(Path(train_pairs))
    pair_queries = np.asarray(pair_arrays["query_ids"]).astype(str)
    layout_train_queries = set(np.asarray(layout.query_ids)[train_rows].astype(str).tolist())
    if set(pair_queries.tolist()) != layout_train_queries:
        raise ValueError("RGB spatial train pairs and frozen train layout query coverage differ")
    support_geometry_path = Path(support_geometry_index)
    if (
        not support_geometry_path.is_file()
        or file_sha256_short(support_geometry_path)
        != str(layout.metadata.get("support_geometry_index_sha256", ""))
    ):
        raise ValueError("RGB spatial support geometry differs from the frozen layout")
    bank_tracks, bank_xyz, bank_metadata = _load_bank_xyz(Path(projected_landmark_bank))
    candidate_xyz = _candidate_xyz_for_layout(
        layout=layout, bank_tracks=bank_tracks, bank_xyz=bank_xyz
    )
    camera_by_query = _query_camera_parameters(
        colmap_model_dir=Path(colmap_model_dir), query_ids=pair_queries
    )

    layout_rows_by_query = {
        query_id: train_rows[np.asarray(layout.query_ids)[train_rows].astype(str) == query_id]
        for query_id in sorted(layout_train_queries)
    }
    pair_point_offsets = [0]
    pair_source_ids: list[np.ndarray] = []
    correct_parts: list[np.ndarray] = []
    correct_valid_parts: list[np.ndarray] = []
    wrong_parts: list[np.ndarray] = []
    wrong_valid_parts: list[np.ndarray] = []
    source_ids = np.asarray(layout.source_point_ids[train_rows], dtype=np.int64)
    source_query_ids = np.asarray(layout.query_ids[train_rows]).astype(str)
    registered_targets: RegisteredQueryObservationTargets | None = None
    images_path = Path(colmap_model_dir) / "images.bin"
    if supervision_mode == "registered_exact_identity":
        if not images_path.is_file():
            raise FileNotFoundError(
                f"registered-identity RGB spatial targets are missing {images_path}"
            )
        images_by_name = {
            str(image.image_name): image
            for image in read_colmap_images_binary(images_path).values()
        }
        registered_targets = registered_query_observation_targets(
            query_ids=source_query_ids,
            query_xy=np.asarray(layout.xy[train_rows], dtype=np.float32),
            images_by_name=images_by_name,
            max_distance_px=identity_radius,
        )
    source_positions = {
        int(source_id): position for position, source_id in enumerate(source_ids.tolist())
    }
    target_projection_offsets = np.zeros(
        (*layout.candidate_track_ids[train_rows].shape, 2), dtype=np.float32
    )
    target_projection_valid = np.zeros(
        layout.candidate_track_ids[train_rows].shape, dtype=bool
    )
    target_projection_resolved = np.zeros((len(train_rows),), dtype=bool)
    for pair_index, query_id in enumerate(pair_queries.tolist()):
        rows = np.asarray(layout_rows_by_query[str(query_id)], dtype=np.int64)
        if len(rows) == 0:
            raise RuntimeError("RGB spatial target pair lost frozen query points")
        focal, principal_x, principal_y, radial_k, width, height = camera_by_query[str(query_id)]
        query_xy = np.asarray(layout.xy[rows], dtype=np.float32)
        xyz = candidate_xyz[rows]
        correct_pose = np.broadcast_to(
            np.asarray(pair_arrays["correct_poses_w2c"][pair_index], dtype=np.float64),
            (len(rows), 4, 4),
        ).copy()
        wrong_pose = np.broadcast_to(
            np.asarray(pair_arrays["coherent_wrong_poses_w2c"][pair_index], dtype=np.float64),
            (len(rows), 4, 4),
        ).copy()
        correct_offsets, correct_valid = project_simple_radial_offsets(
            xyz=xyz,
            poses_w2c=correct_pose,
            query_xy=query_xy,
            focal_length=focal,
            principal_x=principal_x,
            principal_y=principal_y,
            radial_k=radial_k,
            image_width=width,
            image_height=height,
        )
        wrong_offsets, wrong_valid = project_simple_radial_offsets(
            xyz=xyz,
            poses_w2c=wrong_pose,
            query_xy=query_xy,
            focal_length=focal,
            principal_x=principal_x,
            principal_y=principal_y,
            radial_k=radial_k,
            image_width=width,
            image_height=height,
        )
        candidate_valid = np.asarray(layout.candidate_track_ids[rows] >= 0, dtype=bool)
        correct_valid &= candidate_valid
        wrong_valid &= candidate_valid
        source_ids_for_pair = np.asarray(layout.source_point_ids[rows], dtype=np.int64)
        target_positions = np.asarray(
            [source_positions[int(source_id)] for source_id in source_ids_for_pair],
            dtype=np.int64,
        )
        already_resolved = target_projection_resolved[target_positions]
        if np.any(already_resolved):
            if (
                not np.all(already_resolved)
                or not np.allclose(
                    target_projection_offsets[target_positions], correct_offsets, atol=1e-5, rtol=0.0
                )
                or not np.array_equal(target_projection_valid[target_positions], correct_valid)
            ):
                raise ValueError("RGB spatial train pairs disagree on a correct-pose projection")
        else:
            target_projection_offsets[target_positions] = correct_offsets
            target_projection_valid[target_positions] = correct_valid
            target_projection_resolved[target_positions] = True
        pair_source_ids.append(source_ids_for_pair)
        correct_parts.append(correct_offsets)
        correct_valid_parts.append(correct_valid)
        wrong_parts.append(wrong_offsets)
        wrong_valid_parts.append(wrong_valid)
        pair_point_offsets.append(pair_point_offsets[-1] + len(rows))

    if not np.all(target_projection_resolved):
        raise RuntimeError("RGB spatial targets did not resolve every frozen train point")
    registered_identity_audit: dict[str, object] | None = None
    if supervision_mode == "geometry_projected":
        target_offsets, target_observed, target_dustbin = (
            spatial_supervision_from_correct_projections(
                projection_offsets_xy=target_projection_offsets,
                projection_valid=target_projection_valid,
                spatial_search_radius_px=float(spatial_search_radius_px),
            )
        )
        target_supervised = np.ones(target_observed.shape, dtype=bool)
        target_format = CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT
        target_semantics = (
            "correct_pose_projected_offset_inside_fixed_local_support_or_dustbin_v1"
        )
    else:
        if registered_targets is None:
            raise RuntimeError("registered-identity RGB spatial targets were not initialized")
        (
            target_offsets,
            target_observed,
            target_dustbin,
            target_supervised,
            registered_identity_audit,
        ) = spatial_supervision_from_registered_identity(
            projection_offsets_xy=target_projection_offsets,
            projection_valid=target_projection_valid,
            candidate_track_ids=np.asarray(
                layout.candidate_track_ids[train_rows], dtype=np.int64
            ),
            registered_targets=registered_targets,
            spatial_search_radius_px=float(spatial_search_radius_px),
        )
        target_format = CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        target_semantics = (
            "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
        )
    if not np.any(target_observed):
        raise ValueError("RGB spatial targets contain no exact local supervision")
    metadata: dict[str, object] = {
        "format": target_format,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "rgb_spatial_layout_sha256": file_sha256_short(layout_path),
        "train_pairs_sha256": file_sha256_short(Path(train_pairs)),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_path),
        "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
        "projection_space_id": str(layout.metadata["projection_space_id"]),
        "descriptor_space_id": str(layout.metadata["descriptor_space_id"]),
        "spatial_search_radius_px": float(spatial_search_radius_px),
        "spatial_supervision_mode": supervision_mode,
        "spatial_target_semantics": target_semantics,
        "hypothesis_projection_semantics": "simple_radial_current_topl_track_projection_v1",
        "hypothesis_semantic_lineage": pair_metadata["hypothesis_semantic_lineage"],
        "inputs": {
            "rgb_spatial_layout": str(layout_path),
            "train_pairs": str(train_pairs),
            "support_geometry_index": str(support_geometry_path),
            "projected_landmark_bank": str(projected_landmark_bank),
            "colmap_cameras_bin": str(Path(colmap_model_dir) / "cameras.bin"),
            "colmap_images_bin": str(images_path),
            "support_geometry_usage": "frozen_layout_lineage_only_query_images_excluded",
            "projected_landmark_bank_format": bank_metadata.get("format"),
        },
    }
    if registered_identity_audit is not None:
        metadata.update(
            {
                "registered_identity_radius_px": identity_radius,
                "colmap_images_sha256": file_sha256_short(images_path),
                "registered_identity_audit": registered_identity_audit,
                "spatial_class_balance": "per_batch_observed_dustbin_mean_v1",
            }
        )
    targets = CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=source_ids,
        query_ids=source_query_ids,
        spatial_target_offsets_xy=target_offsets,
        spatial_target_observed=target_observed,
        spatial_target_dustbin=target_dustbin,
        spatial_target_supervised=target_supervised,
        pair_query_ids=pair_queries,
        pair_ids=np.arange(len(pair_queries), dtype=np.int64),
        pair_point_offsets=np.asarray(pair_point_offsets, dtype=np.int64),
        pair_source_point_ids=np.concatenate(pair_source_ids, axis=0),
        correct_projection_offsets_xy=np.concatenate(correct_parts, axis=0),
        correct_projection_valid=np.concatenate(correct_valid_parts, axis=0),
        coherent_wrong_projection_offsets_xy=np.concatenate(wrong_parts, axis=0),
        coherent_wrong_projection_valid=np.concatenate(wrong_valid_parts, axis=0),
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_training_targets(targets, output)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_targets",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "source_point_count": targets.source_point_count,
        "candidate_count": targets.candidate_count,
        "pair_count": targets.pair_count,
        "train_query_count": int(len(layout_train_queries)),
        "spatial_non_dustbin_count": int(np.count_nonzero(targets.spatial_target_observed)),
        "spatial_dustbin_count": int(np.count_nonzero(targets.spatial_target_dustbin)),
        "spatial_supervised_count": int(np.count_nonzero(targets.spatial_target_supervised)),
        "spatial_supervision_mode": supervision_mode,
        "registered_identity_audit": registered_identity_audit,
        "protocol": {
            "train_only_target_artifact": True,
            "runtime_layout_remains_target_free": True,
            "pose_or_ground_truth_not_available_to_runtime_scorer": True,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_training_targets(
        rgb_spatial_layout=Path(args.rgb_spatial_layout),
        train_pairs=Path(args.train_pairs),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        spatial_search_radius_px=float(args.spatial_search_radius_px),
        spatial_supervision_mode=str(args.spatial_supervision_mode),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
