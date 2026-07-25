"""Build broad, train-only candidate rows aligned to real coherent-wrong poses.

This widens sparse P1 pose supervision without changing its target-free
runtime layout.  Each row starts from a real train-query SfM observation, uses
one same-track mapping support plus global RADIO-PCA ANN hard negatives, and
then joins a correct/coherent-wrong pose pair only to materialize projected
offset targets.  The saved artifact is forbidden at inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_targets import (
    _load_train_pairs,
    _query_camera_parameters,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_points3d_binary
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT,
    CandidatePoseRGBSpatialHardPosePairs,
    save_candidate_pose_rgb_spatial_hard_pose_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    load_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    project_simple_radial_offsets,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--anchors-per-pose-pair", type=int, default=32)
    parser.add_argument("--anchor-jitter-radius-px", type=int, default=4)
    parser.add_argument("--spatial-search-radius-px", type=float, default=8.0)
    parser.add_argument("--rgb-patch-radius-px", type=float, default=20.0)
    parser.add_argument("--min-wrong-positive-offset-delta-px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _stable_hash(*values: object) -> int:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _sample_nonzero_jitter(
    *, count: int, radius_px: int, generator: np.random.Generator
) -> np.ndarray:
    """Sample deterministic local anchor shifts without an all-zero shortcut."""

    size = int(count)
    radius = int(radius_px)
    if size <= 0 or radius <= 0:
        raise ValueError("hard-pose jitter arguments are invalid")
    values = np.asarray(
        [
            (x, y)
            for y in range(-radius, radius + 1)
            for x in range(-radius, radius + 1)
            if x != 0 or y != 0
        ],
        dtype=np.float32,
    )
    if len(values) == 0:
        raise RuntimeError("hard-pose jitter grid is empty")
    return values[generator.integers(0, len(values), size=size)]


def _inside_patch(
    *, xy: np.ndarray, width: int, height: int, radius_px: float
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float32)
    radius = float(radius_px)
    if points.ndim != 2 or points.shape[1] != 2 or radius <= 0.0:
        raise ValueError("hard-pose patch bounds inputs are invalid")
    return (
        (points[:, 0] >= radius)
        & (points[:, 0] <= float(width - 1) - radius)
        & (points[:, 1] >= radius)
        & (points[:, 1] <= float(height - 1) - radius)
    )


def _candidate_xyz(
    *, track_ids: np.ndarray, xyz_by_track: dict[int, np.ndarray]
) -> np.ndarray:
    tracks = np.asarray(track_ids, dtype=np.int64)
    if tracks.ndim != 2 or len(tracks) == 0:
        raise ValueError("hard-pose candidate track matrix is invalid")
    try:
        output = np.stack(
            [[np.asarray(xyz_by_track[int(track)], dtype=np.float64) for track in row] for row in tracks],
            axis=0,
        )
    except KeyError as error:
        raise ValueError("hard-pose candidate track lacks COLMAP xyz") from error
    if output.shape != (*tracks.shape, 3) or not np.isfinite(output).all():
        raise ValueError("hard-pose candidate xyz is invalid")
    return output


def _project_pair_offsets(
    *,
    candidate_xyz: np.ndarray,
    anchors_xy: np.ndarray,
    correct_pose_w2c: np.ndarray,
    wrong_pose_w2c: np.ndarray,
    camera: tuple[float, float, float, float, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(candidate_xyz, dtype=np.float64)
    anchors = np.asarray(anchors_xy, dtype=np.float32)
    if points.ndim != 3 or points.shape[0] != len(anchors):
        raise ValueError("hard-pose projection candidates do not match anchors")
    focal, principal_x, principal_y, radial_k, width, height = camera
    correct = np.broadcast_to(
        np.asarray(correct_pose_w2c, dtype=np.float64), (len(anchors), 4, 4)
    ).copy()
    wrong = np.broadcast_to(
        np.asarray(wrong_pose_w2c, dtype=np.float64), (len(anchors), 4, 4)
    ).copy()
    correct_offsets, correct_valid = project_simple_radial_offsets(
        xyz=points,
        poses_w2c=correct,
        query_xy=anchors,
        focal_length=float(focal),
        principal_x=float(principal_x),
        principal_y=float(principal_y),
        radial_k=float(radial_k),
        image_width=int(width),
        image_height=int(height),
    )
    wrong_offsets, wrong_valid = project_simple_radial_offsets(
        xyz=points,
        poses_w2c=wrong,
        query_xy=anchors,
        focal_length=float(focal),
        principal_x=float(principal_x),
        principal_y=float(principal_y),
        radial_k=float(radial_k),
        image_width=int(width),
        image_height=int(height),
    )
    return correct_offsets, correct_valid, wrong_offsets, wrong_valid


def build_candidate_pose_rgb_spatial_hard_pose_pairs(
    *,
    observation_pairs: Path,
    train_pairs: Path,
    colmap_model_dir: Path,
    output: Path,
    summary_json: Path,
    anchors_per_pose_pair: int,
    anchor_jitter_radius_px: int,
    spatial_search_radius_px: float,
    rgb_patch_radius_px: float,
    min_wrong_positive_offset_delta_px: float,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    """Materialize fixed broad candidate inputs and train-only hard-pose targets."""

    group_size = int(anchors_per_pose_pair)
    jitter_radius = int(anchor_jitter_radius_px)
    search_radius = float(spatial_search_radius_px)
    patch_radius = float(rgb_patch_radius_px)
    minimum_delta = float(min_wrong_positive_offset_delta_px)
    if (
        group_size < 2
        or jitter_radius <= 0
        or not np.isfinite(search_radius)
        or not np.isfinite(patch_radius)
        or not np.isfinite(minimum_delta)
        or search_radius <= 0.0
        or patch_radius < search_radius
        or minimum_delta <= 0.0
    ):
        raise ValueError("hard-pose pair build arguments are invalid")
    source_path = Path(observation_pairs)
    pair_path = Path(train_pairs)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite hard-pose pair outputs")
    source = load_candidate_pose_rgb_spatial_observation_pairs(source_path)
    pair_arrays, pair_metadata = _load_train_pairs(pair_path)
    pair_query_ids = np.asarray(pair_arrays["query_ids"]).astype(str)
    source_query_ids = set(np.asarray(source.query_image_ids).astype(str).tolist())
    if set(pair_query_ids.tolist()) != source_query_ids:
        raise ValueError("hard-pose train pairs and observation-pair query coverage differ")
    model_dir = Path(colmap_model_dir)
    images_path = model_dir / "images.bin"
    points_path = model_dir / "points3D.bin"
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError("hard-pose pair builder lacks COLMAP images or points")
    camera_by_query = _query_camera_parameters(
        colmap_model_dir=model_dir, query_ids=pair_query_ids
    )
    xyz_by_track = {
        int(track_id): np.asarray(point.xyz, dtype=np.float64)
        for track_id, point in read_colmap_points3d_binary(points_path).items()
    }
    if not xyz_by_track:
        raise ValueError("hard-pose COLMAP point map is empty")
    rows_by_query = {
        query_id: np.flatnonzero(np.asarray(source.query_image_ids).astype(str) == query_id)
        for query_id in sorted(source_query_ids)
    }

    query_ids_out: list[np.ndarray] = []
    query_xy_out: list[np.ndarray] = []
    track_out: list[np.ndarray] = []
    support_ids_out: list[np.ndarray] = []
    support_xy_out: list[np.ndarray] = []
    correct_offsets_out: list[np.ndarray] = []
    correct_valid_out: list[np.ndarray] = []
    wrong_offsets_out: list[np.ndarray] = []
    wrong_valid_out: list[np.ndarray] = []
    splits_out: list[np.ndarray] = []
    group_ids_out: list[np.ndarray] = []
    discarded_short_groups = 0
    candidate_probe_count = max(group_size * 4, group_size)
    delta_values: list[np.ndarray] = []

    for pose_pair_id, query_id in enumerate(pair_query_ids.tolist()):
        available = rows_by_query[str(query_id)]
        if len(available) < group_size:
            discarded_short_groups += 1
            continue
        generator = np.random.default_rng(
            _stable_hash(seed, pose_pair_id, query_id, "hard_pose_anchor")
        )
        probe_size = min(len(available), candidate_probe_count)
        probe_rows = np.asarray(
            generator.choice(available, size=probe_size, replace=False), dtype=np.int64
        )
        base_xy = np.asarray(source.query_xy[probe_rows], dtype=np.float32)
        anchors = base_xy + _sample_nonzero_jitter(
            count=len(probe_rows), radius_px=jitter_radius, generator=generator
        )
        focal, principal_x, principal_y, radial_k, width, height = camera_by_query[str(query_id)]
        inside_patch = _inside_patch(
            xy=anchors, width=int(width), height=int(height), radius_px=patch_radius
        )
        candidate_tracks = np.concatenate(
            [
                np.asarray(source.positive_track_ids[probe_rows], dtype=np.int64)[:, None],
                np.asarray(source.negative_track_ids[probe_rows], dtype=np.int64),
            ],
            axis=1,
        )
        candidate_xyz = _candidate_xyz(track_ids=candidate_tracks, xyz_by_track=xyz_by_track)
        correct_offsets, correct_valid, wrong_offsets, wrong_valid = _project_pair_offsets(
            candidate_xyz=candidate_xyz,
            anchors_xy=anchors,
            correct_pose_w2c=np.asarray(pair_arrays["correct_poses_w2c"][pose_pair_id]),
            wrong_pose_w2c=np.asarray(pair_arrays["coherent_wrong_poses_w2c"][pose_pair_id]),
            camera=(focal, principal_x, principal_y, radial_k, width, height),
        )
        correct_local = np.max(np.abs(correct_offsets[:, 0]), axis=1) <= search_radius
        positive_delta = np.linalg.norm(correct_offsets[:, 0] - wrong_offsets[:, 0], axis=1)
        eligible = (
            inside_patch
            & correct_valid[:, 0]
            & wrong_valid[:, 0]
            & correct_local
            & (positive_delta >= minimum_delta)
        )
        eligible_rows = np.flatnonzero(eligible)
        if len(eligible_rows) < group_size:
            discarded_short_groups += 1
            continue
        chosen = np.asarray(
            generator.choice(eligible_rows, size=group_size, replace=False), dtype=np.int64
        )
        selected_source_rows = probe_rows[chosen]
        query_ids_out.append(np.asarray(source.query_image_ids[selected_source_rows]).astype(str))
        query_xy_out.append(anchors[chosen].astype(np.float32, copy=False))
        track_out.append(candidate_tracks[chosen])
        support_ids_out.append(
            np.concatenate(
                [
                    np.asarray(source.positive_support_image_ids[selected_source_rows]).astype(str)[:, None],
                    np.asarray(source.negative_support_image_ids[selected_source_rows]).astype(str),
                ],
                axis=1,
            )
        )
        support_xy_out.append(
            np.concatenate(
                [
                    np.asarray(source.positive_support_xy[selected_source_rows], dtype=np.float32)[:, None, :],
                    np.asarray(source.negative_support_xy[selected_source_rows], dtype=np.float32),
                ],
                axis=1,
            )
        )
        correct_offsets_out.append(correct_offsets[chosen])
        correct_valid_out.append(correct_valid[chosen])
        wrong_offsets_out.append(wrong_offsets[chosen])
        wrong_valid_out.append(wrong_valid[chosen])
        splits_out.append(np.asarray(source.split_names[selected_source_rows]).astype(str))
        group_ids_out.append(np.full((group_size,), pose_pair_id, dtype=np.int64))
        delta_values.append(positive_delta[chosen])

    if not group_ids_out:
        raise ValueError("hard-pose pair builder retained no complete pose group")
    group_ids = np.concatenate(group_ids_out)
    query_ids = np.concatenate(query_ids_out)
    splits = np.concatenate(splits_out)
    if set(splits.tolist()) != {"inner_train", "inner_validation"}:
        raise RuntimeError("hard-pose pair artifact lost an inner train or validation split")
    observed = np.zeros((len(query_ids), int(source.negative_count) + 1), dtype=bool)
    observed[:, 0] = True
    metadata: dict[str, object] = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_scorer_must_not_load_this_artifact": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "candidate_set": "same_track_mapping_support_plus_radio_pca_global_landmark_ann_hard_negatives",
        "pose_target_semantics": "correct_vs_coherent_wrong_simple_radial_candidate_projection_offsets_v1",
        "candidate_count": int(source.negative_count) + 1,
        "anchors_per_pose_pair": group_size,
        "spatial_search_radius_px": search_radius,
        "rgb_patch_radius_px": patch_radius,
        "anchor_jitter_radius_px": jitter_radius,
        "min_wrong_positive_offset_delta_px": minimum_delta,
        "seed": int(seed),
        "source_observation_pairs": str(source_path.resolve()),
        "source_observation_pairs_sha256": file_sha256_short(source_path),
        "train_pairs": str(pair_path.resolve()),
        "train_pairs_sha256": file_sha256_short(pair_path),
        "colmap_images_bin": str(images_path.resolve()),
        "colmap_images_sha256": file_sha256_short(images_path),
        "colmap_points3d_bin": str(points_path.resolve()),
        "colmap_points3d_sha256": file_sha256_short(points_path),
        "source_pair_protocol": {
            "hard_negative_semantics": source.metadata.get("hard_negative_semantics"),
            "mapping_support_query_overlap_count": source.metadata.get(
                "mapping_support_query_overlap_count"
            ),
        },
        "hard_pose_source": pair_metadata.get("coherent_wrong_pose_source"),
    }
    artifact = CandidatePoseRGBSpatialHardPosePairs(
        row_ids=np.arange(len(query_ids), dtype=np.int64),
        pose_pair_ids=group_ids,
        query_image_ids=query_ids,
        query_xy=np.concatenate(query_xy_out).astype(np.float32, copy=False),
        candidate_track_ids=np.concatenate(track_out),
        support_image_ids=np.concatenate(support_ids_out),
        support_xy=np.concatenate(support_xy_out).astype(np.float32, copy=False),
        correct_projection_offsets_xy=np.concatenate(correct_offsets_out).astype(
            np.float32, copy=False
        ),
        correct_projection_valid=np.concatenate(correct_valid_out),
        coherent_wrong_projection_offsets_xy=np.concatenate(wrong_offsets_out).astype(
            np.float32, copy=False
        ),
        coherent_wrong_projection_valid=np.concatenate(wrong_valid_out),
        spatial_target_observed=observed,
        spatial_target_dustbin=~observed,
        split_names=splits,
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_hard_pose_pairs(artifact, output_path)
    group_splits = {
        str(split): int(
            np.count_nonzero(
                [
                    artifact.split_names[np.flatnonzero(artifact.pose_pair_ids == group_id)[0]]
                    == str(split)
                    for group_id in np.unique(artifact.pose_pair_ids)
                ]
            )
        )
        for split in ("inner_train", "inner_validation")
    }
    values = np.concatenate(delta_values)
    summary: dict[str, object] = {
        "stage": "build_candidate_pose_rgb_spatial_hard_pose_pairs",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(artifact.row_count),
        "group_count": int(artifact.group_count),
        "groups_by_inner_split": group_splits,
        "discarded_short_pose_groups": int(discarded_short_groups),
        "positive_correct_vs_wrong_offset_delta_px": {
            "quantiles": np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
        },
        "protocol": {
            "train_query_only": True,
            "runtime_scorer_loads_artifact": False,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
            "candidate_inputs_fixed_before_pose_targets_joined": True,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            build_candidate_pose_rgb_spatial_hard_pose_pairs(
                observation_pairs=Path(args.observation_pairs),
                train_pairs=Path(args.train_pairs),
                colmap_model_dir=Path(args.colmap_model_dir),
                output=Path(args.output),
                summary_json=Path(args.summary_json),
                anchors_per_pose_pair=int(args.anchors_per_pose_pair),
                anchor_jitter_radius_px=int(args.anchor_jitter_radius_px),
                spatial_search_radius_px=float(args.spatial_search_radius_px),
                rgb_patch_radius_px=float(args.rgb_patch_radius_px),
                min_wrong_positive_offset_delta_px=float(
                    args.min_wrong_positive_offset_delta_px
                ),
                seed=int(args.seed),
                force=bool(args.force),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
