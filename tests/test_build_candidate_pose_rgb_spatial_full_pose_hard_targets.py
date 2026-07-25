from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_full_pose_hard_targets import (
    _load_target_free_score_overlay,
    resolve_full_pose_hard_base_spatial_semantics,
    select_diverse_full_pose_hard_mode_indices,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_full_pool_scores import (
    CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
    CandidatePoseRGBSpatialTrainFullPoolScores,
    save_candidate_pose_rgb_spatial_train_full_pool_scores,
)


def _pose_with_camera_center_x(center_x: float, rotation_deg: float = 0.0) -> np.ndarray:
    radians = np.deg2rad(float(rotation_deg))
    rotation = np.asarray(
        [
            [np.cos(radians), 0.0, np.sin(radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(radians), 0.0, np.cos(radians)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ np.asarray([center_x, 0.0, 0.0], dtype=np.float64)
    return pose


def test_full_pose_hard_selector_excludes_correct_and_near_duplicate_modes() -> None:
    poses = np.stack(
        [
            _pose_with_camera_center_x(0.0),  # Correct, even though its score is highest.
            _pose_with_camera_center_x(1.0),
            _pose_with_camera_center_x(1.05),  # Duplicate of row one under diversity gate.
            _pose_with_camera_center_x(2.0, rotation_deg=2.0),
        ],
        axis=0,
    )
    selected = select_diverse_full_pose_hard_mode_indices(
        poses_w2c=poses,
        scores=np.asarray([10.0, 9.0, 8.0, 7.0]),
        translation_errors_m=np.asarray([0.01, 0.4, 0.4, 0.5]),
        rotation_errors_deg=np.asarray([0.1, 0.5, 0.5, 2.0]),
        correct_10cm_5deg=np.asarray([True, False, False, False]),
        source_row_indices=np.asarray([40, 10, 20, 30], dtype=np.int64),
        max_modes=3,
        score_pool_size=4,
        minimum_translation_error_m=0.25,
        minimum_rotation_error_deg=2.0,
        minimum_pose_translation_diversity_m=0.2,
        minimum_pose_rotation_diversity_deg=0.75,
    )
    assert selected.tolist() == [1, 3]


def test_full_pose_hard_selector_uses_source_row_for_score_ties() -> None:
    poses = np.stack(
        [
            _pose_with_camera_center_x(1.0),
            _pose_with_camera_center_x(2.0),
            _pose_with_camera_center_x(3.0),
        ],
        axis=0,
    )
    selected = select_diverse_full_pose_hard_mode_indices(
        poses_w2c=poses,
        scores=np.asarray([2.0, 2.0, np.nan]),
        translation_errors_m=np.asarray([0.4, 0.5, 0.6]),
        rotation_errors_deg=np.asarray([0.1, 0.1, 0.1]),
        correct_10cm_5deg=np.asarray([False, False, False]),
        source_row_indices=np.asarray([9, 4, 1], dtype=np.int64),
        max_modes=2,
        score_pool_size=3,
        minimum_translation_error_m=0.25,
        minimum_rotation_error_deg=2.0,
        minimum_pose_translation_diversity_m=0.1,
        minimum_pose_rotation_diversity_deg=0.1,
    )
    assert selected.tolist() == [1, 0]


def test_full_pose_hard_selector_returns_empty_when_only_near_correct_modes_exist() -> None:
    poses = np.stack([_pose_with_camera_center_x(0.0)], axis=0)
    selected = select_diverse_full_pose_hard_mode_indices(
        poses_w2c=poses,
        scores=np.asarray([1.0]),
        translation_errors_m=np.asarray([0.12]),
        rotation_errors_deg=np.asarray([0.5]),
        correct_10cm_5deg=np.asarray([False]),
        source_row_indices=np.asarray([0], dtype=np.int64),
        max_modes=1,
        score_pool_size=1,
        minimum_translation_error_m=0.25,
        minimum_rotation_error_deg=2.0,
        minimum_pose_translation_diversity_m=0.1,
        minimum_pose_rotation_diversity_deg=0.1,
    )
    assert selected.shape == (0,)


def test_full_pose_hard_base_accepts_current_geometry_projected_contract() -> None:
    mode, semantics = resolve_full_pose_hard_base_spatial_semantics(
        {
            "format": "candidate_pose_rgb_spatial_targets_v1",
            "spatial_target_semantics": (
                "correct_pose_projected_offset_inside_fixed_local_support_or_dustbin_v1"
            ),
        }
    )
    assert mode == "geometry_projected"
    assert semantics == "correct_pose_projected_offset_inside_fixed_local_support_or_dustbin_v1"


def test_full_pose_hard_base_rejects_ambiguous_target_contract() -> None:
    with pytest.raises(ValueError, match="geometry-projected"):
        resolve_full_pose_hard_base_spatial_semantics(
            {
                "format": "candidate_pose_rgb_spatial_targets_v1",
                "spatial_target_semantics": "unknown",
            }
        )


def _overlay_metadata(*, frozen_path: str, frozen_sha256: str, complete: bool) -> dict[str, object]:
    return {
        "format": CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "runtime_layout_is_target_free": True,
        "projection_after_network_only": True,
        "train_rows_only": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "complete_train_coverage": complete,
        "score_component": "combined",
        "fixed_global_topl": True,
        "explicit_null": True,
        "hypothesis_artifacts": [{"path": frozen_path, "sha256": frozen_sha256}],
        "hypothesis_semantic_hash": "semantic",
        "rgb_spatial_layout_sha256": "layout",
        "evaluation_label": "label",
        "checkpoint": {"path": "checkpoint.pt", "sha256": "checkpoint"},
    }


def _write_overlay(tmp_path, *, complete: bool) -> tuple[object, object]:
    frozen = tmp_path / "frozen.npz"
    frozen.write_bytes(b"immutable-frozen-source")
    scores = CandidatePoseRGBSpatialTrainFullPoolScores(
        source_artifact_indices=np.asarray([0], dtype=np.int64),
        source_row_indices=np.asarray([7], dtype=np.int64),
        query_ids=np.asarray(["train/query.png"]),
        split_names=np.asarray(["train"]),
        evaluation_labels=np.asarray(["label"]),
        hypothesis_indices=np.asarray([3], dtype=np.int64),
        pose_log_likelihood_ratios=np.asarray([0.5], dtype=np.float32),
        metadata=_overlay_metadata(
            frozen_path=str(frozen),
            frozen_sha256=file_sha256_short(frozen),
            complete=complete,
        ),
    )
    overlay = tmp_path / "scores.npz"
    save_candidate_pose_rgb_spatial_train_full_pool_scores(scores, overlay)
    return frozen, overlay


def test_full_pose_overlay_requires_complete_combined_target_free_scores(tmp_path) -> None:
    frozen, overlay = _write_overlay(tmp_path, complete=True)
    mapping, lineage = _load_target_free_score_overlay(
        path=overlay,
        hypothesis_artifacts=[frozen],
        expected_semantic_hash="semantic",
        layout_sha256="layout",
        evaluation_label="label",
    )
    assert mapping[(0, 7)] == ("train/query.png", "label", 3, 0.5)
    assert lineage["row_count"] == 1


def test_full_pose_overlay_rejects_limited_smoke_artifact(tmp_path) -> None:
    frozen, overlay = _write_overlay(tmp_path, complete=False)
    with pytest.raises(ValueError, match="not a complete combined likelihood"):
        _load_target_free_score_overlay(
            path=overlay,
            hypothesis_artifacts=[frozen],
            expected_semantic_hash="semantic",
            layout_sha256="layout",
            evaluation_label="label",
        )
