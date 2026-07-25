from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    CandidatePoseRGBSpatialSelectorInput,
    select_target_free_spatial_quota,
    slice_target_free_edge_prediction,
    target_free_layout_point_quality,
    target_free_rgb_point_quality,
    target_free_selector_scores,
)


def _selector_input() -> CandidatePoseRGBSpatialSelectorInput:
    count = 8
    candidate = np.tile(np.asarray([[0.45, 0.35]], dtype=np.float32), (count, 1))
    return CandidatePoseRGBSpatialSelectorInput(
        source_point_ids=np.arange(100, 100 + count, dtype=np.int64),
        xy=np.asarray(
            [
                [10.0, 10.0],
                [30.0, 30.0],
                [70.0, 10.0],
                [90.0, 30.0],
                [10.0, 70.0],
                [30.0, 90.0],
                [70.0, 70.0],
                [90.0, 90.0],
            ],
            dtype=np.float32,
        ),
        point_sources=np.asarray(["alike", "radio_final"] * 4),
        candidate_probabilities=candidate,
        null_probabilities=np.full((count,), 0.2, dtype=np.float32),
        support_view_valid=np.ones((count, 2, 1), dtype=bool),
        support_view_weights=np.ones((count, 2, 1), dtype=np.float32),
        support_coverage_counts=np.tile(np.asarray([[[2], [8]]], dtype=np.int32), (count, 1, 1)),
    )


def _runtime_and_prediction() -> tuple[CandidatePoseRGBSpatialRuntime, CandidatePoseRGBSpatialEdgePrediction]:
    selector_input = _selector_input()
    count = selector_input.point_count
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.zeros((count,), dtype=torch.long),
        query_xy=torch.from_numpy(selector_input.xy),
        support_image_indices=torch.ones((count, 2, 1), dtype=torch.long),
        support_xy=torch.zeros((count, 2, 1, 2), dtype=torch.float32),
        support_view_valid=torch.ones((count, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((count, 2, 1), dtype=torch.float32),
        candidate_probabilities=torch.from_numpy(selector_input.candidate_probabilities),
        null_probabilities=torch.from_numpy(selector_input.null_probabilities),
    )
    offsets = torch.tensor(
        [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    logits = torch.zeros((count, 2, 1, len(offsets)), dtype=torch.float32)
    # Later rows have increasingly sharp target-free local densities.
    logits[..., 4] = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1)
    joint = torch.log_softmax(
        torch.cat([logits, torch.zeros((count, 2, 1, 1), dtype=torch.float32)], dim=-1), dim=-1
    )
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=logits,
        non_dustbin_logits=torch.zeros((count, 2, 1), dtype=torch.float32),
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        context_log_likelihood_ratios=torch.zeros((count, 2, 1), dtype=torch.float32),
        edge_usable=torch.ones((count, 2, 1), dtype=torch.bool),
    )
    return runtime, prediction


def test_target_free_spatial_quota_is_deterministic_and_diverse() -> None:
    selector_input = _selector_input()
    quality = np.arange(selector_input.point_count, dtype=np.float32)
    selected = select_target_free_spatial_quota(
        selector_input=selector_input,
        quality_scores=quality,
        point_budget=4,
        grid_rows=2,
        grid_columns=2,
        image_size=(100, 100),
    )
    # One highest-ranked point per 2x2 cell, then source order is irrelevant.
    assert selected.tolist() == [1, 3, 5, 7]
    repeated = select_target_free_spatial_quota(
        selector_input=selector_input,
        quality_scores=quality,
        point_budget=4,
        grid_rows=2,
        grid_columns=2,
        image_size=(100, 100),
    )
    assert np.array_equal(selected, repeated)


def test_target_free_quality_never_needs_pose_or_registered_identity() -> None:
    selector_input = _selector_input()
    layout_quality = target_free_layout_point_quality(selector_input)
    assert set(layout_quality) >= {"coarse_margin", "support_coverage"}
    runtime, prediction = _runtime_and_prediction()
    rgb_quality = target_free_rgb_point_quality(runtime=runtime, prediction=prediction)
    assert rgb_quality["rgb_peakiness"][-1] > rgb_quality["rgb_peakiness"][0]
    score = target_free_selector_scores(
        selector_input=selector_input,
        policy="coarse_margin_rgb_peakiness",
        rgb_quality=rgb_quality,
    )
    assert score.shape == (selector_input.point_count,)
    assert np.isfinite(score).all()
    static_score = target_free_selector_scores(
        selector_input=selector_input,
        policy="coarse_margin_support_coverage",
    )
    assert static_score.shape == (selector_input.point_count,)
    assert np.isfinite(static_score).all()
    with pytest.raises(ValueError, match="requires RGB quality"):
        target_free_selector_scores(selector_input=selector_input, policy="rgb_peakiness")


def test_static_support_coverage_excludes_rgb_crop_unusable_edges() -> None:
    selector_input = _selector_input()
    rgb_usable = np.array(selector_input.support_view_valid, copy=True)
    rgb_usable[0] = False
    crop_aware = CandidatePoseRGBSpatialSelectorInput(
        source_point_ids=selector_input.source_point_ids,
        xy=selector_input.xy,
        point_sources=selector_input.point_sources,
        candidate_probabilities=selector_input.candidate_probabilities,
        null_probabilities=selector_input.null_probabilities,
        support_view_valid=selector_input.support_view_valid,
        support_view_weights=selector_input.support_view_weights,
        support_coverage_counts=selector_input.support_coverage_counts,
        support_view_rgb_usable=rgb_usable,
    )
    baseline = target_free_layout_point_quality(selector_input)
    quality = target_free_layout_point_quality(crop_aware)
    assert quality["support_coverage"][0] == 0.0
    assert quality["support_coverage"][0] < baseline["support_coverage"][0]
    assert quality["rgb_context_usable_fraction"][0] == 0.0


def test_target_free_selector_rejects_invalid_probability_mass() -> None:
    with pytest.raises(ValueError, match="prior mass"):
        CandidatePoseRGBSpatialSelectorInput(
            source_point_ids=np.asarray([1], dtype=np.int64),
            xy=np.asarray([[1.0, 1.0]], dtype=np.float32),
            point_sources=np.asarray(["alike"]),
            candidate_probabilities=np.asarray([[0.8, 0.3]], dtype=np.float32),
            null_probabilities=np.asarray([0.0], dtype=np.float32),
            support_view_valid=np.ones((1, 2, 1), dtype=bool),
            support_view_weights=np.ones((1, 2, 1), dtype=np.float32),
            support_coverage_counts=np.ones((1, 2, 1), dtype=np.int32),
        )


def test_target_free_prediction_slice_preserves_only_selected_rows() -> None:
    _runtime, prediction = _runtime_and_prediction()
    subset = slice_target_free_edge_prediction(prediction=prediction, positions=np.asarray([1, 7]))
    assert subset.spatial_logits.shape[0] == 2
    assert torch.equal(subset.spatial_logits[0], prediction.spatial_logits[1])
    with pytest.raises(ValueError, match="slice positions"):
        slice_target_free_edge_prediction(prediction=prediction, positions=np.asarray([1, 1]))
