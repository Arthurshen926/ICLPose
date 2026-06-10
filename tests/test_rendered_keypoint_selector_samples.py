from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.rendered_keypoint_selector_samples import (
    RenderedKeypointSelectorSampleConfig,
    build_rendered_keypoint_selector_samples,
    render_keypoint_reprojection_errors,
)
from feature_extract.vfm.gaussian_vfm_field_projection import project_gaussian_vfm_field_features
from feature_extract.vfm.patch_selector_training import ResidualGatedPatchSelector, SafePatchSelectorTrainingRun, SafePatchSelectorTrainingSummary


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=20, height=20, params=(10.0, 10.0, 10.0, 10.0))


def test_render_keypoint_reprojection_errors_uses_render_depth_and_query_pose() -> None:
    camera = _camera()
    depth = np.full((20, 20), 4.0, dtype=np.float32)
    query_xy = np.asarray([[10.0, 10.0], [14.0, 10.0]], dtype=np.float64)
    render_xy = np.asarray([[10.0, 10.0]], dtype=np.float64)

    errors, valid = render_keypoint_reprojection_errors(
        query_xy,
        render_xy,
        rendered_depth=depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=np.eye(4, dtype=np.float64),
        render_width=20,
        render_height=20,
    )

    assert valid.tolist() == [True]
    assert errors.shape == (2, 1)
    assert np.allclose(errors[:, 0], [0.0, 4.0], atol=1e-6)


def test_build_rendered_keypoint_selector_samples_mines_hard_negatives() -> None:
    query_desc = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    render_desc = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    errors = np.asarray([[2.0, 64.0, 80.0]], dtype=np.float32)

    samples = build_rendered_keypoint_selector_samples(
        query_desc,
        render_desc,
        errors,
        RenderedKeypointSelectorSampleConfig(
            positive_threshold_px=16.0,
            negative_threshold_px=32.0,
            hard_negatives_per_keypoint=1,
            max_positives_per_keypoint=1,
            keypoint_stride_px=16.0,
        ),
    )

    assert samples.sample_count == 1
    assert np.allclose(samples.positive_features[0, 0], render_desc[0])
    assert np.allclose(samples.negative_features[0, 0], render_desc[1])
    assert np.allclose(samples.positive_reprojection_distances[0, 0], 2.0 / 16.0)
    assert np.allclose(samples.negative_reprojection_distances[0, 0], 64.0 / 16.0)


def test_project_gaussian_vfm_field_features_preserves_geometry_and_metadata() -> None:
    field = GaussianVFMField(
        xyz=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        opacity=np.asarray([0.5, 0.7], dtype=np.float32),
        scale=np.asarray([0.1, 0.2], dtype=np.float32),
        gaussian_indices=np.asarray([10, 11], dtype=np.int64),
        nearest_track_ids=np.asarray([-1, -1], dtype=np.int64),
        support_counts=np.asarray([3, 4], dtype=np.int64),
        mean_distances=np.asarray([0.2, 0.3], dtype=np.float32),
        metadata={"source": "unit"},
    )
    selector = ResidualGatedPatchSelector(input_dim=4, output_dim=2, group_size=2)
    run = SafePatchSelectorTrainingRun(
        model=selector,
        active_group_mask=np.ones((2,), dtype=np.float32),
        summary=SafePatchSelectorTrainingSummary(
            selector_arch="residual_gated",
            initial_loss=0.0,
            final_loss=0.0,
            raw_train_top1_acc=0.0,
            raw_eval_top1_acc=0.0,
            train_top1_acc=0.0,
            eval_top1_acc=0.0,
            inlier_train_accuracy=0.0,
            inlier_eval_accuracy=0.0,
            sample_count=0,
            train_sample_count=0,
            eval_sample_count=0,
            input_dim=4,
            output_dim=2,
            residual_hidden_dim=256,
            transformer_dim=0,
            transformer_layers=0,
            transformer_heads=0,
            transformer_ff_dim=0,
            transformer_dropout=0.0,
            steps=0,
            batch_size=1,
            group_size=2,
            group_count=2,
            active_group_count=2,
            active_channel_count=4,
            active_group_fraction=1.0,
            gate_mean=1.0,
            gate_min=1.0,
            gate_max=1.0,
            parameter_count=0,
            inlier_loss_weight=0.0,
            group_lasso_weight=0.0,
            hard_gate_keep_fraction=1.0,
        ),
    )

    projected = project_gaussian_vfm_field_features(field, run, device="cpu", batch_size=2)

    assert projected.feature_dim == 2
    assert np.allclose(projected.xyz, field.xyz)
    assert np.array_equal(projected.gaussian_indices, field.gaussian_indices)
    assert projected.metadata["source"] == "unit"
    assert projected.metadata["selector_output_dim"] == 2
