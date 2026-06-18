from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineAdapter,
    MatchaCoarseFineTrainingConfig,
    MatchaCoarseFineTrainingSet,
    _dual_softmax_descriptor_loss_and_confidence,
    _fine_coordinate_loss_and_metrics,
    _loss,
    _mine_negative_render_features,
    build_matcha_coarse_fine_training_set,
    merge_matcha_coarse_fine_training_sets,
    predict_matcha_pair_heads_for_matches,
    project_feature_map_with_matcha_keypoint_logits,
    project_feature_map_with_matcha_adapter_full,
    project_feature_map_with_matcha_adapter,
    train_matcha_coarse_fine_adapter,
)
from feature_extract.vfm.matcha_keypoint_distillation import build_keypoint_label_map
from feature_extract.vfm.matcha_rgb_keypoint_detector import (
    MatchaRgbKeypointDetector,
    decode_keypoints_from_logits,
    keypoint_xy_to_feature_cell_indices,
    matcha_alike_distillation_loss,
)
from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervision
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


def _supervision() -> MatchaCoarseSupervision:
    return MatchaCoarseSupervision(
        query_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        render_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_xy=np.zeros((4, 2), dtype=np.float64),
        render_xy=np.zeros((4, 2), dtype=np.float64),
        query_offset_labels=np.asarray([0, 9, 18, 36], dtype=np.int64),
        render_offset_labels=np.asarray([36, 18, 9, 0], dtype=np.int64),
        roundtrip_errors_px=np.zeros((4,), dtype=np.float32),
    )


def test_build_matcha_coarse_fine_training_set_mines_negatives() -> None:
    query = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)
    render = query.copy()
    samples = build_matcha_coarse_fine_training_set(
        query,
        render,
        _supervision(),
        hard_negatives_per_match=2,
    )

    assert samples.sample_count == 4
    assert samples.query_features.shape == (4, 4)
    assert samples.render_features.shape == (4, 4)
    assert samples.negative_render_features.shape == (4, 2, 4)
    assert samples.query_offset_labels.tolist() == [0, 9, 18, 36]


def test_coarse_fine_negative_mining_excludes_patch_level_alternatives() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    render = np.asarray(
        [
            [0.90, 0.10],
            [1.00, 0.00],
            [0.00, 1.00],
        ],
        dtype=np.float32,
    )

    negatives = _mine_negative_render_features(
        query,
        render,
        np.asarray([0], dtype=np.int64),
        count=1,
        excluded_render_indices_by_query=(np.asarray([0, 1], dtype=np.int64),),
    )

    assert np.allclose(negatives[0, 0], render[2])


def test_matcha_coarse_fine_adapter_projects_feature_map_and_offset_logits() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=3, residual_hidden_dim=8, group_size=2)
    feature = np.random.default_rng(0).normal(size=(4, 2, 2)).astype(np.float32)

    descriptors, offset_logits = project_feature_map_with_matcha_adapter(adapter, feature, device="cpu", batch_size=2)

    assert descriptors.shape == (3, 2, 2)
    assert offset_logits.shape == (65, 2, 2)
    norms = np.linalg.norm(descriptors.reshape(3, -1), axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_matcha_adapter_full_projection_returns_detector_logits() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=3, residual_hidden_dim=8, group_size=2)
    feature = np.random.default_rng(1).normal(size=(4, 2, 2)).astype(np.float32)

    descriptors, offset_logits, detector_logits = project_feature_map_with_matcha_adapter_full(
        adapter,
        feature,
        device="cpu",
        batch_size=2,
    )

    assert descriptors.shape == (3, 2, 2)
    assert offset_logits.shape == (65, 2, 2)
    assert detector_logits.shape == (2, 2)
    projected_keypoint_logits = project_feature_map_with_matcha_keypoint_logits(adapter, feature, device="cpu", batch_size=2)
    assert projected_keypoint_logits.shape == (65, 2, 2)
    with torch.no_grad():
        keypoint_logits = adapter.keypoint_logits(torch.as_tensor(feature.reshape(4, -1).T))
    assert keypoint_logits.shape == (4, 65)


def test_build_keypoint_label_map_uses_64_as_non_keypoint() -> None:
    keypoints = np.asarray([[4.0, 4.0], [15.0, 15.0]], dtype=np.float32)
    labels, stats = build_keypoint_label_map(
        keypoints,
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        offset_bins=8,
    )

    assert labels.shape == (2, 2)
    assert labels[0, 0] == 36
    assert labels[1, 1] == 63
    assert labels[0, 1] == 64
    assert stats["positive_count"] == 2


def test_matcha_rgb_keypoint_detector_outputs_65_bins_per_8px_cell() -> None:
    detector = MatchaRgbKeypointDetector()
    image = torch.rand(2, 3, 32, 40)

    logits = detector(image)

    assert logits.shape == (2, 65, 4, 5)


def test_matcha_alike_distillation_loss_keeps_non_keypoint_dustbin() -> None:
    logits = torch.zeros(1, 65, 2, 2)
    logits[:, 64, :, :] = 3.0
    labels = torch.full((1, 2, 2), 64, dtype=torch.long)
    labels[0, 0, 0] = 36
    logits[0, 36, 0, 0] = 6.0

    loss, metrics = matcha_alike_distillation_loss(logits, labels, non_keypoint_divisor=1, seed=5)

    assert loss.item() < 1.0
    assert metrics["positive_count"] == 1
    assert metrics["non_keypoint_count"] == 1
    assert metrics["acc"] > 0.99


def test_decode_keypoints_from_logits_recovers_cell_offset() -> None:
    logits = torch.full((1, 65, 2, 2), -10.0)
    logits[:, 64, :, :] = 2.0
    logits[0, 36, 0, 0] = 8.0

    keypoints, scores, labels = decode_keypoints_from_logits(
        logits,
        image_width=16,
        image_height=16,
        top_k=4,
        threshold=0.1,
    )

    assert keypoints.shape[0] == 1
    assert labels.tolist() == [36]
    assert np.allclose(keypoints[0], [4.5, 4.5])
    assert scores[0] > 0.9


def test_keypoint_xy_to_feature_cell_indices_maps_detector_points_to_descriptor_grid() -> None:
    keypoints = np.asarray([[4.5, 4.5], [15.5, 7.5], [19.9, 9.9]], dtype=np.float32)

    indices = keypoint_xy_to_feature_cell_indices(
        keypoints,
        image_width=20,
        image_height=10,
        feature_grid_width=4,
        feature_grid_height=2,
    )

    assert indices.tolist() == [0, 7]


def test_matcha_adapter_pair_heads_score_matches_and_predict_pair_fine_logits() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    query = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)
    render = query.copy()
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([4.0, 4.0]),
            render_xy=np.asarray([4.0, 4.0]),
            similarity=0.9,
            ratio=0.1,
            dual_softmax_confidence=0.25,
        ),
        KeypointFeatureMatch(
            query_index=3,
            render_index=3,
            query_xy=np.asarray([12.0, 12.0]),
            render_xy=np.asarray([12.0, 12.0]),
            similarity=0.8,
            ratio=0.2,
            dual_softmax_confidence=0.2,
        ),
    ]

    confidences, fine_logits = predict_matcha_pair_heads_for_matches(
        adapter,
        query,
        render,
        matches,
        device="cpu",
    )

    assert confidences.shape == (2,)
    assert np.all((confidences >= 0.0) & (confidences <= 1.0))
    assert fine_logits.shape == (2, 64)


def test_matcha_adapter_pair_fine_head_outputs_64_coordinate_bins() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    descriptors = torch.eye(4, dtype=torch.float32)

    logits = adapter.pair_fine_logits(descriptors, descriptors)

    assert logits.shape == (4, 64)


def test_fine_coordinate_loss_ignores_dustbin_and_uses_confidence_weights() -> None:
    logits = torch.zeros(3, 64)
    logits[0, 1] = 4.0
    logits[2, 3] = -4.0
    labels = torch.asarray([1, 64, 3], dtype=torch.long)
    confidence = torch.asarray([0.9, 1.0, 0.1], dtype=torch.float32)

    loss, metrics = _fine_coordinate_loss_and_metrics(logits, labels, confidence=confidence)

    valid = labels < 64
    per_row = torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction="none")
    weights = confidence[valid] / confidence[valid].sum()
    assert loss is not None
    assert torch.allclose(loss, torch.sum(per_row * weights))
    assert not torch.allclose(loss, torch.mean(per_row))
    assert metrics["valid_count"] == 2.0
    assert metrics["acc"] == 1.0
    assert "epe_bins" in metrics
    assert "uncertainty_bins" in metrics
    assert metrics["epe_bins"] >= 0.0
    assert metrics["uncertainty_bins"] >= 0.0


def test_fine_coordinate_loss_trains_continuous_offset_and_learned_uncertainty() -> None:
    logits = torch.full((2, 64), -6.0)
    logits[0, 9] = 6.0
    logits[1, 18] = 6.0
    labels = torch.asarray([9, 18], dtype=torch.long)
    learned_log_sigma = torch.zeros((2,), dtype=torch.float32)

    loss, metrics = _fine_coordinate_loss_and_metrics(
        logits,
        labels,
        continuous_loss_weight=0.25,
        uncertainty_log_sigma=learned_log_sigma,
        uncertainty_loss_weight=0.1,
    )

    assert loss is not None
    assert metrics["continuous_epe_bins"] < 0.05
    assert metrics["learned_uncertainty_bins"] == pytest.approx(1.0)
    assert metrics["uncertainty_nll"] < 0.1


def test_adapter_pair_fine_loss_ignores_dustbin_labels() -> None:
    class _FineOnlyAdapter(torch.nn.Module):
        def forward(self, features):
            return features, torch.zeros(features.shape[0], 65, device=features.device)

        def pair_fine_logits(self, query_z, render_z):
            logits = torch.full((query_z.shape[0], 64), -10.0, device=query_z.device)
            logits[0, 1] = 10.0
            logits[1, 0] = 10.0
            logits[2, 2] = 10.0
            return logits

    features = torch.eye(3, dtype=torch.float32)
    labels = torch.asarray([1, 64, 2], dtype=torch.long)

    loss = _loss(
        _FineOnlyAdapter(),
        features,
        features,
        labels,
        labels,
        torch.zeros(3, 0, 3),
        MatchaCoarseFineTrainingConfig(
            output_dim=3,
            steps=1,
            batch_size=3,
            dual_softmax_weight=0.0,
            offset_loss_weight=0.0,
            pair_fine_loss_weight=1.0,
            confidence_loss_weight=0.0,
            detector_loss_weight=0.0,
            keypoint_loss_weight=0.0,
            hard_negative_weight=0.0,
            device="cpu",
        ),
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) < 1e-3


def test_train_matcha_coarse_fine_adapter_learns_toy_offsets() -> None:
    features = np.eye(8, dtype=np.float32)
    samples = MatchaCoarseFineTrainingSet(
        query_features=features,
        render_features=features,
        query_offset_labels=np.arange(8, dtype=np.int64),
        render_offset_labels=np.arange(8, dtype=np.int64),
        negative_render_features=np.roll(features, shift=1, axis=0)[:, None, :],
        roundtrip_errors_px=np.zeros((8,), dtype=np.float32),
        query_keypoint_features=features,
        query_keypoint_labels=np.asarray([0, 1, 2, 64, 64, 5, 6, 7], dtype=np.int64),
        render_keypoint_features=features,
        render_keypoint_labels=np.asarray([0, 1, 2, 64, 64, 5, 6, 7], dtype=np.int64),
        metadata={"toy": True},
    )

    run = train_matcha_coarse_fine_adapter(
        samples,
        MatchaCoarseFineTrainingConfig(
            output_dim=8,
            residual_hidden_dim=16,
            steps=120,
            batch_size=8,
            lr=5e-3,
            offset_loss_weight=1.0,
            keypoint_loss_weight=0.5,
            hard_negative_weight=0.0,
            eval_split_fraction=0.0,
            group_size=4,
            device="cpu",
            seed=7,
        ),
    )

    assert run.summary["train_top1_acc"] >= 0.99
    assert run.summary["query_offset_acc"] >= 0.75
    assert run.summary["render_offset_acc"] >= 0.75
    assert "query_keypoint_acc" in run.summary
    assert "query_keypoint_positive_acc" in run.summary
    assert "query_keypoint_non_keypoint_acc" in run.summary
    assert run.summary["query_keypoint_positive_count"] == 6
    assert run.summary["query_keypoint_non_keypoint_count"] == 2
    assert 0.0 <= run.summary["query_keypoint_confidence_mean"] <= 1.0
    with torch.no_grad():
        z, logits, detector_logits = run.model.forward_full(torch.as_tensor(features, dtype=torch.float32))
        keypoint_logits = run.model.keypoint_logits_from_descriptor(z)
        confidence_logits = run.model.pair_confidence_logits(z, z)
        pair_fine_logits = run.model.pair_fine_logits(z, z)
    assert logits.shape == (8, 65)
    assert detector_logits.shape == (8,)
    assert keypoint_logits.shape == (8, 65)
    assert confidence_logits.shape == (8,)
    assert pair_fine_logits.shape == (8, 64)


def test_matcha_style_detector_targets_use_dual_softmax_confidence() -> None:
    descriptors = torch.eye(4, dtype=torch.float32)

    loss, confidence = _dual_softmax_descriptor_loss_and_confidence(descriptors, descriptors, temperature=0.07)

    assert loss.item() < 1e-3
    assert confidence.shape == (4,)
    assert torch.all(confidence > 0.99)


def test_training_config_accepts_matcha_confidence_detector_target() -> None:
    config = MatchaCoarseFineTrainingConfig(detector_target_mode="matcha_confidence")

    assert config.detector_target_mode == "matcha_confidence"


def test_merge_matcha_training_sets_caps_and_records_sources() -> None:
    features = np.eye(4, dtype=np.float32)
    first = MatchaCoarseFineTrainingSet(
        query_features=features[:2],
        render_features=features[:2],
        query_offset_labels=np.asarray([1, 2], dtype=np.int64),
        render_offset_labels=np.asarray([2, 1], dtype=np.int64),
        negative_render_features=np.ones((2, 1, 4), dtype=np.float32),
        metadata={"source": "a"},
    )
    second = MatchaCoarseFineTrainingSet(
        query_features=features[2:],
        render_features=features[2:],
        query_offset_labels=np.asarray([3, 4], dtype=np.int64),
        render_offset_labels=np.asarray([4, 3], dtype=np.int64),
        negative_render_features=np.ones((2, 1, 4), dtype=np.float32),
        metadata={"source": "b"},
    )

    merged = merge_matcha_coarse_fine_training_sets([first, second], max_samples=3, seed=5)

    assert merged.sample_count == 3
    assert merged.query_features.shape == (3, 4)
    assert merged.negative_render_features.shape == (3, 1, 4)
    assert merged.metadata["merged"] is True
    assert merged.metadata["source_count"] == 2
