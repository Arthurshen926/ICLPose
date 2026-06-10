from __future__ import annotations

import torch

from feature_extract.vfm.matcha_rgb_keypoint_detector import (
    MatchaRgbKeypointDetector,
    matcha_alike_distillation_loss,
    matcha_keypoint_position_loss,
)


def test_detector_outputs_original_matcha_65_bins_per_8_pixel_cell() -> None:
    detector = MatchaRgbKeypointDetector()
    image = torch.rand(2, 3, 32, 40)

    logits = detector(image)

    assert logits.shape == (2, 65, 4, 5)


def test_alike_distillation_uses_original_positive_divided_dustbin_sampling() -> None:
    logits = torch.zeros(1, 65, 1, 4)
    labels = torch.full((1, 1, 4), 64, dtype=torch.long)
    labels[0, 0, 0] = 5
    logits[0, 5, 0, 0] = 10.0

    loss, metrics = matcha_alike_distillation_loss(logits, labels, non_keypoint_divisor=32, seed=7)

    assert metrics["positive_count"] == 1
    assert metrics["non_keypoint_count"] == 0
    assert metrics["acc"] == 1.0
    assert float(loss.item()) < 1e-2


def test_alike_distillation_skips_images_without_alike_keypoints() -> None:
    logits = torch.zeros(1, 65, 2, 2, requires_grad=True)
    labels = torch.full((1, 2, 2), 64, dtype=torch.long)

    loss, metrics = matcha_alike_distillation_loss(logits, labels, non_keypoint_divisor=32, seed=7)

    assert float(loss.item()) == 0.0
    assert metrics["positive_count"] == 0
    assert metrics["non_keypoint_count"] == 0
    loss.backward()
    assert logits.grad is not None


def test_geometric_keypoint_position_loss_handles_batched_correspondences() -> None:
    source_logits = torch.full((2, 65, 1, 1), -10.0)
    target_logits = torch.full((2, 65, 1, 1), -10.0)
    source_logits[:, 0, 0, 0] = 10.0
    target_logits[0, 9, 0, 0] = 10.0
    target_logits[1, 18, 0, 0] = 10.0
    source_points = torch.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    target_points = torch.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    batches = torch.asarray([0, 1], dtype=torch.long)

    loss, acc, metrics = matcha_keypoint_position_loss(
        source_logits,
        target_logits,
        source_points,
        target_points,
        point_batch_indices=batches,
    )

    assert metrics["valid_count"] == 2
    assert metrics["source_candidate_count"] == 2
    assert acc == 1.0
    assert float(loss.item()) < 1e-3
