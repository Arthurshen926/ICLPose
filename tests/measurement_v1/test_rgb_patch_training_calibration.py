from __future__ import annotations

import torch
import pytest

from feature_extract.vfm.measurement_v1.rgb_patch_training import _binary_ece


def test_binary_ece_is_low_for_calibrated_probability_bins() -> None:
    probabilities = torch.tensor([0.1, 0.1, 0.9, 0.9], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.bool)

    assert _binary_ece(probabilities, labels, bin_count=2) == pytest.approx(0.1)


def test_binary_ece_is_high_for_overconfident_wrong_predictions() -> None:
    probabilities = torch.tensor([0.9, 0.9, 0.1, 0.1], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.bool)

    assert _binary_ece(probabilities, labels, bin_count=2) == pytest.approx(0.9)
