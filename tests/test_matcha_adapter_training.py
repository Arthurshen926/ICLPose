from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.matcha_adapter_training import (
    MatchaAdapterTrainingConfig,
    dual_softmax_descriptor_loss,
    train_matcha_dual_softmax_selector,
)
from feature_extract.vfm.patch_selector_training import PatchSelectorTrainingSet


def test_dual_softmax_descriptor_loss_prefers_aligned_pairs() -> None:
    query = torch.eye(4, dtype=torch.float32)
    positive = torch.eye(4, dtype=torch.float32)
    shuffled = positive[[1, 0, 3, 2]]

    aligned_loss = dual_softmax_descriptor_loss(query, positive, temperature=0.1)
    shuffled_loss = dual_softmax_descriptor_loss(query, shuffled, temperature=0.1)

    assert float(aligned_loss) < float(shuffled_loss)


def test_train_matcha_dual_softmax_selector_reduces_loss_on_toy_pairs() -> None:
    query = np.eye(4, dtype=np.float32)
    positives = query[:, None, :]
    negatives = np.stack([np.roll(query, shift=1, axis=0), np.roll(query, shift=2, axis=0)], axis=1)
    samples = PatchSelectorTrainingSet(
        query_features=query,
        positive_features=positives,
        positive_mask=np.ones((4, 1), dtype=bool),
        negative_features=negatives,
        positive_reprojection_distances=np.zeros((4, 1), dtype=np.float32),
        negative_reprojection_distances=np.ones((4, 2), dtype=np.float32) * 4.0,
    )

    run = train_matcha_dual_softmax_selector(
        samples,
        MatchaAdapterTrainingConfig(
            output_dim=4,
            steps=40,
            batch_size=4,
            lr=1e-2,
            temperature=0.1,
            dual_softmax_weight=1.0,
            hard_negative_weight=0.2,
            inlier_loss_weight=0.0,
            anchor_loss_weight=0.0,
            eval_split_fraction=0.0,
            group_size=2,
            device="cpu",
            seed=3,
        ),
    )

    assert run.summary.final_loss < run.summary.initial_loss
    assert run.summary.train_top1_acc >= 0.75
