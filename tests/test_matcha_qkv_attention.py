from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.matcha_qkv_attention import (
    QKVAttentionTrainingConfig,
    ResidualQKVCrossAttention,
    apply_qkv_attention_to_feature_maps,
    load_qkv_attention_checkpoint,
    save_qkv_attention_checkpoint,
    train_qkv_attention,
)


def test_qkv_cross_attention_preserves_shape_and_normalizes() -> None:
    model = ResidualQKVCrossAttention(input_dim=4, attention_dim=4, alpha=0.25)
    query = np.random.default_rng(0).normal(size=(4, 2, 3)).astype(np.float32)
    render = np.random.default_rng(1).normal(size=(4, 2, 2)).astype(np.float32)

    query_out, render_out = apply_qkv_attention_to_feature_maps(model, query, render, device="cpu")

    assert query_out.shape == query.shape
    assert render_out.shape == render.shape
    assert np.allclose(np.linalg.norm(query_out.reshape(4, -1), axis=0), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(render_out.reshape(4, -1), axis=0), 1.0, atol=1e-5)


def test_qkv_attention_checkpoint_round_trip(tmp_path) -> None:
    model = ResidualQKVCrossAttention(input_dim=3, attention_dim=5, alpha=0.1)
    with torch.no_grad():
        model.query_proj.weight.fill_(0.2)
        model.render_proj.weight.fill_(0.3)
    checkpoint = tmp_path / "qkv.pt"

    save_qkv_attention_checkpoint(model, checkpoint, summary={"toy": True})
    loaded, summary = load_qkv_attention_checkpoint(checkpoint, device="cpu")

    assert summary["toy"] is True
    assert loaded.input_dim == 3
    assert loaded.attention_dim == 5
    assert loaded.alpha == 0.1
    query = np.random.default_rng(2).normal(size=(3, 1, 2)).astype(np.float32)
    render = np.random.default_rng(3).normal(size=(3, 1, 2)).astype(np.float32)
    before = apply_qkv_attention_to_feature_maps(model, query, render, device="cpu")
    after = apply_qkv_attention_to_feature_maps(loaded, query, render, device="cpu")
    assert np.allclose(before[0], after[0], atol=1e-6)
    assert np.allclose(before[1], after[1], atol=1e-6)


def test_train_qkv_attention_learns_toy_candidate_ranking() -> None:
    features = np.eye(6, dtype=np.float32)
    negatives = np.roll(features, shift=1, axis=0)[:, None, :]

    run = train_qkv_attention(
        features,
        features,
        negatives,
        QKVAttentionTrainingConfig(
            attention_dim=6,
            alpha=0.25,
            steps=80,
            batch_size=6,
            lr=1e-3,
            temperature=0.07,
            device="cpu",
            seed=11,
        ),
    )

    assert run.summary["train_top1_acc"] >= 0.99
    assert run.summary["final_loss"] <= run.summary["initial_loss"]
