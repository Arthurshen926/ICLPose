from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.matcha_light_fusion import (
    AttentionFusionNet,
    MatchaLightAttentionBlock,
    MatchaLightDecoderRefinementBlock,
    MatchaLightFineMatcher,
    MatchaLightKeypointHead,
    maybe_fuse_feature_map,
    local_attention_fuse_feature_map,
)


def test_local_attention_fusion_preserves_shape_and_normalizes_cells() -> None:
    feature = np.zeros((2, 3, 3), dtype=np.float32)
    feature[0, :, :] = 1.0
    feature[1, 1, 1] = 1.0

    fused = local_attention_fuse_feature_map(feature, radius=1, temperature=5.0, alpha=0.5, device="cpu")

    assert fused.shape == feature.shape
    norms = np.linalg.norm(fused.reshape(2, -1), axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_local_attention_fusion_alpha_zero_returns_normalized_input() -> None:
    feature = np.random.default_rng(2).normal(size=(4, 2, 3)).astype(np.float32)

    fused = local_attention_fuse_feature_map(feature, radius=1, temperature=5.0, alpha=0.0, device="cpu")
    expected = feature.reshape(4, -1)
    expected = expected / np.maximum(np.linalg.norm(expected, axis=0, keepdims=True), 1e-8)

    assert np.allclose(fused.reshape(4, -1), expected, atol=1e-6)


def test_maybe_fuse_feature_map_supports_none_mode() -> None:
    feature = np.random.default_rng(3).normal(size=(4, 2, 3)).astype(np.float32)

    out = maybe_fuse_feature_map(feature, mode="none")

    assert out is feature


def test_attention_fusion_net_returns_original_matcha_like_maps() -> None:
    model = AttentionFusionNet(
        coarse_input_dim=3,
        fine_input_dim=5,
        output_dim=4,
        hidden_dim=8,
        decoder_depth=1,
        num_heads=2,
        patch_size=2,
        refinement_blocks=1,
    )
    coarse = torch.randn(2, 3, 2, 3)
    fine = torch.randn(2, 5, 4, 6)

    coarse_desc, fine_desc, heatmap_logits = model.forward_fuse_feature(coarse, fine)

    assert coarse_desc.shape == (2, 4, 4, 6)
    assert fine_desc.shape == (2, 4, 4, 6)
    assert heatmap_logits.shape == (2, 1, 4, 6)
    assert torch.allclose(torch.linalg.norm(coarse_desc, dim=1), torch.ones(2, 4, 6), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(fine_desc, dim=1), torch.ones(2, 4, 6), atol=1e-5)


def test_attention_fusion_net_exposes_joint_training_hooks() -> None:
    model = AttentionFusionNet(
        coarse_input_dim=4,
        fine_input_dim=4,
        output_dim=8,
        hidden_dim=16,
        decoder_depth=1,
        num_heads=4,
        patch_size=2,
        upsample_mode="pixel_shuffle",
        refinement_blocks=1,
        fine_matcher_hidden_dim=16,
    )
    feature_maps = torch.randn(1, 8, 4, 6)
    images = torch.randn(1, 3, 32, 48)

    descriptor_map, heatmap_logits, offset_logits = model.forward_feature_map(feature_maps)
    keypoint_logits = model.forward_rgb_keypoints(images)
    rows = descriptor_map.permute(0, 2, 3, 1).reshape(-1, 8)
    fine_logits = model.pair_fine_logits(rows[:4], rows[4:8])
    query_fine_logits = model.query_pair_fine_logits(rows[:4], rows[4:8])
    output = model(feature_maps=feature_maps, image=images)

    assert descriptor_map.shape == (1, 8, 4, 6)
    assert heatmap_logits.shape == (1, 1, 4, 6)
    assert offset_logits.shape == (1, 65, 4, 6)
    assert keypoint_logits.shape == (1, 65, 4, 6)
    assert fine_logits.shape == (4, 64)
    assert query_fine_logits.shape == (4, 64)
    assert output.fine_descriptors.shape == descriptor_map.shape
    assert output.descriptor_map.shape == descriptor_map.shape
    assert output.offset_logits is not None
    assert output.keypoint_logits is not None
    assert output.coarse_context is not None
    assert output.fine_context is not None
    assert output.as_matcha_tuple()[1].shape == descriptor_map.shape
    assert output.as_joint_training_tuple()[2].shape == offset_logits.shape


def test_attention_fusion_net_accepts_separate_feature_maps_in_forward() -> None:
    model = AttentionFusionNet(
        coarse_input_dim=2,
        fine_input_dim=3,
        output_dim=4,
        hidden_dim=8,
        decoder_depth=1,
        num_heads=2,
        patch_size=1,
        refinement_blocks=1,
    )
    output = model(
        feat_c=torch.randn(1, 2, 3, 5),
        feat_f=torch.randn(1, 3, 3, 5),
    )

    assert output.coarse_descriptors.shape == (1, 4, 3, 5)
    assert output.fine_descriptors.shape == (1, 4, 3, 5)
    assert output.heatmap_logits.shape == (1, 1, 3, 5)


def test_attention_fusion_net_validates_pixel_shuffle_hidden_channels() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        AttentionFusionNet(
            coarse_input_dim=4,
            fine_input_dim=4,
            output_dim=4,
            hidden_dim=10,
            patch_size=2,
            upsample_mode="pixel_shuffle",
        )


def test_light_attention_block_applies_self_and_cross_attention() -> None:
    block = MatchaLightAttentionBlock(hidden_dim=8, num_heads=2, mlp_ratio=2)
    fine_tokens = torch.randn(2, 5, 8)
    coarse_tokens = torch.randn(2, 3, 8)

    fine_out, coarse_out = block(fine_tokens, coarse_tokens)

    assert fine_out.shape == fine_tokens.shape
    assert coarse_out.shape == coarse_tokens.shape
    assert not torch.allclose(fine_out, fine_tokens)
    assert not torch.allclose(coarse_out, coarse_tokens)


def test_decoder_refinement_block_preserves_grid_and_changes_channels() -> None:
    block = MatchaLightDecoderRefinementBlock(in_channels=6, hidden_channels=10)

    refined = block(torch.randn(2, 6, 4, 5))

    assert refined.shape == (2, 10, 4, 5)


def test_keypoint_head_predicts_65_bins_per_8x8_window() -> None:
    head = MatchaLightKeypointHead(window_size=8, stem_channels=4, hidden_channels=8)

    logits = head(torch.randn(2, 3, 16, 24))

    assert logits.shape == (2, 65, 2, 3)


def test_fine_matcher_predicts_64_spatial_bins_from_pairs() -> None:
    matcher = MatchaLightFineMatcher(descriptor_dim=8, hidden_dim=16)

    logits = matcher(torch.randn(4, 8), torch.randn(4, 8))

    assert logits.shape == (4, 64)
