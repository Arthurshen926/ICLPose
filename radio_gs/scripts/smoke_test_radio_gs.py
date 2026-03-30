#!/usr/bin/env python3
"""End-to-end smoke test for RADIO-GS framework (CPU-only, no real data needed).

Tests all modules: HCD codec, both architectures, FeatSharp-3D, task heads,
losses, config system, and a mini training loop.

Usage:
    python radio_gs/scripts/smoke_test_radio_gs.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. HCD Codec roundtrip
# ---------------------------------------------------------------------------

def test_hcd_codec():
    from radio_gs.models.hcd_codec import HCDCodec

    codec = HCDCodec(input_dim=1280, bottleneck_dim=64, dual_stream=True)
    x = torch.randn(2, 1280, 8, 10)

    with torch.no_grad():
        z = codec.encode(x)
        x_hat = codec.decode(z)

    assert z.shape == (2, 64, 8, 10), f"Encoded shape mismatch: {z.shape}"
    assert x_hat.shape == (2, 1280, 8, 10), f"Decoded shape mismatch: {x_hat.shape}"

    # Full forward pass (encode → decode)
    with torch.no_grad():
        x_rt = codec(x)
    assert x_rt.shape == x.shape, f"Roundtrip shape mismatch: {x_rt.shape}"

    compression = codec.compression_ratio
    assert compression == 1280 / 64, f"Compression ratio wrong: {compression}"

    print(f"    encode: {tuple(x.shape)} → {tuple(z.shape)}")
    print(f"    decode: {tuple(z.shape)} → {tuple(x_hat.shape)}")
    print(f"    compression ratio: {compression:.1f}×")


# ---------------------------------------------------------------------------
# 2. Explicit Gaussian model
# ---------------------------------------------------------------------------

def test_explicit_gaussian():
    from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian

    N = 100
    latent_dim = 64
    model = ExplicitFeatureGaussian(latent_dim=latent_dim)

    # Manually populate geometry buffers (no PLY needed)
    model._xyz = torch.randn(N, 3)
    model._rotation = torch.randn(N, 4)
    model._scaling = torch.randn(N, 3)
    model._opacity = torch.randn(N, 1)
    model._features_dc = torch.randn(N, 1, 3)

    # Initialise learnable features
    model.init_features_random()

    assert model.num_gaussians == N, f"num_gaussians={model.num_gaussians}"
    assert model.get_xyz().shape == (N, 3)
    assert model.get_features().shape == (N, latent_dim)
    assert model.get_rotation().shape == (N, 4)
    assert model.get_scaling().shape == (N, 3)
    assert model.get_opacity().shape == (N, 1)

    # Features should be L2-normalised
    feat_norms = model.get_features().norm(dim=-1)
    assert torch.allclose(feat_norms, torch.ones(N), atol=1e-5), "Features not L2-normalised"

    params = model.trainable_parameters()
    assert len(params) == 1, f"Expected 1 trainable parameter group, got {len(params)}"
    total_trainable = sum(p.numel() for p in params)
    assert total_trainable == N * latent_dim, f"Trainable params: {total_trainable}"

    print(f"    {N} Gaussians, latent_dim={latent_dim}")
    print(f"    trainable params: {total_trainable:,}")
    print(f"    get_features shape: {tuple(model.get_features().shape)}")


# ---------------------------------------------------------------------------
# 3. Hybrid Gaussian model
# ---------------------------------------------------------------------------

def test_hybrid_gaussian():
    from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian

    N = 100
    latent_dim = 16
    output_dim = 128
    model = HybridFeatureGaussian(
        latent_dim=latent_dim,
        hash_output_dim=48,
        output_dim=output_dim,
        num_levels=4,          # fewer levels for speed
        log2_hashmap_size=10,  # smaller hash table for test
    )

    # Manually populate geometry buffers
    model._xyz = torch.randn(N, 3)
    model._rotation = torch.randn(N, 4)
    model._scaling = torch.randn(N, 3)
    model._opacity = torch.randn(N, 1)
    model._features_dc = torch.randn(N, 3)

    # Init learnable latent codes
    model._latent = nn.Parameter(torch.randn(N, latent_dim) * 0.01)

    assert model.num_gaussians == N
    # get_xyz is a property on HybridFeatureGaussian (not a method)
    assert model.get_xyz.shape == (N, 3)
    assert model.get_features().shape == (N, latent_dim)

    params = model.trainable_parameters()
    total_trainable = sum(p.numel() for p in params)
    assert total_trainable > 0, "No trainable parameters found"

    # Test screen-space decode forward pass
    B, H, W = 2, 8, 10
    latent_map = torch.randn(B, latent_dim, H, W)
    # Normalise positions to [0, 1] for hash grid
    position_map = torch.rand(B, 3, H, W)

    with torch.no_grad():
        out = model.decode_screen_space(latent_map, position_map)
    assert out.shape == (B, output_dim, H, W), f"Output shape: {out.shape}"

    print(f"    {N} Gaussians, latent_dim={latent_dim}, output_dim={output_dim}")
    print(f"    trainable params: {total_trainable:,}")
    print(f"    decode_screen_space: {tuple(latent_map.shape)} → {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# 4. FeatSharp-3D (all modes)
# ---------------------------------------------------------------------------

def test_featsharp():
    from radio_gs.models.featsharp_3d import FeatSharp3D

    B, C, H, W = 2, 64, 8, 10
    feat = torch.randn(B, C, H, W)

    # --- Analytical mode ---
    fs_analytical = FeatSharp3D(mode='analytical', feature_dim=C)
    with torch.no_grad():
        out_a = fs_analytical(feat)
    assert out_a.shape == (B, C, H, W), f"analytical: {out_a.shape}"
    print(f"    analytical: {tuple(feat.shape)} → {tuple(out_a.shape)}")

    # --- Learned mode ---
    fs_learned = FeatSharp3D(mode='learned', feature_dim=C)
    with torch.no_grad():
        out_l = fs_learned(feat)
    assert out_l.shape == (B, C, H, W), f"learned: {out_l.shape}"
    n_params = sum(p.numel() for p in fs_learned.parameters())
    print(f"    learned: {tuple(feat.shape)} → {tuple(out_l.shape)} ({n_params:,} params)")

    # --- Multiview mode ---
    fs_mv = FeatSharp3D(mode='multiview', feature_dim=C)
    depth = torch.rand(B, H, W) * 5.0 + 0.1
    viewmat_ref = torch.eye(4).unsqueeze(0).expand(B, -1, -1).contiguous()
    # Create a slightly offset source viewmat
    viewmat_src = viewmat_ref.clone()
    viewmat_src[:, 0, 3] = 0.1  # small translation
    K = torch.tensor([[32., 0., W / 2.], [0., 32., H / 2.], [0., 0., 1.]])

    feat_src = torch.randn(B, C, H, W)
    with torch.no_grad():
        out_mv = fs_mv(
            feat,
            features_sources=[feat_src],
            depths_ref=depth,
            viewmats_ref=viewmat_ref,
            viewmats_sources=[viewmat_src],
            K=K,
        )
    assert out_mv.shape == (B, C, H, W), f"multiview: {out_mv.shape}"
    n_params_mv = sum(p.numel() for p in fs_mv.parameters())
    print(f"    multiview: {tuple(feat.shape)} → {tuple(out_mv.shape)} ({n_params_mv:,} params)")


# ---------------------------------------------------------------------------
# 5. Task heads
# ---------------------------------------------------------------------------

def test_task_heads():
    from radio_gs.heads.depth_head import DepthHead
    from radio_gs.heads.segmentation_head import SegmentationHead
    from radio_gs.heads.grounding_head import GroundingHead

    B, C, H, W = 2, 1280, 8, 10
    feat = torch.randn(B, C, H, W)

    # --- Depth head ---
    depth_head = DepthHead(feature_dim=C, head_type='mlp')
    with torch.no_grad():
        depth_out = depth_head(feat)
    assert depth_out.shape == (B, 1, H, W), f"DepthHead: {depth_out.shape}"
    assert (depth_out > 0).all(), "Depth should be positive"
    print(f"    DepthHead(mlp): {tuple(feat.shape)} → {tuple(depth_out.shape)}")

    # --- Segmentation head ---
    num_classes = 40
    seg_head = SegmentationHead(feature_dim=C, num_classes=num_classes)
    with torch.no_grad():
        seg_out = seg_head(feat)
    assert seg_out.shape == (B, num_classes, H, W), f"SegHead: {seg_out.shape}"
    print(f"    SegmentationHead: {tuple(feat.shape)} → {tuple(seg_out.shape)}")

    # Segmentation loss
    gt_labels = torch.randint(0, num_classes, (B, H, W))
    seg_loss = nn.CrossEntropyLoss()(seg_out, gt_labels)
    assert seg_loss.ndim == 0, "Seg loss should be scalar"
    print(f"    seg loss: {seg_loss.item():.4f}")

    # --- Grounding head ---
    N_queries = 3
    grounding_head = GroundingHead(feature_dim=C, use_adaptor=False)
    text_emb = torch.randn(B, N_queries, C)
    with torch.no_grad():
        ground_out = grounding_head(feat, text_emb)
    assert ground_out.shape == (B, N_queries, H, W), f"GroundingHead: {ground_out.shape}"
    print(f"    GroundingHead: feat {tuple(feat.shape)} + text {tuple(text_emb.shape)} → {tuple(ground_out.shape)}")


# ---------------------------------------------------------------------------
# 6. Loss functions
# ---------------------------------------------------------------------------

def test_losses():
    from radio_gs.losses.distillation_loss import DistillationLoss, TotalVariationLoss

    B, C, H, W = 2, 1280, 8, 10
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randn(B, C, H, W)

    # --- DistillationLoss ---
    dist_loss = DistillationLoss(l2_weight=1.0, cosine_weight=0.5)
    result = dist_loss(pred, target)
    assert 'total' in result, f"Missing 'total' key; keys={list(result.keys())}"
    assert 'l2' in result
    assert 'cosine' in result
    assert result['total'].ndim == 0, "Total loss should be scalar"

    # Backward pass
    result['total'].backward()
    assert pred.grad is not None, "Gradient not computed"
    print(f"    DistillationLoss: total={result['total'].item():.4f}, "
          f"l2={result['l2'].item():.4f}, cosine={result['cosine'].item():.4f}")

    # --- TotalVariationLoss ---
    feat = torch.randn(B, 64, H, W, requires_grad=True)
    tv_loss = TotalVariationLoss()
    tv = tv_loss(feat)
    assert tv.ndim == 0, "TV loss should be scalar"
    tv.backward()
    assert feat.grad is not None, "TV gradient not computed"
    print(f"    TotalVariationLoss: {tv.item():.4f}")


# ---------------------------------------------------------------------------
# 7. Config system roundtrip
# ---------------------------------------------------------------------------

def test_config_roundtrip():
    from radio_gs.config import RadioGSConfig, load_config, save_config

    # Create default config and modify some fields
    cfg = RadioGSConfig(
        exp_name="smoke_test",
        latent_dim=32,
        dual_stream=False,
        lr_features=5e-4,
    )

    # Save → reload → compare
    tmp_path = os.path.join(os.path.dirname(__file__), '_smoke_test_config.yaml')
    try:
        save_config(cfg, tmp_path)
        loaded = load_config(tmp_path)

        assert loaded.exp_name == cfg.exp_name, \
            f"exp_name: {loaded.exp_name} != {cfg.exp_name}"
        assert loaded.latent_dim == cfg.latent_dim, \
            f"latent_dim: {loaded.latent_dim} != {cfg.latent_dim}"
        assert loaded.dual_stream == cfg.dual_stream, \
            f"dual_stream: {loaded.dual_stream} != {cfg.dual_stream}"
        assert loaded.lr_features == cfg.lr_features, \
            f"lr_features: {loaded.lr_features} != {cfg.lr_features}"
        assert loaded.seed == cfg.seed, \
            f"seed: {loaded.seed} != {cfg.seed}"

        print(f"    save/load roundtrip OK (file: {tmp_path})")
        print(f"    exp_name={loaded.exp_name}, latent_dim={loaded.latent_dim}, "
              f"dual_stream={loaded.dual_stream}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# 8. Mini training loop
# ---------------------------------------------------------------------------

def test_mini_training_loop():
    from radio_gs.models.hcd_codec import HCDCodec
    from radio_gs.models.featsharp_3d import FeatSharp3D
    from radio_gs.losses.distillation_loss import DistillationLoss

    B, C_in, C_compact, H, W = 2, 1280, 64, 30, 40

    codec = HCDCodec(input_dim=C_in, bottleneck_dim=C_compact, dual_stream=True)
    sharpener = FeatSharp3D(mode='learned', feature_dim=C_compact)
    loss_fn = DistillationLoss(l2_weight=1.0, cosine_weight=0.5)

    # Only train the decoder + sharpener (simulate distillation scenario)
    codec.freeze_encoder()
    optimizer = torch.optim.Adam(
        list(codec.decoder.parameters()) + list(sharpener.parameters()),
        lr=1e-3,
    )

    # Ground-truth RADIO features (fixed)
    gt_features = torch.randn(B, C_in, H, W)

    # Pre-compute GT compact codes (frozen encoder)
    with torch.no_grad():
        gt_compact = codec.encode(gt_features)

    losses = []
    num_iters = 3

    for i in range(num_iters):
        optimizer.zero_grad()

        # Simulate rendered compact features (GT compact + noise, decaying)
        noise_scale = 0.5 * (0.8 ** i)
        rendered_compact = gt_compact + torch.randn_like(gt_compact) * noise_scale

        # Sharpen the rendered compact features
        sharpened = sharpener(rendered_compact)

        # Decode back to 1280d
        decoded = codec.decode(sharpened)

        # Compute distillation loss
        result = loss_fn(decoded, gt_features)
        total_loss = result['total']

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        print(f"    iter {i+1}/{num_iters}: loss={total_loss.item():.4f}")

    # Verify loss decreased (with some tolerance for stochastic noise)
    assert losses[-1] < losses[0] + 0.1, \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"    loss trend: {losses[0]:.4f} → {losses[-1]:.4f} ({'↓ OK' if losses[-1] < losses[0] else '~ stable'})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_hcd_codec,
        test_explicit_gaussian,
        test_hybrid_gaussian,
        test_featsharp,
        test_task_heads,
        test_losses,
        test_config_roundtrip,
        test_mini_training_loop,
    ]

    passed, failed = 0, 0
    for t in tests:
        print(f"\n  Running {t.__name__}...")
        try:
            t()
            print(f'  ✓ {t.__name__}')
            passed += 1
        except Exception as e:
            traceback.print_exc()
            print(f'  ✗ {t.__name__}: {e}')
            failed += 1

    print(f'\n{"="*50}')
    print(f'{passed} passed, {failed} failed')
    sys.exit(failed)
