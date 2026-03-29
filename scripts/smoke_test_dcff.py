"""
Smoke test for DCFF (Deferred Cascaded Feature Field).

Creates synthetic data and runs a few training iterations to verify
the entire pipeline works: 2DGS rasterization → latent rendering →
fine/coarse decode → loss computation → backward → optimizer step.

Usage:
    python scripts/smoke_test_dcff.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcff.hybrid_gaussian import HybridGaussianModel
from dcff.hash_grid import SpatialHashGrid
from dcff.deferred_renderer import DeferredCascadedRenderer
from dcff.losses import DCFFLoss


def create_synthetic_gaussians(n_points=1000, latent_dim=16):
    """Create a simple synthetic Gaussian model."""
    model = HybridGaussianModel(sh_degree=0, latent_dim=latent_dim)

    xyz = (torch.rand(n_points, 3) * 4 - 2).numpy()  # [-2, 2]
    colors = torch.rand(n_points, 3).numpy()

    model.create_from_pcd(xyz, colors, spatial_lr_scale=4.0)
    return model


def main():
    print("=" * 60)
    print("  DCFF Smoke Test")
    print("=" * 60)

    device = "cuda"
    latent_dim = 16
    feature_dim = 64
    H, W = 68, 120  # RADIO-like resolution
    img_H, img_W = 270, 480  # Rendering resolution

    # ── 1. Create models ──
    print("\n[1] Creating models...")
    gaussians = create_synthetic_gaussians(n_points=2000, latent_dim=latent_dim)

    import argparse
    args = argparse.Namespace(
        position_lr_init=0.001, position_lr_final=0.00001,
        feature_lr=0.0025, opacity_lr=0.05,
        scaling_lr=0.005, rotation_lr=0.001,
        latent_lr=0.0005, percent_dense=0.01,
        iterations=100,
    )
    gaussians.training_setup(args)
    print(f"  Gaussians: {gaussians.num_points:,}, latent_dim={latent_dim}")

    hash_grid = SpatialHashGrid(
        scene_extent=3.0, feature_dim=feature_dim,
        latent_dim=latent_dim, n_levels=8,
        log2_hashmap_size=16, max_resolution=512,
    ).to(device)

    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid, latent_dim=latent_dim,
        fine_feature_dim=feature_dim, coarse_feature_dim=feature_dim,
    ).to(device)

    loss_fn = DCFFLoss()
    print("  ✓ All models created")

    # ── 2. Synthetic targets ──
    print("\n[2] Creating synthetic targets...")
    gt_rgb = torch.rand(1, 3, img_H, img_W, device=device)
    radio_geo = torch.randn(1, feature_dim, H, W, device=device)
    radio_sem = torch.randn(1, feature_dim, H, W, device=device)

    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = -5.0  # Camera at z=-5 looking forward
    K = torch.tensor([
        [500, 0, img_W / 2],
        [0, 500, img_H / 2],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

    # ── 3. Forward pass (all phases) ──
    for phase in [1, 2, 3]:
        print(f"\n[3.{phase}] Testing Phase {phase} forward pass...")
        try:
            result = renderer(
                gaussians, viewmat=viewmat, K=K,
                width=img_W, height=img_H,
                render_coarse=(phase >= 3),
                feature_height=H, feature_width=W,
            )
            print(f"  RGB:    {result['rgb'].shape}")
            print(f"  Depth:  {result['depth'].shape}, range=[{result['depth'].min():.2f}, {result['depth'].max():.2f}]")
            print(f"  Alpha:  {result['alpha'].shape}, mean={result['alpha'].mean():.3f}")
            print(f"  Z_map:  {result['z_map'].shape}")
            print(f"  Fine:   {result['fine_features'].shape}")
            if result['coarse_features'] is not None:
                print(f"  Coarse: {result['coarse_features'].shape}")

            losses = loss_fn.compute(
                render_result=result, gt_rgb=gt_rgb,
                radio_geo=radio_geo if phase >= 2 else None,
                radio_sem=radio_sem if phase >= 3 else None,
                hash_grid=hash_grid if phase >= 3 else None,
                phase=phase,
            )
            print(f"  Total loss: {losses['total'].item():.4f}")
            for k, v in losses.items():
                if k != 'total' and isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.item():.4f}")
            print(f"  ✓ Phase {phase} OK")

        except Exception as e:
            print(f"  ✗ Phase {phase} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return

    # ── 4. Training loop test ──
    print(f"\n[4] Testing training loop (5 iterations, Phase 3)...")
    dcff_optimizer = torch.optim.Adam([
        {'params': hash_grid.parameters(), 'lr': 1e-3},
        {'params': renderer.fine_decoder.parameters(), 'lr': 1e-4},
    ])

    losses_history = []
    for i in range(5):
        result = renderer(
            gaussians, viewmat=viewmat, K=K,
            width=img_W, height=img_H,
            render_coarse=True,
            feature_height=H, feature_width=W,
        )

        losses = loss_fn.compute(
            render_result=result, gt_rgb=gt_rgb,
            radio_geo=radio_geo, radio_sem=radio_sem,
            hash_grid=hash_grid, phase=3,
        )

        losses['total'].backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        dcff_optimizer.step()
        dcff_optimizer.zero_grad(set_to_none=True)

        losses_history.append(losses['total'].item())
        print(f"  Iter {i+1}: total={losses['total'].item():.4f}, "
              f"N={gaussians.num_points}")

    # Check loss is not NaN/Inf
    for l in losses_history:
        assert not (np.isnan(l) or np.isinf(l)), f"NaN/Inf loss detected: {l}"
    print(f"  ✓ All iterations completed, no NaN/Inf")

    # ── 5. Memory report ──
    print(f"\n[5] Memory report:")
    print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"  GPU cached:    {torch.cuda.memory_reserved()/1e6:.1f} MB")

    n_total = (sum(p.numel() for p in hash_grid.parameters()) +
               sum(p.numel() for p in renderer.fine_decoder.parameters()) +
               gaussians.num_points * (3 + 4 + 2 + 1 + latent_dim))
    print(f"  Total params:  {n_total:,}")

    print(f"\n{'='*60}")
    print(f"  ✓ DCFF SMOKE TEST PASSED")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
