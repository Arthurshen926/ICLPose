#!/usr/bin/env python3
"""Pre-compute channel-selected PCA-no-mean features for both real and rendered images.

Same pipeline as precompute_pca_nomean.py but applies channel selection first:
  raw 512d → select top-K channels → PCA-nomean on selected channels → compressed features

Two output directories:
  1. query features (from real stored raw features)
  2. reference features (from 3DGS-rendered images, freshly extracted)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_selected_pca.py \
        --raw_dir output/features_multiscale_stride7/OldHospital \
        --analysis_dir output/channel_analysis/OldHospital \
        --traj_path output/features_pca_nomean/OldHospital_indexed/traj_w_c.txt \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --query_out output/features_selected_pca/OldHospital_indexed \
        --ref_out output/features_selected_pca_rendered/OldHospital_indexed \
        --top_k 256
"""
import sys
import os
import argparse
import shutil

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import feature_3dgs.train_2dgs_geometry as _tmod


class _FakeArgs:
    init_from_depth = False
    sensor_depth_dir = None


_tmod.args = _FakeArgs()

from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS
from gsplat import rasterization_2dgs

SCALES = ['coarse', 'mid', 'fine_sd', 'fine_dino']

TARGET_DIMS = {
    'coarse': 32,
    'mid': 64,
    'fine_sd': 64,
    'fine_dino': 64,
}

# Raw feature dimensions before ODISE compression
# (extracted by scripts/extract_multiscale_features.py with stacking)
SCALE_RAW_DIMS = {
    'coarse': 512,
    'mid': 512,
    'fine_sd': 512,
    'fine_dino': 768,
}


def load_trajectory(traj_path):
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def selected_pca_nomean_project(feat_tensor, selected_channels, components):
    """Select channels then project via PCA-nomean.

    Args:
        feat_tensor: [C_raw, H, W] raw feature tensor
        selected_channels: list of int, channel indices to select
        components: [D_pca, K] PCA components (K = len(selected_channels))
    Returns:
        [D_pca, H, W] projected features
    """
    # Select channels
    feat_sel = feat_tensor[selected_channels]  # [K, H, W]
    K, H, W = feat_sel.shape
    flat = feat_sel.reshape(K, -1).T  # [HW, K]
    projected = flat @ components.T  # [HW, D_pca] — NO mean subtraction
    D_pca = projected.shape[1]
    return projected.T.reshape(D_pca, H, W)


def fit_pca_on_raw(raw_dir, scale, selected_channels, target_dim, max_samples=50000, max_files=200):
    """Fit PCA components on raw stored features (selected channels only)."""
    scale_dir = Path(raw_dir) / scale
    files = sorted(scale_dir.glob('*.pt'))

    # Subsample files
    if len(files) > max_files:
        indices = np.linspace(0, len(files) - 1, max_files, dtype=int)
        files = [files[i] for i in indices]

    all_pixels = []
    for fn in tqdm(files, desc=f"Loading {scale} for PCA", leave=False):
        feat = torch.load(fn, map_location='cpu').float()
        feat_sel = feat[selected_channels]  # [K, H, W]
        K, H, W = feat_sel.shape
        all_pixels.append(feat_sel.reshape(K, -1).T)  # [HW, K]

    X = torch.cat(all_pixels, dim=0).numpy()  # [N, K]

    rng = np.random.RandomState(42)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    print(f"  PCA-nomean SVD on {X_sub.shape}...")
    _, S, Vt = np.linalg.svd(X_sub, full_matrices=False)
    components = Vt[:target_dim]  # [target_dim, K]
    explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  Explained variance: {explained * 100:.1f}%")

    return components.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--analysis_dir', required=True)
    parser.add_argument('--traj_path', required=True)
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--query_out', required=True)
    parser.add_argument('--ref_out', required=True)
    parser.add_argument('--top_k', type=int, default=256)
    parser.add_argument('--dino_stride', type=int, default=7)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--skip_ref', action='store_true',
                        help='Skip reference (rendered) features, only compute query')
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load channel selections and fit PCA for each scale
    print("=== Loading channel selections and fitting PCA ===")
    selected_channels = {}
    pca_components = {}

    for scale in SCALES:
        scores_path = os.path.join(args.analysis_dir, f'{scale}_channel_scores.npz')
        data = np.load(scores_path)
        ranked = data['ranked_indices']
        combined = data['combined']

        n_ch = len(combined)
        k = min(args.top_k, n_ch)
        sel = sorted(ranked[:k].tolist())
        selected_channels[scale] = sel

        total_score = combined.sum()
        sel_score = combined[sel].sum()
        print(f"  {scale}: selected {k}/{n_ch} channels ({sel_score / total_score * 100:.1f}% importance)")

        # Fit PCA-nomean on selected channels
        target_dim = TARGET_DIMS[scale]
        components = fit_pca_on_raw(args.raw_dir, scale, sel, target_dim)
        pca_components[scale] = torch.from_numpy(components).float().to(device)
        print(f"  {scale}: PCA components {components.shape}")

    # ============================
    # Phase 1: Query features
    # ============================
    print("\n=== Phase 1: Query features from stored raw ===")
    query_out = Path(args.query_out)
    for scale in SCALES:
        (query_out / scale).mkdir(parents=True, exist_ok=True)

    traj_dest = query_out / 'traj_w_c.txt'
    if Path(args.traj_path).resolve() != traj_dest.resolve():
        shutil.copy2(args.traj_path, traj_dest)

    # Get sorted raw files
    raw_files = {}
    for scale in SCALES:
        d = Path(args.raw_dir) / scale
        raw_files[scale] = sorted(d.glob('*.pt'))
        print(f"  {scale}: {len(raw_files[scale])} files")

    n_frames = len(raw_files['coarse'])

    for idx in tqdm(range(n_frames), desc="Query selected-PCA"):
        for scale in SCALES:
            raw = torch.load(raw_files[scale][idx], map_location=device).float()
            projected = selected_pca_nomean_project(
                raw, selected_channels[scale], pca_components[scale]
            ).cpu()
            D, H, W = projected.shape
            fname = f"rgb_{idx}_{scale}_{D}x{H}x{W}.pt"
            torch.save(projected, query_out / scale / fname)

    # Save PCA params + channel selection
    pca_out = query_out / 'pca_params'
    pca_out.mkdir(exist_ok=True)
    for scale in SCALES:
        np.savez(
            pca_out / f'{scale}_pca.npz',
            components=pca_components[scale].cpu().numpy(),
            mean=np.zeros(len(selected_channels[scale]), dtype=np.float32),
            selected_channels=np.array(selected_channels[scale]),
            top_k=args.top_k,
        )

    print(f"  Query features saved to {query_out}")

    if args.skip_ref:
        print("\n  Skipping reference features (--skip_ref)")
        print("Done!")
        return

    # ============================
    # Phase 2: Reference features
    # ============================
    print("\n=== Phase 2: Reference features from rendered images ===")
    ref_out = Path(args.ref_out)
    for scale in SCALES:
        (ref_out / scale).mkdir(parents=True, exist_ok=True)

    ref_traj_dest = ref_out / 'traj_w_c.txt'
    if Path(args.traj_path).resolve() != ref_traj_dest.resolve():
        shutil.copy2(args.traj_path, ref_traj_dest)

    poses = load_trajectory(args.traj_path)
    assert len(poses) == n_frames, f"Pose count {len(poses)} != frame count {n_frames}"

    # Load 3DGS
    print("Loading 3DGS model...")
    gs = GaussianModel2DGS(sh_degree=3)
    gs.load_ply(args.ply_path)
    means = gs.get_xyz
    opacity = gs.get_opacity.squeeze(-1)
    scales_2d = gs.get_scaling
    scales = torch.cat([scales_2d, torch.ones(scales_2d.shape[0], 1, device=device)], dim=-1)
    rotations = gs.get_rotation
    colors = gs.get_features
    sh_degree = gs.active_sh_degree
    K_mat = torch.tensor(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )

    # Load feature extractor
    print("Loading feature extractor...")
    from feature_extraction.multiscale_extractor import MultiScaleFeatureExtractor
    extractor = MultiScaleFeatureExtractor(device=str(device), dino_stride=args.dino_stride)

    for idx in tqdm(range(n_frames), desc="Ref render+extract"):
        pose_c2w = poses[idx]
        pose_w2c = torch.from_numpy(np.linalg.inv(pose_c2w)).float().to(device)

        # Render RGB
        with torch.no_grad():
            render_out, *_ = rasterization_2dgs(
                means=means, quats=rotations, scales=scales,
                opacities=opacity, colors=colors,
                viewmats=pose_w2c[None], Ks=K_mat[None],
                width=args.img_width, height=args.img_height,
                packed=False, sh_degree=sh_degree,
                backgrounds=torch.zeros(1, 4, device=device),
                near_plane=0.01, far_plane=500,
                render_mode="RGB+ED",
            )

        rgb = render_out[0, :, :, :3]
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_np)

        # Extract features
        with torch.no_grad():
            ms = extractor.extract(pil_img)

        # Channel-select + PCA-nomean + save
        for scale in SCALES:
            feat = getattr(ms, scale).to(device).float()
            projected = selected_pca_nomean_project(
                feat, selected_channels[scale], pca_components[scale]
            ).cpu()
            D, H, W = projected.shape
            fname = f"rgb_{idx}_{scale}_{D}x{H}x{W}.pt"
            torch.save(projected, ref_out / scale / fname)

    # Save PCA params to ref dir too
    pca_ref_out = ref_out / 'pca_params'
    pca_ref_out.mkdir(exist_ok=True)
    for scale in SCALES:
        np.savez(
            pca_ref_out / f'{scale}_pca.npz',
            components=pca_components[scale].cpu().numpy(),
            mean=np.zeros(len(selected_channels[scale]), dtype=np.float32),
            selected_channels=np.array(selected_channels[scale]),
            top_k=args.top_k,
        )

    print(f"\n  Reference features saved to {ref_out}")
    print("\nDone!")


if __name__ == '__main__':
    main()
