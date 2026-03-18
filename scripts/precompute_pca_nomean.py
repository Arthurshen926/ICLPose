#!/usr/bin/env python3
"""Pre-compute PCA-no-mean features for both real and rendered images.

PCA-no-mean: project features onto PCA components WITHOUT mean subtraction.
This preserves cross-domain correlation much better:
  - PCA with mean: cosine ~0.0 (real vs rendered)
  - PCA no mean:   cosine ~0.73 (real vs rendered, coarse)

Two output directories:
  1. query features (from real images, using stored raw features)
  2. reference features (from 3DGS-rendered images, freshly extracted)

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/precompute_pca_nomean.py \
        --raw_dir output/features_multiscale_stride7/OldHospital \
        --pca_dir output/features_multiscale_pca/OldHospital_indexed/pca_params \
        --traj_path output/features_multiscale_pca/OldHospital_indexed/traj_w_c.txt \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --query_out output/features_pca_nomean/OldHospital_indexed \
        --ref_out output/features_pca_nomean_rendered/OldHospital_indexed
"""
import sys, os, argparse, shutil, re
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


SCALE_RAW_DIMS = {
    'coarse': '512x15x26',
    'mid': '512x30x53',
    'fine_sd': '512x69x121',
    'fine_dino': '768x69x121',
}


def load_trajectory(traj_path):
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def pca_nomean_project(feat_tensor, components):
    """Project features onto PCA components WITHOUT mean subtraction.

    Args:
        feat_tensor: [C, H, W] raw feature tensor
        components: [D_pca, C] PCA components
    Returns:
        [D_pca, H, W] projected features
    """
    C, H, W = feat_tensor.shape
    flat = feat_tensor.reshape(C, -1).T          # [HW, C]
    projected = flat @ components.T               # [HW, D_pca] — NO mean subtraction
    D_pca = projected.shape[1]
    return projected.T.reshape(D_pca, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True,
                        help='Dir with stored raw features (features_multiscale_stride7/OldHospital)')
    parser.add_argument('--pca_dir', required=True,
                        help='Dir with PCA params (pca_params/)')
    parser.add_argument('--traj_path', required=True)
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--query_out', required=True,
                        help='Output dir for query (real) PCA-nomean features')
    parser.add_argument('--ref_out', required=True,
                        help='Output dir for reference (rendered) PCA-nomean features')
    parser.add_argument('--dino_stride', type=int, default=7)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=1080)
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load PCA components (no mean needed)
    print("Loading PCA components...")
    pca_comp = {}
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        d = np.load(f"{args.pca_dir}/{scale}_pca.npz")
        pca_comp[scale] = torch.from_numpy(d['components']).float().to(device)
        print(f"  {scale}: components {pca_comp[scale].shape}")

    # ===============================
    # Phase 1: Query features (real)
    # ===============================
    print("\n=== Phase 1: Query features from stored raw (real) ===")
    query_out = Path(args.query_out)
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        (query_out / scale).mkdir(parents=True, exist_ok=True)

    # Copy trajectory
    shutil.copy2(args.traj_path, query_out / 'traj_w_c.txt')

    # Get sorted raw files and build index
    raw_files = {}
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        d = Path(args.raw_dir) / scale
        files = sorted(d.glob('*.pt'))
        raw_files[scale] = files
        print(f"  {scale}: {len(files)} files")

    n_frames = len(raw_files['coarse'])

    for idx in tqdm(range(n_frames), desc="Query PCA-nomean"):
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            raw = torch.load(raw_files[scale][idx], map_location=device)
            projected = pca_nomean_project(raw, pca_comp[scale]).cpu()
            D, H, W = projected.shape
            fname = f"rgb_{idx}_{scale}_{D}x{H}x{W}.pt"
            torch.save(projected, query_out / scale / fname)

    # Also save PCA params (components only, no mean) for eval
    pca_out = query_out / 'pca_params'
    pca_out.mkdir(exist_ok=True)
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        np.savez(pca_out / f'{scale}_pca.npz',
                 components=pca_comp[scale].cpu().numpy(),
                 mean=np.zeros_like(pca_comp[scale][0].cpu().numpy()))  # zero mean

    print(f"  Query features saved to {query_out}")

    # ===============================
    # Phase 2: Reference features (rendered)
    # ===============================
    print("\n=== Phase 2: Reference features from rendered images ===")
    ref_out = Path(args.ref_out)
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        (ref_out / scale).mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.traj_path, ref_out / 'traj_w_c.txt')

    # Load poses
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
    K = torch.tensor(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
        dtype=torch.float32, device=device
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
                viewmats=pose_w2c[None], Ks=K[None],
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

        # PCA-nomean and save
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            feat = getattr(ms, scale).to(device)
            projected = pca_nomean_project(feat, pca_comp[scale]).cpu()
            D, H, W = projected.shape
            fname = f"rgb_{idx}_{scale}_{D}x{H}x{W}.pt"
            torch.save(projected, ref_out / scale / fname)

    # Copy PCA params to ref dir too
    pca_ref_out = ref_out / 'pca_params'
    pca_ref_out.mkdir(exist_ok=True)
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        np.savez(pca_ref_out / f'{scale}_pca.npz',
                 components=pca_comp[scale].cpu().numpy(),
                 mean=np.zeros_like(pca_comp[scale][0].cpu().numpy()))

    print(f"\n  Reference features saved to {ref_out}")
    print("\nDone!")


if __name__ == '__main__':
    main()
