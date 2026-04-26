#!/usr/bin/env python3
"""Pre-compute render-extracted features for all training frames.

For each frame:
  1. Read GT pose from trajectory file
  2. Render RGB from 3DGS at that pose
  3. Extract SD+DINOv2 features from rendered image
  4. Apply PCA compression
  5. Save in the same format as stored PCA features

Output directory structure mirrors the existing PCA features:
  {output_dir}/{scale}/rgb_{idx}_{scale}_{CxHxW}.pt
  {output_dir}/traj_w_c.txt (copied)

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/precompute_render_extracted.py \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --pca_dir output/features_multiscale_pca/OldHospital_indexed/pca_params \
        --traj_path output/features_multiscale_pca/OldHospital_indexed/traj_w_c.txt \
        --output_dir output/features_render_extracted/OldHospital_indexed \
        --dino_stride 7
"""
import sys, os, argparse, shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Monkey-patch for load_scene
import feature_gaussian.legacy_3dgs.train_2dgs_geometry as _tmod
class _FakeArgs:
    init_from_depth = False
    sensor_depth_dir = None
_tmod.args = _FakeArgs()

from feature_gaussian.legacy_3dgs.train_2dgs_geometry import GaussianModel2DGS
from gsplat import rasterization_2dgs


def load_trajectory(traj_path):
    """Load poses from trajectory file. Each line = 16 floats (4x4 row-major)."""
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                pose = np.array(vals).reshape(4, 4)
                poses.append(pose)
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--pca_dir', required=True)
    parser.add_argument('--traj_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dino_stride', type=int, default=7)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start from this frame index (for resume)')
    parser.add_argument('--no_mean', action='store_true',
                        help='PCA without mean subtraction (PCA-nomean)')
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load poses
    poses = load_trajectory(args.traj_path)
    n_frames = len(poses)
    print(f"Loaded {n_frames} poses from {args.traj_path}")

    # Load 3DGS model
    print("Loading 3DGS model...")
    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(args.ply_path)

    means = gaussians.get_xyz
    opacity = gaussians.get_opacity.squeeze(-1)
    scales_2d = gaussians.get_scaling
    scales = torch.cat([scales_2d, torch.ones(scales_2d.shape[0], 1, device=device)], dim=-1)
    rotations = gaussians.get_rotation
    colors = gaussians.get_features
    sh_degree = gaussians.active_sh_degree

    K = torch.tensor(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
        dtype=torch.float32, device=device
    )
    print(f"  3DGS: {means.shape[0]} Gaussians")

    # Load feature extractor
    print("Loading feature extractor (SD + DINOv2)...")
    from feature_extraction.multiscale_extractor import MultiScaleFeatureExtractor
    extractor = MultiScaleFeatureExtractor(device=str(device), dino_stride=args.dino_stride)

    # Load PCA parameters
    print("Loading PCA parameters...")
    pca = {}
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        d = np.load(f"{args.pca_dir}/{scale}_pca.npz")
        pca[scale] = {
            'mean': torch.from_numpy(d['mean']).float().to(device),
            'components': torch.from_numpy(d['components']).float().to(device),
        }

    # Create output directories
    out_dir = Path(args.output_dir)
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        (out_dir / scale).mkdir(parents=True, exist_ok=True)

    # Copy trajectory file
    shutil.copy2(args.traj_path, out_dir / 'traj_w_c.txt')

    # Process each frame
    print(f"\nProcessing {n_frames} frames (starting from {args.start_idx})...")
    for idx in tqdm(range(args.start_idx, n_frames), desc="Render+Extract"):
        pose_c2w = poses[idx]
        pose_w2c = np.linalg.inv(pose_c2w)
        pose_w2c_t = torch.from_numpy(pose_w2c).float().to(device)

        # 1. Render RGB
        with torch.no_grad():
            render_out, *_ = rasterization_2dgs(
                means=means,
                quats=rotations,
                scales=scales,
                opacities=opacity,
                colors=colors,
                viewmats=pose_w2c_t[None],
                Ks=K[None],
                width=args.img_width,
                height=args.img_height,
                packed=False,
                sh_degree=sh_degree,
                backgrounds=torch.zeros(1, 4, device=device),
                near_plane=0.01, far_plane=500,
                render_mode="RGB+ED",
            )

        rgb = render_out[0, :, :, :3]  # [H, W, 3]
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_np)

        # 2. Extract features
        with torch.no_grad():
            ms = extractor.extract(pil_img)

        # 3. PCA compress and save
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            feat = getattr(ms, scale).to(device)  # [C, H, W]
            C, H, W = feat.shape

            mean = pca[scale]['mean']
            comp = pca[scale]['components']

            flat = feat.reshape(C, -1).T          # [HW, C]
            if args.no_mean:
                projected = flat @ comp.T         # [HW, D_pca] — NO mean subtraction
            else:
                projected = (flat - mean) @ comp.T    # [HW, D_pca]
            D_pca = projected.shape[1]
            result = projected.T.reshape(D_pca, H, W).cpu()  # [D_pca, H, W]

            fname = f"rgb_{idx}_{scale}_{D_pca}x{H}x{W}.pt"
            torch.save(result, out_dir / scale / fname)

    print(f"\nDone! Saved to {out_dir}")
    print(f"Frames processed: {n_frames - args.start_idx}")


if __name__ == '__main__':
    main()
