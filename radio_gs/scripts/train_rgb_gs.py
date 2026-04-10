#!/usr/bin/env python3
"""Train a clean 3DGS RGB model on Replica using gsplat.

Initializes Gaussians from GT depth maps for dense coverage,
then optimizes with standard 3DGS training (no densification needed
since depth initialization is already dense).

Usage:
    CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/train_rgb_gs.py \
        --scene room_0 --sequence Sequence_1 --iters 10000

The trained model + rendered RGB images are saved for use as RGB guide
in the RADIO-GS feature distillation pipeline.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsplat import rasterization
from radio_gs.data.benchmark_paths import extract_feature_frame_index


# ─── helpers ───────────────────────────────────────────────────────

def load_replica_data(scene: str, sequence: str, dataset_root: str = "dataset",
                      subsample: int = 1):
    """Load RGB images, depth maps, and poses for a Replica sequence."""
    base = Path(dataset_root) / scene / sequence
    rgb_dir = base / "rgb"
    depth_dir = base / "depth"
    pose_file = base / "traj_w_c.txt"

    rgb_files = sorted(
        glob.glob(str(rgb_dir / "rgb_*.png")),
        key=lambda p: extract_feature_frame_index(Path(p)),
    )
    depth_files = sorted(
        glob.glob(str(depth_dir / "depth_*.png")),
        key=lambda p: extract_feature_frame_index(Path(p)),
    )
    poses_flat = np.loadtxt(str(pose_file))
    poses = poses_flat.reshape(-1, 4, 4)  # camera-to-world

    n = min(len(rgb_files), len(depth_files), len(poses))
    indices = list(range(0, n, subsample))

    images, depths, c2ws = [], [], []
    for i in indices:
        img = np.array(Image.open(rgb_files[i]).convert("RGB")).astype(np.float32) / 255.0
        dep = np.array(Image.open(depth_files[i])).astype(np.float32) / 1000.0  # mm→m
        images.append(torch.from_numpy(img))       # [H, W, 3]
        depths.append(torch.from_numpy(dep))        # [H, W]
        c2ws.append(torch.from_numpy(poses[i].astype(np.float32)))

    print(f"Loaded {len(images)} frames from {base}")
    return images, depths, c2ws


def depth_to_points(depth: Tensor, c2w: Tensor, fx: float, fy: float,
                    cx: float, cy: float, stride: int = 4) -> Tensor:
    """Unproject depth map to 3D points in world coordinates.

    Args:
        depth: [H, W] depth in meters.
        c2w: [4, 4] camera-to-world matrix.
        stride: subsampling stride to reduce point count.

    Returns:
        points [M, 3] in world coords.
    """
    H, W = depth.shape
    device = depth.device

    v, u = torch.meshgrid(
        torch.arange(0, H, stride, device=device, dtype=torch.float32),
        torch.arange(0, W, stride, device=device, dtype=torch.float32),
        indexing="ij",
    )
    d = depth[::stride, ::stride]
    mask = d > 0.01  # valid depth

    u = u[mask]
    v = v[mask]
    d = d[mask]

    x_cam = (u - cx) / fx * d
    y_cam = (v - cy) / fy * d
    z_cam = d

    pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)  # [M, 4]
    pts_world = (c2w @ pts_cam.T).T[:, :3]  # [M, 3]

    return pts_world


def init_gaussians_from_depth(images, depths, c2ws, fx, fy, cx, cy,
                              n_init_frames: int = 50, stride: int = 8,
                              max_points: int = 200_000):
    """Create initial Gaussian parameters from depth-unprojected points."""
    # Sample frames uniformly
    n = len(images)
    frame_ids = np.linspace(0, n - 1, min(n_init_frames, n), dtype=int)

    all_points = []
    all_colors = []
    for idx in frame_ids:
        pts = depth_to_points(depths[idx], c2ws[idx], fx, fy, cx, cy, stride=stride)
        # Get colors for these points
        H, W = depths[idx].shape
        d = depths[idx][::stride, ::stride]
        mask = d > 0.01
        colors = images[idx][::stride, ::stride][mask]  # [M, 3]
        all_points.append(pts)
        all_colors.append(colors)

    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)

    # Subsample if too many
    if points.shape[0] > max_points:
        perm = torch.randperm(points.shape[0])[:max_points]
        points = points[perm]
        colors = colors[perm]

    N = points.shape[0]
    print(f"Initialized {N:,} Gaussians from {len(frame_ids)} depth frames")

    # SH DC from colors: color = C0 * sh_dc + 0.5 → sh_dc = (color - 0.5) / C0
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0  # [N, 3]

    # Initial scales (small, based on point density)
    log_scales = torch.full((N, 3), math.log(0.005))  # ~5mm

    # Random quaternions (identity-ish)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    # Initial opacity (logit of 0.5 = 0)
    logit_opacity = torch.full((N, 1), 0.0)

    return {
        "means": points,            # [N, 3]
        "sh_dc": sh_dc,             # [N, 3]
        "log_scales": log_scales,   # [N, 3]
        "quats": quats,             # [N, 4]
        "logit_opacity": logit_opacity,  # [N, 1]
    }


# ─── training ─────────────────────────────────────────────────────

class SimpleGaussianModel(nn.Module):
    """Minimal 3DGS model for RGB training."""

    def __init__(self, init_data: dict, sh_degree: int = 0):
        super().__init__()
        N = init_data["means"].shape[0]
        self.sh_degree = sh_degree
        n_sh = (sh_degree + 1) ** 2

        self.means = nn.Parameter(init_data["means"].clone())
        self.quats = nn.Parameter(init_data["quats"].clone())
        self.log_scales = nn.Parameter(init_data["log_scales"].clone())
        self.logit_opacity = nn.Parameter(init_data["logit_opacity"].clone())

        # SH coefficients: [N, K, 3] where K = (sh_degree+1)^2
        sh_all = torch.zeros(N, n_sh, 3)
        sh_all[:, 0, :] = init_data["sh_dc"]
        self.sh_coeffs = nn.Parameter(sh_all)

    def render(self, viewmat: Tensor, K: Tensor, W: int, H: int,
               bg: Tensor = None) -> dict:
        """Render a single view."""
        if bg is None:
            bg = torch.zeros(3, device=self.means.device)

        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.logit_opacity).squeeze(-1)
        quats_n = F.normalize(self.quats, p=2, dim=-1)

        renders, alphas, info = rasterization(
            means=self.means,
            quats=quats_n,
            scales=scales,
            opacities=opacities,
            colors=self.sh_coeffs,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            backgrounds=bg.unsqueeze(0),
            sh_degree=self.sh_degree if self.sh_degree > 0 else None,
        )
        rgb = renders[0]  # [H, W, 3]
        alpha = alphas[0, :, :, 0]  # [H, W]
        return {"rgb": rgb, "alpha": alpha}


def l1_loss(pred, gt):
    return (pred - gt).abs().mean()


def ssim_loss(pred, gt, window_size=11):
    """Simple SSIM loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # pred, gt: [H, W, 3] → [1, 3, H, W]
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)

    pad = window_size // 2
    mu_p = F.avg_pool2d(p, window_size, stride=1, padding=pad)
    mu_g = F.avg_pool2d(g, window_size, stride=1, padding=pad)

    mu_pp = mu_p ** 2
    mu_gg = mu_g ** 2
    mu_pg = mu_p * mu_g

    sigma_pp = F.avg_pool2d(p ** 2, window_size, stride=1, padding=pad) - mu_pp
    sigma_gg = F.avg_pool2d(g ** 2, window_size, stride=1, padding=pad) - mu_gg
    sigma_pg = F.avg_pool2d(p * g, window_size, stride=1, padding=pad) - mu_pg

    ssim_map = ((2 * mu_pg + C1) * (2 * sigma_pg + C2)) / \
               ((mu_pp + mu_gg + C1) * (sigma_pp + sigma_gg + C2))

    return 1.0 - ssim_map.mean()


def train(args):
    device = torch.device("cuda")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Camera intrinsics (Replica standard)
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    W, H = 640, 480
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=device)

    # Load data
    images, depths, c2ws = load_replica_data(
        args.scene, args.sequence, args.dataset_root, subsample=1
    )
    images = [img.to(device) for img in images]
    depths = [d.to(device) for d in depths]
    c2ws = [p.to(device) for p in c2ws]
    n_frames = len(images)

    # Pre-compute w2c matrices
    w2cs = [torch.inverse(c2w) for c2w in c2ws]

    # Initialize from depth
    init_data = init_gaussians_from_depth(
        images, depths, c2ws, fx, fy, cx, cy,
        n_init_frames=args.init_frames,
        stride=args.init_stride,
        max_points=args.max_points,
    )
    for k, v in init_data.items():
        init_data[k] = v.to(device)

    model = SimpleGaussianModel(init_data, sh_degree=args.sh_degree).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with per-parameter learning rates
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": args.lr_means},
        {"params": [model.sh_coeffs], "lr": args.lr_sh},
        {"params": [model.log_scales], "lr": args.lr_scale},
        {"params": [model.quats], "lr": args.lr_quat},
        {"params": [model.logit_opacity], "lr": args.lr_opacity},
    ])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.01 ** (1.0 / args.iters)
    )

    # Output directory
    out_dir = Path(args.output_dir) / f"{args.scene}_{args.sequence}_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = torch.zeros(3, device=device)  # black background
    best_psnr = 0.0

    pbar = tqdm(range(args.iters), desc="Training RGB GS")
    for step in pbar:
        # Random frame
        idx = random.randint(0, n_frames - 1)
        gt_img = images[idx]     # [H, W, 3]
        w2c = w2cs[idx]          # [4, 4]

        result = model.render(w2c, K, W, H, bg)
        pred_rgb = result["rgb"].clamp(0, 1)  # [H, W, 3]

        # Loss: 0.8 * L1 + 0.2 * SSIM
        loss = 0.8 * l1_loss(pred_rgb, gt_img) + 0.2 * ssim_loss(pred_rgb, gt_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0 or step == 0:
            with torch.no_grad():
                mse = ((pred_rgb - gt_img) ** 2).mean()
                psnr = -10 * math.log10(mse.item() + 1e-10)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.1f}")

        if (step + 1) % 2000 == 0:
            # Evaluate on a few frames
            psnr_sum = 0.0
            eval_frames = list(range(0, n_frames, max(1, n_frames // 10)))
            with torch.no_grad():
                for ei in eval_frames:
                    res = model.render(w2cs[ei], K, W, H, bg)
                    mse = ((res["rgb"].clamp(0, 1) - images[ei]) ** 2).mean()
                    psnr_sum += -10 * math.log10(mse.item() + 1e-10)
            avg_psnr = psnr_sum / len(eval_frames)
            print(f"\n  [Iter {step+1}] Eval PSNR: {avg_psnr:.2f} dB ({len(eval_frames)} frames)")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), str(out_dir / "best.pth"))
                print(f"  → New best: {best_psnr:.2f} dB")

    # Final save
    torch.save(model.state_dict(), str(out_dir / "final.pth"))
    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Model saved to {out_dir}")

    # Render and save all frames
    if args.render_all:
        render_all_frames(model, w2cs, K, W, H, bg, out_dir, n_frames)


def render_all_frames(model, w2cs, K, W, H, bg, out_dir, n_frames):
    """Render and save RGB for all frames."""
    rgb_out = out_dir / "rendered_rgb"
    rgb_out.mkdir(exist_ok=True)
    model.eval()

    psnr_sum = 0.0
    with torch.no_grad():
        for i in tqdm(range(n_frames), desc="Rendering all frames"):
            res = model.render(w2cs[i], K, W, H, bg)
            rgb = res["rgb"].clamp(0, 1).cpu().numpy()
            Image.fromarray((rgb * 255).astype(np.uint8)).save(
                str(rgb_out / f"rgb_{i:04d}.png")
            )
    print(f"Rendered {n_frames} frames to {rgb_out}")


def main():
    parser = argparse.ArgumentParser(description="Train 3DGS RGB model on Replica")
    parser.add_argument("--scene", default="room_0")
    parser.add_argument("--sequence", default="Sequence_1")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--output_dir", default="output/radio_gs/rgb_models")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--init_frames", type=int, default=50)
    parser.add_argument("--init_stride", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_quat", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--render_all", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
