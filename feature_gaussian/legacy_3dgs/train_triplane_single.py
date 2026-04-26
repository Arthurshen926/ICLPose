#!/usr/bin/env python3
"""
Single-Scale Tri-Plane Feature Training
========================================
Train a tri-plane + MLP decoder to represent ONLY fine-scale DA3 features (64d).

Simplification over train_triplane.py:
  - Single head (fine only, 64d) instead of multi-scale (coarse+mid+fine=160d)
  - Renders only 64 channels → less VRAM, better convergence
  - Target: beat per-Gaussian quality (best_loss=0.017)

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python feature_3dgs/train_triplane_single.py \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --feature_dir output/features_da3_unified/OldHospital \
        --traj_path output/features_da3_unified/OldHospital/traj_w_c.txt \
        --output_dir output/feature_3dgs/oldhospital_triplane_single \
        --num_iters 20000 --precache_gpu
"""

import os
import sys
import argparse
import time
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_gaussian.legacy_3dgs.triplane_feature_model import TriPlaneFeatureModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer


# ============================================================
# Single-scale loss
# ============================================================

def feature_loss(pred, gt, cos_weight=2.0):
    """L1 + cosine loss between rendered and GT features.

    Args:
        pred: (D, H, W) rendered features
        gt: (D, H, W) GT features
        cos_weight: weight for cosine loss term

    Returns:
        loss, loss_dict
    """
    # L2 normalize pred for consistency
    pred_n = F.normalize(pred, p=2, dim=0)

    l1 = torch.abs(pred_n - gt).mean()
    cos = 1.0 - F.cosine_similarity(pred_n, gt, dim=0).mean()
    total = l1 + cos_weight * cos
    return total, {'l1': l1.item(), 'cos': cos.item(), 'total': total.item()}


# ============================================================
# Dataset
# ============================================================

class SingleScaleDataset:
    """Load fine-scale features for supervision."""

    def __init__(self, feature_dir: str, traj_path: str, max_frames: int = None):
        fine_dir = os.path.join(feature_dir, 'fine')
        if not os.path.isdir(fine_dir):
            raise FileNotFoundError(f"Fine dir not found: {fine_dir}")

        # Load poses (c2w → w2c)
        poses_c2w = self._load_poses(traj_path)

        # Scan feature files
        self.files = {}
        for f in sorted(os.listdir(fine_dir)):
            if f.endswith('.pt'):
                m = re.match(r'rgb_(\d+)_', f)
                if m:
                    self.files[int(m.group(1))] = os.path.join(fine_dir, f)

        # Get info from first file
        first = torch.load(next(iter(self.files.values())), map_location='cpu')
        self.feat_dim = first.shape[0]
        self.feat_h = first.shape[1]
        self.feat_w = first.shape[2]

        # Build samples
        self.samples = []
        ids = sorted(self.files.keys())
        if max_frames:
            ids = ids[:max_frames]
        for fid in ids:
            if fid < len(poses_c2w):
                c2w = poses_c2w[fid].astype(np.float32)
                w2c = np.linalg.inv(c2w)
                self.samples.append({
                    'frame_id': fid,
                    'pose': torch.from_numpy(w2c).float(),
                })

        print(f"[Dataset] {len(self.samples)} frames, feat={self.feat_dim}d @ {self.feat_h}×{self.feat_w}")

    def _load_poses(self, path):
        poses = []
        with open(path, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 16:
                    poses.append(np.array(vals).reshape(4, 4))
                elif len(vals) == 12:
                    mat = np.eye(4)
                    mat[:3, :] = np.array(vals).reshape(3, 4)
                    poses.append(mat)
        return poses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        gt = torch.load(self.files[s['frame_id']], map_location='cpu')
        return {'frame_id': s['frame_id'], 'pose': s['pose'], 'gt_feat': gt}


class GPUCachedDataset:
    """Pre-cache all features on GPU for fast training."""

    def __init__(self, dataset: SingleScaleDataset, device):
        n = len(dataset)
        print(f"[GPUCache] Pre-caching {n} frames...")
        self.frame_ids = []
        self.poses = []
        self.gt_feats = []

        for i in range(n):
            s = dataset[i]
            self.frame_ids.append(s['frame_id'])
            self.poses.append(s['pose'])
            self.gt_feats.append(s['gt_feat'])

        self.poses = torch.stack(self.poses).to(device)
        self.gt_feats = torch.stack(self.gt_feats).to(device).float()

        total_mb = (self.poses.nelement() * 4 + self.gt_feats.nelement() * 4) / 1024**2
        print(f"  Cached {n} frames → {total_mb:.0f} MB")

    def __len__(self):
        return len(self.frame_ids)

    def random_sample(self):
        i = random.randint(0, len(self) - 1)
        return {
            'frame_id': self.frame_ids[i],
            'pose': self.poses[i],
            'gt_feat': self.gt_feats[i],
        }


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Single head: only fine
    head_dims = {"fine": args.feat_dim}

    print(f"\n{'='*60}")
    print(f"Single-Scale Tri-Plane Feature Training")
    print(f"  Planes: R={args.plane_resolution}, C={args.plane_channels}")
    print(f"  Decoder: trunk_dim={args.trunk_dim}, output={args.feat_dim}d")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model
    model = TriPlaneFeatureModel(
        plane_resolution=args.plane_resolution,
        plane_channels=args.plane_channels,
        trunk_dim=args.trunk_dim,
        head_dims=head_dims,
    )
    model.load_ply(args.ply_path)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f}M")

    # 2. Dataset
    dataset = SingleScaleDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        max_frames=args.max_frames,
    )
    cached = GPUCachedDataset(dataset, device) if args.precache_gpu else None

    # 3. Rendering config
    render_h, render_w = dataset.feat_h, dataset.feat_w
    scale_x = render_w / args.img_width
    scale_y = render_h / args.img_height
    render_fx = args.fx * scale_x
    render_fy = args.fy * scale_y
    render_cx = args.cx * scale_x
    render_cy = args.cy * scale_y
    print(f"  Render: {render_w}×{render_h}, fx={render_fx:.2f} fy={render_fy:.2f}")

    # 4. Optimizer
    param_groups = model.trainable_parameters()
    optimizer = torch.optim.Adam([
        {"params": param_groups[0]["params"], "lr": args.lr_planes},
        {"params": param_groups[1]["params"], "lr": args.lr_decoder},
    ], eps=1e-15)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=args.lr_planes * 0.01
    )

    # 5. Training loop
    best_loss = float('inf')
    best_iter = 0
    loss_history = []
    t_start = time.time()

    for iteration in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        iter_loss = 0.0
        iter_l1 = 0.0
        iter_cos = 0.0
        for _ in range(args.grad_accum):
            if cached:
                s = cached.random_sample()
                gt = s['gt_feat']
                pose = s['pose']
            else:
                idx = random.randint(0, len(dataset) - 1)
                s = dataset[idx]
                gt = s['gt_feat'].to(device).float()
                pose = s['pose'].to(device)

            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=pose,
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                img_height=render_h, img_width=render_w,
                feature_height=render_h, feature_width=render_w,
                norm_feat_before_render=True,
                norm_feat_after_render=False,
                max_channels_per_chunk=64,
            )
            rendered = result['feature_map']  # (64, H, W)

            loss, ld = feature_loss(rendered, gt, cos_weight=args.cos_weight)
            (loss / args.grad_accum).backward()
            iter_loss += loss.item() / args.grad_accum
            iter_l1 += ld['l1'] / args.grad_accum
            iter_cos += ld['cos'] / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(iter_loss)

        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            it_s = iteration / elapsed if elapsed > 0 else 0
            lr = optimizer.param_groups[0]['lr']
            best_mark = " ★" if iter_loss <= best_loss else ""
            print(f"  Iter {iteration:5d}/{args.num_iters} "
                  f"loss={iter_loss:.6f} L1={iter_l1:.4f} cos={iter_cos:.4f} "
                  f"lr={lr:.6f} | {it_s:.1f} it/s{best_mark}")

        if iter_loss < best_loss:
            best_loss = iter_loss
            best_iter = iteration
            save_checkpoint(model, iteration, iter_loss, dataset, output_dir / 'best_model.pth')

        if iteration % args.save_interval == 0:
            save_checkpoint(model, iteration, iter_loss, dataset, output_dir / f'checkpoint_{iteration}.pth')

    save_checkpoint(model, args.num_iters, loss_history[-1], dataset, output_dir / 'final_model.pth')

    elapsed = time.time() - t_start
    print(f"\n✓ Training complete!")
    print(f"  Best loss: {best_loss:.6f} at iter {best_iter}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def save_checkpoint(model, iteration, loss, dataset, path):
    ckpt = {
        'iteration': iteration,
        'loss': loss,
        'plane_resolution': model.plane_resolution,
        'plane_channels': model.plane_channels,
        'trunk_dim': model.decoder.trunk[0].out_features,
        'head_dims': model.head_dims,
        'plane_xy': model.plane_xy.data.cpu(),
        'plane_xz': model.plane_xz.data.cpu(),
        'plane_yz': model.plane_yz.data.cpu(),
        'decoder_state_dict': model.decoder.state_dict(),
        'bbox_min': model.bbox_min.cpu(),
        'bbox_max': model.bbox_max.cpu(),
        'feat_dim': dataset.feat_dim,
        'feat_hw': (dataset.feat_h, dataset.feat_w),
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Single-scale tri-plane training")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Feature directory with fine/ subdirectory")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--plane_resolution", type=int, default=256)
    parser.add_argument("--plane_channels", type=int, default=32)
    parser.add_argument("--trunk_dim", type=int, default=128)
    parser.add_argument("--feat_dim", type=int, default=64)

    # Training
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--lr_planes", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--cos_weight", type=float, default=2.0)

    # Data
    parser.add_argument("--img_height", type=int, default=1080)
    parser.add_argument("--img_width", type=int, default=1920)
    parser.add_argument("--fx", type=float, default=1663.12)
    parser.add_argument("--fy", type=float, default=1663.12)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--precache_gpu", action="store_true")

    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
