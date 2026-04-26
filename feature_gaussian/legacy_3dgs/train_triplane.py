#!/usr/bin/env python3
"""
Tri-Plane Feature Model Training
=================================
Train a GSFF-style tri-plane + MLP decoder to distill FlowFeat features
into a compact 3DGS feature representation.

The tri-plane model replaces per-Gaussian feature vectors with:
  - Three 2D feature planes (XY, XZ, YZ) at configurable resolution
  - A shared MLP trunk with multi-head output (coarse/mid/fine)

Training: render tri-plane features at known camera poses, compare with
pre-extracted FlowFeat PCA features using L1 + cosine similarity loss.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python feature_3dgs/train_triplane.py \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --feature_dir output/features_flowfeat_pca/OldHospital \
        --traj_path output/features_selected_pca/OldHospital_indexed/traj_w_c.txt \
        --output_dir output/feature_3dgs/oldhospital_triplane \
        --num_iters 10000 --precache_gpu
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_gaussian.legacy_3dgs.triplane_feature_model import TriPlaneFeatureModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer


# ============================================================
# Loss
# ============================================================

def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


def multi_scale_loss(rendered_map, gt_maps, scale_weights, cos_weight=1.0):
    """Compute per-scale L1 + cosine loss.

    Args:
        rendered_map: (total_D, H, W) concatenated rendered features
        gt_maps: Dict[str, (D_i, H_i, W_i)] per-scale GT features
        scale_weights: Dict[str, float] loss weights per scale
        cos_weight: cosine loss multiplier

    Returns:
        total_loss, loss_dict
    """
    total = 0.0
    loss_dict = {}

    # Split rendered into scales
    offset = 0
    for scale in sorted(gt_maps.keys()):
        gt = gt_maps[scale]
        D_i = gt.shape[0]
        rendered_scale = rendered_map[offset:offset + D_i]  # (D_i, H, W)
        offset += D_i

        # Resize rendered to match GT resolution if needed
        rH, rW = rendered_scale.shape[1:]
        gH, gW = gt.shape[1:]
        if rH != gH or rW != gW:
            rendered_scale = F.interpolate(
                rendered_scale.unsqueeze(0), size=(gH, gW),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        # L2 normalize
        rendered_scale = F.normalize(rendered_scale, p=2, dim=0)

        l1 = l1_loss(rendered_scale, gt)
        cos = cosine_loss(rendered_scale, gt)
        w = scale_weights.get(scale, 1.0)
        scale_loss = w * (l1 + cos_weight * cos)
        total += scale_loss
        loss_dict[f'{scale}_l1'] = l1.item()
        loss_dict[f'{scale}_cos'] = cos.item()

    loss_dict['total'] = total.item()
    return total, loss_dict


# ============================================================
# Dataset
# ============================================================

class TriPlaneFeatureDataset:
    """Load per-scale PCA-compressed FlowFeat features for supervision."""

    # scale → (subdir, dim, H, W) — auto-detected from filenames
    def __init__(self, feature_dir: str, traj_path: str,
                 scale_names=('coarse', 'mid', 'fine'),
                 max_frames: int = None):
        self.feature_dir = feature_dir
        self.scale_names = list(scale_names)

        # Load poses (c2w → w2c)
        poses_c2w = self._load_poses(traj_path)

        # Scan feature files
        self.samples = []
        self.scale_info = {}

        for scale in self.scale_names:
            scale_dir = os.path.join(feature_dir, scale)
            if not os.path.isdir(scale_dir):
                raise FileNotFoundError(f"Scale dir not found: {scale_dir}")

            files = {}
            for f in sorted(os.listdir(scale_dir)):
                if f.endswith('.pt'):
                    import re
                    m = re.match(r'rgb_(\d+)_', f)
                    if m:
                        files[int(m.group(1))] = os.path.join(scale_dir, f)

            # Get dim info from first file
            first = torch.load(next(iter(files.values())), map_location='cpu')
            self.scale_info[scale] = {
                'dim': first.shape[0],
                'height': first.shape[1],
                'width': first.shape[2],
            }

            if not hasattr(self, '_file_maps'):
                self._file_maps = {}
            self._file_maps[scale] = files

        # Build sample list: frames that have all scales
        common_ids = set.intersection(
            *[set(self._file_maps[s].keys()) for s in self.scale_names]
        )
        common_ids = sorted(common_ids)
        if max_frames and len(common_ids) > max_frames:
            common_ids = common_ids[:max_frames]

        for fid in common_ids:
            if fid < len(poses_c2w):
                c2w = poses_c2w[fid].astype(np.float32)
                w2c = np.linalg.inv(c2w)
                self.samples.append({
                    'frame_id': fid,
                    'pose': torch.from_numpy(w2c).float(),
                })

        print(f"[TriPlaneDataset] {len(self.samples)} frames, "
              f"scales={self.scale_names}, info={self.scale_info}")

    def _load_poses(self, path):
        poses = []
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
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
        fid = s['frame_id']

        gt_feats = {}
        for scale in self.scale_names:
            gt_feats[scale] = torch.load(
                self._file_maps[scale][fid], map_location='cpu'
            )

        return {
            'frame_id': fid,
            'pose': s['pose'],
            'gt_feats': gt_feats,
        }


class GPUCachedTriPlaneDataset:
    """Pre-cache all features on GPU."""

    def __init__(self, dataset: TriPlaneFeatureDataset, device):
        n = len(dataset)
        print(f"[GPUCache] Pre-caching {n} frames...")

        self.frame_ids = []
        self.poses = []
        self.gt_feats = {s: [] for s in dataset.scale_names}

        for i in range(n):
            sample = dataset[i]
            self.frame_ids.append(sample['frame_id'])
            self.poses.append(sample['pose'])
            for s in dataset.scale_names:
                self.gt_feats[s].append(sample['gt_feats'][s])

        self.poses = torch.stack(self.poses).to(device)
        for s in dataset.scale_names:
            self.gt_feats[s] = torch.stack(self.gt_feats[s]).to(device)

        total_mb = self.poses.nelement() * 4
        for s in dataset.scale_names:
            total_mb += self.gt_feats[s].nelement() * 4
        total_mb /= 1024**2
        print(f"  Cached {n} frames → {total_mb:.0f} MB")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return {
            'frame_id': self.frame_ids[idx],
            'pose': self.poses[idx],
            'gt_feats': {s: self.gt_feats[s][idx] for s in self.gt_feats},
        }

    def random_sample(self):
        return self[random.randint(0, len(self) - 1)]


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse head dims
    head_dims = {
        'coarse': args.coarse_dim,
        'mid': args.mid_dim,
        'fine': args.fine_dim,
    }
    total_dim = sum(head_dims.values())

    print(f"\n{'='*60}")
    print(f"Tri-Plane Feature Training")
    print(f"  Planes: R={args.plane_resolution}, C={args.plane_channels}")
    print(f"  Heads: {head_dims} (total={total_dim}d)")
    print(f"{'='*60}")

    # Output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create model
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

    # 2. Load dataset
    dataset = TriPlaneFeatureDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        scale_names=['coarse', 'mid', 'fine'],
        max_frames=args.max_frames,
    )

    if args.precache_gpu:
        cached = GPUCachedTriPlaneDataset(dataset, device)
    else:
        cached = None

    # 3. Rendering config
    fine_info = dataset.scale_info['fine']
    render_h, render_w = fine_info['height'], fine_info['width']
    scale_x = render_w / args.img_width
    scale_y = render_h / args.img_height
    render_fx = args.fx * scale_x
    render_fy = args.fy * scale_y
    render_cx = args.cx * scale_x
    render_cy = args.cy * scale_y
    print(f"  Render: {render_w}×{render_h}, fx={render_fx:.2f}")

    # 4. Optimizer
    param_groups = model.trainable_parameters()
    optimizer = torch.optim.Adam([
        {"params": param_groups[0]["params"], "lr": args.lr_planes},
        {"params": param_groups[1]["params"], "lr": args.lr_decoder},
    ], eps=1e-15)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=args.lr_planes * 0.01
    )

    scale_weights = {
        'coarse': args.w_coarse,
        'mid': args.w_mid,
        'fine': args.w_fine,
    }

    # 5. Training loop
    best_loss = float('inf')
    loss_history = []
    t_start = time.time()

    for iteration in range(1, args.num_iters + 1):
        optimizer.zero_grad()

        iter_loss = 0.0
        for _ in range(args.grad_accum):
            if cached:
                sample = cached.random_sample()
                gt_feats = sample['gt_feats']
                pose = sample['pose']
            else:
                idx = random.randint(0, len(dataset) - 1)
                sample = dataset[idx]
                gt_feats = {s: v.to(device) for s, v in sample['gt_feats'].items()}
                pose = sample['pose'].to(device)

            # Render: all features concatenated [total_dim, H, W] at fine resolution
            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=pose,
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                img_height=render_h, img_width=render_w,
                feature_height=render_h, feature_width=render_w,
                norm_feat_before_render=True,   # uses get_loc_feature property (already L2-norm per head)
                norm_feat_after_render=False,
                max_channels_per_chunk=64,
            )
            rendered = result['feature_map']  # (total_dim, H, W)

            loss, _ = multi_scale_loss(
                rendered, gt_feats, scale_weights,
                cos_weight=args.cos_weight,
            )
            (loss / args.grad_accum).backward()
            iter_loss += loss.item() / args.grad_accum

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(iter_loss)

        # Logging
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            it_s = iteration / elapsed if elapsed > 0 else 0
            lr = optimizer.param_groups[0]['lr']
            print(f"  Iter {iteration:5d}/{args.num_iters} "
                  f"loss={iter_loss:.6f} lr={lr:.6f} "
                  f"| {it_s:.1f} it/s")

        # Save best
        if iter_loss < best_loss:
            best_loss = iter_loss
            save_triplane_checkpoint(
                model, iteration, iter_loss, dataset.scale_info,
                output_dir / 'best_model.pth'
            )

        # Periodic save
        if iteration % args.save_interval == 0:
            save_triplane_checkpoint(
                model, iteration, iter_loss, dataset.scale_info,
                output_dir / f'checkpoint_{iteration}.pth'
            )

    # Final save
    save_triplane_checkpoint(
        model, args.num_iters, loss_history[-1], dataset.scale_info,
        output_dir / 'final_model.pth'
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def save_triplane_checkpoint(model, iteration, loss, scale_info, path):
    """Save tri-plane model checkpoint."""
    ckpt = {
        'iteration': iteration,
        'loss': loss,
        'plane_resolution': model.plane_resolution,
        'plane_channels': model.plane_channels,
        'trunk_dim': model.decoder.trunk[0].out_features,  # Linear(3*C → trunk_dim)
        'head_dims': model.head_dims,
        'plane_xy': model.plane_xy.data.cpu(),
        'plane_xz': model.plane_xz.data.cpu(),
        'plane_yz': model.plane_yz.data.cpu(),
        'decoder_state_dict': model.decoder.state_dict(),
        'bbox_min': model.bbox_min.cpu(),
        'bbox_max': model.bbox_max.cpu(),
        'scale_info': scale_info,
    }
    # Also save per-scale feature info for renderer compatibility
    # The renderer needs: feature_dim, loc_feature, resolution per scale
    for scale_name, info in scale_info.items():
        ckpt[f'{scale_name}_resolution'] = (info['height'], info['width'])
        ckpt[f'{scale_name}_dim'] = info['dim']

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train tri-plane feature model")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="FlowFeat PCA feature directory")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--plane_resolution", type=int, default=256)
    parser.add_argument("--plane_channels", type=int, default=32)
    parser.add_argument("--trunk_dim", type=int, default=128)
    parser.add_argument("--coarse_dim", type=int, default=32)
    parser.add_argument("--mid_dim", type=int, default=64)
    parser.add_argument("--fine_dim", type=int, default=64)

    # Training
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--lr_planes", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--cos_weight", type=float, default=1.0)
    parser.add_argument("--w_coarse", type=float, default=1.0)
    parser.add_argument("--w_mid", type=float, default=1.0)
    parser.add_argument("--w_fine", type=float, default=1.0)

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
    parser.add_argument("--save_interval", type=int, default=2000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
