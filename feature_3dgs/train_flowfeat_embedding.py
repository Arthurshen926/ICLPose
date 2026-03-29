"""
FlowFeat Per-Gaussian Feature Embedding Training
=================================================
Train per-Gaussian embeddings for FlowFeat PCA features (3 scales: coarse/mid/fine).

Layout: [fine(64d) | mid(64d) | coarse(32d)] = 160d per Gaussian.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -m feature_3dgs.train_flowfeat_embedding \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --feature_dir output/features_flowfeat_pca/OldHospital \
        --traj_path dataset/OldHospital/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/oldhospital_flowfeat_pg \
        --img_height 1080 --img_width 1920 \
        --fx 1663.12 --fy 1663.12 --cx 960.0 --cy 540.0 \
        --num_iters 15000 --grad_accum 4 --precache_gpu
"""
import os
import sys
import re
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


# ── Scale configuration ──
# FlowFeat: 3 scales with PCA-compressed features
# Layout in per-Gaussian embedding: [fine | mid | coarse]
SCALES = ['fine', 'mid', 'coarse']


def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


class FlowFeatDataset:
    """Load FlowFeat PCA features for per-Gaussian training."""

    def __init__(self, feature_dir, traj_path, normalize=True, max_frames=None):
        self.feature_dir = Path(feature_dir)
        self.normalize = normalize

        # Load poses (c2w → w2c)
        traj = np.loadtxt(traj_path).reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(traj).astype(np.float32)

        # Scan frame IDs from coarse directory
        coarse_dir = self.feature_dir / 'coarse'
        self.frame_ids = []
        for fpath in sorted(coarse_dir.glob('rgb_*_coarse_*.pt')):
            m = re.search(r'rgb_(\d+)_coarse_', fpath.name)
            if m:
                fid = int(m.group(1))
                if fid < len(self.poses):
                    self.frame_ids.append(fid)
        self.frame_ids.sort()
        if max_frames:
            self.frame_ids = self.frame_ids[:max_frames]

        # Auto-detect dimensions from first frame
        s0 = self._load_features(self.frame_ids[0])
        self.scale_info = {}
        for name in SCALES:
            feat = s0[name]
            self.scale_info[name] = {'dim': feat.shape[0], 'H': feat.shape[1], 'W': feat.shape[2]}

        self.fine_dim = self.scale_info['fine']['dim']
        self.mid_dim = self.scale_info['mid']['dim']
        self.coarse_dim = self.scale_info['coarse']['dim']
        self.total_dim = self.fine_dim + self.mid_dim + self.coarse_dim

        print(f"[FlowFeatDataset] {len(self.frame_ids)} frames")
        for name in SCALES:
            si = self.scale_info[name]
            print(f"  {name}: {si['dim']}d @ {si['W']}×{si['H']}")
        print(f"  Total embed dim: {self.total_dim}")

    def _load_features(self, frame_id):
        result = {}
        for name in SCALES:
            subdir = self.feature_dir / name
            matches = list(subdir.glob(f'rgb_{frame_id}_{name}_*.pt'))
            if not matches:
                raise FileNotFoundError(f"No {name} feature for frame {frame_id}")
            feat = torch.load(str(matches[0]), map_location='cpu').float()
            if self.normalize:
                feat = F.normalize(feat, p=2, dim=0)
            result[name] = feat
        return result

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        feats = self._load_features(fid)
        pose = torch.tensor(self.poses[fid], dtype=torch.float32)
        return {'feats': feats, 'pose': pose, 'frame_id': fid}


class GPUCachedFlowFeatDataset:
    """Pre-cache all features on GPU for fast training."""

    def __init__(self, dataset: FlowFeatDataset, device):
        n = len(dataset)
        print(f"[GPUCache] Pre-caching {n} frames...")
        self.frame_ids = []
        feat_lists = {name: [] for name in SCALES}
        pose_list = []

        for i in range(n):
            sample = dataset[i]
            for name in SCALES:
                feat_lists[name].append(sample['feats'][name])
            pose_list.append(sample['pose'])
            self.frame_ids.append(sample['frame_id'])

        self.feats = {name: torch.stack(feat_lists[name]).to(device) for name in SCALES}
        self.poses = torch.stack(pose_list).to(device)
        self.scale_info = dataset.scale_info

        mem_mb = sum(v.nelement() * v.element_size() for v in self.feats.values())
        mem_mb = (mem_mb + self.poses.nelement() * self.poses.element_size()) / 1024 / 1024
        print(f"  Cache size: {mem_mb:.0f} MB")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return {
            'feats': {name: self.feats[name][idx] for name in SCALES},
            'pose': self.poses[idx],
            'frame_id': self.frame_ids[idx],
        }

    def random_sample(self):
        return self[random.randint(0, len(self) - 1)]


def render_per_scale(model, pose, scale_dims, scale_intrinsics):
    """Render features per scale at native resolution.

    Args:
        model: GaussianFeatureModel with _loc_feature [N, total_dim]
        pose: [4, 4] w2c pose
        scale_dims: dict of {scale_name: (start, end)} offsets
        scale_intrinsics: dict of {scale_name: {fx, fy, cx, cy, H, W}}
    """
    raw_feat = model._loc_feature  # [N, total_dim]
    rendered = {}
    for name in SCALES:
        start, end = scale_dims[name]
        si = scale_intrinsics[name]
        colors = F.normalize(raw_feat[:, start:end], p=2, dim=-1)
        r = FeatureRenderer.render_features(
            gaussian_model=model, viewmat=pose,
            fx=si['fx'], fy=si['fy'], cx=si['cx'], cy=si['cy'],
            img_height=si['H'], img_width=si['W'],
            norm_feat_before_render=False, norm_feat_after_render=False,
            colors_override=colors,
        )
        rendered[name] = F.normalize(r['feature_map'], p=2, dim=0)
    return rendered


def train(args):
    device = torch.device('cuda')
    print(f"Device: {device}")

    # 1. Load 3DGS model
    print("\n=== Loading 3DGS model ===")
    model = GaussianFeatureModel(feature_dim=160)  # placeholder, will reinit
    model.load_ply(args.ply_path)
    model = model.to(device)
    N = model._loc_feature.shape[0]

    # 2. Load dataset
    print("\n=== Loading dataset ===")
    dataset = FlowFeatDataset(
        args.feature_dir, args.traj_path,
        normalize=True, max_frames=args.max_frames,
    )

    total_dim = dataset.total_dim
    fine_dim = dataset.fine_dim
    mid_dim = dataset.mid_dim
    coarse_dim = dataset.coarse_dim

    # Reinit feature embeddings to match dataset
    print(f"  Initializing loc_feature: ({N}, {total_dim})")
    model._loc_feature = nn.Parameter(
        torch.randn(N, total_dim, device=device) * 0.01
    )
    model.feature_dim = total_dim

    # Scale dimension offsets: [fine | mid | coarse]
    scale_dims = {
        'fine': (0, fine_dim),
        'mid': (fine_dim, fine_dim + mid_dim),
        'coarse': (fine_dim + mid_dim, total_dim),
    }
    print(f"  Layout: fine=[0:{fine_dim}], mid=[{fine_dim}:{fine_dim+mid_dim}], "
          f"coarse=[{fine_dim+mid_dim}:{total_dim}]")

    # GPU cache
    if args.precache_gpu:
        cached = GPUCachedFlowFeatDataset(dataset, device)
    else:
        cached = dataset

    # 3. Intrinsics per scale
    ref_h, ref_w = args.img_height, args.img_width
    def _scale_intr(H, W):
        return {
            'fx': args.fx * W / ref_w, 'fy': args.fy * H / ref_h,
            'cx': args.cx * W / ref_w, 'cy': args.cy * H / ref_h,
            'H': H, 'W': W,
        }
    scale_intrinsics = {}
    for name in SCALES:
        si = dataset.scale_info[name]
        scale_intrinsics[name] = _scale_intr(si['H'], si['W'])

    print(f"  Resolutions: " + ", ".join(
        f"{name}={scale_intrinsics[name]['W']}×{scale_intrinsics[name]['H']}"
        for name in SCALES))

    # 4. Optimizer
    optimizer = torch.optim.Adam([model._loc_feature], lr=args.lr, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=args.lr * 0.01)

    # 5. Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Training loop
    n_frames = len(cached)
    best_loss = float('inf')
    print(f"\n=== Training ({args.num_iters} iters, accum={args.grad_accum}) ===")
    t_start = time.time()
    running_loss = 0.0
    running_count = 0

    for iteration in range(1, args.num_iters + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            sample = cached.random_sample() if hasattr(cached, 'random_sample') \
                else cached[random.randint(0, n_frames - 1)]
            gt_feats = sample['feats']
            pose = sample['pose']
            if pose.device != device:
                pose = pose.to(device)

            rendered = render_per_scale(model, pose, scale_dims, scale_intrinsics)

            loss = torch.tensor(0.0, device=device)
            for name, w in [('fine', args.w_fine), ('mid', args.w_mid), ('coarse', args.w_coarse)]:
                gt = gt_feats[name]
                if gt.device != device:
                    gt = gt.to(device)
                rd = rendered[name]
                scale_loss = l1_loss(rd, gt) + args.cos_weight * cosine_loss(rd, gt)
                loss = loss + w * scale_loss

            (loss / args.grad_accum).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_([model._loc_feature], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = accum_loss / args.grad_accum
        running_loss += avg_loss
        running_count += 1

        if iteration % args.log_interval == 0:
            avg = running_loss / running_count
            elapsed = time.time() - t_start
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{iteration:>6}/{args.num_iters}] loss={avg:.4f} lr={lr:.6f} "
                  f"({elapsed:.0f}s)")
            running_loss = 0.0
            running_count = 0

        if iteration % args.save_interval == 0 or iteration == args.num_iters:
            ckpt = {
                'loc_feature': model._loc_feature.data,
                'iteration': iteration,
                'loss': avg_loss,
                'scale_dims': scale_dims,
                'fine_dim': fine_dim,
                'mid_dim': mid_dim,
                'coarse_dim': coarse_dim,
            }
            torch.save(ckpt, output_dir / f'checkpoint_{iteration}.pth')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'loc_feature': model._loc_feature.data,
                'iteration': iteration,
                'loss': best_loss,
                'scale_dims': scale_dims,
                'fine_dim': fine_dim,
                'mid_dim': mid_dim,
                'coarse_dim': coarse_dim,
            }
            torch.save(ckpt, output_dir / 'best_model.pth')

    # 7. Save per-scale models (compatible with MultiScaleRenderer)
    print(f"\n=== Saving per-scale models ===")
    per_scale_dir = output_dir / 'per_scale'
    per_scale_dir.mkdir(exist_ok=True)

    raw = model._loc_feature.data
    for name in SCALES:
        start, end = scale_dims[name]
        si = dataset.scale_info[name]
        torch.save({
            'loc_feature': raw[:, start:end],
            'feature_dim': end - start,
            'resolution': (si['H'], si['W']),
        }, per_scale_dir / f'{name}.pth')
        print(f"  {name}: [{start}:{end}] dim={end-start} res={si['H']}×{si['W']}")

    # Final model
    torch.save({
        'loc_feature': raw,
        'iteration': args.num_iters,
        'loss': best_loss,
        'scale_dims': scale_dims,
        'fine_dim': fine_dim,
        'mid_dim': mid_dim,
        'coarse_dim': coarse_dim,
    }, output_dir / 'final_model.pth')

    elapsed = time.time() - t_start
    print(f"\nDone! Best loss: {best_loss:.4f}, Time: {elapsed:.0f}s")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--traj_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)
    parser.add_argument('--num_iters', type=int, default=15000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--cos_weight', type=float, default=1.0)
    parser.add_argument('--w_fine', type=float, default=1.0)
    parser.add_argument('--w_mid', type=float, default=1.0)
    parser.add_argument('--w_coarse', type=float, default=1.0)
    parser.add_argument('--precache_gpu', action='store_true')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    args = parser.parse_args()
    train(args)
