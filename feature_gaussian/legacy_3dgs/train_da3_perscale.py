"""
DA3 Per-Scale Independent Feature Embedding Training
=====================================================
按尺度独立训练 DA3 特征的 3DGS 嵌入。

核心思路:
  - DA3联合训练160d的best_loss=0.45，是SD+DINO独立训练0.04的10倍
  - 联合训练中不同分辨率(15x26→69x121)的梯度互相干扰
  - 按尺度独立训练消除干扰，每个尺度的Gaussian特征参数独立优化

DA3三尺度:
  - coarse: 32d @ 15×26  (DPT stage-4)
  - mid:    64d @ 30×53  (DPT stage-3)
  - fine:   64d @ 69×121 (DPT stage-2)

用法:
    # 训练单个尺度
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m feature_3dgs.train_da3_perscale \
        --scale fine \
        --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --feature_dir output/features_da3/OldHospital_indexed \
        --traj_path dataset/OldHospital/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/oldhospital_da3_perscale/fine \
        --img_height 1080 --img_width 1920 \
        --fx 1663.12 --fy 1663.12 --cx 960.0 --cy 540.0 \
        --num_iters 15000 --grad_accum 4 --precache_gpu

    # 训练所有尺度 (串行)
    PYTHONPATH=. python -m feature_3dgs.train_da3_perscale --scale all ...
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

from feature_gaussian.legacy_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer

DA3_SCALES = ['coarse', 'mid', 'fine', 'fine_sd', 'fine_dino']


# ============================================================
# Loss
# ============================================================

def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


# ============================================================
# Dataset
# ============================================================

class DA3ScaleDataset:
    """加载单个尺度的DA3 PCA特征。自动检测维度和分辨率。"""

    def __init__(self, feature_dir, traj_path, scale, normalize=True, max_frames=None):
        self.feature_dir = Path(feature_dir) / scale
        self.scale = scale
        self.normalize = normalize

        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"特征目录不存在: {self.feature_dir}")

        # 加载 poses (c2w → w2c)
        traj = np.loadtxt(traj_path).reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(traj).astype(np.float32)

        # 扫描帧 ID
        self.frame_ids = []
        self.file_map = {}
        for fpath in sorted(self.feature_dir.glob(f'rgb_*_{scale}_*.pt')):
            m = re.search(r'rgb_(\d+)_', fpath.name)
            if m:
                fid = int(m.group(1))
                if fid < len(self.poses):
                    self.frame_ids.append(fid)
                    self.file_map[fid] = fpath
        self.frame_ids.sort()

        if max_frames:
            self.frame_ids = self.frame_ids[:max_frames]

        # 从第一帧自动检测维度和分辨率
        sample = torch.load(str(self.file_map[self.frame_ids[0]]), map_location='cpu').float()
        self.feat_dim = sample.shape[0]
        self.feat_h = sample.shape[1]
        self.feat_w = sample.shape[2]

        print(f"[DA3ScaleDataset] scale={scale}, {len(self.frame_ids)} frames")
        print(f"  dim={self.feat_dim}, resolution={self.feat_w}×{self.feat_h}")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        feat = torch.load(str(self.file_map[fid]), map_location='cpu').float()
        if self.normalize:
            feat = F.normalize(feat, p=2, dim=0)
        pose = torch.tensor(self.poses[fid], dtype=torch.float32)
        return {'feat': feat, 'pose': pose, 'frame_id': fid}


class GPUCachedDA3Dataset:
    """预缓存单尺度DA3特征到GPU。"""

    def __init__(self, dataset: DA3ScaleDataset, device):
        n = len(dataset)
        print(f"[GPUCache] 预缓存 {n} 帧 (scale={dataset.scale})...")
        self.frame_ids = []
        feat_list, pose_list = [], []
        for i in range(n):
            s = dataset[i]
            feat_list.append(s['feat'])
            pose_list.append(s['pose'])
            self.frame_ids.append(s['frame_id'])

        self.feats = torch.stack(feat_list).to(device)
        self.poses = torch.stack(pose_list).to(device)
        mem_mb = (self.feats.nelement() * self.feats.element_size()) / 1024**2
        print(f"  缓存: {self.feats.shape} → {mem_mb:.0f} MB")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return {'feat': self.feats[idx], 'pose': self.poses[idx], 'frame_id': self.frame_ids[idx]}

    def random_sample(self):
        return self[random.randint(0, len(self) - 1)]


# ============================================================
# Training
# ============================================================

def train_single_scale(args, scale):
    """训练单个尺度的DA3特征嵌入。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"训练尺度: {scale}")
    print(f"{'='*60}")

    # 输出目录
    if args.scale == 'all':
        output_dir = Path(args.output_dir) / scale
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'

    # 1. 加载数据集 (先加载以获取维度)
    print("\n[1] 加载数据集...")
    dataset = DA3ScaleDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        scale=scale,
        normalize=True,
        max_frames=args.max_frames,
    )
    feat_dim = dataset.feat_dim
    feat_h = dataset.feat_h
    feat_w = dataset.feat_w

    # GPU 预缓存
    if args.precache_gpu:
        cached = GPUCachedDA3Dataset(dataset, device)
    else:
        cached = None

    # 2. 加载 3DGS/2DGS 模型
    print("\n[2] 加载 3DGS 模型...")
    model = GaussianFeatureModel(feature_dim=feat_dim)
    model.load_ply(args.ply_path)
    model = model.to(device)

    N = model.num_gaussians
    param_mb = N * feat_dim * 4 / 1024**2
    print(f"  特征嵌入: [{N}, {feat_dim}] = {param_mb:.0f} MB")
    print(f"  估计 GPU 总占用: {param_mb * 4:.0f} MB (含 Adam 状态)")

    # 3. 相机内参缩放
    scale_x = feat_w / args.img_width
    scale_y = feat_h / args.img_height
    render_fx = args.fx * scale_x
    render_fy = args.fy * scale_y
    render_cx = args.cx * scale_x
    render_cy = args.cy * scale_y
    print(f"  渲染内参: fx={render_fx:.2f} fy={render_fy:.2f} | {feat_w}×{feat_h}")

    # 4. 优化器
    optimizer = torch.optim.Adam([model._loc_feature], lr=args.lr, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=args.lr * 0.01
    )

    # 5. 训练循环
    print(f"\n[3] 开始训练")
    print(f"  迭代: 1→{args.num_iters} | grad_accum={args.grad_accum}")

    best_loss = float('inf')
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

    for iteration in range(1, args.num_iters + 1):
        optimizer.zero_grad()
        iter_loss = 0.0
        iter_l1 = 0.0
        iter_cos = 0.0

        for _ in range(args.grad_accum):
            if cached is not None:
                sample = cached.random_sample()
                gt_feat = sample['feat']
                pose = sample['pose']
            else:
                idx = random.randint(0, len(dataset) - 1)
                sample = dataset[idx]
                gt_feat = sample['feat'].to(device)
                pose = sample['pose'].to(device)

            # 渲染
            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=pose,
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                img_height=feat_h, img_width=feat_w,
                feature_height=feat_h, feature_width=feat_w,
                norm_feat_before_render=True,
                norm_feat_after_render=False,
            )
            rendered = result['feature_map']  # [D, H, W]
            rendered = F.normalize(rendered, p=2, dim=0)

            # Loss: L1 + cosine
            loss_l1 = l1_loss(rendered, gt_feat)
            loss_cos = cosine_loss(rendered, gt_feat)
            loss = loss_l1 + args.cos_weight * loss_cos

            # Spatial diversity regularization: penalize high similarity between neighbors
            if args.diversity_weight > 0:
                cos_h = F.cosine_similarity(rendered[:, :, :-1], rendered[:, :, 1:], dim=0)
                cos_v = F.cosine_similarity(rendered[:, :-1, :], rendered[:, 1:, :], dim=0)
                diversity_loss = 0.5 * (cos_h.mean() + cos_v.mean())
                loss = loss + args.diversity_weight * diversity_loss

            (loss / args.grad_accum).backward()

            iter_loss += loss.item() / args.grad_accum
            iter_l1 += loss_l1.item() / args.grad_accum
            iter_cos += loss_cos.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_([model._loc_feature], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += iter_loss
        running_count += 1

        # 日志
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            avg_loss = running_loss / running_count
            lr = optimizer.param_groups[0]['lr']
            it_s = iteration / elapsed if elapsed > 0 else 0

            msg = (f"[{scale}] Iter {iteration:5d}/{args.num_iters} "
                   f"loss={iter_loss:.6f} (avg={avg_loss:.4f}) "
                   f"L1={iter_l1:.4f} cos={iter_cos:.4f} "
                   f"lr={lr:.6f} | {it_s:.2f} it/s")
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            running_loss = 0.0
            running_count = 0

        # 保存 best (兼容 MultiScaleRenderer 加载格式)
        if iter_loss < best_loss:
            best_loss = iter_loss
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': best_loss,
                'scale': scale,
                'feature_dim': feat_dim,
                'resolution': (feat_h, feat_w),
            }, output_dir / 'best_model.pth')

        # 定期保存
        if iteration % args.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'scale': scale,
                'feature_dim': feat_dim,
                'resolution': (feat_h, feat_w),
            }, output_dir / f'checkpoint_{iteration}.pth')

    elapsed_total = time.time() - t_start
    print(f"\n[{scale}] 训练完成!")
    print(f"  最佳 loss: {best_loss:.6f}")
    print(f"  总时间: {elapsed_total:.0f}s")
    print(f"  输出: {output_dir}")

    return best_loss


def main():
    parser = argparse.ArgumentParser(description='DA3 Per-Scale Feature Embedding Training')
    parser.add_argument('--scale', type=str, default='all',
                        choices=['coarse', 'mid', 'fine', 'fine_sd', 'fine_dino', 'all'])
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='DA3 feature directory (contains coarse/mid/fine subdirs)')
    parser.add_argument('--traj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # 图像尺寸 (用于计算渲染内参)
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)

    # 训练参数
    parser.add_argument('--num_iters', type=int, default=15000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cos_weight', type=float, default=0.5)
    parser.add_argument('--diversity_weight', type=float, default=0.0,
                        help='Weight for spatial diversity regularization (penalizes uniform renderings)')
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--precache_gpu', action='store_true')

    # 日志/保存
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)

    args = parser.parse_args()

    if args.scale == 'all':
        results = {}
        for scale in DA3_SCALES:
            best = train_single_scale(args, scale)
            results[scale] = best
        print(f"\n{'='*60}")
        print(f"所有尺度训练完成:")
        for s, v in results.items():
            print(f"  {s}: best_loss={v:.6f}")
    else:
        train_single_scale(args, args.scale)


if __name__ == '__main__':
    main()
