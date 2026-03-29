"""
DA3 Per-Scale Feature Embedding Training v3
============================================
v1/v2 的改进版。关键改进:
  1. 支持 warm start (从已有 checkpoint 加载嵌入)
  2. LR warmup (避免初始阶段破坏 warm start 权重)
  3. 可调 cosine loss 权重 (cos 是最终评判指标)
  4. 定期 cosine similarity 评估 (监控真实质量)
  5. 更大 grad_accum (coarse/mid 像素少→需要更多视角积累梯度)
  6. Multi-view batch rendering (同时渲染多帧减少方差)

诊断发现:
  - coarse(15×26=390px): self-sim=0.42, 特征多样性高, 极度过参数化(300K Gaussians)
  - mid(30×53=1590px): self-sim=0.67
  - fine(69×121=8349px): self-sim=0.94, 近乎均匀→易学
  核心问题: 低分辨率→梯度稀疏, 需要更多视角和更保守的优化

用法:
    PYTHONPATH=. python -m feature_3dgs.train_da3_perscale_v3 \
        --scale coarse \
        --warmstart output/feature_3dgs/oldhospital_da3_perscale_v2/coarse/best_model.pth \
        --num_iters 30000 --lr 0.003 --cos_weight 2.0 --grad_accum 16 \
        --warmup_iters 1000 --eval_interval 2000 --precache_gpu \
        --ply_path ... --feature_dir ... --traj_path ... --output_dir ...
"""

import os
import sys
import re
import argparse
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


DA3_SCALES = ['coarse', 'mid', 'fine']


# ============================================================
# Loss
# ============================================================

def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    """Per-pixel cosine loss, averaged."""
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


def combined_loss(pred, gt, cos_weight=2.0):
    """L1 + weighted cosine loss."""
    loss_l1 = l1_loss(pred, gt)
    loss_cos = cosine_loss(pred, gt)
    return loss_l1 + cos_weight * loss_cos, loss_l1.item(), loss_cos.item()


# ============================================================
# Dataset (same as v1/v2)
# ============================================================

class DA3ScaleDataset:
    def __init__(self, feature_dir, traj_path, scale, normalize=True, max_frames=None):
        self.feature_dir = Path(feature_dir) / scale
        self.scale = scale
        self.normalize = normalize

        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"特征目录不存在: {self.feature_dir}")

        traj = np.loadtxt(traj_path).reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(traj).astype(np.float32)

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

        sample = torch.load(str(self.file_map[self.frame_ids[0]]), map_location='cpu').float()
        self.feat_dim = sample.shape[0]
        self.feat_h = sample.shape[1]
        self.feat_w = sample.shape[2]

        print(f"[DA3ScaleDataset] scale={scale}, {len(self.frame_ids)} frames, "
              f"dim={self.feat_dim}, {self.feat_w}×{self.feat_h}")

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
# LR Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing."""
    def __init__(self, optimizer, warmup_iters, total_iters, eta_min_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.base_lr = optimizer.param_groups[0]['lr']
        self.eta_min = self.base_lr * eta_min_ratio
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_iters:
            # Linear warmup
            lr = self.base_lr * self._step / self.warmup_iters
        else:
            # Cosine annealing
            progress = (self._step - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ============================================================
# Evaluation
# ============================================================

def evaluate_cosine(model, dataset, renderer_args, device, n_eval_frames=10):
    """评估渲染特征与GT的平均 cosine similarity。"""
    model.eval()
    n = len(dataset)
    # 均匀采样 eval frames
    if n <= n_eval_frames:
        eval_indices = list(range(n))
    else:
        eval_indices = [int(i * n / n_eval_frames) for i in range(n_eval_frames)]

    cos_scores = []
    with torch.no_grad():
        for idx in eval_indices:
            sample = dataset[idx]
            gt_feat = sample['feat']
            if not isinstance(gt_feat, torch.Tensor):
                gt_feat = torch.tensor(gt_feat)
            gt_feat = gt_feat.to(device)
            pose = sample['pose']
            if not isinstance(pose, torch.Tensor):
                pose = torch.tensor(pose)
            pose = pose.to(device)

            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=pose,
                **renderer_args,
            )
            rendered = F.normalize(result['feature_map'], p=2, dim=0)
            gt_norm = F.normalize(gt_feat, p=2, dim=0)

            cos = F.cosine_similarity(rendered, gt_norm, dim=0).mean().item()
            cos_scores.append(cos)

    model.train()
    return np.mean(cos_scores), cos_scores


# ============================================================
# Visualization
# ============================================================

def visualize_features_pca(gt_feat, rendered_feat, save_path, iteration, scale, cos_sim):
    """PCA降维到3通道，对比GT与渲染特征。"""
    C, H, W = gt_feat.shape
    gt_flat = gt_feat.reshape(C, -1).T.float()      # [H*W, C]
    rend_flat = rendered_feat.reshape(C, -1).T.float()  # [H*W, C]

    # 合并后拟合PCA（SVD）
    all_feats = torch.cat([gt_flat, rend_flat], dim=0)  # [2*H*W, C]
    mean = all_feats.mean(0, keepdim=True)
    centered = all_feats - mean
    try:
        _, _, V = torch.linalg.svd(centered, full_matrices=False)
        pca_proj = centered @ V[:3].T  # [2*H*W, 3]
    except Exception:
        # fallback: 直接取前3个通道
        pca_proj = centered[:, :3]

    pmin = pca_proj.min(0, keepdim=True)[0]
    pmax = pca_proj.max(0, keepdim=True)[0]
    pca_proj = (pca_proj - pmin) / (pmax - pmin + 1e-8)
    pca_proj = pca_proj.clamp(0, 1)

    gt_pca = pca_proj[:H * W].reshape(H, W, 3).cpu().numpy()
    rend_pca = pca_proj[H * W:].reshape(H, W, 3).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gt_pca)
    axes[0].set_title('GT Features (PCA)', fontsize=11)
    axes[0].axis('off')
    axes[1].imshow(rend_pca)
    axes[1].set_title(f'Rendered (cos={cos_sim:.3f})', fontsize=11)
    axes[1].axis('off')
    fig.suptitle(f'{scale} @ iter {iteration}', fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Training
# ============================================================

def train_single_scale(args, scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  v3 训练: {scale}")
    print(f"{'='*60}")

    # 输出目录 (使用绝对路径避免 CWD 相关问题)
    if args.scale == 'all':
        output_dir = Path(args.output_dir).resolve() / scale
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'
    metrics_file = output_dir / 'metrics.json'

    # 1. 加载数据集
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

    if args.precache_gpu:
        cached = GPUCachedDA3Dataset(dataset, device)
    else:
        cached = None

    # 2. 加载模型
    print("\n[2] 加载模型...")
    model = GaussianFeatureModel(feature_dim=feat_dim)
    model.load_ply(args.ply_path)
    model = model.to(device)

    # 3. Warm start
    if args.warmstart and os.path.exists(args.warmstart):
        print(f"\n[3] Warm start: {args.warmstart}")
        ckpt = torch.load(args.warmstart, map_location=device)
        if 'loc_feature' in ckpt:
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            print(f"  加载嵌入成功 (loss={ckpt.get('loss', '?')})")
        else:
            print(f"  [WARN] Checkpoint 中无 loc_feature，随机初始化")
    else:
        if args.warmstart:
            print(f"  [WARN] Warmstart 文件不存在: {args.warmstart}")
        print("[3] 随机初始化")

    N = model.num_gaussians
    print(f"  Gaussians: {N}, 嵌入: [{N}, {feat_dim}]")

    # 4. 渲染参数
    scale_x = feat_w / args.img_width
    scale_y = feat_h / args.img_height
    renderer_args = {
        'fx': args.fx * scale_x,
        'fy': args.fy * scale_y,
        'cx': args.cx * scale_x,
        'cy': args.cy * scale_y,
        'img_height': feat_h,
        'img_width': feat_w,
        'feature_height': feat_h,
        'feature_width': feat_w,
        'norm_feat_before_render': True,
        'norm_feat_after_render': False,
    }
    print(f"  渲染: fx={renderer_args['fx']:.2f} fy={renderer_args['fy']:.2f} | {feat_w}×{feat_h}")

    # 5. 初始评估 (skip for speed — first eval at eval_interval)
    init_cos = -1.0
    print(f"\n  跳过初始评估 (首次评估在 iter {args.eval_interval})")

    # 6. 优化器 + 调度器
    eval_ds = cached if cached is not None else dataset
    optimizer = torch.optim.Adam([model._loc_feature], lr=args.lr, eps=1e-15)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        total_iters=args.num_iters,
        eta_min_ratio=0.01,
    )

    # 7. 训练循环
    print(f"\n[4] 开始训练")
    print(f"  迭代: {args.num_iters} | grad_accum={args.grad_accum} | cos_weight={args.cos_weight}")
    print(f"  warmup: {args.warmup_iters} iters | eval: 每{args.eval_interval}步")

    best_cos = init_cos
    best_iter = 0
    running_loss = 0.0
    running_count = 0
    metrics_history = []
    t_start = time.time()

    # 可视化目录
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

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

            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=pose,
                **renderer_args,
            )
            rendered = F.normalize(result['feature_map'], p=2, dim=0)

            loss, l1_val, cos_val = combined_loss(rendered, gt_feat, args.cos_weight)

            (loss / args.grad_accum).backward()
            iter_loss += loss.item() / args.grad_accum
            iter_l1 += l1_val / args.grad_accum
            iter_cos += cos_val / args.grad_accum

        torch.nn.utils.clip_grad_norm_([model._loc_feature], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += iter_loss
        running_count += 1

        # 日志
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            avg_loss = running_loss / running_count
            lr = scheduler.get_lr()
            it_s = iteration / elapsed if elapsed > 0 else 0

            msg = (f"[{scale}] Iter {iteration:5d}/{args.num_iters} "
                   f"loss={iter_loss:.6f} (avg={avg_loss:.4f}) "
                   f"L1={iter_l1:.4f} cos_loss={iter_cos:.4f} "
                   f"lr={lr:.6f} | {it_s:.1f} it/s")
            print(msg, flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)  # 防止目录被意外删除
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            running_loss = 0.0
            running_count = 0

        # 定期评估 (真实 cosine similarity)
        if iteration % args.eval_interval == 0 or iteration == args.num_iters:
            eval_cos, _ = evaluate_cosine(model, eval_ds, renderer_args, device)
            elapsed = time.time() - t_start

            is_best = eval_cos > best_cos
            if is_best:
                best_cos = eval_cos
                best_iter = iteration
                # 保存 best
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'loc_feature': model._loc_feature.data,
                    'loss': iter_loss,
                    'cosine_similarity': eval_cos,
                    'scale': scale,
                    'feature_dim': feat_dim,
                    'resolution': (feat_h, feat_w),
                }, output_dir / 'best_model.pth')

            marker = " ★ BEST" if is_best else ""
            eval_msg = (f"[{scale}] EVAL @ iter {iteration}: "
                        f"cos={eval_cos:.4f} (best={best_cos:.4f}@{best_iter}){marker}")
            print(eval_msg, flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(eval_msg + '\n')

            metrics_history.append({
                'iteration': iteration,
                'cosine_similarity': float(eval_cos),
                'loss': float(iter_loss),
                'elapsed': float(elapsed),
                'is_best': bool(is_best),
            })
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)

        # 定期可视化
        if args.vis_interval > 0 and (iteration % args.vis_interval == 0 or iteration == args.num_iters):
            model.eval()
            with torch.no_grad():
                eval_ds = cached if cached is not None else dataset
                vis_idx = random.randint(0, len(eval_ds) - 1)
                vis_sample = eval_ds[vis_idx]
                vis_gt = vis_sample['feat'].to(device)
                vis_pose = vis_sample['pose'].to(device)
                if vis_gt.dim() == 3:
                    vis_gt = F.normalize(vis_gt, p=2, dim=0)
                vis_result = FeatureRenderer.render_features(
                    gaussian_model=model, viewmat=vis_pose, **renderer_args)
                vis_rendered = F.normalize(vis_result['feature_map'], p=2, dim=0)
                vis_cos = F.cosine_similarity(vis_rendered, vis_gt, dim=0).mean().item()
                vis_path = vis_dir / f'iter_{iteration:06d}.png'
                visualize_features_pca(vis_gt, vis_rendered, vis_path, iteration, scale, vis_cos)
                print(f"[{scale}] 可视化保存: {vis_path.name} (cos={vis_cos:.3f})")
            model.train()

        # 定期保存
        if iteration % args.save_interval == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'scale': scale,
                'feature_dim': feat_dim,
                'resolution': (feat_h, feat_w),
            }, output_dir / f'checkpoint_{iteration}.pth')

    elapsed_total = time.time() - t_start
    print(f"\n[{scale}] v3 训练完成!")
    print(f"  最佳 cosine: {best_cos:.4f} @ iter {best_iter}")
    print(f"  初始 cosine: {init_cos:.4f} → 改进: {best_cos - init_cos:+.4f}")
    print(f"  总时间: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    return best_cos


def main():
    parser = argparse.ArgumentParser(description='DA3 Per-Scale Feature Training v3')
    parser.add_argument('--scale', type=str, default='all', choices=DA3_SCALES + ['all'])
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--traj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Warm start
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Path to best_model.pth for warm start')

    # Camera
    parser.add_argument('--img_height', type=int, default=1080)
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--fx', type=float, default=1663.12)
    parser.add_argument('--fy', type=float, default=1663.12)
    parser.add_argument('--cx', type=float, default=960.0)
    parser.add_argument('--cy', type=float, default=540.0)

    # Training
    parser.add_argument('--num_iters', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--cos_weight', type=float, default=2.0)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--precache_gpu', action='store_true')

    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--vis_interval', type=int, default=5000,
                        help='可视化间隔(iter)，0=禁用')

    args = parser.parse_args()

    if args.scale == 'all':
        results = {}
        for scale in DA3_SCALES:
            best = train_single_scale(args, scale)
            results[scale] = best
        print(f"\n{'='*60}")
        print(f"v3 所有尺度完成:")
        for s, v in results.items():
            print(f"  {s}: best_cos={v:.4f}")
    else:
        train_single_scale(args, args.scale)


if __name__ == '__main__':
    main()
