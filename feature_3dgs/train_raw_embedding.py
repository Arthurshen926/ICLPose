"""
Raw Feature Embedding Training (Per-Scale)
============================================
直接使用原始未压缩特征进行 3DGS 嵌入训练。

核心设计:
  - 按尺度独立训练 (解决 3968d 总维度超出 24GB 显存的问题)
  - 每个尺度独立优化一组 Gaussian 特征参数
  - 训练完成后各尺度特征可分别渲染，最终组合使用

内存估算 (417K Gaussians):
  fine_sd(640d):   ~4GB total → GPU 0/1 均可
  fine_dino(768d): ~5GB total → GPU 0/1 均可
  mid(1280d):      ~8GB total → GPU 0/1 均可
  coarse(1280d):   ~8GB total → GPU 0/1 均可

用法:
    # 训练 fine_sd (640d)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m feature_3dgs.train_raw_embedding \
        --scale fine_sd \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --feature_dir output/features_multiscale/room_0 \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/room_0_raw/fine_sd \
        --num_iters 5000 --grad_accum 4 --precache_gpu

    # 训练 mid (1280d) 在另一张 GPU
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m feature_3dgs.train_raw_embedding \
        --scale mid ...

    # 一键训练所有尺度 (串行)
    PYTHONPATH=. python -m feature_3dgs.train_raw_embedding --scale all --output_dir output/feature_3dgs/room_0_raw ...
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.raw_gaussian_model import RawScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.raw_multiscale_dataset import RawScaleFeatureDataset, RAW_SCALE_CONFIGS


# ============================================================
# Loss
# ============================================================

def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


# ============================================================
# GPU Cached Dataset
# ============================================================

class GPUCachedScaleDataset:
    """将单尺度所有特征预加载到 GPU。"""

    def __init__(self, dataset: RawScaleFeatureDataset, device: torch.device):
        n = len(dataset)
        print(f"[GPUCachedScaleDataset] 预缓存 {n} 帧 (scale={dataset.scale})...")

        self.frame_ids = []
        feat_list, pose_list = [], []

        for i in range(n):
            sample = dataset[i]
            feat_list.append(sample['feat'])
            pose_list.append(sample['pose'])
            self.frame_ids.append(sample['frame_id'])

        self.feats = torch.stack(feat_list).to(device)   # [N, D, H, W]
        self.poses = torch.stack(pose_list).to(device)    # [N, 4, 4]

        mem_mb = (self.feats.nelement() * self.feats.element_size() +
                  self.poses.nelement() * self.poses.element_size()) / 1024**2
        print(f"  缓存: {self.feats.shape} → {mem_mb:.0f} MB")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return {
            'feat': self.feats[idx],
            'pose': self.poses[idx],
            'frame_id': self.frame_ids[idx],
        }

    def random_sample(self):
        idx = random.randint(0, len(self) - 1)
        return self[idx]


# ============================================================
# PCA Visualization
# ============================================================

def visualize_pca(rendered, gt, save_path, title=""):
    """渲染特征 vs GT 的 PCA→RGB 可视化。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        def feat_to_rgb(feat_chw):
            C, H, W = feat_chw.shape
            flat = feat_chw.reshape(C, -1).T
            if flat.shape[0] < 3:
                return np.zeros((H, W, 3))
            pca = PCA(n_components=3)
            rgb = pca.fit_transform(flat)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            return rgb.reshape(H, W, 3)

        r_np = rendered.detach().cpu().numpy()
        g_np = gt.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(feat_to_rgb(r_np))
        axes[0].set_title(f'{title}Rendered [{r_np.shape[0]}d]')
        axes[0].axis('off')
        axes[1].imshow(feat_to_rgb(g_np))
        axes[1].set_title(f'GT [{g_np.shape[0]}d]')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  可视化失败: {e}")


# ============================================================
# Single-Scale Training
# ============================================================

def train_single_scale(args, scale: str):
    """训练单个尺度的特征嵌入。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scale_cfg = RAW_SCALE_CONFIGS[scale]
    feat_dim = scale_cfg['dim']
    feat_h, feat_w = scale_cfg['resolution']

    print(f"\n{'='*60}")
    print(f"训练尺度: {scale} ({feat_dim}d @ {feat_w}×{feat_h})")
    print(f"{'='*60}")

    # --- 输出目录 ---
    if args.scale == 'all':
        output_dir = Path(args.output_dir) / scale
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'

    # --- 1. 加载 3DGS 模型 ---
    print("\n[1] 加载 3DGS 模型...")
    model = RawScaleGaussianModel(scale=scale)
    model.load_ply(args.ply_path)
    model = model.to(device)
    print(f"  {model.summary()}")

    # 内存估算
    param_mb = model._loc_feature.nelement() * 4 / 1024**2
    total_est_mb = param_mb * 4  # param + grad + adam(m,v)
    print(f"  估计 GPU 占用: {total_est_mb:.0f} MB (含 Adam 状态)")

    # --- 2. 断点续训 ---
    start_iter = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            start_iter = ckpt.get('iteration', 0) + 1
            print(f"  从 checkpoint 恢复: iter={start_iter-1}")

    # --- 3. 加载数据集 ---
    print("\n[2] 加载数据集...")
    intrinsics = {'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy}
    dataset = RawScaleFeatureDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        scale=scale,
        intrinsics=intrinsics,
        normalize_features=True,
        max_frames=args.max_frames,
    )

    # GPU 预缓存
    if args.precache_gpu:
        cached = GPUCachedScaleDataset(dataset, device)
    else:
        cached = None

    # --- 4. 相机内参缩放 ---
    scale_x = feat_w / args.img_width
    scale_y = feat_h / args.img_height
    render_fx = args.fx * scale_x
    render_fy = args.fy * scale_y
    render_cx = args.cx * scale_x
    render_cy = args.cy * scale_y
    print(f"  渲染内参: fx={render_fx:.2f} fy={render_fy:.2f} | {feat_w}×{feat_h}")

    # --- 5. 优化器 ---
    optimizer = torch.optim.Adam([model._loc_feature], lr=args.lr, eps=1e-15)
    total_iters = args.num_iters - start_iter + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=args.lr * 0.01
    )

    # 解析可视化帧
    vis_frame_ids = []
    if args.vis_frames:
        vis_frame_ids = [int(x) for x in args.vis_frames.split(',')]

    # --- 6. 训练循环 ---
    print(f"\n[3] 开始训练")
    print(f"  迭代: {start_iter}→{args.num_iters} | grad_accum={args.grad_accum}")
    print(f"  GPU 预缓存: {'是' if cached else '否'}")

    best_loss = float('inf')
    loss_history = []
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

    for iteration in range(start_iter, args.num_iters + 1):
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

            # L2 归一化渲染结果
            rendered = F.normalize(rendered, p=2, dim=0)

            # Loss
            loss_l1 = l1_loss(rendered, gt_feat)
            loss_cos = cosine_loss(rendered, gt_feat)
            loss = loss_l1 + args.cos_weight * loss_cos
            (loss / args.grad_accum).backward()

            iter_loss += loss.item() / args.grad_accum
            iter_l1 += loss_l1.item() / args.grad_accum
            iter_cos += loss_cos.item() / args.grad_accum

        optimizer.step()
        scheduler.step()

        loss_history.append(iter_loss)
        running_loss += iter_loss
        running_count += 1

        # --- 日志 ---
        if iteration % args.log_interval == 0 or iteration == start_iter:
            elapsed = time.time() - t_start
            avg_loss = running_loss / running_count
            lr = optimizer.param_groups[0]['lr']
            it_s = (iteration - start_iter + 1) / elapsed if elapsed > 0 else 0

            msg = (f"[{scale}] Iter {iteration:5d}/{args.num_iters} "
                   f"loss={iter_loss:.6f} (avg={avg_loss:.4f}) "
                   f"L1={iter_l1:.4f} cos={iter_cos:.4f} "
                   f"lr={lr:.6f} | {it_s:.2f} it/s")
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

            running_loss = 0.0
            running_count = 0

        # --- 可视化 ---
        if args.vis_interval > 0 and iteration % args.vis_interval == 0 and vis_frame_ids:
            with torch.no_grad():
                for fid in vis_frame_ids:
                    if cached is not None:
                        try:
                            fidx = cached.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = cached[fidx]
                    else:
                        try:
                            fidx = dataset.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = dataset[fidx]
                        s = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in s.items()}

                    r = FeatureRenderer.render_features(
                        gaussian_model=model, viewmat=s['pose'],
                        fx=render_fx, fy=render_fy,
                        cx=render_cx, cy=render_cy,
                        img_height=feat_h, img_width=feat_w,
                        feature_height=feat_h, feature_width=feat_w,
                        norm_feat_before_render=True, norm_feat_after_render=True,
                    )
                    save_path = vis_dir / f'iter{iteration:05d}_f{fid}.png'
                    visualize_pca(r['feature_map'], s['feat'], str(save_path),
                                  f"[{scale}] Iter{iteration} F{fid} ")

        # --- 保存 ---
        if iter_loss < best_loss:
            best_loss = iter_loss
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'scale': scale,
                'feature_dim': feat_dim,
                'resolution': (feat_h, feat_w),
            }, output_dir / 'best_model.pth')

        if iteration % args.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'scale': scale,
                'feature_dim': feat_dim,
            }, output_dir / f'checkpoint_{iteration}.pth')

    # --- 最终保存 ---
    elapsed_total = time.time() - t_start
    print(f"\n[{scale}] 训练完成!")
    print(f"  最终 loss: {loss_history[-1]:.6f}")
    print(f"  最佳 loss: {best_loss:.6f}")
    print(f"  耗时: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")

    # 保存 PLY
    ply_out = output_dir / f'point_cloud_{scale}.ply'
    model.save_ply_with_features(str(ply_out))

    # 保存最终 checkpoint
    torch.save({
        'iteration': args.num_iters,
        'loc_feature': model._loc_feature.data,
        'loss': loss_history[-1],
        'scale': scale,
        'feature_dim': feat_dim,
        'resolution': (feat_h, feat_w),
        'loss_history': loss_history,
    }, output_dir / 'final_model.pth')

    # 最终可视化
    if vis_frame_ids:
        with torch.no_grad():
            for fid in vis_frame_ids:
                if cached is not None:
                    try:
                        fidx = cached.frame_ids.index(fid)
                    except ValueError:
                        continue
                    s = cached[fidx]
                else:
                    try:
                        fidx = dataset.frame_ids.index(fid)
                    except ValueError:
                        continue
                    s = dataset[fidx]
                    s = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in s.items()}

                r = FeatureRenderer.render_features(
                    gaussian_model=model, viewmat=s['pose'],
                    fx=render_fx, fy=render_fy,
                    cx=render_cx, cy=render_cy,
                    img_height=feat_h, img_width=feat_w,
                    feature_height=feat_h, feature_width=feat_w,
                    norm_feat_before_render=True, norm_feat_after_render=True,
                )
                save_path = vis_dir / f'final_f{fid}.png'
                visualize_pca(r['feature_map'], s['feat'], str(save_path),
                              f"[{scale}] Final F{fid} ")

    print(f"  结果: {output_dir}")
    return model, loss_history


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Raw Feature Embedding Training (Per-Scale)')

    # 尺度选择
    parser.add_argument('--scale', type=str, default='fine_sd',
                        choices=['fine_sd', 'fine_dino', 'mid', 'coarse', 'all'],
                        help='训练哪个尺度 (all=串行训练所有尺度)')

    # 数据路径
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='原始特征目录 (e.g. output/features_multiscale/room_0)')
    parser.add_argument('--traj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_3dgs/room_0_raw')

    # 相机
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)

    # 训练
    parser.add_argument('--num_iters', type=int, default=5000,
                        help='每个尺度的训练迭代数 (原始特征维度高，loss 收敛更快)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cos_weight', type=float, default=0.1)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=None)

    # 加速
    parser.add_argument('--precache_gpu', action='store_true')

    # 可视化
    parser.add_argument('--vis_interval', type=int, default=1000)
    parser.add_argument('--vis_frames', type=str, default='0,100,450')

    # 断点续训
    parser.add_argument('--resume', type=str, default=None)

    # 日志
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=2000)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.scale == 'all':
        # 串行训练所有尺度
        print("=" * 60)
        print("训练全部尺度 (串行)")
        print("=" * 60)
        all_scales = ['fine_sd', 'fine_dino', 'mid', 'coarse']
        for i, scale in enumerate(all_scales):
            print(f"\n[{i+1}/{len(all_scales)}] 训练 {scale}...")
            train_single_scale(args, scale)
            # 释放 GPU 内存
            torch.cuda.empty_cache()
    else:
        train_single_scale(args, args.scale)


if __name__ == '__main__':
    main()
