"""
Multi-Scale Feature Embedding Training (v2 - 加速版)
=====================================================
相比 v1 的改进:
  1. GPU 预缓存: 所有特征/位姿预加载到 GPU (~900MB)，消除磁盘 I/O
  2. 多帧梯度累积: 每步渲染 N 帧累积梯度，等效 batch_size=N
  3. 可视化验证: 定期渲染 PCA 特征图与 GT 对比
  4. 断点续训: 支持从 checkpoint 恢复
  5. 自动早停 / 合理默认迭代次数

用法:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m feature_3dgs.train_multiscale_embedding_v2 \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --feature_dir output/features_multiscale_compressed/room_0 \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/room_0_multiscale_v2 \
        --num_iters 10000 --grad_accum 4 --precache_gpu \
        --vis_interval 1000 --vis_frames 0,100,450,700
"""
import os
import sys
import argparse
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_gaussian.legacy_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer
from feature_gaussian.legacy_3dgs.multiscale_dataset import MultiScaleFeatureDataset, ColmapMultiScaleFeatureDataset


def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    return 1.0 - F.cosine_similarity(pred, gt, dim=0).mean()


def visualize_features_pca(rendered, gt, save_path, title_prefix=""):
    """渲染特征 vs GT 特征的 PCA→RGB 可视化。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        def feat_to_rgb(feat_chw):
            """[C,H,W] → [H,W,3] via PCA"""
            C, H, W = feat_chw.shape
            flat = feat_chw.reshape(C, -1).T  # [H*W, C]
            if flat.shape[0] < 3:
                return np.zeros((H, W, 3))
            pca = PCA(n_components=3)
            rgb = pca.fit_transform(flat)  # [H*W, 3]
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            return rgb.reshape(H, W, 3)

        scales = list(rendered.keys())
        n_scales = len(scales)
        fig, axes = plt.subplots(2, n_scales, figsize=(6 * n_scales, 10))

        for j, scale in enumerate(scales):
            r_np = rendered[scale].detach().cpu().numpy()
            g_np = gt[scale].detach().cpu().numpy()

            r_rgb = feat_to_rgb(r_np)
            g_rgb = feat_to_rgb(g_np)

            axes[0, j].imshow(r_rgb)
            axes[0, j].set_title(f'{title_prefix}Rendered {scale}\n{r_np.shape}')
            axes[0, j].axis('off')

            axes[1, j].imshow(g_rgb)
            axes[1, j].set_title(f'GT {scale}\n{g_np.shape}')
            axes[1, j].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  可视化失败: {e}")


class GPUCachedDataset:
    """将所有特征和位姿预加载到 GPU 显存中，消除训练时的磁盘 I/O。"""

    def __init__(self, dataset: MultiScaleFeatureDataset, device: torch.device):
        n = len(dataset)
        print(f"[GPUCachedDataset] 预缓存 {n} 帧到 GPU...")

        self.frame_ids = []
        fine_list, mid_list, coarse_list, pose_list = [], [], [], []

        for i in range(n):
            sample = dataset[i]
            fine_list.append(sample['fine_feat'])
            mid_list.append(sample['mid_feat'])
            coarse_list.append(sample['coarse_feat'])
            pose_list.append(sample['pose'])
            self.frame_ids.append(sample['frame_id'])

        self.fine_feats = torch.stack(fine_list).to(device)      # [N, 128, 35, 46]
        self.mid_feats = torch.stack(mid_list).to(device)        # [N, 64, 15, 20]
        self.coarse_feats = torch.stack(coarse_list).to(device)  # [N, 32, 7, 10]
        self.poses = torch.stack(pose_list).to(device)           # [N, 4, 4]

        mem_mb = (self.fine_feats.nelement() * self.fine_feats.element_size() +
                  self.mid_feats.nelement() * self.mid_feats.element_size() +
                  self.coarse_feats.nelement() * self.coarse_feats.element_size() +
                  self.poses.nelement() * self.poses.element_size()) / 1024 / 1024
        print(f"  缓存大小: {mem_mb:.0f} MB")
        print(f"  fine: {self.fine_feats.shape}, mid: {self.mid_feats.shape}, "
              f"coarse: {self.coarse_feats.shape}")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        return {
            'fine_feat': self.fine_feats[idx],
            'mid_feat': self.mid_feats[idx],
            'coarse_feat': self.coarse_feats[idx],
            'pose': self.poses[idx],
            'frame_id': self.frame_ids[idx],
        }

    def random_sample(self):
        idx = random.randint(0, len(self) - 1)
        return self[idx]


def render_per_scale(model, pose, scale_intrinsics):
    """按尺度独立渲染：每个尺度在其原生分辨率直接渲染。支持可配置维度。"""
    raw_feat = model._loc_feature  # [N, total_dim]

    # Fine: fine_sd + fine_dino at fine resolution
    fi = scale_intrinsics['fine']
    fine_colors = F.normalize(raw_feat[:, model._fine_sd_start:model._fine_end], p=2, dim=-1)
    r_fine = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=fi['fx'], fy=fi['fy'], cx=fi['cx'], cy=fi['cy'],
        img_height=fi['H'], img_width=fi['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=fine_colors,
    )
    rendered_fine = F.normalize(r_fine['feature_map'], p=2, dim=0)

    # Mid at mid resolution
    mi = scale_intrinsics['mid']
    mid_colors = F.normalize(raw_feat[:, model._mid_start:model._mid_end], p=2, dim=-1)
    r_mid = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=mi['fx'], fy=mi['fy'], cx=mi['cx'], cy=mi['cy'],
        img_height=mi['H'], img_width=mi['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=mid_colors,
    )
    rendered_mid = F.normalize(r_mid['feature_map'], p=2, dim=0)

    # Coarse at coarse resolution
    ci = scale_intrinsics['coarse']
    coarse_colors = F.normalize(raw_feat[:, model._coarse_start:model._coarse_end], p=2, dim=-1)
    r_coarse = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=ci['fx'], fy=ci['fy'], cx=ci['cx'], cy=ci['cy'],
        img_height=ci['H'], img_width=ci['W'],
        norm_feat_before_render=False, norm_feat_after_render=False,
        colors_override=coarse_colors,
    )
    rendered_coarse = F.normalize(r_coarse['feature_map'], p=2, dim=0)

    return {'fine': rendered_fine, 'mid': rendered_mid, 'coarse': rendered_coarse}


@torch.no_grad()
def build_knn_indices(xyz, k=8):
    """构建 KNN 索引用于空间平滑正则化。优先使用 faiss，回退到分块 cdist。"""
    try:
        import faiss
        xyz_np = xyz.detach().cpu().float().numpy().copy()
        index = faiss.IndexFlatL2(3)
        index.add(xyz_np)
        _, knn_idx = index.search(xyz_np, k + 1)
        return torch.from_numpy(knn_idx[:, 1:]).long().to(xyz.device)
    except ImportError:
        N = xyz.shape[0]
        device = xyz.device
        chunk_size = min(2048, N)
        all_indices = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            dists = torch.cdist(xyz[i:end], xyz)  # [C, N]
            self_idx = torch.arange(end - i, device=device)
            dists[self_idx, self_idx + i] = float('inf')
            _, idx = dists.topk(k, largest=False, dim=1)
            all_indices.append(idx)
        return torch.cat(all_indices, dim=0)


def knn_smooth_loss(features, knn_indices, num_samples=10000):
    """空间平滑损失：鼓励相邻 Gaussian 拥有相似特征（cosine 距离）。"""
    N = features.shape[0]
    if num_samples < N:
        idx = torch.randperm(N, device=features.device)[:num_samples]
    else:
        idx = torch.arange(N, device=features.device)
    feat = F.normalize(features[idx], p=2, dim=-1)
    neighbor_feats = F.normalize(features[knn_indices[idx]], p=2, dim=-1)  # [M, K, D]
    cos_sim = (feat.unsqueeze(1) * neighbor_feats).sum(dim=-1)  # [M, K]
    return (1 - cos_sim).mean()


def compute_loss(rendered, gt_fine, gt_mid, gt_coarse, args):
    """计算多尺度加权损失。"""
    loss_fine = l1_loss(rendered['fine'], gt_fine) + args.cos_weight * cosine_loss(rendered['fine'], gt_fine)
    loss_mid = l1_loss(rendered['mid'], gt_mid) + args.cos_weight * cosine_loss(rendered['mid'], gt_mid)
    loss_coarse = l1_loss(rendered['coarse'], gt_coarse) + args.cos_weight * cosine_loss(rendered['coarse'], gt_coarse)

    total = args.w_fine * loss_fine + args.w_mid * loss_mid + args.w_coarse * loss_coarse
    return total, loss_fine.item(), loss_mid.item(), loss_coarse.item()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # === 1. 加载 3DGS 模型 (geometry only initially) ===
    print("\n=== 加载 3DGS 模型 ===")
    model = MultiScaleGaussianModel()
    model.load_ply(args.ply_path)
    model = model.to(device)
    ply_feat_dim = model._loc_feature.shape[1]
    print(f"  PLY 特征维度: {ply_feat_dim}")

    # Dataset dimensions will be detected after loading; feature reinit happens after dataset load

    # === 2. 断点续训 ===
    start_iter = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            start_iter = ckpt.get('iteration', 0) + 1
            print(f"  从 checkpoint 恢复: {ckpt_path} (iter={start_iter-1}, loss={ckpt.get('loss', '?')})")
        else:
            print(f"  警告: checkpoint 不存在: {ckpt_path}")

    # === 3. 加载数据集 ===
    print("\n=== 加载数据集 ===")
    intrinsics = {'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy}

    if args.colmap_dir is not None:
        # COLMAP-based dataset (e.g. OldHospital with subdir feature stems)
        dataset = ColmapMultiScaleFeatureDataset(
            feature_dir=args.feature_dir,
            colmap_dir=args.colmap_dir,
            intrinsics=intrinsics if (args.fx != 320.0 or args.fy != 320.0) else None,
            normalize_features=True,
            max_frames=args.max_frames,
        )
        # Update intrinsics from dataset (may have been auto-detected)
        intrinsics = dataset.intrinsics
    elif args.traj_path is not None:
        dataset = MultiScaleFeatureDataset(
            feature_dir=args.feature_dir,
            traj_path=args.traj_path,
            intrinsics=intrinsics,
            img_size=(args.img_height, args.img_width),
            normalize_features=True,
            max_frames=args.max_frames,
        )
    else:
        raise ValueError("必须提供 --traj_path 或 --colmap_dir 之一")
    fine_H, fine_W = dataset.fine_hw
    mid_H, mid_W = dataset.mid_hw
    coarse_H, coarse_W = dataset.coarse_hw

    # === 3b. Reinitialize model features to match dataset dimensions ===
    target_dim = dataset.total_dim
    N = model._loc_feature.shape[0]
    # Compute per-scale dims from dataset
    fine_sd_dim = dataset.fine_dim // 2  # fine = fine_sd + fine_dino
    fine_dino_dim = dataset.fine_dim - fine_sd_dim
    mid_dim = dataset.mid_dim
    coarse_dim = dataset.coarse_dim
    
    if ply_feat_dim != target_dim:
        print(f"  特征维度不匹配: PLY={ply_feat_dim}, 数据集={target_dim}")
        print(f"  重新初始化 loc_feature: ({N}, {target_dim})")
        model._loc_feature = nn.Parameter(
            torch.randn(N, target_dim, device=device) * 0.01
        )
    # Update model's dimension info
    model._fine_sd_dim = fine_sd_dim
    model._fine_dino_dim = fine_dino_dim
    model._mid_dim = mid_dim
    model._coarse_dim = coarse_dim
    model._total_dim = target_dim
    model._fine_sd_start = 0
    model._fine_sd_end = fine_sd_dim
    model._fine_dino_start = fine_sd_dim
    model._fine_dino_end = fine_sd_dim + fine_dino_dim
    model._fine_end = fine_sd_dim + fine_dino_dim
    model._mid_start = model._fine_end
    model._mid_end = model._mid_start + mid_dim
    model._coarse_start = model._mid_end
    model._coarse_end = model._coarse_start + coarse_dim
    print(f"  Gaussians: {N:,}  特征维度: {target_dim} "
          f"(fine_sd={fine_sd_dim}, fine_dino={fine_dino_dim}, mid={mid_dim}, coarse={coarse_dim})")

    # GPU 预缓存
    if args.precache_gpu:
        cached = GPUCachedDataset(dataset, device)
    else:
        cached = None

    # === 4. 相机内参缩放 (各尺度独立) ===
    # 使用实际内参（COLMAP数据集可能已自动检测）
    _fx = intrinsics['fx']
    _fy = intrinsics['fy']
    _cx = intrinsics['cx']
    _cy = intrinsics['cy']
    # 参考分辨率：COLMAP数据集从cameras.bin中读取，否则使用命令行参数
    if args.colmap_dir and hasattr(dataset, 'img_w'):
        _ref_w = dataset.img_w
        _ref_h = dataset.img_h
    else:
        _ref_w = args.img_width
        _ref_h = args.img_height

    def _scale_intrinsics(H, W):
        return {
            'fx': _fx * W / _ref_w,
            'fy': _fy * H / _ref_h,
            'cx': _cx * W / _ref_w,
            'cy': _cy * H / _ref_h,
            'H': H, 'W': W,
        }
    scale_intrinsics = {
        'fine': _scale_intrinsics(fine_H, fine_W),
        'mid': _scale_intrinsics(mid_H, mid_W),
        'coarse': _scale_intrinsics(coarse_H, coarse_W),
    }
    fi = scale_intrinsics['fine']
    print(f"  Fine 内参: fx={fi['fx']:.2f}, fy={fi['fy']:.2f}")
    print(f"  分辨率: fine={fine_W}×{fine_H}, mid={mid_W}×{mid_H}, coarse={coarse_W}×{coarse_H}")

    # === 4b. KNN 空间平滑索引 ===
    knn_indices = None
    if args.knn_smooth_weight > 0:
        print(f"\n  构建 KNN 索引 (k={args.knn_k})...")
        knn_indices = build_knn_indices(model.get_xyz, k=args.knn_k)
        print(f"  KNN 索引: {knn_indices.shape}")

    # === 4c. 解冻几何参数 (可选) ===
    init_opacity = None
    init_scaling = None
    if args.unfreeze_opacity:
        init_opacity = model._opacity.detach().clone()
        delattr(model, '_opacity')
        model._opacity = nn.Parameter(init_opacity.clone())
        print(f"  ✓ 解冻 opacity (lr_scale={args.geom_lr_scale})")
    if args.unfreeze_scaling:
        init_scaling = model._scaling.detach().clone()
        delattr(model, '_scaling')
        model._scaling = nn.Parameter(init_scaling.clone())
        print(f"  ✓ 解冻 scaling (lr_scale={args.geom_lr_scale})")

    # === 5. 优化器 ===
    optim_params = [{'params': [model._loc_feature], 'lr': args.lr}]
    if args.unfreeze_opacity:
        optim_params.append({'params': [model._opacity], 'lr': args.lr * args.geom_lr_scale})
    if args.unfreeze_scaling:
        optim_params.append({'params': [model._scaling], 'lr': args.lr * args.geom_lr_scale})
    optimizer = torch.optim.Adam(optim_params, lr=args.lr, eps=1e-15)
    total_iters = args.num_iters - start_iter + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=args.lr * 0.01
    )

    # === 6. 输出目录 ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'

    # 解析可视化帧
    vis_frame_ids = []
    if args.vis_frames:
        vis_frame_ids = [int(x) for x in args.vis_frames.split(',')]

    # === 7. 训练循环 ===
    n_frames = len(cached) if cached is not None else len(dataset)
    frame_losses = None
    if args.hard_mining:
        frame_losses = torch.ones(n_frames, device=device)

    print(f"\n=== 开始训练 (v3 改进版) ===")
    print(f"  迭代: {start_iter}→{args.num_iters} ({total_iters} steps)")
    print(f"  梯度累积: {args.grad_accum} 帧/步")
    print(f"  等效 batch: {args.grad_accum} views/step")
    print(f"  GPU 预缓存: {'是' if args.precache_gpu else '否'}")
    print(f"  改进: 按尺度独立渲染 ✓")
    if knn_indices is not None:
        print(f"  改进: KNN 空间平滑 (k={args.knn_k}, w={args.knn_smooth_weight}) ✓")
    if args.unfreeze_opacity or args.unfreeze_scaling:
        print(f"  改进: 解冻几何 (opacity={args.unfreeze_opacity}, scaling={args.unfreeze_scaling}) ✓")
    if args.hard_mining:
        print(f"  改进: Hard example mining (ema={args.mining_ema}) ✓")
    print(f"  可视化间隔: {args.vis_interval}  帧: {vis_frame_ids}")

    best_loss = float('inf')
    loss_history = []
    running_loss = 0.0
    running_count = 0
    t_start = time.time()

    for iteration in range(start_iter, args.num_iters + 1):
        optimizer.zero_grad()

        iter_loss = 0.0
        iter_fine = 0.0
        iter_mid = 0.0
        iter_coarse = 0.0

        # ── 多帧梯度累积 ──
        for _ in range(args.grad_accum):
            if args.hard_mining and frame_losses is not None:
                probs = frame_losses / frame_losses.sum()
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = random.randint(0, n_frames - 1)

            if cached is not None:
                sample = cached[idx]
                gt_fine = sample['fine_feat']
                gt_mid = sample['mid_feat']
                gt_coarse = sample['coarse_feat']
                pose = sample['pose']
            else:
                sample = dataset[idx]
                gt_fine = sample['fine_feat'].to(device)
                gt_mid = sample['mid_feat'].to(device)
                gt_coarse = sample['coarse_feat'].to(device)
                pose = sample['pose'].to(device)

            rendered = render_per_scale(model, pose, scale_intrinsics)

            loss, lf, lm, lc = compute_loss(rendered, gt_fine, gt_mid, gt_coarse, args)
            (loss / args.grad_accum).backward()

            iter_loss += loss.item() / args.grad_accum
            iter_fine += lf / args.grad_accum
            iter_mid += lm / args.grad_accum
            iter_coarse += lc / args.grad_accum

            if args.hard_mining and frame_losses is not None:
                frame_losses[idx] = args.mining_ema * frame_losses[idx] + (1 - args.mining_ema) * loss.item()

        # KNN 空间平滑 + 几何正则化
        aux_loss = torch.tensor(0.0, device=device)
        if knn_indices is not None and args.knn_smooth_weight > 0:
            aux_loss = aux_loss + args.knn_smooth_weight * knn_smooth_loss(
                model._loc_feature, knn_indices, num_samples=args.knn_samples)
        if args.unfreeze_opacity and init_opacity is not None:
            aux_loss = aux_loss + args.opacity_reg * F.mse_loss(model._opacity, init_opacity)
        if args.unfreeze_scaling and init_scaling is not None:
            aux_loss = aux_loss + args.scaling_reg * F.mse_loss(model._scaling, init_scaling)
        if aux_loss.requires_grad:
            aux_loss.backward()

        optimizer.step()
        scheduler.step()

        loss_history.append(iter_loss)
        running_loss += iter_loss
        running_count += 1

        # ── 日志 ──
        if iteration % args.log_interval == 0 or iteration == start_iter:
            elapsed = time.time() - t_start
            avg_loss = running_loss / running_count
            lr = optimizer.param_groups[0]['lr']
            it_per_sec = (iteration - start_iter + 1) / elapsed

            msg = (f"[Iter {iteration:5d}/{args.num_iters}] "
                   f"loss={iter_loss:.6f} (avg={avg_loss:.4f}) "
                   f"(fine={iter_fine:.4f} mid={iter_mid:.4f} coarse={iter_coarse:.4f}) "
                   f"lr={lr:.6f} | {it_per_sec:.1f} it/s")
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

            running_loss = 0.0
            running_count = 0

        # ── 可视化 ──
        if args.vis_interval > 0 and iteration % args.vis_interval == 0 and vis_frame_ids:
            print(f"  生成可视化 (iter={iteration})...")
            with torch.no_grad():
                for fid in vis_frame_ids:
                    if fid >= len(dataset):
                        continue
                    if cached is not None:
                        # 找到 fid 对应的 index
                        try:
                            idx = cached.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = cached[idx]
                    else:
                        # 在 dataset 中找 fid 对应的 index
                        try:
                            idx = dataset.frame_ids.index(fid)
                        except ValueError:
                            continue
                        s = dataset[idx]
                        s = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in s.items()}

                    r = render_per_scale(model, s['pose'], scale_intrinsics)
                    gt = {'fine': s['fine_feat'], 'mid': s['mid_feat'], 'coarse': s['coarse_feat']}
                    save_path = vis_dir / f'iter{iteration:05d}_frame{fid}.png'
                    visualize_features_pca(r, gt, str(save_path), f"Iter{iteration} F{fid} ")

        # ── 保存 ──
        if iter_loss < best_loss:
            best_loss = iter_loss
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'feature_dim': model.TOTAL_DIM,
                'scale_dims': {
                    'fine_sd': model.FINE_SD_DIM, 'fine_dino': model.FINE_DINO_DIM,
                    'mid': model.MID_DIM, 'coarse': model.COARSE_DIM,
                },
            }, output_dir / 'best_model.pth')

        if iteration % args.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': iter_loss,
                'feature_dim': model.TOTAL_DIM,
            }, output_dir / f'checkpoint_{iteration}.pth')

    # === 8. 最终保存 ===
    elapsed_total = time.time() - t_start
    print(f"\n=== 训练完成 ===")
    print(f"  迭代: {start_iter}→{args.num_iters}")
    print(f"  最终 loss: {loss_history[-1]:.6f}")
    print(f"  最佳 loss: {best_loss:.6f}")
    print(f"  耗时: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"  等效训练视图: {(args.num_iters - start_iter + 1) * args.grad_accum}")

    ply_output = output_dir / 'point_cloud_with_features.ply'
    model.save_ply_with_features(str(ply_output))

    torch.save({
        'iteration': args.num_iters,
        'loc_feature': model._loc_feature.data,
        'loss': loss_history[-1],
        'feature_dim': model.TOTAL_DIM,
        'loss_history': loss_history,
        'scale_dims': {
            'fine_sd': model.FINE_SD_DIM, 'fine_dino': model.FINE_DINO_DIM,
            'mid': model.MID_DIM, 'coarse': model.COARSE_DIM,
        },
    }, output_dir / 'final_model.pth')

    # 最终可视化
    if vis_frame_ids:
        print("  生成最终可视化...")
        with torch.no_grad():
            for fid in vis_frame_ids:
                if fid >= len(dataset):
                    continue
                if cached is not None:
                    try:
                        idx = cached.frame_ids.index(fid)
                    except ValueError:
                        continue
                    s = cached[idx]
                else:
                    try:
                        idx = dataset.frame_ids.index(fid)
                    except ValueError:
                        continue
                    s = dataset[idx]
                    s = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in s.items()}

                r = render_per_scale(model, s['pose'], scale_intrinsics)
                gt = {'fine': s['fine_feat'], 'mid': s['mid_feat'], 'coarse': s['coarse_feat']}
                save_path = vis_dir / f'final_frame{fid}.png'
                visualize_features_pca(r, gt, str(save_path), f"Final F{fid} ")

    print(f"\n结果保存至: {output_dir}")
    return model, loss_history


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Scale Feature Embedding Training (v2 加速版)')

    # 数据路径
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--traj_path', type=str, default=None,
                        help='C2W trajectory file (N×16 values); 与 --colmap_dir 二选一')
    parser.add_argument('--colmap_dir', type=str, default=None,
                        help='COLMAP sparse dir (含 images.bin/cameras.bin)；与 --traj_path 二选一')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_3dgs/room_0_multiscale_v2')

    # 相机内参
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)

    # 训练参数
    parser.add_argument('--num_iters', type=int, default=10000,
                        help='训练迭代次数 (默认 10K, 配合 grad_accum=4 等效 40K views)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cos_weight', type=float, default=0.1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='梯度累积帧数 (等效 batch_size)')

    # 多尺度权重
    parser.add_argument('--w_fine', type=float, default=1.0)
    parser.add_argument('--w_mid', type=float, default=0.5)
    parser.add_argument('--w_coarse', type=float, default=0.25)

    # 加速
    parser.add_argument('--precache_gpu', action='store_true',
                        help='预加载所有特征到 GPU 显存')

    # 可视化
    parser.add_argument('--vis_interval', type=int, default=1000,
                        help='可视化间隔 (0=不可视化)')
    parser.add_argument('--vis_frames', type=str, default='0,100,450,700',
                        help='可视化帧 ID，逗号分隔')

    # 断点续训
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')

    # 日志
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=2000)

    # ── 改进: KNN 空间平滑 ──
    parser.add_argument('--knn_smooth_weight', type=float, default=0.01,
                        help='KNN 空间平滑损失权重 (0=禁用)')
    parser.add_argument('--knn_k', type=int, default=8,
                        help='KNN 邻居数')
    parser.add_argument('--knn_samples', type=int, default=10000,
                        help='每步采样的 Gaussian 数 (降低显存)')

    # ── 改进: 解冻几何微调 ──
    parser.add_argument('--unfreeze_opacity', action='store_true',
                        help='解冻 opacity 允许特征训练微调')
    parser.add_argument('--unfreeze_scaling', action='store_true',
                        help='解冻 scaling 允许特征训练微调')
    parser.add_argument('--opacity_reg', type=float, default=0.1,
                        help='opacity 偏移正则化权重')
    parser.add_argument('--scaling_reg', type=float, default=0.1,
                        help='scaling 偏移正则化权重')
    parser.add_argument('--geom_lr_scale', type=float, default=0.1,
                        help='几何参数学习率倍率 (相对于特征 LR)')

    # ── 改进: Hard example mining ──
    parser.add_argument('--hard_mining', action='store_true',
                        help='启用 loss 加权采样')
    parser.add_argument('--mining_ema', type=float, default=0.9,
                        help='帧 loss EMA 系数')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
