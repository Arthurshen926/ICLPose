"""
Multi-Scale Feature Embedding Training
========================================
训练 3DGS 多尺度特征嵌入 (v2-iterative-routing Phase 2)。

每个 Gaussian 存储 224d 特征向量:
  [fine_sd(64) | fine_dino(64) | mid(64) | coarse(32)]

训练策略:
  1. 在 fine 分辨率 (35×46) 渲染完整 224d 特征图
  2. 拆分为 fine/mid/coarse 三个尺度
  3. Mid/Coarse 部分 downsample 到各自的 GT 分辨率
  4. 各尺度独立计算 L1 + Cosine 损失，加权求和

用法:
    CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_multiscale_embedding \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --feature_dir output/features_multiscale_compressed/room_0 \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/room_0_multiscale_v1 \
        --num_iters 30000
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

from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.multiscale_dataset import MultiScaleFeatureDataset


def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    cos_sim = F.cosine_similarity(pred, gt, dim=0).mean()
    return 1.0 - cos_sim


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # === 1. 加载 3DGS 模型 (224d 特征嵌入) ===
    print("\n=== 加载 3DGS 模型 ===")
    model = MultiScaleGaussianModel()
    model.load_ply(args.ply_path)
    model = model.to(device)
    N = model.num_gaussians
    print(f"  Gaussians: {N:,}")
    print(f"  总特征维度: {model.TOTAL_DIM} "
          f"(fine_sd={model.FINE_SD_DIM} + fine_dino={model.FINE_DINO_DIM} "
          f"+ mid={model.MID_DIM} + coarse={model.COARSE_DIM})")

    # === 2. 加载多尺度数据集 ===
    print("\n=== 加载多尺度数据集 ===")
    intrinsics = {
        'fx': args.fx, 'fy': args.fy,
        'cx': args.cx, 'cy': args.cy,
    }
    dataset = MultiScaleFeatureDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        intrinsics=intrinsics,
        img_size=(args.img_height, args.img_width),
        normalize_features=True,
        max_frames=args.max_frames,
    )
    fine_H, fine_W = dataset.fine_hw
    mid_H, mid_W = dataset.mid_hw
    coarse_H, coarse_W = dataset.coarse_hw

    print(f"  渲染分辨率: fine={fine_W}×{fine_H}, mid={mid_W}×{mid_H}, coarse={coarse_W}×{coarse_H}")

    # === 3. 优化器 ===
    optimizer = torch.optim.Adam(
        [model._loc_feature],
        lr=args.lr,
        eps=1e-15,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_iters,
        eta_min=args.lr * 0.01,
    )

    # === 4. 缩放相机内参到 fine 分辨率 ===
    # 原始内参对应 img_size (640×480); fine 分辨率是 46×35
    # 缩放因子
    scale_x = fine_W / args.img_width
    scale_y = fine_H / args.img_height
    fine_fx = args.fx * scale_x
    fine_fy = args.fy * scale_y
    fine_cx = args.cx * scale_x
    fine_cy = args.cy * scale_y
    print(f"  Fine 内参: fx={fine_fx:.2f}, fy={fine_fy:.2f}, cx={fine_cx:.2f}, cy={fine_cy:.2f}")

    # === 5. 训练循环 ===
    print(f"\n=== 开始多尺度训练 ===")
    print(f"  迭代次数: {args.num_iters}")
    print(f"  学习率: {args.lr}")
    print(f"  损失权重: fine={args.w_fine}, mid={args.w_mid}, coarse={args.w_coarse}")
    print(f"  数据帧数: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train_log.txt'

    best_loss = float('inf')
    loss_history = []
    t_start = time.time()

    for iteration in range(1, args.num_iters + 1):
        # 随机采样一帧
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        gt_fine = sample['fine_feat'].to(device)      # [128, 35, 46]
        gt_mid = sample['mid_feat'].to(device)        # [64, 15, 20]
        gt_coarse = sample['coarse_feat'].to(device)  # [32, 7, 10]
        pose = sample['pose'].to(device)              # [4, 4]

        # ── 渲染 224d 特征图 (在 fine 分辨率下) ──
        result = FeatureRenderer.render_features(
            gaussian_model=model,
            viewmat=pose,
            fx=fine_fx, fy=fine_fy,
            cx=fine_cx, cy=fine_cy,
            img_height=fine_H,
            img_width=fine_W,
            feature_height=fine_H,
            feature_width=fine_W,
            norm_feat_before_render=True,
            norm_feat_after_render=False,  # 分尺度归一化
        )
        rendered_full = result['feature_map']  # [224, fine_H, fine_W]

        # ── 拆分为多尺度 ──
        rendered_split = MultiScaleGaussianModel.split_feature_map(rendered_full)
        # rendered_split['fine']:   [128, fine_H, fine_W]
        # rendered_split['mid']:    [64, fine_H, fine_W]  ← 需要 downsample
        # rendered_split['coarse']: [32, fine_H, fine_W]  ← 需要 downsample

        # 各尺度 L2 归一化
        rendered_fine = F.normalize(rendered_split['fine'], p=2, dim=0)
        rendered_mid_ds = F.interpolate(
            rendered_split['mid'].unsqueeze(0),
            size=(mid_H, mid_W), mode='bilinear', align_corners=False
        ).squeeze(0)
        rendered_mid = F.normalize(rendered_mid_ds, p=2, dim=0)

        rendered_coarse_ds = F.interpolate(
            rendered_split['coarse'].unsqueeze(0),
            size=(coarse_H, coarse_W), mode='bilinear', align_corners=False
        ).squeeze(0)
        rendered_coarse = F.normalize(rendered_coarse_ds, p=2, dim=0)

        # ── 各尺度损失 ──
        loss_fine_l1 = l1_loss(rendered_fine, gt_fine)
        loss_fine_cos = cosine_loss(rendered_fine, gt_fine)
        loss_fine = loss_fine_l1 + args.cos_weight * loss_fine_cos

        loss_mid_l1 = l1_loss(rendered_mid, gt_mid)
        loss_mid_cos = cosine_loss(rendered_mid, gt_mid)
        loss_mid = loss_mid_l1 + args.cos_weight * loss_mid_cos

        loss_coarse_l1 = l1_loss(rendered_coarse, gt_coarse)
        loss_coarse_cos = cosine_loss(rendered_coarse, gt_coarse)
        loss_coarse = loss_coarse_l1 + args.cos_weight * loss_coarse_cos

        # ── 加权总损失 ──
        total_loss = (args.w_fine * loss_fine +
                      args.w_mid * loss_mid +
                      args.w_coarse * loss_coarse)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        # 日志
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            it_per_sec = iteration / elapsed
            lr = optimizer.param_groups[0]['lr']

            msg = (f"[Iter {iteration:5d}/{args.num_iters}] "
                   f"loss={loss_val:.6f} "
                   f"(fine={loss_fine.item():.4f} mid={loss_mid.item():.4f} coarse={loss_coarse.item():.4f}) "
                   f"lr={lr:.6f} | {it_per_sec:.1f} it/s | frame={sample['frame_id']}")
            print(msg)

            with open(log_file, 'a') as f:
                f.write(msg + '\n')

        # 保存
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': loss_val,
                'feature_dim': model.TOTAL_DIM,
                'scale_dims': {
                    'fine_sd': model.FINE_SD_DIM,
                    'fine_dino': model.FINE_DINO_DIM,
                    'mid': model.MID_DIM,
                    'coarse': model.COARSE_DIM,
                },
            }, output_dir / 'best_model.pth')

        if iteration % args.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': loss_val,
                'feature_dim': model.TOTAL_DIM,
            }, output_dir / f'checkpoint_{iteration}.pth')

    # === 6. 保存最终结果 ===
    print(f"\n=== 训练完成 ===")
    print(f"  总迭代: {args.num_iters}")
    print(f"  最终 loss: {loss_history[-1]:.6f}")
    print(f"  最佳 loss: {best_loss:.6f}")
    print(f"  耗时: {time.time() - t_start:.1f}s")

    # 保存带特征的 PLY
    ply_output = output_dir / 'point_cloud_with_features.ply'
    model.save_ply_with_features(str(ply_output))

    # 保存最终 checkpoint
    torch.save({
        'iteration': args.num_iters,
        'loc_feature': model._loc_feature.data,
        'loss': loss_history[-1],
        'feature_dim': model.TOTAL_DIM,
        'loss_history': loss_history,
        'scale_dims': {
            'fine_sd': model.FINE_SD_DIM,
            'fine_dino': model.FINE_DINO_DIM,
            'mid': model.MID_DIM,
            'coarse': model.COARSE_DIM,
        },
    }, output_dir / 'final_model.pth')

    print(f"\n结果保存至: {output_dir}")
    return model, loss_history


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Scale Feature Embedding Training for 3DGS (v2)')

    # 数据路径
    parser.add_argument('--ply_path', type=str, required=True,
                        help='预训练 3DGS PLY 文件路径')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='多尺度压缩特征目录 (含 fine_sd/fine_dino/mid/coarse 子目录)')
    parser.add_argument('--traj_path', type=str, required=True,
                        help='位姿文件路径 (traj_w_c.txt)')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_3dgs/room_0_multiscale_v1',
                        help='输出目录')

    # 相机内参 (Replica room_0)
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)

    # 训练参数
    parser.add_argument('--num_iters', type=int, default=30000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--cos_weight', type=float, default=0.1, help='Cosine loss 权重')
    parser.add_argument('--max_frames', type=int, default=None, help='最大训练帧数')

    # 多尺度损失权重
    parser.add_argument('--w_fine', type=float, default=1.0, help='Fine 层损失权重')
    parser.add_argument('--w_mid', type=float, default=0.5, help='Mid 层损失权重')
    parser.add_argument('--w_coarse', type=float, default=0.25, help='Coarse 层损失权重')

    # 日志
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存间隔')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
