"""
Feature Embedding Training Script
==================================
训练3DGS特征嵌入，实现几何/外观与特征的解耦重建。

用法（原始768维融合特征）:
    python -m feature_3dgs.train_feature_embedding \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --feature_dir output/features_fixed/features_fused \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/room_0_v3_768 \
        --feature_dim 768 \
        --num_iters 30000

用法（压缩256维特征）:
    python -m feature_3dgs.train_feature_embedding \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --feature_dir dataset/room_0/Sequence_1/features_compressed/fused \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --output_dir output/feature_3dgs/room_0_v3_256 \
        --feature_dim 256 \
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
from torch.utils.data import DataLoader
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.feature_dataset import FeatureEmbeddingDataset


def l1_loss(pred, gt):
    """L1 loss (mean absolute error)"""
    return torch.abs(pred - gt).mean()


def cosine_loss(pred, gt):
    """Cosine similarity loss (1 - cos_sim)"""
    cos_sim = F.cosine_similarity(pred, gt, dim=0).mean()
    return 1.0 - cos_sim


def train(args):
    """主训练循环"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # --- 1. 加载3DGS模型 ---
    print("\n=== 加载3DGS模型 ===")
    model = GaussianFeatureModel(feature_dim=args.feature_dim)
    model.load_ply(args.ply_path)
    model = model.to(device)
    
    # --- 2. 加载数据集 ---
    print("\n=== 加载数据集 ===")
    intrinsics = {
        'fx': args.fx, 'fy': args.fy,
        'cx': args.cx, 'cy': args.cy,
    }
    dataset = FeatureEmbeddingDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        intrinsics=intrinsics,
        img_size=(args.img_height, args.img_width),
        normalize_features=True,
        max_frames=args.max_frames,
        feature_type=args.feature_type,
    )
    
    # --- 3. 设置优化器 ---
    # 仅优化特征嵌入
    optimizer = torch.optim.Adam(
        [model._loc_feature],
        lr=args.lr,
        eps=1e-15,
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_iters,
        eta_min=args.lr * 0.01,
    )
    
    # --- 4. 训练 ---
    print(f"\n=== 开始训练 ===")
    print(f"  迭代次数: {args.num_iters}")
    print(f"  学习率: {args.lr}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  数据帧数: {len(dataset)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = output_dir / 'train_log.txt'
    
    best_loss = float('inf')
    loss_history = []
    t_start = time.time()
    
    for iteration in range(1, args.num_iters + 1):
        # 随机采样一帧
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        gt_feature_map = sample['feature_map'].to(device)  # [D, fH, fW]  (35×46 原始分辨率)
        pose = sample['pose'].to(device)                    # [4, 4]
        intr = sample['intrinsics']

        # ── 全分辨率监督 (参考 STDLoc) ──
        # GT 特征先上采样到训练分辨率，再与渲染图对比
        # 好处: 梯度信号密度提升 ~200×，特征边界更锐利
        train_H = args.train_height if hasattr(args, 'train_height') and args.train_height else args.img_height
        train_W = args.train_width  if hasattr(args, 'train_width')  and args.train_width  else args.img_width

        gt_feature_map_full = F.interpolate(
            gt_feature_map.unsqueeze(0),
            size=(train_H, train_W),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)                                          # [D, train_H, train_W]
        gt_feature_map_full = F.normalize(gt_feature_map_full, p=2, dim=0)

        # 渲染特征图（直接渲染到 train 分辨率，无需二次 resize）
        result = FeatureRenderer.render_features(
            gaussian_model=model,
            viewmat=pose,
            fx=intr['fx'], fy=intr['fy'],
            cx=intr['cx'], cy=intr['cy'],
            img_height=train_H,
            img_width=train_W,
            feature_height=train_H,   # 保持一致，不做额外 resize
            feature_width=train_W,
            norm_feat_before_render=True,
            norm_feat_after_render=True,
        )

        rendered_feat = result['feature_map']  # [D, train_H, train_W]

        # 计算损失
        loss_l1 = l1_loss(rendered_feat, gt_feature_map_full)
        loss_cos = cosine_loss(rendered_feat, gt_feature_map_full)
        
        # 总损失: L1 + cosine loss
        loss = loss_l1 + args.cos_weight * loss_cos
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # 日志
        if iteration % args.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            it_per_sec = iteration / elapsed
            lr = optimizer.param_groups[0]['lr']
            
            msg = (f"[Iter {iteration:5d}/{args.num_iters}] "
                   f"loss={loss_val:.6f} (L1={loss_l1.item():.6f}, cos={loss_cos.item():.6f}) "
                   f"lr={lr:.6f} | {it_per_sec:.1f} it/s | frame={sample['frame_id']}")
            print(msg)
            
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
        
        # 保存checkpoint
        if iteration % args.save_interval == 0 or loss_val < best_loss:
            if loss_val < best_loss:
                best_loss = loss_val
                ckpt_path = output_dir / 'best_model.pth'
                torch.save({
                    'iteration': iteration,
                    'loc_feature': model._loc_feature.data,
                    'loss': loss_val,
                    'feature_dim': args.feature_dim,
                }, ckpt_path)
        
        if iteration % args.save_interval == 0:
            ckpt_path = output_dir / f'checkpoint_{iteration}.pth'
            torch.save({
                'iteration': iteration,
                'loc_feature': model._loc_feature.data,
                'loss': loss_val,
                'feature_dim': args.feature_dim,
            }, ckpt_path)
    
    # --- 5. 保存最终结果 ---
    print(f"\n=== 训练完成 ===")
    print(f"  总迭代: {args.num_iters}")
    print(f"  最终loss: {loss_history[-1]:.6f}")
    print(f"  最佳loss: {best_loss:.6f}")
    print(f"  总耗时: {time.time() - t_start:.1f}s")
    
    # 保存带特征的PLY
    ply_output = output_dir / 'point_cloud_with_features.ply'
    model.save_ply_with_features(str(ply_output))
    
    # 保存最终checkpoint
    torch.save({
        'iteration': args.num_iters,
        'loc_feature': model._loc_feature.data,
        'loss': loss_history[-1],
        'feature_dim': args.feature_dim,
        'loss_history': loss_history,
    }, output_dir / 'final_model.pth')
    
    print(f"\n结果保存至: {output_dir}")
    return model, loss_history


def parse_args():
    parser = argparse.ArgumentParser(description='Feature Embedding Training for 3DGS')
    
    # 数据路径
    parser.add_argument('--ply_path', type=str, required=True, help='预训练3DGS PLY文件路径')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='特征图目录。原始768维: output/features_fixed/features_fused；'
                             '压缩256维: dataset/.../features_compressed/fused')
    parser.add_argument('--traj_path', type=str, required=True, help='位姿文件路径')
    parser.add_argument('--output_dir', type=str, default='output/feature_3dgs/room_0', help='输出目录')
    parser.add_argument('--feature_type', type=str, default='auto',
                        choices=['auto', 'raw', 'compressed'],
                        help='特征文件类型: auto=自动识别, raw=原始768维, compressed=压缩256维')

    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='特征嵌入维度: 768=原始融合特征(推荐), 256=压缩特征')
    
    # 相机内参
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width',  type=int, default=640)
    parser.add_argument('--train_height', type=int, default=None,
                        help='训练时特征图高度 (默认=img_height, 即全分辨率监督)')
    parser.add_argument('--train_width',  type=int, default=None,
                        help='训练时特征图宽度 (默认=img_width,  即全分辨率监督)')
    
    # 训练参数
    parser.add_argument('--num_iters', type=int, default=3000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--cos_weight', type=float, default=0.1, help='cosine loss权重')
    parser.add_argument('--max_frames', type=int, default=None, help='最大训练帧数')
    
    # 日志
    parser.add_argument('--log_interval', type=int, default=50, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='保存间隔')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
