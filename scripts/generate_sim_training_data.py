#!/usr/bin/env python3
"""
仿真训练数据生成脚本
=====================

利用 3DGS 作为仿真器, 在场景中采样有效位姿 → 渲染 RGB → 提取 DINO 特征 → 缓存为 .pt

工作流程:
  1. 加载 3DGS 模型 + 原始轨迹
  2. 定义场景有效区域 (基于轨迹包围盒 + margin)
  3. 采样 N 个有效位姿 (碰撞检测 + 视角合理性)
  4. 渲染 RGB → 提取 DINO 特征 → 保存

输出结构 (output_dir/):
  sim_poses.npy                              # (N, 4, 4) c2w 位姿
  fine_dino/sim_{idx}_fine_dino_768x35x46.pt # DINO 特征
  depth/sim_{idx}_depth.pt                   # 渲染深度图 (用于训练)
  rgb_preview/sim_{idx}.png                  # RGB 预览 (可选)

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_sim_training_data.py \\
        --num_poses 5000 \\
        --output_dir output/sim_training_data/room_0 \\
        --margin 0.3
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer
from data.dataset_v3 import load_poses_c2w


# ============================================================================
# Pose Sampler — 在场景有效区域内采样合理位姿
# ============================================================================

class ScenePoseSampler:
    """
    基于原始轨迹的有界位姿采样器
    
    策略:
      - 位置: 在轨迹包围盒 + margin 内均匀采样, 混合轨迹附近高斯采样
      - 朝向: 基于原始轨迹朝向分布采样 + 小幅随机扰动
      - 碰撞检测: 渲染 alpha 图, 拒绝可视性过低的位姿
      - 深度检测: 拒绝离几何面太近的位姿 (相机在物体内部)
    """
    
    def __init__(
        self,
        traj_c2w: np.ndarray,          # (N, 4, 4) 原始轨迹 c2w
        gaussian_xyz: np.ndarray,      # (M, 3)  Gaussian 中心坐标
        margin: float = 0.3,           # 包围盒外扩 margin (米)
        min_depth: float = 0.1,        # 最小深度阈值 (米)
        min_alpha_ratio: float = 0.5,  # alpha 覆盖率最低阈值
        nearby_ratio: float = 0.7,     # 轨迹附近样本占比(0~1)
        nearby_pos_std: float = 0.3,   # 轨迹附近位置扰动 std (米)
        nearby_rot_std: float = 15.0,  # 轨迹附近朝向扰动 std (度)
    ):
        self.traj_c2w = traj_c2w
        self.N_traj = len(traj_c2w)
        self.margin = margin
        self.min_depth = min_depth
        self.min_alpha_ratio = min_alpha_ratio
        self.nearby_ratio = nearby_ratio
        self.nearby_pos_std = nearby_pos_std
        self.nearby_rot_std = nearby_rot_std
        
        # 提取轨迹位置与朝向
        self.traj_positions = traj_c2w[:, :3, 3]  # (N, 3)
        self.traj_rotations = traj_c2w[:, :3, :3]  # (N, 3, 3)
        
        # 计算场景包围盒 (基于轨迹 + Gaussian 热点区域)
        traj_min = self.traj_positions.min(axis=0)
        traj_max = self.traj_positions.max(axis=0)
        
        # 使用 Gaussian 中心的 P5-P95 范围与轨迹范围的交集+margin
        gauss_p5 = np.percentile(gaussian_xyz, 5, axis=0)
        gauss_p95 = np.percentile(gaussian_xyz, 95, axis=0)
        
        # 场景有效区域 = 轨迹范围 ± margin (不超出 Gaussian 分布)
        self.bbox_min = np.maximum(traj_min - margin, gauss_p5)
        self.bbox_max = np.minimum(traj_max + margin, gauss_p95)
        
        print(f"[PoseSampler] 轨迹范围:")
        print(f"  X: [{traj_min[0]:.2f}, {traj_max[0]:.2f}]")
        print(f"  Y: [{traj_min[1]:.2f}, {traj_max[1]:.2f}]")
        print(f"  Z: [{traj_min[2]:.2f}, {traj_max[2]:.2f}]")
        print(f"[PoseSampler] 采样范围 (margin={margin}m):")
        print(f"  X: [{self.bbox_min[0]:.2f}, {self.bbox_max[0]:.2f}]")
        print(f"  Y: [{self.bbox_min[1]:.2f}, {self.bbox_max[1]:.2f}]")
        print(f"  Z: [{self.bbox_min[2]:.2f}, {self.bbox_max[2]:.2f}]")
    
    def _sample_nearby_pose(self) -> np.ndarray:
        """在原始轨迹附近采样: 选一个轨迹帧, 加小扰动"""
        idx = np.random.randint(0, self.N_traj)
        c2w = self.traj_c2w[idx].copy()
        
        # 位置扰动
        c2w[:3, 3] += np.random.randn(3) * self.nearby_pos_std
        
        # 朝向扰动 (绕随机轴小角度旋转)
        angle_deg = np.random.randn() * self.nearby_rot_std
        angle_rad = angle_deg * np.pi / 180.0
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        # Rodrigues 旋转
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        delta_R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        c2w[:3, :3] = delta_R @ c2w[:3, :3]
        
        # 确保位置在包围盒内
        c2w[:3, 3] = np.clip(c2w[:3, 3], self.bbox_min, self.bbox_max)
        
        return c2w
    
    def _sample_uniform_pose(self) -> np.ndarray:
        """在包围盒内均匀采样位置, 朝向从轨迹分布中随机选取+扰动"""
        c2w = np.eye(4)
        
        # 均匀位置
        c2w[:3, 3] = np.random.uniform(self.bbox_min, self.bbox_max)
        
        # 朝向: 随机选一个轨迹帧的朝向 + 较大扰动
        idx = np.random.randint(0, self.N_traj)
        base_R = self.traj_rotations[idx].copy()
        
        angle_deg = np.random.randn() * 30.0  # 更大的朝向扰动
        angle_rad = angle_deg * np.pi / 180.0
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        delta_R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        c2w[:3, :3] = delta_R @ base_R
        
        return c2w
    
    def sample_candidate(self) -> np.ndarray:
        """采样一个候选位姿 (c2w), 按 nearby_ratio 概率分配"""
        if np.random.rand() < self.nearby_ratio:
            return self._sample_nearby_pose()
        else:
            return self._sample_uniform_pose()
    
    @staticmethod
    def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        w2c = np.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ t
        return w2c
    
    def validate_pose(
        self,
        c2w: np.ndarray,
        gaussian_model,
        fx: float, fy: float, cx: float, cy: float,
        img_height: int, img_width: int,
    ) -> bool:
        """
        验证位姿合理性:
          1. alpha 覆盖率 >= min_alpha_ratio
          2. 最近深度 >= min_depth (相机不在物体内部)
        """
        w2c = self.c2w_to_w2c(c2w)
        viewmat = torch.from_numpy(w2c).float().to(gaussian_model.get_xyz.device)
        
        with torch.no_grad():
            # 渲染 alpha (用 render_rgb 的 alpha 通道)
            result = FeatureRenderer.render_rgb(
                gaussian_model, viewmat, fx, fy, cx, cy, img_height, img_width
            )
            alpha = result['alpha']  # [1, H, W]
            alpha_ratio = (alpha > 0.5).float().mean().item()
            
            if alpha_ratio < self.min_alpha_ratio:
                return False
            
            # 渲染深度检查
            depth = FeatureRenderer.render_depth(
                gaussian_model, viewmat, fx, fy, cx, cy, img_height, img_width
            )
            # 有效深度 (alpha > 0.5 区域)
            valid_mask = alpha.squeeze(0) > 0.5
            if valid_mask.sum() < 100:
                return False
            valid_depth = depth[valid_mask]
            
            # 最小深度过滤 (相机不能在面内)
            min_d = valid_depth.min().item()
            if min_d < self.min_depth:
                return False
        
        return True


# ============================================================================
# DINO Feature Extractor (仅 fine_dino)
# ============================================================================

class DINOFeatureExtractor:
    """
    从 RGB 图像 (tensor 或 numpy) 提取 DINO fine 特征
    
    与 MultiScaleFeatureExtractor 类似, 但只提取 fine_dino:
      - 不加载 SD 模型 (省显存)
      - 输入可以是 tensor (来自 3DGS RGB 渲染)
    """
    
    def __init__(self, device='cuda', target_hw=(35, 46)):
        self.device = device
        self.target_hw = target_hw
        
        from feature_extraction.extractor_dino import ViTExtractor
        self.extractor = ViTExtractor('dinov2_vitb14', stride=14, device=device)
        
        # DINO 标准归一化
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        print(f"[DINOFeatureExtractor] 加载完成, target_hw={target_hw}")
    
    @torch.no_grad()
    def extract_from_tensor(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        从 RGB tensor 提取 DINO fine_dino 特征
        
        Args:
            rgb_tensor: [3, H, W] float, range [0, 1] (来自 3DGS render_rgb)
        
        Returns:
            [768, tH, tW] fine_dino 特征
        """
        # DINO 需要输入尺寸为 14 的倍数
        _, H, W = rgb_tensor.shape
        dino_h = int(math.ceil(H / 14) * 14)
        dino_w = int(math.ceil(W / 14) * 14)
        
        # Resize to DINO-compatible resolution
        img = F.interpolate(
            rgb_tensor.unsqueeze(0), size=(dino_h, dino_w),
            mode='bilinear', align_corners=False
        )
        
        # 归一化 (ImageNet mean/std)
        img = (img - self.mean) / self.std
        
        # 提取 patch tokens
        tokens_h, tokens_w = dino_h // 14, dino_w // 14
        feats = self.extractor.extract_descriptors(
            img, layer=11, facet='token', include_cls=False
        )
        # [B, 1, num_patches, 768] → [768, tokens_h, tokens_w]
        feats = feats.squeeze(0).squeeze(0)  # [num_patches, 768]
        feats = feats.T.reshape(768, tokens_h, tokens_w)  # [768, tH, tW]
        
        # Resize to target resolution (与预提取特征一致)
        if (tokens_h, tokens_w) != self.target_hw:
            feats = F.interpolate(
                feats.unsqueeze(0), size=self.target_hw,
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return feats.cpu()


# ============================================================================
# Main Generation Pipeline
# ============================================================================

def generate_sim_data(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    (output_dir / 'fine_dino').mkdir(parents=True, exist_ok=True)
    (output_dir / 'depth').mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        (output_dir / 'rgb_preview').mkdir(parents=True, exist_ok=True)
    
    # ── 1. 加载 3DGS 模型 ──
    print("加载 3DGS 模型...")
    gaussian_model = GaussianFeatureModel(feature_dim=768)
    gaussian_model.load_ply(args.ply_path)
    
    # 加载 feature embedding (用于渲染深度/RGB的几何, 不需要特征权重但需要初始化维度)
    if args.feature_model_path and os.path.exists(args.feature_model_path):
        import torch.nn as nn
        ckpt = torch.load(args.feature_model_path, map_location='cpu')
        loc_feature = ckpt['loc_feature']
        with torch.no_grad():
            gaussian_model._loc_feature = nn.Parameter(loc_feature)
        print(f"  加载特征权重: {args.feature_model_path}")
    
    # 冻结所有参数
    for param in gaussian_model.parameters():
        param.requires_grad = False
    gaussian_model = gaussian_model.to(device)
    gaussian_model.eval()
    
    # ── 2. 加载原始轨迹 ──
    print("加载原始轨迹...")
    traj_c2w = load_poses_c2w(args.traj_path)  # (N, 4, 4)
    print(f"  轨迹帧数: {len(traj_c2w)}")
    
    # ── 3. 初始化位姿采样器 ──
    gaussian_xyz = gaussian_model.get_xyz.detach().cpu().numpy()  # (M, 3)
    sampler = ScenePoseSampler(
        traj_c2w=traj_c2w,
        gaussian_xyz=gaussian_xyz,
        margin=args.margin,
        min_depth=args.min_depth,
        min_alpha_ratio=args.min_alpha_ratio,
        nearby_ratio=args.nearby_ratio,
        nearby_pos_std=args.nearby_pos_std,
        nearby_rot_std=args.nearby_rot_std,
    )
    
    # ── 4. 初始化 DINO 提取器 ──
    print("加载 DINO 特征提取器...")
    dino_extractor = DINOFeatureExtractor(device=str(device), target_hw=(35, 46))
    
    # ── 5. 采样 + 渲染 + 提取 ──
    print(f"\n开始生成 {args.num_poses} 个仿真训练样本...")
    
    # 相机参数 (Replica room_0, 640x480)
    fx, fy, cx, cy = 320.0, 320.0, 319.5, 239.5
    img_h, img_w = 480, 640
    
    # 低分辨率验证 (加速碰撞检测)
    val_h, val_w = 120, 160
    val_fx = fx * val_w / img_w
    val_fy = fy * val_h / img_h
    val_cx = cx * val_w / img_w
    val_cy = cy * val_h / img_h
    
    valid_poses = []
    attempts = 0
    max_attempts = args.num_poses * 10
    
    pbar = tqdm(total=args.num_poses, desc="生成仿真数据")
    
    while len(valid_poses) < args.num_poses and attempts < max_attempts:
        attempts += 1
        
        # 采样候选位姿
        c2w = sampler.sample_candidate()
        
        # 低分辨率快速验证
        if not sampler.validate_pose(
            c2w, gaussian_model, val_fx, val_fy, val_cx, val_cy, val_h, val_w
        ):
            continue
        
        idx = len(valid_poses)
        w2c = sampler.c2w_to_w2c(c2w)
        viewmat = torch.from_numpy(w2c).float().to(device)
        
        with torch.no_grad():
            # 渲染 RGB (全分辨率)
            result = FeatureRenderer.render_rgb(
                gaussian_model, viewmat, fx, fy, cx, cy, img_h, img_w
            )
            rgb = result['rgb']  # [3, H, W], float [0, 1]
            
            # 提取 DINO 特征
            dino_feat = dino_extractor.extract_from_tensor(rgb)  # [768, 35, 46]
            
            # 渲染深度 (特征分辨率)
            feat_fx = fx * 46 / img_w
            feat_fy = fy * 35 / img_h
            feat_cx = cx * 46 / img_w
            feat_cy = cy * 35 / img_h
            depth = FeatureRenderer.render_depth(
                gaussian_model, viewmat, feat_fx, feat_fy, feat_cx, feat_cy, 35, 46
            )  # [35, 46]
        
        # 保存特征 (float16 节省空间: 4.7MB → 2.4MB/file)
        shape_str = 'x'.join(str(s) for s in dino_feat.shape)
        torch.save(
            dino_feat.half(),
            output_dir / 'fine_dino' / f'sim_{idx:04d}_fine_dino_{shape_str}.pt'
        )
        
        # 保存深度 (float16 同样够用)
        torch.save(depth.cpu().half(), output_dir / 'depth' / f'sim_{idx:04d}_depth.pt')
        
        # 保存 RGB 预览 (可选, 每 N 帧)
        if args.save_rgb and idx % args.rgb_interval == 0:
            from PIL import Image
            rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rgb_np).save(
                output_dir / 'rgb_preview' / f'sim_{idx:04d}.png'
            )
        
        valid_poses.append(c2w)
        pbar.update(1)
        
        if idx % 100 == 0 and idx > 0:
            pbar.set_postfix({
                'accept_rate': f'{len(valid_poses)/attempts:.1%}',
            })
    
    pbar.close()
    
    # 保存所有位姿 (c2w)
    poses_c2w = np.stack(valid_poses)  # (N, 4, 4)
    np.save(output_dir / 'sim_poses_c2w.npy', poses_c2w)
    
    # 同时保存 w2c 版本 (训练时直接用)
    poses_w2c = np.stack([sampler.c2w_to_w2c(p) for p in valid_poses])
    np.save(output_dir / 'sim_poses_w2c.npy', poses_w2c)
    
    print(f"\n{'='*60}")
    print(f"✓ 完成! 生成 {len(valid_poses)} 个仿真样本")
    print(f"  尝试次数: {attempts}, 接受率: {len(valid_poses)/attempts:.1%}")
    print(f"  位姿文件: {output_dir / 'sim_poses_c2w.npy'}")
    print(f"  DINO特征: {output_dir / 'fine_dino/'}")
    print(f"  深度图:   {output_dir / 'depth/'}")
    if args.save_rgb:
        print(f"  RGB预览:  {output_dir / 'rgb_preview/'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='3DGS 仿真训练数据生成'
    )
    
    # I/O
    parser.add_argument('--output_dir', type=str,
                        default='output/sim_training_data/room_0',
                        help='输出目录')
    parser.add_argument('--ply_path', type=str,
                        default='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply')
    parser.add_argument('--feature_model_path', type=str,
                        default='output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
                        help='Feature 3DGS 模型路径 (用于几何)')
    parser.add_argument('--traj_path', type=str,
                        default='dataset/room_0/Sequence_1/traj_w_c.txt',
                        help='原始轨迹路径')
    
    # 采样参数
    parser.add_argument('--num_poses', type=int, default=5000,
                        help='生成样本数')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='包围盒外扩 margin (米)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='最小深度阈值 (米)')
    parser.add_argument('--min_alpha_ratio', type=float, default=0.5,
                        help='最小 alpha 覆盖率')
    parser.add_argument('--nearby_ratio', type=float, default=0.7,
                        help='轨迹附近样本占比')
    parser.add_argument('--nearby_pos_std', type=float, default=0.3,
                        help='轨迹附近位置扰动 std (米)')
    parser.add_argument('--nearby_rot_std', type=float, default=15.0,
                        help='轨迹附近朝向扰动 std (度)')
    
    # RGB 保存
    parser.add_argument('--save_rgb', action='store_true',
                        help='保存 RGB 预览图')
    parser.add_argument('--rgb_interval', type=int, default=50,
                        help='RGB 预览保存间隔')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    generate_sim_data(args)


if __name__ == '__main__':
    main()
