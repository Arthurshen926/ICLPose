"""
重叠区域检测模块 - 借鉴 ICL-I2PReg

ICL-I2PReg 的核心设计:
1. OverlapEstimator: 预测每个3D点是否在图像视锥内（mask confidence）
2. VertexPred: 从点云特征预测相机的粗略位姿（视锥中心和方向）
3. 两阶段流程：先预测重叠mask → 再进行精细配准

这解决了"网络不知道选哪些点"的问题：
- 网络显式学习判断哪些点在视锥内
- 不依赖GT位姿进行裁剪
- 对初始位姿误差更鲁棒

参考: reference/ICL-I2PReg/kitti/stage_2/fusion_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class OverlapEstimator(nn.Module):
    """
    重叠区域估计器
    
    预测每个3D点是否在图像视锥内
    
    输入:
        - pcd_feats: (N, C) 点云特征（所有点）
        - global_img_feats: (B, C) 全局图像特征
        - pcd_points: (N, 3) 点云坐标
        - pcd_lengths: (B,) 每个batch的点数
    
    输出:
        - overlap_mask: (N,) 每个点的视锥内置信度 [0, 1]
    """
    
    def __init__(
        self,
        pcd_feat_dim: int = 256,
        img_feat_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 图像特征投影
        self.img_mlp = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 点云特征投影
        self.pcd_mlp = nn.Sequential(
            nn.Linear(pcd_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 3D坐标编码
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # 重叠预测器
        # 输入: pcd_feat(256) + img_feat(256) + point_feat(128) = 640
        self.overlap_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 输出 [0, 1] 置信度
        )
        
        print(f"📐 OverlapEstimator initialized:")
        print(f"   - PCD feat dim: {pcd_feat_dim}")
        print(f"   - IMG feat dim: {img_feat_dim}")
    
    def forward(
        self,
        pcd_feats: torch.Tensor,      # (N, C) 或 (B, N, C)
        global_img_feats: torch.Tensor,  # (B, C)
        pcd_points: torch.Tensor,     # (N, 3) 或 (B, N, C)
        pcd_lengths: torch.Tensor = None,  # (B,) 每个batch的点数
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            overlap_mask: (N,) 或 (B, N) 每个点的视锥内置信度
        """
        # 处理不同的输入格式
        if pcd_feats.dim() == 3:
            # (B, N, C) 格式 - 已对齐
            B, N, C = pcd_feats.shape
            
            # 投影特征
            pcd_feats_proj = self.pcd_mlp(pcd_feats)  # (B, N, hidden)
            img_feats_proj = self.img_mlp(global_img_feats)  # (B, hidden)
            point_feats = self.point_mlp(pcd_points)  # (B, N, 128)
            
            # 扩展图像特征到每个点
            img_feats_expand = img_feats_proj.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden)
            
            # 拼接
            fusion_feats = torch.cat([
                pcd_feats_proj,
                img_feats_expand,
                point_feats,
            ], dim=-1)  # (B, N, hidden*2 + 128)
            
            overlap_mask = self.overlap_estimator(fusion_feats).squeeze(-1)  # (B, N)
            
        else:
            # (N, C) 格式 - 需要用 pcd_lengths 分割
            assert pcd_lengths is not None, "需要提供 pcd_lengths"
            
            # 投影特征
            pcd_feats_proj = self.pcd_mlp(pcd_feats)  # (N, hidden)
            img_feats_proj = self.img_mlp(global_img_feats)  # (B, hidden)
            point_feats = self.point_mlp(pcd_points)  # (N, 128)
            
            # 扩展图像特征到对应的点
            img_feats_expand = img_feats_proj.repeat_interleave(pcd_lengths, dim=0)  # (N, hidden)
            
            # 拼接
            fusion_feats = torch.cat([
                pcd_feats_proj,
                img_feats_expand,
                point_feats,
            ], dim=-1)  # (N, hidden*2 + 128)
            
            overlap_mask = self.overlap_estimator(fusion_feats).squeeze(-1)  # (N,)
        
        return overlap_mask


class FrustumPosePredictor(nn.Module):
    """
    视锥位姿预测器（粗定位）
    
    从点云特征和重叠mask预测相机的粗略位姿
    输出: 视锥中心 (x, z) 和方向 (cos, sin)
    
    这是 ICL-I2PReg 的 VertexPred 模块的简化版
    """
    
    def __init__(
        self,
        pcd_feat_dim: int = 256,
        img_feat_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 重叠mask编码
        self.mask_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        
        # 3D坐标编码
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # 点云特征投影
        self.pcd_mlp = nn.Sequential(
            nn.Linear(pcd_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # 图像特征投影
        self.img_mlp = nn.Sequential(
            nn.Linear(img_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # 聚合 MLP
        # 输入: mask(64) + point(128) + pcd(128) + img(128) = 448
        self.agg_mlp1 = nn.Sequential(
            nn.Linear(448, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        
        self.agg_mlp2 = nn.Sequential(
            nn.Linear(512, 512),  # local + global
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        
        # 中心预测器 (x, z)
        self.center_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # (x, z)
        )
        
        # 方向预测器 (cos, sin)
        self.dir_estimator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # (cos, sin) → 会被归一化
        )
    
    def forward(
        self,
        overlap_mask: torch.Tensor,    # (B, N) 重叠置信度
        pcd_feats: torch.Tensor,       # (B, N, C) 点云特征
        pcd_points: torch.Tensor,      # (B, N, 3) 点云坐标
        global_img_feats: torch.Tensor,  # (B, C) 全局图像特征
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            vertex: (B, 4) [center_x, center_z, dir_cos, dir_sin]
        """
        B, N, _ = pcd_feats.shape
        
        # 编码各个特征
        mask_feats = self.mask_mlp(overlap_mask.unsqueeze(-1))  # (B, N, 64)
        point_feats = self.point_mlp(pcd_points)  # (B, N, 128)
        pcd_feats_proj = self.pcd_mlp(pcd_feats)  # (B, N, 128)
        img_feats_proj = self.img_mlp(global_img_feats)  # (B, 128)
        
        # 扩展图像特征
        img_feats_expand = img_feats_proj.unsqueeze(1).expand(-1, N, -1)  # (B, N, 128)
        
        # 拼接
        fusion_feats = torch.cat([
            mask_feats,
            point_feats,
            pcd_feats_proj,
            img_feats_expand,
        ], dim=-1)  # (B, N, 448)
        
        # 局部聚合
        local_feats = self.agg_mlp1(fusion_feats)  # (B, N, 256)
        
        # 使用重叠mask加权平均（关注重叠区域）
        weights = overlap_mask.unsqueeze(-1)  # (B, N, 1)
        global_feats = (local_feats * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # (B, 256)
        
        # 全局+局部聚合
        global_feats_expand = global_feats.unsqueeze(1).expand(-1, N, -1)
        combined_feats = torch.cat([local_feats, global_feats_expand], dim=-1)  # (B, N, 512)
        combined_feats = self.agg_mlp2(combined_feats)  # (B, N, 512)
        
        # 再次加权平均
        final_feats = (combined_feats * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # (B, 512)
        
        # 预测中心和方向
        center = self.center_estimator(final_feats)  # (B, 2)
        direction = F.normalize(self.dir_estimator(final_feats), dim=-1)  # (B, 2) 归一化
        
        vertex = torch.cat([center, direction], dim=-1)  # (B, 4)
        
        return vertex
    
    def vertex_to_pose(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        将 vertex (center_x, center_z, cos, sin) 转换为 4x4 位姿矩阵
        
        这是一个简化版本，假设相机只在 XZ 平面旋转（Y轴朝上）
        
        Args:
            vertex: (B, 4) [center_x, center_z, cos, sin]
        
        Returns:
            pose: (B, 4, 4) 位姿矩阵 (camera-to-world)
        """
        B = vertex.shape[0]
        device = vertex.device
        
        center_x = vertex[:, 0]
        center_z = vertex[:, 1]
        cos_theta = vertex[:, 2]
        sin_theta = vertex[:, 3]
        
        # 构建旋转矩阵（绕Y轴）
        R = torch.zeros(B, 3, 3, device=device)
        R[:, 0, 0] = cos_theta
        R[:, 0, 2] = -sin_theta
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = sin_theta
        R[:, 2, 2] = cos_theta
        
        # 构建平移向量
        t = torch.zeros(B, 3, device=device)
        t[:, 0] = center_x
        t[:, 2] = center_z
        
        # 组合为 4x4 矩阵
        pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        pose[:, :3, :3] = R
        pose[:, :3, 3] = t
        
        return pose


class OverlapDetectionModule(nn.Module):
    """
    完整的重叠检测模块
    
    整合了:
    1. OverlapEstimator: 预测点云-图像重叠mask
    2. FrustumPosePredictor: 从重叠区域预测粗略位姿
    
    工作流程:
    1. 输入: 全局点云特征 + 全局图像特征 + 点云坐标
    2. 预测: 每个点的重叠置信度
    3. 输出: 粗略位姿 + 重叠mask（用于后续精细配准）
    """
    
    def __init__(
        self,
        pcd_feat_dim: int = 256,
        img_feat_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.overlap_estimator = OverlapEstimator(
            pcd_feat_dim=pcd_feat_dim,
            img_feat_dim=img_feat_dim,
            hidden_dim=hidden_dim,
        )
        
        self.frustum_predictor = FrustumPosePredictor(
            pcd_feat_dim=pcd_feat_dim,
            img_feat_dim=img_feat_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(
        self,
        pcd_feats: torch.Tensor,       # (B, N, C) 点云特征
        global_img_feats: torch.Tensor,  # (B, C) 全局图像特征
        pcd_points: torch.Tensor,      # (B, N, 3) 点云坐标
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            overlap_mask: (B, N) 重叠置信度 [0, 1]
            coarse_pose: (B, 4, 4) 粗略位姿
            vertex: (B, 4) 视锥顶点参数
        """
        # 预测重叠mask
        overlap_mask = self.overlap_estimator(
            pcd_feats, global_img_feats, pcd_points
        )
        
        # 预测粗略位姿
        vertex = self.frustum_predictor(
            overlap_mask, pcd_feats, pcd_points, global_img_feats
        )
        
        # 转换为位姿矩阵
        coarse_pose = self.frustum_predictor.vertex_to_pose(vertex)
        
        return overlap_mask, coarse_pose, vertex


class OverlapLoss(nn.Module):
    """
    重叠检测损失函数
    
    监督信号: 使用 GT 位姿计算每个点是否真正在视锥内
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def compute_gt_overlap_mask(
        self,
        pcd_points: torch.Tensor,  # (B, N, 3)
        gt_pose: torch.Tensor,     # (B, 4, 4) c2w
        intrinsics: torch.Tensor,  # (B, 3, 3)
        image_size: Tuple[int, int] = (640, 480),
    ) -> torch.Tensor:
        """
        使用 GT 位姿计算真实的重叠mask
        
        Returns:
            gt_mask: (B, N) bool, True = 在视锥内
        """
        B, N, _ = pcd_points.shape
        W, H = image_size
        
        # c2w → w2c
        w2c = torch.inverse(gt_pose)  # (B, 4, 4)
        
        # 转换到相机坐标系
        R = w2c[:, :3, :3]  # (B, 3, 3)
        t = w2c[:, :3, 3:4]  # (B, 3, 1)
        
        points_cam = torch.bmm(R, pcd_points.transpose(1, 2)) + t  # (B, 3, N)
        points_cam = points_cam.transpose(1, 2)  # (B, N, 3)
        
        # 投影到图像平面
        z = points_cam[:, :, 2:3]  # (B, N, 1)
        points_proj = torch.bmm(intrinsics, points_cam.transpose(1, 2))  # (B, 3, N)
        points_proj = points_proj.transpose(1, 2)  # (B, N, 3)
        
        u = points_proj[:, :, 0] / (z.squeeze(-1) + 1e-8)  # (B, N)
        v = points_proj[:, :, 1] / (z.squeeze(-1) + 1e-8)  # (B, N)
        
        # 判断是否在视锥内
        in_front = z.squeeze(-1) > 0.05  # 在相机前方
        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)  # 在图像内
        
        gt_mask = in_front & in_image  # (B, N)
        
        return gt_mask
    
    def forward(
        self,
        pred_mask: torch.Tensor,   # (B, N) 预测的重叠置信度
        pcd_points: torch.Tensor,  # (B, N, 3)
        gt_pose: torch.Tensor,     # (B, 4, 4)
        intrinsics: torch.Tensor,  # (B, 3, 3)
        image_size: Tuple[int, int] = (640, 480),
    ) -> torch.Tensor:
        """
        计算重叠检测损失
        
        Returns:
            loss: 二分类交叉熵损失
        """
        # 计算 GT mask
        gt_mask = self.compute_gt_overlap_mask(
            pcd_points, gt_pose, intrinsics, image_size
        ).float()
        
        # 二分类损失（带类别权重）
        pos_mask = gt_mask > 0.5
        neg_mask = ~pos_mask
        
        # 加权 BCE
        bce = F.binary_cross_entropy(pred_mask, gt_mask, reduction='none')
        
        pos_loss = (bce * pos_mask.float()).sum() / (pos_mask.sum() + 1e-8)
        neg_loss = (bce * neg_mask.float()).sum() / (neg_mask.sum() + 1e-8)
        
        loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss
        
        return loss


if __name__ == '__main__':
    """测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 测试 OverlapDetectionModule ===")
    
    module = OverlapDetectionModule(
        pcd_feat_dim=256,
        img_feat_dim=256,
    ).to(device)
    
    B, N, C = 4, 1000, 256
    pcd_feats = torch.randn(B, N, C).to(device)
    global_img_feats = torch.randn(B, C).to(device)
    pcd_points = torch.randn(B, N, 3).to(device) * 5  # 范围 [-5, 5]
    
    overlap_mask, coarse_pose, vertex = module(pcd_feats, global_img_feats, pcd_points)
    
    print(f"Overlap mask: {overlap_mask.shape}, range: [{overlap_mask.min():.3f}, {overlap_mask.max():.3f}]")
    print(f"Coarse pose: {coarse_pose.shape}")
    print(f"Vertex: {vertex.shape}")
    
    # 测试损失函数
    print("\n=== 测试 OverlapLoss ===")
    
    loss_fn = OverlapLoss()
    
    gt_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device)
    gt_pose[:, :3, 3] = torch.randn(B, 3).to(device)
    
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    intrinsics[:, 0, 2] = 319.5
    intrinsics[:, 1, 2] = 239.5
    
    loss = loss_fn(overlap_mask, pcd_points, gt_pose, intrinsics)
    print(f"Overlap loss: {loss.item():.4f}")
    
    # 参数量
    params = sum(p.numel() for p in module.parameters())
    print(f"\nTotal params: {params / 1e6:.2f}M")
