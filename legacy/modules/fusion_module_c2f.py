"""
跨模态融合模块 - 增强版（支持 Coarse-to-Fine 辅助损失）

借鉴 MaRepo 的关键设计：
1. 中间层辅助损失 - 每隔2层输出一次特征，计算中间位姿损失
2. 动态位置编码 - 使用相机内参归一化的坐标 (u-cx)/fx
3. 周期性 Skip Connection - 每4层做一次残差连接

参考: reference/marepo/transformer/transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .transformer import TransformerLayer
import math


class PositionalEncoding2DIntrinsic(nn.Module):
    """
    相机感知的2D位置编码
    
    MaRepo 的关键改进：使用焦距归一化的坐标而不是简单像素坐标
    
    输入: (u, v) 像素坐标
    处理: ((u - cx) / fx, (v - cy) / fy) → 归一化平面坐标 → 正弦编码
    意义: 编码的是"光线方向"，而不是"像素位置"
    """
    
    def __init__(self, feature_dim: int = 256, num_freqs: int = 8):
        """
        Args:
            feature_dim: 输出特征维度
            num_freqs: 频率数量（类似 NeRF 位置编码）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_freqs = num_freqs
        
        # 频率带: 2^0, 2^1, ..., 2^(num_freqs-1)
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # 编码后维度: 2 (x,y) * (1 + 2*num_freqs) = 2 + 4*num_freqs
        embed_dim = 2 + 4 * num_freqs  # include_input=True
        
        # 投影到目标维度
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self, 
        coords_2d: torch.Tensor,  # (B, N, 2) 像素坐标 (u, v)
        intrinsics: torch.Tensor = None,  # (B, 3, 3) 相机内参
    ) -> torch.Tensor:
        """
        Args:
            coords_2d: (B, N, 2) 像素坐标
            intrinsics: (B, 3, 3) 相机内参矩阵
        
        Returns:
            pos_embed: (B, N, feature_dim) 位置编码
        """
        B, N, _ = coords_2d.shape
        
        # === 焦距归一化 ===
        if intrinsics is not None:
            # 提取内参
            fx = intrinsics[:, 0, 0].view(B, 1, 1)  # (B, 1, 1)
            fy = intrinsics[:, 1, 1].view(B, 1, 1)
            cx = intrinsics[:, 0, 2].view(B, 1, 1)
            cy = intrinsics[:, 1, 2].view(B, 1, 1)
            
            # 归一化到光线方向: (u - cx) / fx, (v - cy) / fy
            u = coords_2d[..., 0:1]  # (B, N, 1)
            v = coords_2d[..., 1:2]
            
            x = (u - cx) / (fx + 1e-8)  # 归一化平面 x
            y = (v - cy) / (fy + 1e-8)  # 归一化平面 y
            
            coords_norm = torch.cat([x, y], dim=-1)  # (B, N, 2)
        else:
            # 简单归一化到 [-1, 1]
            coords_norm = coords_2d / 320.0 - 1.0
        
        # === 正弦位置编码 (NeRF-style) ===
        # 输出: [x, y, sin(2^0*x), cos(2^0*x), sin(2^0*y), cos(2^0*y), ...]
        embed_list = [coords_norm]  # 原始坐标
        
        for freq in self.freq_bands:
            embed_list.append(torch.sin(freq * coords_norm))
            embed_list.append(torch.cos(freq * coords_norm))
        
        embed = torch.cat(embed_list, dim=-1)  # (B, N, 2 + 4*num_freqs)
        
        # 投影到目标维度
        pos_embed = self.proj(embed)  # (B, N, feature_dim)
        
        return pos_embed


class PositionalEncoding3DNeRF(nn.Module):
    """
    3D坐标的 NeRF 风格位置编码
    
    对 3D 点云坐标进行多频率编码，增强高频细节表达
    """
    
    def __init__(
        self, 
        feature_dim: int = 256, 
        num_freqs: int = 5,
        coord_range: float = 50.0,
    ):
        """
        Args:
            feature_dim: 输出特征维度
            num_freqs: 频率数量
            coord_range: 坐标范围（用于归一化到 [-π, π]）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_freqs = num_freqs
        self.coord_range = coord_range
        
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # 编码后维度: 3 (x,y,z) * (1 + 2*num_freqs)
        embed_dim = 3 * (1 + 2 * num_freqs)
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 场景中心（可设置）
        self.register_buffer('scene_center', torch.zeros(3))
    
    def set_scene_center(self, center: torch.Tensor):
        """设置场景中心"""
        self.scene_center = center.view(3)
    
    def forward(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_3d: (B, N, 3) 3D坐标
        
        Returns:
            pos_embed: (B, N, feature_dim) 位置编码
        """
        # 中心化并归一化到 [-π, π]
        coords = coords_3d - self.scene_center.view(1, 1, 3)
        coords = torch.clamp(coords / self.coord_range, -1, 1) * math.pi
        
        # NeRF 编码
        embed_list = [coords]
        for freq in self.freq_bands:
            embed_list.append(torch.sin(freq * coords))
            embed_list.append(torch.cos(freq * coords))
        
        embed = torch.cat(embed_list, dim=-1)
        pos_embed = self.proj(embed)
        
        return pos_embed


class CrossModalFusionModuleC2F(nn.Module):
    """
    跨模态融合模块 - Coarse-to-Fine 版本
    
    关键改进（借鉴 MaRepo）：
    1. 中间层辅助损失 - 每隔 output_interval 层输出一次
    2. 动态位置编码 - 支持相机内参归一化
    3. 周期性 Skip Connection - 稳定深层网络训练
    
    用法:
        module = CrossModalFusionModuleC2F(num_layers=8, output_interval=2)
        # 返回 4 个阶段的特征（第2、4、6、8层）
        query_stages, img_tokens, pcd_tokens = module(...)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = 'ReLU',
        output_interval: int = 2,  # 每隔几层输出一次
        use_skip_connection: bool = True,
        skip_interval: int = 4,  # Skip connection 间隔
    ):
        """
        Args:
            feature_dim: 特征维度
            num_layers: 总层数（必须是2的倍数）
            num_heads: 注意力头数
            dropout: Dropout概率
            output_interval: 中间输出间隔（每隔多少层输出一次）
            use_skip_connection: 是否使用周期性 skip connection
            skip_interval: Skip connection 间隔
        """
        super().__init__()
        
        # 确保层数是2的倍数
        if num_layers % 2 != 0:
            num_layers = (num_layers // 2) * 2
            if num_layers < 2:
                num_layers = 2
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_blocks = num_layers // 2
        self.output_interval = output_interval
        self.use_skip_connection = use_skip_connection
        self.skip_interval = skip_interval
        
        # 计算输出阶段数
        self.num_stages = num_layers // output_interval
        
        # Token投影层
        self.img_in_proj = nn.Linear(feature_dim, feature_dim)
        self.pcd_in_proj = nn.Linear(feature_dim, feature_dim)
        
        # Self-Attention层
        self.self_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # Cross-Attention层
        self.cross_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # 每个阶段的输出投影（共享权重或独立）
        self.stage_out_projs = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(self.num_stages)
        ])
        
        print(f"📐 CrossModalFusionModuleC2F initialized:")
        print(f"   - Num layers: {num_layers}")
        print(f"   - Output interval: {output_interval} (→ {self.num_stages} stages)")
        print(f"   - Skip connection: every {skip_interval} layers" if use_skip_connection else "   - Skip connection: disabled")
        
    def forward(
        self,
        query_feats: torch.Tensor,
        img_feats: torch.Tensor,
        pcd_feats: torch.Tensor,
        query_pos_embeds: Optional[torch.Tensor] = None,
        img_pos_embeds: Optional[torch.Tensor] = None,
        pcd_pos_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            query_stages: List[(B, N, C)] 每个阶段的query特征（用于计算辅助损失）
            img_tokens: 处理后的图像tokens
            pcd_tokens: 处理后的点云tokens
        """
        # 投影tokens
        img_tokens = self.img_in_proj(img_feats)
        pcd_tokens = self.pcd_in_proj(pcd_feats)
        
        query = query_feats
        query_skip = query.clone() if self.use_skip_connection else None
        
        query_stages = []
        stage_idx = 0
        
        # 遍历所有blocks
        for block_idx in range(self.num_blocks):
            layer_idx = block_idx * 2
            
            # Layer 1: Self-Attention + Cross-Attention with Image
            query_s1 = self.self_attn_layers[layer_idx](
                q=query,
                k=query,
                v=query,
                q_embeds=query_pos_embeds,
                k_embeds=query_pos_embeds,
            )
            query_c1 = self.cross_attn_layers[layer_idx](
                q=query_s1,
                k=img_tokens,
                v=img_tokens,
                q_embeds=query_pos_embeds,
                k_embeds=img_pos_embeds,
            )
            
            # 检查是否需要输出（layer_idx + 1 是当前完成的层数）
            current_layer = layer_idx + 1
            if current_layer % self.output_interval == 0:
                stage_output = self.stage_out_projs[stage_idx](query_c1)
                query_stages.append(stage_output)
                stage_idx += 1
            
            # Skip connection
            if self.use_skip_connection and current_layer % self.skip_interval == 0:
                query_c1 = query_c1 + query_skip
                query_skip = query_c1.clone()
            
            # Layer 2: Self-Attention + Cross-Attention with PointCloud
            query_s2 = self.self_attn_layers[layer_idx + 1](
                q=query_c1,
                k=query_c1,
                v=query_c1,
                q_embeds=query_pos_embeds,
                k_embeds=query_pos_embeds,
            )
            query_c2 = self.cross_attn_layers[layer_idx + 1](
                q=query_s2,
                k=pcd_tokens,
                v=pcd_tokens,
                q_embeds=query_pos_embeds,
                k_embeds=pcd_pos_embeds,
            )
            
            # 检查是否需要输出
            current_layer = layer_idx + 2
            if current_layer % self.output_interval == 0:
                stage_output = self.stage_out_projs[stage_idx](query_c2)
                query_stages.append(stage_output)
                stage_idx += 1
            
            # Skip connection
            if self.use_skip_connection and current_layer % self.skip_interval == 0:
                query_c2 = query_c2 + query_skip
                query_skip = query_c2.clone()
            
            # 更新query
            query = query_c2
        
        return query_stages, img_tokens, pcd_tokens


class ICPoseNetC2F(nn.Module):
    """
    ICPoseNet - Coarse-to-Fine 版本
    
    核心改进:
    1. 中间层辅助损失 - 多阶段监督
    2. 相机感知位置编码 - 焦距归一化
    3. NeRF 风格 3D 编码 - 增强高频细节
    
    输出:
        - pose_stages: List[(B, 4, 4)] 每个阶段的位姿预测
        - final_pose: (B, 4, 4) 最终位姿
        - keypoints: (img_kp, pcd_kp) 检测到的关键点
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_queries: int = 64,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_interval: int = 2,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_stages = num_layers // output_interval
        
        # 可学习的 Query Embedding
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        
        # 动态位置编码
        self.pos_encoding_2d = PositionalEncoding2DIntrinsic(feature_dim)
        self.pos_encoding_3d = PositionalEncoding3DNeRF(feature_dim)
        
        # 跨模态融合（C2F版本）
        self.fusion = CrossModalFusionModuleC2F(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_interval=output_interval,
            use_skip_connection=True,
            skip_interval=4,
        )
        
        # 关键点检测头（2D和3D）
        self.keypoint_head_2d = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # (u, v)
        )
        
        self.keypoint_head_3d = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # (x, y, z)
        )
        
        # 每个阶段共享的位姿回归头
        self.pose_head = PoseRegressionHead(feature_dim, hidden_dim)
        
        print(f"📐 ICPoseNetC2F initialized:")
        print(f"   - Feature dim: {feature_dim}")
        print(f"   - Num queries: {num_queries}")
        print(f"   - Num stages: {self.num_stages}")
    
    def forward(
        self,
        img_feats: torch.Tensor,  # (B, C, H, W) 图像特征
        pcd_feats: torch.Tensor,  # (B, N_pcd, C) 点云特征
        pcd_coords: torch.Tensor,  # (B, N_pcd, 3) 点云坐标
        intrinsics: torch.Tensor = None,  # (B, 3, 3) 相机内参
        initial_pose: torch.Tensor = None,  # (B, 4, 4) 初始位姿
    ):
        """
        前向传播
        
        Returns:
            output_dict: {
                'pose_stages': List[(B, 4, 4)],
                'final_pose': (B, 4, 4),
                'img_keypoints': (B, N_query, 2),
                'pcd_keypoints': (B, N_query, 3),
                'rotation_6d': (B, 6),
                'translation': (B, 3),
            }
        """
        B = img_feats.shape[0]
        device = img_feats.device
        
        # 展平图像特征
        if img_feats.dim() == 4:
            _, C, H, W = img_feats.shape
            img_feats_flat = img_feats.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            
            # 生成图像位置坐标
            y_coords = torch.arange(H, device=device).float()
            x_coords = torch.arange(W, device=device).float()
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            img_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
            img_coords = img_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
        else:
            img_feats_flat = img_feats
            img_coords = None
        
        # 位置编码
        if img_coords is not None:
            img_pos_embeds = self.pos_encoding_2d(img_coords, intrinsics)
        else:
            img_pos_embeds = None
        
        pcd_pos_embeds = self.pos_encoding_3d(pcd_coords)
        
        # Query embedding
        query_feats = self.query_embed.expand(B, -1, -1)  # (B, N_query, C)
        query_pos_embeds = None  # Query 不需要位置编码
        
        # 跨模态融合 (C2F)
        query_stages, _, _ = self.fusion(
            query_feats=query_feats,
            img_feats=img_feats_flat,
            pcd_feats=pcd_feats,
            query_pos_embeds=query_pos_embeds,
            img_pos_embeds=img_pos_embeds,
            pcd_pos_embeds=pcd_pos_embeds,
        )
        
        # 关键点检测（使用最后一个阶段的特征）
        final_query = query_stages[-1]
        img_keypoints = self.keypoint_head_2d(final_query)  # (B, N_query, 2)
        pcd_keypoints = self.keypoint_head_3d(final_query)  # (B, N_query, 3)
        
        # 每个阶段的位姿回归
        pose_stages = []
        for stage_query in query_stages:
            pose, rot_6d, trans = self.pose_head(
                stage_query, img_keypoints, pcd_keypoints
            )
            pose_stages.append(pose)
        
        # 获取最终位姿
        final_pose = pose_stages[-1]
        
        # 如果有初始位姿，组合相对位姿
        if initial_pose is not None:
            composed_stages = []
            for pose in pose_stages:
                composed = initial_pose @ pose
                composed_stages.append(composed)
            pose_stages = composed_stages
            final_pose = pose_stages[-1]
        
        return {
            'pose_stages': pose_stages,
            'final_pose': final_pose,
            'pose_matrix': final_pose,  # 兼容性
            'img_keypoints': img_keypoints,
            'pcd_keypoints': pcd_keypoints,
            'rotation_6d': rot_6d,
            'translation': trans,
        }


class PoseRegressionHead(nn.Module):
    """
    位姿回归头（共享权重）
    
    从 query 特征回归 6-DOF 位姿
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        # 2D keypoint 编码
        self.kp_2d_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
        )
        
        # 3D keypoint 编码
        self.kp_3d_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
        )
        
        # Query 投影
        self.query_proj = nn.Linear(feature_dim, 128)
        
        # 聚合
        self.aggregation = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 旋转头 (6D)
        self.rotation_head = nn.Linear(hidden_dim, 6)
        
        # 平移头
        self.translation_head = nn.Linear(hidden_dim, 3)
    
    def forward(
        self,
        query_feats: torch.Tensor,  # (B, N, C)
        img_keypoints: torch.Tensor,  # (B, N, 2)
        pcd_keypoints: torch.Tensor,  # (B, N, 3)
    ):
        """
        Returns:
            pose: (B, 4, 4) 位姿矩阵
            rotation_6d: (B, 6)
            translation: (B, 3)
        """
        # 编码
        kp_2d_feats = self.kp_2d_encoder(img_keypoints)
        kp_3d_feats = self.kp_3d_encoder(pcd_keypoints)
        query_feats = self.query_proj(query_feats)
        
        # 拼接并聚合
        combined = torch.cat([kp_2d_feats, query_feats, kp_3d_feats], dim=-1)
        global_feat = combined.mean(dim=1)  # (B, 384)
        
        feat = self.aggregation(global_feat)
        
        # 回归
        rotation_6d = self.rotation_head(feat)
        translation = self.translation_head(feat)
        
        # 构建位姿矩阵
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)
        
        B = rotation_6d.shape[0]
        pose = torch.eye(4, device=rotation_6d.device).unsqueeze(0).expand(B, -1, -1).clone()
        pose[:, :3, :3] = rotation_matrix
        pose[:, :3, 3] = translation
        
        return pose, rotation_6d, translation


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D → 旋转矩阵"""
    a1 = d6[:, :3]
    a2 = d6[:, 3:]
    
    b1 = F.normalize(a1, dim=1)
    dot = (a2 * b1).sum(dim=1, keepdim=True)
    u2 = a2 - dot * b1
    b2 = F.normalize(u2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    return torch.stack([b1, b2, b3], dim=-1)


if __name__ == '__main__':
    """测试代码"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 C2F Fusion Module
    print("=== 测试 CrossModalFusionModuleC2F ===")
    fusion = CrossModalFusionModuleC2F(
        feature_dim=256,
        num_layers=8,
        output_interval=2,
    ).to(device)
    
    B, N_query, N_img, N_pcd, C = 4, 64, 100, 200, 256
    query = torch.randn(B, N_query, C).to(device)
    img = torch.randn(B, N_img, C).to(device)
    pcd = torch.randn(B, N_pcd, C).to(device)
    
    stages, _, _ = fusion(query, img, pcd)
    print(f"Number of stages: {len(stages)}")
    for i, s in enumerate(stages):
        print(f"  Stage {i+1}: {s.shape}")
    
    # 测试位置编码
    print("\n=== 测试 PositionalEncoding2DIntrinsic ===")
    pos_enc = PositionalEncoding2DIntrinsic(256).to(device)
    coords = torch.randn(B, 100, 2).to(device) * 320 + 320
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    intrinsics[:, 0, 2] = 319.5
    intrinsics[:, 1, 2] = 239.5
    
    pos_embed = pos_enc(coords, intrinsics)
    print(f"Position embedding: {pos_embed.shape}")
    
    # 测试完整模型
    print("\n=== 测试 ICPoseNetC2F ===")
    model = ICPoseNetC2F(
        feature_dim=256,
        num_queries=64,
        num_layers=8,
        output_interval=2,
    ).to(device)
    
    img_feats = torch.randn(B, 256, 30, 40).to(device)
    pcd_feats = torch.randn(B, 200, 256).to(device)
    pcd_coords = torch.randn(B, 200, 3).to(device)
    
    output = model(img_feats, pcd_feats, pcd_coords, intrinsics)
    
    print(f"Pose stages: {len(output['pose_stages'])}")
    print(f"Final pose: {output['final_pose'].shape}")
    print(f"Image keypoints: {output['img_keypoints'].shape}")
    print(f"PCD keypoints: {output['pcd_keypoints'].shape}")
    
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {params / 1e6:.2f}M")
