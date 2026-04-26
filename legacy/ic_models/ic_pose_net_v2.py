"""
ICPoseNet V2: 改进版隐式对应关系位姿估计网络

关键改进：
1. C2F (Coarse-to-Fine): 多阶段辅助损失，强制浅层学习粗略特征
2. 重叠检测: 预测点云-图像重叠区域，对初始位姿误差鲁棒
3. 动态位置编码: 使用相机内参归一化2D坐标
4. MaRepo风格Transformer: 12层带跳跃连接
5. 旋转损失Warmup: 先用L1/Cosine，后用测地线距离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from modules.fusion_module_c2f import (
    CrossModalFusionModuleC2F,
    PositionalEncoding2DIntrinsic,
    PositionalEncoding3DNeRF,
    PoseRegressionHead,
)
from modules.overlap_detection import (
    OverlapEstimator,
    FrustumPosePredictor,
    OverlapDetectionModule,
)
from modules.pose_regressor import rotation_6d_to_matrix


class ICPoseNetV2(nn.Module):
    """
    ICPoseNet V2 - 集成C2F和重叠检测的位姿估计网络
    
    架构流程:
    1. 输入: 预提取的2D图像特征 + 3D点云特征
    2. 重叠检测: 预测哪些3D点在图像视野内
    3. C2F跨模态融合: 多阶段输出的Transformer
    4. 位姿回归: 多阶段位姿预测 + 软关键点提取
    
    训练策略:
    - 前50 epoch: 旋转用L1+Cosine损失 (warmup)
    - 后续: 逐渐过渡到测地线距离
    - C2F辅助损失: 每2层一个stage，浅层权重较低
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_queries: int = 128,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_interval: int = 2,
        use_overlap_detection: bool = True,
        use_c2f: bool = True,
        nerf_frequencies: int = 10,
    ):
        """
        Args:
            feature_dim: 特征维度
            num_queries: 可学习query数量
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout概率
            output_interval: C2F输出间隔（每几层输出一次）
            use_overlap_detection: 是否使用重叠检测
            use_c2f: 是否使用C2F多阶段输出
            nerf_frequencies: NeRF位置编码频率数
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.use_overlap_detection = use_overlap_detection
        self.use_c2f = use_c2f
        self.output_interval = output_interval
        
        # 可学习的query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.xavier_uniform_(self.query_embed)
        
        # 位置编码模块（NeRF风格3D + 内参归一化2D）
        self.pos_enc_3d = PositionalEncoding3DNeRF(
            feature_dim=feature_dim,
            num_freqs=nerf_frequencies,
        )
        self.pos_enc_2d = PositionalEncoding2DIntrinsic(
            feature_dim=feature_dim,
            num_freqs=nerf_frequencies,
        )
        
        # 重叠检测模块（可选）
        if use_overlap_detection:
            self.overlap_module = OverlapDetectionModule(
                pcd_feat_dim=feature_dim,
                img_feat_dim=feature_dim,
                hidden_dim=256,
            )
        
        # C2F跨模态融合模块
        self.fusion_module = CrossModalFusionModuleC2F(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_interval=output_interval if use_c2f else num_layers,
        )
        
        # 计算C2F阶段数
        self.num_stages = num_layers // output_interval if use_c2f else 1
        
        # 每个阶段的位姿回归头
        self.pose_heads = nn.ModuleList([
            PoseRegressionHead(
                feature_dim=feature_dim,
                hidden_dim=512,
            )
            for _ in range(self.num_stages)
        ])
        
    def forward(
        self,
        img_feats: torch.Tensor,
        pcd_feats: torch.Tensor,
        img_pixels: torch.Tensor,
        pcd_points: torch.Tensor,
        intrinsics: torch.Tensor = None,
        initial_pose: torch.Tensor = None,
        img_global_feats: torch.Tensor = None,
        return_all_stages: bool = True,
    ):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            img_pixels: (B, N_img, 2) 2D像素坐标
            pcd_points: (B, N_pcd, 3) 3D点坐标
            intrinsics: (B, 3, 3) 相机内参矩阵（可选，用于归一化2D坐标）
            initial_pose: (B, 4, 4) 初始位姿（可选，用于重叠检测基准）
            img_global_feats: (B, C) 全局图像特征（可选，用于重叠检测）
            return_all_stages: 是否返回所有阶段的输出
            
        Returns:
            dict: {
                'pose_matrix': (B, 4, 4) 最终位姿矩阵,
                'rotation_6d': (B, 6) 6D旋转,
                'translation': (B, 3) 平移,
                'stage_outputs': list of stage outputs if return_all_stages,
                'overlap_mask': (B, N_pcd) 重叠掩码 if use_overlap_detection,
                'keypoints_2d': (B, N_query, 2) 2D关键点,
                'keypoints_3d': (B, N_query, 3) 3D关键点,
            }
        """
        batch_size = img_feats.shape[0]
        device = img_feats.device
        
        outputs = {}
        
        # ============ 1. 重叠检测（可选）============
        overlap_mask = None
        coarse_pose = None
        
        if self.use_overlap_detection and img_global_feats is not None:
            overlap_mask, coarse_pose, vertex = self.overlap_module(
                pcd_feats=pcd_feats,
                global_img_feats=img_global_feats,
                pcd_points=pcd_points,
            )
            outputs['overlap_mask'] = overlap_mask
            outputs['coarse_pose'] = coarse_pose
            outputs['vertex'] = vertex
            
            # 使用重叠掩码加权点云特征
            # 高重叠分数的点权重更大
            pcd_feats_weighted = pcd_feats * overlap_mask.unsqueeze(-1)
        else:
            pcd_feats_weighted = pcd_feats
        
        # ============ 2. 位置编码 ============
        # 3D位置编码（NeRF风格）
        pcd_pos_embeds = self.pos_enc_3d(pcd_points)  # (B, N_pcd, C)
        
        # 2D位置编码（使用内参归一化）
        if intrinsics is not None:
            img_pos_embeds = self.pos_enc_2d(img_pixels, intrinsics)  # (B, N_img, C)
        else:
            img_pos_embeds = self.pos_enc_2d(img_pixels)  # (B, N_img, C)
        
        # ============ 3. Query Embeddings ============
        query_feats = self.query_embed.expand(batch_size, -1, -1)  # (B, N_query, C)
        
        # ============ 4. C2F跨模态融合 ============
        stage_query_feats, final_img_tokens, final_pcd_tokens = self.fusion_module(
            query_feats=query_feats,
            img_feats=img_feats,
            pcd_feats=pcd_feats_weighted,
            query_pos_embeds=None,
            img_pos_embeds=img_pos_embeds,
            pcd_pos_embeds=pcd_pos_embeds,
        )
        
        # stage_query_feats: list of (B, N_query, C)
        # final_img_tokens: (B, N_img, C)
        # final_pcd_tokens: (B, N_pcd, C)
        
        # ============ 5. 多阶段位姿回归 ============
        stage_outputs = []
        
        for stage_idx, (query_feat, pose_head) in enumerate(zip(stage_query_feats, self.pose_heads)):
            # 计算attention heatmaps
            # 2D heatmap
            img_attn = torch.matmul(
                query_feat,
                final_img_tokens.transpose(1, 2)
            ) / (self.feature_dim ** 0.5)
            img_attn = F.softmax(img_attn, dim=-1)  # (B, N_query, N_img)
            
            # 3D heatmap
            pcd_attn = torch.matmul(
                query_feat,
                final_pcd_tokens.transpose(1, 2)
            ) / (self.feature_dim ** 0.5)
            pcd_attn = F.softmax(pcd_attn, dim=-1)  # (B, N_query, N_pcd)
            
            # Soft-argmax提取关键点
            keypoints_2d = torch.matmul(img_attn, img_pixels)  # (B, N_query, 2)
            keypoints_3d = torch.matmul(pcd_attn, pcd_points)  # (B, N_query, 3)
            
            # 位姿回归
            pose_matrix, rotation_6d, translation = pose_head(query_feat, keypoints_2d, keypoints_3d)
            
            stage_outputs.append({
                'rotation_6d': rotation_6d,
                'translation': translation,
                'pose_matrix': pose_matrix,
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'img_attn': img_attn,
                'pcd_attn': pcd_attn,
            })
        
        # ============ 6. 最终输出（最后一个阶段）============
        final_stage = stage_outputs[-1]
        outputs['pose_matrix'] = final_stage['pose_matrix']
        outputs['rotation_6d'] = final_stage['rotation_6d']
        outputs['translation'] = final_stage['translation']
        outputs['keypoints_2d'] = final_stage['keypoints_2d']
        outputs['keypoints_3d'] = final_stage['keypoints_3d']
        outputs['img_keypoint_heatmap'] = final_stage['img_attn']
        outputs['pcd_keypoint_heatmap'] = final_stage['pcd_attn']
        
        if return_all_stages:
            outputs['stage_outputs'] = stage_outputs
        
        return outputs
    
    def get_trainable_params(self, lr_base: float = 1e-4):
        """
        获取分层学习率的参数组
        
        Returns:
            list: 参数组列表，用于优化器
        """
        param_groups = []
        
        # 重叠检测模块 - 较高学习率
        if self.use_overlap_detection:
            param_groups.append({
                'params': self.overlap_module.parameters(),
                'lr': lr_base * 2.0,
                'name': 'overlap_module'
            })
        
        # 位置编码 - 标准学习率
        param_groups.append({
            'params': list(self.pos_enc_3d.parameters()) + list(self.pos_enc_2d.parameters()),
            'lr': lr_base,
            'name': 'pos_encoding'
        })
        
        # Query embeddings - 标准学习率
        param_groups.append({
            'params': [self.query_embed],
            'lr': lr_base,
            'name': 'query_embed'
        })
        
        # Fusion模块 - 标准学习率
        param_groups.append({
            'params': self.fusion_module.parameters(),
            'lr': lr_base,
            'name': 'fusion_module'
        })
        
        # 位姿回归头 - 不同阶段不同学习率
        for i, head in enumerate(self.pose_heads):
            stage_lr = lr_base * (1.0 + 0.5 * i / len(self.pose_heads))  # 深层略高
            param_groups.append({
                'params': head.parameters(),
                'lr': stage_lr,
                'name': f'pose_head_{i}'
            })
        
        return param_groups


class ICPoseNetV2Lite(nn.Module):
    """
    ICPoseNet V2 Lite - 轻量版，不使用重叠检测
    
    适用于初始位姿较准确的场景
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_queries: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_interval: int = 2,
    ):
        super().__init__()
        
        # 简化版，不使用重叠检测
        self.model = ICPoseNetV2(
            feature_dim=feature_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_interval=output_interval,
            use_overlap_detection=False,
            use_c2f=True,
        )
        
    def forward(self, img_feats, pcd_feats, img_pixels, pcd_points, intrinsics=None):
        return self.model(
            img_feats=img_feats,
            pcd_feats=pcd_feats,
            img_pixels=img_pixels,
            pcd_points=pcd_points,
            intrinsics=intrinsics,
            return_all_stages=True,
        )


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("=" * 60)
    print("Testing ICPoseNet V2")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ICPoseNetV2(
        feature_dim=256,
        num_queries=128,
        num_layers=12,
        num_heads=8,
        dropout=0.1,
        output_interval=2,
        use_overlap_detection=True,
        use_c2f=True,
    ).to(device)
    
    print(f"\n模型参数量: {count_parameters(model) / 1e6:.2f}M")
    
    # 测试输入
    batch_size = 4
    n_img = 2000
    n_pcd = 1000
    feature_dim = 256
    
    img_feats = torch.randn(batch_size, n_img, feature_dim).to(device)
    pcd_feats = torch.randn(batch_size, n_pcd, feature_dim).to(device)
    img_pixels = torch.rand(batch_size, n_img, 2).to(device) * 640
    pcd_points = torch.randn(batch_size, n_pcd, 3).to(device) * 10
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] = 600  # fx
    intrinsics[:, 1, 1] = 600  # fy
    intrinsics[:, 0, 2] = 320  # cx
    intrinsics[:, 1, 2] = 240  # cy
    img_global_feats = torch.randn(batch_size, feature_dim).to(device)
    
    # 前向传播
    print("\n前向传播测试...")
    with torch.no_grad():
        outputs = model(
            img_feats=img_feats,
            pcd_feats=pcd_feats,
            img_pixels=img_pixels,
            pcd_points=pcd_points,
            intrinsics=intrinsics,
            img_global_feats=img_global_feats,
            return_all_stages=True,
        )
    
    print(f"  Pose matrix: {outputs['pose_matrix'].shape}")
    print(f"  Rotation 6D: {outputs['rotation_6d'].shape}")
    print(f"  Translation: {outputs['translation'].shape}")
    print(f"  Keypoints 2D: {outputs['keypoints_2d'].shape}")
    print(f"  Keypoints 3D: {outputs['keypoints_3d'].shape}")
    
    if 'overlap_mask' in outputs:
        print(f"  Overlap mask: {outputs['overlap_mask'].shape}")
        print(f"    Range: [{outputs['overlap_mask'].min():.3f}, {outputs['overlap_mask'].max():.3f}]")
    
    if 'stage_outputs' in outputs:
        print(f"  Number of stages: {len(outputs['stage_outputs'])}")
        for i, stage in enumerate(outputs['stage_outputs']):
            print(f"    Stage {i}: pose_matrix {stage['pose_matrix'].shape}")
    
    # 测试梯度
    print("\n梯度测试...")
    outputs = model(
        img_feats=img_feats,
        pcd_feats=pcd_feats,
        img_pixels=img_pixels,
        pcd_points=pcd_points,
        intrinsics=intrinsics,
        img_global_feats=img_global_feats,
    )
    
    loss = outputs['translation'].sum() + outputs['rotation_6d'].sum()
    loss.backward()
    
    # 检查梯度
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name.split('.')[0]] = grad_norms.get(name.split('.')[0], 0) + param.grad.norm().item()
    
    print("  各模块梯度范数:")
    for name, norm in sorted(grad_norms.items()):
        print(f"    {name}: {norm:.4f}")
    
    print("\n✅ ICPoseNet V2 测试通过!")
    
    # 测试Lite版本
    print("\n" + "=" * 60)
    print("Testing ICPoseNet V2 Lite (without overlap detection)")
    print("=" * 60)
    
    model_lite = ICPoseNetV2Lite(
        feature_dim=256,
        num_queries=128,
        num_layers=8,
        num_heads=8,
    ).to(device)
    
    print(f"\nLite模型参数量: {count_parameters(model_lite) / 1e6:.2f}M")
    
    with torch.no_grad():
        outputs_lite = model_lite(
            img_feats=img_feats,
            pcd_feats=pcd_feats,
            img_pixels=img_pixels,
            pcd_points=pcd_points,
            intrinsics=intrinsics,
        )
    
    print(f"  Pose matrix: {outputs_lite['pose_matrix'].shape}")
    print(f"  Stages: {len(outputs_lite['stage_outputs'])}")
    
    print("\n✅ ICPoseNet V2 Lite 测试通过!")
