"""
损失函数工厂模块

根据配置创建不同版本的损失函数，提供统一的接口
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


def create_loss_function(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    根据配置创建损失函数
    
    Args:
        config: 损失函数配置字典，包含:
            - version: 损失版本 ('standard', 'c2f')
            - use_kendall: 是否使用Kendall自动权重
            - ... 其他损失特定参数
        device: 设备
        
    Returns:
        criterion: 创建的损失函数实例
    """
    loss_cfg = config.get('loss', config)  # 兼容直接传入loss配置
    version = loss_cfg.get('version', 'standard')
    
    if version == 'c2f':
        return _create_c2f_loss(loss_cfg, config, device)
    else:
        return _create_standard_loss(loss_cfg, device)


def _create_standard_loss(loss_cfg: Dict, device: torch.device) -> nn.Module:
    """创建标准损失函数"""
    from losses.pose_loss import PoseLoss, PoseLossKendall
    
    use_kendall = loss_cfg.get('use_kendall', True)
    
    if use_kendall:
        criterion = PoseLossKendall(
            init_log_var_r=loss_cfg.get('init_log_var_rotation', 0.0),
            init_log_var_t=loss_cfg.get('init_log_var_translation', 0.0),
            rotation_loss_type=loss_cfg.get('rotation_loss', 'geodesic'),
            translation_loss_type=loss_cfg.get('translation_loss', 'l2'),
        )
        print(f"  📉 创建损失函数: PoseLossKendall (自动权重)")
    else:
        criterion = PoseLoss(
            rotation_weight=loss_cfg.get('rotation_weight', 1.0),
            translation_weight=loss_cfg.get('translation_weight', 1.0),
            rotation_loss_type=loss_cfg.get('rotation_loss', 'geodesic'),
            translation_loss_type=loss_cfg.get('translation_loss', 'l2'),
        )
        print(f"  📉 创建损失函数: PoseLoss (固定权重)")
    
    return criterion.to(device)


def _create_c2f_loss(loss_cfg: Dict, full_config: Dict, device: torch.device) -> nn.Module:
    """创建C2F损失函数"""
    from losses.pose_loss_c2f import PoseLossC2F
    
    # 获取模型配置来确定阶段数
    model_cfg = full_config.get('model', {})
    num_layers = model_cfg.get('num_layers', 12)
    output_interval = model_cfg.get('output_interval', 2)
    num_stages = num_layers // output_interval
    
    # 获取阶段权重（如果未配置，自动生成）
    stage_weights = loss_cfg.get('stage_weights', None)
    if stage_weights is None:
        # 自动生成: 浅层权重低，深层权重高
        stage_weights = [0.2 + 0.8 * i / (num_stages - 1) for i in range(num_stages)]
    
    # 获取warmup参数
    warmup_epochs = 0
    if loss_cfg.get('use_rotation_warmup', True):
        warmup_epochs = loss_cfg.get('rotation_warmup_epochs', 50)
    
    criterion = PoseLossC2F(
        num_stages=num_stages,
        stage_weights=stage_weights,
        rotation_weight=loss_cfg.get('rotation_weight', 1.0),
        translation_weight=loss_cfg.get('translation_weight', 1.0),
        rotation_loss_type=loss_cfg.get('rotation_loss', 'geodesic'),
        translation_loss_type=loss_cfg.get('translation_loss', 'l2'),
        warmup_epochs=warmup_epochs,
        warmup_rotation_loss=loss_cfg.get('warmup_type', 'l1'),
        use_dyntanh=loss_cfg.get('use_dyntanh', True),
        soft_clamp=loss_cfg.get('dyntanh_start', 100.0),
        soft_clamp_min=loss_cfg.get('dyntanh_end', 10.0),
    )
    
    print(f"  📉 创建损失函数: PoseLossC2F (多阶段)")
    print(f"     - 阶段数: {num_stages}")
    print(f"     - 阶段权重: {stage_weights}")
    print(f"     - 旋转Warmup: {warmup_epochs} epochs")
    print(f"     - Dyntanh: {'启用' if loss_cfg.get('use_dyntanh', True) else '禁用'}")
    
    return criterion.to(device)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    整合位姿损失、重叠检测损失、Diversity损失、重投影损失等
    """
    
    def __init__(
        self,
        pose_loss: nn.Module,
        use_overlap_loss: bool = False,
        use_diversity_loss: bool = True,
        use_reprojection_loss: bool = True,
        overlap_weight: float = 0.5,
        diversity_weight: float = 0.05,
        reprojection_weight: float = 0.1,
        diversity_margin_2d: float = 16.0,
        diversity_margin_3d: float = 0.16,
    ):
        super().__init__()
        
        self.pose_loss = pose_loss
        self.use_overlap_loss = use_overlap_loss
        self.use_diversity_loss = use_diversity_loss
        self.use_reprojection_loss = use_reprojection_loss
        
        self.overlap_weight = overlap_weight
        self.diversity_weight = diversity_weight
        self.reprojection_weight = reprojection_weight
        
        # 初始化辅助损失
        if use_overlap_loss:
            from modules.overlap_detection import OverlapLoss
            self.overlap_loss = OverlapLoss()
        
        if use_diversity_loss:
            from modules.diversity_loss import DiversityLoss
            self.diversity_loss = DiversityLoss(
                margin_2d=diversity_margin_2d,
                margin_3d=diversity_margin_3d,
            )
        
        if use_reprojection_loss:
            from losses.reprojection_loss import ReprojectionLoss
            self.reprojection_loss = ReprojectionLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_pose: torch.Tensor,
        pcd_points: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            outputs: 模型输出，包含:
                - pose_matrix 或 stage_outputs
                - overlap_mask (可选)
                - keypoints_2d, keypoints_3d (可选)
            gt_pose: GT位姿
            pcd_points: 点云坐标（用于重叠损失）
            intrinsics: 相机内参（用于重叠和重投影损失）
            epoch: 当前epoch
            
        Returns:
            losses: 损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 位姿损失
        if 'stage_outputs' in outputs:
            # C2F模式：使用多阶段输出
            pose_stages = [s['pose_matrix'] for s in outputs['stage_outputs']]
            
            # 设置epoch（用于warmup）
            if hasattr(self.pose_loss, 'set_epoch'):
                self.pose_loss.set_epoch(epoch)
            
            pose_loss_dict = self.pose_loss(pose_stages, gt_pose, return_components=True)
        else:
            # 标准模式：单阶段输出
            pose_pred = outputs.get('pose_matrix')
            if pose_pred is not None:
                pose_loss_dict = self.pose_loss(pose_pred, gt_pose)
            else:
                pose_loss_dict = {'loss': torch.tensor(0.0)}
        
        losses.update(pose_loss_dict)
        total_loss = total_loss + pose_loss_dict['loss']
        
        # 2. 重叠检测损失
        if self.use_overlap_loss and 'overlap_mask' in outputs:
            overlap_mask = outputs['overlap_mask']
            if pcd_points is not None and intrinsics is not None:
                overlap_loss_val = self.overlap_loss(
                    overlap_mask, pcd_points, gt_pose, intrinsics
                )
                losses['overlap_loss'] = overlap_loss_val
                total_loss = total_loss + self.overlap_weight * overlap_loss_val
        
        # 3. Diversity损失
        if self.use_diversity_loss:
            keypoints_2d = outputs.get('keypoints_2d')
            keypoints_3d = outputs.get('keypoints_3d')
            if keypoints_2d is not None and keypoints_3d is not None:
                div_loss = self.diversity_loss(keypoints_2d, keypoints_3d)
                losses['diversity_loss'] = div_loss
                total_loss = total_loss + self.diversity_weight * div_loss
        
        # 4. 重投影损失
        if self.use_reprojection_loss:
            keypoints_2d = outputs.get('keypoints_2d')
            keypoints_3d = outputs.get('keypoints_3d')
            if keypoints_2d is not None and keypoints_3d is not None and intrinsics is not None:
                reproj_result = self.reprojection_loss(
                    keypoints_2d, keypoints_3d, gt_pose, intrinsics
                )
                # ReprojectionLoss 返回 (weighted_loss, info) 元组
                if isinstance(reproj_result, tuple):
                    reproj_loss, reproj_info = reproj_result
                    # 注意：ReprojectionLoss 内部已经应用了 weight，这里不再乘 weight
                    losses['reprojection_loss'] = reproj_loss
                    total_loss = total_loss + reproj_loss
                else:
                    losses['reprojection_loss'] = reproj_result
                    total_loss = total_loss + self.reprojection_weight * reproj_result
        
        losses['total_loss'] = total_loss
        
        return losses


def create_combined_loss(config: Dict[str, Any], device: torch.device) -> CombinedLoss:
    """
    创建组合损失函数
    
    Args:
        config: 完整配置
        device: 设备
        
    Returns:
        combined_loss: 组合损失函数
    """
    loss_cfg = config.get('loss', {})
    model_cfg = config.get('model', {})
    
    # 创建基础位姿损失
    pose_loss = create_loss_function(config, device)
    
    # 判断是否需要重叠损失
    use_overlap_loss = model_cfg.get('use_overlap_detection', False)
    
    combined = CombinedLoss(
        pose_loss=pose_loss,
        use_overlap_loss=use_overlap_loss,
        use_diversity_loss=loss_cfg.get('diversity_weight', 0.0) > 0,
        use_reprojection_loss=loss_cfg.get('reprojection_weight', 0.0) > 0,
        overlap_weight=loss_cfg.get('overlap_loss_weight', 0.5),
        diversity_weight=loss_cfg.get('diversity_weight', 0.05),
        reprojection_weight=loss_cfg.get('reprojection_weight', 0.1),
        diversity_margin_2d=loss_cfg.get('diversity_margin_2d', 16.0),
        diversity_margin_3d=loss_cfg.get('diversity_margin_3d', 0.16),
    )
    
    return combined.to(device)


def get_loss_info(criterion: nn.Module) -> Dict[str, Any]:
    """获取损失函数信息"""
    if isinstance(criterion, CombinedLoss):
        return {
            'type': 'CombinedLoss',
            'pose_loss_type': criterion.pose_loss.__class__.__name__,
            'use_overlap_loss': criterion.use_overlap_loss,
            'use_diversity_loss': criterion.use_diversity_loss,
            'use_reprojection_loss': criterion.use_reprojection_loss,
        }
    else:
        return {
            'type': criterion.__class__.__name__,
        }
