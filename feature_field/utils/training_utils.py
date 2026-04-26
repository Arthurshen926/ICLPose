"""
训练工具模块

封装通用的训练逻辑和辅助函数
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import math


def compute_relative_pose(pose_target: torch.Tensor, pose_init: torch.Tensor) -> torch.Tensor:
    """
    计算相对位姿: pose_rel = inv(pose_init) @ pose_target
    
    Args:
        pose_target: (B, 4, 4) 目标位姿（绝对）
        pose_init: (B, 4, 4) 初始位姿（绝对）
        
    Returns:
        pose_rel: (B, 4, 4) 相对位姿
    """
    R_init = pose_init[:, :3, :3]
    t_init = pose_init[:, :3, 3]
    
    R_target = pose_target[:, :3, :3]
    t_target = pose_target[:, :3, 3]
    
    # 相对旋转: R_rel = R_init^T @ R_target
    R_rel = torch.bmm(R_init.transpose(1, 2), R_target)
    
    # 相对平移: t_rel = R_init^T @ (t_target - t_init)
    t_rel = torch.bmm(R_init.transpose(1, 2), (t_target - t_init).unsqueeze(-1)).squeeze(-1)
    
    # 组合成4x4变换矩阵
    batch_size = pose_target.shape[0]
    pose_rel = torch.eye(4, device=pose_target.device).unsqueeze(0).expand(batch_size, 4, 4).clone()
    pose_rel[:, :3, :3] = R_rel
    pose_rel[:, :3, 3] = t_rel
    
    return pose_rel


def compose_pose(pose_rel: torch.Tensor, pose_init: torch.Tensor) -> torch.Tensor:
    """
    组合相对位姿和初始位姿: pose_target = pose_init @ pose_rel
    
    Args:
        pose_rel: (B, 4, 4) 相对位姿
        pose_init: (B, 4, 4) 初始位姿
        
    Returns:
        pose_target: (B, 4, 4) 目标位姿
    """
    return torch.bmm(pose_init, pose_rel)


def compute_pose_error(pose_pred: torch.Tensor, pose_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算位姿误差（旋转误差度数，平移误差米）
    
    Args:
        pose_pred: (B, 4, 4) 预测位姿
        pose_gt: (B, 4, 4) GT位姿
        
    Returns:
        rotation_error: (B,) 旋转误差（度）
        translation_error: (B,) 平移误差（米）
    """
    R_pred = pose_pred[:, :3, :3]
    t_pred = pose_pred[:, :3, 3]
    R_gt = pose_gt[:, :3, :3]
    t_gt = pose_gt[:, :3, 3]
    
    # 旋转误差（测地距离）
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    trace = torch.clamp(trace, -1.0, 3.0)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    rotation_error = torch.acos(cos_angle) * 180.0 / math.pi
    
    # 平移误差（欧氏距离）
    translation_error = torch.norm(t_pred - t_gt, dim=1)
    
    return rotation_error, translation_error


def prepare_model_inputs_v1(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    use_relative_pose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    准备 V1 模型的输入
    
    Args:
        batch: 数据批次
        model: 模型
        use_relative_pose: 是否使用相对位姿
        
    Returns:
        model_inputs: 模型输入字典
    """
    model_inputs = {
        'img_feats': batch['img_feats'],
        'pcd_feats': batch['pcd_feats'],
        'img_pixels': batch['img_pixels'],
        'pcd_points': batch['pcd_points'],
    }
    
    # 位置编码
    if 'img_pos_embeds' in batch:
        model_inputs['img_pos_embeds'] = batch['img_pos_embeds']
    if 'pcd_pos_embeds' in batch:
        model_inputs['pcd_pos_embeds'] = batch['pcd_pos_embeds']
    
    return model_inputs


def prepare_model_inputs_v2(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    use_relative_pose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    准备 V2 模型的输入
    
    Args:
        batch: 数据批次
        model: 模型
        use_relative_pose: 是否使用相对位姿
        
    Returns:
        model_inputs: 模型输入字典
    """
    model_inputs = {
        'img_feats': batch['img_feats'],
        'pcd_feats': batch['pcd_feats'],
        'img_pixels': batch['img_pixels'],
        'pcd_points': batch['pcd_points'],
    }
    
    # 相机内参（V2需要用于位置编码和重叠检测）
    if 'intrinsics' in batch:
        model_inputs['intrinsics'] = batch['intrinsics']
    
    # 全局图像特征（用于重叠检测）
    if 'img_global_feats' in batch:
        model_inputs['img_global_feats'] = batch['img_global_feats']
    
    # 初始位姿（用于重叠检测参考）
    if 'initial_pose' in batch:
        model_inputs['initial_pose'] = batch['initial_pose']
    
    return model_inputs


def prepare_model_inputs(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    根据模型版本准备输入
    
    Args:
        batch: 数据批次
        model: 模型
        config: 配置
        
    Returns:
        model_inputs: 模型输入字典
    """
    from feature_field.utils.model_factory import get_model_info
    
    model_info = get_model_info(model)
    use_relative_pose = config.get('loss', {}).get('use_relative_pose', True)
    
    if model_info['is_v2']:
        return prepare_model_inputs_v2(batch, model, use_relative_pose)
    else:
        return prepare_model_inputs_v1(batch, model, use_relative_pose)


def extract_pose_from_outputs(
    outputs: Dict[str, torch.Tensor],
    model: nn.Module,
) -> torch.Tensor:
    """
    从模型输出中提取位姿矩阵
    
    Args:
        outputs: 模型输出
        model: 模型
        
    Returns:
        pose_matrix: (B, 4, 4) 位姿矩阵
    """
    if isinstance(outputs, dict):
        if 'pose_matrix' in outputs:
            return outputs['pose_matrix']
        elif 'stage_outputs' in outputs:
            # C2F: 返回最后一个阶段的位姿
            return outputs['stage_outputs'][-1]['pose_matrix']
    elif isinstance(outputs, tuple):
        # V1模型返回元组
        return outputs[0]  # pose_matrix
    
    raise ValueError(f"无法从输出中提取位姿: {type(outputs)}")


class GradientClipper:
    """梯度裁剪器"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        """
        裁剪梯度并返回梯度范数
        
        Returns:
            total_norm: 裁剪前的梯度范数
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_norm, norm_type=self.norm_type
        )
        
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


class EMATracker:
    """指数移动平均跟踪器"""
    
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.values = {}
    
    def update(self, key: str, value: float):
        """更新跟踪值"""
        if key not in self.values:
            self.values[key] = value
        else:
            self.values[key] = self.alpha * self.values[key] + (1 - self.alpha) * value
    
    def get(self, key: str, default: float = 0.0) -> float:
        """获取跟踪值"""
        return self.values.get(key, default)
    
    def reset(self):
        """重置所有跟踪值"""
        self.values = {}


class MetricLogger:
    """指标记录器"""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, key: str, value: float, n: int = 1):
        """更新指标"""
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0
        self.metrics[key] += value * n
        self.counts[key] += n
    
    def get_average(self, key: str) -> float:
        """获取平均值"""
        if key not in self.metrics or self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]
    
    def get_all_averages(self) -> Dict[str, float]:
        """获取所有平均值"""
        return {k: self.get_average(k) for k in self.metrics}
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
        self.counts = {}


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    additional_params: Optional[List[nn.Parameter]] = None,
) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        config: 训练配置
        additional_params: 额外的参数（如Kendall loss的参数）
        
    Returns:
        optimizer: 优化器
    """
    training_cfg = config.get('training', {})
    
    # 收集参数
    params = list(model.parameters())
    if additional_params:
        params.extend(additional_params)
    
    optimizer_type = training_cfg.get('optimizer', 'adamw').lower()
    lr = training_cfg.get('learning_rate', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 1e-4)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay,
            momentum=training_cfg.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")
    
    print(f"  🔧 创建优化器: {optimizer_type.upper()}")
    print(f"     - 学习率: {lr}")
    print(f"     - 权重衰减: {weight_decay}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 训练配置
        
    Returns:
        scheduler: 学习率调度器（或None）
    """
    training_cfg = config.get('training', {})
    scheduler_type = training_cfg.get('scheduler', 'cosine').lower()
    num_epochs = training_cfg.get('num_epochs', 100)
    
    if scheduler_type == 'none':
        return None
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=training_cfg.get('min_lr', 1e-6),
        )
        print(f"  📅 创建调度器: CosineAnnealing")
        print(f"     - T_max: {num_epochs}")
        print(f"     - 最小学习率: {training_cfg.get('min_lr', 1e-6)}")
        
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_cfg.get('lr_decay_step', 10),
            gamma=training_cfg.get('lr_decay_gamma', 0.5),
        )
        print(f"  📅 创建调度器: StepLR")
        
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_cfg.get('lr_decay_gamma', 0.95),
        )
        print(f"  📅 创建调度器: ExponentialLR")
        
    else:
        print(f"  ⚠️ 未知调度器类型: {scheduler_type}，不使用调度器")
        return None
    
    return scheduler
