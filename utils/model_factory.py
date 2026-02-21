"""
模型工厂模块

根据配置创建不同版本的模型，提供统一的接口
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置字典，包含:
            - version: 模型版本 ('v1', 'v2', 'v2_lite')
            - model_type: v1模型类型 ('standard', 'iterative', 'iterative_lite')
            - feature_dim: 特征维度
            - num_queries: Query数量
            - ... 其他模型特定参数
        device: 设备
        
    Returns:
        model: 创建的模型实例
    """
    model_cfg = config.get('model', config)  # 兼容直接传入model配置
    version = model_cfg.get('version', 'v1')
    
    if version == 'v2':
        return _create_model_v2(model_cfg, device)
    elif version == 'v2_lite':
        return _create_model_v2_lite(model_cfg, device)
    else:
        return _create_model_v1(model_cfg, device)


def _create_model_v1(model_cfg: Dict, device: torch.device) -> nn.Module:
    """创建 V1 版本模型（原版）"""
    from ic_models.ic_pose_net import ICPoseNet
    from ic_models.ic_pose_net_iterative import ICPoseNetIterative, ICPoseNetIterativeLite
    
    model_type = model_cfg.get('model_type', 'standard')
    
    if model_type == 'iterative':
        model = ICPoseNetIterative(
            feature_dim=model_cfg.get('feature_dim', 256),
            num_queries=model_cfg.get('num_queries', 128),
            num_stages=model_cfg.get('num_stages', 3),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            num_heads=model_cfg.get('num_heads', 8),
            dropout=model_cfg.get('dropout', 0.1),
        )
        print(f"  📦 创建模型: ICPoseNetIterative (V1迭代版)")
        print(f"     - 精化阶段数: {model_cfg.get('num_stages', 3)}")
        
    elif model_type == 'iterative_lite':
        model = ICPoseNetIterativeLite(
            feature_dim=model_cfg.get('feature_dim', 256),
            num_queries=model_cfg.get('num_queries', 128),
            num_stages=model_cfg.get('num_stages', 3),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            num_heads=model_cfg.get('num_heads', 8),
            dropout=model_cfg.get('dropout', 0.1),
        )
        print(f"  📦 创建模型: ICPoseNetIterativeLite (V1轻量迭代版)")
        print(f"     - 精化阶段数: {model_cfg.get('num_stages', 3)} (共享参数)")
        
    else:  # standard
        model = ICPoseNet(
            feature_dim=model_cfg.get('feature_dim', 256),
            num_queries=model_cfg.get('num_queries', 128),
            fusion_layers=model_cfg.get('fusion_layers', 6),
            num_heads=model_cfg.get('num_heads', 8),
            dropout=model_cfg.get('dropout', 0.1),
            attention_temperature=model_cfg.get('attention_temperature', None),
        )
        print(f"  📦 创建模型: ICPoseNet (V1标准版)")
        print(f"     - 融合层数: {model_cfg.get('fusion_layers', 6)}")
        print(f"     - attention_temperature: {model_cfg.get('attention_temperature', 'default(√dim)')}")
    
    return model.to(device)


def _create_model_v2(model_cfg: Dict, device: torch.device) -> nn.Module:
    """创建 V2 版本模型（C2F + 重叠检测）"""
    from ic_models.ic_pose_net_v2 import ICPoseNetV2
    
    model = ICPoseNetV2(
        feature_dim=model_cfg.get('feature_dim', 256),
        num_queries=model_cfg.get('num_queries', 128),
        num_layers=model_cfg.get('num_layers', 12),
        num_heads=model_cfg.get('num_heads', 8),
        dropout=model_cfg.get('dropout', 0.1),
        output_interval=model_cfg.get('output_interval', 2),
        use_overlap_detection=model_cfg.get('use_overlap_detection', True),
        use_c2f=model_cfg.get('use_c2f', True),
        nerf_frequencies=model_cfg.get('nerf_frequencies', 10),
    )
    
    print(f"  📦 创建模型: ICPoseNetV2 (C2F + 重叠检测)")
    print(f"     - Transformer层数: {model_cfg.get('num_layers', 12)}")
    print(f"     - C2F阶段间隔: {model_cfg.get('output_interval', 2)}")
    print(f"     - 重叠检测: {'启用' if model_cfg.get('use_overlap_detection', True) else '禁用'}")
    print(f"     - C2F多阶段: {'启用' if model_cfg.get('use_c2f', True) else '禁用'}")
    
    return model.to(device)


def _create_model_v2_lite(model_cfg: Dict, device: torch.device) -> nn.Module:
    """创建 V2 Lite 版本模型（不含重叠检测）"""
    from ic_models.ic_pose_net_v2 import ICPoseNetV2Lite
    
    model = ICPoseNetV2Lite(
        feature_dim=model_cfg.get('feature_dim', 256),
        num_queries=model_cfg.get('num_queries', 128),
        num_layers=model_cfg.get('num_layers', 8),
        num_heads=model_cfg.get('num_heads', 8),
        dropout=model_cfg.get('dropout', 0.1),
        output_interval=model_cfg.get('output_interval', 2),
    )
    
    print(f"  📦 创建模型: ICPoseNetV2Lite (轻量版)")
    print(f"     - Transformer层数: {model_cfg.get('num_layers', 8)}")
    
    return model.to(device)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型信息
    
    Returns:
        dict: {
            'total_params': 总参数量,
            'trainable_params': 可训练参数量,
            'model_type': 模型类型名称,
            'is_v2': 是否为V2版本,
            'has_overlap_detection': 是否有重叠检测,
            'has_c2f': 是否有C2F,
        }
    """
    # 处理 DDP 包装
    model_unwrapped = model.module if hasattr(model, 'module') else model
    
    total_params = sum(p.numel() for p in model_unwrapped.parameters())
    trainable_params = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
    
    model_type = model_unwrapped.__class__.__name__
    
    # 检测模型特性
    is_v2 = 'V2' in model_type
    has_overlap_detection = hasattr(model_unwrapped, 'overlap_module') and model_unwrapped.use_overlap_detection
    has_c2f = hasattr(model_unwrapped, 'use_c2f') and model_unwrapped.use_c2f
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_type': model_type,
        'is_v2': is_v2,
        'has_overlap_detection': has_overlap_detection,
        'has_c2f': has_c2f,
    }


def print_model_summary(model: nn.Module):
    """打印模型摘要"""
    info = get_model_info(model)
    
    print(f"  📊 模型摘要:")
    print(f"     - 类型: {info['model_type']}")
    print(f"     - 总参数量: {info['total_params'] / 1e6:.2f}M")
    print(f"     - 可训练参数量: {info['trainable_params'] / 1e6:.2f}M")
    if info['is_v2']:
        print(f"     - 重叠检测: {'✅' if info['has_overlap_detection'] else '❌'}")
        print(f"     - C2F多阶段: {'✅' if info['has_c2f'] else '❌'}")
