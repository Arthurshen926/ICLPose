"""
隐式对应关系位姿估计网络训练脚本
完整的训练流程，包含数据加载、模型训练、验证、checkpoint保存等
支持多GPU分布式训练
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# SplatLoc相关导入（已集成到本仓库）
from splatloc_modules.gaussian_splatting.scene.gaussian_model import GaussianModel
from splatloc_modules.models.decoders import FeatureDecoder
from splatloc_modules.gaussian_splatting.gaussian_renderer import render

# Feature3DGS相关导入
from feature_3dgs.feature_3dgs_provider import Feature3DGSProvider

# 隐式对应关系模块导入
from data.dataset import CorrespondenceDataset, collate_fn
from losses.pose_loss import PoseLoss, PoseLossKendall
from ic_models.ic_pose_net import ICPoseNet
from ic_models.ic_pose_net_iterative import ICPoseNetIterative, ICPoseNetIterativeLite
from utils.visualization import (
    visualize_attention_maps,
    visualize_2d3d_correspondence,
    visualize_pose_prediction,
    visualize_feature_similarity_matrix
)


def compute_relative_pose(pose_target, pose_init):
    """
    计算相对位姿: pose_rel = inv(pose_init) @ pose_target
    
    即：从init坐标系到target坐标系的变换
    满足：pose_target = pose_init @ pose_rel
    
    Args:
        pose_target: (B, 4, 4) 目标位姿（绝对）- camera-to-world变换
        pose_init: (B, 4, 4) 初始位姿（绝对）- camera-to-world变换
        
    Returns:
        pose_rel: (B, 4, 4) 相对位姿（从init相机坐标系到target相机坐标系的变换）
    
    数学推导:
        pose_rel = inv(pose_init) @ pose_target
        
        其中 inv(T) = [R^T, -R^T @ t; 0, 1]
        
        所以:
        R_rel = R_init^T @ R_target
        t_rel = R_init^T @ (t_target - t_init)
    """
    R_init = pose_init[:, :3, :3]  # (B, 3, 3)
    t_init = pose_init[:, :3, 3]   # (B, 3)
    
    R_target = pose_target[:, :3, :3]  # (B, 3, 3)
    t_target = pose_target[:, :3, 3]   # (B, 3)
    
    # 相对旋转: R_rel = R_init^T @ R_target
    # 注意：这是 inv(pose_init) @ pose_target 的正确公式
    R_rel = torch.bmm(R_init.transpose(1, 2), R_target)
    
    # 相对平移: t_rel = R_init^T @ (t_target - t_init)
    t_rel = torch.bmm(R_init.transpose(1, 2), (t_target - t_init).unsqueeze(-1)).squeeze(-1)
    
    # 组合成4x4变换矩阵
    batch_size = pose_target.shape[0]
    pose_rel = torch.eye(4, device=pose_target.device).unsqueeze(0).expand(batch_size, 4, 4).clone()
    pose_rel[:, :3, :3] = R_rel
    pose_rel[:, :3, 3] = t_rel
    
    return pose_rel


def compose_pose(pose_rel, pose_init):
    """
    组合相对位姿和初始位姿: pose_target = pose_init @ pose_rel
    
    Args:
        pose_rel: (B, 4, 4) 相对位姿
        pose_init: (B, 4, 4) 初始位姿（绝对）
        
    Returns:
        pose_target: (B, 4, 4) 目标位姿（绝对）
    """
    return torch.bmm(pose_init, pose_rel)


class ICPoseTrainer:
    """隐式对应关系位姿估计训练器（支持分布式训练）"""
    
    def __init__(self, config: Dict, local_rank: int = -1, resume_path: str = None):
        """
        初始化训练器
        
        参数:
            config: 配置字典
            local_rank: 分布式训练的本地rank（-1表示单卡）
            resume_path: checkpoint路径（用于恢复训练）
        """
        self.config = config
        self.local_rank = local_rank
        self.resume_path = resume_path
        self.is_distributed = local_rank >= 0
        
        # 设置设备
        if self.is_distributed:
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            self.is_main_process = (local_rank == 0)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main_process = True
        
        # 创建输出目录（仅主进程）
        if self.is_main_process:
            self.output_dir = Path(config['output_dir'])
            self.checkpoint_dir = self.output_dir / 'checkpoints'
            self.log_dir = self.output_dir / 'logs'
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 可视化目录（如果启用）
            if config.get('visualization', {}).get('enable', False):
                self.vis_dir = self.output_dir / 'visualizations'
                self.vis_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.vis_dir = None
            
            # TensorBoard日志
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            
            # 创建训练日志文件
            self.log_file = self.output_dir / 'training.log'
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"="*80 + "\n\n")
        else:
            self.output_dir = Path(config['output_dir'])
            self.checkpoint_dir = None
            self.log_dir = None
            self.writer = None
            self.log_file = None
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        if self.is_main_process:
            print("=" * 80)
            print("隐式对应关系位姿估计网络 - 训练脚本")
            if self.is_distributed:
                print(f"分布式训练: {dist.get_world_size()} GPUs")
            print("=" * 80)
            print(f"设备: {self.device}")
            print(f"输出目录: {self.output_dir}")
        
        # 确定输入类型 ('splatloc' 或 'feature3dgs')
        self.input_type = config.get('input_type', 'splatloc')
        self.feature3dgs_provider: Optional[Feature3DGSProvider] = None

        # 1. 加载场景模型
        # feature3dgs 模式：只需 Gaussians 几何（视锥裁切），跳过 FeatureDecoder
        # splatloc    模式：完整加载 Gaussians + FeatureDecoder
        skip_decoder = (self.input_type == 'feature3dgs')
        self._load_splatloc_models(skip_decoder=skip_decoder)
        if self.input_type == 'feature3dgs':
            self._load_feature3dgs_model()

        if self.is_main_process:
            print(f"  输入类型: {self.input_type}")

        # 2. 初始化ICPoseNet
        self._init_icposenet()
        
        # 3. 准备数据集
        self._prepare_datasets()
        
        # 4. 初始化损失函数（需要在优化器之前，因为Kendall's Loss有可学习参数）
        self._init_loss_functions()
        
        # 5. 初始化优化器
        self._init_optimizer()
        
        # 6. 从checkpoint恢复（如果需要）
        if self.resume_path:
            self._load_checkpoint(self.resume_path)
    
    def _load_feature3dgs_model(self):
        """加载预训练的 GaussianFeatureModel（特征3DGS），用于 feature3dgs 输入模式"""
        if self.is_main_process:
            print("\n[步骤 1b] 加载 Feature3DGS 模型...")

        f3dgs_cfg = self.config.get('feature3dgs', {})
        feature_ply_path = f3dgs_cfg.get('feature_ply_path')
        feature_dim      = f3dgs_cfg.get('feature_dim', self.config['model']['feature_dim'])
        depth_min        = f3dgs_cfg.get('depth_min', 0.01)
        depth_max        = f3dgs_cfg.get('depth_max', 20.0)

        if feature_ply_path is None:
            raise ValueError(
                "[Feature3DGS] 配置中缺少 feature3dgs.feature_ply_path，"
                "请先运行 feature_3dgs/train_feature_embedding.py 训练特征嵌入"
            )

        self.feature3dgs_provider = Feature3DGSProvider(
            feature_ply_path=feature_ply_path,
            feature_dim=feature_dim,
            device=str(self.device),
            depth_min=depth_min,
            depth_max=depth_max,
        )

        if self.is_main_process:
            print(f"  ✓ Feature3DGS 加载完成: {feature_ply_path}")
            print(f"  ✓ Gaussian 数量: {self.feature3dgs_provider.num_gaussians}")

    def _load_splatloc_models(self, skip_decoder: bool = False):
        """
        加载 SplatLoc 预训练模型。

        Args:
            skip_decoder: True → 只加载 Gaussian 几何，跳过 FeatureDecoder
                          （feature3dgs 模式下 FeatureDecoder 完全用不到）
        """
        print("\n[步骤 1] 加载SplatLoc预训练模型...")
        if skip_decoder:
            print("  (feature3dgs 模式：跳过 FeatureDecoder 加载)")

        splatloc_cfg = self.config['splatloc']
        gaussians_path = splatloc_cfg['gaussians_path']
        decoder_path = splatloc_cfg['decoder_path']
        splatloc_config_path = splatloc_cfg.get('config_path', None)
        
        # 加载SplatLoc配置
        if splatloc_config_path and os.path.exists(splatloc_config_path):
            print(f"  - 加载SplatLoc配置: {splatloc_config_path}")
            import yaml
            with open(splatloc_config_path, 'r') as f:
                splatloc_config = yaml.safe_load(f)
            # 确保有Training配置
            if 'Training' not in splatloc_config:
                splatloc_config['Training'] = {'primitive_reg': False}
        else:
            print(f"  - 使用默认配置")
            # 使用train_config.yaml中的配置
            splatloc_config = {
                'Training': {'primitive_reg': False},
                'scene': self.config.get('scene', {
                    'bound': [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                    'voxel_sdf': 0.05
                }),
                'decoder': self.config.get('decoder', {
                    'enc': 'HashGrid',
                    'num_layers': 4,
                    'hidden_dim': 128,
                    'final_dim': 256
                })
            }
        
        # 加载Gaussian模型
        print(f"  - 加载Gaussian模型: {gaussians_path}")
        self.gaussians = GaussianModel(sh_degree=0, config=splatloc_config)
        self.gaussians.load_ply(gaussians_path)
        
        # GaussianModel使用tensor属性，不是nn.Module
        # 将其移到CUDA并设置为eval模式（如果有的话）
        print(f"    ✓ Gaussian点数: {self.gaussians.get_xyz.shape[0]}")

        # feature3dgs 模式下不需要 FeatureDecoder，直接跳过
        if skip_decoder:
            self.feat_decoder = None
            # 仍需初始化渲染参数（视锥裁切时用）
            from munch import munchify
            self.pipeline_params = munchify({
                'convert_SHs_python': False,
                'compute_cov3D_python': False,
                'debug': False
            })
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
            if self.is_main_process:
                print("  ✓ Gaussian 加载完成（FeatureDecoder 已跳过）")
            return

        # 加载特征解码器
        print(f"  - 加载特征解码器: {decoder_path}")

        # 确保splatloc_config有必需的scene和decoder配置
        if 'scene' not in splatloc_config:
            splatloc_config['scene'] = self.config.get('scene', {
                'bound': [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                'voxel_sdf': 0.05
            })
        if 'decoder' not in splatloc_config:
            splatloc_config['decoder'] = self.config.get('decoder', {
                'enc': 'HashGrid',
                'num_layers': 4,
                'hidden_dim': 128,
                'final_dim': 256
            })
        
        self.feat_decoder = FeatureDecoder(config=splatloc_config, input_ch=3).to(self.device)
        
        # 手动将bounding_box移到GPU
        self.feat_decoder.bounding_box = self.feat_decoder.bounding_box.to(self.device)
        
        # 加载解码器权重
        checkpoint = torch.load(decoder_path, map_location=self.device)
        if 'decoder_state_dict' in checkpoint:
            ckpt_state = checkpoint['decoder_state_dict']
        else:
            ckpt_state = checkpoint
        
        # 手动加载参数，跳过大小不匹配的
        model_state = self.feat_decoder.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in ckpt_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        self.feat_decoder.load_state_dict(model_state)
        
        # 打印加载信息
        if self.is_main_process:
            print(f"    ✓ 加载了 {len(loaded_keys)} 个参数")
            if skipped_keys:
                print(f"    ⚠ 跳过了 {len(skipped_keys)} 个参数:")
                for key in skipped_keys[:3]:  # 只显示前3个
                    print(f"       - {key}")
                if len(skipped_keys) > 3:
                    print(f"       ... 和 {len(skipped_keys) - 3} 个其他参数")
            
            # 检查HashGrid配置是否匹配
            current_bound = self.config['scene']['bound']
            current_voxel = self.config['scene']['voxel_sdf']
            decoder_bound = self.feat_decoder.bounding_box.cpu().numpy().tolist()
            decoder_resolution = self.feat_decoder.resolution_sdf
            expected_resolution = int(max([b[1]-b[0] for b in current_bound]) / current_voxel)
            
            if abs(decoder_resolution - expected_resolution) > 1:
                print(f"    ⚠ 警告：HashGrid分辨率不匹配！")
                print(f"       当前配置: resolution={expected_resolution} (bound={current_bound}, voxel={current_voxel})")
                print(f"       解码器训练时: resolution={decoder_resolution}")
                print(f"       建议：检查并更新配置文件中的 scene.bound 和 scene.voxel_sdf")
        
        # 根据配置决定是否冻结解码器
        if splatloc_cfg.get('freeze_decoder', True):
            for param in self.feat_decoder.parameters():
                param.requires_grad = False
            self.feat_decoder.eval()
            if self.is_main_process:
                print("    ✓ 特征解码器已冻结")
        else:
            if self.is_main_process:
                print("    ✓ 特征解码器可训练")
        
        # 初始化渲染参数
        from munch import munchify
        self.pipeline_params = munchify({
            'convert_SHs_python': False,
            'compute_cov3D_python': False,
            'debug': False
        })
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        if self.is_main_process:
            print("  ✓ SplatLoc模型加载完成")
    
    def _init_icposenet(self):
        """初始化ICPoseNet"""
        if self.is_main_process:
            print("\n[步骤 2] 初始化ICPoseNet...")
        
        model_cfg = self.config['model']
        
        # 选择模型架构
        model_type = model_cfg.get('model_type', 'standard')  # standard, iterative, iterative_lite
        
        if model_type == 'iterative':
            # 迭代精化版本（每阶段独立参数）
            self.model = ICPoseNetIterative(
                feature_dim=model_cfg['feature_dim'],
                num_queries=model_cfg['num_queries'],
                num_stages=model_cfg.get('num_stages', 3),
                hidden_dim=model_cfg.get('hidden_dim', 512),
                num_heads=model_cfg['num_heads'],
                dropout=model_cfg['dropout']
            ).to(self.device)
            self.use_iterative_model = True
            if self.is_main_process:
                print(f"  - 模型类型: ICPoseNetIterative (迭代精化)")
                print(f"  - 精化阶段数: {model_cfg.get('num_stages', 3)}")
        elif model_type == 'iterative_lite':
            # 轻量级迭代版本（共享参数）
            self.model = ICPoseNetIterativeLite(
                feature_dim=model_cfg['feature_dim'],
                num_queries=model_cfg['num_queries'],
                num_stages=model_cfg.get('num_stages', 3),
                hidden_dim=model_cfg.get('hidden_dim', 512),
                num_heads=model_cfg['num_heads'],
                dropout=model_cfg['dropout']
            ).to(self.device)
            self.use_iterative_model = True
            if self.is_main_process:
                print(f"  - 模型类型: ICPoseNetIterativeLite (轻量级迭代)")
                print(f"  - 精化阶段数: {model_cfg.get('num_stages', 3)} (共享参数)")
        else:
            # 标准版本（单阶段）
            self.model = ICPoseNet(
                feature_dim=model_cfg['feature_dim'],
                num_queries=model_cfg['num_queries'],
                fusion_layers=model_cfg['fusion_layers'],
                num_heads=model_cfg['num_heads'],
                dropout=model_cfg['dropout'],
                attention_temperature=model_cfg.get('attention_temperature', None),
            ).to(self.device)
            self.use_iterative_model = False
            if self.is_main_process:
                print(f"  - 模型类型: ICPoseNet (标准单阶段)")
        
        # 包装为DDP模型
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            if self.is_main_process:
                print(f"  - 使用DistributedDataParallel包装模型")
        
        # 统计参数量
        model_for_count = self.model.module if self.is_distributed else self.model
        total_params = sum(p.numel() for p in model_for_count.parameters())
        trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"  - 特征维度: {model_cfg['feature_dim']}")
            print(f"  - Query数量: {model_cfg['num_queries']}")
            if not self.use_iterative_model:
                print(f"  - 融合层数: {model_cfg['fusion_layers']}")
            print(f"  - 注意力头数: {model_cfg['num_heads']}")
            print(f"  - 总参数量: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print("  ✓ ICPoseNet初始化完成")
    
    def _prepare_datasets(self):
        """准备训练集和验证集"""
        print("\n[步骤 3] 准备数据集...")
        
        data_cfg = self.config['dataset']
        
        # 训练集
        use_relative_pose = self.config['loss'].get('use_relative_pose', False)
        self.train_dataset = CorrespondenceDataset(
            data_root=data_cfg['data_root'],
            scene_name=data_cfg['train_scene'],
            image_size=tuple(data_cfg['image_size']),
            augment=True,
            max_samples=data_cfg.get('max_train_samples'),
            use_depth=data_cfg.get('use_depth', False),
            gaussian_path=data_cfg.get('gaussian_path'),
            fx=data_cfg['fx'],
            fy=data_cfg['fy'],
            cx=data_cfg['cx'],
            cy=data_cfg['cy'],
            sample_step=data_cfg.get('train_step', 1),
            use_initial_pose=use_relative_pose,
            pose_noise_rot_deg=data_cfg.get('pose_noise_rot_deg', 5.0),
            pose_noise_trans_m=data_cfg.get('pose_noise_trans_m', 0.1),
            # 🆕 视锥裁切和负样本配置
            use_init_pose_for_culling=data_cfg.get('use_init_pose_for_culling', False),
            frustum_margin=data_cfg.get('frustum_margin', 0.0),
            negative_ratio=data_cfg.get('negative_ratio', 0.0),
        )
        
        # 验证集 - 验证时总是使用GT位姿裁切，不使用负样本
        self.val_dataset = CorrespondenceDataset(
            data_root=data_cfg['data_root'],
            scene_name=data_cfg['val_scene'],
            image_size=tuple(data_cfg['image_size']),
            augment=False,
            max_samples=data_cfg.get('max_val_samples'),
            use_depth=data_cfg.get('use_depth', False),
            gaussian_path=data_cfg.get('gaussian_path'),
            fx=data_cfg['fx'],
            fy=data_cfg['fy'],
            cx=data_cfg['cx'],
            cy=data_cfg['cy'],
            sample_step=data_cfg.get('val_step', 1),
            use_initial_pose=use_relative_pose,
            pose_noise_rot_deg=data_cfg.get('pose_noise_rot_deg', 5.0),
            pose_noise_trans_m=data_cfg.get('pose_noise_trans_m', 0.1),
            # 验证集不使用初始位姿裁切和负样本
            use_init_pose_for_culling=False,
            frustum_margin=0.0,
            negative_ratio=0.0,
        )
        
        # DataLoader
        train_cfg = self.config['training']
        
        # 分布式训练使用DistributedSampler
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=train_cfg['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg['val_batch_size'],
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=train_cfg['num_workers'],
            pin_memory=True,
        )
        
        if self.is_main_process:
            print(f"  - 训练集: {len(self.train_dataset)} 样本, {len(self.train_loader)} batches")
            print(f"  - 验证集: {len(self.val_dataset)} 样本, {len(self.val_loader)} batches")
        
        # 计算平移归一化参数（使用训练集统计）
        self._compute_normalization_params()
        
        if self.is_main_process:
            print("  ✓ 数据集准备完成")
    
    def _compute_normalization_params(self):
        """计算平移归一化参数"""
        normalize_translation = self.config['loss'].get('normalize_translation', True)
        
        if not normalize_translation:
            self.translation_scale = 1.0
            if self.is_main_process:
                print("  - 不使用平移归一化")
            return
        
        if self.is_main_process:
            print("  - 计算平移归一化参数...")
        
        # 从训练集采样计算平移幅度
        translations = []
        sample_size = min(len(self.train_dataset), 500)
        
        for i in range(sample_size):
            sample = self.train_dataset[i]
            pose = sample['pose']  # (4, 4)
            t = pose[:3, 3].numpy()
            translations.append(np.linalg.norm(t))
        
        # 使用中位数作为归一化尺度（更稳健）
        self.translation_scale = float(np.median(translations))
        
        if self.is_main_process:
            print(f"    平移归一化尺度: {self.translation_scale:.4f} m")
            print(f"    归一化后平移范围约为: ±{3.0/self.translation_scale:.2f} (标准化到相似量级)")
    
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("\n[步骤 5] 初始化优化器...")
        
        train_cfg = self.config['training']
        
        # 收集需要训练的参数
        params_to_train = []
        
        # ICPoseNet参数
        params_to_train.extend([
            {'params': self.model.parameters(), 'lr': train_cfg['learning_rate']}
        ])
        
        # Kendall's Loss参数（如果使用）
        if hasattr(self, 'loss_parameters') and self.loss_parameters:
            params_to_train.extend(self.loss_parameters)
            print("  - 添加Kendall's Loss参数到优化器")
        
        # 如果解码器可训练，添加其参数
        if not self.config['splatloc'].get('freeze_decoder', True):
            params_to_train.extend([
                {'params': self.feat_decoder.parameters(), 'lr': train_cfg['decoder_lr']}
            ])
        
        # 优化器
        optimizer_type = train_cfg.get('optimizer', 'adamw').lower()
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                params_to_train,
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg.get('weight_decay', 1e-4),
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                params_to_train,
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 学习率调度器
        scheduler_type = train_cfg.get('scheduler', 'cosine').lower()
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg['num_epochs'],
                eta_min=train_cfg.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg.get('lr_decay_step', 10),
                gamma=train_cfg.get('lr_decay_gamma', 0.5)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=train_cfg.get('lr_decay_gamma', 0.95)
            )
        else:
            self.scheduler = None
        
        if self.is_main_process:
            print(f"  - 优化器: {optimizer_type}")
            print(f"  - 学习率: {train_cfg['learning_rate']}")
            print(f"  - 权重衰减: {train_cfg.get('weight_decay', 1e-4)}")
            if self.scheduler:
                print(f"  - 学习率调度器: {scheduler_type}")
            print("  ✓ 优化器初始化完成")
    
    def _init_loss_functions(self):
        """初始化损失函数"""
        print("\n[步骤 5] 初始化损失函数...")
        
        loss_cfg = self.config['loss']
        
        # 使用Kendall's Loss（自动学习权重）
        use_kendall = loss_cfg.get('use_kendall', True)
        
        if use_kendall:
            print("  使用Kendall's Loss（自动权重）")
            self.pose_loss = PoseLossKendall(
                rotation_loss_type='rotation_6d',  # 6D旋转表示
                translation_loss_type=loss_cfg.get('translation_loss', 'l2'),
                reduction='mean',
                init_log_var_rotation=loss_cfg.get('init_log_var_rotation', 0.0),
                init_log_var_translation=loss_cfg.get('init_log_var_translation', 0.0),
            ).to(self.device)
            
            # 将loss的参数加入优化器
            self.loss_parameters = [
                {'params': [self.pose_loss.log_var_rotation], 'lr': self.config['training']['learning_rate']},
                {'params': [self.pose_loss.log_var_translation], 'lr': self.config['training']['learning_rate']},
            ]
        else:
            if self.is_main_process:
                print("  使用固定权重损失")
            self.pose_loss = PoseLoss(
                rotation_loss_type=loss_cfg['rotation_loss'],
                translation_loss_type=loss_cfg['translation_loss'],
                rotation_weight=loss_cfg['rotation_weight'],
                translation_weight=loss_cfg['translation_weight'],
                reduction='mean'
            )
            self.loss_parameters = []
        
        if self.is_main_process:
            print("  ✓ 损失函数初始化完成")
    
    def _extract_features(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从图像和点云提取特征（根据 input_type 自动调度）

        input_type='splatloc'    → 原始路径（FeatureDecoder + fused_feature图采样）
        input_type='feature3dgs' → 新路径（Feature3DGS渲染深度反投影 + 渲染特征图采样）

        新实现:
        - 2D特征: 从预提取的融合特征图中采样
        - 3D特征: 使用FeatureDecoder直接查询3D点
        
        参数:
            batch: 数据批次，包含images, poses, K, fused_feature等
            
        返回:
            img_feats: (B, N_img, C) 图像特征
            pcd_feats: (B, N_pcd, C) 点云特征
            img_pos_embeds: (B, N_img, C) 图像位置编码
            pcd_pos_embeds: (B, N_pcd, C) 点云位置编码
            img_pixels: (B, N_img, 2) 图像像素坐标 (u, v)
            pcd_points: (B, N_pcd, 3) 3D点云坐标 (x, y, z)
        """
        import numpy as np
        
        # ---- 根据 input_type 分支 ----
        if self.input_type == 'feature3dgs':
            return self._extract_features_feature3dgs(batch)

        images = batch['image'].to(self.device)  # [B, 3, H, W]
        poses = batch['pose'].to(self.device)    # [B, 4, 4]
        K = batch['intrinsics'].to(self.device)  # [B, 3, 3]
        pts_2d = batch['points_2d'].to(self.device)  # [total_N, 2] (u, v)
        pts_3d = batch['points_3d'].to(self.device)  # [total_N, 3]
        sample_indices = batch['sample_indices'].to(self.device)  # [total_N]
        batch_size = batch['batch_size']
        feature_dim = self.config['model']['feature_dim']
        
        # 获取融合特征
        fused_features = batch.get('fused_feature', None)  # [B, 256, H, W] or None
        if fused_features is not None:
            fused_features = fused_features.to(self.device)
        
        # === 提取3D点云特征 ===
        # pts_3d已经是flat的 [total_N, 3]，直接查询
        with torch.no_grad():
            pcd_feats_flat = self.feat_decoder(pts_3d)  # [total_N, feature_dim]
        
        # 按照sample_indices重新组织成batch格式
        # 首先统计每个batch中的点数
        pts_per_sample = []
        for b in range(batch_size):
            n_pts = (sample_indices == b).sum().item()
            pts_per_sample.append(n_pts)
        
        max_pts = max(pts_per_sample)
        
        # 创建padded tensors
        pcd_feats = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)
        
        for b in range(batch_size):
            mask = sample_indices == b
            n_pts = mask.sum().item()
            pcd_feats[b, :n_pts] = pcd_feats_flat[mask]
        
        # L2归一化3D特征（在添加位置编码之前）
        for b in range(batch_size):
            pcd_feats_batch = pcd_feats[b, :pts_per_sample[b]]
            if pcd_feats_batch.shape[0] > 0:
                pcd_feats[b, :pts_per_sample[b]] = torch.nn.functional.normalize(
                    pcd_feats_batch, p=2, dim=-1
                )
        
        # === 提取2D图像特征 ===
        img_feats = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)
        
        if fused_features is not None:
            # 从融合特征图中采样
            for b in range(batch_size):
                mask = sample_indices == b
                if not mask.any():
                    continue
                
                # 获取该样本的2D坐标
                pts_2d_b = pts_2d[mask]  # [N_b, 2] (u, v)
                n_pts = pts_2d_b.shape[0]
                
                # 注意：融合特征现在是[256, 35, 46]，而图像是[640, 480]
                # 需要将图像坐标映射到特征坐标
                H_feat, W_feat = fused_features.shape[2], fused_features.shape[3]  # 35, 46
                H_img, W_img = 480, 640  # 图像尺寸
                
                # 将图像坐标缩放到特征图坐标
                u_feat = pts_2d_b[:, 0] * (W_feat / W_img)  # [N_b]
                v_feat = pts_2d_b[:, 1] * (H_feat / H_img)  # [N_b]
                
                # 归一化到[-1, 1] for grid_sample
                grid_x = 2.0 * u_feat / (W_feat - 1) - 1.0  # u -> x
                grid_y = 2.0 * v_feat / (H_feat - 1) - 1.0  # v -> y
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N_b, 2]
                
                # 采样特征
                feat_b = fused_features[b:b+1]  # [1, 256, 35, 46]
                sampled_feats = torch.nn.functional.grid_sample(
                    feat_b, grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )  # [1, 256, 1, N_b]
                
                sampled_feats = sampled_feats.squeeze(2).squeeze(0).permute(1, 0)  # [N_b, 256]
                img_feats[b, :n_pts] = sampled_feats
            
            # L2归一化2D特征（在添加位置编码之前）
            # 使2D和3D特征都在单位超球面上，特征匹配只关注方向
            for b in range(batch_size):
                img_feats_batch = img_feats[b, :pts_per_sample[b]]  # [N_b, 256]
                if img_feats_batch.shape[0] > 0:
                    img_feats[b, :pts_per_sample[b]] = torch.nn.functional.normalize(
                        img_feats_batch, p=2, dim=-1
                    )
        else:
            # Fallback: 使用3D特征
            print("警告: 没有融合特征，使用3D特征作为2D特征（不推荐）")
            img_feats = pcd_feats.clone()
        
        # 获取实际模型（处理DDP包装）
        actual_model = self.model.module if self.is_distributed else self.model
        
        # 添加位置编码
        # 2D位置编码: 基于图像坐标 (u, v)
        coords_2d_norm = pts_2d.clone()  # [total_N, 2]
        coords_2d_norm[:, 0] = coords_2d_norm[:, 0] / 639.0  # u归一化到[0,1]
        coords_2d_norm[:, 1] = coords_2d_norm[:, 1] / 479.0  # v归一化到[0,1]
        
        # 重组为batch格式
        coords_2d_batch = torch.zeros(batch_size, max_pts, 2, device=self.device)
        for b in range(batch_size):
            mask = (sample_indices == b)
            n_pts = mask.sum()
            coords_2d_batch[b, :n_pts] = coords_2d_norm[mask]
        
        # 生成2D位置编码（不再直接添加到特征）
        # 改为通过embeds参数传递给Transformer
        pos_enc_2d = actual_model.pos_enc_2d(coords_2d_batch)  # [B, N, 256]
        img_pos_embeds = pos_enc_2d
        
        # 3D位置编码: 基于世界坐标 (x, y, z)
        coords_3d_batch = torch.zeros(batch_size, max_pts, 3, device=self.device)
        for b in range(batch_size):
            mask = (sample_indices == b)
            n_pts = mask.sum()
            coords_3d_batch[b, :n_pts] = pts_3d[mask]
        
        # 生成3D位置编码
        pos_enc_3d = actual_model.pos_enc_3d(coords_3d_batch)  # [B, N, 258]
        # 需要投影到25 6维
        pos_enc_3d_proj = actual_model.pos_enc_3d_proj(pos_enc_3d)  # [B, N, 256]
        pcd_pos_embeds = pos_enc_3d_proj
        
        # 🆕 返回坐标（用于keypoint提取）
        # img_pixels: (B, N, 2) - 像素坐标 (u, v)
        # pcd_points: (B, N, 3) - 3D点坐标 (x, y, z)
        img_pixels = coords_2d_batch  # 已经归一化到[0,1]，需要恢复到像素坐标
        # 恢复到像素坐标
        img_pixels = img_pixels.clone()
        img_pixels[:, :, 0] = img_pixels[:, :, 0] * 639.0  # u
        img_pixels[:, :, 1] = img_pixels[:, :, 1] * 479.0  # v
        
        pcd_points = coords_3d_batch  # 已经是世界坐标
        
        # 返回特征和位置编码（位置编码通过embeds参数传递）
        return img_feats, pcd_feats, img_pos_embeds, pcd_pos_embeds, img_pixels, pcd_points

    # ==================================================================
    # Feature3DGS 输入模式的特征提取
    # ==================================================================

    def _extract_features_feature3dgs(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Feature3DGS 模式的特征提取:

        数据流:
          1. 用 GT 位姿（或初始位姿）渲染 Feature3DGS → 深度图 + 特征图
          2. 深度图反投影 → 世界坐标 3D 点 + 对应像素坐标
          3. 在渲染特征图上采样 → 3D 特征 (与 DINO+SD 特征对齐)
          4. 在预提取 fused_feature 上采样同一像素 → 2D 特征
          5. 用像素坐标/世界坐标生成位置编码

        Args:
            batch: DataLoader 的 batch，必须含 'pose', 'fused_feature', 'intrinsics'

        Returns:
            同 _extract_features(): img_feats, pcd_feats, img_pos_embeds,
                                     pcd_pos_embeds, img_pixels, pcd_points
        """
        assert self.feature3dgs_provider is not None, (
            "feature3dgs_provider 未初始化，请检查 input_type 和 feature3dgs 配置"
        )

        data_cfg    = self.config['dataset']
        f3dgs_cfg   = self.config.get('feature3dgs', {})
        feature_dim = self.config['model']['feature_dim']

        # 相机参数
        fx = float(data_cfg['fx'])
        fy = float(data_cfg['fy'])
        cx = float(data_cfg['cx'])
        cy = float(data_cfg['cy'])
        img_h, img_w = int(data_cfg['image_size'][1]), int(data_cfg['image_size'][0])

        # 每帧采样点数
        num_samples = f3dgs_cfg.get('num_samples', data_cfg.get('num_pairs', 1024))

        # 使用 GT 还是初始位姿渲染？
        # - use_gt_pose_for_rendering=True (默认)：GT 位姿渲染，3D 点准确
        # - False：初始（带噪声）位姿渲染，更接近真实定位场景
        use_gt = f3dgs_cfg.get('use_gt_pose_for_rendering', True)

        poses_gt   = batch['pose'].to(self.device)          # (B, 4, 4) c2w GT
        batch_size = poses_gt.shape[0]

        if (not use_gt) and ('initial_pose' in batch):
            render_poses = batch['initial_pose'].to(self.device)  # (B, 4, 4) noisy
        else:
            render_poses = poses_gt  # (B, 4, 4) GT

        # fused_feature: (B, 256, fH, fW)
        fused_features = batch.get('fused_feature', None)
        if fused_features is not None:
            fused_features = fused_features.to(self.device)

        # --------  逐帧渲染（共享投影, CUDA上顺序执行）  --------
        all_pts3d   = []   # per-sample: (N_b, 3)
        all_pix     = []   # per-sample: (N_b, 2)  (u, v)
        all_pcd_f   = []   # per-sample: (N_b, feature_dim)
        pts_per_sample = []

        for b in range(batch_size):
            result = self.feature3dgs_provider.render_and_backproject(
                c2w=render_poses[b],
                fx=fx, fy=fy, cx=cx, cy=cy,
                img_height=img_h, img_width=img_w,
                num_samples=num_samples,
                norm_features=True,
            )
            N_b = result['points_3d'].shape[0]
            all_pts3d.append(result['points_3d'])   # (N_b, 3)
            all_pix.append(result['pixel_coords'])   # (N_b, 2)
            all_pcd_f.append(result['pcd_feats'])    # (N_b, feature_dim)
            pts_per_sample.append(N_b)

        max_pts = max(pts_per_sample) if pts_per_sample else num_samples

        # ----  组成 padded batch tensors  ----
        pcd_points  = torch.zeros(batch_size, max_pts, 3,           device=self.device)
        img_pixels  = torch.zeros(batch_size, max_pts, 2,           device=self.device)
        pcd_feats   = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)
        img_feats   = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)

        for b in range(batch_size):
            nb = pts_per_sample[b]
            if nb == 0:
                continue
            pcd_points[b, :nb] = all_pts3d[b]
            img_pixels[b, :nb] = all_pix[b]
            pcd_feats[b, :nb]  = all_pcd_f[b]

        # ----  2D 特征：从 fused_feature 在 img_pixels 位置采样  ----
        if fused_features is not None:
            fH, fW = fused_features.shape[2], fused_features.shape[3]

            for b in range(batch_size):
                nb = pts_per_sample[b]
                if nb == 0:
                    continue
                uv = img_pixels[b, :nb]          # (nb, 2)  (u, v) pixel coords
                u_feat = uv[:, 0] * (fW / img_w)
                v_feat = uv[:, 1] * (fH / img_h)
                gx = 2.0 * u_feat / (fW - 1) - 1.0
                gy = 2.0 * v_feat / (fH - 1) - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,nb,2)
                sampled = torch.nn.functional.grid_sample(
                    fused_features[b:b+1],                # (1, D, fH, fW)
                    grid,
                    mode='bilinear', padding_mode='border', align_corners=True
                )  # (1, D, 1, nb)
                sampled = sampled.squeeze(2).squeeze(0).T  # (nb, D)
                img_feats[b, :nb] = torch.nn.functional.normalize(sampled, p=2, dim=-1)
        else:
            # Fallback: 直接用渲染的 3D 特征也作为 2D 特征（调试用）
            if self.is_main_process:
                print("[WARNING] feature3dgs 模式下未找到 fused_feature，2D 特征将使用渲染特征代替")
            img_feats = pcd_feats.clone()

        # ----  位置编码  ----
        actual_model = self.model.module if self.is_distributed else self.model

        # 2D 位置编码：归一化像素坐标 (u/W, v/H)
        pix_norm = img_pixels.clone()
        pix_norm[:, :, 0] = pix_norm[:, :, 0] / (img_w - 1)
        pix_norm[:, :, 1] = pix_norm[:, :, 1] / (img_h - 1)
        img_pos_embeds = actual_model.pos_enc_2d(pix_norm)    # (B, N, C)

        # 3D 位置编码：世界坐标
        pos_enc_3d      = actual_model.pos_enc_3d(pcd_points)           # (B, N, C_raw)
        pcd_pos_embeds  = actual_model.pos_enc_3d_proj(pos_enc_3d)      # (B, N, C)

        return img_feats, pcd_feats, img_pos_embeds, pcd_pos_embeds, img_pixels, pcd_points

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        参数:
            epoch: 当前epoch编号
            
        返回:
            metrics: 训练指标字典
        """
        self.model.train()
        if not self.config['splatloc'].get('freeze_decoder', True):
            self.feat_decoder.train()
        
        # 分布式训练：设置sampler的epoch（保证每个epoch的shuffle不同）
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
                   disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 1. 提取特征和位置编码 + 坐标
                img_feats, pcd_feats, img_pos_embeds, pcd_pos_embeds, img_pixels, pcd_points = self._extract_features(batch)
                
                # 2. 准备初始位姿（迭代模型需要）
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                if use_relative_pose and 'initial_pose' in batch:
                    initial_pose = batch['initial_pose'].to(self.device)
                else:
                    initial_pose = None
                
                # 3. 前向传播（根据模型类型选择不同的调用方式）
                if getattr(self, 'use_iterative_model', False):
                    # 迭代精化模型：需要传入initial_pose
                    if initial_pose is None:
                        raise ValueError("迭代精化模型需要initial_pose！请启用use_relative_pose")
                    pose_matrix_pred, pose_9d, rotation_6d, translation_rel, \
                        img_heatmap, img_keypoints, pcd_keypoints, stage_poses = self.model(
                        img_feats, pcd_feats, img_pixels, pcd_points,
                        img_pos_embeds, pcd_pos_embeds,
                        initial_pose=initial_pose
                    )
                else:
                    # 标准模型
                    pose_matrix_pred, pose_9d, rotation_6d, translation_rel, \
                        img_heatmap, img_keypoints, pcd_keypoints = self.model(
                        img_feats, pcd_feats, img_pixels, pcd_points,
                        img_pos_embeds, pcd_pos_embeds
                    )
                    stage_poses = None
                
                # 4. 准备GT
                gt_poses_abs = batch['pose'].to(self.device)  # (B, 4, 4) 绝对位姿
                
                # 相对位姿选项
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                normalize_translation = self.config['loss'].get('normalize_translation', False)
                
                if use_relative_pose:
                    # 使用每帧的初始位姿（带噪声）
                    if 'initial_pose' not in batch:
                        raise ValueError("启用use_relative_pose但数据集中没有initial_pose！请检查数据集配置")
                    pose_init = batch['initial_pose'].to(self.device)  # (B, 4, 4) 每帧独立的初始位姿
                    # 计算相对GT：从pose_init到gt_poses_abs的变换
                    gt_poses = compute_relative_pose(gt_poses_abs, pose_init)
                else:
                    # 使用绝对位姿
                    gt_poses = gt_poses_abs
                
                # 4. 计算损失
                from modules.pose_regressor import rotation_6d_to_matrix
                R_pred = rotation_6d_to_matrix(rotation_6d)  # (B, 3, 3)
                
                # 准备用于loss计算的平移（归一化或原始）
                translation_for_loss = translation_rel.clone()
                gt_translation_for_loss = gt_poses[:, :3, 3].clone()
                
                # 归一化平移（仅用于loss计算）
                if normalize_translation and hasattr(self, 'translation_scale') and self.translation_scale > 0:
                    translation_for_loss = translation_for_loss / self.translation_scale
                    gt_translation_for_loss = gt_translation_for_loss / self.translation_scale
                
                # 检查是否使用Kendall's Loss
                use_kendall = self.config['loss'].get('use_kendall', False)
                
                if use_kendall:
                    # Kendall's Loss需要传入(R, t)元组
                    loss_dict = self.pose_loss(
                        pose_pred=(R_pred, translation_for_loss),
                        pose_gt=gt_poses,
                        return_components=True
                    )
                else:
                    # 标准PoseLoss需要完整位姿矩阵
                    # 构建用于loss计算的位姿矩阵（使用归一化后的平移）
                    pose_pred_full = torch.eye(4, device=R_pred.device).unsqueeze(0).expand(R_pred.shape[0], 4, 4).clone()
                    pose_pred_full[:, :3, :3] = R_pred
                    pose_pred_full[:, :3, 3] = translation_for_loss
                    
                    gt_poses_for_loss = gt_poses.clone()
                    gt_poses_for_loss[:, :3, 3] = gt_translation_for_loss
                    
                    loss_dict = self.pose_loss(
                        pose_pred=pose_pred_full,
                        pose_gt=gt_poses_for_loss,
                        return_components=True
                    )
                
                loss = loss_dict['loss']
                
                # 🆕 添加Diversity Loss
                from modules.diversity_loss import diversity_loss
                # 配置参数：margin和权重
                diversity_margin_2d = self.config['loss'].get('diversity_margin_2d', 10.0)  # 像素
                diversity_margin_3d = self.config['loss'].get('diversity_margin_3d', 0.1)  # 米
                diversity_weight = self.config['loss'].get('diversity_weight', 0.01)
                
                div_loss_2d = diversity_loss(img_keypoints, diversity_margin_2d)
                div_loss_3d = diversity_loss(pcd_keypoints, diversity_margin_3d)
                div_loss_total = diversity_weight * (div_loss_2d + div_loss_3d)
                
                # 添加到总loss
                loss = loss + div_loss_total
                
                # 记录diversity loss
                loss_dict['diversity_loss_2d'] = div_loss_2d.item()
                loss_dict['diversity_loss_3d'] = div_loss_3d.item()
                loss_dict['diversity_loss_total'] = div_loss_total.item()
                
                # 🆕 添加Reprojection Loss（几何一致性约束）
                from losses.reprojection_loss import reprojection_loss
                reprojection_weight = self.config['loss'].get('reprojection_weight', 0.1)
                
                if reprojection_weight > 0:
                    # 注意：使用绝对位姿gt_poses_abs计算重投影
                    # 因为img_keypoints和pcd_keypoints都是在绝对坐标系下检测的
                    intrinsics_batch = batch['intrinsics'].to(self.device)  # (B, 3, 3)
                    
                    reproj_loss, reproj_valid_ratio = reprojection_loss(
                        img_keypoints=img_keypoints,  # (B, N_query, 2) 像素坐标
                        pcd_keypoints=pcd_keypoints,  # (B, N_query, 3) 世界坐标
                        gt_pose=gt_poses_abs,         # (B, 4, 4) 绝对GT位姿
                        intrinsics=intrinsics_batch,
                        image_size=(640, 480),
                        reduction='mean',
                    )
                    
                    reproj_loss_weighted = reprojection_weight * reproj_loss
                    loss = loss + reproj_loss_weighted
                    
                    # 记录reprojection loss
                    loss_dict['reprojection_loss'] = reproj_loss.item()
                    loss_dict['reprojection_loss_weighted'] = reproj_loss_weighted.item()
                    loss_dict['reprojection_valid_ratio'] = reproj_valid_ratio.item()
                
                # 4. 检测NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ 检测到NaN/Inf损失在batch {batch_idx}! 跳过此batch")
                    print(f"  Loss: {loss.item()}")
                    print(f"  Rotation loss: {loss_dict['rotation_loss']}")
                    print(f"  Translation loss: {loss_dict['translation_loss']}")
                    continue
                
                # 5. 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 6. 检查梯度
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        if torch.isnan(param_norm) or torch.isinf(param_norm):
                            print(f"\n⚠️ 检测到NaN/Inf梯度在batch {batch_idx}! 跳过此batch")
                            self.optimizer.zero_grad()
                            continue
                total_grad_norm = total_grad_norm ** 0.5
                
                # 7. 梯度裁剪
                if self.config['training'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
                
                # 8. 统计
                total_loss += loss.item()
                total_rot_loss += loss_dict['rotation_loss']
                total_trans_loss += loss_dict['translation_loss']
                
                # 🆕 统计diversity loss
                if 'diversity_loss_total' in loss_dict:
                    if not hasattr(self, 'total_div_loss'):
                        self.total_div_loss = 0.0
                        self.total_div_loss_2d = 0.0
                        self.total_div_loss_3d = 0.0
                    self.total_div_loss += loss_dict['diversity_loss_total']
                    self.total_div_loss_2d += loss_dict['diversity_loss_2d']
                    self.total_div_loss_3d += loss_dict['diversity_loss_3d']
                
                # 9. 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rot': f"{loss_dict['rotation_loss']:.4f}",
                    'trans': f"{loss_dict['translation_loss']:.4f}",
                    'div': f"{loss_dict.get('diversity_loss_total', 0.0):.4f}",
                    'reproj': f"{loss_dict.get('reprojection_loss', 0.0):.2f}",
                    'grad': f"{total_grad_norm:.2f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # 10. TensorBoard日志（每N步）
                if self.global_step % self.config['training'].get('log_interval', 10) == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/rotation_loss', loss_dict['rotation_loss'], self.global_step)
                    self.writer.add_scalar('train/translation_loss', loss_dict['translation_loss'], self.global_step)
                    self.writer.add_scalar('train/gradient_norm', total_grad_norm, self.global_step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                    
                    # 🆕 Diversity Loss日志
                    if 'diversity_loss_total' in loss_dict:
                        self.writer.add_scalar('train/diversity_loss_total', loss_dict['diversity_loss_total'], self.global_step)
                        self.writer.add_scalar('train/diversity_loss_2d', loss_dict['diversity_loss_2d'], self.global_step)
                        self.writer.add_scalar('train/diversity_loss_3d', loss_dict['diversity_loss_3d'], self.global_step)
                    
                    # 🆕 Reprojection Loss日志
                    if 'reprojection_loss' in loss_dict:
                        self.writer.add_scalar('train/reprojection_loss', loss_dict['reprojection_loss'], self.global_step)
                        self.writer.add_scalar('train/reprojection_loss_weighted', loss_dict['reprojection_loss_weighted'], self.global_step)
                        self.writer.add_scalar('train/reprojection_valid_ratio', loss_dict['reprojection_valid_ratio'], self.global_step)
                    
                    # Kendall's Loss权重监控
                    if hasattr(self.pose_loss, 'log_var_rotation'):
                        self.writer.add_scalar('train/log_var_rotation', self.pose_loss.log_var_rotation.item(), self.global_step)
                        self.writer.add_scalar('train/log_var_translation', self.pose_loss.log_var_translation.item(), self.global_step)
                        self.writer.add_scalar('train/weight_rotation', loss_dict['effective_weight_rotation'].item(), self.global_step)
                        self.writer.add_scalar('train/weight_translation', loss_dict['effective_weight_translation'].item(), self.global_step)
                
                self.global_step += 1
                
            except Exception as e:
                if self.is_main_process:
                    print(f"\n错误发生在batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # 计算平均指标
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'rotation_loss': total_rot_loss / num_batches,
            'translation_loss': total_trans_loss / num_batches,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证模型
        
        参数:
            epoch: 当前epoch编号
            
        返回:
            metrics: 验证指标字典
        """
        self.model.eval()
        self.feat_decoder.eval()
        
        total_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        
        # 位姿误差统计
        rotation_errors = []
        translation_errors = []
        
        # 🆕 初始位姿误差统计（作为基准对比）
        init_rotation_errors = []
        init_translation_errors = []
        
        # 可视化相关
        vis_config = self.config.get('visualization', {})
        should_visualize = (
            self.is_main_process and 
            vis_config.get('enable', False) and 
            self.vis_dir is not None and
            epoch % vis_config.get('vis_interval', 5) == 0
        )
        vis_samples_saved = 0
        vis_num_samples = 1  # 只保存1个样本的置信度热力图
        
        pbar = tqdm(self.val_loader, desc=f"Validation",
                   disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 1. 提取特征和位置编码 + 坐标
                img_feats, pcd_feats, img_pos_embeds, pcd_pos_embeds, img_pixels, pcd_points = self._extract_features(batch)
                
                # 2. 准备初始位姿（迭代模型需要）
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                if use_relative_pose and 'initial_pose' in batch:
                    initial_pose = batch['initial_pose'].to(self.device)
                else:
                    initial_pose = None
                
                # 3. 前向传播（根据模型类型选择不同的调用方式）
                if getattr(self, 'use_iterative_model', False):
                    # 迭代精化模型
                    if initial_pose is None:
                        raise ValueError("迭代精化模型需要initial_pose！请启用use_relative_pose")
                    pose_matrix_pred, pose_9d, rotation_6d, translation, \
                        img_heatmap, img_keypoints, pcd_keypoints, stage_poses = self.model(
                        img_feats, pcd_feats, img_pixels, pcd_points,
                        img_pos_embeds, pcd_pos_embeds,
                        initial_pose=initial_pose
                    )
                else:
                    # 标准模型
                    pose_matrix_pred, pose_9d, rotation_6d, translation, \
                        img_heatmap, img_keypoints, pcd_keypoints = self.model(
                        img_feats, pcd_feats, img_pixels, pcd_points,
                        img_pos_embeds, pcd_pos_embeds
                    )
                    stage_poses = None
                
                # 4. 准备GT（与训练保持一致）
                gt_poses_abs = batch['pose'].to(self.device)
                
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                normalize_translation = self.config['loss'].get('normalize_translation', False)
                
                if use_relative_pose:
                    # 🔧 修复：使用每帧独立的初始位姿，与训练保持一致
                    if 'initial_pose' not in batch:
                        raise ValueError("启用use_relative_pose但数据集中没有initial_pose！请检查数据集配置")
                    pose_init = batch['initial_pose'].to(self.device)  # (B, 4, 4) 每帧独立的初始位姿
                    gt_poses = compute_relative_pose(gt_poses_abs, pose_init)
                else:
                    gt_poses = gt_poses_abs
                
                # 4. 计算损失
                from modules.pose_regressor import rotation_6d_to_matrix
                R_pred = rotation_6d_to_matrix(rotation_6d)
                
                # 准备用于loss计算的平移（归一化或原始）
                translation_for_loss = translation.clone()
                gt_translation_for_loss = gt_poses[:, :3, 3].clone()
                
                if normalize_translation and hasattr(self, 'translation_scale') and self.translation_scale > 0:
                    translation_for_loss = translation_for_loss / self.translation_scale
                    gt_translation_for_loss = gt_translation_for_loss / self.translation_scale
                
                use_kendall = self.config['loss'].get('use_kendall', False)
                
                if use_kendall:
                    loss_dict = self.pose_loss(
                        pose_pred=(R_pred, translation_for_loss),
                        pose_gt=gt_poses,
                        return_components=True
                    )
                else:
                    pose_pred_full = torch.eye(4, device=R_pred.device).unsqueeze(0).expand(R_pred.shape[0], 4, 4).clone()
                    pose_pred_full[:, :3, :3] = R_pred
                    pose_pred_full[:, :3, 3] = translation_for_loss
                    
                    gt_poses_for_loss = gt_poses.clone()
                    gt_poses_for_loss[:, :3, 3] = gt_translation_for_loss
                    
                    loss_dict = self.pose_loss(
                        pose_pred=pose_pred_full,
                        pose_gt=gt_poses_for_loss,
                        return_components=True
                    )
                
                loss = loss_dict['loss']
                
                # 5. 统计
                total_loss += loss.item()
                total_rot_loss += loss_dict['rotation_loss']
                total_trans_loss += loss_dict['translation_loss']
                
                # 6. 计算位姿误差（使用绝对位姿进行评估）
                if getattr(self, 'use_iterative_model', False):
                    # 迭代模型直接输出绝对位姿
                    pose_pred_eval = pose_matrix_pred
                else:
                    # 标准模型：重建完整的绝对位姿用于评估
                    pose_pred_eval = torch.eye(4, device=R_pred.device).unsqueeze(0).expand(R_pred.shape[0], 4, 4).clone()
                    pose_pred_eval[:, :3, :3] = R_pred
                    pose_pred_eval[:, :3, 3] = translation  # 使用原始translation
                    
                    # 如果使用了相对位姿，需要转回绝对位姿
                    if use_relative_pose:
                        pose_init = batch['initial_pose'].to(self.device)
                        pose_pred_eval = compose_pose(pose_pred_eval, pose_init)
                
                rot_error, trans_error = self._compute_pose_error(pose_pred_eval, gt_poses_abs)
                rotation_errors.extend(rot_error.cpu().numpy().tolist())
                translation_errors.extend(trans_error.cpu().numpy().tolist())
                
                # 🆕 计算初始位姿的误差作为基准
                if use_relative_pose and 'initial_pose' in batch:
                    init_pose = batch['initial_pose'].to(self.device)
                    init_rot_error, init_trans_error = self._compute_pose_error(init_pose, gt_poses_abs)
                    init_rotation_errors.extend(init_rot_error.cpu().numpy().tolist())
                    init_translation_errors.extend(init_trans_error.cpu().numpy().tolist())
                
                # 7. 可视化（仅在主进程且满足条件时）
                if should_visualize and vis_samples_saved < vis_num_samples:
                    self._save_confidence_heatmap(
                        epoch=epoch,
                        sample_idx=vis_samples_saved,
                        batch=batch,
                        img_feats=img_feats,
                        pcd_feats=pcd_feats,
                        img_heatmap=img_heatmap,  # 🆕 传递真正的attention heatmap
                        rot_error=rot_error[0].item(),
                        trans_error=trans_error[0].item()
                    )
                    vis_samples_saved += 1
                
                # 8. 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rot': f"{loss_dict['rotation_loss']:.4f}",
                    'trans': f"{loss_dict['translation_loss']:.4f}",
                })
                
            except Exception as e:
                if self.is_main_process:
                    print(f"\n验证错误发生在batch {batch_idx}: {str(e)}")
                continue
        
        # 分布式训练：同步所有GPU的统计结果
        if self.is_distributed:
            # 将统计数据转为tensor
            stats = torch.tensor([
                total_loss, total_rot_loss, total_trans_loss,
                len(rotation_errors), len(translation_errors),
                sum(rotation_errors), sum(translation_errors)
            ], device=self.device)
            
            # 汇总所有GPU的统计数据
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            
            # 从tensor中提取统计值
            total_loss = stats[0].item()
            total_rot_loss = stats[1].item()
            total_trans_loss = stats[2].item()
            total_samples = int(stats[3].item())
            
            # 收集所有GPU的误差列表（只在主进程中）
            if self.is_main_process:
                # 创建接收列表
                all_rotation_errors = [torch.zeros(len(rotation_errors), device=self.device) 
                                      for _ in range(dist.get_world_size())]
                all_translation_errors = [torch.zeros(len(translation_errors), device=self.device)
                                         for _ in range(dist.get_world_size())]
                
                # 收集所有进程的数据
                dist.gather(torch.tensor(rotation_errors, device=self.device), all_rotation_errors)
                dist.gather(torch.tensor(translation_errors, device=self.device), all_translation_errors)
                
                # 合并
                rotation_errors = torch.cat(all_rotation_errors).cpu().numpy()
                translation_errors = torch.cat(all_translation_errors).cpu().numpy()
            else:
                # 非主进程只需发送数据
                dist.gather(torch.tensor(rotation_errors, device=self.device))
                dist.gather(torch.tensor(translation_errors, device=self.device))
        
        # 计算平均指标（只在主进程或单GPU模式）
        if self.is_main_process:
            num_batches = len(self.val_loader) * (dist.get_world_size() if self.is_distributed else 1)
            rotation_errors_np = np.array(rotation_errors) if not self.is_distributed else rotation_errors
            translation_errors_np = np.array(translation_errors) if not self.is_distributed else translation_errors
            
            # 🆕 初始位姿误差
            init_rotation_errors_np = np.array(init_rotation_errors) if init_rotation_errors else None
            init_translation_errors_np = np.array(init_translation_errors) if init_translation_errors else None
        else:
            # 非主进程返回空指标
            return {}
        
        metrics = {
            'loss': total_loss / num_batches,
            'rotation_loss': total_rot_loss / num_batches,
            'translation_loss': total_trans_loss / num_batches,
            'rotation_error_mean': rotation_errors_np.mean(),
            'rotation_error_median': np.median(rotation_errors_np),
            'translation_error_mean': translation_errors_np.mean(),
            'translation_error_median': np.median(translation_errors_np),
        }
        
        # 🆕 添加初始位姿误差作为基准
        if init_rotation_errors_np is not None and len(init_rotation_errors_np) > 0:
            metrics['init_rotation_error_mean'] = init_rotation_errors_np.mean()
            metrics['init_translation_error_mean'] = init_translation_errors_np.mean()
        
        # 记录到TensorBoard（只在主进程）
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return metrics
    
    def _log(self, message: str):
        """记录日志到文件和控制台"""
        if self.is_main_process:
            print(message)
            if self.log_file is not None:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
    
    def _compute_pose_error(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算位姿误差（角度误差和平移误差）
        
        参数:
            pred_pose: (B, 4, 4) 预测位姿
            gt_pose: (B, 4, 4) 真实位姿
            
        返回:
            rotation_error: (B,) 旋转角度误差（度）
            translation_error: (B,) 平移距离误差（米）
        """
        # 提取旋转矩阵
        pred_R = pred_pose[:, :3, :3]
        gt_R = gt_pose[:, :3, :3]
        
        # 计算旋转误差（角度）
        R_diff = torch.matmul(pred_R, gt_R.transpose(-2, -1))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        rotation_error = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        rotation_error = rotation_error * 180.0 / np.pi  # 转换为度
        
        # 计算平移误差（欧氏距离）
        pred_t = pred_pose[:, :3, 3]
        gt_t = gt_pose[:, :3, 3]
        translation_error = torch.norm(pred_t - gt_t, dim=1)
        
        return rotation_error, translation_error
    
    def _save_visualizations(self, epoch, batch_idx, sample_idx, batch, 
                            img_feats, pcd_feats, pose_pred, pose_gt, 
                            rot_error, trans_error, vis_types):
        """
        保存可视化结果
        
        参数:
            epoch: 当前epoch
            batch_idx: 当前batch索引
            sample_idx: 当前样本索引
            batch: 数据batch
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            pose_pred: (B, 4, 4) 预测位姿
            pose_gt: (B, 4, 4) 真实位姿
            rot_error: (B,) 旋转误差
            trans_error: (B,) 平移误差
            vis_types: 可视化类型配置
        """
        import cv2
        
        try:
            # 只可视化batch中的第一个样本
            idx = 0
            
            # 创建epoch专用目录
            epoch_vis_dir = self.vis_dir / f'epoch_{epoch:04d}'
            epoch_vis_dir.mkdir(exist_ok=True)
            
            # 准备图像（从batch中获取）
            if 'image' in batch:
                img = batch['image'][idx].cpu().numpy()  # (3, H, W)
                if img.shape[0] == 3:  # 如果是CHW格式
                    img = np.transpose(img, (1, 2, 0))  # 转为HWC
                
                # 反归一化 (ImageNet标准化)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean  # 反归一化到[0, 1]
                img = np.clip(img, 0, 1)  # 裁剪到合法范围
                
                # 转换到[0, 255]
                img = (img * 255).astype(np.uint8)
            else:
                # 如果没有原始图像，创建空白图像
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 1. 位姿预测可视化
            if vis_types.get('pose_prediction', True):
                pose_vis_path = epoch_vis_dir / f'sample_{sample_idx:02d}_pose.png'
                visualize_pose_prediction(
                    gt_pose=pose_gt[idx].cpu().numpy(),
                    pred_pose=pose_pred[idx].cpu().numpy(),
                    img=img,
                    save_path=str(pose_vis_path)
                )
            
            # 2. 2D-3D特征相似度可视化
            if vis_types.get('correspondence', True) and 'pcd_xyz' in batch:
                correspondence_vis_path = epoch_vis_dir / f'sample_{sample_idx:02d}_correspondence.png'
                # 提取点云坐标
                pcd_xyz = batch['pcd_xyz'][idx].cpu().numpy()  # (N_pcd, 3)
                visualize_2d3d_correspondence(
                    img_feats=img_feats[idx].cpu().numpy(),  # (N_img, C)
                    pcd_feats=pcd_feats[idx].cpu().numpy(),  # (N_pcd, C)
                    img=img,
                    pcd_xyz=pcd_xyz,
                    save_path=str(correspondence_vis_path)
                )
            
            # 3. 特征相似度矩阵
            if vis_types.get('feature_similarity', True):
                similarity_vis_path = epoch_vis_dir / f'sample_{sample_idx:02d}_similarity.png'
                
                # 调试信息：检查特征统计
                img_f = img_feats[idx].cpu().numpy()
                pcd_f = pcd_feats[idx].cpu().numpy()
                self._log(f"  [Debug] 2D特征: shape={img_f.shape}, mean={img_f.mean():.4f}, std={img_f.std():.4f}, range=[{img_f.min():.4f}, {img_f.max():.4f}]")
                self._log(f"  [Debug] 3D特征: shape={pcd_f.shape}, mean={pcd_f.mean():.4f}, std={pcd_f.std():.4f}, range=[{pcd_f.min():.4f}, {pcd_f.max():.4f}]")
                
                visualize_feature_similarity_matrix(
                    img_feats=img_f,
                    pcd_feats=pcd_f,
                    save_path=str(similarity_vis_path)
                )
            
            if sample_idx == 0:  # 只在第一个样本时打印
                self._log(f"  ✓ 可视化已保存至: {epoch_vis_dir}")
        
        except Exception as e:
            self._log(f"  ⚠ 可视化保存失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _save_confidence_heatmap(self, epoch, sample_idx, batch, 
                                 img_feats, pcd_feats, img_heatmap, rot_error, trans_error):
        """
        保存置信度热力图（原始图像尺寸）
        
        参数:
            epoch: 当前epoch
            sample_idx: 当前样本索引
            batch: 数据batch
            img_feats: (B, N_img, C) 2D图像特征（用于获取坐标）
            pcd_feats: (B, N_pcd, C) 3D点云特征
            img_heatmap: (B, N_query, N_img) Query的attention权重
            rot_error: 旋转误差(标量)
            trans_error: 平移误差(标量)
        """
        import cv2
        import matplotlib.pyplot as plt
        
        try:
            idx = 0  # 第一个样本
            
            # 创建epoch专用目录
            epoch_vis_dir = self.vis_dir / f'epoch_{epoch:04d}'
            epoch_vis_dir.mkdir(exist_ok=True)
            
            # 1. 准备原始图像
            if 'image' in batch:
                img = batch['image'][idx].cpu().numpy()  # (3, H, W)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))  # 转为HWC
                
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
            else:
                img = np.zeros((480, 640, 3))
            
            H, W = img.shape[:2]  # 480, 640
            
            # 2. 🆕 使用真正的Attention Heatmap（ICL-I2PReg方式）
            # img_heatmap: (N_query, N_img) - 每个query对所有2D点的attention权重
            heatmap_data = img_heatmap[idx].cpu()  # (N_query, N_img)
            
            # 对所有queries取最大值 → 每个2D点被关注的最大程度
            confidence_scores = heatmap_data.max(dim=0)[0].numpy()  # (N_img,)
            
            # 3. 获取2D坐标
            pts_2d = batch['points_2d']  # [total_N, 2]
            sample_indices = batch['sample_indices']  # [total_N]
            mask = (sample_indices == idx)
            pts_2d_sample = pts_2d[mask].cpu().numpy()  # (N, 2) - (u, v)
            
            # 4. 创建原始图像尺寸的置信度热力图
            heatmap = np.zeros((H, W), dtype=np.float32)
            
            # 将点的置信度分配到最近的像素
            for i, (u, v) in enumerate(pts_2d_sample):
                x, y = int(round(u)), int(round(v))
                if 0 <= x < W and 0 <= y < H:
                    heatmap[y, x] = max(heatmap[y, x], confidence_scores[i])
            
            # 高斯平滑使热力图更连续
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            # 5. 可视化
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 左图：原始图像
            axes[0].imshow(img)
            axes[0].set_title('Input Image', fontsize=14)
            axes[0].axis('off')
            
            # 右图：置信度热力图叠加（自适应归一化）
            # 🆕 统计attention质量
            heatmap_max = heatmap.max() if heatmap.max() > 0 else 1e-6
            heatmap_mean = heatmap[heatmap > 0].mean() if (heatmap > 0).any() else 0
            heatmap_std = heatmap[heatmap > 0].std() if (heatmap > 0).any() else 0
            
            # 🆕 打印attention统计信息
            raw_attn_max = heatmap_data.max().item()
            raw_attn_min = heatmap_data.min().item()
            raw_attn_std = heatmap_data.std().item()
            self._log(f"  📊 Attention Stats: max={raw_attn_max:.6f}, min={raw_attn_min:.6f}, std={raw_attn_std:.6f}")
            
            axes[1].imshow(img)
            # 🆕 使用自适应归一化：vmax=heatmap_max 而非固定的 1.0
            im = axes[1].imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=heatmap_max)
            axes[1].set_title(f'Confidence Heatmap (max={heatmap_max:.4f})\nRot: {rot_error:.2f}°, Trans: {trans_error:.3f}m', 
                            fontsize=14)
            axes[1].axis('off')
            
            # 添加colorbar
            plt.colorbar(im, ax=axes[1], label='2D-3D Match Confidence', fraction=0.046)
            
            plt.tight_layout()
            save_path = epoch_vis_dir / f'confidence_heatmap_epoch{epoch:04d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self._log(f"  ✓ 置信度热力图已保存: {save_path}")
            
        except Exception as e:
            self._log(f"  ⚠ 可视化保存失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存checkpoint
        
        参数:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        # 获取模型state_dict（处理DDP包装）
        if self.is_distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if not self.config['splatloc'].get('freeze_decoder', True):
            checkpoint['decoder_state_dict'] = self.feat_decoder.state_dict()
        
        # 保存latest.pth（只在主进程）
        if self.is_main_process:
            latest_path = self.checkpoint_dir / 'latest.pth'
            torch.save(checkpoint, latest_path)
            print(f"  ✓ 保存checkpoint: {latest_path}")
        
            # 保存best.pth
            if is_best:
                best_path = self.checkpoint_dir / 'best.pth'
                torch.save(checkpoint, best_path)
                print(f"  ✓ 保存最佳模型: {best_path}")
        
            # 定期保存epoch checkpoint
            if epoch % self.config['training'].get('save_interval', 10) == 0:
                epoch_path = self.checkpoint_dir / f'epoch_{epoch:04d}.pth'
                torch.save(checkpoint, epoch_path)
                print(f"  ✓ 保存epoch checkpoint: {epoch_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        从checkpoint恢复训练
        
        参数:
            checkpoint_path: checkpoint文件路径
        """
        if self.is_main_process:
            print(f"\n从checkpoint恢复: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型（处理DDP包装）
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复调度器
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复解码器（如果需要）
        if not self.config['splatloc'].get('freeze_decoder', True) and 'decoder_state_dict' in checkpoint:
            self.feat_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # 恢复训练状态
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.is_main_process:
            print(f"  ✓ 从epoch {self.epoch}恢复训练")
            print(f"  ✓ 全局步数: {self.global_step}")
            print(f"  ✓ 最佳验证损失: {self.best_val_loss:.6f}")
    
    def train(self):
        """主训练循环"""
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80)
        
        num_epochs = self.config['training']['num_epochs']
        start_epoch = self.epoch + 1
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                self.epoch = epoch
                
                if self.is_main_process:
                    self._log(f"\n{'='*60}")
                    self._log(f"Epoch {epoch}/{num_epochs}")
                    self._log(f"{'='*60}")
                
                # 训练
                train_metrics = self.train_epoch(epoch)
                if self.is_main_process:
                    train_msg = (f"\n[训练] Loss: {train_metrics['loss']:.6f} | "
                                f"Rot: {train_metrics['rotation_loss']:.6f} | "
                                f"Trans: {train_metrics['translation_loss']:.6f}")
                    self._log(train_msg)
                
                # 验证
                if epoch % self.config['training'].get('val_interval', 1) == 0:
                    val_metrics = self.validate(epoch)
                    if self.is_main_process and val_metrics:  # 只有主进程有metrics
                        val_msg = (f"[验证] Loss: {val_metrics['loss']:.6f} | "
                                  f"Rot: {val_metrics['rotation_loss']:.6f} | "
                                  f"Trans: {val_metrics['translation_loss']:.6f}")
                        self._log(val_msg)
                        error_msg = (f"       角度误差: {val_metrics['rotation_error_mean']:.2f}° "
                                    f"(中位数: {val_metrics['rotation_error_median']:.2f}°)")
                        self._log(error_msg)
                        trans_msg = (f"       平移误差: {val_metrics['translation_error_mean']:.4f}m "
                                    f"(中位数: {val_metrics['translation_error_median']:.4f}m)")
                        self._log(trans_msg)
                        
                        # 🆕 显示初始位姿误差作为基准对比
                        if 'init_rotation_error_mean' in val_metrics:
                            init_msg = (f"       [基准] 初始位姿误差: {val_metrics['init_rotation_error_mean']:.2f}° / "
                                       f"{val_metrics['init_translation_error_mean']:.4f}m")
                            self._log(init_msg)
                            # 计算改进率
                            rot_improve = (1 - val_metrics['rotation_error_mean'] / val_metrics['init_rotation_error_mean']) * 100
                            trans_improve = (1 - val_metrics['translation_error_mean'] / val_metrics['init_translation_error_mean']) * 100
                            improve_msg = f"       [改进] 旋转: {rot_improve:+.1f}% | 平移: {trans_improve:+.1f}%"
                            self._log(improve_msg)
                    
                        # 检查是否为最佳模型
                        is_best = val_metrics['loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_metrics['loss']
                            best_msg = f"  ★ 新的最佳模型! 验证损失: {self.best_val_loss:.6f}"
                            self._log(best_msg)
                    
                        # 保存checkpoint
                        self.save_checkpoint(epoch, is_best=is_best)
                else:
                    # 即使不验证，也保存latest checkpoint
                    if self.is_main_process:
                        self.save_checkpoint(epoch, is_best=False)
                
                # 更新学习率
                if self.scheduler is not None:
                    self.scheduler.step()
                
        except KeyboardInterrupt:
            if self.is_main_process:
                self._log("\n训练被用户中断")
                self._log("保存当前状态...")
                self.save_checkpoint(self.epoch, is_best=False)
            
        except Exception as e:
            if self.is_main_process:
                self._log(f"\n训练过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                self._log("保存当前状态...")
                self.save_checkpoint(self.epoch, is_best=False)
            raise
        
        finally:
            if self.writer is not None and self.is_main_process:
                self.writer.close()
            if self.is_main_process:
                self._log("\n训练完成!")
                self._log(f"最佳验证损失: {self.best_val_loss:.6f}")
                self._log(f"输出目录: {self.output_dir}")
            
            # 清理分布式进程组
            if self.is_distributed:
                dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='隐式对应关系位姿估计网络训练脚本')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练（路径到.pth文件）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（覆盖配置文件中的设置）')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的local rank（由torch.distributed.launch自动设置）')
    
    args = parser.parse_args()
    
    # 初始化分布式训练
    if 'LOCAL_RANK' in os.environ:
        # 使用torchrun启动时
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    elif args.local_rank >= 0:
        # 使用torch.distributed.launch启动时
        local_rank = args.local_rank
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        # 单卡训练
        local_rank = -1
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 加载配置
    if local_rank <= 0:  # 只在主进程打印
        print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 覆盖输出目录
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 创建训练器并开始训练
    trainer = ICPoseTrainer(config, local_rank=local_rank, resume_path=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
