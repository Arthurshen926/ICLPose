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

# SplatLoc相关导入
sys.path.append(str(Path(__file__).parent.parent))
from gaussian_splatting.scene.gaussian_model import GaussianModel
from models.decoders import FeatureDecoder
from gaussian_splatting.gaussian_renderer import render
from utils.camera_utils import Camera

# 隐式对应关系模块导入
from data.dataset import CorrespondenceDataset, collate_fn
from losses.pose_loss import PoseLoss, PoseLossKendall
from ic_models.ic_pose_net import ICPoseNet


def compute_relative_pose(pose_target, pose_init):
    """
    计算相对位姿: pose_rel = pose_target @ inv(pose_init)
    
    Args:
        pose_target: (B, 4, 4) 目标位姿（绝对）
        pose_init: (B, 4, 4) 初始位姿（绝对）
        
    Returns:
        pose_rel: (B, 4, 4) 相对位姿
    """
    # inv(pose_init) @ pose_target 等价于从init坐标系到target坐标系的变换
    R_init = pose_init[:, :3, :3]  # (B, 3, 3)
    t_init = pose_init[:, :3, 3]   # (B, 3)
    
    R_target = pose_target[:, :3, :3]  # (B, 3, 3)
    t_target = pose_target[:, :3, 3]   # (B, 3)
    
    # 相对旋转: R_rel = R_target @ R_init^T
    R_rel = torch.bmm(R_target, R_init.transpose(1, 2))
    
    # 相对平移: t_rel = R_init^T @ (t_target - t_init)
    t_rel = torch.bmm(R_init.transpose(1, 2), (t_target - t_init).unsqueeze(-1)).squeeze(-1)
    
    # 组合
    pose_rel = torch.eye(4, device=pose_target.device).unsqueeze(0).expand(pose_target.shape[0], 4, 4).clone()
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
        
        # 1. 加载SplatLoc预训练模型
        self._load_splatloc_models()
        
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
    
    def _load_splatloc_models(self):
        """加载SplatLoc已训练的gaussians和feat_decoder"""
        print("\n[步骤 1] 加载SplatLoc预训练模型...")
        
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
        self.model = ICPoseNet(
            feature_dim=model_cfg['feature_dim'],
            num_queries=model_cfg['num_queries'],
            fusion_layers=model_cfg['fusion_layers'],
            num_heads=model_cfg['num_heads'],
            dropout=model_cfg['dropout']
        ).to(self.device)
        
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
        )
        
        # 验证集
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
    
    def _extract_features(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从图像和点云提取特征
        
        新实现:
        - 2D特征: 从预提取的融合特征图中采样
        - 3D特征: 使用FeatureDecoder直接查询3D点
        
        参数:
            batch: 数据批次，包含images, poses, K, fused_feature等
            
        返回:
            img_feats: (B, N_img, C) 图像特征
            pcd_feats: (B, N_pcd, C) 点云特征
        """
        import numpy as np
        
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
        
        # 生成2D位置编码
        pos_enc_2d = actual_model.pos_enc_2d(coords_2d_batch)  # [B, N, 256]
        img_feats = img_feats + pos_enc_2d  # 特征 + 位置编码
        
        # 3D位置编码: 基于世界坐标 (x, y, z)
        coords_3d_batch = torch.zeros(batch_size, max_pts, 3, device=self.device)
        for b in range(batch_size):
            mask = (sample_indices == b)
            n_pts = mask.sum()
            coords_3d_batch[b, :n_pts] = pts_3d[mask]
        
        # 生成3D位置编码
        pos_enc_3d = actual_model.pos_enc_3d(coords_3d_batch)  # [B, N, 258]
        # 需要投影到256维
        pos_enc_3d_proj = actual_model.pos_enc_3d_proj(pos_enc_3d)  # [B, N, 256]
        pcd_feats = pcd_feats + pos_enc_3d_proj  # 特征 + 位置编码
        
        return img_feats, pcd_feats
    
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
                # 1. 提取特征
                img_feats, pcd_feats = self._extract_features(batch)
                
                # 2. 前向传播
                pose_matrix_pred, pose_9d, rotation_6d, translation_rel = self.model(
                    img_feats, pcd_feats
                )
                
                # 3. 准备GT
                gt_poses_abs = batch['pose'].to(self.device)  # (B, 4, 4) 绝对位姿
                
                # 相对位姿选项
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                normalize_translation = self.config['loss'].get('normalize_translation', False)
                
                if use_relative_pose:
                    # 初始位姿：batch第一帧
                    pose_init = gt_poses_abs[0:1].expand(gt_poses_abs.shape[0], 4, 4).clone()
                    # 计算相对GT
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
                
                # 9. 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rot': f"{loss_dict['rotation_loss']:.4f}",
                    'trans': f"{loss_dict['translation_loss']:.4f}",
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
        
        pbar = tqdm(self.val_loader, desc=f"Validation",
                   disable=not self.is_main_process)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 1. 提取特征
                img_feats, pcd_feats = self._extract_features(batch)
                
                # 2. 前向传播
                pose_matrix_pred, pose_9d, rotation_6d, translation = self.model(
                    img_feats, pcd_feats
                )
                
                # 3. 准备GT（与训练保持一致）
                gt_poses_abs = batch['pose'].to(self.device)
                
                use_relative_pose = self.config['loss'].get('use_relative_pose', False)
                normalize_translation = self.config['loss'].get('normalize_translation', False)
                
                if use_relative_pose:
                    pose_init = gt_poses_abs[0:1].expand(gt_poses_abs.shape[0], 4, 4).clone()
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
                # 重建完整的绝对位姿用于评估
                pose_pred_eval = torch.eye(4, device=R_pred.device).unsqueeze(0).expand(R_pred.shape[0], 4, 4).clone()
                pose_pred_eval[:, :3, :3] = R_pred
                pose_pred_eval[:, :3, 3] = translation  # 使用原始translation（模型直接输出，从未归一化）
                
                # 如果使用了相对位姿，需要转回绝对位姿
                if use_relative_pose:
                    pose_init = gt_poses_abs[0:1].expand(gt_poses_abs.shape[0], 4, 4).clone()
                    pose_pred_eval = compose_pose(pose_pred_eval, pose_init)
                
                rot_error, trans_error = self._compute_pose_error(pose_pred_eval, gt_poses_abs)
                rotation_errors.extend(rot_error.cpu().numpy().tolist())
                translation_errors.extend(trans_error.cpu().numpy().tolist())
                
                # 7. 更新进度条
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
