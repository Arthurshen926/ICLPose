#!/usr/bin/env python3
"""
ICPoseNet V2 训练脚本

支持功能:
1. ICPoseNetV2 (C2F + 重叠检测)
2. C2F多阶段辅助损失
3. 旋转损失Warmup
4. 模块化设计，所有功能可配置

使用方法:
    python train_v2.py --config configs/exp020_config.yaml
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

# 工具模块（新增）
from utils.model_factory import create_model, get_model_info, print_model_summary
from utils.loss_factory import create_loss_function, create_combined_loss
from utils.training_utils import (
    compute_relative_pose,
    compose_pose,
    compute_pose_error,
    GradientClipper,
    MetricLogger,
    create_optimizer,
    create_scheduler,
)

# SplatLoc相关导入
from splatloc_modules.gaussian_splatting.scene.gaussian_model import GaussianModel
from splatloc_modules.models.decoders import FeatureDecoder

# 数据集
from data.dataset import CorrespondenceDataset, collate_fn


class ICPoseTrainerV2:
    """
    ICPoseNet V2 训练器
    
    模块化设计:
    - 使用工厂函数创建模型和损失
    - 所有功能通过配置文件控制
    """
    
    def __init__(self, config: Dict, local_rank: int = -1, resume_path: str = None):
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
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        self._setup_output_dirs()
        
        # 初始化组件
        self._print_header()
        self._load_splatloc_models()
        self._init_model()
        self._prepare_datasets()
        self._init_loss()
        self._init_optimizer()
        
        # 从checkpoint恢复
        if resume_path:
            self._load_checkpoint(resume_path)
    
    def _print_header(self):
        if self.is_main_process:
            print("=" * 70)
            print("ICPoseNet V2 训练器")
            print("=" * 70)
            model_version = self.config.get('model', {}).get('version', 'v1')
            loss_version = self.config.get('loss', {}).get('version', 'standard')
            print(f"  模型版本: {model_version}")
            print(f"  损失版本: {loss_version}")
            print(f"  设备: {self.device}")
            if self.is_distributed:
                print(f"  分布式训练: {dist.get_world_size()} GPUs")
            print("=" * 70)
    
    def _setup_output_dirs(self):
        """创建输出目录"""
        if self.is_main_process:
            self.output_dir = Path(self.config['output_dir'])
            self.checkpoint_dir = self.output_dir / 'checkpoints'
            self.log_dir = self.output_dir / 'logs'
            self.vis_dir = self.output_dir / 'visualizations'
            
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.get('visualization', {}).get('enable', False):
                self.vis_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            
            # 保存配置
            config_save_path = self.output_dir / 'config.yaml'
            with open(config_save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            self.output_dir = Path(self.config['output_dir'])
            self.writer = None
    
    def _load_splatloc_models(self):
        """加载SplatLoc预训练模型"""
        if self.is_main_process:
            print("\n[1/5] 加载SplatLoc模型...")
        
        splatloc_cfg = self.config['splatloc']
        
        # 加载配置
        splatloc_config_path = splatloc_cfg.get('config_path')
        if splatloc_config_path and os.path.exists(splatloc_config_path):
            with open(splatloc_config_path, 'r') as f:
                splatloc_config = yaml.safe_load(f)
            if 'Training' not in splatloc_config:
                splatloc_config['Training'] = {'primitive_reg': False}
        else:
            splatloc_config = {
                'Training': {'primitive_reg': False},
                'scene': self.config.get('scene', {}),
                'decoder': self.config.get('decoder', {}),
            }
        
        # 确保配置完整
        if 'scene' not in splatloc_config:
            splatloc_config['scene'] = self.config.get('scene', {})
        if 'decoder' not in splatloc_config:
            splatloc_config['decoder'] = self.config.get('decoder', {})
        
        # 加载Gaussian模型
        self.gaussians = GaussianModel(sh_degree=0, config=splatloc_config)
        self.gaussians.load_ply(splatloc_cfg['gaussians_path'])
        
        if self.is_main_process:
            print(f"  ✓ Gaussian点数: {self.gaussians.get_xyz.shape[0]}")
        
        # 加载特征解码器
        self.feat_decoder = FeatureDecoder(config=splatloc_config, input_ch=3).to(self.device)
        self.feat_decoder.bounding_box = self.feat_decoder.bounding_box.to(self.device)
        
        # 加载权重
        checkpoint = torch.load(splatloc_cfg['decoder_path'], map_location=self.device)
        ckpt_state = checkpoint.get('decoder_state_dict', checkpoint)
        
        model_state = self.feat_decoder.state_dict()
        loaded_keys = 0
        for key, value in ckpt_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys += 1
        self.feat_decoder.load_state_dict(model_state)
        
        if self.is_main_process:
            print(f"  ✓ 解码器加载了 {loaded_keys} 个参数")
        
        # 冻结解码器
        if splatloc_cfg.get('freeze_decoder', True):
            for param in self.feat_decoder.parameters():
                param.requires_grad = False
            self.feat_decoder.eval()
            if self.is_main_process:
                print(f"  ✓ 解码器已冻结")
        
        # 渲染参数
        from munch import munchify
        self.pipeline_params = munchify({
            'convert_SHs_python': False,
            'compute_cov3D_python': False,
            'debug': False
        })
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
    
    def _init_model(self):
        """初始化模型"""
        if self.is_main_process:
            print("\n[2/5] 初始化模型...")
        
        # 使用工厂函数创建模型
        self.model = create_model(self.config, self.device)
        
        # DDP包装
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,  # V2模型可能有未使用的参数
            )
        
        # 打印模型信息
        if self.is_main_process:
            print_model_summary(self.model)
    
    def _prepare_datasets(self):
        """准备数据集"""
        if self.is_main_process:
            print("\n[3/5] 准备数据集...")
        
        data_cfg = self.config['dataset']
        loss_cfg = self.config.get('loss', {})
        use_relative_pose = loss_cfg.get('use_relative_pose', True)
        
        # 检查是否使用多序列训练
        train_scenes = data_cfg.get('train_scenes', None)
        if train_scenes is None:
            # 兼容旧配置格式
            train_scenes = [{'name': data_cfg['train_scene'], 'weight': 1.0}]
        
        # 创建多个训练数据集
        train_datasets = []
        for scene_info in train_scenes:
            if isinstance(scene_info, str):
                scene_name = scene_info
            else:
                scene_name = scene_info['name']
            
            ds = CorrespondenceDataset(
                data_root=data_cfg['data_root'],
                scene_name=scene_name,
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
                pose_noise_trans_m=data_cfg.get('pose_noise_trans_m', 0.3),
                use_init_pose_for_culling=data_cfg.get('use_init_pose_for_culling', False),
                frustum_margin=data_cfg.get('frustum_margin', 0.0),
                negative_ratio=data_cfg.get('negative_ratio', 0.0),
                num_pairs=data_cfg.get('num_pairs', 1024),
            )
            train_datasets.append(ds)
            if self.is_main_process:
                print(f"    ✓ {scene_name}: {len(ds)} 样本")
        
        # 合并训练数据集
        if len(train_datasets) == 1:
            self.train_dataset = train_datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            self.train_dataset = ConcatDataset(train_datasets)
        
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
            use_initial_pose=use_relative_pose,
            pose_noise_rot_deg=data_cfg.get('pose_noise_rot_deg', 5.0),
            pose_noise_trans_m=data_cfg.get('pose_noise_trans_m', 0.3),
            use_init_pose_for_culling=False,
            num_pairs=data_cfg.get('num_pairs', 1024),
            frustum_margin=0.0,
            negative_ratio=0.0,
        )
        
        # DataLoader
        train_cfg = self.config['training']
        
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=train_cfg.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg.get('val_batch_size', train_cfg['batch_size']),
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=train_cfg.get('num_workers', 4),
            pin_memory=True,
        )
        
        if self.is_main_process:
            print(f"  ✓ 训练集总计: {len(self.train_dataset)} 样本 ({len(train_scenes)} 序列)")
            print(f"  ✓ 验证集: {len(self.val_dataset)} 样本")
    
    def _init_loss(self):
        """初始化损失函数"""
        if self.is_main_process:
            print("\n[4/5] 初始化损失函数...")
        
        # 使用组合损失（自动处理C2F、重叠检测等）
        self.criterion = create_combined_loss(self.config, self.device)
        
        # 收集损失函数的可学习参数（如Kendall权重）
        self.loss_parameters = []
        for name, param in self.criterion.named_parameters():
            if param.requires_grad:
                self.loss_parameters.append(param)
                if self.is_main_process:
                    print(f"  + 可学习损失参数: {name}")
    
    def _init_optimizer(self):
        """初始化优化器"""
        if self.is_main_process:
            print("\n[5/5] 初始化优化器...")
        
        train_cfg = self.config['training']
        
        # 收集参数
        param_groups = [
            {'params': self.model.parameters(), 'lr': train_cfg['learning_rate']},
        ]
        
        # 损失函数参数
        if self.loss_parameters:
            param_groups.append({
                'params': self.loss_parameters,
                'lr': train_cfg['learning_rate'],
                'name': 'loss_params'
            })
        
        # 解码器参数（如果可训练）
        if not self.config['splatloc'].get('freeze_decoder', True):
            param_groups.append({
                'params': self.feat_decoder.parameters(),
                'lr': train_cfg.get('decoder_lr', train_cfg['learning_rate'] * 0.1),
                'name': 'decoder'
            })
        
        self.optimizer = create_optimizer(self.model, self.config)
        
        # 添加其他参数组
        for pg in param_groups[1:]:
            self.optimizer.add_param_group(pg)
        
        self.scheduler = create_scheduler(self.optimizer, self.config)
        
        # 梯度裁剪
        self.grad_clipper = GradientClipper(
            max_norm=train_cfg.get('grad_clip', 1.0)
        )
        
        if self.is_main_process:
            print("  ✓ 优化器初始化完成")
    
    def _extract_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        提取特征（统一接口）
        
        Returns:
            dict: 包含所有需要的特征和坐标
        """
        images = batch['image'].to(self.device)
        poses = batch['pose'].to(self.device)
        K = batch['intrinsics'].to(self.device)
        pts_2d = batch['points_2d'].to(self.device)
        pts_3d = batch['points_3d'].to(self.device)
        sample_indices = batch['sample_indices'].to(self.device)
        batch_size = batch['batch_size']
        feature_dim = self.config['model']['feature_dim']
        
        fused_features = batch.get('fused_feature')
        if fused_features is not None:
            fused_features = fused_features.to(self.device)
        
        # 提取3D特征
        with torch.no_grad():
            pcd_feats_flat = self.feat_decoder(pts_3d)
        
        # 统计每个样本的点数
        pts_per_sample = [(sample_indices == b).sum().item() for b in range(batch_size)]
        max_pts = max(pts_per_sample)
        
        # 重组为batch格式
        pcd_feats = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)
        pcd_points = torch.zeros(batch_size, max_pts, 3, device=self.device)
        img_pixels = torch.zeros(batch_size, max_pts, 2, device=self.device)
        img_feats = torch.zeros(batch_size, max_pts, feature_dim, device=self.device)
        
        for b in range(batch_size):
            mask = sample_indices == b
            n_pts = pts_per_sample[b]
            pcd_feats[b, :n_pts] = pcd_feats_flat[mask]
            pcd_points[b, :n_pts] = pts_3d[mask]
            img_pixels[b, :n_pts] = pts_2d[mask]
        
        # 注意: FeatureDecoder已经在输出时进行L2归一化 (decoders.py L66)
        # 因此这里不需要再次归一化
        
        # 提取2D特征
        if fused_features is not None:
            # 从config获取图像尺寸
            img_width = self.config['dataset']['image_size'][0]
            img_height = self.config['dataset']['image_size'][1]
            
            for b in range(batch_size):
                mask = sample_indices == b
                if not mask.any():
                    continue
                
                pts_2d_b = pts_2d[mask]
                n_pts = pts_2d_b.shape[0]
                
                H_feat, W_feat = fused_features.shape[2], fused_features.shape[3]
                
                u_feat = pts_2d_b[:, 0] * (W_feat / img_width)
                v_feat = pts_2d_b[:, 1] * (H_feat / img_height)
                
                grid_x = 2.0 * u_feat / (W_feat - 1) - 1.0
                grid_y = 2.0 * v_feat / (H_feat - 1) - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
                
                sampled_feats = torch.nn.functional.grid_sample(
                    fused_features[b:b+1], grid,
                    mode='bilinear', padding_mode='border', align_corners=True
                ).squeeze(2).squeeze(0).permute(1, 0)
                
                img_feats[b, :n_pts] = sampled_feats
            
            # 注意: 融合特征已经在提取时归一化
            # 因此这里不需要再次归一化
        else:
            img_feats = pcd_feats.clone()
        
        # 生成全局图像特征（用于V2重叠检测）
        if fused_features is not None:
            # 全局平均池化
            img_global_feats = fused_features.mean(dim=(2, 3))  # (B, C)
        else:
            img_global_feats = img_feats.mean(dim=1)  # (B, C)
        
        return {
            'img_feats': img_feats,
            'pcd_feats': pcd_feats,
            'img_pixels': img_pixels,
            'pcd_points': pcd_points,
            'img_global_feats': img_global_feats,
            'intrinsics': K,
            'poses': poses,
            'pts_per_sample': pts_per_sample,
        }
    
    def _forward_v2(self, features: Dict, batch: Dict) -> Dict:
        """V2模型前向传播"""
        model = self.model.module if self.is_distributed else self.model
        
        outputs = model(
            img_feats=features['img_feats'],
            pcd_feats=features['pcd_feats'],
            img_pixels=features['img_pixels'],
            pcd_points=features['pcd_points'],
            intrinsics=features['intrinsics'],
            img_global_feats=features['img_global_feats'],
            return_all_stages=True,
        )
        
        return outputs
    
    def _forward_v1(self, features: Dict, batch: Dict) -> Dict:
        """V1模型前向传播"""
        model = self.model.module if self.is_distributed else self.model
        
        # 生成位置编码 - 分别归一化u和v坐标
        img_width = self.config['dataset']['image_size'][0]
        img_height = self.config['dataset']['image_size'][1]
        img_pixels_norm = features['img_pixels'].clone()
        img_pixels_norm[:, :, 0] = img_pixels_norm[:, :, 0] / img_width   # u / width
        img_pixels_norm[:, :, 1] = img_pixels_norm[:, :, 1] / img_height  # v / height
        img_pos_embeds = model.pos_enc_2d(img_pixels_norm)
        pcd_pos_embeds = model.pos_enc_3d(features['pcd_points'])
        pcd_pos_embeds = model.pos_enc_3d_proj(pcd_pos_embeds)
        
        outputs = model(
            img_feats=features['img_feats'],
            pcd_feats=features['pcd_feats'],
            img_pixels=features['img_pixels'],
            pcd_points=features['pcd_points'],
            img_pos_embeds=img_pos_embeds,
            pcd_pos_embeds=pcd_pos_embeds,
        )
        
        # 转换为dict格式
        if isinstance(outputs, tuple):
            return {
                'pose_matrix': outputs[0],
                'pose_9d': outputs[1],
                'rotation_6d': outputs[2],
                'translation': outputs[3],
                'img_keypoint_heatmap': outputs[4],
                'keypoints_2d': outputs[5],
                'keypoints_3d': outputs[6],
            }
        return outputs
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        # 设置epoch（用于C2F warmup）
        if hasattr(self.criterion, 'pose_loss') and hasattr(self.criterion.pose_loss, 'set_epoch'):
            self.criterion.pose_loss.set_epoch(epoch)
        
        metrics = MetricLogger()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main_process
        )
        
        model_info = get_model_info(self.model)
        use_relative_pose = self.config.get('loss', {}).get('use_relative_pose', True)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 提取特征
                features = self._extract_features(batch)
                
                # 前向传播
                if model_info['is_v2']:
                    outputs = self._forward_v2(features, batch)
                else:
                    outputs = self._forward_v1(features, batch)
                
                # 准备GT位姿
                gt_pose = features['poses']
                
                if use_relative_pose and 'initial_pose' in batch:
                    initial_pose = batch['initial_pose'].to(self.device)
                    gt_pose = compute_relative_pose(gt_pose, initial_pose)
                
                # 计算损失
                loss_dict = self.criterion(
                    outputs=outputs,
                    gt_pose=gt_pose,
                    pcd_points=features['pcd_points'],
                    intrinsics=features['intrinsics'],
                    epoch=epoch,
                )
                
                loss = loss_dict.get('total_loss', loss_dict.get('loss'))
                
                # NaN保护：跳过产生NaN loss的batch
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.is_main_process:
                        print(f"\n⚠️ Batch {batch_idx}: loss={loss.item()}, 跳过")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                grad_norm = self.grad_clipper(self.model)
                
                self.optimizer.step()
                
                # 记录指标
                metrics.update('loss', loss.item())
                metrics.update('grad_norm', grad_norm)
                
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        metrics.update(key, value.item())
                
                # 计算位姿误差
                if 'pose_matrix' in outputs:
                    pred_pose = outputs['pose_matrix']
                elif 'stage_outputs' in outputs:
                    pred_pose = outputs['stage_outputs'][-1]['pose_matrix']
                else:
                    pred_pose = None
                
                if pred_pose is not None:
                    rot_err, trans_err = compute_pose_error(pred_pose, gt_pose)
                    metrics.update('rot_err', rot_err.mean().item())
                    metrics.update('trans_err', trans_err.mean().item())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rot': f"{metrics.get_average('rot_err'):.2f}°",
                    'trans': f"{metrics.get_average('trans_err'):.3f}m",
                })
                
                self.global_step += 1
                
                # TensorBoard日志
                if self.writer and self.global_step % self.config['training'].get('log_interval', 10) == 0:
                    for key, value in metrics.get_all_averages().items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
            except Exception as e:
                if self.is_main_process:
                    print(f"\n⚠️ Batch {batch_idx} 出错: {e}")
                continue
        
        return metrics.get_all_averages()
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        metrics = MetricLogger()
        model_info = get_model_info(self.model)
        use_relative_pose = self.config.get('loss', {}).get('use_relative_pose', True)
        
        for batch in tqdm(self.val_loader, desc="Validating", disable=not self.is_main_process):
            try:
                features = self._extract_features(batch)
                
                if model_info['is_v2']:
                    outputs = self._forward_v2(features, batch)
                else:
                    outputs = self._forward_v1(features, batch)
                
                gt_pose = features['poses']
                if use_relative_pose and 'initial_pose' in batch:
                    initial_pose = batch['initial_pose'].to(self.device)
                    gt_pose = compute_relative_pose(gt_pose, initial_pose)
                
                loss_dict = self.criterion(
                    outputs=outputs,
                    gt_pose=gt_pose,
                    pcd_points=features['pcd_points'],
                    intrinsics=features['intrinsics'],
                    epoch=epoch,
                )
                
                loss = loss_dict.get('total_loss', loss_dict.get('loss'))
                metrics.update('loss', loss.item())
                
                # 位姿误差
                if 'pose_matrix' in outputs:
                    pred_pose = outputs['pose_matrix']
                elif 'stage_outputs' in outputs:
                    pred_pose = outputs['stage_outputs'][-1]['pose_matrix']
                else:
                    continue
                
                rot_err, trans_err = compute_pose_error(pred_pose, gt_pose)
                metrics.update('rot_err', rot_err.mean().item())
                metrics.update('trans_err', trans_err.mean().item())
                
            except Exception as e:
                continue
        
        return metrics.get_all_averages()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存checkpoint"""
        if not self.is_main_process:
            return
        
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # 定期保存
        if epoch % self.config['training'].get('save_interval', 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch:04d}.pth')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def _load_checkpoint(self, path: str):
        """加载checkpoint"""
        if self.is_main_process:
            print(f"\n加载checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        model = self.model.module if self.is_distributed else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.is_main_process:
            print(f"  ✓ 从 epoch {self.epoch} 恢复")
    
    def train(self):
        """主训练循环"""
        num_epochs = self.config['training']['num_epochs']
        val_interval = self.config['training'].get('val_interval', 2)
        
        if self.is_main_process:
            print("\n" + "=" * 70)
            print("开始训练")
            print("=" * 70)
        
        for epoch in range(self.epoch + 1, num_epochs + 1):
            self.epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                print(f"\nEpoch {epoch} 训练完成:")
                print(f"  Loss: {train_metrics.get('loss', 0):.4f}")
                print(f"  旋转误差: {train_metrics.get('rot_err', 0):.2f}°")
                print(f"  平移误差: {train_metrics.get('trans_err', 0):.3f}m")
            
            # 验证
            if epoch % val_interval == 0:
                val_metrics = self.validate(epoch)
                
                if self.is_main_process:
                    print(f"\n验证结果:")
                    print(f"  Loss: {val_metrics.get('loss', 0):.4f}")
                    print(f"  旋转误差: {val_metrics.get('rot_err', 0):.2f}°")
                    print(f"  平移误差: {val_metrics.get('trans_err', 0):.3f}m")
                    
                    # TensorBoard
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'val/{key}', value, self.global_step)
                
                # 检查是否最佳
                val_loss = val_metrics.get('loss', float('inf'))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, is_best)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
        
        if self.is_main_process:
            print("\n" + "=" * 70)
            print("训练完成!")
            print(f"最佳验证Loss: {self.best_val_loss:.4f}")
            print("=" * 70)
            
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='ICPoseNet V2 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化分布式训练
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建训练器并训练
    trainer = ICPoseTrainerV2(
        config=config,
        local_rank=args.local_rank,
        resume_path=args.resume,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
