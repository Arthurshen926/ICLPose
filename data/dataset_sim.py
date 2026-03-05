"""
SimPoseDataset: 仿真训练数据集
================================
使用 3DGS 仿真生成的数据训练 CorrPoseNet

与 PoseDatasetV3 的关键区别:
  - Query 特征: 来自 3DGS 渲染 RGB → DINO 提取 (或预缓存)
  - GT 位姿: 仿真采样的有效位姿 (非真实轨迹)
  - 深度: 3DGS 渲染深度 (非传感器深度)
  - 训练集大小: 可任意扩展

可与 PoseDatasetV3 的真实数据混合训练 (MixedPoseDataset)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, List, Optional, Tuple

from data.dataset_v3 import (
    PoseDatasetV3, collate_v3, perturb_pose, c2w_to_w2c
)


# 仿真数据文件命名模式
SIM_FEAT_PATTERN = 'sim_{idx:04d}_fine_dino_768x35x46.pt'
SIM_DEPTH_PATTERN = 'sim_{idx:04d}_depth.pt'


class SimPoseDataset(Dataset):
    """
    仿真训练数据集
    
    数据来源: generate_sim_training_data.py 预生成
    
    Args:
        sim_data_dir: 仿真数据根目录 (包含 fine_dino/, depth/, sim_poses_w2c.npy)
        scale_names: 使用的特征尺度 (目前仅 ['fine_dino'])
        noise_rot_deg: 初始位姿旋转噪声 (度)
        noise_trans_m: 初始位姿平移噪声 (米)
        max_samples: 最大使用样本数 (None = 全部)
    """
    
    def __init__(
        self,
        sim_data_dir: str,
        scale_names: List[str] = None,
        noise_rot_deg: float = 15.0,
        noise_trans_m: float = 0.5,
        max_samples: int = None,
    ):
        self.sim_data_dir = sim_data_dir
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m
        
        if scale_names is None:
            scale_names = ['fine_dino']
        self.scale_names = scale_names
        
        # 加载位姿
        w2c_path = os.path.join(sim_data_dir, 'sim_poses_w2c.npy')
        self.poses_w2c = np.load(w2c_path)  # (N, 4, 4)
        
        self.num_total = len(self.poses_w2c)
        if max_samples is not None:
            self.num_total = min(self.num_total, max_samples)
        
        # 验证特征文件
        self._verify()
        
        print(f"[SimPoseDataset] {self.num_total} 仿真训练样本, "
              f"scales={scale_names}, noise_rot={noise_rot_deg}°, "
              f"noise_trans={noise_trans_m}m")
    
    def _verify(self):
        """快速检查前几个文件"""
        missing = 0
        for i in range(min(5, self.num_total)):
            feat_path = os.path.join(
                self.sim_data_dir, 'fine_dino',
                SIM_FEAT_PATTERN.format(idx=i)
            )
            if not os.path.exists(feat_path):
                print(f"  [Warning] Missing: {feat_path}")
                missing += 1
        if missing > 0:
            print(f"  [Warning] {missing} feature files missing in first 5!")
    
    def __len__(self):
        return self.num_total
    
    def __getitem__(self, i: int) -> Dict[str, object]:
        # 1. Query 特征 (DINO from rendered RGB, 预缓存)
        query_feats = {}
        for scale in self.scale_names:
            if scale == 'fine_dino':
                fname = SIM_FEAT_PATTERN.format(idx=i)
            else:
                raise ValueError(f"仿真数据暂不支持尺度: {scale}")
            fpath = os.path.join(self.sim_data_dir, scale, fname)
            query_feats[scale] = torch.load(fpath, map_location='cpu', weights_only=True).float()
        
        # 2. GT 位姿 (w2c)
        pose_gt = torch.from_numpy(self.poses_w2c[i]).float()
        
        # 3. 初始位姿 = GT + 噪声
        initial_pose = perturb_pose(
            pose_gt.clone(),
            self.noise_rot_deg,
            self.noise_trans_m,
        )
        
        # 4. 深度图
        depth_path = os.path.join(
            self.sim_data_dir, 'depth', SIM_DEPTH_PATTERN.format(idx=i)
        )
        depth = None
        if os.path.exists(depth_path):
            depth = torch.load(depth_path, map_location='cpu', weights_only=True).float()
        
        result = {
            'query_feats': query_feats,
            'pose_gt': pose_gt,
            'initial_pose': initial_pose,
            'frame_idx': -(i + 1),  # 负数标记为仿真数据
        }
        if depth is not None:
            result['depth'] = depth
        
        return result


class MixedPoseDataset(Dataset):
    """
    混合数据集: 真实数据 + 仿真数据
    
    每个 epoch 按比例混合采样:
      - real_ratio 比例的真实数据 (PoseDatasetV3)
      - (1 - real_ratio) 比例的仿真数据 (SimPoseDataset)
    
    Args:
        real_dataset: PoseDatasetV3 实例
        sim_dataset: SimPoseDataset 实例
        real_ratio: 真实数据比例 (0~1, 默认 0.3)
        epoch_size: 每 epoch 总样本数 (None = real + sim)
    """
    
    def __init__(
        self,
        real_dataset: PoseDatasetV3,
        sim_dataset: SimPoseDataset,
        real_ratio: float = 0.3,
        epoch_size: int = None,
    ):
        self.real_dataset = real_dataset
        self.sim_dataset = sim_dataset
        self.real_ratio = real_ratio
        
        if epoch_size is None:
            epoch_size = len(real_dataset) + len(sim_dataset)
        self.epoch_size = epoch_size
        
        self._resample()
        
        print(f"[MixedPoseDataset] epoch_size={epoch_size}, "
              f"real_ratio={real_ratio:.0%} "
              f"({len(real_dataset)} real + {len(sim_dataset)} sim)")
    
    def _resample(self):
        """每个 epoch 开始时重新采样索引"""
        n_real = int(self.epoch_size * self.real_ratio)
        n_sim = self.epoch_size - n_real
        
        real_indices = np.random.choice(
            len(self.real_dataset), size=n_real, replace=True
        )
        sim_indices = np.random.choice(
            len(self.sim_dataset), size=n_sim, replace=True
        )
        
        # (source, index) 对, 打乱顺序
        self._samples = []
        for idx in real_indices:
            self._samples.append(('real', int(idx)))
        for idx in sim_indices:
            self._samples.append(('sim', int(idx)))
        np.random.shuffle(self._samples)
    
    def resample_epoch(self):
        """供训练循环调用, 每 epoch 重新打乱"""
        self._resample()
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, i: int) -> Dict[str, object]:
        source, idx = self._samples[i]
        if source == 'real':
            return self.real_dataset[idx]
        else:
            return self.sim_dataset[idx]
