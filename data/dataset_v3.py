"""
DatasetV3: ICPoseNetV3 训练/验证数据集
=======================================
为迭代 Render-and-Compare 设计的数据加载器

训练模式:
  - 加载预提取的多尺度查询特征 (.pt 文件)
  - 加载 GT 位姿（c2w → 转换为 w2c）
  - 应用位姿扰动模拟初始位姿噪声
  - 加载深度图（用于 flow loss）

验证模式:
  - 同上，但使用 NetVLAD 检索的初始位姿（如果可用）
"""

import os
import glob as glob_module
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


# 默认尺度配置
SCALE_FILE_PATTERNS = {
    'coarse':    'rgb_{idx}_coarse_*.pt',
    'mid':       'rgb_{idx}_mid_*.pt',
    'fine_sd':   'rgb_{idx}_fine_sd_*.pt',
    'fine_dino': 'rgb_{idx}_fine_dino_*.pt',
}


def load_poses_c2w(traj_path: str) -> np.ndarray:
    """加载 traj_w_c.txt 中的 c2w 位姿 (每行 16 float)
    Returns: (N, 4, 4)
    """
    raw = np.loadtxt(traj_path)
    N = raw.shape[0]
    return raw.reshape(N, 4, 4)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """c2w (4, 4) → w2c (4, 4)"""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=c2w.dtype)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def perturb_pose(
    pose_w2c: torch.Tensor,
    noise_rot_deg: float = 15.0,
    noise_trans_m: float = 0.5,
) -> torch.Tensor:
    """
    对 w2c 位姿添加随机扰动，模拟初始位姿噪声
    
    Args:
        pose_w2c: (4, 4) 
        noise_rot_deg: 旋转噪声标准差 (度)
        noise_trans_m: 平移噪声标准差 (米)
    """
    from modules.lie_algebra import se3_exp
    
    # 生成 se(3) 噪声
    noise_rot_rad = noise_rot_deg * np.pi / 180.0
    xi = torch.zeros(6)
    xi[:3] = torch.randn(3) * noise_trans_m
    xi[3:] = torch.randn(3) * noise_rot_rad
    
    delta = se3_exp(xi.unsqueeze(0)).squeeze(0)  # (4, 4)
    return delta.to(pose_w2c.device) @ pose_w2c


class PoseDatasetV3(Dataset):
    """
    ICPoseNetV3 训练/验证数据集
    
    Args:
        feature_base_dir: 多尺度特征根目录  
            e.g. 'output/features_multiscale/room_0'
        traj_path: 位姿文件  
            e.g. 'dataset/room_0/Sequence_1/traj_w_c.txt'
        depth_dir: 深度图目录  
            e.g. 'dataset/room_0/Sequence_1/depth'
        frame_indices: 使用哪些帧 (None = 全部)
        scale_names: 加载哪些尺度
        depth_scale: 深度图缩放因子 (uint16 → 米)
        noise_rot_deg: 训练时旋转噪声 (度)
        noise_trans_m: 训练时平移噪声 (米)
        is_train: 是否训练模式（决定是否加扰动）
        netvlad_poses_path: NetVLAD 初始位姿文件（验证时使用）
    """
    
    def __init__(
        self,
        feature_base_dir: str,
        traj_path: str,
        depth_dir: str = None,
        frame_indices: List[int] = None,
        scale_names: List[str] = None,
        depth_scale: float = 1000.0,
        noise_rot_deg: float = 15.0,
        noise_trans_m: float = 0.5,
        is_train: bool = True,
        netvlad_poses_path: str = None,
        depth_resize: Tuple[int, int] = None,
        depth_pattern: str = 'depth_{idx}.png',
    ):
        self.feature_base_dir = feature_base_dir
        self.depth_dir = depth_dir
        self.is_train = is_train
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m
        self.depth_scale = depth_scale
        self.depth_resize = depth_resize  # (H, W) for flow loss resolution
        self.depth_pattern = depth_pattern
        
        if scale_names is None:
            scale_names = ['coarse', 'fine_sd', 'fine_dino']
        self.scale_names = scale_names
        
        # 加载位姿
        poses_c2w = load_poses_c2w(traj_path)  # (N, 4, 4)
        
        if frame_indices is not None:
            poses_c2w = poses_c2w[frame_indices]
            self.frame_indices = frame_indices
        else:
            self.frame_indices = list(range(len(poses_c2w)))
        
        # 转换为 w2c
        self.poses_w2c = np.stack([c2w_to_w2c(p) for p in poses_c2w])  # (N, 4, 4)
        
        # NetVLAD 初始位姿（验证时）
        self.netvlad_poses = None
        if netvlad_poses_path and os.path.exists(netvlad_poses_path):
            all_netvlad = np.load(netvlad_poses_path)  # (900, 4, 4) w2c
            if frame_indices is not None:
                self.netvlad_poses = all_netvlad[frame_indices]
            else:
                self.netvlad_poses = all_netvlad
            print(f"[DatasetV3] Loaded NetVLAD poses: {self.netvlad_poses.shape}")
        
        # 验证特征文件存在
        self._verify_features()
        
        print(f"[DatasetV3] {len(self)} frames, scales={self.scale_names}, "
              f"train={is_train}, noise_rot={noise_rot_deg}°, noise_trans={noise_trans_m}m")
    
    def _verify_features(self):
        """检查所有特征文件是否存在"""
        missing = 0
        for idx in self.frame_indices[:5]:
            for scale in self.scale_names:
                pattern = SCALE_FILE_PATTERNS[scale]
                glob_pattern = os.path.join(self.feature_base_dir, scale,
                                            pattern.format(idx=idx))
                matches = glob_module.glob(glob_pattern)
                if not matches:
                    print(f"  [Warning] Missing: {glob_pattern}")
                    missing += 1
        if missing > 0:
            print(f"  [Warning] {missing} feature files missing in first 5 frames!")
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, i: int) -> Dict[str, object]:
        frame_idx = self.frame_indices[i]
        
        # 1. 加载多尺度查询特征
        query_feats = {}
        for scale in self.scale_names:
            pattern = SCALE_FILE_PATTERNS[scale]
            glob_pattern = os.path.join(self.feature_base_dir, scale,
                                        pattern.format(idx=frame_idx))
            matches = glob_module.glob(glob_pattern)
            if not matches:
                raise FileNotFoundError(
                    f"找不到帧 {frame_idx} 的 {scale} 特征: {glob_pattern}")
            query_feats[scale] = torch.load(matches[0], map_location='cpu')
        
        # 2. GT 位姿 (w2c)
        pose_gt = torch.from_numpy(self.poses_w2c[i]).float()
        
        # 3. 初始位姿
        if self.is_train:
            # 训练: GT + 随机噪声
            initial_pose = perturb_pose(
                pose_gt.clone(), 
                self.noise_rot_deg, 
                self.noise_trans_m,
            )
        elif self.netvlad_poses is not None:
            # 验证: 使用 NetVLAD 检索位姿
            initial_pose = torch.from_numpy(self.netvlad_poses[i]).float()
        else:
            # 验证无 NetVLAD: 用较大噪声
            initial_pose = perturb_pose(pose_gt.clone(), 25.0, 1.0)
        
        # 4. 深度图（用于 flow loss）
        depth = None
        if self.depth_dir is not None:
            depth_path = os.path.join(
                self.depth_dir, self.depth_pattern.format(idx=frame_idx))
            if os.path.exists(depth_path):
                depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth = torch.from_numpy(
                    depth_uint16.astype(np.float32) / self.depth_scale
                )  # (H, W) in meters
                # Resize depth to model output resolution for flow loss
                if self.depth_resize is not None:
                    tH, tW = self.depth_resize
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0).unsqueeze(0),
                        size=(tH, tW), mode='nearest'
                    ).squeeze(0).squeeze(0)
        
        result = {
            'query_feats': query_feats,
            'pose_gt': pose_gt,
            'initial_pose': initial_pose,
            'frame_idx': frame_idx,
        }
        
        if depth is not None:
            result['depth'] = depth
        
        return result


def collate_v3(batch: List[Dict]) -> Dict:
    """
    自定义 collate: 将 query_feats 按尺度 stack
    """
    B = len(batch)
    
    # Stack query features per scale
    scale_names = list(batch[0]['query_feats'].keys())
    query_feats = {}
    for scale in scale_names:
        query_feats[scale] = torch.stack(
            [b['query_feats'][scale] for b in batch], dim=0
        )
    
    # Stack poses
    pose_gt = torch.stack([b['pose_gt'] for b in batch], dim=0)
    initial_pose = torch.stack([b['initial_pose'] for b in batch], dim=0)
    frame_indices = [b['frame_idx'] for b in batch]
    
    result = {
        'query_feats': query_feats,
        'pose_gt': pose_gt,
        'initial_pose': initial_pose,
        'frame_idx': frame_indices,
    }
    
    # Depth (optional)
    if 'depth' in batch[0]:
        result['depth'] = torch.stack([b['depth'] for b in batch], dim=0)
    
    return result
