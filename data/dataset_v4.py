"""
DatasetV4: SD-Primary Coarse-to-Fine Flow Pose 训练/验证数据集
===============================================================
支持两种特征格式:
  v1: output/features_multiscale/room_0/{coarse,mid,fine_sd,fine_dino}/
      naming: rgb_{idx}_{scale}_{CxHxW}.pt
      resolutions: coarse 7×10, mid 15×20, fine_sd 35×46, fine_dino 35×46
  v2: output/features_v2/room_0/{sd_s5,sd_s4,sd_s3,dino}/
      naming: rgb_{idx}_{subdir}_{CxHxW}.pt
      resolutions: coarse 8×10, mid 16×20, fine_sd 32×40, fine_dino 35×46

自动检测格式 (优先 v1, 因为已有数据).
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


# v1 特征布局: subdir名 = scale名, 文件名包含 scale名
# 注: ODISE backbone 将所有 SD 特征投影到 512 维 (非原始 640/1280)
# 压缩后特征维度更低 (coarse=32, mid/fine=64), 但目录结构相同
V1_SCALE_CONFIG = {
    'coarse':    {'subdir': 'coarse',    'dim': 512},
    'mid':       {'subdir': 'mid',       'dim': 512},
    'fine_sd':   {'subdir': 'fine_sd',   'dim': 512},
    'fine_dino': {'subdir': 'fine_dino', 'dim': 768},
}

# v2 特征布局: subdir名 = SD层名/dino
V2_SCALE_CONFIG = {
    'coarse':    {'subdir': 'sd_s5',  'dim': 512},
    'mid':       {'subdir': 'sd_s4',  'dim': 512},
    'fine_sd':   {'subdir': 'sd_s3',  'dim': 512},
    'fine_dino': {'subdir': 'dino',   'dim': 768},
}

# FlowFeat 特征布局: 3 scales from DPT intermediate layers (PCA compressed)
FLOWFEAT_SCALE_CONFIG = {
    'coarse': {'subdir': 'coarse', 'dim': 32},
    'mid':    {'subdir': 'mid',    'dim': 64},
    'fine':   {'subdir': 'fine',   'dim': 64},
}


def _orthogonalize_rotations(poses: np.ndarray) -> np.ndarray:
    """SVD-orthogonalize rotation matrices to ensure det(R)=1, R@R^T=I.
    Some datasets (e.g. 7-Scenes stairs) have slightly non-orthogonal rotations
    (det≈0.9998) which causes ~40% pose error inflation and solver accuracy floors.
    """
    fixed = poses.copy()
    for i in range(len(poses)):
        R = poses[i, :3, :3]
        U, _, Vh = np.linalg.svd(R)
        R_orth = U @ Vh
        if np.linalg.det(R_orth) < 0:
            R_orth = U @ np.diag([1, 1, -1]) @ Vh
        fixed[i, :3, :3] = R_orth
    return fixed


def load_poses_c2w(traj_path: str) -> np.ndarray:
    """加载 traj_w_c.txt 中的 c2w 位姿, Returns: (N, 4, 4)"""
    raw = np.loadtxt(traj_path)
    if raw.size % 16 != 0:
        raise ValueError(f"Pose file {traj_path} has {raw.size} elements, not divisible by 16")
    poses = raw.reshape(-1, 4, 4)
    poses = _orthogonalize_rotations(poses)
    return poses


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
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
    """对 w2c 位姿添加随机扰动"""
    from modules.lie_algebra import se3_exp
    noise_rot_rad = noise_rot_deg * np.pi / 180.0
    xi = torch.zeros(6)
    xi[:3] = torch.randn(3) * noise_trans_m
    xi[3:] = torch.randn(3) * noise_rot_rad
    delta = se3_exp(xi.unsqueeze(0)).squeeze(0)
    return delta.to(pose_w2c.device) @ pose_w2c


class PoseDatasetV4(Dataset):
    """
    SD-Primary Coarse-to-Fine Flow Pose 数据集

    每帧返回:
      - 4 个尺度的查询特征 (coarse/mid/fine_sd/fine_dino)
      - GT w2c 位姿
      - 带扰动的初始位姿
      - 深度图 (35×46, 匹配 fine_dino 分辨率, 用于 Image Jacobian)

    自动检测 v1/v2 特征格式, 通过目录结构判断.

    Args:
        feature_base_dir: 特征根目录
        traj_path: 位姿文件
        depth_dir: 深度图目录
        frame_indices: 使用哪些帧
        scale_names: 加载哪些尺度
        noise_rot_deg: 旋转噪声 (度)
        noise_trans_m: 平移噪声 (米)
        is_train: 训练模式
        netvlad_poses_path: NetVLAD 检索位姿 (.npy)
    """

    # 最终 flow 输出所在的分辨率 (fine_dino)
    FLOW_RESOLUTION = (35, 46)

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
        flow_resolution: Tuple[int, int] = None,
        cache_in_memory: bool = False,
    ):
        self.feature_base_dir = feature_base_dir
        self.depth_dir = depth_dir
        self.is_train = is_train
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m
        self.depth_scale = depth_scale
        if flow_resolution is not None:
            self.FLOW_RESOLUTION = tuple(flow_resolution)

        # 自动检测 v1/v2/flowfeat 格式
        self.scale_config = self._detect_format(feature_base_dir)

        if scale_names is None:
            if self.scale_config is FLOWFEAT_SCALE_CONFIG:
                scale_names = ['coarse', 'mid', 'fine']
            else:
                scale_names = ['coarse', 'mid', 'fine_sd', 'fine_dino']
        self.scale_names = scale_names

        # 加载位姿
        poses_c2w = load_poses_c2w(traj_path)
        if frame_indices is not None:
            poses_c2w = poses_c2w[frame_indices]
            self.frame_indices = frame_indices
        else:
            self.frame_indices = list(range(len(poses_c2w)))
        self.poses_w2c = np.stack(
            [c2w_to_w2c(p.astype(np.float32)) for p in poses_c2w]
        )

        # NetVLAD 初始位姿
        self.netvlad_poses = None
        if netvlad_poses_path and os.path.exists(netvlad_poses_path):
            all_netvlad = np.load(netvlad_poses_path)
            if frame_indices is not None:
                self.netvlad_poses = all_netvlad[frame_indices]
            else:
                self.netvlad_poses = all_netvlad
            print(f"[DatasetV4] Loaded NetVLAD poses: {self.netvlad_poses.shape}")

        # 扫描并缓存特征文件映射
        self.file_maps = {}  # {scale: {frame_idx: path}}
        for scale in self.scale_names:
            cfg = self.scale_config[scale]
            subdir_path = os.path.join(feature_base_dir, cfg['subdir'])
            if not os.path.isdir(subdir_path):
                raise FileNotFoundError(
                    f"特征目录不存在: {subdir_path}, scale={scale}")
            file_map = {}
            # 通用 pattern: rgb_{idx}_{anything}_{shape}.pt
            for fname in sorted(os.listdir(subdir_path)):
                if not fname.endswith('.pt'):
                    continue
                # 提取帧号: rgb_{idx}_{...}.pt
                m = re.match(r'rgb_(\d+)_', fname)
                if m:
                    fid = int(m.group(1))
                    file_map[fid] = os.path.join(subdir_path, fname)
            self.file_maps[scale] = file_map

        # 验证
        self._verify_features()

        # 特征缓存: 预加载所有 .pt 到内存, 消除训练时磁盘 I/O
        self._feature_cache = None
        if cache_in_memory:
            self._feature_cache = {}
            total_bytes = 0
            for scale in self.scale_names:
                self._feature_cache[scale] = {}
                for fid, fpath in self.file_maps[scale].items():
                    t = torch.load(fpath, map_location='cpu', weights_only=True)
                    self._feature_cache[scale][fid] = t
                    total_bytes += t.nelement() * t.element_size()
            print(f"[DatasetV4] Cached {len(self.frame_indices)} frames "
                  f"× {len(self.scale_names)} scales = "
                  f"{total_bytes / 1024**2:.0f} MB in CPU memory")

        fmt = "flowfeat" if self.scale_config is FLOWFEAT_SCALE_CONFIG else \
              "v1" if self.scale_config is V1_SCALE_CONFIG else "v2"
        cache_str = ", cached" if cache_in_memory else ""
        print(f"[DatasetV4] {len(self)} frames, {fmt} format, "
              f"scales={self.scale_names}, train={is_train}, "
              f"noise=({noise_rot_deg}°, {noise_trans_m}m){cache_str}")

    @staticmethod
    def _detect_format(feature_base_dir: str) -> Dict:
        """自动检测 v1/v2/flowfeat 特征布局."""
        # flowfeat: 有 fine/ 子目录 (而非 fine_sd/ 或 fine_dino/)
        has_fine = os.path.isdir(os.path.join(feature_base_dir, 'fine'))
        has_fine_sd = os.path.isdir(os.path.join(feature_base_dir, 'fine_sd'))
        if has_fine and not has_fine_sd:
            return FLOWFEAT_SCALE_CONFIG
        # v1: 有 coarse/ 子目录
        if os.path.isdir(os.path.join(feature_base_dir, 'coarse')):
            return V1_SCALE_CONFIG
        # v2: 有 sd_s5/ 子目录
        if os.path.isdir(os.path.join(feature_base_dir, 'sd_s5')):
            return V2_SCALE_CONFIG
        raise FileNotFoundError(
            f"无法检测特征格式: {feature_base_dir}, "
            f"需要 coarse/ (v1) 或 sd_s5/ (v2) 或 fine/ (flowfeat) 子目录")

    def _verify_features(self):
        missing = 0
        for idx in self.frame_indices[:3]:
            for scale in self.scale_names:
                if idx not in self.file_maps[scale]:
                    print(f"  [Warning] Missing {scale} frame {idx}")
                    missing += 1
        if missing > 0:
            print(f"  [Warning] {missing} files missing in first 3 frames!")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, i: int) -> Dict[str, object]:
        frame_idx = self.frame_indices[i]

        # 1. 查询特征
        query_feats = {}
        for scale in self.scale_names:
            if self._feature_cache is not None:
                query_feats[scale] = self._feature_cache[scale][frame_idx]
            else:
                fpath = self.file_maps[scale].get(frame_idx)
                if fpath is None:
                    raise FileNotFoundError(
                        f"特征不存在: scale={scale}, frame={frame_idx}")
                query_feats[scale] = torch.load(
                    fpath, map_location='cpu', weights_only=True
                )

        # 2. GT w2c 位姿
        pose_gt = torch.from_numpy(self.poses_w2c[i]).float()

        # 3. 初始位姿
        if self.is_train:
            initial_pose = perturb_pose(
                pose_gt.clone(), self.noise_rot_deg, self.noise_trans_m)
        elif self.netvlad_poses is not None:
            initial_pose = torch.from_numpy(self.netvlad_poses[i]).float()
        else:
            initial_pose = perturb_pose(pose_gt.clone(), 25.0, 1.0)

        # 4. 深度图 (35×46, 用于 Image Jacobian)
        depth = None
        if self.depth_dir is not None:
            depth_path = os.path.join(self.depth_dir, f'depth_{frame_idx}.png')
            if os.path.exists(depth_path):
                depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_uint16 is None:
                    pass  # leave depth = None
                else:
                    if depth_uint16.ndim == 3:
                        depth_uint16 = depth_uint16[..., 0]
                    depth = torch.from_numpy(
                        depth_uint16.astype(np.float32) / self.depth_scale
                    )
                    tH, tW = self.FLOW_RESOLUTION
                    depth = F.interpolate(
                        depth.unsqueeze(0).unsqueeze(0),
                        size=(tH, tW), mode='nearest',
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


def collate_v4(batch: List[Dict]) -> Dict:
    """自定义 collate: 按尺度 stack 查询特征, stack 位姿和深度."""
    scale_names = list(batch[0]['query_feats'].keys())
    query_feats = {}
    for scale in scale_names:
        query_feats[scale] = torch.stack(
            [b['query_feats'][scale] for b in batch], dim=0
        )

    result = {
        'query_feats': query_feats,
        'pose_gt': torch.stack([b['pose_gt'] for b in batch], dim=0),
        'initial_pose': torch.stack([b['initial_pose'] for b in batch], dim=0),
        'frame_idx': [b['frame_idx'] for b in batch],
    }

    if 'depth' in batch[0] and batch[0]['depth'] is not None:
        result['depth'] = torch.stack([b['depth'] for b in batch], dim=0)

    return result
