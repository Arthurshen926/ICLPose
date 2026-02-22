"""
Raw Multi-Scale Feature Embedding Dataset
==========================================
直接加载原始未压缩的多尺度特征，用于 3DGS 特征嵌入训练。

原始特征维度:
  - fine_sd   : 640d @ 35×46  (SD s3 原始)
  - fine_dino : 768d @ 35×46  (DINO Patch 原始)
  - mid       : 1280d @ 15×20 (SD s4 原始)
  - coarse    : 1280d @ 7×10  (SD s5 原始)

支持两种加载模式:
  1. 单尺度 (--scale=fine_sd): 只加载 fine_sd [640, 35, 46]
  2. 全尺度: 加载并拼接所有尺度 (需要按最大分辨率对齐或分别处理)
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List


# ============================================================
# 原始特征层级定义 (直接从提取器输出的维度)
# ============================================================
RAW_SCALE_CONFIGS = {
    'fine_sd':   {'dim': 640,  'subdir': 'fine_sd',   'resolution': (35, 46)},
    'fine_dino': {'dim': 768,  'subdir': 'fine_dino', 'resolution': (35, 46)},
    'mid':       {'dim': 1280, 'subdir': 'mid',       'resolution': (15, 20)},
    'coarse':    {'dim': 1280, 'subdir': 'coarse',    'resolution': (7, 10)},
}


class RawScaleFeatureDataset(Dataset):
    """
    单尺度原始特征数据集。

    用于按尺度独立训练 3DGS 特征嵌入。

    目录结构:
        feature_dir/fine_sd/rgb_0_fine_sd_640x35x46.pt
        feature_dir/fine_dino/rgb_0_fine_dino_768x35x46.pt
        feature_dir/mid/rgb_0_mid_1280x15x20.pt
        feature_dir/coarse/rgb_0_coarse_1280x7x10.pt
    """

    def __init__(
        self,
        feature_dir: str,
        traj_path: str,
        scale: str = 'fine_sd',
        intrinsics: dict = None,
        img_size: tuple = (480, 640),
        normalize_features: bool = True,
        max_frames: int = None,
    ):
        """
        Args:
            feature_dir: 原始特征根目录 (e.g. output/features_multiscale/room_0)
            traj_path: 位姿文件 (C2W)
            scale: 尺度名称: 'fine_sd', 'fine_dino', 'mid', 'coarse'
            intrinsics: 相机内参, 默认 Replica room_0
            normalize_features: 是否 L2 归一化
            max_frames: 限制帧数
        """
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.normalize_features = normalize_features
        self.img_size = img_size
        self.scale = scale

        if scale not in RAW_SCALE_CONFIGS:
            raise ValueError(f"未知尺度 '{scale}', 可选: {list(RAW_SCALE_CONFIGS.keys())}")

        self.scale_cfg = RAW_SCALE_CONFIGS[scale]
        self.feat_dim = self.scale_cfg['dim']
        self.feat_hw = self.scale_cfg['resolution']  # (H, W)

        if intrinsics is None:
            intrinsics = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
        self.intrinsics = intrinsics

        # 加载位姿 (C2W → W2C)
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(c2w_poses).astype(np.float32)

        # 扫描帧 ID
        subdir = self.feature_dir / self.scale_cfg['subdir']
        if not subdir.exists():
            raise FileNotFoundError(f"找不到 {scale} 特征目录: {subdir}")

        self.frame_ids = []
        self.file_map: Dict[int, Path] = {}
        pattern = re.compile(rf'rgb_(\d+)_{self.scale_cfg["subdir"]}_')

        for fpath in sorted(subdir.glob(f'rgb_*_{self.scale_cfg["subdir"]}_*.pt')):
            match = pattern.search(fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(self.poses):
                    self.frame_ids.append(fid)
                    self.file_map[fid] = fpath

        self.frame_ids.sort()
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]

        # 验证第一帧维度
        sample = torch.load(str(self.file_map[self.frame_ids[0]]), map_location='cpu')
        actual_dim, actual_h, actual_w = sample.shape
        assert actual_dim == self.feat_dim, (
            f"维度不匹配: 期望 {self.feat_dim}, 实际 {actual_dim}")
        self.feat_hw = (actual_h, actual_w)

        print(f"[RawScaleFeatureDataset] scale={scale}")
        print(f"  特征维度: {self.feat_dim}d @ {self.feat_hw[1]}×{self.feat_hw[0]}")
        print(f"  帧数: {len(self.frame_ids)}")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        feat = torch.load(str(self.file_map[frame_id]), map_location='cpu').float()

        if self.normalize_features:
            feat = F.normalize(feat, p=2, dim=0)

        pose = torch.tensor(self.poses[frame_id], dtype=torch.float32)

        return {
            'feat': feat,              # [D, H, W]
            'pose': pose,              # [4, 4]
            'frame_id': frame_id,
            'intrinsics': self.intrinsics,
        }


class RawMultiScaleFeatureDataset(Dataset):
    """
    多尺度原始特征数据集 (所有尺度一起加载)。

    每帧返回所有 4 个尺度的原始特征。
    """

    def __init__(
        self,
        feature_dir: str,
        traj_path: str,
        intrinsics: dict = None,
        img_size: tuple = (480, 640),
        normalize_features: bool = True,
        max_frames: int = None,
        scales: List[str] = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.normalize_features = normalize_features

        if scales is None:
            scales = list(RAW_SCALE_CONFIGS.keys())
        self.scales = scales

        if intrinsics is None:
            intrinsics = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
        self.intrinsics = intrinsics

        # 位姿
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(c2w_poses).astype(np.float32)

        # 扫描帧 (以第一个尺度为基准)
        first_scale = self.scales[0]
        cfg = RAW_SCALE_CONFIGS[first_scale]
        subdir = self.feature_dir / cfg['subdir']

        self.frame_ids = []
        pattern = re.compile(rf'rgb_(\d+)_{cfg["subdir"]}_')

        for fpath in sorted(subdir.glob(f'rgb_*_{cfg["subdir"]}_*.pt')):
            match = pattern.search(fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(self.poses):
                    self.frame_ids.append(fid)

        self.frame_ids.sort()
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]

        # 收集各尺度信息
        self.scale_info = {}
        for sname in self.scales:
            scfg = RAW_SCALE_CONFIGS[sname]
            sdir = self.feature_dir / scfg['subdir']
            sample_file = list(sdir.glob(f'rgb_{self.frame_ids[0]}_{scfg["subdir"]}_*.pt'))[0]
            sample = torch.load(str(sample_file), map_location='cpu')
            self.scale_info[sname] = {
                'dim': sample.shape[0],
                'hw': (sample.shape[1], sample.shape[2]),
                'subdir': sdir,
            }

        total_dim = sum(info['dim'] for info in self.scale_info.values())
        print(f"[RawMultiScaleFeatureDataset] 加载 {len(self.frame_ids)} 帧, {len(self.scales)} 尺度")
        for sname, info in self.scale_info.items():
            print(f"  {sname}: {info['dim']}d @ {info['hw'][1]}×{info['hw'][0]}")
        print(f"  总维度: {total_dim}d")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        result = {'frame_id': frame_id, 'intrinsics': self.intrinsics}

        # 位姿
        result['pose'] = torch.tensor(self.poses[frame_id], dtype=torch.float32)

        # 各尺度特征
        for sname in self.scales:
            info = self.scale_info[sname]
            cfg = RAW_SCALE_CONFIGS[sname]
            sdir = info['subdir']
            matches = list(sdir.glob(f'rgb_{frame_id}_{cfg["subdir"]}_*.pt'))
            feat = torch.load(str(matches[0]), map_location='cpu').float()
            if self.normalize_features:
                feat = F.normalize(feat, p=2, dim=0)
            result[sname] = feat

        return result
