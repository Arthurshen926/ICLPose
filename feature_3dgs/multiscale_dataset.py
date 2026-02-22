"""
Multi-Scale Feature Embedding Dataset
======================================
加载多尺度压缩特征 + 相机位姿，用于多尺度 3DGS 特征嵌入训练。

特征层级:
  - fine_sd   : 64d @ 35×46  (SD s3 压缩)
  - fine_dino : 64d @ 35×46  (DINO Patch 压缩)
  - mid       : 64d @ 15×20  (SD s4 压缩)
  - coarse    : 32d @ 7×10   (SD s5 压缩)

每帧返回:
  - 拼接后的 fine 特征: [128, 35, 46] (fine_sd + fine_dino)
  - mid 特征:   [64, 15, 20]
  - coarse 特征: [32, 7, 10]
  - W2C 位姿:    [4, 4]
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


# 多尺度特征层级定义
SCALE_CONFIGS = {
    'fine_sd':   {'dim': 64, 'subdir': 'fine_sd',   'pattern': 'rgb_{fid}_fine_sd_*.pt'},
    'fine_dino': {'dim': 64, 'subdir': 'fine_dino', 'pattern': 'rgb_{fid}_fine_dino_*.pt'},
    'mid':       {'dim': 64, 'subdir': 'mid',       'pattern': 'rgb_{fid}_mid_*.pt'},
    'coarse':    {'dim': 32, 'subdir': 'coarse',    'pattern': 'rgb_{fid}_coarse_*.pt'},
}


class MultiScaleFeatureDataset(Dataset):
    """
    多尺度特征嵌入训练数据集。

    目录结构 (feature_dir):
        fine_sd/rgb_0_fine_sd_64x35x46.pt
        fine_dino/rgb_0_fine_dino_64x35x46.pt
        mid/rgb_0_mid_64x15x20.pt
        coarse/rgb_0_coarse_32x7x10.pt
    """

    def __init__(
        self,
        feature_dir: str,
        traj_path: str,
        intrinsics: dict = None,
        img_size: tuple = (480, 640),
        normalize_features: bool = True,
        max_frames: int = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.normalize_features = normalize_features
        self.img_size = img_size

        if intrinsics is None:
            intrinsics = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
        self.intrinsics = intrinsics

        # 加载位姿 (C2W → W2C)
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)
        self.poses = np.linalg.inv(c2w_poses).astype(np.float32)

        # 扫描所有帧 ID (以 fine_sd 为基准)
        self.frame_ids = []
        fine_sd_dir = self.feature_dir / 'fine_sd'
        if not fine_sd_dir.exists():
            raise FileNotFoundError(f"找不到 fine_sd 目录: {fine_sd_dir}")

        for fpath in sorted(fine_sd_dir.glob('rgb_*_fine_sd_*.pt')):
            match = re.search(r'rgb_(\d+)_fine_sd_', fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(self.poses):
                    self.frame_ids.append(fid)

        self.frame_ids.sort()
        if max_frames is not None:
            self.frame_ids = self.frame_ids[:max_frames]

        # 预扫描特征尺寸 (只读第一帧)
        sample0 = self._load_scale_features(self.frame_ids[0])
        self.fine_hw = sample0['fine_sd'].shape[1:]   # (35, 46)
        self.mid_hw = sample0['mid'].shape[1:]         # (15, 20)
        self.coarse_hw = sample0['coarse'].shape[1:]   # (7, 10)
        self.fine_dim = 64 + 64   # fine_sd + fine_dino
        self.mid_dim = 64
        self.coarse_dim = 32
        self.total_dim = self.fine_dim + self.mid_dim + self.coarse_dim  # 224

        print(f"[MultiScaleFeatureDataset] 加载 {len(self.frame_ids)} 帧")
        print(f"  Fine:   {self.fine_dim}d @ {self.fine_hw[1]}×{self.fine_hw[0]}")
        print(f"  Mid:    {self.mid_dim}d @ {self.mid_hw[1]}×{self.mid_hw[0]}")
        print(f"  Coarse: {self.coarse_dim}d @ {self.coarse_hw[1]}×{self.coarse_hw[0]}")
        print(f"  Total embed dim: {self.total_dim}")

    def _load_scale_features(self, frame_id: int) -> dict:
        """加载单帧的所有尺度特征"""
        result = {}
        for name, cfg in SCALE_CONFIGS.items():
            subdir = self.feature_dir / cfg['subdir']
            # 匹配文件 (glob pattern)
            pattern = f"rgb_{frame_id}_{cfg['subdir']}_*.pt"
            matches = list(subdir.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"找不到帧 {frame_id} 的 {name} 特征: {subdir / pattern}")
            result[name] = torch.load(str(matches[0]), map_location='cpu').float()
        return result

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        feats = self._load_scale_features(frame_id)

        # L2 归一化 (per-pixel, 沿通道维度)
        if self.normalize_features:
            for key in feats:
                feats[key] = F.normalize(feats[key], p=2, dim=0)

        # 拼接 fine 层: [128, fH, fW]
        fine_feat = torch.cat([feats['fine_sd'], feats['fine_dino']], dim=0)

        pose = torch.tensor(self.poses[frame_id], dtype=torch.float32)

        return {
            'fine_feat': fine_feat,           # [128, 35, 46]
            'mid_feat': feats['mid'],         # [64, 15, 20]
            'coarse_feat': feats['coarse'],   # [32, 7, 10]
            'pose': pose,                     # [4, 4]
            'frame_id': frame_id,
            'intrinsics': self.intrinsics,
        }
