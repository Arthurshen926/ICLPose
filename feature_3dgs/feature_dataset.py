"""
Feature Embedding Dataset
=========================
加载预提取的特征图 + 相机位姿，用于特征嵌入训练。

支持两种特征格式：
  - 原始融合特征 (raw):       rgb_X_fused_768x35x46.pt        shape [1, D, H, W]
  - 压缩特征     (compressed): rgb_X_fused_*_compressed.pt     dict{'compressed': [H,W,D]}
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class FeatureEmbeddingDataset(Dataset):
    """
    用于特征嵌入训练的数据集。

    加载:
    - 特征图 (支持 768维原始 或 256维压缩，自动识别)
    - 相机位姿 (4x4 W2C 矩阵)

    注意: traj_w_c.txt 存储的实际是 C2W (camera-to-world) 矩阵,
    本dataset在加载时自动转换为 W2C (world-to-camera)，供gsplat使用。

    Args:
        feature_dir: 特征图目录
        traj_path: 位姿文件路径 (traj_w_c.txt, 每行16个float = 4x4 C2W矩阵 行优先)
        intrinsics: 相机内参 dict {fx, fy, cx, cy}
        img_size: (H, W) 图像分辨率
        normalize_features: 是否对GT特征图做L2归一化
        max_frames: 最大帧数 (None=全部)
        feature_type: 'auto'|'raw'|'compressed'
            auto: 优先查找原始特征，再查找压缩特征
            raw: 只查找 rgb_*_fused_*.pt (非compressed)
            compressed: 只查找 rgb_*_fused_*_compressed.pt
    """

    def __init__(
        self,
        feature_dir: str,
        traj_path: str,
        intrinsics: dict = None,
        img_size: tuple = (480, 640),
        normalize_features: bool = True,
        max_frames: int = None,
        feature_type: str = 'auto',
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.normalize_features = normalize_features
        self.img_size = img_size
        self.feature_type = feature_type

        # 默认内参 (Replica room_0)
        if intrinsics is None:
            intrinsics = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
        self.intrinsics = intrinsics

        # 加载位姿 (traj_w_c.txt实际存储C2W，需转换为W2C供gsplat使用)
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)  # [N_frames, 4, 4] C2W
        self.poses = np.linalg.inv(c2w_poses).astype(np.float32)  # [N_frames, 4, 4] W2C

        # 查找匹配的特征文件
        self.samples = []
        self._detected_type = None
        self._scan_features()

        # 按frame_id排序
        self.samples.sort(key=lambda x: x['frame_id'])

        if max_frames is not None:
            self.samples = self.samples[:max_frames]

        print(f"[FeatureEmbeddingDataset] 加载 {len(self.samples)} 帧  "
              f"(特征类型: {self._detected_type})")
        print(f"  特征目录: {feature_dir}")
        print(f"  位姿帧数: {len(self.poses)}")
        print(f"  内参: fx={intrinsics['fx']}, fy={intrinsics['fy']}")

    def _scan_features(self):
        """根据 feature_type 扫描并登记特征文件。"""
        use_raw        = self.feature_type in ('raw', 'auto')
        use_compressed = self.feature_type in ('compressed', 'auto')

        # ── 尝试原始特征 (优先) ──────────────────────────────────────────
        if use_raw:
            # 匹配 rgb_X_fused_<shape>.pt（不含 _compressed）
            raw_files = sorted(
                f for f in self.feature_dir.glob('rgb_*_fused_*.pt')
                if '_compressed' not in f.name
            )
            if raw_files:
                self._detected_type = 'raw'
                for fpath in raw_files:
                    match = re.search(r'rgb_(\d+)_fused_', fpath.name)
                    if match:
                        frame_id = int(match.group(1))
                        if frame_id < len(self.poses):
                            self.samples.append({
                                'frame_id': frame_id,
                                'feature_path': str(fpath),
                                'pose': self.poses[frame_id],
                                'fmt': 'raw',
                            })
                return

        # ── 回退到压缩特征 ───────────────────────────────────────────────
        if use_compressed:
            comp_files = sorted(self.feature_dir.glob('rgb_*_fused_*_compressed.pt'))
            if comp_files:
                self._detected_type = 'compressed'
                for fpath in comp_files:
                    match = re.search(r'rgb_(\d+)_fused_', fpath.name)
                    if match:
                        frame_id = int(match.group(1))
                        if frame_id < len(self.poses):
                            self.samples.append({
                                'frame_id': frame_id,
                                'feature_path': str(fpath),
                                'pose': self.poses[frame_id],
                                'fmt': 'compressed',
                            })
                return

        if not self.samples:
            raise FileNotFoundError(
                f"在 {self.feature_dir} 中未找到特征文件\n"
                f"  期望格式: rgb_X_fused_*.pt 或 rgb_X_fused_*_compressed.pt"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fmt = sample.get('fmt', self._detected_type)

        data = torch.load(sample['feature_path'], map_location='cpu')

        if fmt == 'raw':
            # 原始特征: Tensor [1, D, H, W] 或 [D, H, W]
            feat = data.float()
            if feat.dim() == 4:
                feat = feat.squeeze(0)          # [D, H, W]
        else:
            # 压缩特征: dict{'compressed': [H,W,D] 或 [D,H,W]} 或直接 Tensor
            if isinstance(data, dict):
                feat = data['compressed'].float()
            else:
                feat = data.float()
            if feat.dim() == 4:
                feat = feat.squeeze(0)
            # channels-last → channels-first
            if feat.dim() == 3 and feat.shape[-1] < feat.shape[0]:
                feat = feat.permute(2, 0, 1).contiguous()  # [H,W,D] → [D,H,W]

        # L2归一化 (per-pixel, 沿通道维度)
        if self.normalize_features:
            feat = F.normalize(feat, p=2, dim=0)

        pose = torch.tensor(sample['pose'], dtype=torch.float32)  # [4, 4] W2C

        return {
            'feature_map': feat,              # [D, fH, fW]
            'pose': pose,                     # [4, 4]
            'frame_id': sample['frame_id'],
            'intrinsics': self.intrinsics,
        }
