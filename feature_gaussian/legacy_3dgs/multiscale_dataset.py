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
import struct
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

        # 预扫描特征尺寸 (只读第一帧, 自动检测维度)
        sample0 = self._load_scale_features(self.frame_ids[0])
        self.fine_hw = sample0['fine_sd'].shape[1:]   # (H, W)
        self.mid_hw = sample0['mid'].shape[1:]
        self.coarse_hw = sample0['coarse'].shape[1:]
        self.fine_dim = sample0['fine_sd'].shape[0] + sample0['fine_dino'].shape[0]
        self.mid_dim = sample0['mid'].shape[0]
        self.coarse_dim = sample0['coarse'].shape[0]
        self.total_dim = self.fine_dim + self.mid_dim + self.coarse_dim

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


# ── COLMAP 辅助函数 ──────────────────────────────────────────────────────────

def _read_images_binary(path: str) -> dict:
    """读取 COLMAP images.bin，返回 {img_id: (qvec, tvec, name)} dict。"""
    images = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('I', f.read(4))[0]
            qvec = np.array(struct.unpack('4d', f.read(32)))   # w, x, y, z
            tvec = np.array(struct.unpack('3d', f.read(24)))
            _cam_id = struct.unpack('I', f.read(4))[0]
            name_bytes = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_bytes += c
            name = name_bytes.decode()
            num_pts = struct.unpack('Q', f.read(8))[0]
            f.read(num_pts * 24)
            images[img_id] = (qvec, tvec, name)
    return images


def _read_cameras_binary(path: str) -> dict:
    """读取 COLMAP cameras.bin，返回 {cam_id: params_dict}。"""
    _PARAMS_LEN = {0: 3, 1: 4, 2: 4, 3: 4, 4: 5, 5: 8, 6: 8, 7: 7, 8: 8, 9: 12}
    cameras = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack('I', f.read(4))[0]
            model = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            n_params = _PARAMS_LEN.get(model, 4)
            params = struct.unpack(f'{n_params}d', f.read(8 * n_params))
            cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) → 3×3 rotation matrix (W2C)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def _image_name_to_stem(name: str) -> str:
    """
    将 COLMAP image name 转为特征文件前缀。
    例: 'seq1/frame00001.png' → 'seq1_frame00001'
         'frame000001.png'    → 'frame000001'
    """
    no_ext = Path(name).stem          # 去扩展名
    parent = Path(name).parent.name   # 父目录名 (flat图像时为 '.')
    if parent and parent != '.':
        return f'{parent}_{no_ext}'
    return no_ext


class ColmapMultiScaleFeatureDataset(Dataset):
    """
    基于 COLMAP 模型的多尺度特征嵌入数据集。

    适用于特征文件以 ``{subdir}_{stem}_{scale}_*.pt`` 命名的场景
    （如 OldHospital: seq1_frame00001_fine_sd_64x35x61.pt）。

    Args:
        feature_dir: 压缩特征根目录（包含 fine_sd/, fine_dino/, mid/, coarse/ 子目录）。
        colmap_dir:  COLMAP 稀疏模型目录（含 images.bin, cameras.bin）。
        intrinsics:  可选固定内参；若为 None 则自动从 cameras.bin 的第一个相机读取。
        normalize_features: 是否对特征做 L2 归一化。
        max_frames:  最多使用多少帧（按 COLMAP image_id 排序后截断）。
    """

    def __init__(
        self,
        feature_dir: str,
        colmap_dir: str,
        intrinsics: dict = None,
        normalize_features: bool = True,
        max_frames: int = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        self.normalize_features = normalize_features

        colmap_dir = Path(colmap_dir)
        images_bin = colmap_dir / 'images.bin'
        cameras_bin = colmap_dir / 'cameras.bin'

        colmap_images = _read_images_binary(str(images_bin))

        if intrinsics is None and cameras_bin.exists():
            cams = _read_cameras_binary(str(cameras_bin))
            # 取所有相机焦距均值作为固定内参
            fxs = []
            for c in cams.values():
                fxs.append(c['params'][0])
            fx = float(np.mean(fxs))
            # 假设 SIMPLE_RADIAL / PINHOLE: params = (f or fx, cx, cy, ...)
            sample_cam = next(iter(cams.values()))
            cx = float(sample_cam['params'][1])
            cy = float(sample_cam['params'][2])
            img_w = int(sample_cam['width'])
            img_h = int(sample_cam['height'])
            intrinsics = {'fx': fx, 'fy': fx, 'cx': cx, 'cy': cy,
                          'width': img_w, 'height': img_h}

        self.intrinsics = intrinsics or {'fx': 960.0, 'fy': 960.0, 'cx': 960.0, 'cy': 540.0,
                                         'width': 1920, 'height': 1080}
        self.img_w = self.intrinsics.get('width', 1920)
        self.img_h = self.intrinsics.get('height', 1080)

        # 构建帧列表：只保留存在压缩特征文件的 COLMAP 图像
        fine_sd_dir = self.feature_dir / 'fine_sd'
        if not fine_sd_dir.exists():
            raise FileNotFoundError(f"找不到 fine_sd 目录: {fine_sd_dir}")

        self.stems = []          # 特征文件前缀
        self.w2c_poses = []     # W2C 4×4

        for img_id in sorted(colmap_images.keys()):
            qvec, tvec, name = colmap_images[img_id]
            stem = _image_name_to_stem(name)

            # 检查 fine_sd 特征是否存在
            matches = list(fine_sd_dir.glob(f'{stem}_fine_sd_*.pt'))
            if not matches:
                continue  # 该帧无特征，跳过

            # 构建 W2C 4×4
            R = _qvec2rotmat(qvec)          # 3×3 W2C rotation
            t = tvec.astype(np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            self.stems.append(stem)
            self.w2c_poses.append(w2c)

        if max_frames is not None:
            self.stems = self.stems[:max_frames]
            self.w2c_poses = self.w2c_poses[:max_frames]

        # 预扫描特征尺寸
        sample0 = self._load_scale_features(0)
        self.fine_hw = sample0['fine_sd'].shape[1:]
        self.mid_hw = sample0['mid'].shape[1:]
        self.coarse_hw = sample0['coarse'].shape[1:]
        self.fine_dim = 64 + 64
        self.mid_dim = 64
        self.coarse_dim = 32
        self.total_dim = self.fine_dim + self.mid_dim + self.coarse_dim

        print(f"[ColmapMultiScaleFeatureDataset] 加载 {len(self.stems)} 帧 "
              f"(COLMAP共{len(colmap_images)}张图像，{len(colmap_images)-len(self.stems)}张无特征跳过)")
        print(f"  Fine:   {self.fine_dim}d @ {self.fine_hw[1]}×{self.fine_hw[0]}")
        print(f"  Mid:    {self.mid_dim}d @ {self.mid_hw[1]}×{self.mid_hw[0]}")
        print(f"  Coarse: {self.coarse_dim}d @ {self.coarse_hw[1]}×{self.coarse_hw[0]}")
        print(f"  Intrinsics: fx={self.intrinsics['fx']:.1f} cx={self.intrinsics['cx']:.1f} cy={self.intrinsics['cy']:.1f}")

    def _load_scale_features(self, idx: int) -> dict:
        stem = self.stems[idx]
        result = {}
        scale_subdirs = {
            'fine_sd':   'fine_sd',
            'fine_dino': 'fine_dino',
            'mid':       'mid',
            'coarse':    'coarse',
        }
        for name, subdir in scale_subdirs.items():
            d = self.feature_dir / subdir
            matches = list(d.glob(f'{stem}_{subdir}_*.pt'))
            if not matches:
                raise FileNotFoundError(f"找不到帧 {idx} ({stem}) 的 {name} 特征: {d / stem}")
            result[name] = torch.load(str(matches[0]), map_location='cpu').float()
        return result

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        feats = self._load_scale_features(idx)

        if self.normalize_features:
            for key in feats:
                feats[key] = F.normalize(feats[key], p=2, dim=0)

        fine_feat = torch.cat([feats['fine_sd'], feats['fine_dino']], dim=0)
        pose = torch.tensor(self.w2c_poses[idx], dtype=torch.float32)

        return {
            'fine_feat': fine_feat,
            'mid_feat': feats['mid'],
            'coarse_feat': feats['coarse'],
            'pose': pose,
            'frame_id': idx,
            'intrinsics': self.intrinsics,
        }
