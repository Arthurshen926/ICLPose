"""
RadioLocDataset: Dataset for RADIO-DCFF Localization Training
==============================================================
Loads pre-cached RADIO features (fine_geo + coarse_sem), camera poses from
COLMAP, and depth maps for training the RadioLocNet.

Data layout expected::

    feature_dir/
        fine_geo/
            {image_stem}.pt      # (64, H, W) RADIO shallow PCA features
        coarse_sem/
            {image_stem}.pt      # (64, H, W) RADIO deep PCA features

    colmap_dir/
        cameras.bin              # COLMAP cameras
        images.bin               # COLMAP image extrinsics

    depth_dir/
        {image_stem}.pt          # (H, W) rendered depth from 2DGS

Split files (dataset_train.txt / dataset_test.txt) list image filenames,
one per line.
"""

import os
import struct
import collections
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  COLMAP Binary Readers
# ═══════════════════════════════════════════════════════════════════════════════

CameraModel = collections.namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])

CAMERA_MODELS = {
    0: CameraModel(0, 'SIMPLE_PINHOLE', 3),
    1: CameraModel(1, 'PINHOLE', 4),
    2: CameraModel(2, 'SIMPLE_RADIAL', 4),
    3: CameraModel(3, 'RADIAL', 5),
    4: CameraModel(4, 'OPENCV', 8),
    5: CameraModel(5, 'OPENCV_FISHEYE', 8),
    6: CameraModel(6, 'FULL_OPENCV', 12),
    7: CameraModel(7, 'FOV', 5),
    8: CameraModel(8, 'SIMPLE_RADIAL_FISHEYE', 4),
    9: CameraModel(9, 'RADIAL_FISHEYE', 5),
    10: CameraModel(10, 'THIN_PRISM_FISHEYE', 12),
}

Camera = collections.namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])
ImageMeta = collections.namedtuple('ImageMeta', ['id', 'qvec', 'tvec', 'camera_id', 'name'])


def read_colmap_cameras(path: str) -> Dict[int, Camera]:
    """Read cameras.bin → dict of camera_id → Camera."""
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('i', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = np.array(struct.unpack(f'{num_params}d', f.read(8 * num_params)))
            cameras[cam_id] = Camera(
                cam_id, CAMERA_MODELS[model_id].model_name, width, height, params,
            )
    return cameras


def read_colmap_images(path: str) -> Dict[int, ImageMeta]:
    """Read images.bin → dict of image_id → ImageMeta(qvec, tvec, camera_id, name)."""
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack('I', f.read(4))[0]
            qvec = np.array(struct.unpack('4d', f.read(32)))
            tvec = np.array(struct.unpack('3d', f.read(24)))
            cam_id = struct.unpack('I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack('Q', f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D point data
            images[img_id] = ImageMeta(img_id, qvec, tvec, cam_id, name)
    return images


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])
    return R


def colmap_to_w2c(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP (qvec, tvec) to 4×4 world-to-camera matrix."""
    R = qvec_to_rotmat(qvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def add_pose_noise(
    pose_w2c: np.ndarray,
    rot_deg: float,
    trans_m: float,
) -> np.ndarray:
    """Add random camera-centre/rotation perturbation to a w2c pose.

    Rotation noise: random axis-angle with magnitude ~ N(0, rot_deg) degrees.
    Translation noise: isotropic Gaussian camera-centre offset in world metres.

    COLMAP stores ``w2c`` as ``X_cam = R X_world + t`` with camera centre
    ``C = -R^T t``.  Adding noise directly to ``t`` makes the measured
    camera-centre error depend on the scene coordinate magnitude and couples a
    small rotation perturbation into a large apparent translation error.  Keep
    the noise in the metric space used by evaluation instead: rotate the camera
    orientation, perturb ``C`` in world coordinates, then recompute ``t``.

    Args:
        pose_w2c: (4, 4) world-to-camera matrix
        rot_deg: rotation noise standard deviation in degrees
        trans_m: translation noise standard deviation in meters

    Returns:
        (4, 4) noisy w2c pose
    """
    pose = np.asarray(pose_w2c, dtype=np.float64)

    # Rotation noise via axis-angle
    angle = np.random.randn() * (rot_deg * np.pi / 180.0)
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    R_gt = pose[:3, :3]
    t_gt = pose[:3, 3]
    C_gt = -(R_gt.T @ t_gt)

    R_noisy = R_noise @ R_gt
    C_noisy = C_gt + np.random.randn(3) * trans_m

    noisy = pose.copy()
    noisy[:3, :3] = R_noisy
    noisy[:3, 3] = -(R_noisy @ C_noisy)

    return noisy.astype(pose_w2c.dtype, copy=False)


def camera_params_to_intrinsics(
    cam: Camera,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """Extract {fx, fy, cx, cy} from a COLMAP Camera, optionally rescaled.

    Supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, and RADIAL models.
    """
    model = cam.model
    params = cam.params

    if model == 'PINHOLE':
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model == 'SIMPLE_PINHOLE':
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in ('SIMPLE_RADIAL', 'RADIAL'):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        # Fallback: assume first 4 params are [fx, fy, cx, cy]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]

    intr = {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)}

    if target_hw is not None:
        tH, tW = target_hw
        sx = tW / cam.width
        sy = tH / cam.height
        intr = {
            'fx': intr['fx'] * sx,
            'fy': intr['fy'] * sy,
            'cx': intr['cx'] * sx,
            'cy': intr['cy'] * sy,
        }

    return intr


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class RadioLocDataset(Dataset):
    """Dataset for training the RADIO-DCFF localization network.

    Loads:
      - Pre-cached RADIO features (fine_geo + coarse_sem) for query images
      - Camera poses from COLMAP sparse model
      - Depth maps (pre-rendered from 2DGS at GT pose)

    For training, adds noise to GT pose as the initial pose estimate.

    Args:
        feature_dir: root dir containing fine_geo/ and coarse_sem/ subdirs
        teacher_feature_dir: optional secondary feature root for teacher-query curriculum
        colmap_dir:  path to COLMAP sparse model (cameras.bin, images.bin)
        depth_dir:   path to pre-rendered depth maps (*.pt files)
        split:       'train' or 'test'
        split_file:  path to split file listing image names, one per line.
                     If None, uses all images.
        noise_rot_deg:  rotation noise std in degrees (training only)
        noise_trans_m:  translation noise std in meters (training only)
        coarse_hw:  target coarse resolution (H, W)
        fine_hw:    target fine resolution (H, W)
        cache_in_memory: whether to cache all features in RAM
    """

    @staticmethod
    def _normalize_image_name(name: str) -> str:
        return str(name).replace('\\', '/')

    @classmethod
    def _discover_feature_index_by_name(
        cls,
        feature_dir: str,
        source_dir: Optional[str],
        colmap_dir: Optional[str],
    ) -> Dict[str, int]:
        feature_root = Path(feature_dir)

        export_index_path = feature_root / 'export_index.json'
        if export_index_path.is_file():
            export_index = json.loads(export_index_path.read_text(encoding='utf-8'))
            mapping: Dict[str, int] = {}
            for entry in export_index:
                teacher_idx = int(entry['teacher_idx'])
                sample_name = cls._normalize_image_name(entry['sample_name'])
                keys = {sample_name, os.path.basename(sample_name)}
                if sample_name.startswith('images/'):
                    keys.add(sample_name[len('images/'):])
                for key in keys:
                    mapping.setdefault(key, teacher_idx)
            return mapping

        candidate_roots: List[Path] = []
        if source_dir:
            candidate_roots.append(Path(source_dir))
        if colmap_dir:
            colmap_path = Path(colmap_dir)
            candidate_roots.extend([
                colmap_path,
                colmap_path.parent,
                colmap_path.parent.parent,
            ])

        seen_roots = set()
        discovery_patterns = [
            'seq*/*.png',
            'seq*/*.jpg',
            'images/*.png',
            'images/*.jpg',
            '*.png',
            '*.jpg',
        ]

        for root in candidate_roots:
            try:
                root = root.resolve()
            except FileNotFoundError:
                continue
            if root in seen_roots or not root.is_dir():
                continue
            seen_roots.add(root)

            image_paths = []
            for pattern in discovery_patterns:
                image_paths = sorted(root.glob(pattern))
                if image_paths:
                    break
            if not image_paths:
                continue

            mapping: Dict[str, int] = {}
            for idx, image_path in enumerate(image_paths):
                rel_name = cls._normalize_image_name(image_path.relative_to(root).as_posix())
                keys = {rel_name, image_path.name}
                if rel_name.startswith('images/'):
                    keys.add(rel_name[len('images/'):])
                for key in keys:
                    mapping.setdefault(key, idx)
            return mapping

        return {}

    @classmethod
    def _build_feature_source(
        cls,
        feature_dir: str,
        source_dir: Optional[str] = None,
        colmap_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        fine_geo_dir = os.path.join(feature_dir, 'fine_geo')
        coarse_sem_dir = os.path.join(feature_dir, 'coarse_sem')
        if not os.path.isdir(fine_geo_dir) or not os.path.isdir(coarse_sem_dir):
            raise RuntimeError(
                f"Expected fine_geo/ and coarse_sem/ under {feature_dir}"
            )

        sample_files = sorted(os.listdir(fine_geo_dir))
        sample_file = sample_files[0] if sample_files else ''
        name_to_feature_idx = cls._discover_feature_index_by_name(
            feature_dir,
            source_dir=source_dir,
            colmap_dir=colmap_dir,
        )
        return {
            'fine_geo_dir': fine_geo_dir,
            'coarse_sem_dir': coarse_sem_dir,
            'use_colmap_id_naming': sample_file.startswith('rgb_'),
            'name_to_feature_idx': name_to_feature_idx,
        }

    @staticmethod
    def _resolve_feature_paths(
        source: Dict[str, object],
        img_id: int,
        image_name: str,
    ) -> Optional[Tuple[str, str]]:
        if source['use_colmap_id_naming']:
            import glob as glob_mod

            normalized_name = RadioLocDataset._normalize_image_name(image_name)
            candidate_indices = []
            name_to_feature_idx = source.get('name_to_feature_idx') or {}
            for key in [normalized_name, os.path.basename(normalized_name)]:
                feature_idx = name_to_feature_idx.get(key)
                if feature_idx is not None and feature_idx not in candidate_indices:
                    candidate_indices.append(int(feature_idx))
            if img_id not in candidate_indices:
                candidate_indices.append(int(img_id))

            for feature_idx in candidate_indices:
                fine_pattern = os.path.join(source['fine_geo_dir'], f'rgb_{feature_idx}_fine_geo_*.pt')
                coarse_pattern = os.path.join(source['coarse_sem_dir'], f'rgb_{feature_idx}_coarse_sem_*.pt')
                fine_matches = glob_mod.glob(fine_pattern)
                coarse_matches = glob_mod.glob(coarse_pattern)
                if fine_matches and coarse_matches:
                    return fine_matches[0], coarse_matches[0]
            return None

        stem = os.path.splitext(os.path.basename(image_name))[0]
        fine_path = os.path.join(source['fine_geo_dir'], f'{stem}.pt')
        coarse_path = os.path.join(source['coarse_sem_dir'], f'{stem}.pt')
        if not os.path.isfile(fine_path) or not os.path.isfile(coarse_path):
            return None
        return fine_path, coarse_path

    def __init__(
        self,
        feature_dir: str,
        colmap_dir: str,
        teacher_feature_dir: Optional[str] = None,
        source_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        split: str = 'train',
        split_file: Optional[str] = None,
        noise_rot_deg: float = 5.0,
        noise_trans_m: float = 0.2,
        coarse_hw: Tuple[int, int] = (17, 30),
        fine_hw: Tuple[int, int] = (68, 120),
        cache_in_memory: bool = True,
        normalize_features: bool = False,
    ):
        super().__init__()
        self.feature_dir = feature_dir
        self.teacher_feature_dir = teacher_feature_dir
        self.source_dir = source_dir
        self.depth_dir = depth_dir
        self.split = split
        self.noise_rot_deg = noise_rot_deg
        self.noise_trans_m = noise_trans_m
        self.coarse_hw = tuple(coarse_hw)
        self.fine_hw = tuple(fine_hw)
        self.cache_in_memory = cache_in_memory
        self.normalize_features = normalize_features
        self.norm_stats = None

        # Load normalization stats if requested
        if normalize_features:
            stats_path = os.path.join(feature_dir, 'normalization_stats.pt')
            if os.path.exists(stats_path):
                self.norm_stats = torch.load(stats_path)
                logging.info(f"Loaded feature normalization stats from {stats_path}")
            else:
                logging.warning(f"normalize_features=True but no stats at {stats_path}")

        # Load COLMAP model
        cameras = read_colmap_cameras(os.path.join(colmap_dir, 'cameras.bin'))
        images = read_colmap_images(os.path.join(colmap_dir, 'images.bin'))

        # Determine image list from split file or use all
        # Cambridge Landmarks split files have format:
        #   Visual Landmark Dataset V1
        #   ImageFile, Camera Position [X Y Z W P Q R]
        #   (empty line)
        #   seq9/frame00001.png ...
        split_names = None
        if split_file is not None and os.path.isfile(split_file):
            split_names = set()
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                        continue
                    # First token is the image path (e.g., seq9/frame00001.png)
                    parts = line.split()
                    if len(parts) >= 1:
                        img_name = parts[0]
                        # Normalize .jpg/.png extension
                        base = os.path.splitext(img_name)[0]
                        split_names.add(base + '.png')
                        split_names.add(base + '.jpg')

        feature_source = self._build_feature_source(
            feature_dir,
            source_dir=source_dir,
            colmap_dir=colmap_dir,
        )
        teacher_feature_source = (
            self._build_feature_source(
                teacher_feature_dir,
                source_dir=source_dir,
                colmap_dir=colmap_dir,
            )
            if teacher_feature_dir is not None
            else None
        )

        # Build ordered sample list
        self.samples: List[Dict] = []
        self.intrinsics: Optional[Dict[str, float]] = None

        for img_id in sorted(images.keys()):
            meta = images[img_id]

            # Filter by split file
            if split_names is not None and meta.name not in split_names:
                continue

            # Find feature files by COLMAP image_id or image name
            feature_paths = self._resolve_feature_paths(feature_source, img_id, meta.name)
            if feature_paths is None:
                continue
            fine_path, coarse_path = feature_paths

            teacher_fine_path = None
            teacher_coarse_path = None
            if teacher_feature_source is not None:
                teacher_paths = self._resolve_feature_paths(
                    teacher_feature_source, img_id, meta.name,
                )
                if teacher_paths is None:
                    continue
                teacher_fine_path, teacher_coarse_path = teacher_paths

            # Depth path (optional) — indexed by COLMAP image_id
            depth_path = None
            if depth_dir is not None:
                dp = os.path.join(depth_dir, f'depth_{img_id}.pt')
                if os.path.isfile(dp):
                    depth_path = dp

            # Pose: COLMAP gives w2c
            pose_w2c = colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32)

            # Intrinsics (from first camera seen)
            cam = cameras[meta.camera_id]
            if self.intrinsics is None:
                self.intrinsics = camera_params_to_intrinsics(cam)

            self.samples.append({
                'img_id': img_id,
                'image_name': meta.name,
                'fine_path': fine_path,
                'coarse_path': coarse_path,
                'teacher_fine_path': teacher_fine_path,
                'teacher_coarse_path': teacher_coarse_path,
                'depth_path': depth_path,
                'pose_w2c': pose_w2c,
                'camera_id': meta.camera_id,
            })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {feature_dir} "
                f"(colmap_dir={colmap_dir}, split_file={split_file})"
            )

        # Pre-load features into memory
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        if cache_in_memory:
            print(f"[RadioLocDataset] Caching {len(self.samples)} samples "
                  f"({split}) into memory...")
            for idx in range(len(self.samples)):
                self._cache[idx] = self._load_features(idx)

        print(f"[RadioLocDataset] split={split}, samples={len(self.samples)}, "
              f"fine_hw={self.fine_hw}, coarse_hw={self.coarse_hw}, "
              f"noise={self.noise_rot_deg}°/{self.noise_trans_m}m")

    def _load_features(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load fine and coarse features for a single sample."""
        s = self.samples[idx]
        fine_feat = torch.load(s['fine_path'], map_location='cpu', weights_only=True)
        coarse_feat = torch.load(s['coarse_path'], map_location='cpu', weights_only=True)
        result = {'fine': fine_feat, 'coarse': coarse_feat}
        if s.get('teacher_fine_path') is not None and s.get('teacher_coarse_path') is not None:
            result['teacher_fine'] = torch.load(
                s['teacher_fine_path'], map_location='cpu', weights_only=True,
            )
            result['teacher_coarse'] = torch.load(
                s['teacher_coarse_path'], map_location='cpu', weights_only=True,
            )
        if s['depth_path'] is not None:
            result['depth'] = torch.load(
                s['depth_path'], map_location='cpu', weights_only=True,
            )
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Load or retrieve cached features
        if idx in self._cache:
            cached = self._cache[idx]
        else:
            cached = self._load_features(idx)

        fine_feat = cached['fine'].clone().float()
        coarse_feat = cached['coarse'].clone().float()

        # Apply per-channel normalization to match DCFF training targets
        if self.normalize_features and self.norm_stats is not None:
            fm = self.norm_stats['fine_mean'].view(-1, 1, 1)
            fs = self.norm_stats['fine_std'].view(-1, 1, 1).clamp(min=1e-6)
            fine_feat = (fine_feat - fm) / fs
            cm = self.norm_stats['coarse_mean'].view(-1, 1, 1)
            cs = self.norm_stats['coarse_std'].view(-1, 1, 1).clamp(min=1e-6)
            coarse_feat = (coarse_feat - cm) / cs

        # Ensure 3D: (C, H, W)
        if fine_feat.ndim == 2:
            raise ValueError(
                f"Expected 3D tensor (C,H,W) for fine features, got {fine_feat.shape}"
            )

        # Resize to target resolutions
        fine_feat = F.interpolate(
            fine_feat.unsqueeze(0), self.fine_hw, mode='bilinear', align_corners=False,
        ).squeeze(0)
        coarse_feat = F.interpolate(
            coarse_feat.unsqueeze(0), self.coarse_hw, mode='bilinear', align_corners=False,
        ).squeeze(0)

        # GT pose (w2c)
        pose_gt = torch.tensor(s['pose_w2c'].tolist(), dtype=torch.float32)

        # Add noise for initial pose (optionally randomize noise magnitude)
        rot_noise = self.noise_rot_deg
        trans_noise = self.noise_trans_m
        if getattr(self, 'noise_rot_min', None) is not None:
            rot_noise = np.random.uniform(self.noise_rot_min, self.noise_rot_deg)
            trans_noise = np.random.uniform(self.noise_trans_min, self.noise_trans_m)
        pose_init = torch.tensor(
            add_pose_noise(s['pose_w2c'], rot_noise, trans_noise).tolist(),
            dtype=torch.float32,
        )

        result = {
            'query_fine': fine_feat,         # (64, H_f, W_f)
            'query_coarse': coarse_feat,     # (64, H_c, W_c)
            'pose_gt': pose_gt,              # (4, 4)
            'pose_init': pose_init,          # (4, 4)
            'image_id': s['img_id'],
            'image_name': s.get('image_name', ''),
            'intrinsics': self.intrinsics,   # {fx, fy, cx, cy}
        }

        # Depth (optional)
        if 'depth' in cached:
            depth = cached['depth'].clone().float()
            if depth.ndim == 3:
                depth = depth.squeeze(0)
            result['depth'] = depth  # (H, W)

        if 'teacher_fine' in cached and 'teacher_coarse' in cached:
            teacher_fine = cached['teacher_fine'].clone().float()
            teacher_coarse = cached['teacher_coarse'].clone().float()

            if self.normalize_features and self.norm_stats is not None:
                fm = self.norm_stats['fine_mean'].view(-1, 1, 1)
                fs = self.norm_stats['fine_std'].view(-1, 1, 1).clamp(min=1e-6)
                teacher_fine = (teacher_fine - fm) / fs
                cm = self.norm_stats['coarse_mean'].view(-1, 1, 1)
                cs = self.norm_stats['coarse_std'].view(-1, 1, 1).clamp(min=1e-6)
                teacher_coarse = (teacher_coarse - cm) / cs

            teacher_fine = F.interpolate(
                teacher_fine.unsqueeze(0), self.fine_hw, mode='bilinear', align_corners=False,
            ).squeeze(0)
            teacher_coarse = F.interpolate(
                teacher_coarse.unsqueeze(0), self.coarse_hw, mode='bilinear', align_corners=False,
            ).squeeze(0)
            result['teacher_query_fine'] = teacher_fine
            result['teacher_query_coarse'] = teacher_coarse

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """Custom collate that stacks tensors and preserves dicts/strings."""
    elem = batch[0]
    collated = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], dict):
            # Intrinsics: all samples share the same dict
            collated[key] = vals[0]
        elif isinstance(vals[0], str):
            collated[key] = vals
        else:
            collated[key] = vals
    return collated
