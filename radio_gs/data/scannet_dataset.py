"""ScanNet dataset loader for RADIO-GS evaluation.

ScanNet provides:
- GT depth maps (16-bit PNG, depth_in_mm)
- Semantic labels (NYU40 or ScanNet20)
- Camera intrinsics per frame

Layout:
    dataset/scannet/scene0000_00/
        color/0.jpg, 1.jpg, ...
        depth/0.png, 1.png, ...
        label-filt/0.png, 1.png, ...
        pose/0.txt, 1.txt, ...  (4x4 c2w per file)
        intrinsic/intrinsic_depth.txt  (4x4 intrinsic matrix)
    output/radio_features/scene0000_00/
        backbone/rgb_{idx}.pt
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _load_gray16(path: str) -> Optional[np.ndarray]:
    """Load 16-bit grayscale image."""
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.uint16)
    if _HAS_PIL:
        return np.array(Image.open(path), dtype=np.uint16)
    raise ImportError("Either PIL or cv2 is required for image loading")


def _load_label(path: str) -> Optional[np.ndarray]:
    """Load semantic label image."""
    if _HAS_PIL:
        return np.array(Image.open(path))
    if _HAS_CV2:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    raise ImportError("Either PIL or cv2 is required for image loading")


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if _HAS_PIL:
        return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    raise ImportError("Either PIL or cv2 is required for resizing")


def _resize_bilinear(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    if _HAS_PIL:
        return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
    raise ImportError("Either PIL or cv2 is required for resizing")


def _is_valid_pose(pose: np.ndarray) -> bool:
    """Check that a 4×4 pose matrix contains no inf or NaN."""
    return bool(np.isfinite(pose).all())


def _load_scannet_pose(path: str) -> Optional[np.ndarray]:
    """Load a single 4×4 pose from a text file. Returns None if invalid."""
    try:
        pose = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
    except (ValueError, FileNotFoundError):
        return None
    if not _is_valid_pose(pose):
        return None
    return pose


def _load_intrinsic(path: str) -> np.ndarray:
    """Load a 4×4 intrinsic matrix from a text file."""
    return np.loadtxt(path, dtype=np.float32).reshape(4, 4)


class ScanNetDataset(Dataset):
    """ScanNet dataset with pre-extracted RADIO features, GT depth, and semantics.

    Args:
        scene_root: Path to ScanNet scene (e.g. ``dataset/scannet/scene0000_00``).
        feature_dir: Path to pre-extracted RADIO features (contains ``backbone/``).
        load_depth: Whether to load GT depth maps.
        load_semantics: Whether to load semantic label images.
        feature_height: Target spatial height (match feature resolution).
        feature_width: Target spatial width.
        max_frames: Optional cap on frame count (for debugging / quick eval).
    """

    def __init__(
        self,
        scene_root: str,
        feature_dir: str,
        load_depth: bool = True,
        load_semantics: bool = True,
        feature_height: int = 30,
        feature_width: int = 40,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.scene_root = Path(scene_root)
        self.feature_dir = Path(feature_dir)
        self.load_depth = load_depth
        self.load_semantics = load_semantics
        self.feature_height = feature_height
        self.feature_width = feature_width

        # --- discover feature files ---
        backbone_dir = self.feature_dir / "backbone"
        if not backbone_dir.exists():
            backbone_dir = self.feature_dir
        feature_paths_all = sorted(
            backbone_dir.glob("rgb_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if len(feature_paths_all) == 0:
            raise FileNotFoundError(
                f"No rgb_*.pt features found in {backbone_dir}"
            )

        # --- per-frame poses: pose/{i}.txt ---
        pose_dir = self.scene_root / "pose"
        if not pose_dir.exists():
            raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

        # Build valid frame list (intersect features with valid poses)
        self.feature_paths: List[Path] = []
        self.frame_indices: List[int] = []
        self.poses_w2c: List[np.ndarray] = []

        for feat_path in feature_paths_all:
            frame_idx = int(feat_path.stem.split("_")[1])
            pose_path = pose_dir / f"{frame_idx}.txt"
            c2w = _load_scannet_pose(str(pose_path))
            if c2w is None:
                logger.debug("Skipping frame %d: invalid/missing pose", frame_idx)
                continue

            self.feature_paths.append(feat_path)
            self.frame_indices.append(frame_idx)
            self.poses_w2c.append(np.linalg.inv(c2w))

        if len(self.feature_paths) == 0:
            raise RuntimeError(
                f"No valid frames after filtering poses in {scene_root}"
            )

        # Apply frame cap
        if max_frames is not None and max_frames < len(self.feature_paths):
            self.feature_paths = self.feature_paths[:max_frames]
            self.frame_indices = self.frame_indices[:max_frames]
            self.poses_w2c = self.poses_w2c[:max_frames]

        # Stack poses for efficient access
        self.poses_w2c_np = np.stack(self.poses_w2c, axis=0)

        # --- optional directories ---
        self.depth_dir = self.scene_root / "depth"
        if self.load_depth and not self.depth_dir.exists():
            logger.warning("Depth directory not found: %s — disabling", self.depth_dir)
            self.load_depth = False

        self.label_dir = self.scene_root / "label-filt"
        if self.load_semantics and not self.label_dir.exists():
            logger.warning("Label directory not found: %s — disabling", self.label_dir)
            self.load_semantics = False

        # --- intrinsics (optional, stored if available) ---
        self.intrinsic: Optional[np.ndarray] = None
        intrinsic_path = self.scene_root / "intrinsic" / "intrinsic_depth.txt"
        if intrinsic_path.exists():
            self.intrinsic = _load_intrinsic(str(intrinsic_path))

        logger.info(
            "ScanNetDataset: %d valid frames (of %d features), depth=%s, semantics=%s",
            len(self.feature_paths),
            len(feature_paths_all),
            self.load_depth,
            self.load_semantics,
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    def get_intrinsic(self) -> Optional[torch.Tensor]:
        """Return depth intrinsic matrix as [4,4] tensor, or None."""
        if self.intrinsic is not None:
            return torch.from_numpy(self.intrinsic.copy())
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx = self.frame_indices[idx]

        # --- features ---
        radio_feat = torch.load(self.feature_paths[idx], map_location="cpu")
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)

        # --- pose ---
        pose_w2c = torch.from_numpy(self.poses_w2c_np[idx])

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

        # --- depth (ScanNet stores depth in mm) ---
        if self.load_depth:
            depth_path = self.depth_dir / f"{frame_idx}.png"
            if depth_path.exists():
                raw = _load_gray16(str(depth_path))
                depth_m = raw.astype(np.float32) / 1000.0
                depth_m = _resize_bilinear(depth_m, self.feature_height, self.feature_width)
                out["depth"] = torch.from_numpy(depth_m)
            else:
                out["depth"] = torch.zeros(
                    self.feature_height, self.feature_width, dtype=torch.float32
                )

        # --- semantics ---
        if self.load_semantics:
            label_path = self.label_dir / f"{frame_idx}.png"
            if label_path.exists():
                raw = _load_label(str(label_path))
                raw = _resize_nearest(raw, self.feature_height, self.feature_width)
                out["semantics"] = torch.from_numpy(raw.astype(np.int64))
            else:
                out["semantics"] = torch.zeros(
                    self.feature_height, self.feature_width, dtype=torch.long
                )

        return out
