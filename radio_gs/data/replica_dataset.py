"""Replica dataset loader for RADIO-GS evaluation.

Replica provides:
- GT depth maps (16-bit PNG, depth_in_meters = pixel_value / 1000.0)
- Semantic labels (NYU40 classes, 40 categories)
- Camera trajectories (4x4 c2w in traj_w_c.txt)

Layout:
    dataset/room_0/
        Sequence_1/traj_w_c.txt
        Sequence_1/rgb/rgb_{idx}.png
        Sequence_1/depth/depth_{idx}.png
        Sequence_1/semantic_class/semantic_class_{idx}.png
    output/radio_features/room_0/
        backbone/rgb_{idx}.pt
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

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

# fmt: off
NYU40_CLASSES: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor_mat",
    "clothes", "ceiling", "books", "refrigerator", "television",
    "paper", "towel", "shower_curtain", "box", "whiteboard",
    "person", "night_stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
]
# fmt: on


def _load_image_gray16(path: str) -> Optional[np.ndarray]:
    """Load a 16-bit grayscale PNG, returns H×W uint16 array or None."""
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.uint16)
    if _HAS_PIL:
        img = Image.open(path)
        return np.array(img, dtype=np.uint16)
    raise ImportError("Either PIL or cv2 is required for image loading")


def _load_image_label(path: str) -> Optional[np.ndarray]:
    """Load a semantic label PNG as uint8/uint16 array."""
    if _HAS_PIL:
        img = Image.open(path)
        return np.array(img)
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img
    raise ImportError("Either PIL or cv2 is required for image loading")


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize 2D array with nearest-neighbor interpolation."""
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if _HAS_PIL:
        img = Image.fromarray(arr)
        img = img.resize((w, h), Image.NEAREST)
        return np.array(img)
    raise ImportError("Either PIL or cv2 is required for resizing")


def _resize_bilinear(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize 2D float array with bilinear interpolation."""
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    if _HAS_PIL:
        img = Image.fromarray(arr)
        img = img.resize((w, h), Image.BILINEAR)
        return np.array(img)
    raise ImportError("Either PIL or cv2 is required for resizing")


class ReplicaDataset(Dataset):
    """Replica dataset with pre-extracted RADIO features, GT depth, and semantics.

    Args:
        scene_root: Path to scene directory (e.g. ``dataset/room_0``).
        feature_dir: Path to pre-extracted RADIO features (contains ``backbone/``).
        split: Trajectory subdirectory name (default ``Sequence_1``).
        load_depth: Whether to load GT depth maps.
        load_semantics: Whether to load semantic class labels.
        feature_height: Target spatial height for depth/semantics (match feature res).
        feature_width: Target spatial width for depth/semantics.
    """

    def __init__(
        self,
        scene_root: str,
        feature_dir: str,
        split: str = "Sequence_1",
        load_depth: bool = True,
        load_semantics: bool = True,
        feature_height: int = 30,
        feature_width: int = 40,
    ) -> None:
        super().__init__()
        self.scene_root = Path(scene_root)
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.load_depth = load_depth
        self.load_semantics = load_semantics
        self.feature_height = feature_height
        self.feature_width = feature_width

        # --- discover feature files ---
        backbone_dir = self.feature_dir / "backbone"
        if not backbone_dir.exists():
            backbone_dir = self.feature_dir
        self.feature_paths = sorted(
            backbone_dir.glob("rgb_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if len(self.feature_paths) == 0:
            raise FileNotFoundError(
                f"No rgb_*.pt features found in {backbone_dir}"
            )

        # --- extract frame indices from feature filenames ---
        self.frame_indices = [
            int(p.stem.split("_")[1]) for p in self.feature_paths
        ]

        # --- load poses ---
        split_dir = self.scene_root / split
        pose_file = split_dir / "traj_w_c.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        c2w_all = np.loadtxt(str(pose_file)).reshape(-1, 4, 4).astype(np.float32)
        # Invert c2w → w2c
        self.poses_w2c = np.linalg.inv(c2w_all)
        logger.info("Loaded %d poses from %s", len(self.poses_w2c), pose_file)

        # --- discover depth files ---
        self.depth_dir = split_dir / "depth"
        if self.load_depth and not self.depth_dir.exists():
            logger.warning("Depth directory not found: %s — disabling depth", self.depth_dir)
            self.load_depth = False

        # --- discover semantic files ---
        self.semantic_dir = split_dir / "semantic_class"
        if self.load_semantics and not self.semantic_dir.exists():
            logger.warning(
                "Semantic directory not found: %s — disabling semantics",
                self.semantic_dir,
            )
            self.load_semantics = False

        logger.info(
            "ReplicaDataset: %d frames, depth=%s, semantics=%s",
            len(self.feature_paths),
            self.load_depth,
            self.load_semantics,
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx = self.frame_indices[idx]

        # --- features ---
        radio_feat = torch.load(self.feature_paths[idx], map_location="cpu")
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)  # [1,C,H,W] → [C,H,W]

        # --- pose ---
        if frame_idx < len(self.poses_w2c):
            pose_w2c = torch.from_numpy(self.poses_w2c[frame_idx])
        else:
            logger.warning("Frame %d exceeds pose count; using identity", frame_idx)
            pose_w2c = torch.eye(4, dtype=torch.float32)

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

        # --- depth ---
        if self.load_depth:
            depth_path = self.depth_dir / f"depth_{frame_idx}.png"
            if depth_path.exists():
                raw = _load_image_gray16(str(depth_path))
                depth_m = raw.astype(np.float32) / 1000.0
                depth_m = _resize_bilinear(depth_m, self.feature_height, self.feature_width)
                out["depth"] = torch.from_numpy(depth_m)
            else:
                out["depth"] = torch.zeros(
                    self.feature_height, self.feature_width, dtype=torch.float32
                )

        # --- semantics ---
        if self.load_semantics:
            sem_path = self.semantic_dir / f"semantic_class_{frame_idx}.png"
            if sem_path.exists():
                raw = _load_image_label(str(sem_path))
                raw = _resize_nearest(raw, self.feature_height, self.feature_width)
                out["semantics"] = torch.from_numpy(raw.astype(np.int64))
            else:
                out["semantics"] = torch.zeros(
                    self.feature_height, self.feature_width, dtype=torch.long
                )

        return out
