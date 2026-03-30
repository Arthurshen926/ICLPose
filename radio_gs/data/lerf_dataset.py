"""LERF dataset loader for RADIO-GS text grounding evaluation.

LERF provides:
- Scene images + camera poses (in COLMAP or transforms.json format)
- Text query annotations with ground-truth relevancy masks

Layout:
    dataset/lerf/{scene_name}/
        images/
        transforms.json  (NeRF-style, with camera params + file paths)
    dataset/lerf/{scene_name}/annotations/
        {query_text}/  (directory per query)
            frame_{idx}.png  (binary relevancy mask)
    output/radio_features/lerf/{scene_name}/
        backbone/rgb_{idx}.pt
"""

import json
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


def _load_mask(path: str) -> np.ndarray:
    """Load a binary mask as uint8 array (0/1)."""
    if _HAS_PIL:
        img = np.array(Image.open(path).convert("L"))
        return (img > 127).astype(np.uint8)
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (img > 127).astype(np.uint8)
    raise ImportError("Either PIL or cv2 is required for image loading")


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if _HAS_PIL:
        return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    raise ImportError("Either PIL or cv2 is required for resizing")


# OpenGL (NeRF convention) → OpenCV camera convention
# OpenGL: +X right, +Y up, -Z forward
# OpenCV: +X right, +Y down, +Z forward
_GL_TO_CV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _parse_transforms_json(path: str) -> Dict:
    """Parse a NeRF-style transforms.json file.

    Returns:
        dict with keys ``c2w_list`` (list of 4×4 np arrays in OpenCV convention),
        ``file_paths`` (list of image path strings), and camera parameters.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    c2w_list: List[np.ndarray] = []
    file_paths: List[str] = []

    for frame in frames:
        mat = np.array(frame["transform_matrix"], dtype=np.float32)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0, 0, 0, 1]])
        # Convert OpenGL → OpenCV convention
        mat = mat @ _GL_TO_CV
        c2w_list.append(mat)
        file_paths.append(frame.get("file_path", ""))

    result = {
        "c2w_list": c2w_list,
        "file_paths": file_paths,
    }
    # Propagate camera params if present
    for key in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h"):
        if key in data:
            result[key] = data[key]

    return result


class LERFDataset(Dataset):
    """LERF dataset with pre-extracted RADIO features and text grounding annotations.

    Args:
        scene_root: Path to LERF scene (contains ``images/`` and ``transforms.json``).
        feature_dir: Path to pre-extracted RADIO features (contains ``backbone/``).
        annotation_dir: Optional path to text query annotations.
            If None, tries ``{scene_root}/annotations``.
        feature_height: Target spatial height for grounding masks.
        feature_width: Target spatial width for grounding masks.
    """

    def __init__(
        self,
        scene_root: str,
        feature_dir: str,
        annotation_dir: Optional[str] = None,
        feature_height: int = 30,
        feature_width: int = 40,
    ) -> None:
        super().__init__()
        self.scene_root = Path(scene_root)
        self.feature_dir = Path(feature_dir)
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
        self.frame_indices = [
            int(p.stem.split("_")[1]) for p in self.feature_paths
        ]

        # --- load poses from transforms.json ---
        transforms_path = self.scene_root / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

        parsed = _parse_transforms_json(str(transforms_path))
        c2w_list = parsed["c2w_list"]
        self.file_paths = parsed["file_paths"]

        # Invert c2w → w2c, store as array
        self.poses_w2c = np.stack(
            [np.linalg.inv(c) for c in c2w_list], axis=0
        )
        logger.info("Loaded %d poses from %s", len(self.poses_w2c), transforms_path)

        # Store camera params for downstream use
        self.camera_params = {
            k: parsed[k]
            for k in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h")
            if k in parsed
        }

        # --- text query annotations ---
        if annotation_dir is not None:
            self.annotation_dir: Optional[Path] = Path(annotation_dir)
        elif (self.scene_root / "annotations").exists():
            self.annotation_dir = self.scene_root / "annotations"
        else:
            self.annotation_dir = None

        self.text_queries: List[str] = []
        if self.annotation_dir is not None and self.annotation_dir.exists():
            self.text_queries = self.get_text_queries(str(self.annotation_dir))

        logger.info(
            "LERFDataset: %d frames, %d text queries, annotations=%s",
            len(self.feature_paths),
            len(self.text_queries),
            self.annotation_dir is not None,
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    @classmethod
    def get_text_queries(cls, annotation_dir: str) -> List[str]:
        """Discover text query strings from annotation subdirectories.

        Each subdirectory under ``annotation_dir`` is treated as a query.

        Args:
            annotation_dir: Path containing one subdirectory per query.

        Returns:
            Sorted list of query strings.
        """
        ann_path = Path(annotation_dir)
        if not ann_path.exists():
            return []
        queries = sorted(
            d.name for d in ann_path.iterdir() if d.is_dir()
        )
        return queries

    @staticmethod
    def get_text_embeddings(
        query_texts: List[str],
        radio_model: Optional[object] = None,
    ) -> torch.Tensor:
        """Compute text embeddings for query strings.

        If ``radio_model`` is provided and has a ``encode_text`` method, uses it.
        Otherwise returns random unit vectors (for testing / stub).

        Args:
            query_texts: List of N query strings.
            radio_model: Optional model with ``encode_text(texts) → [N, D]``.

        Returns:
            Tensor of shape ``[N, D]`` (D=1280 by default).
        """
        n = len(query_texts)
        if n == 0:
            return torch.empty(0, 1280)

        if radio_model is not None and hasattr(radio_model, "encode_text"):
            embeddings = radio_model.encode_text(query_texts)
            if isinstance(embeddings, torch.Tensor):
                return embeddings.detach().cpu()
            return torch.from_numpy(np.array(embeddings))

        # Stub: deterministic pseudo-random embeddings for testing
        logger.warning(
            "radio_model not available; returning random text embeddings for %d queries",
            n,
        )
        gen = torch.Generator().manual_seed(hash(tuple(query_texts)) % (2**31))
        emb = torch.randn(n, 1280, generator=gen)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def _load_grounding_masks(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Load binary relevancy masks for all text queries at a given frame.

        Returns:
            [N_queries, H, W] float tensor, or None if annotations unavailable.
        """
        if self.annotation_dir is None or len(self.text_queries) == 0:
            return None

        masks = []
        for query in self.text_queries:
            mask_path = self.annotation_dir / query / f"frame_{frame_idx}.png"
            if mask_path.exists():
                raw = _load_mask(str(mask_path))
                raw = _resize_nearest(raw, self.feature_height, self.feature_width)
                masks.append(torch.from_numpy(raw.astype(np.float32)))
            else:
                masks.append(
                    torch.zeros(self.feature_height, self.feature_width, dtype=torch.float32)
                )

        return torch.stack(masks, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx = self.frame_indices[idx]

        # --- features ---
        radio_feat = torch.load(self.feature_paths[idx], map_location="cpu")
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)

        # --- pose ---
        if frame_idx < len(self.poses_w2c):
            pose_w2c = torch.from_numpy(self.poses_w2c[frame_idx].copy())
        else:
            logger.warning("Frame %d exceeds pose count; using identity", frame_idx)
            pose_w2c = torch.eye(4, dtype=torch.float32)

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

        # --- text queries + grounding masks ---
        if self.text_queries:
            out["text_queries"] = self.text_queries  # type: ignore[assignment]
            masks = self._load_grounding_masks(frame_idx)
            if masks is not None:
                out["grounding_masks"] = masks

        return out
