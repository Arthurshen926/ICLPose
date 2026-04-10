from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def resolve_dataset_type(config_or_type) -> str:
    if isinstance(config_or_type, str):
        dataset_type = config_or_type
    else:
        dataset_type = getattr(config_or_type, "dataset_type", "replica")
    return str(dataset_type or "replica").lower()


def resolve_scene_root(config) -> Path:
    explicit = getattr(config, "scene_root", "")
    if explicit:
        return Path(explicit)
    scene = getattr(config, "scene", "")
    dataset_type = resolve_dataset_type(config)
    if dataset_type == "scannet":
        return Path("dataset") / "scannet" / scene
    if dataset_type == "lerf":
        return Path("dataset") / "lerf" / scene
    return Path("dataset") / scene


def extract_feature_frame_index(path: Path) -> int:
    stem = path.stem
    if "_" in stem:
        suffix = stem.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"Could not infer frame index from feature path: {path}")


def list_feature_paths(feature_dir: str | Path, frame_ids: Optional[Sequence[int]] = None) -> list[Path]:
    feature_root = Path(feature_dir)
    backbone_dir = feature_root / "backbone"
    if not backbone_dir.exists():
        backbone_dir = feature_root
    feature_paths = sorted(
        backbone_dir.glob("rgb_*.pt"),
        key=extract_feature_frame_index,
    )
    if frame_ids is None:
        return feature_paths
    wanted = {int(fid) for fid in frame_ids}
    return [p for p in feature_paths if extract_feature_frame_index(p) in wanted]


def load_frame_id_list(path: str | Path | None) -> Optional[list[int]]:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Frame-id list not found: {src}")
    if src.suffix.lower() == ".json":
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = None
            for key in ("frame_ids", "frames", "indices"):
                if key in data:
                    items = data[key]
                    break
            if items is None:
                raise ValueError(
                    f"Unsupported JSON frame-id format in {src}; "
                    "expected list or dict with frame_ids/frames/indices"
                )
        else:
            raise ValueError(f"Unsupported JSON frame-id payload in {src}")
        return [int(item) for item in items]

    tokens = src.read_text(encoding="utf-8").replace(",", " ").split()
    return [int(tok) for tok in tokens]


def resolve_split_feature_dir(config, split: str) -> Path:
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    base = Path(getattr(config, "feature_dir", ""))
    if split == "train":
        return base

    explicit = getattr(config, "val_feature_dir", "")
    if explicit:
        return Path(explicit)

    dataset_type = resolve_dataset_type(config)
    if dataset_type == "replica":
        candidate = Path(str(base).replace(train_split, val_split))
        if candidate.exists():
            return candidate
    return base


def resolve_split_pose_source(config, split: str) -> Tuple[Optional[str], Optional[str]]:
    if split == "train":
        explicit_file = getattr(config, "pose_file", "")
        explicit_dir = getattr(config, "pose_dir", "")
    else:
        explicit_file = getattr(config, "val_pose_file", "")
        explicit_dir = getattr(config, "val_pose_dir", "")

    if explicit_file:
        return explicit_file, None
    if explicit_dir:
        return None, explicit_dir

    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    split_name = train_split if split == "train" else val_split

    if dataset_type == "replica":
        return str(scene_root / split_name / "traj_w_c.txt"), None
    if dataset_type == "scannet":
        return None, str(scene_root / "pose")
    return None, None


def resolve_split_data_dir(config, split: str, kind: str) -> Optional[Path]:
    if kind not in {"rgb", "depth", "semantics", "instance"}:
        raise ValueError(f"Unsupported data kind: {kind}")

    explicit_field = {
        ("train", "rgb"): "rgb_dir",
        ("val", "rgb"): "val_rgb_dir",
        ("train", "depth"): "depth_dir",
        ("val", "depth"): "val_depth_dir",
        ("train", "semantics"): "semantics_dir",
        ("val", "semantics"): "val_semantics_dir",
        ("train", "instance"): "instance_dir",
        ("val", "instance"): "val_instance_dir",
    }[(split, kind)]
    explicit = getattr(config, explicit_field, "")
    if explicit:
        return Path(explicit)

    dataset_type = resolve_dataset_type(config)
    scene_root = resolve_scene_root(config)
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    split_name = train_split if split == "train" else val_split

    if dataset_type == "replica":
        subdir = {
            "rgb": "rgb",
            "depth": "depth",
            "semantics": "semantic_class",
            "instance": "instance_class",
        }[kind]
        return scene_root / split_name / subdir

    if dataset_type == "scannet":
        subdir = {
            "rgb": "color",
            "depth": "depth",
            "semantics": "label-filt",
            "instance": "instance-filt",
        }[kind]
        return scene_root / subdir

    if dataset_type == "lerf":
        subdir = {
            "rgb": "images",
            "depth": "",
            "semantics": "",
            "instance": "",
        }[kind]
        return scene_root / subdir if subdir else None

    return None


def resolve_split_frame_ids(config, split: str) -> Optional[list[int]]:
    field = "train_frame_ids_path" if split == "train" else "val_frame_ids_path"
    return load_frame_id_list(getattr(config, field, ""))


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_rgb_path(rgb_dir: str | Path | None, frame_idx: int, dataset_type: str) -> Optional[Path]:
    if rgb_dir is None:
        return None
    root = Path(rgb_dir)
    if resolve_dataset_type(dataset_type) == "scannet":
        candidates = [
            root / f"{frame_idx}.jpg",
            root / f"{frame_idx}.png",
            root / f"{frame_idx:06d}.jpg",
            root / f"{frame_idx:06d}.png",
        ]
    else:
        candidates = [
            root / f"rgb_{frame_idx}.png",
            root / f"rgb_{frame_idx}.jpg",
            root / f"{frame_idx}.png",
            root / f"{frame_idx}.jpg",
        ]
    return _first_existing(candidates) or candidates[0]


def resolve_depth_path(depth_dir: str | Path | None, frame_idx: int, dataset_type: str) -> Optional[Path]:
    if depth_dir is None:
        return None
    root = Path(depth_dir)
    if resolve_dataset_type(dataset_type) == "scannet":
        candidates = [
            root / f"{frame_idx}.png",
            root / f"depth_{frame_idx}.png",
            root / f"{frame_idx:06d}.png",
        ]
    else:
        candidates = [
            root / f"depth_{frame_idx}.png",
            root / f"{frame_idx}.png",
        ]
    return _first_existing(candidates) or candidates[0]


def resolve_semantics_path(
    semantics_dir: str | Path | None,
    frame_idx: int,
    dataset_type: str,
) -> Optional[Path]:
    if semantics_dir is None:
        return None
    root = Path(semantics_dir)
    if resolve_dataset_type(dataset_type) == "scannet":
        candidates = [
            root / f"{frame_idx}.png",
            root / f"{frame_idx:06d}.png",
            root / f"semantic_class_{frame_idx}.png",
        ]
    else:
        candidates = [
            root / f"semantic_class_{frame_idx}.png",
            root / f"{frame_idx}.png",
        ]
    return _first_existing(candidates) or candidates[0]


def load_w2c_from_pose_file(pose_file: str | Path, frame_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    pose_path = Path(pose_file)
    raw = np.loadtxt(str(pose_path)).reshape(-1, 4, 4).astype(np.float32)
    w2c = np.linalg.inv(raw)
    if frame_indices is None:
        return w2c
    max_frame = max(int(fid) for fid in frame_indices) if frame_indices else -1
    if max_frame >= len(w2c):
        raise IndexError(
            f"Pose file {pose_path} has {len(w2c)} frames, but frame {max_frame} was requested"
        )
    return np.stack([w2c[int(fid)] for fid in frame_indices], axis=0)


def load_w2c_from_pose_dir(pose_dir: str | Path, frame_indices: Sequence[int]) -> np.ndarray:
    pose_root = Path(pose_dir)
    poses = []
    for frame_idx in frame_indices:
        pose_path = pose_root / f"{int(frame_idx)}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")
        pose = np.loadtxt(str(pose_path)).reshape(4, 4).astype(np.float32)
        if not np.isfinite(pose).all():
            raise ValueError(f"Invalid pose values in {pose_path}")
        poses.append(np.linalg.inv(pose))
    return np.stack(poses, axis=0)
