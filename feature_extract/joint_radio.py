from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from feature_field.utils.project_config import load_joint_radio_config


DEFAULT_JOINT_RADIO_CONFIG: Dict = {
    "exp_name": "joint_radio_dcff_oh_v1",
    "output_dir": "feature_extract/output",
    "dataset": {
        "source_dir": "dataset/OldHospital",
        "feature_dir": "feature_extract/output/features_radio_dual/OldHospital",
        "train_split": "dataset/OldHospital/dataset_train.txt",
        "val_split": "dataset/OldHospital/dataset_test.txt",
        "image_patterns": [
            "seq*/*.png",
            "seq*/*.jpg",
            "images/*.png",
            "images/*.jpg",
            "*.png",
            "*.jpg",
        ],
        "patch_size": 16,
        "input_hw": [1088, 1920],
        "feature_hw": [68, 120],
        "coarse_feature_hw": None,
        "cache_teacher": False,
        "fallback_val_ratio": 0.1,
        "max_train_samples": None,
        "max_val_samples": None,
        "synthetic_if_missing": False,
    },
    "model": {
        "feature_dim": 64,
        "fine_feature_dim": None,
        "coarse_feature_dim": None,
        "base_channels": 32,
        "stage_dims": [32, 64, 96, 128],
        "dropout": 0.0,
        "l2_normalize": True,
        "predict_magnitude": False,
        "fine_init_norm": 1.0,
        "coarse_init_norm": 1.0,
        "magnitude_min": 1e-4,
        "warmstart_strict": True,
    },
    "training": {
        "device": "cuda",
        "seed": 42,
        "epochs": 8,
        "batch_size": 2,
        "num_workers": 2,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "amp": True,
        "log_every": 10,
        "save_every_epochs": 1,
        "val_every_epochs": 1,
        "max_steps": None,
    },
    "loss": {
        "fine_l1_weight": 1.0,
        "fine_cos_weight": 1.0,
        "coarse_l1_weight": 1.0,
        "coarse_cos_weight": 1.0,
        "teacher_norm_weight": 0.0,
        "query_teacher_infonce_weight": 0.0,
        "infonce_temperature": 0.07,
        "infonce_samples": 256,
        "infonce_cross_batch": False,
    },
    "retrieval": {
        "enabled": False,
        "feature_dir": None,
        "teacher_subdir": "cls",
        "student_dim": 768,
        "hidden_dim": 256,
        "dropout": 0.0,
        "l2_normalize": True,
        "cache_teacher": False,
        "l1_weight": 0.0,
        "cos_weight": 0.0,
        "infonce_weight": 0.0,
        "similarity_weight": 0.0,
        "temperature": 0.07,
    },
    "map_supervision": {
        "enabled": False,
        "config_path": None,
        "colmap_dir": None,
        "cache_rendered": True,
        "trainable": False,
        "map_lr_scale": 0.1,
        "hash_mlp_lr_scale": 0.05,
        "train_fine_decoder": False,
        "train_coarse_fusion": False,
        "train_feat_sharp": False,
        "train_hash_mlp": False,
        "detach_query_features": False,
        "coarse_smoothing_kernel": 1,
        "coarse_start_epoch": 999999,
        "query_fine_weight": 0.0,
        "query_fine_raw_weight": 0.0,
        "query_coarse_weight": 0.0,
        "query_fine_infonce_weight": 0.0,
        "query_coarse_infonce_weight": 0.0,
        "rendered_teacher_fine_weight": 0.0,
        "rendered_teacher_fine_raw_weight": 0.0,
        "rendered_teacher_coarse_weight": 0.0,
        "rendered_teacher_fine_infonce_weight": 0.0,
        "rendered_teacher_coarse_infonce_weight": 0.0,
        "infonce_cross_batch": True,
        "variance_target_std": 0.05,
        "query_variance_weight": 0.0,
        "map_variance_weight": 0.0,
        "query_covariance_weight": 0.0,
        "map_covariance_weight": 0.0,
    },
    "visualization": {
        "num_val_vis": 4,
        "save_root": "feature_extract/output/visualizations/feature_track",
    },
}


def load_config(path: str) -> Dict:
    return load_joint_radio_config(path, default_config=DEFAULT_JOINT_RADIO_CONFIG)


def safe_torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_cambridge_split(split_path: str | Path) -> set[str]:
    names: set[str] = set()
    split_path = Path(split_path)
    if not split_path.is_file():
        return names
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("Visual")
                or line.startswith("ImageFile")
            ):
                continue
            image_name = line.split()[0].replace("\\", "/")
            stem = str(Path(image_name).with_suffix(""))
            names.add(image_name)
            names.add(stem + ".png")
            names.add(stem + ".jpg")
    return names


def discover_images(source_dir: str | Path, patterns: list[str]) -> list[Path]:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    for pattern in patterns:
        found = sorted(source_dir.glob(pattern))
        if found:
            return found
    return []


class TeacherFeatureStore:
    def __init__(self, feature_dir: str | Path, cache_in_memory: bool = False):
        self.feature_dir = Path(feature_dir)
        self.fine_dir = self.feature_dir / "fine_geo"
        self.coarse_dir = self.feature_dir / "coarse_sem"
        if not self.fine_dir.is_dir() or not self.coarse_dir.is_dir():
            raise FileNotFoundError(
                f"Expected fine_geo/ and coarse_sem/ under {self.feature_dir}"
            )

        self.fine_files = self._discover_files(self.fine_dir, "fine_geo")
        self.coarse_files = self._discover_files(self.coarse_dir, "coarse_sem")
        self.indices = sorted(set(self.fine_files) & set(self.coarse_files))
        if not self.indices:
            raise RuntimeError(f"No paired teacher features found in {self.feature_dir}")

        sample = safe_torch_load(self.fine_files[self.indices[0]]).float()
        coarse_sample = safe_torch_load(self.coarse_files[self.indices[0]]).float()
        self.fine_feature_dim = int(sample.shape[0])
        self.coarse_feature_dim = int(coarse_sample.shape[0])
        self.feature_dim = self.fine_feature_dim
        self.feature_hw = (int(sample.shape[1]), int(sample.shape[2]))
        self.coarse_feature_hw = (int(coarse_sample.shape[1]), int(coarse_sample.shape[2]))
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _discover_files(root_dir: Path, scale_name: str) -> dict[int, Path]:
        mapping = {}
        pattern = re.compile(rf"rgb_(\d+)_{re.escape(scale_name)}_.*\.pt$")
        for path in sorted(root_dir.glob("*.pt")):
            match = pattern.match(path.name)
            if match:
                mapping[int(match.group(1))] = path
        return mapping

    def load_pair(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_memory and index in self._cache:
            return self._cache[index]

        fine = safe_torch_load(self.fine_files[index]).float()
        coarse = safe_torch_load(self.coarse_files[index]).float()
        if self.cache_in_memory:
            self._cache[index] = (fine, coarse)
        return fine, coarse


def sample_name_to_feature_stem(sample_name: str) -> str:
    return Path(sample_name).with_suffix("").as_posix().replace("/", "_")


class RetrievalTeacherStore:
    def __init__(self, feature_dir: str | Path, subdir: str = "cls", cache_in_memory: bool = False):
        root = Path(feature_dir)
        self.root_dir = root / subdir if subdir else root
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Retrieval teacher directory not found: {self.root_dir}")

        pattern = re.compile(r"(.+)_cls_.*\.pt$")
        self.files: dict[str, Path] = {}
        for path in sorted(self.root_dir.glob("*.pt")):
            match = pattern.match(path.name)
            if match:
                self.files[match.group(1)] = path
        if not self.files:
            raise RuntimeError(f"No CLS teacher descriptors found in {self.root_dir}")

        sample = safe_torch_load(next(iter(self.files.values()))).float().view(-1)
        self.feature_dim = int(sample.numel())
        self.cache_in_memory = cache_in_memory
        self._cache: dict[str, torch.Tensor] = {}

    def load(self, sample_name: str) -> torch.Tensor:
        stem = sample_name_to_feature_stem(sample_name)
        if self.cache_in_memory and stem in self._cache:
            return self._cache[stem]

        path = self.files.get(stem)
        if path is None:
            raise KeyError(f"Missing retrieval teacher descriptor for {sample_name} ({stem})")

        descriptor = safe_torch_load(path).float().view(-1)
        if self.cache_in_memory:
            self._cache[stem] = descriptor
        return descriptor


def build_all_records(dataset_cfg: Dict, teacher_store: TeacherFeatureStore, allow_synthetic: bool = False) -> list[dict]:
    images = discover_images(dataset_cfg["source_dir"], dataset_cfg["image_patterns"])
    records = []
    if images:
        for teacher_idx in teacher_store.indices:
            if teacher_idx >= len(images):
                continue
            image_path = images[teacher_idx]
            rel_name = image_path.relative_to(dataset_cfg["source_dir"]).as_posix()
            records.append(
                {
                    "teacher_idx": teacher_idx,
                    "image_path": str(image_path),
                    "sample_name": rel_name,
                    "normalized_name": rel_name.replace("\\", "/"),
                }
            )
    elif allow_synthetic:
        for teacher_idx in teacher_store.indices:
            records.append(
                {
                    "teacher_idx": teacher_idx,
                    "image_path": None,
                    "sample_name": f"synthetic_{teacher_idx:05d}",
                    "normalized_name": f"synthetic_{teacher_idx:05d}",
                }
            )
    else:
        raise FileNotFoundError(
            "No source RGB images found. Set dataset.synthetic_if_missing=true or use --smoke-test."
        )

    if not records:
        raise RuntimeError("No records could be paired with teacher features.")
    return records


def split_records(all_records: list[dict], dataset_cfg: Dict) -> tuple[list[dict], list[dict]]:
    train_split = parse_cambridge_split(dataset_cfg.get("train_split"))
    val_split = parse_cambridge_split(dataset_cfg.get("val_split"))

    if train_split and val_split:
        train_records = [record for record in all_records if record["normalized_name"] in train_split]
        val_records = [record for record in all_records if record["normalized_name"] in val_split]
    else:
        val_ratio = float(dataset_cfg.get("fallback_val_ratio", 0.1))
        split_idx = max(1, int(round(len(all_records) * (1.0 - val_ratio))))
        train_records = all_records[:split_idx]
        val_records = all_records[split_idx:]

    if not val_records:
        val_records = train_records[: max(1, min(8, len(train_records)))]
    if not train_records:
        raise RuntimeError("Training split is empty after pairing images and teacher caches.")

    if dataset_cfg.get("max_train_samples") is not None:
        train_records = train_records[: int(dataset_cfg["max_train_samples"])]
    if dataset_cfg.get("max_val_samples") is not None:
        val_records = val_records[: int(dataset_cfg["max_val_samples"])]

    return train_records, val_records


class JointRADIOQueryDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        teacher_store: TeacherFeatureStore,
        input_hw: tuple[int, int] | list[int],
        synthetic_rgb: bool = False,
        retrieval_teacher_store: RetrievalTeacherStore | None = None,
    ):
        self.records = records
        self.teacher_store = teacher_store
        self.input_hw = tuple(input_hw)
        self.synthetic_rgb = synthetic_rgb
        self.retrieval_teacher_store = retrieval_teacher_store

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, record: dict) -> torch.Tensor:
        if record["image_path"] is None:
            generator = torch.Generator().manual_seed(record["teacher_idx"])
            return torch.rand(3, self.input_hw[0], self.input_hw[1], generator=generator)

        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            if tuple(reversed(self.input_hw)) != image.size:
                image = image.resize((self.input_hw[1], self.input_hw[0]), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        rgb = self._load_rgb(record)
        teacher_fine, teacher_coarse = self.teacher_store.load_pair(record["teacher_idx"])
        item = {
            "rgb": rgb,
            "teacher_fine": teacher_fine,
            "teacher_coarse": teacher_coarse,
            "teacher_idx": record["teacher_idx"],
            "sample_name": record["sample_name"],
        }
        if self.retrieval_teacher_store is not None:
            item["teacher_retrieval"] = self.retrieval_teacher_store.load(record["sample_name"])
        return item
