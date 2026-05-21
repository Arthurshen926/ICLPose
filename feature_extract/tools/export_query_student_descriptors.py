#!/usr/bin/env python3
"""Export compact query-student descriptors without writing dense feature maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract import (  # noqa: E402
    load_config,
    safe_torch_load,
)
from feature_extract.export_impl import build_model  # noqa: E402
from feature_extract.joint_radio import discover_images, parse_cambridge_split_ordered  # noqa: E402
from feature_extract.localizability.reference_pose_scoring import descriptor_from_dense_feature, save_descriptor_bank  # noqa: E402


def _normalize_name(name: str | Path) -> str:
    return Path(str(name).replace("\\", "/")).as_posix()


def _candidate_names(name: str | Path) -> list[str]:
    norm = _normalize_name(name)
    candidates = [norm]
    if norm.startswith("images/"):
        candidates.append(norm[len("images/"):])
    candidates.append(Path(norm).name)
    candidates.append(str(Path(norm).with_suffix("")))
    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _has_teacher_feature_subdirs(path: Path) -> bool:
    return (path / "fine_geo").is_dir() and (path / "coarse_sem").is_dir()


def resolve_feature_dir(feature_dir: str | Path) -> Path:
    """Resolve legacy result/feature_extract roots after result/result migration."""

    path = Path(feature_dir)
    if _has_teacher_feature_subdirs(path):
        return path

    text = str(path)
    marker = "/result/feature_extract/"
    if marker in text:
        nested = Path(text.replace(marker, "/result/result/feature_extract/", 1))
        if _has_teacher_feature_subdirs(nested):
            return nested
    return path


def apply_student_feature_hw_defaults(cfg: dict) -> None:
    dataset_cfg = cfg.setdefault("dataset", {})
    if dataset_cfg.get("student_feature_hw") is not None:
        dataset_cfg["feature_hw"] = list(dataset_cfg["student_feature_hw"])
    if dataset_cfg.get("student_coarse_feature_hw") is not None:
        dataset_cfg["coarse_feature_hw"] = list(dataset_cfg["student_coarse_feature_hw"])


def prepare_checkpoint_for_rgb_export(checkpoint: dict) -> tuple[dict, list[str]]:
    state_dict = checkpoint.get("model_state_dict", {})
    legacy_prefixes = ("fine_loc_head.", "fine_loc_highres_fuse.")
    legacy_names = {"fine_loc_scale", "fine_loc_highres_scale"}
    dropped = [
        key
        for key in state_dict
        if key in legacy_names or any(key.startswith(prefix) for prefix in legacy_prefixes)
    ]
    if not dropped:
        return checkpoint, []
    cleaned = dict(checkpoint)
    cleaned["model_state_dict"] = {
        key: value for key, value in state_dict.items() if key not in set(dropped)
    }
    return cleaned, dropped


def resolve_descriptor_key(descriptor_key: str, *, fine_key: str) -> str:
    key = str(descriptor_key)
    if key in {"export_fine", "config_fine"}:
        return str(fine_key)
    return key


def _records_from_split_order(records: list[dict], split_order: list[str], split_names: set[str]) -> list[dict]:
    record_by_key: dict[str, dict] = {}
    for record in records:
        for key in _candidate_names(record["sample_name"]):
            if key in split_names:
                record_by_key.setdefault(key, record)

    selected: list[dict] = []
    used: set[int] = set()
    for split_name in split_order:
        for key in _candidate_names(split_name):
            record = record_by_key.get(key)
            if record is None:
                continue
            record_id = id(record)
            if record_id not in used:
                selected.append(record)
                used.add(record_id)
            break
    return selected


def build_rgb_records(dataset_cfg: dict, split: str, limit: int | None) -> list[dict]:
    images = discover_images(dataset_cfg["source_dir"], dataset_cfg.get("image_patterns", ["*.png", "*.jpg"]))
    if not images:
        raise FileNotFoundError(f"No RGB images found under {dataset_cfg['source_dir']}")

    source_dir = Path(dataset_cfg["source_dir"])
    records = [
        {
            "record_idx": int(idx),
            "image_path": str(image_path),
            "sample_name": _normalize_name(image_path.relative_to(source_dir)),
            "normalized_name": _normalize_name(image_path.relative_to(source_dir)),
        }
        for idx, image_path in enumerate(images)
    ]
    if split != "all":
        split_key = "train_split" if split == "train" else "val_split"
        split_order, split_names = parse_cambridge_split_ordered(dataset_cfg.get(split_key))
        records = _records_from_split_order(records, split_order, split_names) if split_names else []
    if limit is not None:
        records = records[:limit]
    return records


class RGBOnlyQueryDataset(Dataset):
    def __init__(self, records: list[dict], input_hw: tuple[int, int] | list[int]):
        self.records = records
        self.input_hw = tuple(input_hw)

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if tuple(reversed(self.input_hw)) != image.size:
                image = image.resize((self.input_hw[1], self.input_hw[0]), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        return {
            "rgb": self._load_rgb(record["image_path"]),
            "sample_name": record["sample_name"],
            "record_idx": record["record_idx"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-key", default="fine")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if bool(cfg.get("model", {}).get("teacher_fine_condition", False)):
        feature_dir = resolve_feature_dir(cfg["dataset"]["feature_dir"])
        raise RuntimeError(
            "RGB-only descriptor export does not support teacher_fine_condition=true. "
            f"A teacher-backed export path is required for feature_dir={feature_dir}."
        )
    apply_student_feature_hw_defaults(cfg)
    records = build_rgb_records(cfg["dataset"], args.split, args.limit)
    if not records:
        raise RuntimeError("No records selected for descriptor export")

    dataset = RGBOnlyQueryDataset(
        records,
        input_hw=cfg["dataset"]["input_hw"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    checkpoint, dropped_keys = prepare_checkpoint_for_rgb_export(safe_torch_load(args.checkpoint))
    model = build_model(cfg, checkpoint, device)
    fine_key = str(
        cfg.get("export", {}).get(
            "fine_key",
            cfg.get("model", {}).get("export_fine_key", "fine"),
        )
    )
    descriptor_key = resolve_descriptor_key(args.descriptor_key, fine_key=fine_key)
    descriptors: dict[str, torch.Tensor] = {}
    use_amp = device.type == "cuda"
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(rgb)
        if descriptor_key not in pred:
            raise KeyError(f"descriptor_key={descriptor_key!r} not found in model outputs")
        feature = pred[descriptor_key].detach().float().cpu()
        for idx, sample_name in enumerate(batch["sample_name"]):
            descriptors[str(sample_name)] = descriptor_from_dense_feature(feature[idx])

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rgb_only": True,
        "split": str(args.split),
        "limit": args.limit,
        "num_records": len(records),
        "descriptor_key": descriptor_key,
        "descriptor_dim": int(next(iter(descriptors.values())).numel()) if descriptors else 0,
        "batch_size": int(args.batch_size),
        "dropped_legacy_checkpoint_keys": dropped_keys,
    }
    save_descriptor_bank(args.output_path, descriptors, metadata=metadata)
    sidecar = Path(args.output_path).with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output_path": str(args.output_path)}, indent=2))


if __name__ == "__main__":
    main()
