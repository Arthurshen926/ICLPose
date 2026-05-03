#!/usr/bin/env python3
"""Export query-student absolute pose hypotheses as real-init pose caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    save_retrieval_init_entries,
)
from feature_extract.train_impl import (  # noqa: E402
    TeacherFeatureStore,
    build_all_records,
    build_radio_query_student,
    resolve_query_feature_dims,
    safe_torch_load,
)


def build_pose_init_entries_from_predictions(
    *,
    query_samples: Sequence[Dict],
    pose_candidates,
    scores,
    source_name: str,
    save_path: str | None = None,
) -> Tuple[list[Dict], Dict]:
    pose_candidates = np.asarray(pose_candidates, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if pose_candidates.ndim != 4 or pose_candidates.shape[-2:] != (4, 4):
        raise ValueError(f"pose_candidates must have shape [N,K,4,4], got {pose_candidates.shape}")
    if scores.shape != pose_candidates.shape[:2]:
        raise ValueError(f"scores must have shape {pose_candidates.shape[:2]}, got {scores.shape}")
    if len(query_samples) != pose_candidates.shape[0]:
        raise ValueError(
            f"query_samples length {len(query_samples)} does not match predictions {pose_candidates.shape[0]}"
        )

    topk = pose_candidates.shape[1]
    entries = []
    for idx, sample in enumerate(query_samples):
        order = np.argsort(-scores[idx])
        ordered_poses = pose_candidates[idx, order]
        ordered_scores = scores[idx, order]
        entries.append(
            {
                "query_img_id": int(sample["img_id"]),
                "query_image_name": sample["image_name"],
                "query_image_stem": sample["image_stem"],
                "pose_init": ordered_poses[0].astype(np.float32),
                "init_source": source_name,
                "retrieval_frame_id": -1,
                "retrieval_image_name": "",
                "retrieval_score": float(ordered_scores[0]),
                "pose_init_candidates": ordered_poses.astype(np.float32),
                "candidate_valid_mask": np.ones((topk,), dtype=bool),
                "retrieval_frame_ids_candidates": np.full((topk,), -1, dtype=np.int64),
                "retrieval_image_names_candidates": np.array([""] * topk),
                "retrieval_scores_candidates": ordered_scores.astype(np.float32),
            }
        )

    stats = {
        "method_requested": "query_student_pose_init",
        "method_used": source_name,
        "retrieval_topk_requested": int(topk),
        "num_query_samples": int(len(entries)),
        "counts_by_source": {source_name: int(len(entries))},
    }
    if save_path:
        save_retrieval_init_entries(entries, stats, save_path)
    return entries, stats


def _load_rgb(path: str, input_hw: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if tuple(reversed(input_hw)) != img.size:
            img = img.resize((input_hw[1], input_hw[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _records_by_sample_name(records: Sequence[Dict]) -> Dict[str, Dict]:
    by_name = {}
    for record in records:
        name = str(record["sample_name"]).replace("\\", "/")
        by_name[name] = record
        by_name.setdefault(Path(name).name, record)
    return by_name


@torch.no_grad()
def export_query_student_pose_init(
    *,
    config_path: str,
    checkpoint_path: str,
    colmap_dir: str,
    query_split: str,
    save_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    source_name: str | None = None,
) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
    model = build_radio_query_student(cfg, fine_feature_dim=fine_dim, coarse_feature_dim=coarse_dim).to(device_obj)
    checkpoint = safe_torch_load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    all_records = build_all_records(cfg["dataset"], teacher_store, allow_synthetic=False)
    record_by_name = _records_by_sample_name(all_records)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    input_hw = tuple(cfg["dataset"]["input_hw"])

    pose_parts = []
    score_parts = []
    used_samples = []
    for start in range(0, len(query_samples), int(batch_size)):
        chunk = query_samples[start : start + int(batch_size)]
        rgbs = []
        kept = []
        for sample in chunk:
            record = record_by_name.get(sample["image_name"]) or record_by_name.get(Path(sample["image_name"]).name)
            if record is None or record.get("image_path") is None:
                continue
            rgbs.append(_load_rgb(record["image_path"], input_hw))
            kept.append(sample)
        if not rgbs:
            continue
        rgb_batch = torch.stack(rgbs, dim=0).to(device_obj)
        outputs = model(rgb_batch)
        if "pose_init" not in outputs:
            raise RuntimeError("Model did not return pose_init; set model.pose_init_head=true")
        pose_parts.append(outputs["pose_init"]["pose_w2c"].detach().cpu().numpy())
        score_parts.append(outputs["pose_init"]["scores"].detach().cpu().numpy())
        used_samples.extend(kept)

    if not pose_parts:
        raise RuntimeError("No pose init predictions were exported; check split/image paths")
    pose_candidates = np.concatenate(pose_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    source = source_name or f"query_student_pose_init_{Path(checkpoint_path).parent.parent.name}"
    _entries, stats = build_pose_init_entries_from_predictions(
        query_samples=used_samples,
        pose_candidates=pose_candidates,
        scores=scores,
        source_name=source,
        save_path=save_path,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export query-student absolute pose init cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_query_student_pose_init(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        colmap_dir=args.colmap_dir,
        query_split=args.query_split,
        save_path=args.save_path,
        batch_size=args.batch_size,
        device=args.device,
        source_name=args.source_name,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
