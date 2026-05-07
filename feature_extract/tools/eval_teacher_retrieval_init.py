#!/usr/bin/env python3
"""Evaluate cached teacher/map descriptors as a top-K pose initializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.radio_loc_dataset import colmap_to_w2c, read_colmap_images
from feature_extract.train_impl import (
    TeacherFeatureStore,
    _global_descriptor_from_teacher_pair,
    build_all_records,
    descriptor_pose_retrieval_metrics,
    load_config,
    split_records,
)


def _pose_lookup(colmap_dir: str | Path) -> dict[str, torch.Tensor]:
    images_bin = Path(colmap_dir) / "images.bin"
    if not images_bin.is_file():
        raise FileNotFoundError(f"COLMAP images.bin not found: {images_bin}")
    lookup: dict[str, torch.Tensor] = {}
    for _image_id, meta in read_colmap_images(str(images_bin)).items():
        name = meta.name.replace("\\", "/")
        pose = torch.from_numpy(colmap_to_w2c(meta.qvec, meta.tvec)).float()
        lookup[name] = pose
        lookup.setdefault(Path(name).name, pose)
    return lookup


def _record_pose(record: dict, lookup: dict[str, torch.Tensor]) -> torch.Tensor | None:
    sample_name = record["sample_name"].replace("\\", "/")
    pose = lookup.get(sample_name)
    if pose is not None:
        return pose
    return lookup.get(Path(sample_name).name)


def _descriptors_for_records(records, teacher_store, pose_lookup, feature_source):
    descs = []
    poses = []
    kept = []
    for record in records:
        pose = _record_pose(record, pose_lookup)
        if pose is None:
            continue
        fine, coarse = teacher_store.load_pair(record["teacher_idx"])
        descs.append(_global_descriptor_from_teacher_pair(fine, coarse, feature_source=feature_source))
        poses.append(pose)
        kept.append(record["sample_name"])
    if not descs:
        raise RuntimeError("No records with both teacher features and COLMAP poses were found.")
    return torch.stack(descs, dim=0), torch.stack(poses, dim=0), kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-source", default="coarse", choices=["coarse", "fine", "fine_coarse"])
    parser.add_argument("--topk", default="1,5,10,20")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    teacher_store = TeacherFeatureStore(
        dataset_cfg["feature_dir"],
        cache_in_memory=bool(dataset_cfg.get("cache_teacher", False)),
    )
    all_records = build_all_records(dataset_cfg, teacher_store)
    train_records, val_records = split_records(all_records, dataset_cfg)
    pose_lookup = _pose_lookup(dataset_cfg["colmap_dir"])

    bank_desc, bank_pose, bank_names = _descriptors_for_records(
        train_records,
        teacher_store,
        pose_lookup,
        args.feature_source,
    )
    query_desc, query_pose, query_names = _descriptors_for_records(
        val_records,
        teacher_store,
        pose_lookup,
        args.feature_source,
    )
    topk = tuple(int(part) for part in args.topk.split(",") if part.strip())
    metrics = descriptor_pose_retrieval_metrics(
        query_desc,
        query_pose,
        bank_desc,
        bank_pose,
        topk=topk,
        prefix=f"teacher_{args.feature_source}",
    )
    result = {
        "config": str(args.config),
        "feature_source": args.feature_source,
        "num_bank": len(bank_names),
        "num_query": len(query_names),
        "metrics": {key: float(value.item()) for key, value in metrics.items()},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
