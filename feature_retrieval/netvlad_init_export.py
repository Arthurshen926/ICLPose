#!/usr/bin/env python3
"""Export NetVLAD descriptor retrieval poses as real-init caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples, save_retrieval_init_entries  # noqa: E402
from feature_extract.student_feature_bank_init_export import (  # noqa: E402
    build_student_feature_bank_entries,
    search_student_feature_bank,
)


def _normalize_image_name(name: str) -> str:
    return str(name).replace("\\", "/")


def search_netvlad_descriptors(
    query_descriptors: torch.Tensor,
    train_descriptors: torch.Tensor,
    *,
    topk: int,
    query_image_names: list[str] | None = None,
    train_image_names: list[str] | None = None,
    exclude_self: bool = False,
):
    if not exclude_self:
        return search_student_feature_bank(query_descriptors, train_descriptors, topk=topk)
    if query_image_names is None or train_image_names is None:
        raise ValueError("query_image_names and train_image_names are required when exclude_self=True")
    if len(query_image_names) != int(torch.as_tensor(query_descriptors).shape[0]):
        raise ValueError("query_image_names length does not match query descriptors")
    if len(train_image_names) != int(torch.as_tensor(train_descriptors).shape[0]):
        raise ValueError("train_image_names length does not match train descriptors")

    query = F.normalize(torch.as_tensor(query_descriptors, dtype=torch.float32), dim=1, eps=1e-8)
    train = F.normalize(torch.as_tensor(train_descriptors, dtype=torch.float32), dim=1, eps=1e-8)
    if query.ndim != 2 or train.ndim != 2:
        raise ValueError(f"query/train descriptors must be 2D, got {tuple(query.shape)} and {tuple(train.shape)}")
    if query.shape[1] != train.shape[1]:
        raise ValueError(f"descriptor dims differ: query={query.shape[1]} train={train.shape[1]}")

    scores = query @ train.t()
    train_name_to_indices: dict[str, list[int]] = {}
    for idx, name in enumerate(train_image_names):
        train_name_to_indices.setdefault(_normalize_image_name(name), []).append(idx)
    for qidx, name in enumerate(query_image_names):
        for tidx in train_name_to_indices.get(_normalize_image_name(name), []):
            scores[qidx, tidx] = -torch.inf

    k = max(1, min(int(topk), int(train.shape[0])))
    top_scores, top_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    if not torch.isfinite(top_scores).all():
        raise ValueError("exclude_self=True left at least one query without enough finite retrieval candidates")
    return top_indices, top_scores


def export_netvlad_init(
    *,
    train_descs_path: str,
    query_descs_path: str,
    colmap_dir: str,
    train_split: str,
    query_split: str,
    save_path: str,
    topk: int = 10,
    source_name: str = "netvlad_top10",
    exclude_self: bool = False,
):
    train_desc = torch.from_numpy(np.load(train_descs_path).astype(np.float32))
    query_desc = torch.from_numpy(np.load(query_descs_path).astype(np.float32))
    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    if train_desc.shape[0] != len(train_samples):
        raise ValueError(f"train descriptors {train_desc.shape[0]} != train samples {len(train_samples)}")
    if query_desc.shape[0] != len(query_samples):
        raise ValueError(f"query descriptors {query_desc.shape[0]} != query samples {len(query_samples)}")
    indices, scores = search_netvlad_descriptors(
        query_desc,
        train_desc,
        topk=topk,
        query_image_names=[sample["image_name"] for sample in query_samples],
        train_image_names=[sample["image_name"] for sample in train_samples],
        exclude_self=exclude_self,
    )
    entries, stats = build_student_feature_bank_entries(
        query_samples=query_samples,
        train_samples=train_samples,
        indices=indices,
        scores=scores,
        source_name=source_name,
        save_path=None,
    )
    stats.update(
        {
            "method_requested": "netvlad_retrieval",
            "descriptor_dim": int(train_desc.shape[1]),
            "train_descs_path": str(train_descs_path),
            "query_descs_path": str(query_descs_path),
            "exclude_self": bool(exclude_self),
        }
    )
    save_retrieval_init_entries(entries, stats, save_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NetVLAD real-init cache")
    parser.add_argument("--train_descs", required=True)
    parser.add_argument("--query_descs", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_name", default="netvlad_top10")
    parser.add_argument("--exclude_self", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_netvlad_init(
        train_descs_path=args.train_descs,
        query_descs_path=args.query_descs,
        colmap_dir=args.colmap_dir,
        train_split=args.train_split,
        query_split=args.query_split,
        save_path=args.save_path,
        topk=args.topk,
        source_name=args.source_name,
        exclude_self=args.exclude_self,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
