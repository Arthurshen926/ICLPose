#!/usr/bin/env python3
"""Export NetVLAD descriptor retrieval poses as real-init caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples, save_retrieval_init_entries  # noqa: E402
from feature_extract.student_feature_bank_init_export import (  # noqa: E402
    build_student_feature_bank_entries,
    search_student_feature_bank,
)


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
):
    train_desc = torch.from_numpy(np.load(train_descs_path).astype(np.float32))
    query_desc = torch.from_numpy(np.load(query_descs_path).astype(np.float32))
    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    if train_desc.shape[0] != len(train_samples):
        raise ValueError(f"train descriptors {train_desc.shape[0]} != train samples {len(train_samples)}")
    if query_desc.shape[0] != len(query_samples):
        raise ValueError(f"query descriptors {query_desc.shape[0]} != query samples {len(query_samples)}")
    indices, scores = search_student_feature_bank(query_desc, train_desc, topk=topk)
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
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
