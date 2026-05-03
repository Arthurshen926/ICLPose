#!/usr/bin/env python3
"""Export RADIO summary-token retrieval poses as real-init caches."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    save_retrieval_init_entries,
)
from feature_extract.student_feature_bank_init_export import (  # noqa: E402
    build_student_feature_bank_entries,
    search_student_feature_bank,
)


def _safe_load_vector(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return torch.as_tensor(value, dtype=torch.float32).view(-1)


def discover_summary_files(feature_dir: str) -> dict[int, Path]:
    summary_dir = Path(feature_dir) / "summary"
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"summary directory not found: {summary_dir}")
    mapping: dict[int, Path] = {}
    pattern = re.compile(r"rgb_(\d+)_summary_.*\.pt$")
    for path in sorted(summary_dir.glob("*.pt")):
        match = pattern.match(path.name)
        if match:
            mapping[int(match.group(1))] = path
    if not mapping:
        raise RuntimeError(f"No RADIO summary files found in {summary_dir}")
    return mapping


def extract_radio_summary_descriptors(
    feature_dir: str,
    samples: Sequence[Dict],
) -> tuple[list[Dict], torch.Tensor]:
    files = discover_summary_files(feature_dir)
    used_samples = []
    descriptors = []
    for sample in samples:
        path = files.get(int(sample["img_id"]))
        if path is None:
            continue
        descriptors.append(F.normalize(_safe_load_vector(path).unsqueeze(0), dim=1, eps=1e-8))
        used_samples.append(sample)
    if not descriptors:
        raise RuntimeError(f"No RADIO summary descriptors matched split samples in {feature_dir}")
    return used_samples, torch.cat(descriptors, dim=0)


def export_radio_summary_init(
    *,
    feature_dir: str,
    colmap_dir: str,
    train_split: str,
    query_split: str,
    save_path: str,
    topk: int = 10,
    source_name: str = "radio_summary_topk",
) -> Dict:
    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    used_train, bank_desc = extract_radio_summary_descriptors(feature_dir, train_samples)
    used_query, query_desc = extract_radio_summary_descriptors(feature_dir, query_samples)
    indices, scores = search_student_feature_bank(query_desc, bank_desc, topk=topk)
    entries, stats = build_student_feature_bank_entries(
        query_samples=used_query,
        train_samples=used_train,
        indices=indices,
        scores=scores,
        source_name=source_name,
        save_path=None,
    )
    stats.update(
        {
            "method_requested": "radio_summary_retrieval",
            "descriptor_dim": int(bank_desc.shape[1]),
            "descriptor_source": str(Path(feature_dir) / "summary"),
            "num_query_samples_requested": int(len(query_samples)),
            "num_train_samples_requested": int(len(train_samples)),
        }
    )
    save_retrieval_init_entries(entries, stats, save_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RADIO summary retrieval real-init cache")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_name", default="radio_summary_top10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_radio_summary_init(
        feature_dir=args.feature_dir,
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
