from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_retrieval.netvlad_init_export import search_netvlad_descriptors


def test_search_netvlad_descriptors_can_exclude_same_image_name():
    query = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    bank = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )

    indices, scores = search_netvlad_descriptors(
        query,
        bank,
        topk=2,
        query_image_names=["seq/frame001.png"],
        train_image_names=["seq/frame001.png", "seq/frame002.png", "seq/frame003.png"],
        exclude_self=True,
    )

    assert indices.tolist() == [[1, 2]]
    assert torch.isfinite(scores).all()
    assert scores[0, 0] > scores[0, 1]
