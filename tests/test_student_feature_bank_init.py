from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_extract.student_feature_bank_init_export import (
    build_student_feature_bank_entries,
    extract_cached_student_descriptors,
    pool_student_descriptor,
    search_student_feature_bank,
)


def test_pool_student_descriptor_concatenates_normalized_fine_and_coarse_means():
    outputs = {
        "fine": torch.tensor([[[[1.0]], [[0.0]]]]),
        "coarse": torch.tensor([[[[0.0]], [[1.0]]]]),
    }

    desc = pool_student_descriptor(outputs)

    assert desc.shape == (1, 4)
    assert torch.allclose(desc.norm(dim=1), torch.ones(1), atol=1e-6)
    assert desc[0, 0] > 0
    assert desc[0, 3] > 0


def test_search_student_feature_bank_returns_topk_cosine_matches():
    query = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    bank = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.7, 0.3],
        ],
        dtype=torch.float32,
    )

    indices, scores = search_student_feature_bank(query, bank, topk=2)

    assert indices.tolist() == [[1, 2]]
    assert scores[0, 0] > scores[0, 1]


def test_build_student_feature_bank_entries_write_retrieval_schema(tmp_path):
    query_samples = [
        {"img_id": 5, "image_name": "seq/frame5.png", "image_stem": "seq_frame5"},
    ]
    train_samples = [
        {"img_id": 1, "image_name": "seq/frame1.png", "pose_w2c": np.eye(4, dtype=np.float32)},
        {"img_id": 2, "image_name": "seq/frame2.png", "pose_w2c": np.eye(4, dtype=np.float32) * 2.0},
    ]
    indices = torch.tensor([[1, 0]])
    scores = torch.tensor([[0.8, 0.2]])

    entries, stats = build_student_feature_bank_entries(
        query_samples=query_samples,
        train_samples=train_samples,
        indices=indices,
        scores=scores,
        source_name="student_bank_test",
        save_path=str(tmp_path / "student_bank_init.npz"),
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(tmp_path / "student_bank_init.npz"))

    assert stats["method_used"] == "student_bank_test"
    assert stats["retrieval_topk_requested"] == 2
    assert entries[0]["retrieval_frame_id"] == 2
    assert entries[0]["retrieval_image_name"] == "seq/frame2.png"
    assert loaded_entries[0]["pose_init_candidates"].shape == (2, 4, 4)
    assert loaded_entries[0]["retrieval_frame_ids_candidates"].tolist() == [2, 1]
    assert loaded_stats["num_query_samples"] == 1


def test_extract_cached_student_descriptors_loads_by_img_id(tmp_path):
    feature_dir = tmp_path / "features"
    fine_dir = feature_dir / "fine_geo"
    coarse_dir = feature_dir / "coarse_sem"
    fine_dir.mkdir(parents=True)
    coarse_dir.mkdir(parents=True)
    torch.save(torch.ones(2, 2, 2), fine_dir / "rgb_7_fine_geo_2x2x2.pt")
    torch.save(torch.zeros(1, 1, 1), coarse_dir / "rgb_7_coarse_sem_1x1x1.pt")
    torch.save(torch.zeros(2, 2, 2), fine_dir / "rgb_8_fine_geo_2x2x2.pt")
    torch.save(torch.ones(1, 1, 1), coarse_dir / "rgb_8_coarse_sem_1x1x1.pt")
    samples = [
        {"img_id": 7, "image_name": "a.png", "image_stem": "a"},
        {"img_id": 8, "image_name": "b.png", "image_stem": "b"},
    ]

    used, desc = extract_cached_student_descriptors(str(feature_dir), samples)

    assert [sample["img_id"] for sample in used] == [7, 8]
    assert desc.shape == (2, 3)
    assert torch.allclose(desc.norm(dim=1), torch.ones(2), atol=1e-6)
    assert desc[0, 0] > desc[0, 2]
    assert desc[1, 2] > desc[1, 0]
