from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_extract.coarse_pose_bank_init_export import (
    build_coarse_pose_bank_entries,
    extract_cached_coarse_descriptors,
    extract_rendered_map_coarse_descriptors_from_renderer,
    pool_coarse_descriptor,
    search_coarse_pose_bank,
    summarize_topk_pose_recall,
)


def _pose_at(center_x_m: float, yaw_deg: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    yaw = np.deg2rad(float(yaw_deg))
    c = np.cos(yaw)
    s = np.sin(yaw)
    pose[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose[0, 3] = -float(center_x_m)
    return pose


def test_pool_coarse_descriptor_normalizes_spatial_mean():
    feature = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    desc = pool_coarse_descriptor(feature)

    assert desc.shape == (1, 2)
    assert torch.allclose(desc.norm(dim=1), torch.ones(1), atol=1e-6)
    assert desc[0, 0] > 0.99
    assert desc[0, 1] == 0.0


def test_pool_coarse_descriptor_uses_masked_mean_when_mask_is_provided():
    feature = torch.tensor(
        [
            [[1.0, 9.0], [1.0, 9.0]],
            [[2.0, 0.0], [2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    desc = pool_coarse_descriptor(feature, mask=mask)

    expected = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0]]), dim=1)
    assert torch.allclose(desc, expected, atol=1e-6)


def test_search_coarse_pose_bank_returns_sorted_matches():
    query = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    bank = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.9],
            [0.2, 0.7],
        ],
        dtype=torch.float32,
    )

    indices, scores = search_coarse_pose_bank(query, bank, topk=2)

    assert indices.tolist() == [[1, 2]]
    assert scores[0, 0] > scores[0, 1]


def test_extract_cached_coarse_descriptors_loads_coarse_sem_by_img_id(tmp_path):
    feature_dir = tmp_path / "features"
    coarse_dir = feature_dir / "coarse_sem"
    coarse_dir.mkdir(parents=True)
    torch.save(torch.ones(2, 2, 2), coarse_dir / "rgb_7_coarse_sem_2x2x2.pt")
    torch.save(torch.tensor([[[0.0]], [[3.0]]]), coarse_dir / "rgb_8_coarse_sem_2x1x1.pt")
    samples = [
        {"img_id": 7, "image_name": "a.png", "image_stem": "a"},
        {"img_id": 8, "image_name": "b.png", "image_stem": "b"},
        {"img_id": 9, "image_name": "missing.png", "image_stem": "missing"},
    ]

    used, desc = extract_cached_coarse_descriptors(str(feature_dir), samples)

    assert [sample["img_id"] for sample in used] == [7, 8]
    assert desc.shape == (2, 2)
    assert torch.allclose(desc.norm(dim=1), torch.ones(2), atol=1e-6)
    assert desc[0, 0] == pytest.approx(desc[0, 1])
    assert desc[1, 1] > 0.99


def test_extract_rendered_map_coarse_descriptors_uses_map_side_renderer():
    class FakeRenderer:
        def _render_single(self, image_name, require_grad=False):
            assert require_grad is False
            if image_name == "a.png":
                coarse = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)
            else:
                coarse = torch.tensor([[[[0.0]], [[2.0]]]], dtype=torch.float32)
            mask = torch.ones(1, 1, 1, 1)
            return None, None, coarse, mask, None, None, None, None

    samples = [
        {"img_id": 7, "image_name": "a.png", "image_stem": "a"},
        {"img_id": 8, "image_name": "b.png", "image_stem": "b"},
    ]

    used, desc = extract_rendered_map_coarse_descriptors_from_renderer(FakeRenderer(), samples)

    assert [sample["image_name"] for sample in used] == ["a.png", "b.png"]
    assert desc.shape == (2, 2)
    assert desc[0, 0] > 0.99
    assert desc[1, 1] > 0.99


def test_build_coarse_pose_bank_entries_exports_existing_retrieval_schema(tmp_path):
    query_samples = [
        {"img_id": 5, "image_name": "seq/frame5.png", "image_stem": "seq_frame5"},
    ]
    train_samples = [
        {"img_id": 1, "image_name": "seq/frame1.png", "pose_w2c": _pose_at(1.0)},
        {"img_id": 2, "image_name": "seq/frame2.png", "pose_w2c": _pose_at(2.0)},
    ]
    indices = torch.tensor([[1, 0]])
    scores = torch.tensor([[0.8, 0.2]])
    save_path = tmp_path / "coarse_bank_init.npz"

    entries, stats = build_coarse_pose_bank_entries(
        query_samples=query_samples,
        train_samples=train_samples,
        indices=indices,
        scores=scores,
        source_name="coarse_bank_test",
        save_path=str(save_path),
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert stats["method_requested"] == "coarse_pose_bank"
    assert stats["method_used"] == "coarse_bank_test"
    assert entries[0]["retrieval_frame_id"] == 2
    assert loaded_entries[0]["pose_init_candidates"].shape == (2, 4, 4)
    assert loaded_entries[0]["retrieval_frame_ids_candidates"].tolist() == [2, 1]
    assert list(loaded_entries[0]["retrieval_image_names_candidates"]) == ["seq/frame2.png", "seq/frame1.png"]
    assert loaded_stats["num_query_samples"] == 1


def test_summarize_topk_pose_recall_uses_best_valid_candidate_within_k():
    entries = [
        {
            "query_image_name": "q1.png",
            "pose_init": _pose_at(0.5),
            "pose_init_candidates": np.stack([_pose_at(0.5), _pose_at(0.05)], axis=0),
            "candidate_valid_mask": np.array([True, True], dtype=bool),
        },
        {
            "query_image_name": "q2.png",
            "pose_init": _pose_at(0.8, yaw_deg=4.0),
            "pose_init_candidates": np.stack([_pose_at(0.8, yaw_deg=4.0), _pose_at(0.2, yaw_deg=0.5)], axis=0),
            "candidate_valid_mask": np.array([True, False], dtype=bool),
        },
    ]
    gt = {
        "q1.png": _pose_at(0.0),
        "q2.png": _pose_at(0.0),
    }

    metrics = summarize_topk_pose_recall(entries, gt, topks=(1, 2))

    assert metrics["num_samples"] == 2
    assert metrics["top1_trans_median"] == pytest.approx(650.0)
    assert metrics["top2_trans_median"] == pytest.approx(425.0)
    assert metrics["top1_joint_1deg_100mm"] == 0.0
    assert metrics["top2_joint_1deg_100mm"] == 50.0
    assert metrics["top2_joint_5deg_250mm"] == 50.0
