from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_extract.pose_init_export import build_pose_init_entries_from_predictions
from feature_extract.students.radio_query_student import FeatureBankPoseInitHead, RadioQueryStudent
from feature_extract.train_impl import (
    compute_pose_init_losses,
    farthest_point_anchor_centers,
    pose_feature_bank_from_map_renderer,
    pose_feature_bank_from_dataset,
    pose_anchors_from_dataset,
)
from feature_retrieval.evaluate_impl import summarize_full_pipeline_metrics


def test_feature_bank_pose_init_scores_query_descriptor_against_anchor_descriptors():
    anchors = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    anchor_desc = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    head = FeatureBankPoseInitHead(
        token_dim=4,
        anchor_centers=anchors,
        anchor_descriptors=anchor_desc,
        fine_dim=2,
        coarse_dim=1,
        hypotheses=1,
        temperature=0.05,
        residual_scale=0.0,
        projector="identity",
    )
    fine = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)
    coarse = torch.tensor([[[[0.0]]]], dtype=torch.float32)
    out = head(torch.zeros(1, 4), fine=fine, coarse=coarse)

    assert out["anchor_indices"].item() == 1
    assert torch.allclose(out["center"][0, 0], anchors[1], atol=1e-6)
    assert out["anchor_logits"][0, 1] > out["anchor_logits"][0, 0]


def test_feature_bank_pose_init_can_use_coarse_only_for_rough_localization():
    anchors = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    anchor_desc = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    head = FeatureBankPoseInitHead(
        token_dim=4,
        anchor_centers=anchors,
        anchor_descriptors=anchor_desc,
        fine_dim=2,
        coarse_dim=1,
        hypotheses=1,
        temperature=0.05,
        residual_scale=0.0,
        projector="identity",
        feature_source="coarse",
    )
    fine = torch.tensor([[[[0.0]], [[100.0]]]], dtype=torch.float32)
    coarse = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    out = head(torch.zeros(1, 4), fine=fine, coarse=coarse)

    assert out["query_descriptor"].shape == (1, 1)
    assert out["anchor_indices"].item() == 0
    assert torch.allclose(out["center"][0, 0], anchors[0], atol=1e-6)


def test_radio_query_student_query_channel_gate_exposes_query_feature_selection():
    model = RadioQueryStudent(
        feature_dim=16,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        stage_dims=(8, 12, 16, 20),
        output_hw=(8, 10),
        coarse_output_hw=(4, 5),
        input_hw=(64, 80),
        query_channel_gate_enabled=True,
        query_channel_gate_zero_init=False,
    )

    out = model(torch.randn(2, 3, 64, 80))
    gates = out["query_channel_weights"]

    assert gates["fine"].shape == (2, 16)
    assert gates["coarse"].shape == (2, 8)
    assert torch.isfinite(gates["fine"]).all()
    assert torch.isfinite(gates["coarse"]).all()
    assert gates["fine"].std().item() > 0.0
    assert gates["coarse"].std().item() > 0.0


def test_pose_init_loss_uses_best_hypothesis_and_reports_recall_metrics():
    gt = torch.eye(4).unsqueeze(0)
    outputs = {
        "pose_init": {
            "center": torch.tensor([[[1.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.5, 0.0, 0.0]]]),
            "rotmat": torch.eye(3).view(1, 1, 3, 3).expand(1, 3, 3, 3).clone(),
            "scores": torch.tensor([[0.0, 1.0, -1.0]]),
            "log_var": torch.zeros(1, 3, 2),
        }
    }

    loss, metrics = compute_pose_init_losses(
        outputs,
        {"pose_gt": gt},
        {
            "pose_init": {
                "enabled": True,
                "trans_weight": 1.0,
                "rot_weight": 1.0,
                "score_weight": 1.0,
                "uncertainty_weight": 0.1,
            }
        },
    )

    assert loss.item() > 0.0
    assert metrics["pose_init_best_idx"].item() == 1.0
    assert metrics["pose_init_best_trans_mm"].item() < 11.0
    assert metrics["pose_init_joint_5deg_1000mm"].item() == 100.0


def test_farthest_point_anchor_centers_are_deterministic_and_cover_scene_extent():
    centers = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    anchors_a = farthest_point_anchor_centers(centers, num_anchors=3)
    anchors_b = farthest_point_anchor_centers(centers, num_anchors=3)

    assert torch.allclose(anchors_a, anchors_b)
    assert anchors_a.shape == (3, 3)
    assert any(torch.allclose(anchor, torch.tensor([0.0, 0.0, 0.0])) for anchor in anchors_a)
    assert any(torch.allclose(anchor, torch.tensor([10.1, 0.0, 0.0])) for anchor in anchors_a)


def test_pose_anchors_from_dataset_can_use_all_training_views_as_pose_bank():
    def make_pose(center_x, yaw90=False):
        pose = torch.eye(4)
        if yaw90:
            pose[:3, :3] = torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        center = torch.tensor([center_x, 0.0, 0.0], dtype=torch.float32)
        pose[:3, 3] = -(pose[:3, :3] @ center)
        return pose

    class TinyPoseDataset:
        records = [
            {"sample_name": "a.png"},
            {"sample_name": "b.png"},
            {"sample_name": "c.png"},
        ]

        name_to_pose = {
            "a.png": make_pose(0.0),
            "b.png": make_pose(1.0, yaw90=True),
            "c.png": make_pose(2.0),
        }
        basename_to_pose = {}

    centers, rotmats = pose_anchors_from_dataset(TinyPoseDataset(), num_anchors=1, sampling="train_all")

    assert centers.shape == (3, 3)
    assert rotmats.shape == (3, 3, 3)
    assert torch.allclose(centers[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(rotmats[1], TinyPoseDataset.name_to_pose["b.png"][:3, :3])


def test_pose_feature_bank_from_dataset_uses_teacher_feature_global_descriptors():
    def make_pose(center_x):
        pose = torch.eye(4)
        pose[0, 3] = -float(center_x)
        return pose

    class TinyPoseDataset:
        records = [
            {"sample_name": "a.png", "teacher_idx": 0},
            {"sample_name": "b.png", "teacher_idx": 1},
        ]

        name_to_pose = {"a.png": make_pose(0.0), "b.png": make_pose(1.0)}
        basename_to_pose = {}

    class TinyTeacherStore:
        def load_pair(self, idx):
            fine = torch.full((2, 2, 3), float(idx + 1))
            coarse = torch.full((1, 1, 2), float(idx + 3))
            return fine, coarse

    centers, rotmats, descriptors = pose_feature_bank_from_dataset(
        TinyPoseDataset(),
        TinyTeacherStore(),
        num_anchors=1,
        sampling="train_all",
    )

    assert centers.shape == (2, 3)
    assert rotmats.shape == (2, 3, 3)
    assert descriptors.shape == (2, 3)
    assert torch.allclose(descriptors[0], torch.tensor([1.0, 1.0, 3.0]))
    assert torch.allclose(descriptors[1], torch.tensor([2.0, 2.0, 4.0]))


def test_pose_feature_bank_from_dataset_can_build_coarse_only_descriptors():
    def make_pose(center_x):
        pose = torch.eye(4)
        pose[0, 3] = -float(center_x)
        return pose

    class TinyPoseDataset:
        records = [
            {"sample_name": "a.png", "teacher_idx": 0},
            {"sample_name": "b.png", "teacher_idx": 1},
        ]

        name_to_pose = {"a.png": make_pose(0.0), "b.png": make_pose(1.0)}
        basename_to_pose = {}

    class TinyTeacherStore:
        def load_pair(self, idx):
            fine = torch.full((2, 2, 3), float(idx + 1))
            coarse = torch.full((1, 1, 2), float(idx + 3))
            return fine, coarse

    _centers, _rotmats, descriptors = pose_feature_bank_from_dataset(
        TinyPoseDataset(),
        TinyTeacherStore(),
        num_anchors=1,
        sampling="train_all",
        feature_source="coarse",
    )

    assert descriptors.shape == (2, 1)
    assert torch.allclose(descriptors[:, 0], torch.tensor([3.0, 4.0]))


def test_pose_feature_bank_from_map_renderer_uses_reconstructed_map_features():
    def make_pose(center_x):
        pose = torch.eye(4)
        pose[0, 3] = -float(center_x)
        return pose

    class TinyPoseDataset:
        records = [
            {"sample_name": "a.png"},
            {"sample_name": "b.png"},
        ]

        name_to_pose = {"a.png": make_pose(0.0), "b.png": make_pose(1.0)}
        basename_to_pose = {}

    class TinyMapRenderer:
        def _render_single(self, sample_name, require_grad=False):
            idx = 0 if sample_name == "a.png" else 1
            fine = torch.full((1, 2, 1, 1), float(idx + 1))
            coarse = torch.full((1, 1, 1, 1), float(idx + 5))
            mask = torch.ones(1, 1, 1, 1)
            filler = torch.empty(0)
            return fine, fine, coarse, mask, filler, filler, filler, filler

    centers, rotmats, descriptors = pose_feature_bank_from_map_renderer(
        TinyPoseDataset(),
        TinyMapRenderer(),
        num_anchors=1,
        sampling="train_all",
        feature_source="coarse",
    )

    assert centers.shape == (2, 3)
    assert rotmats.shape == (2, 3, 3)
    assert descriptors.shape == (2, 1)
    assert torch.allclose(descriptors[:, 0], torch.tensor([5.0, 6.0]))


def test_full_pipeline_metrics_report_required_recalls_and_gain():
    metrics = summarize_full_pipeline_metrics(
        init_rot_errs=[6.0, 0.5, 0.2],
        init_trans_errs=[1200.0, 80.0, 40.0],
        final_rot_errs=[4.0, 0.4, 0.2],
        final_trans_errs=[200.0, 40.0, 30.0],
    )

    assert metrics["protocol"] == "real_init_full_pipeline"
    assert metrics["init"]["joint_5deg_1000mm"] == 66.66666666666666
    assert metrics["final"]["joint_1deg_50mm"] == 66.66666666666666
    assert metrics["final"]["joint_1deg_100mm"] == 66.66666666666666
    assert metrics["final"]["joint_2deg_100mm"] == 66.66666666666666
    assert metrics["final"]["joint_5deg_250mm"] == 100.0
    assert metrics["gain"]["trans_median_mm"] == 40.0


def test_pose_init_predictions_export_to_retrieval_init_schema(tmp_path):
    query_samples = [
        {
            "img_id": 7,
            "image_name": "seq1/frame00007.png",
            "image_stem": "seq1_frame00007",
            "pose_w2c": torch.eye(4).numpy(),
        }
    ]
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1).numpy()
    scores = torch.tensor([[0.1, 2.0, -0.5]]).numpy()

    entries, stats = build_pose_init_entries_from_predictions(
        query_samples=query_samples,
        pose_candidates=poses,
        scores=scores,
        source_name="pose_init_test",
        save_path=str(tmp_path / "init_pose.npz"),
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(tmp_path / "init_pose.npz"))

    assert stats["method_used"] == "pose_init_test"
    assert stats["retrieval_topk_requested"] == 3
    assert entries[0]["init_source"] == "pose_init_test"
    assert entries[0]["retrieval_frame_id"] == -1
    assert entries[0]["candidate_valid_mask"].tolist() == [True, True, True]
    assert loaded_entries[0]["query_img_id"] == 7
    assert loaded_entries[0]["pose_init_candidates"].shape == (3, 4, 4)
    assert loaded_entries[0]["retrieval_scores_candidates"][0] == 2.0
    assert loaded_stats["method_used"] == "pose_init_test"
