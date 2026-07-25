from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_p1_mapper_direct import (
    CURRENT_P1_MAPPER_DIRECT_PROPOSAL_FORMAT,
    CURRENT_P1_MAPPER_DIRECT_TARGET_FORMAT,
    CurrentP1MapperDirectTargets,
    CurrentP1MapperRuntime,
    current_p1_mapper_direct_loss,
    dense_descriptor_anchor_distillation_loss,
    fixed_p1_candidate_log_posteriors,
    group_current_p1_runtime_rows_by_query,
    load_current_p1_mapper_direct_supervision,
    sample_mapper_descriptors_at_p1_xy,
)
from feature_extract.vfm.localization.landmark_hybrid import save_landmark_index_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _runtime() -> CurrentP1MapperRuntime:
    return CurrentP1MapperRuntime(
        source_point_ids=np.asarray([4, 8, 12, 16], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png", "query/b.png", "query/b.png"]),
        xy=np.asarray([[3.0, 2.0], [9.0, 7.0], [2.0, 4.0], [8.0, 5.0]], dtype=np.float32),
        candidate_bank_rows=np.asarray([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int64),
        candidate_prior_probabilities=np.asarray(
            [[0.45, 0.30, 0.15], [0.45, 0.30, 0.15], [0.45, 0.30, 0.15], [0.45, 0.30, 0.15]],
            dtype=np.float32,
        ),
        null_probabilities=np.full((4,), 0.10, dtype=np.float32),
        metadata={"format": "target_free_test"},
    )


def _targets() -> CurrentP1MapperDirectTargets:
    positive = np.asarray(
        [[True, False, False], [True, False, False], [False, True, False], [False, True, False]],
        dtype=bool,
    )
    hard = np.asarray(
        [[False, True, False], [False, True, False], [True, False, False], [True, False, False]],
        dtype=bool,
    )
    return CurrentP1MapperDirectTargets(
        positive_mask=positive,
        hard_negative_mask=hard,
        group_hard_mask=np.ones((4,), dtype=bool),
        hard_mode_ids=np.asarray([[7], [7], [11], [11]], dtype=np.int64),
        hard_mode_candidate_mask=hard[:, None, :],
        metadata={"training_only_target_artifact": True},
    )


def test_exact_p1_sampling_uses_endpoint_coordinate_frame() -> None:
    descriptor_maps = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    descriptor_maps[0, :, 0, 0] = torch.tensor([3.0, 4.0])
    descriptor_maps[0, :, 1, 1] = torch.tensor([5.0, 12.0])
    sampled = sample_mapper_descriptors_at_p1_xy(
        descriptor_maps,
        torch.tensor([[[0.0, 0.0], [9.0, 9.0]]]),
        torch.tensor([[10.0, 10.0]]),
    )
    torch.testing.assert_close(sampled[0, 0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(sampled[0, 1], torch.tensor([5.0 / 13.0, 12.0 / 13.0]))
    with pytest.raises(ValueError, match="outside"):
        sample_mapper_descriptors_at_p1_xy(
            descriptor_maps,
            torch.tensor([[[10.0, 0.0]]]),
            torch.tensor([[10.0, 10.0]]),
        )


def test_dense_anchor_distillation_preserves_non_p1_tokens_only() -> None:
    torch.manual_seed(4)
    teacher = torch.randn((1, 3, 5, 5), dtype=torch.float32)
    points = torch.tensor([[[2.0, 2.0]]], dtype=torch.float32)
    image_sizes = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

    p1_only_change = teacher.clone()
    p1_only_change[:, :, 1:4, 1:4] *= -1.0
    ignored_loss, ignored_metrics = dense_descriptor_anchor_distillation_loss(
        student_descriptor_maps=p1_only_change,
        teacher_descriptor_maps=teacher,
        p1_xy=points,
        image_sizes=image_sizes,
        exclusion_radius_tokens=1,
    )
    assert ignored_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert ignored_metrics["anchor_token_count"] == pytest.approx(16.0)
    assert ignored_metrics["anchor_excluded_token_count"] == pytest.approx(9.0)

    student = teacher.clone().requires_grad_(True)
    student.data[:, :, 0, 0] *= -1.0
    loss, metrics = dense_descriptor_anchor_distillation_loss(
        student_descriptor_maps=student,
        teacher_descriptor_maps=teacher,
        p1_xy=points,
        image_sizes=image_sizes,
        exclusion_radius_tokens=1,
    )
    assert loss.item() > 0.0
    assert metrics["anchor_descriptor_cosine_mean"] < 1.0
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[:, :, 0, 0]) > 0
    assert teacher.grad is None


def test_candidate_posteriors_keep_an_explicit_fixed_null_component() -> None:
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    log_posterior, similarity = fixed_p1_candidate_log_posteriors(
        query_descriptors=query,
        candidate_descriptors=candidates,
        candidate_prior_probabilities=torch.tensor([[0.45, 0.45]]),
        null_probabilities=torch.tensor([0.10]),
        temperature=0.1,
    )
    assert float(similarity[0, 0]) > float(similarity[0, 1])
    candidate_mass = log_posterior.exp().sum(dim=1)
    assert bool(torch.all(candidate_mass < 1.0))
    assert float(log_posterior[0, 0]) > float(log_posterior[0, 1])


def test_direct_loss_uses_exact_modes_and_backpropagates() -> None:
    runtime = _runtime()
    targets = _targets()
    bank = torch.eye(3, dtype=torch.float32)
    candidate_descriptors = bank[torch.from_numpy(runtime.candidate_bank_rows)]
    query = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    result = current_p1_mapper_direct_loss(
        query_descriptors=query,
        candidate_descriptors=candidate_descriptors,
        runtime=runtime,
        targets=targets,
        temperature=0.1,
        coherent_margin=0.05,
        coherent_margin_weight=0.5,
        coherent_min_mode_rows=2,
    )
    assert result.metrics["direct_positive_row_count"] == pytest.approx(4.0)
    assert result.metrics["direct_coherent_mode_count"] == pytest.approx(2.0)
    assert result.metrics["direct_coherent_mode_row_count"] == pytest.approx(4.0)
    assert result.metrics["direct_candidate_top1_positive_rate"] == pytest.approx(1.0)
    result.total_loss.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.count_nonzero(query.grad) > 0


def test_runtime_grouping_is_target_free_and_targets_subset_by_exact_rows() -> None:
    runtime = _runtime()
    targets = _targets()
    groups = group_current_p1_runtime_rows_by_query(runtime, targets)
    assert set(groups) == {"query/a.png", "query/b.png"}
    subset = np.asarray([2, 3], dtype=np.int64)
    assert runtime.subset(subset).query_ids.tolist() == ["query/b.png", "query/b.png"]
    assert targets.subset(subset).hard_mode_ids[:, 0].tolist() == [11, 11]


def test_target_subset_without_modes_keeps_positive_only_loss_valid() -> None:
    runtime = _runtime().subset(np.asarray([0, 1], dtype=np.int64))
    targets = CurrentP1MapperDirectTargets(
        positive_mask=np.asarray([[True, False, False], [True, False, False]], dtype=bool),
        hard_negative_mask=np.zeros((2, 3), dtype=bool),
        group_hard_mask=np.zeros((2,), dtype=bool),
        hard_mode_ids=np.full((2, 1), -1, dtype=np.int64),
        hard_mode_candidate_mask=np.zeros((2, 1, 3), dtype=bool),
        metadata={"training_only_target_artifact": True},
    )
    bank = torch.eye(3, dtype=torch.float32)
    result = current_p1_mapper_direct_loss(
        query_descriptors=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        candidate_descriptors=bank[torch.from_numpy(runtime.candidate_bank_rows)],
        runtime=runtime,
        targets=targets,
        temperature=0.1,
        coherent_margin=0.05,
        coherent_margin_weight=0.5,
        coherent_min_mode_rows=2,
    )
    assert result.metrics["direct_coherent_mode_count"] == pytest.approx(0.0)
    assert result.metrics["direct_coherent_margin_loss"] == pytest.approx(0.0)


def _write_toy_direct_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    bank_path = tmp_path / "bank.npz"
    bank = LandmarkMapIndex(
        track_ids=np.asarray([100, 200, 300], dtype=np.int64),
        xyz=np.zeros((3, 3), dtype=np.float64),
        features=np.eye(3, dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64),
        observation_image_ids=(("map/a.png",), ("map/b.png",), ("map/c.png",)),
    )
    save_landmark_index_npz(
        bank,
        bank_path,
        metadata={
            "descriptor_space_id": "space-a",
            "projection_space_id": "projection-a",
        },
    )
    proposal_path = tmp_path / "proposals.npz"
    proposal_metadata = {
        "format": CURRENT_P1_MAPPER_DIRECT_PROPOSAL_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
    }
    np.savez(
        proposal_path,
        layout_row_indices=np.asarray([0, 1], dtype=np.int64),
        source_point_ids=np.asarray([20, 21], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png"]),
        xy=np.asarray([[2.0, 3.0], [7.0, 8.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[100, 200, 300], [100, 200, 300]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        candidate_prior_probabilities=np.asarray([[0.4, 0.3, 0.2], [0.4, 0.3, 0.2]], dtype=np.float32),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(proposal_metadata, sort_keys=True)),
    )
    target_path = tmp_path / "targets.npz"
    target_metadata = {
        "format": CURRENT_P1_MAPPER_DIRECT_TARGET_FORMAT,
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "split_names": ["train"],
        "inputs": {
            "proposals_sha256": file_sha256_short(proposal_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "descriptor_space_id": "space-a",
            "projection_space_id": "projection-a",
        },
    }
    np.savez(
        target_path,
        selected_rows=np.asarray([1, 0], dtype=np.int64),
        selected_columns=np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png"]),
        positive_mask_TARGET_ONLY=np.asarray([[True, False, False], [True, False, False]], dtype=bool),
        hard_negative_mask_TARGET_ONLY=np.asarray([[False, True, False], [False, True, False]], dtype=bool),
        group_hard_mask_TARGET_ONLY=np.asarray([True, True], dtype=bool),
        hard_mode_ids_TARGET_ONLY=np.asarray([[5], [5]], dtype=np.int64),
        hard_mode_candidate_mask_TARGET_ONLY=np.asarray(
            [[[False, True, False]], [[False, True, False]]], dtype=bool
        ),
        metadata_json=np.asarray(json.dumps(target_metadata, sort_keys=True)),
    )
    return proposal_path, target_path, bank_path


def test_loader_keeps_target_fields_out_of_runtime_and_checks_bank_lineage(tmp_path: Path) -> None:
    proposal_path, target_path, bank_path = _write_toy_direct_artifacts(tmp_path)
    runtime, targets, bank, summary = load_current_p1_mapper_direct_supervision(
        proposals_path=proposal_path,
        targets_path=target_path,
        projected_landmark_bank_path=bank_path,
    )
    assert runtime.query_ids.tolist() == ["query/a.png", "query/a.png"]
    assert not hasattr(runtime, "positive_mask")
    assert targets.positive_mask.shape == (2, 3)
    assert bank.shape == (3, 3)
    assert summary["post_training_requirement"] == "rebuild_projected_observation_bank_before_runtime_eval"

    with np.load(proposal_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["ground_truth_label"] = np.asarray([0, 0], dtype=np.int64)
    bad_proposal = tmp_path / "bad_proposals.npz"
    np.savez(bad_proposal, **arrays)
    with pytest.raises(ValueError, match="target fields"):
        load_current_p1_mapper_direct_supervision(
            proposals_path=bad_proposal,
            targets_path=target_path,
            projected_landmark_bank_path=bank_path,
        )
