from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import feature_extract.tools.vfm.train_candidate_pose_llr as pose_llr_train
from feature_extract.tools.vfm.build_candidate_pose_llr_train_pairs import (
    _load_hypothesis_poses,
    validate_train_pair_splits,
    validate_train_pair_hypothesis_sources_match_reference,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    CandidatePoseLLRRuntime,
    CandidateSpecificPoseLLR,
    _alike_shift_correlations,
    _crop_subpixel_grid_tokens,
    bounded_log_likelihood_ratio,
    fixed_candidate_view_mixture_log_ratio,
    grouped_hypothesis_semantic_manifest,
    pairwise_pose_margin_loss,
    query_grouped_pose_margin_loss,
    query_grouped_pose_permutation_soft_hard_margin_loss,
    query_grouped_pose_soft_hard_margin_loss,
    score_candidate_pose_batch,
    validate_grouped_hypothesis_semantic_match,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    crop_anchor_aligned_grid_tokens,
)


def test_zero_llr_keeps_fixed_candidate_null_mixture_neutral() -> None:
    edge_llr = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    edge_usable = torch.ones_like(edge_llr, dtype=torch.bool)
    candidate_weights = torch.tensor(
        [[[0.7, 0.3], [0.4, 0.6]]] * 3, dtype=torch.float32
    )
    candidate_priors = torch.tensor(
        [[0.45, 0.35], [0.45, 0.35], [0.45, 0.35]], dtype=torch.float32
    )
    null_priors = torch.full((3,), 0.2, dtype=torch.float32)

    point_log_ratio, candidate_log_ratio = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=edge_usable,
        candidate_view_weights=candidate_weights,
        candidate_probabilities=candidate_priors,
        null_probabilities=null_priors,
        missing_edge_log_likelihood_ratio=0.0,
    )

    torch.testing.assert_close(point_log_ratio, torch.zeros_like(point_log_ratio))
    torch.testing.assert_close(candidate_log_ratio, torch.zeros_like(candidate_log_ratio))


def test_invalid_view_cannot_create_pose_dependent_boost() -> None:
    edge_llr = torch.tensor([[[[8.0, 0.0]]]], dtype=torch.float32)
    edge_usable = torch.tensor([[[[False, True]]]])
    point_log_ratio, candidate_log_ratio = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=edge_llr,
        edge_usable=edge_usable,
        candidate_view_weights=torch.tensor([[[0.5, 0.5]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
        missing_edge_log_likelihood_ratio=0.0,
    )

    torch.testing.assert_close(candidate_log_ratio, torch.zeros_like(candidate_log_ratio))
    torch.testing.assert_close(point_log_ratio, torch.zeros_like(point_log_ratio))


def test_fixed_probability_mixture_stays_float32_under_half_edge_scores() -> None:
    point_log_ratio, candidate_log_ratio = fixed_candidate_view_mixture_log_ratio(
        edge_log_likelihood_ratios=torch.zeros((1, 1, 1, 1), dtype=torch.float16),
        edge_usable=torch.ones((1, 1, 1, 1), dtype=torch.bool),
        candidate_view_weights=torch.tensor([[[1.0]]], dtype=torch.float32),
        candidate_probabilities=torch.tensor([[0.9]], dtype=torch.float32),
        null_probabilities=torch.tensor([0.1], dtype=torch.float32),
        missing_edge_log_likelihood_ratio=0.0,
    )
    assert point_log_ratio.dtype is torch.float32
    assert candidate_log_ratio.dtype is torch.float32


def test_bounded_llr_and_pairwise_margin_prefer_correct_pose() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    bounded = bounded_log_likelihood_ratio(raw, max_abs_log_ratio=2.5)
    assert float(torch.max(torch.abs(bounded))) <= 2.5
    torch.testing.assert_close(bounded[1], torch.tensor(0.0))

    preferred, preferred_metrics = pairwise_pose_margin_loss(
        correct_scores=torch.tensor([2.0, 1.0]),
        coherent_wrong_scores=torch.tensor([0.0, 0.5]),
        margin=0.25,
    )
    reversed_loss, _ = pairwise_pose_margin_loss(
        correct_scores=torch.tensor([0.0, 0.5]),
        coherent_wrong_scores=torch.tensor([2.0, 1.0]),
        margin=0.25,
    )
    assert float(preferred) < float(reversed_loss)
    assert preferred_metrics["pairwise_correct_win_fraction"] == 1.0


def test_train_pair_builder_rejects_non_train_target_rows() -> None:
    with pytest.raises(ValueError, match="validation/test"):
        validate_train_pair_splits(
            query_ids=np.asarray(["train/a.png", "validation/b.png"]),
            split_names=np.asarray(["train", "validation"]),
        )
    validate_train_pair_splits(
        query_ids=np.asarray(["train/a.png", "train/b.png"]),
        split_names=np.asarray(["train", "train"]),
    )


def test_train_pair_pose_loader_ignores_unselected_validation_rows(tmp_path) -> None:
    path = tmp_path / "mixed_split_hypotheses.npz"
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train/a.png", "validation/b.png"]),
        split_names=np.asarray(["train", "validation"]),
        evaluation_labels=np.asarray(["frozen", "frozen"]),
        poses_w2c=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        verification_log_likelihood_means=np.asarray([1.0, 2.0]),
        metadata_json=np.asarray(
            '{"contains_target_fields": false, "format": "grouped_pose_hypotheses_inference_only_v1"}'
        ),
    )
    poses, scores, _ = _load_hypothesis_poses(
        (path,), evaluation_label="frozen", expected_train_query_ids={"train/a.png"}
    )
    assert set(poses) == {"train/a.png"}
    np.testing.assert_allclose(scores["train/a.png"], [1.0])


def test_hypothesis_semantic_lineage_rejects_old_generation_distribution() -> None:
    current = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
        "inputs": {
            "candidate_artifact_sha256": "current-candidates",
            "score_artifact_sha256": "current-scores",
        },
        "grouped_config": {
            "generation_mode": "grouped_prosac",
            "latent_em_enabled": True,
            "latent_em_seed_count": 256,
        },
    }
    old = {
        **current,
        "inputs": {
            "candidate_artifact_sha256": "old-candidates",
            "score_artifact_sha256": "old-scores",
        },
        "grouped_config": {
            "generation_mode": "grouped_prosac",
            "latent_em_enabled": False,
            "latent_em_seed_count": 16,
        },
    }

    reference = grouped_hypothesis_semantic_manifest(current)
    assert reference["semantic_hash"] == grouped_hypothesis_semantic_manifest(current)["semantic_hash"]
    with pytest.raises(ValueError, match="semantic lineage"):
        validate_grouped_hypothesis_semantic_match(
            expected=reference,
            observed=grouped_hypothesis_semantic_manifest(old),
        )
    evidence_drift = {
        **current,
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v11",
    }
    with pytest.raises(ValueError, match="semantic lineage"):
        validate_grouped_hypothesis_semantic_match(
            expected=reference,
            observed=grouped_hypothesis_semantic_manifest(evidence_drift),
        )


def test_hypothesis_semantic_lineage_requires_candidate_pose_evidence_version() -> None:
    metadata = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "inputs": {"candidate_artifact_sha256": "current"},
        "grouped_config": {"latent_em_enabled": True},
    }

    with pytest.raises(ValueError, match="semantic lineage"):
        grouped_hypothesis_semantic_manifest(metadata)


def test_query_grouped_pose_margin_loss_uses_each_query_hardest_wrong_pose() -> None:
    correct = torch.tensor([1.0, 2.0])
    wrong = torch.tensor([[0.2, 1.3], [1.4, 1.6]])

    loss, metrics = query_grouped_pose_margin_loss(
        correct_scores=correct,
        coherent_wrong_scores=wrong,
        margin=0.25,
    )

    expected = torch.nn.functional.softplus(torch.tensor([0.55, -0.15])).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics["query_count"] == 2.0
    assert metrics["query_correct_win_fraction"] == 0.5
    assert metrics["query_mean_correct_minus_hardest_wrong"] == pytest.approx(0.05)


def test_query_grouped_soft_hard_margin_has_gradient_for_every_wrong_mode() -> None:
    correct = torch.tensor([1.2], requires_grad=True)
    wrong = torch.tensor([[1.0, 0.7, -0.4]], requires_grad=True)

    loss, metrics = query_grouped_pose_soft_hard_margin_loss(
        correct_scores=correct,
        coherent_wrong_scores=wrong,
        margin=0.25,
        temperature=0.35,
    )
    reference_wrong = 0.35 * (
        torch.logsumexp(wrong / 0.35, dim=1) - np.log(3.0)
    )
    reference = torch.nn.functional.softplus(0.25 - (correct - reference_wrong)).mean()
    torch.testing.assert_close(loss, reference)
    loss.backward()

    assert wrong.grad is not None
    assert torch.all(torch.abs(wrong.grad) > 0.0)
    assert metrics["query_soft_hard_effective_wrong_mode_count"] > 1.0
    assert metrics["query_mean_correct_minus_soft_hard_wrong"] < 0.5


def test_query_grouped_permutation_soft_hard_margin_uses_all_modes_in_both_branches() -> None:
    normal_correct = torch.tensor([1.4], requires_grad=True)
    normal_wrong = torch.tensor([[1.1, 0.6, -0.3]], requires_grad=True)
    permuted_correct = torch.tensor([1.0], requires_grad=True)
    permuted_wrong = torch.tensor([[0.9, 0.4, -0.5]], requires_grad=True)

    loss, metrics = query_grouped_pose_permutation_soft_hard_margin_loss(
        normal_correct_scores=normal_correct,
        normal_wrong_scores=normal_wrong,
        permuted_correct_scores=permuted_correct,
        permuted_wrong_scores=permuted_wrong,
        margin=0.05,
        temperature=0.35,
    )

    normal_soft = 0.35 * (
        torch.logsumexp(normal_wrong / 0.35, dim=1) - np.log(3.0)
    )
    permuted_soft = 0.35 * (
        torch.logsumexp(permuted_wrong / 0.35, dim=1) - np.log(3.0)
    )
    expected = torch.nn.functional.softplus(
        0.05 - ((normal_correct - normal_soft) - (permuted_correct - permuted_soft))
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()

    assert normal_wrong.grad is not None
    assert permuted_wrong.grad is not None
    assert torch.all(torch.abs(normal_wrong.grad) > 0.0)
    assert torch.all(torch.abs(permuted_wrong.grad) > 0.0)
    assert metrics["normal_minus_permuted_soft_hard_gap"] > 0.0
    assert metrics["normal_soft_hard_effective_wrong_mode_count"] > 1.0
    assert metrics["permuted_soft_hard_effective_wrong_mode_count"] > 1.0


def test_train_pair_groups_keep_all_modes_for_each_query() -> None:
    pairs = pose_llr_train._TrainPairs(
        query_ids=np.asarray(["query-b.png", "query-a.png", "query-b.png"]),
        correct_poses_w2c=np.broadcast_to(np.eye(4), (3, 4, 4)).copy(),
        coherent_wrong_poses_w2c=np.broadcast_to(np.eye(4), (3, 4, 4)).copy(),
        metadata={},
    )

    groups = pose_llr_train._group_train_pairs_by_query(pairs)

    assert list(groups) == ["query-a.png", "query-b.png"]
    np.testing.assert_array_equal(groups["query-a.png"], [1])
    np.testing.assert_array_equal(groups["query-b.png"], [0, 2])


def test_inner_validation_query_partition_is_stable_and_disjoint() -> None:
    query_ids = tuple(f"query-{index:03d}.png" for index in range(63))

    train_ids, validation_ids = pose_llr_train._partition_train_queries_for_inner_validation(
        query_ids=query_ids,
        fold_count=5,
        fold_index=2,
    )

    assert train_ids == pose_llr_train._partition_train_queries_for_inner_validation(
        query_ids=query_ids,
        fold_count=5,
        fold_index=2,
    )[0]
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids).union(validation_ids) == set(query_ids)
    assert validation_ids
    assert train_ids


def test_inner_validation_checkpoint_selection_prefers_loss_then_wins() -> None:
    assert pose_llr_train._is_better_inner_validation_epoch(
        candidate={"query_grouped_loss": 0.4, "correct_win_fraction": 0.5},
        incumbent=None,
    )
    assert pose_llr_train._is_better_inner_validation_epoch(
        candidate={"query_grouped_loss": 0.4, "correct_win_fraction": 0.7},
        incumbent={"query_grouped_loss": 0.4, "correct_win_fraction": 0.5},
    )
    assert not pose_llr_train._is_better_inner_validation_epoch(
        candidate={"query_grouped_loss": 0.4, "correct_win_fraction": 0.3},
        incumbent={"query_grouped_loss": 0.4, "correct_win_fraction": 0.5},
    )
    assert not pose_llr_train._is_better_inner_validation_epoch(
        candidate={"query_grouped_loss": 0.5, "correct_win_fraction": 1.0},
        incumbent={"query_grouped_loss": 0.4, "correct_win_fraction": 0.0},
    )


def test_train_pair_hypothesis_sources_require_current_reference_lineage(tmp_path) -> None:
    current_metadata = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
        "inputs": {"candidate_artifact_sha256": "current"},
        "grouped_config": {"latent_em_enabled": True},
    }
    old_metadata = {
        **current_metadata,
        "inputs": {"candidate_artifact_sha256": "old"},
        "grouped_config": {"latent_em_enabled": False},
    }
    reference = tmp_path / "reference.npz"
    matching_source = tmp_path / "matching.npz"
    stale_source = tmp_path / "stale.npz"
    for path, metadata in (
        (reference, current_metadata),
        (matching_source, current_metadata),
        (stale_source, old_metadata),
    ):
        np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)))

    lineage = validate_train_pair_hypothesis_sources_match_reference(
        source_paths=(matching_source,), reference_path=reference
    )
    assert lineage["semantic_hash"] == grouped_hypothesis_semantic_manifest(current_metadata)["semantic_hash"]
    with pytest.raises(ValueError, match="semantic lineage"):
        validate_train_pair_hypothesis_sources_match_reference(
            source_paths=(stale_source,), reference_path=reference
        )


def test_train_pair_loader_rejects_missing_hypothesis_semantic_lineage(tmp_path) -> None:
    path = tmp_path / "pairs_without_lineage.npz"
    metadata = {
        "format": "candidate_pose_llr_train_pairs_v1",
        "training_only_target_artifact": True,
        "contains_validation_or_test_targets": False,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
    }
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train/a.png"]),
        split_names=np.asarray(["train"]),
        correct_poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        coherent_wrong_poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="semantic lineage"):
        pose_llr_train._load_train_pairs(path)


def test_train_pair_loader_accepts_serialized_hypothesis_semantic_lineage(tmp_path) -> None:
    hypothesis_metadata = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
        "inputs": {"candidate_artifact_sha256": "current"},
        "grouped_config": {"latent_em_enabled": True},
    }
    path = tmp_path / "pairs_with_serialized_lineage.npz"
    metadata = {
        "format": "candidate_pose_llr_train_pairs_v1",
        "training_only_target_artifact": True,
        "contains_validation_or_test_targets": False,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
        "hypothesis_semantic_lineage": grouped_hypothesis_semantic_manifest(
            hypothesis_metadata
        ),
    }
    np.savez_compressed(
        path,
        query_ids=np.asarray(["train/a.png"]),
        split_names=np.asarray(["train"]),
        correct_poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        coherent_wrong_poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    loaded = pose_llr_train._load_train_pairs(path)

    np.testing.assert_array_equal(loaded.query_ids, ["train/a.png"])
    assert loaded.metadata["hypothesis_semantic_lineage"] == metadata[
        "hypothesis_semantic_lineage"
    ]


def test_train_runtime_extracts_geometry_index_from_loader_tuple(
    monkeypatch, tmp_path
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        pose_llr_train,
        "load_support_observation_geometry_index_npz",
        lambda _path: (sentinel, {"format": "support_observation_geometry_index_npz"}),
    )
    assert pose_llr_train._load_support_geometry_index(tmp_path / "geometry.npz") is sentinel


def test_target_free_score_metadata_rejects_pose_or_target_leakage() -> None:
    metadata = {
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
    }
    validate_target_free_pose_llr_score_metadata(metadata)
    leaked = dict(metadata)
    leaked["pose_or_ground_truth_used_for_scoring"] = True
    with pytest.raises(ValueError, match="target-free"):
        validate_target_free_pose_llr_score_metadata(leaked)


def test_candidate_pose_llr_uses_multiscale_crops_and_keeps_oob_neutral() -> None:
    generator = torch.Generator().manual_seed(7)
    sources = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 4), generator=generator), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 6), generator=generator), dim=-1
        ),
        "alike": torch.nn.functional.normalize(
            torch.randn((3, 32, 32, 5), generator=generator), dim=-1
        ),
    }
    model = CandidateSpecificPoseLLR(
        sources=sources,
        image_sizes=torch.tensor([[100.0, 100.0]] * 3),
        hidden_dim=8,
        max_abs_log_ratio=1.5,
        edge_chunk_size=2,
    )
    runtime = CandidatePoseLLRRuntime(
        query_image_indices=torch.tensor([0]),
        support_image_indices=torch.tensor([[[1]]]),
        support_xy=torch.tensor([[[[50.0, 50.0]]]]),
        support_view_valid=torch.tensor([[[True]]]),
        candidate_view_weights=torch.tensor([[[1.0]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
    )
    scores = score_candidate_pose_batch(
        model=model,
        runtime=runtime,
        candidate_query_xy=torch.tensor(
            [
                [[[50.0, 50.0]]],
                [[[500.0, 500.0]]],
            ]
        ),
        candidate_projection_valid=torch.tensor([[[True]], [[False]]]),
    )

    assert scores.pose_log_likelihood_ratios.shape == (2,)
    assert scores.point_log_likelihood_ratios.shape == (2, 1)
    assert float(torch.max(torch.abs(scores.edge_log_likelihood_ratios))) <= 1.5
    torch.testing.assert_close(scores.pose_log_likelihood_ratios[1], torch.tensor(0.0))
    direct_scores = model(
        runtime,
        torch.tensor([[[[50.0, 50.0]]]]),
        torch.tensor([[[True]]]),
    )
    assert direct_scores.shape == (1,)


def test_candidate_pose_llr_keeps_partial_border_crop_usable() -> None:
    """A valid projection must retain its real border tokens, not become null."""

    generator = torch.Generator().manual_seed(11)
    sources = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 4), generator=generator), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 6), generator=generator), dim=-1
        ),
        "alike": torch.nn.functional.normalize(
            torch.randn((3, 32, 32, 5), generator=generator), dim=-1
        ),
    }
    model = CandidateSpecificPoseLLR(
        sources=sources,
        image_sizes=torch.tensor([[100.0, 100.0]] * 3),
        hidden_dim=8,
        max_abs_log_ratio=1.5,
        edge_chunk_size=2,
    )
    runtime = CandidatePoseLLRRuntime(
        query_image_indices=torch.tensor([0]),
        support_image_indices=torch.tensor([[[1]]]),
        support_xy=torch.tensor([[[[1.0, 1.0]]]]),
        support_view_valid=torch.tensor([[[True]]]),
        candidate_view_weights=torch.tensor([[[1.0]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
    )

    scores = score_candidate_pose_batch(
        model=model,
        runtime=runtime,
        candidate_query_xy=torch.tensor([[[[1.0, 1.0]]]]),
        candidate_projection_valid=torch.tensor([[[True]]]),
    )

    assert bool(scores.edge_usable[0, 0, 0, 0])
    assert torch.isfinite(scores.edge_log_likelihood_ratios).all()


def test_alike_shift_correlations_ignore_masked_border_tokens() -> None:
    query = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [1.0, 0.0],
          [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0]]]
    )
    support = torch.tensor(
        [[[9.0, 0.0], [8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [1.0, 0.0],
          [4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]
    )
    valid = torch.tensor([[False, False, False, False, True, False, False, False, False]])

    reference = _alike_shift_correlations(
        query,
        support,
        window_size=3,
        query_valid=valid,
        support_valid=valid,
    )
    query[:, 0] = torch.tensor([999.0, -999.0])
    support[:, 8] = torch.tensor([-999.0, 999.0])
    perturbed = _alike_shift_correlations(
        query,
        support,
        window_size=3,
        query_valid=valid,
        support_valid=valid,
    )

    torch.testing.assert_close(perturbed, reference)


def test_subpixel_crop_distinguishes_same_discrete_anchor_cell() -> None:
    """Candidate-pose scoring must retain pose shifts below one grid cell."""

    columns = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    rows = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
    grid = torch.cat(
        [columns.expand(1, 5, 5, 1), rows.expand(1, 5, 5, 1)], dim=3
    )
    image_sizes = torch.tensor([[100.0, 100.0]])
    image_indices = torch.tensor([0, 0])
    xy = torch.tensor([[50.0, 50.0], [51.0, 50.0]])

    discrete, _ = crop_anchor_aligned_grid_tokens(
        image_grids=grid,
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        window_size=3,
    )
    continuous, valid = _crop_subpixel_grid_tokens(
        image_grids=grid,
        image_sizes=image_sizes,
        image_indices=image_indices,
        xy=xy,
        window_size=3,
    )

    torch.testing.assert_close(discrete[0], discrete[1])
    assert bool(valid.all())
    assert not torch.allclose(continuous[0], continuous[1])


def test_candidate_pose_llr_activation_checkpointing_preserves_gradients() -> None:
    generator = torch.Generator().manual_seed(17)
    sources = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 4), generator=generator), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn((3, 16, 16, 6), generator=generator), dim=-1
        ),
        "alike": torch.nn.functional.normalize(
            torch.randn((3, 32, 32, 5), generator=generator), dim=-1
        ),
    }
    common = {
        "sources": sources,
        "image_sizes": torch.tensor([[100.0, 100.0]] * 3),
        "hidden_dim": 8,
        "max_abs_log_ratio": 1.5,
        "edge_chunk_size": 3,
    }
    baseline = CandidateSpecificPoseLLR(
        **common, activation_checkpointing=False
    )
    checkpointed = CandidateSpecificPoseLLR(
        **common, activation_checkpointing=True
    )
    checkpointed.load_state_dict(baseline.state_dict())
    for model in (baseline, checkpointed):
        with torch.no_grad():
            model.edge_head[-1].weight.fill_(0.1)
        model.train()

    runtime = CandidatePoseLLRRuntime(
        query_image_indices=torch.tensor([0, 0]),
        support_image_indices=torch.tensor([[[1], [2]], [[2], [1]]]),
        support_xy=torch.tensor(
            [
                [[[50.0, 50.0]], [[60.0, 50.0]]],
                [[[50.0, 60.0]], [[60.0, 60.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1)),
        candidate_probabilities=torch.full((2, 2), 0.4),
        null_probabilities=torch.full((2,), 0.2),
    )
    candidate_xy = torch.tensor(
        [
            [
                [[50.0, 50.0], [60.0, 50.0]],
                [[50.0, 60.0], [60.0, 60.0]],
            ],
            [
                [[51.0, 50.0], [61.0, 50.0]],
                [[51.0, 60.0], [61.0, 60.0]],
            ],
        ]
    )
    candidate_valid = torch.ones((2, 2, 2), dtype=torch.bool)

    baseline_scores = baseline(runtime, candidate_xy, candidate_valid)
    checkpointed_scores = checkpointed(runtime, candidate_xy, candidate_valid)
    baseline_scores.sum().backward()
    checkpointed_scores.sum().backward()

    torch.testing.assert_close(checkpointed_scores, baseline_scores)
    for baseline_parameter, checkpointed_parameter in zip(
        baseline.parameters(), checkpointed.parameters()
    ):
        torch.testing.assert_close(
            checkpointed_parameter.grad, baseline_parameter.grad
        )
