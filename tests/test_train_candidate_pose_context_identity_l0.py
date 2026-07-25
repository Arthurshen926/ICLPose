from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
)
from feature_extract.tools.vfm.train_candidate_pose_context_identity_l0 import (
    CHECKPOINT_FORMAT,
    _context_only_validation,
    _validate_args,
    context_pose_margin_loss,
    fixed_final_epoch_checkpoint_selection,
    load_context_identity_l0_initialization_checkpoint,
    p1_context_hard_repeat_evidence_gate,
    parse_args,
    p1_context_identity_gate,
    resolve_target_free_validation_point_budget,
    select_identity_group_points,
)


def test_identity_point_selection_never_drops_available_strict_positive_rows() -> None:
    group = SimpleNamespace(
        point_count=8,
        source_point_ids=np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64),
        spatial_target_observed=np.asarray(
            [
                [False, False],
                [True, False],
                [False, False],
                [False, True],
                [False, False],
                [False, False],
                [True, False],
                [False, False],
            ]
        ),
    )
    selected = select_identity_group_points(group=group, max_points=5, seed=17)
    assert len(selected) == 5
    assert {1, 3, 6}.issubset(set(selected.tolist()))


def test_target_free_validation_budget_clamps_without_dropping_short_queries() -> None:
    assert resolve_target_free_validation_point_budget(
        requested_point_budget=64, available_point_count=17
    ) == 17
    assert resolve_target_free_validation_point_budget(
        requested_point_budget=16, available_point_count=64
    ) == 16
    with pytest.raises(ValueError, match="at least four"):
        resolve_target_free_validation_point_budget(
            requested_point_budget=64, available_point_count=3
        )


def test_context_pose_margin_joins_projection_targets_after_target_free_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feature_extract.tools.vfm.train_candidate_pose_context_identity_l0 as module

    prediction = SimpleNamespace(joint_log_probabilities=torch.zeros((2, 3, 2, 4)))
    batch = SimpleNamespace(
        correct_projection_offsets_xy=torch.zeros((2, 3, 2)),
        correct_projection_valid=torch.ones((2, 3), dtype=torch.bool),
        wrong_projection_offsets_xy=torch.zeros((2, 2, 3, 2)),
        wrong_projection_valid=torch.ones((2, 2, 3), dtype=torch.bool),
    )
    observed_shapes: list[tuple[int, ...]] = []

    monkeypatch.setattr(
        module,
        "candidate_pose_rgb_spatial_score_component_prediction",
        lambda *, prediction, component: prediction,
    )

    def fake_score(**kwargs: object) -> object:
        offsets = torch.as_tensor(kwargs["candidate_projection_offsets_xy"])
        observed_shapes.append(tuple(offsets.shape))
        if offsets.shape[0] == 1:
            return SimpleNamespace(pose_log_likelihood_ratios=torch.tensor([0.4]))
        return SimpleNamespace(pose_log_likelihood_ratios=torch.tensor([0.1, 0.2]))

    def fake_margin(**kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        torch.testing.assert_close(kwargs["correct_scores"], torch.tensor([0.4]))
        torch.testing.assert_close(kwargs["coherent_wrong_scores"], torch.tensor([[0.1, 0.2]]))
        assert kwargs["margin"] == pytest.approx(0.25)
        return torch.tensor(0.3), {
            "query_mean_correct_minus_hardest_wrong": 0.2,
            "query_correct_win_fraction": 1.0,
        }

    monkeypatch.setattr(module, "score_candidate_pose_rgb_spatial_batch", fake_score)
    monkeypatch.setattr(module, "query_grouped_pose_margin_loss", fake_margin)
    loss, metrics = context_pose_margin_loss(
        runtime=object(),  # type: ignore[arg-type]
        prediction=prediction,
        batch=batch,
        pose_margin=0.25,
    )
    assert loss.item() == pytest.approx(0.3)
    assert metrics["query_mean_correct_minus_hardest_wrong"] == pytest.approx(0.2)
    assert observed_shapes == [(1, 2, 3, 2), (2, 2, 3, 2)]


def test_identity_point_selection_retains_hard_repeat_rows_when_they_fit() -> None:
    group = SimpleNamespace(
        point_count=8,
        source_point_ids=np.asarray([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.int64),
        spatial_target_observed=np.asarray(
            [[False], [True], [False], [False], [False], [False], [False], [False]]
        ),
    )
    selected = select_identity_group_points(
        group=group,
        max_points=5,
        seed=17,
        required_source_point_ids=[103, 106],
    )
    assert {1, 3, 6}.issubset(set(selected.tolist()))


def test_p1_context_gate_rejects_a_position_only_shortcut() -> None:
    visual = {
        "normal_correct_win_fraction": 0.8,
        "normal_mean_correct_minus_hardest_wrong": 0.20,
        "permuted_mean_correct_minus_hardest_wrong": 0.05,
        "query_count": 5.0,
    }
    position = {**visual, "normal_mean_correct_minus_hardest_wrong": 0.19}
    result = p1_context_identity_gate(
        visual_metrics=visual,
        position_only_metrics=position,
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
        minimum_position_only_gap=0.05,
    )
    assert result["passed"] is False
    assert result["checks"]["descriptor_over_position"] is False


def test_direct_hard_repeat_gate_requires_visual_candidate_evidence() -> None:
    metrics = {
        "hard_repeat_query_count": 5.0,
        "hard_repeat_normal_mean_gap": 0.20,
        "hard_repeat_normal_win_fraction": 0.80,
        "hard_repeat_permuted_mean_gap": 0.05,
        "hard_repeat_position_only_mean_gap": 0.03,
    }
    result = p1_context_hard_repeat_evidence_gate(
        metrics=metrics,
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
        minimum_position_only_gap=0.05,
    )
    assert result["passed"] is True

    position_shortcut = {**metrics, "hard_repeat_position_only_mean_gap": 0.18}
    rejected = p1_context_hard_repeat_evidence_gate(
        metrics=position_shortcut,
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
        minimum_position_only_gap=0.05,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["descriptor_over_position_gap"] is False


def test_l0_checkpoint_selection_is_fixed_final_epoch() -> None:
    selection = fixed_final_epoch_checkpoint_selection(epochs=8)
    assert selection == {
        "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
        "selected_epoch": 8,
        "inner_validation_used_for_model_selection": False,
    }


def test_l0_arguments_require_hard_repeat_artifact_when_its_loss_is_enabled() -> None:
    args = parse_args(
        [
            "--rgb-spatial-layout",
            "layout.npz",
            "--training-targets",
            "targets.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--output-dir",
            "out",
            "--hard-repeat-context-loss-weight",
            "1.0",
        ]
    )
    with pytest.raises(ValueError, match="hard-repeat"):
        _validate_args(args)


def test_l0_pretrain_initialization_requires_the_source_pair_artifact() -> None:
    args = parse_args(
        [
            "--rgb-spatial-layout",
            "layout.npz",
            "--training-targets",
            "targets.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--output-dir",
            "out",
            "--context-observation-pretrain-checkpoint",
            "context.pt",
        ]
    )
    with pytest.raises(ValueError, match="observation-pair"):
        _validate_args(args)


def test_l0_arguments_require_absolute_context_for_position_only_control() -> None:
    args = parse_args(
        [
            "--rgb-spatial-layout",
            "layout.npz",
            "--training-targets",
            "targets.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--output-dir",
            "out",
            "--context-encoder-arch",
            "conv_v1",
        ]
    )
    with pytest.raises(ValueError, match="absolute_cross_attention_v3"):
        _validate_args(args)


def test_l0_validation_selects_target_free_layout_before_target_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """The P1 checkpoint gate must never call the target-preserving sampler."""

    import feature_extract.tools.vfm.train_candidate_pose_context_identity_l0 as module
    import torch

    events: list[str] = []
    group = SimpleNamespace(layout_rows=np.asarray([40, 41, 42, 43, 44, 45], dtype=np.int64))
    batch = SimpleNamespace(
        runtime="normal-runtime",
        correct_projection_offsets_xy=torch.zeros((2, 3, 2), dtype=torch.float32),
        correct_projection_valid=torch.ones((2, 3), dtype=torch.bool),
        wrong_projection_offsets_xy=torch.zeros((1, 2, 3, 2), dtype=torch.float32),
        wrong_projection_valid=torch.ones((1, 2, 3), dtype=torch.bool),
    )

    def fake_selector_input(*, layout: object, rows: np.ndarray) -> object:
        assert layout == "target-free-layout"
        np.testing.assert_array_equal(rows, group.layout_rows)
        events.append("selector_input")
        return SimpleNamespace(point_count=6)

    def fake_selector_scores(*, selector_input: object, policy: str) -> np.ndarray:
        assert selector_input.point_count == 6
        assert policy == "coarse_margin"
        events.append("selector_scores")
        return np.asarray([0.1, 0.4, 0.2, 0.3, 0.5, 0.6], dtype=np.float32)

    def fake_select(**kwargs: object) -> np.ndarray:
        assert kwargs["point_budget"] == 4
        events.append("select")
        return np.asarray([1, 3, 4, 5], dtype=np.int64)

    def fake_batch(**kwargs: object) -> object:
        np.testing.assert_array_equal(
            kwargs["point_positions"], np.asarray([1, 3, 4, 5], dtype=np.int64)
        )
        events.append("target_join")
        return batch

    def fake_permute(runtime: object, *, shift: int) -> object:
        assert runtime == "normal-runtime"
        assert shift == 1
        return "permuted-runtime"

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def __call__(self, *, runtime: object, **kwargs: object) -> object:
            assert torch.is_grad_enabled() is False
            assert kwargs == {"context_only": True, "context_appearance_mode": "visual"}
            events.append(f"forward:{runtime}")
            return runtime

    def fake_component(*, prediction: object, component: str) -> object:
        assert component == "context_only"
        return prediction

    def fake_score(**kwargs: object) -> object:
        events.append(f"score:{kwargs['runtime']}")
        return SimpleNamespace(pose_log_likelihood_ratios=torch.tensor([0.3]))

    def fake_margin(**kwargs: object) -> tuple[torch.Tensor, dict[str, float]]:
        assert kwargs["correct_scores"].shape == (1,)
        assert kwargs["coherent_wrong_scores"].shape == (1, 1)
        return torch.tensor(0.2), {
            "query_mean_correct_minus_hardest_wrong": 0.1,
            "query_correct_win_fraction": 1.0,
        }

    monkeypatch.setattr(module, "selector_input_from_target_free_layout", fake_selector_input)
    monkeypatch.setattr(module, "target_free_selector_scores", fake_selector_scores)
    monkeypatch.setattr(module, "select_target_free_spatial_quota", fake_select)
    monkeypatch.setattr(module, "_query_batch_from_group", fake_batch)
    monkeypatch.setattr(module, "permute_runtime_support_image_appearance_only", fake_permute)
    monkeypatch.setattr(module, "_assert_geometry_fixed_support_image_control", lambda **_: None)
    monkeypatch.setattr(module, "candidate_pose_rgb_spatial_score_component_prediction", fake_component)
    monkeypatch.setattr(module, "score_candidate_pose_rgb_spatial_batch", fake_score)
    monkeypatch.setattr(module, "query_grouped_pose_margin_loss", fake_margin)

    metrics = _context_only_validation(
        model=FakeModel(),
        layout="target-free-layout",  # type: ignore[arg-type]
        groups={"query": group},
        complete_runtime="complete-runtime",
        query_ids=("query",),
        image_size=(1024, 768),
        state=SimpleNamespace(
            world_size=1,
            rank=0,
            device=torch.device("cpu"),
            enabled=False,
        ),
        selector_policy="coarse_margin",
        selector_point_budget=4,
        selector_grid_rows=1,
        selector_grid_columns=2,
        pose_margin=0.25,
        amp_enabled=False,
    )

    assert events[:4] == ["selector_input", "selector_scores", "select", "target_join"]
    assert metrics["normal_mean_correct_minus_hardest_wrong"] == pytest.approx(0.1)
    assert metrics["permuted_mean_correct_minus_hardest_wrong"] == pytest.approx(0.1)


def test_context_l0_checkpoint_loader_requires_exact_lineage_and_explicit_failed_gate_override(
    tmp_path,
) -> None:
    torch.manual_seed(31)
    sources = {
        "radio_final": torch.nn.functional.normalize(torch.randn(2, 8, 8, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(2, 8, 8, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(2, 16, 16, 4), dim=-1),
    }
    windows = {"radio_final": 3, "radio_intermediate": 3, "alike": 3}

    def model() -> CandidatePoseRGBSpatialLikelihood:
        return CandidatePoseRGBSpatialLikelihood(
            sources=sources,
            image_sizes=torch.full((2, 2), 64.0),
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            context_windows=windows,
            context_encoder_arch="absolute_cross_attention_v3",
        )

    source_model = model()
    source_cache_paths = {
        "radio_final": tmp_path / "radio_final.npz",
        "radio_intermediate": tmp_path / "radio_intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for name, path in source_cache_paths.items():
        path.write_bytes(name.encode("ascii"))
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "l1_spatial_training_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "appearance_control_geometry_fixed": True,
        "checkpoint_selection_policy": (
            "fixed_final_epoch_without_inner_validation_model_selection_v1"
        ),
        "inner_validation_used_for_model_selection": False,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
            "rgb_patch",
        ],
        "config": {
            "search_radius_px": 2.0,
            "context_radius_px": 2.0,
            "step_px": 1.0,
            "texture_feature_dim": 4,
            "hidden_dim": 8,
            "max_abs_context_log_ratio": 3.0,
            "candidate_count": 2,
            "support_view_count": 2,
            "context_windows": windows,
            "context_encoder_arch": "absolute_cross_attention_v3",
            "context_only": True,
        },
        "training": {
            "inner_validation": {
                "selected_epoch": 2,
                "checkpoint_selection": {
                    "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
                    "selected_epoch": 2,
                    "inner_validation_used_for_model_selection": False,
                },
                "gate": {"passed": False},
            }
        },
        "lineage": {
            "rgb_spatial_layout_sha256": "layout",
            "training_targets_sha256": "targets",
            "radio_final_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_final"]
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                source_cache_paths["radio_intermediate"]
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(source_cache_paths["alike"]),
            "source_image_manifest_sha256": "manifest",
        },
    }
    checkpoint_path = tmp_path / "context_l0.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": source_model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="inner gate did not pass"):
        load_context_identity_l0_initialization_checkpoint(
            path=checkpoint_path,
            model=model(),
            layout_sha256="layout",
            targets_sha256="targets",
            candidate_count=2,
            support_view_count=2,
            source_cache_paths=source_cache_paths,
            source_image_manifest_sha256="manifest",
            search_radius_px=2.0,
            context_radius_px=2.0,
            step_px=1.0,
            texture_feature_dim=4,
            hidden_dim=8,
            max_abs_context_log_ratio=3.0,
            context_windows=windows,
            context_encoder_arch="absolute_cross_attention_v3",
        )
    target_model = model()
    result = load_context_identity_l0_initialization_checkpoint(
        path=checkpoint_path,
        model=target_model,
        layout_sha256="layout",
        targets_sha256="targets",
        candidate_count=2,
        support_view_count=2,
        source_cache_paths=source_cache_paths,
        source_image_manifest_sha256="manifest",
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        max_abs_context_log_ratio=3.0,
        context_windows=windows,
        context_encoder_arch="absolute_cross_attention_v3",
        allow_diagnostic_ineligible=True,
    )
    assert result["kind"] == "p1_context_identity_l0_context_only"
    assert result["eligible_for_p1_finetune"] is False
    assert result["diagnostic_ineligible_override"] is True
    for name, value in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], value)
