from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from feature_extract.tools.vfm.train_candidate_absolute_appearance_fusion import (
    _validate_args,
    _load_expert_initialization,
    configure_fine_rgb_trainable_parameters,
    configure_radio_final_phase_trainable_parameters,
    current_fusion_inner_gate_manifest,
    fusion_inner_gate_decision,
    parse_args,
    source_only_mode,
)
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscaleLikelihood,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityLLR,
    PhaseIdentitySourceConfig,
)


def _args():
    return parse_args(
        [
            "--rgb-spatial-layout", "layout.npz",
            "--geometry-training-targets", "geometry.npz",
            "--registered-identity-targets", "identity.npz",
            "--current-hard-repeat-targets", "current.npz",
            "--current-hard-mining-checkpoint", "current.pt",
            "--static-hard-repeat-targets", "static.npz",
            "--radio-final-context-cache", "final.npz",
            "--radio-intermediate-context-cache", "intermediate.npz",
            "--alike-spatial-context-cache", "alike.npz",
            "--image-root", "rgb",
            "--output-dir", "out",
        ]
    )


def _phase_model() -> CandidateMultiscalePhaseIdentityLLR:
    torch.manual_seed(17)
    sources = {
        name: torch.nn.functional.normalize(torch.randn(3, 8, 8, 8), dim=-1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    configs = {
        name: PhaseIdentitySourceConfig(name=name, window_size=3, shift_radius=1, region_bins=1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    return CandidateMultiscalePhaseIdentityLLR(
        sources=sources,
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        source_configs=configs,
        source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
        hidden_dim=8,
        source_storage_dtype=torch.float32,
    )


def test_fusion_cli_defaults_to_fixed_target_free_validation_selector() -> None:
    args = _args()
    _validate_args(args)
    assert args.validation_selector_policy == "coarse_margin"
    assert args.validation_selector_point_budget == 64
    assert args.inner_validation_fold_index == 1
    assert args.gate_every_epoch is False
    assert source_only_mode("radio_final_phase") == "phase_identity_only"


def test_fusion_gate_requires_fused_pose_and_radio_repeat_controls() -> None:
    args = _args()
    metrics = {
        "fused_pose_win_fraction": 0.7,
        "fused_pose_gap": 0.15,
        "fused_pose_normal_minus_permuted_gap": 0.10,
        "rgb_only_pose_gap": 0.10,
        "fused_static_repeat_eligible_query_fraction": 1.0,
        "fused_static_repeat_win_fraction": 0.7,
        "fused_static_repeat_gap": 0.15,
        "fused_static_repeat_normal_minus_permuted_gap": 0.10,
        "phase_static_repeat_eligible_query_fraction": 1.0,
        "phase_static_repeat_win_fraction": 0.7,
        "phase_static_repeat_gap": 0.15,
        "phase_static_repeat_normal_minus_permuted_gap": 0.10,
    }
    assert fusion_inner_gate_decision(metrics=metrics, args=args)["passed"] is True
    assert fusion_inner_gate_decision(
        metrics={**metrics, "phase_static_repeat_normal_minus_permuted_gap": 0.0}, args=args
    )["passed"] is False
    assert fusion_inner_gate_decision(
        metrics={**metrics, "rgb_only_pose_gap": 0.149}, args=args
    )["passed"] is False


def test_fusion_trainable_scope_excludes_unvalidated_phase_sources_and_rgb_broad_head() -> None:
    phase = _phase_model()
    phase_names = configure_radio_final_phase_trainable_parameters(phase)
    assert phase_names and all(name.startswith("source_heads.radio_final.") for name in phase_names)
    assert all(
        parameter.requires_grad == name.startswith("source_heads.radio_final.")
        for name, parameter in phase.named_parameters()
    )
    rgb = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=2.0,
        texture_feature_dim=8,
        hidden_dim=8,
    )
    rgb_names = configure_fine_rgb_trainable_parameters(rgb)
    assert rgb_names
    assert all(
        parameter.requires_grad
        == (name.startswith("texture_encoder.") or name.startswith("calibrators.fine."))
        for name, parameter in rgb.named_parameters()
    )


def test_fusion_manifest_is_stable_and_has_explicit_control_semantics() -> None:
    first = current_fusion_inner_gate_manifest()
    second = current_fusion_inner_gate_manifest()
    assert first == second
    assert "geometry_fixed_support_image_derangement_recrop_v2" in first["visual_controls"]
    assert first["semantic_sha256"]


def test_fusion_cli_rejects_invalid_initial_phase_prior_strength() -> None:
    args = _args()
    args.initial_radio_final_phase_prior_strength = 2.0
    with pytest.raises(ValueError, match="floating"):
        _validate_args(args)


def test_expert_initialization_loads_only_lineage_compatible_visual_experts(tmp_path) -> None:
    source_phase = _phase_model()
    source_rgb = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=2.0,
        texture_feature_dim=8,
        hidden_dim=8,
    )
    target_phase = _phase_model()
    target_rgb = CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=2.0,
        texture_feature_dim=8,
        hidden_dim=8,
    )
    lineage = {"layout_sha256": "layout", "source_cache_sha256": {"radio_final": "cache"}}
    checkpoint = tmp_path / "experts.pt"
    torch.save(
        {
            "format": "candidate_absolute_appearance_fusion_checkpoint_v1",
            "phase_state_dict": source_phase.state_dict(),
            "rgb_state_dict": source_rgb.state_dict(),
            "fusion_state_dict": {"source_logits": torch.tensor([99.0, -99.0])},
            "metadata": {
                "runtime_layout_is_target_free": True,
                "checkpoint_contains_train_targets": False,
                "render": False,
                "image_retrieval_or_submap": False,
                "lineage": lineage,
            },
        },
        checkpoint,
    )
    report = _load_expert_initialization(
        checkpoint_path=checkpoint,
        phase_model=target_phase,
        rgb_model=target_rgb,
        expected_lineage=lineage,
    )
    assert report["enabled"] is True
    assert report["old_fusion_calibrator_loaded"] is False
    for source, target in zip(source_phase.parameters(), target_phase.parameters()):
        torch.testing.assert_close(source, target)
    for source, target in zip(source_rgb.parameters(), target_rgb.parameters()):
        torch.testing.assert_close(source, target)
    with pytest.raises(ValueError, match="lineage mismatch"):
        _load_expert_initialization(
            checkpoint_path=checkpoint,
            phase_model=target_phase,
            rgb_model=target_rgb,
            expected_lineage={"layout_sha256": "other"},
        )
