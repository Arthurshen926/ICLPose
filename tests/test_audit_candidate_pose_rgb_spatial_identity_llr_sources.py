from __future__ import annotations

import pytest
import torch

from feature_extract.tools.vfm.audit_candidate_pose_rgb_spatial_identity_llr_sources import (
    _derived_metrics,
    _load_checkpoint,
    _source_scale_configurations,
    _validate_checkpoint_source_lineage,
    _write_audit_output,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT,
)
from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_identity_llr import (
    CHECKPOINT_FORMAT as OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES,
)


def test_source_audit_covers_all_only_and_leave_one_out_controls() -> None:
    configurations = _source_scale_configurations()
    names = tuple(CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_VISUAL_SOURCES)
    assert configurations["all"] is None
    for selected in names:
        values = configurations[f"only_{selected}"]
        assert values is not None
        assert set(values) == set(names)
        assert values[selected] == 1.0
        assert sum(values.values()) == 1.0
    for omitted in names:
        values = configurations[f"without_{omitted}"]
        assert values is not None
        assert set(values) == set(names)
        assert values[omitted] == 0.0
        assert sum(values.values()) == float(len(names) - 1)


def test_source_audit_reports_visual_control_gaps_without_redefining_scores() -> None:
    derived = _derived_metrics(
        {
            "normal_mean_correct_minus_hardest_wrong": 0.30,
            "permuted_mean_correct_minus_hardest_wrong": 0.05,
            "position_only_mean_correct_minus_hardest_wrong": 0.10,
            "hard_repeat_mean_correct_minus_coherent_wrong": 0.20,
            "hard_repeat_position_only_mean_correct_minus_coherent_wrong": -0.15,
        }
    )
    assert derived == pytest.approx(
        {
            "normal_minus_permuted_gap": 0.25,
            "normal_minus_position_only_gap": 0.20,
            "hard_repeat_minus_position_only_gap": 0.35,
        }
    )


def test_source_audit_accepts_identity_llr_checkpoint_format(tmp_path) -> None:
    checkpoint = tmp_path / "identity_llr.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": {"edge_head.0.weight": torch.zeros((1, 1))},
            "metadata": {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
            },
        },
        checkpoint,
    )
    state_dict, metadata = _load_checkpoint(checkpoint)
    assert set(state_dict) == {"edge_head.0.weight"}
    assert metadata["format"] == CHECKPOINT_FORMAT


def test_source_audit_accepts_gate_approved_broad_observation_checkpoint_format(tmp_path) -> None:
    checkpoint = tmp_path / "identity_observation_pretrain.pt"
    torch.save(
        {
            "format": OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
            "state_dict": {"edge_head.0.weight": torch.zeros((1, 1))},
            "metadata": {
                "format": OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
            },
        },
        checkpoint,
    )
    state_dict, metadata = _load_checkpoint(checkpoint)
    assert set(state_dict) == {"edge_head.0.weight"}
    assert metadata["format"] == OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT


def test_source_audit_requires_canonical_or_exact_legacy_cache_lineage() -> None:
    source = {
        "radio_final_context_cache_sha256": "final",
        "radio_intermediate_context_cache_sha256": "intermediate",
        "alike_spatial_context_cache_sha256": "alike",
        "source_image_manifest_sha256": "images",
        "rgb_coordinate_bridge": {"version": 1},
    }
    assert (
        _validate_checkpoint_source_lineage(
            checkpoint_lineage=dict(source), source_lineage=source
        )
        == "canonical_source_lineage_v1"
    )
    legacy = {
        "source_image_manifest_sha256": "",
        "rgb_coordinate_bridge": {"version": 1},
        "inputs": {
            "radio_final_context_cache": {"sha256": "final"},
            "radio_intermediate_context_cache": {"sha256": "intermediate"},
            "alike_spatial_context_cache": {"sha256": "alike"},
        },
    }
    assert (
        _validate_checkpoint_source_lineage(
            checkpoint_lineage=legacy, source_lineage=source
        )
        == "legacy_exact_input_cache_hash_fallback_v1"
    )
    legacy["inputs"]["alike_spatial_context_cache"]["sha256"] = "stale"
    with pytest.raises(ValueError, match="cache lineage"):
        _validate_checkpoint_source_lineage(checkpoint_lineage=legacy, source_lineage=source)


def test_source_audit_creates_parent_directory_before_atomic_write(tmp_path) -> None:
    output = tmp_path / "new" / "nested" / "audit.json"
    _write_audit_output(path=output, value={"ok": True})
    assert output.read_text() == '{\n  "ok": true\n}\n'
