from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_pose_rgb_spatial_hard_pose_pretrain_p1 import (
    _checkpoint_request,
    _checkpoint_kind_requires_context_only_replay,
    _p1_point_selection_contract,
    _registered_identity_masks_by_source,
)


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "checkpoint": "checkpoint.pt",
        "checkpoint_kind": "hard_pose_pretrain",
        "hard_pose_pretrain_checkpoint": "",
        "allow_ineligible_diagnostic_checkpoint": False,
        "cross_layout_component_audit": "",
    }
    values.update(overrides)
    return Namespace(**values)


def test_checkpoint_request_accepts_explicit_and_legacy_hard_pose_paths() -> None:
    path, kind = _checkpoint_request(_args())
    assert str(path) == "checkpoint.pt"
    assert kind == "hard_pose_pretrain"

    legacy_path, legacy_kind = _checkpoint_request(
        _args(checkpoint="", hard_pose_pretrain_checkpoint="legacy.pt")
    )
    assert str(legacy_path) == "legacy.pt"
    assert legacy_kind == "hard_pose_pretrain"

    context_path, context_kind = _checkpoint_request(
        _args(checkpoint_kind="context_observation_pretrain")
    )
    assert str(context_path) == "checkpoint.pt"
    assert context_kind == "context_observation_pretrain"

    l0_path, l0_kind = _checkpoint_request(
        _args(
            checkpoint_kind="context_identity_l0",
            allow_ineligible_diagnostic_checkpoint=True,
        )
    )
    assert str(l0_path) == "checkpoint.pt"
    assert l0_kind == "context_identity_l0"

    cross_path, cross_kind = _checkpoint_request(
        _args(
            checkpoint_kind="p1_cross_layout_curriculum_checkpoint",
            cross_layout_component_audit="component_audit.json",
        )
    )
    assert str(cross_path) == "checkpoint.pt"
    assert cross_kind == "p1_cross_layout_curriculum_checkpoint"


def test_checkpoint_request_rejects_ambiguous_or_invalid_override() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        _checkpoint_request(_args(hard_pose_pretrain_checkpoint="other.pt"))
    with pytest.raises(ValueError, match="requires --checkpoint"):
        _checkpoint_request(_args(checkpoint=""))
    with pytest.raises(ValueError, match="only for hard-pose"):
        _checkpoint_request(
            _args(
                checkpoint_kind="observation_pretrain",
                allow_ineligible_diagnostic_checkpoint=True,
            )
        )
    with pytest.raises(ValueError, match="requires --cross-layout-component-audit"):
        _checkpoint_request(_args(checkpoint_kind="p1_cross_layout_curriculum_checkpoint"))
    with pytest.raises(ValueError, match="valid only for a cross-layout"):
        _checkpoint_request(_args(cross_layout_component_audit="component_audit.json"))


def test_context_only_checkpoint_kinds_never_run_random_rgb_spatial_heads() -> None:
    assert _checkpoint_kind_requires_context_only_replay("context_observation_pretrain")
    assert _checkpoint_kind_requires_context_only_replay("context_identity_l0")
    assert not _checkpoint_kind_requires_context_only_replay("observation_pretrain")
    assert not _checkpoint_kind_requires_context_only_replay("p1_checkpoint")


def test_edge_evidence_requires_explicit_registered_identity_masks() -> None:
    exact = Namespace(
        source_point_ids=np.asarray([7, 9], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/b.png"]),
        spatial_target_observed=np.asarray([[True, False], [False, False]], dtype=bool),
        metadata={
            "format": "candidate_pose_rgb_spatial_targets_v2",
            "spatial_supervision_mode": "registered_exact_identity",
            "spatial_target_semantics": (
                "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
            ),
        },
    )
    masks = _registered_identity_masks_by_source(
        registered_identity_targets=exact,
        expected_source_point_ids=np.asarray([7, 9], dtype=np.int64),
        expected_query_ids=np.asarray(["q/a.png", "q/b.png"]),
        candidate_count=2,
    )
    np.testing.assert_array_equal(masks[7], np.asarray([True, False]))
    np.testing.assert_array_equal(masks[9], np.asarray([False, False]))

    # Geometric full-pose visibility can contain several valid candidates. It
    # is never a substitute for the registered physical-track identity mask.
    geometry_visibility = Namespace(
        source_point_ids=np.asarray([7], dtype=np.int64),
        query_ids=np.asarray(["q/a.png"]),
        spatial_target_observed=np.asarray([[True, True]], dtype=bool),
        metadata={
            "format": "candidate_pose_rgb_spatial_targets_v1",
            "spatial_supervision_mode": "geometry_projected",
            "spatial_target_semantics": "correct_pose_projected_offset_inside_fixed_local_support_or_dustbin_v1",
        },
    )
    with pytest.raises(ValueError, match="registered identity"):
        _registered_identity_masks_by_source(
            registered_identity_targets=geometry_visibility,
            expected_source_point_ids=np.asarray([7], dtype=np.int64),
            expected_query_ids=np.asarray(["q/a.png"]),
            candidate_count=2,
        )


class _Group:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count


class _Layout:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata


def _frozen_selector_metadata() -> dict[str, object]:
    return {
        "frozen_rgb_selector": {
            "format": "frozen_rgb_peakiness_p1_subset_layout_v1",
            "runtime_layout_target_free": True,
            "selection_before_train_target_join": True,
            "selection_excludes": [
                "pose_matrix",
                "projection_offset",
                "reprojection_residual",
                "ground_truth_label",
                "track_id",
                "candidate_rank",
                "coarse_score",
            ],
        }
    }


def test_point_selection_contract_accepts_all_rows_of_frozen_target_free_layout() -> None:
    result = _p1_point_selection_contract(
        layout=_Layout(_frozen_selector_metadata()),
        groups={"q0": _Group(32), "q1": _Group(32)},
        query_ids=("q0", "q1"),
        max_points_per_query=32,
    )
    assert result["runtime_eligible"] is True
    assert result["keeps_all_frozen_layout_rows"] is True
    assert result["mode"] == "frozen_target_free_rgb_selector_layout_all_rows_v1"


def test_point_selection_contract_rejects_second_stage_subsampling_or_missing_manifest() -> None:
    second_stage = _p1_point_selection_contract(
        layout=_Layout(_frozen_selector_metadata()),
        groups={"q0": _Group(32)},
        query_ids=("q0",),
        max_points_per_query=16,
    )
    assert second_stage["runtime_eligible"] is False
    assert second_stage["mode"] == "frozen_target_free_layout_with_second_stage_subsampling_v1"

    historic = _p1_point_selection_contract(
        layout=_Layout({}),
        groups={"q0": _Group(32)},
        query_ids=("q0",),
        max_points_per_query=32,
    )
    assert historic["runtime_eligible"] is False
    assert historic["mode"] == "train_target_observation_preserving_sampler_v1"
