from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood import (
    _assert_score_row_alignment,
    _assert_layout_matches_mixed_points,
    _checkpoint_validation_selector_positions,
    _project_candidate_offsets,
    _query_rows,
    _slice_layout,
    _validate_checkpoint_for_target_free_scoring,
    _validate_oof_train_query_checkpoint,
    _validate_query_layout_against_points,
    _QueryGeometry,
    parse_args,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    current_inner_gate_evaluator_manifest,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialRuntime,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([10, 11], dtype=np.int64),
        query_ids=np.asarray(["query.png", "query.png"]),
        split_names=np.asarray(["validation", "validation"]),
        xy=np.asarray([[0.25, 0.5], [1.0, 2.0]], dtype=np.float32),
        point_sources=np.asarray(["alike_high_detail", "radio_final_uniform_context"]),
        candidate_track_ids=np.asarray([[101], [102]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0], [1]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.8], [0.7]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.9], [0.9]], dtype=np.float32),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        support_image_ids=np.asarray([[["support.png"]], [["support.png"]]]),
        support_xy=np.asarray([[[[3.0, 4.0]]], [[[5.0, 6.0]]]], dtype=np.float32),
        support_view_valid=np.ones((2, 1, 1), dtype=bool),
        support_view_weights=np.ones((2, 1, 1), dtype=np.float32),
        support_coverage_counts=np.ones((2, 1, 1), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points-sha",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "projected_landmark_bank_sha256": "bank",
        },
    )


def _points(layout: CandidatePoseRGBSpatialLayout) -> SimpleNamespace:
    return SimpleNamespace(
        source_point_ids=layout.source_point_ids.copy(),
        query_ids=layout.query_ids.copy(),
        split_names=layout.split_names.copy(),
        xy=layout.xy.copy(),
        point_sources=layout.point_sources.copy(),
        candidate_track_ids=layout.candidate_track_ids.copy(),
        candidate_bank_rows=layout.candidate_bank_rows.copy(),
        candidate_coarse_similarities=layout.candidate_coarse_similarities.copy(),
        candidate_prior_probabilities=layout.candidate_prior_probabilities.copy(),
        null_probabilities=layout.null_probabilities.copy(),
        source_detector_rows=np.asarray([4, -1], dtype=np.int64),
        metadata={
            "format": "mixed_multiscale_verification_points_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "descriptor_space_id": "descriptor",
            "projection_space_id": "projection",
            "projected_landmark_bank_sha256": "bank",
        },
    )


def _checkpoint_metadata() -> dict[str, object]:
    return {
        "format": "candidate_pose_rgb_spatial_likelihood_checkpoint_v2",
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "diagnostic_only": True,
        "holdout_evaluation_allowed": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "runtime_layout_is_target_free": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "explicit_null": True,
        "projection_after_network_only": True,
        "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
        "edge_source_availability": {
            "format": "independent_rgb_context_union_v1",
            "rgb_cost_volume": "in_bounds_real_rgb_query_and_support_windows_v1",
            "context_identity": "all_full_map_radio_alike_crops_valid_v1",
            "learned_spatial_residual_and_dustbin": "rgb_and_context_intersection_only_v1",
            "combined_fallback": "rgb_raw_cost_volume_or_context_scalar_with_neutral_missing_peer_v1",
        },
        "image_retrieval_or_submap_used": False,
        "render": False,
        "appearance_control_geometry_fixed": True,
        "checkpoint_selection_policy": (
            "fixed_final_epoch_without_inner_validation_model_selection_v1"
        ),
        "inner_validation_used_for_model_selection": False,
        "inner_gate_evaluator_manifest": current_inner_gate_evaluator_manifest(),
        "train_only_inner_gate_passed": True,
        "encoder_excludes": [
            "pose_matrix",
            "projection_offset",
            "reprojection_residual",
            "ground_truth_label",
            "track_id",
            "candidate_rank",
            "coarse_score",
        ],
        "inputs": {
            "radio_final_context_cache": {"sha256": "final"},
            "radio_intermediate_context_cache": {"sha256": "intermediate"},
            "alike_spatial_context_cache": {"sha256": "alike"},
        },
        "lineage": {
            "descriptor_space_id": "descriptor",
            "projection_space_id": "projection",
            "source_image_manifest_sha256": "images",
        },
        "config": {
            "search_radius_px": 8.0,
            "context_radius_px": 12.0,
            "step_px": 0.5,
            "texture_feature_dim": 32,
            "hidden_dim": 32,
            "max_abs_context_log_ratio": 3.0,
            "max_abs_pose_log_ratio": 6.0,
            "edge_chunk_size": 64,
            "context_windows": {
                "radio_final": 15,
                "radio_intermediate": 15,
                "alike": 13,
            },
            "context_encoder_arch": "absolute_cross_attention_v3",
            "rgb_cost_volume_only": False,
        },
        "training": {
            "support_permutation_contrastive": {
                "rgb_patch_derangement": (
                    "recrop_deranged_support_image_at_fixed_support_coordinate_v2"
                ),
            },
            "inner_validation": {
                "checkpoint_selection": {
                    "policy": "fixed_final_epoch_without_inner_validation_model_selection_v1",
                    "inner_validation_used_for_model_selection": False,
                },
                "support_permutation_control": {
                    "geometry_fixed_image_only": True,
                    "rgb_support_patch": "recrop_deranged_image_at_fixed_coordinate_v2",
                },
            },
        },
    }


def test_layout_slice_preserves_frozen_candidate_and_null_mass() -> None:
    layout = _layout()
    sliced = _slice_layout(layout, np.asarray([1], dtype=np.int64))

    assert sliced.row_count == 1
    np.testing.assert_array_equal(sliced.source_point_ids, np.asarray([11], dtype=np.int64))
    np.testing.assert_allclose(
        sliced.candidate_prior_probabilities.sum(axis=1) + sliced.null_probabilities,
        np.ones((1,), dtype=np.float32),
    )
    assert sliced.metadata == layout.metadata


def test_score_row_alignment_keeps_fixed_support_ownership_point_aligned() -> None:
    arrays = {
        "query_ids": np.asarray(["q.png", "q.png"]),
        "pose_log_likelihood_ratios": np.asarray([0.1, 0.2], dtype=np.float32),
        "verification_source_point_ids": np.asarray([10, 11], dtype=np.int64),
        "candidate_support_image_ids": np.asarray([[['a.png']]], dtype=np.str_),
        "scored_support_image_ids": np.asarray([[['b.png']]], dtype=np.str_),
    }
    _assert_score_row_alignment(arrays)

    invalid = dict(arrays)
    invalid["pose_log_likelihood_ratios"] = np.asarray([0.1], dtype=np.float32)
    with pytest.raises(RuntimeError, match="row arrays"):
        _assert_score_row_alignment(invalid)


def test_layout_and_frozen_points_require_exact_target_free_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood as module

    layout = _layout()
    points = _points(layout)
    monkeypatch.setattr(module, "file_sha256_short", lambda _path: "points-sha")
    _assert_layout_matches_mixed_points(
        layout=layout, points=points, points_path=Path("points.npz")
    )

    points.candidate_track_ids[0, 0] = 999
    with pytest.raises(ValueError, match="candidate_track_ids"):
        _assert_layout_matches_mixed_points(
            layout=layout, points=points, points_path=Path("points.npz")
        )


def test_frozen_rgb_target_free_subset_maps_back_to_parent_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood as module

    parent = _layout()
    points = _points(parent)
    parent = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        query_ids=np.asarray(["query.png", "query.png", "query.png"]),
        split_names=np.asarray(["validation", "validation", "validation"]),
        xy=np.asarray([[0.25, 0.5], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        point_sources=np.asarray([
            "alike_high_detail", "radio_final_uniform_context", "alike_high_detail"
        ]),
        candidate_track_ids=np.asarray([[101], [102], [103]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0], [1], [2]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.8], [0.7], [0.6]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.9], [0.9], [0.9]], dtype=np.float32),
        null_probabilities=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        support_image_ids=np.asarray([[['support.png']], [['support.png']], [['support.png']]]),
        support_xy=np.asarray([[[[3.0, 4.0]]], [[[5.0, 6.0]]], [[[7.0, 8.0]]]], dtype=np.float32),
        support_view_valid=np.ones((3, 1, 1), dtype=bool),
        support_view_weights=np.ones((3, 1, 1), dtype=np.float32),
        support_coverage_counts=np.ones((3, 1, 1), dtype=np.int32),
        metadata=_layout().metadata,
    )
    points = _points(parent)
    selector = {
        "format": "frozen_rgb_peakiness_p1_subset_layout_v1",
        "runtime_layout_target_free": True,
        "selection_before_train_target_join": True,
        "selection_excludes": [
            "pose_matrix", "projection_offset", "reprojection_residual",
            "ground_truth_label", "track_id", "candidate_rank", "coarse_score",
        ],
    }
    subset = CandidatePoseRGBSpatialLayout(
        source_point_ids=parent.source_point_ids[[0, 2]],
        query_ids=parent.query_ids[[0, 2]],
        split_names=parent.split_names[[0, 2]],
        xy=parent.xy[[0, 2]],
        point_sources=parent.point_sources[[0, 2]],
        candidate_track_ids=parent.candidate_track_ids[[0, 2]],
        candidate_bank_rows=parent.candidate_bank_rows[[0, 2]],
        candidate_coarse_similarities=parent.candidate_coarse_similarities[[0, 2]],
        candidate_prior_probabilities=parent.candidate_prior_probabilities[[0, 2]],
        null_probabilities=parent.null_probabilities[[0, 2]],
        support_image_ids=parent.support_image_ids[[0, 2]],
        support_xy=parent.support_xy[[0, 2]],
        support_view_valid=parent.support_view_valid[[0, 2]],
        support_view_weights=parent.support_view_weights[[0, 2]],
        support_coverage_counts=parent.support_coverage_counts[[0, 2]],
        metadata={**parent.metadata, "frozen_rgb_selector": selector},
    )
    monkeypatch.setattr(module, "file_sha256_short", lambda _path: "points-sha")
    rows, lineage = _assert_layout_matches_mixed_points(
        layout=subset, points=points, points_path=Path("points.npz")
    )
    np.testing.assert_array_equal(rows, np.asarray([0, 2], dtype=np.int64))
    assert lineage["mode"] == "frozen_target_free_rgb_peakiness_subset_of_formal_p1_points_v1"

    invalid = CandidatePoseRGBSpatialLayout(
        **{name: getattr(subset, name) for name in (
            "source_point_ids", "query_ids", "split_names", "xy", "point_sources",
            "candidate_track_ids", "candidate_bank_rows", "candidate_coarse_similarities",
            "candidate_prior_probabilities", "null_probabilities", "support_image_ids",
            "support_xy", "support_view_valid", "support_view_weights", "support_coverage_counts",
        )},
        metadata=parent.metadata,
    )
    with pytest.raises(ValueError, match="valid target-free frozen RGB subset"):
        _assert_layout_matches_mixed_points(
            layout=invalid, points=points, points_path=Path("points.npz")
        )


def test_query_alignment_rejects_different_frozen_candidate_order() -> None:
    layout = _layout()
    points = _points(layout)
    rows = _query_rows(layout=layout, query_id="query.png", split_name="validation")
    _validate_query_layout_against_points(
        query_layout=_slice_layout(layout, rows), points=points, points_rows=rows
    )

    points.candidate_track_ids = points.candidate_track_ids[::-1].copy()
    with pytest.raises(ValueError, match="candidate_track_ids"):
        _validate_query_layout_against_points(
            query_layout=_slice_layout(layout, rows), points=points, points_rows=rows
        )


def test_checkpoint_contract_rejects_target_or_stale_context_cache() -> None:
    layout = _layout()
    metadata = _checkpoint_metadata()
    hashes = {
        "radio_final_context_cache": "final",
        "radio_intermediate_context_cache": "intermediate",
        "alike_spatial_context_cache": "alike",
    }
    config = _validate_checkpoint_for_target_free_scoring(
        metadata=metadata,
        layout=layout,
        cache_hashes=hashes,
        source_image_manifest_sha256="images",
    )
    assert int(config["edge_chunk_size"]) == 64

    stale = _checkpoint_metadata()
    stale["inputs"] = dict(stale["inputs"])
    stale["inputs"]["alike_spatial_context_cache"] = {"sha256": "old"}
    with pytest.raises(ValueError, match="stale"):
        _validate_checkpoint_for_target_free_scoring(
            metadata=stale,
            layout=layout,
            cache_hashes=hashes,
            source_image_manifest_sha256="images",
        )

    target_bearing = _checkpoint_metadata()
    target_bearing["contains_target_fields"] = True
    with pytest.raises(ValueError, match="held-out contract"):
        _validate_checkpoint_for_target_free_scoring(
            metadata=target_bearing,
            layout=layout,
            cache_hashes=hashes,
            source_image_manifest_sha256="images",
        )

    old_control = _checkpoint_metadata()
    old_control["appearance_control_geometry_fixed"] = False
    with pytest.raises(ValueError, match="held-out contract"):
        _validate_checkpoint_for_target_free_scoring(
            metadata=old_control,
            layout=layout,
            cache_hashes=hashes,
            source_image_manifest_sha256="images",
        )

    validation_selected = _checkpoint_metadata()
    validation_selected["inner_validation_used_for_model_selection"] = True
    with pytest.raises(ValueError, match="held-out contract"):
        _validate_checkpoint_for_target_free_scoring(
            metadata=validation_selected,
            layout=layout,
            cache_hashes=hashes,
            source_image_manifest_sha256="images",
        )

    stale_evaluator = _checkpoint_metadata()
    stale_evaluator["inner_gate_evaluator_manifest"] = {"version": "legacy"}
    with pytest.raises(ValueError, match="manifest is stale or missing"):
        _validate_checkpoint_for_target_free_scoring(
            metadata=stale_evaluator,
            layout=layout,
            cache_hashes=hashes,
            source_image_manifest_sha256="images",
        )


def test_oof_train_query_requires_serialized_exclusion_proof() -> None:
    from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
        train_query_partition_manifest,
    )

    metadata = _checkpoint_metadata()
    training = dict(metadata["training"])
    inner = dict(training["inner_validation"])
    partition = train_query_partition_manifest(
        all_query_ids=("train/a.png", "train/b.png"),
        inner_train_query_ids=("train/a.png",),
        inner_validation_query_ids=("train/b.png",),
        fold_count=2,
        fold_index=1,
    )
    inner["query_partition"] = partition
    training["inner_validation"] = inner
    metadata["training"] = training

    result = _validate_oof_train_query_checkpoint(
        checkpoint_metadata=metadata, query_id="train/b.png"
    )
    assert result == partition
    with pytest.raises(ValueError, match="not excluded"):
        _validate_oof_train_query_checkpoint(
            checkpoint_metadata=metadata, query_id="train/a.png"
        )
    partition["inner_validation"]["query_ids_sha256"] = "stale"
    with pytest.raises(ValueError, match="digest"):
        _validate_oof_train_query_checkpoint(
            checkpoint_metadata=metadata, query_id="train/b.png"
        )


def test_pose_projection_is_converted_to_query_anchor_offsets() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        support_image_indices=torch.tensor([[[1]]]),
        support_xy=torch.tensor([[[[2.0, 2.0]]]], dtype=torch.float32),
        support_view_valid=torch.ones((1, 1, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 1, 1), dtype=torch.float32),
        candidate_probabilities=torch.tensor([[0.9]], dtype=torch.float32),
        null_probabilities=torch.tensor([0.1], dtype=torch.float32),
    )
    geometry = _QueryGeometry(
        candidate_xyz=torch.tensor([[[2.0, 4.0, 2.0]]]),
        focal_length=1.0,
        principal_x=0.0,
        principal_y=0.0,
        radial_k=0.0,
        image_width=10,
        image_height=10,
    )
    offsets, valid = _project_candidate_offsets(
        geometry=geometry,
        runtime=runtime,
        poses_w2c=np.eye(4, dtype=np.float64)[None],
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(offsets[0, 0, 0], torch.tensor([0.75, 1.5]))
    assert bool(valid[0, 0, 0])


def test_component_ablation_is_explicit_and_defaults_to_combined() -> None:
    base = [
        "--hypothesis-artifact", "hyp.npz", "--baseline-score-artifact", "base.npz",
        "--detector-query-cache", "detector.npz", "--proposals", "proposals.npz",
        "--candidate-artifact", "candidate.npz", "--fixed-candidate-prior-overlay", "prior.npz",
        "--mixed-verification-points-artifact", "points.npz", "--rgb-spatial-layout", "layout.npz",
        "--projected-landmark-bank", "bank.npz", "--colmap-model-dir", "colmap",
        "--radio-final-context-cache", "final.npz", "--radio-intermediate-context-cache", "intermediate.npz",
        "--alike-spatial-context-cache", "alike.npz", "--image-root", "images",
        "--checkpoint", "checkpoint.pt", "--output", "scores.npz",
    ]
    assert parse_args(base).score_component == "combined"
    assert parse_args(base + ["--score-component", "context_only"]).score_component == "context_only"
    assert parse_args(base + ["--query-id", "train/query.png"]).query_id == "train/query.png"


def test_direct_script_import_bootstrap_resolves_repository_root() -> None:
    import feature_extract.tools.vfm.score_candidate_pose_rgb_spatial_likelihood as module

    assert str(module._REPOSITORY_ROOT) in sys.path


def test_checkpoint_validation_selector_is_static_and_uses_no_prediction() -> None:
    base = _layout()
    layout = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        query_ids=np.asarray(["query.png"] * 4),
        split_names=np.asarray(["validation"] * 4),
        xy=np.asarray([[2.0, 2.0], [12.0, 2.0], [2.0, 12.0], [12.0, 12.0]], dtype=np.float32),
        point_sources=np.asarray(["alike_high_detail"] * 4),
        candidate_track_ids=np.asarray([[101], [102], [103], [104]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0], [1], [2], [3]], dtype=np.int64),
        candidate_coarse_similarities=np.full((4, 1), 0.8, dtype=np.float32),
        candidate_prior_probabilities=np.full((4, 1), 0.9, dtype=np.float32),
        null_probabilities=np.full((4,), 0.1, dtype=np.float32),
        support_image_ids=np.asarray([[["support.png"]]] * 4),
        support_xy=np.asarray(
            [[[[3.0, 4.0]]], [[[13.0, 4.0]]], [[[3.0, 13.0]]], [[[13.0, 13.0]]]],
            dtype=np.float32,
        ),
        support_view_valid=np.ones((4, 1, 1), dtype=bool),
        support_view_weights=np.ones((4, 1, 1), dtype=np.float32),
        support_coverage_counts=np.ones((4, 1, 1), dtype=np.int32),
        metadata=base.metadata,
    )
    config = _checkpoint_metadata()["config"]
    config = dict(config)
    config["validation_selector"] = {
        "policy": "support_coverage",
        "point_budget": 4,
        "grid_rows": 2,
        "grid_columns": 2,
        "target_free_static_only": True,
    }
    positions, metadata = _checkpoint_validation_selector_positions(
        layout=layout, config=config, coordinate_image_size=(64, 64)
    )
    np.testing.assert_array_equal(positions, np.arange(4, dtype=np.int64))
    assert metadata["target_free_static_only"] is True
    assert metadata["policy"] == "support_coverage"
