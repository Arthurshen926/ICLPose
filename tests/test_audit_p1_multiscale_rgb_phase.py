from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.audit_p1_multiscale_rgb_phase import (
    BRANCHES,
    RGBPhaseQueryFeatures,
    _checkpoint_inner_query_split,
    _load_frozen_texture_encoder,
    _load_training_feature,
    _validate_edge_encoder_support_view_sweep,
    _write_training_feature,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_multiscale_rgb_phase import RGBPhaseScale
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
)


class _Layout:
    candidate_count = 2
    support_view_count = 1


def _support_view_layout(*, view_count: int) -> SimpleNamespace:
    if view_count < 1:
        raise ValueError("view_count must be positive")
    row_count, candidate_count = 2, 2
    support_ids = np.asarray(
        [
            [[f"support-{row}-{candidate}-{view}.png" for view in range(view_count)]
             for candidate in range(candidate_count)]
            for row in range(row_count)
        ]
    )
    support_xy = (
        np.arange(row_count * candidate_count * 2, dtype=np.float32).reshape(
            row_count, candidate_count, 1, 2
        )
        + 100.0 * np.arange(view_count, dtype=np.float32)[None, None, :, None]
    )
    valid = np.ones((row_count, candidate_count, view_count), dtype=bool)
    coverage = np.full((row_count, candidate_count, view_count), 3, dtype=np.int32)
    return SimpleNamespace(
        candidate_count=candidate_count,
        support_view_count=view_count,
        source_point_ids=np.asarray([10, 11], dtype=np.int64),
        query_ids=np.asarray(["q0.png", "q1.png"]),
        split_names=np.asarray(["train", "train"]),
        xy=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        point_sources=np.asarray(["alike_high_detail", "radio_final_uniform_context"]),
        candidate_track_ids=np.asarray([[101, 102], [103, 104]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.5, 0.4], [0.7, 0.6]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.6, 0.3], [0.5, 0.4]], dtype=np.float32),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        support_image_ids=support_ids,
        support_xy=support_xy,
        support_view_valid=valid,
        support_coverage_counts=coverage,
        metadata={
            "candidate_set": "fixed_full_global_faiss_top_l_tracks",
            "candidate_top_k": 2,
            "candidate_prior_semantics": "fixed_prior",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "support_view_selection": "fixed_maplet_coverage_rank_excluding_query_image_v1",
            "support_coordinate_source": "sfm_observation_xy",
            "image_retrieval_or_submap_used": False,
            "render": False,
            "candidate_reselection": False,
        },
    )


def _checkpoint(path, *, layout_path) -> None:
    encoder = TexturePatchEncoder(
        feature_dim=4,
        hidden_dim=4,
        input_mode="rgb_graygrad",
        encoder_arch="fpn",
    )
    torch.save(
        {
            "state_dict": {
                f"texture_encoder.{name}": value
                for name, value in encoder.state_dict().items()
            },
            "metadata": {
                "format": "candidate_pose_rgb_spatial_likelihood_checkpoint_v1",
                "model_format": "candidate_pose_rgb_spatial_likelihood_v2",
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "fixed_global_topl": True,
                "projection_after_network_only": True,
                "fixed_candidate_top_k": 2,
                "fixed_support_view_count": 1,
                "config": {
                    "rgb_cost_volume_only": True,
                    "texture_feature_dim": 4,
                    "hidden_dim": 4,
                },
                "lineage": {"layout_sha256": file_sha256_short(layout_path)},
                "training": {
                    "coherent_hard_repeat_edge": {"enabled": True},
                    "inner_validation": {
                        "fold_count": 2,
                        "fold_index": 1,
                        "selected_epoch": 1,
                        "gate": {"hard_repeat_passed": True},
                    },
                },
            },
        },
        path,
    )


def test_texture_checkpoint_requires_exact_frozen_layout_lineage(tmp_path) -> None:
    layout_path = tmp_path / "layout.npz"
    layout_path.write_bytes(b"frozen-layout")
    checkpoint = tmp_path / "texture.pt"
    _checkpoint(checkpoint, layout_path=layout_path)
    encoder, contract = _load_frozen_texture_encoder(
        path=checkpoint,
        layout_path=layout_path,
        layout=_Layout(),
        device=torch.device("cpu"),
    )
    assert isinstance(encoder, TexturePatchEncoder)
    assert contract["local_hard_repeat_gate_passed"] is True
    other_layout = tmp_path / "other-layout.npz"
    other_layout.write_bytes(b"different-layout")
    with pytest.raises(ValueError, match="incompatible"):
        _load_frozen_texture_encoder(
            path=checkpoint,
            layout_path=other_layout,
            layout=_Layout(),
            device=torch.device("cpu"),
        )


def test_support_view_sweep_requires_exact_candidate_and_support_prefix() -> None:
    reference = _support_view_layout(view_count=1)
    runtime = _support_view_layout(view_count=3)
    _validate_edge_encoder_support_view_sweep(
        checkpoint_layout=reference,
        runtime_layout=runtime,
    )

    runtime.support_image_ids[0, 0, 0] = "different.png"
    with pytest.raises(ValueError, match="ordered support_image_ids prefix"):
        _validate_edge_encoder_support_view_sweep(
            checkpoint_layout=reference,
            runtime_layout=runtime,
        )


def test_support_view_sweep_rejects_candidate_drift() -> None:
    reference = _support_view_layout(view_count=1)
    runtime = _support_view_layout(view_count=2)
    runtime.candidate_track_ids[1, 1] = 999
    with pytest.raises(ValueError, match="candidate_track_ids"):
        _validate_edge_encoder_support_view_sweep(
            checkpoint_layout=reference,
            runtime_layout=runtime,
        )


def test_target_free_phase_shard_rejects_lineage_drift(tmp_path) -> None:
    scale = RGBPhaseScale("tiny", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    lineage = {"layout_sha256": "layout", "texture_checkpoint_sha256": "checkpoint"}
    uniform = np.full((1, 2, 1, 9), -math.log(9.0), dtype=np.float32)
    feature = RGBPhaseQueryFeatures(
        query_id="q.png",
        source_point_ids=np.asarray([7], dtype=np.int64),
        candidate_view_weights=np.ones((1, 2, 1), dtype=np.float32),
        log_probabilities={branch: {scale.name: uniform} for branch in BRANCHES},
        edge_usable={
            branch: {scale.name: np.ones((1, 2, 1), dtype=bool)} for branch in BRANCHES
        },
    )
    path = tmp_path / "feature.npz"
    _write_training_feature(path=path, features=feature, lineage=lineage)
    loaded = _load_training_feature(
        path=path,
        expected_query_id="q.png",
        expected_lineage=lineage,
        scales=(scale,),
    )
    assert loaded.query_id == "q.png"
    with pytest.raises(ValueError, match="contract"):
        _load_training_feature(
            path=path,
            expected_query_id="q.png",
            expected_lineage={"layout_sha256": "stale"},
            scales=(scale,),
        )


def test_checkpoint_inner_split_keeps_gate_on_held_queries_only() -> None:
    all_ids = ("q0", "q1", "q2", "q3", "q4", "q5")
    held_all, held, train, complete = _checkpoint_inner_query_split(
        all_train_query_ids=all_ids,
        selected_query_ids=all_ids,
        fold_count=3,
        fold_index=1,
    )
    assert held_all == ("q1", "q4")
    assert held == held_all
    assert train == ("q0", "q2", "q3", "q5")
    assert complete is True
    _, partial_held, _, partial_complete = _checkpoint_inner_query_split(
        all_train_query_ids=all_ids,
        selected_query_ids=("q0", "q1"),
        fold_count=3,
        fold_index=1,
    )
    assert partial_held == ("q1",)
    assert partial_complete is False
