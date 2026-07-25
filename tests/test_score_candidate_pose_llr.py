from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.score_candidate_pose_llr import (
    remap_support_image_indices,
    resolve_edge_chunk_size,
    validate_baseline_reference_hypothesis_equivalence,
    validate_checkpoint_for_target_free_scoring,
    validate_checkpoint_hypothesis_semantic_lineage,
)
from feature_extract.vfm.localization.candidate_pose_llr import grouped_hypothesis_semantic_manifest
from feature_extract.vfm.localization.candidate_pose_llr import CandidatePoseLLRRuntime


def _runtime() -> CandidatePoseLLRRuntime:
    return CandidatePoseLLRRuntime(
        query_image_indices=torch.tensor([0, 1]),
        support_image_indices=torch.tensor([[[1, 2]], [[2, 1]]]),
        support_xy=torch.tensor(
            [
                [[[10.0, 20.0], [30.0, 40.0]]],
                [[[50.0, 60.0], [70.0, 80.0]]],
            ]
        ),
        support_view_valid=torch.tensor([[[True, True]], [[True, False]]]),
        candidate_view_weights=torch.tensor([[[0.5, 0.5]], [[1.0, 0.0]]]),
        candidate_probabilities=torch.tensor([[0.8], [0.8]]),
        null_probabilities=torch.tensor([0.2, 0.2]),
    )


def test_support_descriptor_control_remaps_only_support_image_addresses() -> None:
    runtime = _runtime()
    remapped = remap_support_image_indices(
        runtime=runtime,
        image_ids=("query.png", "support-a.png", "support-b.png"),
        descriptor_derangement={
            "query.png": "query.png",
            "support-a.png": "support-b.png",
            "support-b.png": "support-a.png",
        },
    )

    torch.testing.assert_close(remapped.query_image_indices, runtime.query_image_indices)
    torch.testing.assert_close(remapped.support_xy, runtime.support_xy)
    torch.testing.assert_close(remapped.support_view_valid, runtime.support_view_valid)
    torch.testing.assert_close(remapped.candidate_view_weights, runtime.candidate_view_weights)
    torch.testing.assert_close(remapped.candidate_probabilities, runtime.candidate_probabilities)
    torch.testing.assert_close(remapped.null_probabilities, runtime.null_probabilities)
    torch.testing.assert_close(
        remapped.support_image_indices,
        torch.tensor([[[2, 1]], [[1, 2]]]),
    )


def test_target_free_checkpoint_contract_rejects_stale_static_inputs() -> None:
    metadata = {
        "format": "candidate_pose_llr_checkpoint_v1",
        "model_format": "candidate_specific_pose_llr_v3",
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "explicit_null": True,
        "inputs": {
            "verification_points": {"sha256": "a"},
            "maplet_support_index": {"sha256": "b"},
            "support_geometry_index": {"sha256": "c"},
            "projected_landmark_bank": {"sha256": "d"},
            "radio_final_context_cache": {"sha256": "e"},
            "radio_intermediate_context_cache": {"sha256": "f"},
            "alike_spatial_context_cache": {"sha256": "g"},
            "colmap_cameras_bin": {"sha256": "h"},
            "colmap_images_bin": {"sha256": "i"},
        },
    }
    expected = {
        "verification_points": {"sha256": "a"},
        "maplet_support_index": {"sha256": "b"},
        "support_geometry_index": {"sha256": "c"},
        "projected_landmark_bank": {"sha256": "d"},
        "radio_final_context_cache": {"sha256": "e"},
        "radio_intermediate_context_cache": {"sha256": "f"},
        "alike_spatial_context_cache": {"sha256": "g"},
        "colmap_cameras_bin": {"sha256": "h"},
        "colmap_images_bin": {"sha256": "i"},
    }
    validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=expected)

    stale = {key: dict(value) for key, value in expected.items()}
    stale["alike_spatial_context_cache"] = {"sha256": "other"}
    with pytest.raises(ValueError, match="stale"):
        validate_checkpoint_for_target_free_scoring(metadata, expected_inputs=stale)


def test_edge_chunk_override_keeps_checkpoint_default_explicit() -> None:
    assert resolve_edge_chunk_size(checkpoint_edge_chunk_size=2048, requested_edge_chunk_size=0) == 2048
    assert resolve_edge_chunk_size(checkpoint_edge_chunk_size=2048, requested_edge_chunk_size=8192) == 8192
    with pytest.raises(ValueError, match="edge chunk"):
        resolve_edge_chunk_size(checkpoint_edge_chunk_size=0, requested_edge_chunk_size=0)


def test_checkpoint_rejects_heldout_hypothesis_semantic_mismatch() -> None:
    current = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
        "inputs": {"candidate_artifact_sha256": "current"},
        "grouped_config": {"latent_em_enabled": True},
    }
    old = {
        **current,
        "inputs": {"candidate_artifact_sha256": "old"},
        "grouped_config": {"latent_em_enabled": False},
    }
    checkpoint = {"hypothesis_semantic_lineage": grouped_hypothesis_semantic_manifest(current)}

    validate_checkpoint_hypothesis_semantic_lineage(
        checkpoint_metadata=checkpoint,
        hypothesis_metadata=current,
    )
    with pytest.raises(ValueError, match="semantic lineage"):
        validate_checkpoint_hypothesis_semantic_lineage(
            checkpoint_metadata=checkpoint,
            hypothesis_metadata=old,
        )


def test_baseline_reference_bridge_requires_exact_frozen_pose_rows(tmp_path: Path) -> None:
    def write_hypotheses(path: Path, *, pose_offset: float) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        poses[0, 0, 3] = float(pose_offset)
        metadata = {
            "format": "grouped_pose_hypotheses_inference_only_v1",
            "contains_target_fields": False,
            "pose_or_ground_truth_used_for_generation": False,
            "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
            "inputs": {"candidate_artifact_sha256": "frozen"},
            "grouped_config": {"latent_em_enabled": True},
            "row_count": 2,
        }
        np.savez_compressed(
            path,
            query_ids=np.asarray(["query.png", "query.png"]),
            split_names=np.asarray(["validation", "validation"]),
            evaluation_labels=np.asarray(["optional", "optional"]),
            hypothesis_indices=np.asarray([0, 1], dtype=np.int64),
            generation_profiles=np.asarray(["raw", "raw"]),
            selection_modes=np.asarray(["fixed", "fixed"]),
            shortlisted_for_verification=np.asarray([True, True]),
            chosen_for_optional_pose=np.asarray([True, False]),
            poses_w2c=poses,
            preliminary_log_likelihood_means=np.asarray([1.0, 0.5]),
            shortlist_log_likelihood_means=np.asarray([1.0, 0.5]),
            verification_log_likelihood_means=np.asarray([1.0, 0.5]),
            verification_relation_log_likelihood_ratio_means=np.asarray([0.0, 0.0]),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    reference = tmp_path / "reference.npz"
    current = tmp_path / "current.npz"
    drifted = tmp_path / "drifted.npz"
    write_hypotheses(reference, pose_offset=0.1)
    write_hypotheses(current, pose_offset=0.1)
    write_hypotheses(drifted, pose_offset=0.2)

    bridge = validate_baseline_reference_hypothesis_equivalence(
        current_hypothesis_path=current,
        reference_hypothesis_path=reference,
    )

    assert bridge["validated_row_count"] == 2
    assert bridge["rule"] == "exact_target_free_row_pose_selector_equivalence_v1"
    with pytest.raises(ValueError, match="pose"):
        validate_baseline_reference_hypothesis_equivalence(
            current_hypothesis_path=drifted,
            reference_hypothesis_path=reference,
        )
