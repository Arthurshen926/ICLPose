from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_hybrid_context_coverage import (
    summarize_hybrid_context_coverage,
)
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_hybrid_context_manifest import (
    build_hybrid_context_manifest,
)
from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_hybrid_context import (
    HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
    HYBRID_CONTEXT_PROFILE_NAMES,
    HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    load_frozen_fulltrack_per_view_appearance_features,
)


_LINEAGE_KEYS = (
    "source_frozen_appearance_artifact",
    "source_frozen_appearance_artifact_sha256",
    "source_fulltrack_per_view_artifact",
    "source_fulltrack_per_view_artifact_sha256",
    "source_edge_candidate_offsets_sha256",
    "source_edge_geometry_rows_sha256",
    "source_candidate_tracks_sha256",
    "source_candidate_probabilities_sha256",
    "source_null_probabilities_sha256",
    "source_verification_rows_sha256",
    "support_geometry_index_sha256",
)


def _metadata(*, artifact_format: str, semantics: str) -> dict[str, object]:
    return {
        "format": artifact_format,
        "version": 1,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "fixed_candidate_top_k": 20,
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_count_cap": None,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": semantics,
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
        },
        **{key: f"fixture-{key}" for key in _LINEAGE_KEYS},
    }


def _write_component(
    *,
    path: Path,
    profile_names: tuple[str, ...],
    artifact_format: str,
    semantics: str,
    track_delta: int = 0,
) -> None:
    rows = 192
    candidates = 20
    edges = rows * candidates
    score_value = (
        0.25
        if artifact_format == FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT
        else 0.75
    )
    scores = np.full((edges, len(profile_names)), score_value, dtype=np.float16)
    np.savez_compressed(
        path,
        verification_query_ids=np.asarray(["seq1/frame00001.png"] * rows, dtype=np.str_),
        split_names=np.asarray(["train"] * rows, dtype=np.str_),
        verification_source_row_indices=np.arange(rows, dtype=np.int64),
        verification_xy=np.zeros((rows, 2), dtype=np.float32),
        candidate_track_ids=(
            np.arange(rows * candidates, dtype=np.int64).reshape(rows, candidates)
            + int(track_delta)
        ),
        candidate_probabilities=np.full((rows, candidates), 0.045, dtype=np.float32),
        null_probabilities=np.full((rows,), 0.1, dtype=np.float32),
        candidate_support_observation_counts=np.ones((rows, candidates), dtype=np.int64),
        profile_names=np.asarray(profile_names, dtype=np.str_),
        edge_candidate_offsets=np.arange(edges + 1, dtype=np.int64),
        edge_geometry_rows=np.arange(edges, dtype=np.int64),
        edge_profile_scores=scores,
        edge_profile_valid=np.ones(scores.shape, dtype=bool),
        metadata_json=np.asarray(
            json.dumps(_metadata(artifact_format=artifact_format, semantics=semantics))
        ),
    )


def _components(tmp_path: Path) -> tuple[Path, Path]:
    translation = tmp_path / "translation.npz"
    absolute = tmp_path / "absolute.npz"
    _write_component(
        path=translation,
        profile_names=tuple(MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES),
        artifact_format=FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
        semantics=FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    )
    _write_component(
        path=absolute,
        profile_names=tuple(FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES),
        artifact_format=FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
        semantics=FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    )
    return translation, absolute


def test_hybrid_manifest_selects_only_visual_absolute_intermediate_columns(
    tmp_path: Path,
) -> None:
    translation, absolute = _components(tmp_path)
    manifest = tmp_path / "hybrid.json"
    build_hybrid_context_manifest(
        translation_artifacts=(translation,),
        absolute_artifacts=(absolute,),
        output=manifest,
    )
    features = load_frozen_fulltrack_per_view_appearance_features((manifest,))
    assert len(HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES) == 70
    assert len(HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES) == 135
    assert features.profile_names == HYBRID_CONTEXT_PROFILE_NAMES
    assert features.edge_profile_scores.shape == (3840, 205)
    np.testing.assert_allclose(features.edge_profile_scores[:, 0], 0.25)
    np.testing.assert_allclose(features.edge_profile_scores[:, 70], 0.75)
    assert all("coverage_control" not in name for name in features.profile_names)

    coverage = summarize_hybrid_context_coverage(features=features)
    train_top = next(
        row
        for row in coverage
        if row["split"] == "train" and row["rank_bucket"] == "rank_1_5"
    )
    assert train_top["hybrid_candidate_coverage_rate"] == 1.0
    assert train_top["translation_joint_edge_coverage_rate"] == 1.0
    assert train_top["absolute_joint_edge_coverage_rate"] == 1.0


def test_hybrid_manifest_rejects_misaligned_csr_before_writing(tmp_path: Path) -> None:
    translation, absolute = _components(tmp_path)
    _write_component(
        path=absolute,
        profile_names=tuple(FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES),
        artifact_format=FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
        semantics=FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
        track_delta=1,
    )
    with pytest.raises(ValueError, match="hybrid CSR differs for candidate_track_ids"):
        build_hybrid_context_manifest(
            translation_artifacts=(translation,),
            absolute_artifacts=(absolute,),
            output=tmp_path / "hybrid.json",
        )


def test_hybrid_loader_rejects_stale_component_hash(tmp_path: Path) -> None:
    translation, absolute = _components(tmp_path)
    manifest = tmp_path / "hybrid.json"
    build_hybrid_context_manifest(
        translation_artifacts=(translation,),
        absolute_artifacts=(absolute,),
        output=manifest,
    )
    translation.write_bytes(b"stale")
    with pytest.raises(ValueError, match="hybrid translation source is stale"):
        load_frozen_fulltrack_per_view_appearance_features((manifest,))
