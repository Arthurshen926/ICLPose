from __future__ import annotations

import json

import numpy as np

import feature_extract.tools.vfm.build_frozen_fulltrack_global_context as builder
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    load_frozen_fulltrack_appearance_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)


def _source_artifact(path, *, context_cache) -> None:
    count, candidates = 192, 20
    tracks = np.full((count, candidates), -1, dtype=np.int64)
    tracks[:, :2] = np.asarray([10, 11], dtype=np.int64)
    probabilities = np.zeros((count, candidates), dtype=np.float32)
    probabilities[:, :2] = np.asarray([0.45, 0.45], dtype=np.float32)
    metadata = {
        "format": "frozen_multiscale_candidate_absolute_appearance_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "inputs": {
            "radio_final_context_cache": {
                "sha256": file_sha256_short(context_cache)
            }
        },
        "strict_frozen_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "fixed_candidate_top_k": 20,
            "heldout_s0_verification_rows": True,
        },
    }
    np.savez_compressed(
        path,
        verification_query_ids=np.asarray(["q.png"] * count),
        split_names=np.asarray(["train"] * count),
        verification_source_row_indices=np.arange(count, dtype=np.int64),
        verification_xy=np.full((count, 2), 2.0, dtype=np.float32),
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        candidate_view_weights=np.concatenate(
            (
                np.ones((count, 2, 1), dtype=np.float32),
                np.zeros((count, candidates - 2, 1), dtype=np.float32),
            ),
            axis=1,
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _context_cache(path) -> None:
    metadata = {
        "format": "radio_final_context_pca_v1",
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
    }
    np.savez_compressed(
        path,
        image_ids=np.asarray(["a.png", "b.png", "q.png"]),
        global_descriptors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        ),
        summary_descriptors=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _raw_per_view_source(path, *, geometry_path, context_path, s0_path) -> None:
    rows, candidates = 192, 20
    tracks = np.full((rows, candidates), -1, dtype=np.int64)
    tracks[:, :2] = np.asarray([10, 11], dtype=np.int64)
    probabilities = np.zeros((rows, candidates), dtype=np.float32)
    probabilities[:, :2] = np.asarray([0.45, 0.45], dtype=np.float32)
    counts = np.zeros((rows, candidates), dtype=np.int64)
    counts[:, 0] = 2
    counts[:, 1] = 1
    offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(counts.reshape(-1), dtype=np.int64))
    )
    geometry_rows = np.tile(np.asarray([0, 2, 1], dtype=np.int64), rows)
    metadata = {
        "format": "frozen_fulltrack_candidate_per_view_appearance_v1",
        "version": "frozen_fulltrack_candidate_per_view_appearance_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "fixed_candidate_top_k": 20,
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_count_cap": None,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": "raw_aligned_ncc_per_real_sfm_observation_v1",
        "profiles": [{"name": "raw", "source": "fixture", "window_size": 1}],
        "appearance_config": {"fixture": True},
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "context_cache_sha256": {"radio_final": file_sha256_short(context_path)},
        "source_frozen_appearance_artifact": str(s0_path),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(s0_path),
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
        },
    }
    np.savez_compressed(
        path,
        verification_query_ids=np.asarray(["q.png"] * rows),
        split_names=np.asarray(["train"] * rows),
        verification_source_row_indices=np.arange(rows, dtype=np.int64),
        verification_xy=np.full((rows, 2), 2.0, dtype=np.float32),
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=np.full((rows,), 0.1, dtype=np.float32),
        candidate_support_observation_counts=counts,
        source_maplet_support_view_counts=np.where(counts > 0, 1, 0),
        profile_names=np.asarray(["raw"], dtype=np.str_),
        edge_candidate_offsets=offsets,
        edge_geometry_rows=geometry_rows,
        edge_profile_scores=np.ones((len(geometry_rows), 1), dtype=np.float16),
        edge_profile_valid=np.ones((len(geometry_rows), 1), dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_builder_keeps_all_observations_and_exports_named_global_cosines(tmp_path) -> None:
    context_path = tmp_path / "context.npz"
    _context_cache(context_path)
    source_path = tmp_path / "source.npz"
    _source_artifact(source_path, context_cache=context_path)
    geometry_path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png", "b.png"),
            image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
            source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
            track_ids=np.asarray([10, 11, 10], dtype=np.int64),
            xy=np.asarray(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
            ),
            viewing_rays=np.ones((3, 3), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        geometry_path,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    output = tmp_path / "fulltrack_global.npz"
    result = builder.build_frozen_fulltrack_global_context(
        source_appearance_artifact=source_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )

    assert result["candidate_entries_expanded_beyond_maplet"] == 192
    with np.load(output, allow_pickle=False) as data:
        assert data["candidate_support_observation_counts"][0, :2].tolist() == [2, 1]
        assert data["source_maplet_support_view_counts"][0, :2].tolist() == [1, 1]
        np.testing.assert_allclose(data["candidate_probabilities"][0, :2], [0.45, 0.45])
        names = data["feature_names"].astype(str).tolist()
        assert names[:4] == [
            "radio_final_global__uniform_mean_cosine",
            "radio_final_global__uniform_logmeanexp_tau0p05_cosine",
            "radio_final_global__uniform_top4_mean_cosine",
            "radio_final_global__uniform_max_cosine",
        ]
        # Track 10 sees a matching and an orthogonal support image; track 11
        # sees only the matching image.  No support view was selected or dropped.
        np.testing.assert_allclose(
            data["candidate_summary_features"][0, :2, 0], [0.5, 1.0]
        )
        metadata = json.loads(str(data["metadata_json"].item()))
        strict = metadata["strict_fulltrack_appearance_contract"]
        assert strict["all_real_sfm_track_observations_enumerated"] is True
        assert strict["global_context_hard_retrieval_or_candidate_reselection"] is False
    loaded = load_frozen_fulltrack_appearance_features((output,))
    assert loaded.compatibility["descriptor_projection_contract"] == {
        "profile_names": ["radio_final_global", "radio_final_summary"],
        "radio_intermediate_projection": {
            "applicable": False,
            "reason": "no_radio_intermediate_profile_v1",
        },
    }


def test_builder_preserves_current_raw_per_view_csr_edges(tmp_path) -> None:
    context_path = tmp_path / "context.npz"
    _context_cache(context_path)
    geometry_path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png", "b.png"),
            image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
            source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
            track_ids=np.asarray([10, 11, 10], dtype=np.int64),
            xy=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
            viewing_rays=np.ones((3, 3), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        geometry_path,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    s0_path = tmp_path / "s0.npz"
    s0_path.write_bytes(b"immutable-s0-root")
    raw_path = tmp_path / "raw-per-view.npz"
    _raw_per_view_source(
        raw_path,
        geometry_path=geometry_path,
        context_path=context_path,
        s0_path=s0_path,
    )
    output = tmp_path / "global.npz"
    builder.build_frozen_fulltrack_global_context(
        source_appearance_artifact=raw_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=context_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )
    with np.load(raw_path, allow_pickle=False) as raw, np.load(output, allow_pickle=False) as built:
        np.testing.assert_array_equal(
            built["candidate_track_ids"], raw["candidate_track_ids"]
        )
        np.testing.assert_array_equal(
            built["candidate_probabilities"], raw["candidate_probabilities"]
        )
        np.testing.assert_array_equal(
            built["candidate_support_observation_counts"],
            raw["candidate_support_observation_counts"],
        )
        metadata = json.loads(str(built["metadata_json"].item()))
    assert metadata["source_fulltrack_edge_contract"] == "raw_fulltrack_per_view_csr_v1"
    assert metadata["source_fulltrack_per_view_artifact_sha256"] == file_sha256_short(raw_path)
    assert metadata["source_frozen_appearance_artifact_sha256"] == file_sha256_short(s0_path)
    assert metadata["strict_fulltrack_appearance_contract"][
        "source_fulltrack_csr_edges_preserved"
    ] is True
