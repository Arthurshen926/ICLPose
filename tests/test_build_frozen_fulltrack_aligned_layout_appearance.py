from __future__ import annotations

import json

import numpy as np

import feature_extract.tools.vfm.build_frozen_fulltrack_aligned_layout_appearance as builder
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    ALIGNED_LAYOUT_FEATURE_NAMES,
    FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT,
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
)


def _raw_per_view_artifact(path, *, geometry_path, cache_paths) -> None:
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
        "context_cache_sha256": {
            "radio_final": file_sha256_short(cache_paths["radio_final"]),
            "radio_intermediate": file_sha256_short(cache_paths["radio_intermediate"]),
            "alike": file_sha256_short(cache_paths["alike"]),
        },
        "radio_intermediate_projection_override": {"enabled": False},
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
        "implementation": {"fixture": True},
    }
    np.savez_compressed(
        path,
        verification_query_ids=np.asarray(["q.png"] * rows),
        split_names=np.asarray(["train"] * rows),
        verification_source_row_indices=np.arange(rows, dtype=np.int64),
        verification_xy=np.full((rows, 2), 15.0, dtype=np.float32),
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=np.full((rows,), 0.1, dtype=np.float32),
        candidate_support_observation_counts=counts,
        profile_names=np.asarray(["raw_ncc"], dtype=np.str_),
        edge_candidate_offsets=offsets,
        edge_geometry_rows=geometry_rows,
        edge_profile_scores=np.ones((len(geometry_rows), 1), dtype=np.float16),
        edge_profile_valid=np.ones((len(geometry_rows), 1), dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _source() -> ImageGridFeatureSource:
    image_ids = np.asarray(["a.png", "b.png", "q.png"])
    descriptors = np.zeros((3, 16 * 16, 2), dtype=np.float32)
    descriptors[..., 0] = 1.0
    return ImageGridFeatureSource(
        name="fixture",
        image_ids=image_ids,
        image_sizes=np.asarray([[31, 31]] * len(image_ids), dtype=np.int64),
        grid_size=16,
        descriptors=descriptors,
        metadata={},
    )


def test_layout_builder_preserves_frozen_edges_and_exports_spatial_dct(
    tmp_path, monkeypatch
) -> None:
    cache_paths = {
        "radio_final": tmp_path / "final.npz",
        "radio_intermediate": tmp_path / "intermediate.npz",
        "alike": tmp_path / "alike.npz",
    }
    for cache in cache_paths.values():
        cache.write_bytes(b"fixture-cache")
    geometry_path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png", "b.png"),
            image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
            source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
            track_ids=np.asarray([10, 11, 10], dtype=np.int64),
            xy=np.asarray([[15.0, 15.0]] * 3, dtype=np.float32),
            viewing_rays=np.ones((3, 3), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        geometry_path,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    raw = tmp_path / "raw_fulltrack.npz"
    _raw_per_view_artifact(raw, geometry_path=geometry_path, cache_paths=cache_paths)
    source = _source()

    def fake_sources(**kwargs):
        assert kwargs["allow_mismatched_descriptor_dimensions"] is False
        return {
            "radio_final": source,
            "radio_intermediate": source,
            "alike": source,
        }

    monkeypatch.setattr(builder, "_as_image_grid_sources", fake_sources)
    output = tmp_path / "layout.npz"
    result = builder.build_frozen_fulltrack_aligned_layout_appearance(
        source_per_view_artifact=raw,
        support_geometry_index=geometry_path,
        radio_final_context_cache=cache_paths["radio_final"],
        radio_intermediate_context_cache=cache_paths["radio_intermediate"],
        alike_spatial_context_cache=cache_paths["alike"],
        template_batch_size=256,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
        device="cpu",
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )
    assert result["feature_count"] == len(ALIGNED_LAYOUT_FEATURE_NAMES)
    with np.load(raw, allow_pickle=False) as source_payload, np.load(
        output, allow_pickle=False
    ) as payload:
        np.testing.assert_array_equal(
            payload["candidate_track_ids"], source_payload["candidate_track_ids"]
        )
        np.testing.assert_array_equal(
            payload["candidate_probabilities"], source_payload["candidate_probabilities"]
        )
        np.testing.assert_array_equal(
            payload["edge_candidate_offsets"], source_payload["edge_candidate_offsets"]
        )
        np.testing.assert_array_equal(
            payload["edge_geometry_rows"], source_payload["edge_geometry_rows"]
        )
        assert payload["edge_profile_scores"].shape == (
            192 * 3,
            len(ALIGNED_LAYOUT_FEATURE_NAMES),
        )
        assert payload["edge_profile_valid"].all()
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["format"] == FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT
    assert (
        metadata["per_view_edge_feature_semantics"]
        == FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS
    )
    assert metadata["aligned_layout_contract"]["candidate_coordinates_or_pose_used"] is False
    assert metadata["radio_intermediate_projection_override"] == {"enabled": False}
    assert metadata["source_csr_array_hash_scheme"] == "dtype_shape_bytes_sha256_v1"
    loaded = load_frozen_fulltrack_per_view_appearance_features((output,))
    assert fulltrack_per_view_feature_granularity(loaded) == (
        "sparse_per_real_support_view_aligned_spatial_dct_v1"
    )
