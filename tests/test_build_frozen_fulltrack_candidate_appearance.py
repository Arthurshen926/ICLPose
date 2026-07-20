from __future__ import annotations

import json

import numpy as np

import feature_extract.tools.vfm.build_frozen_fulltrack_candidate_appearance as builder
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
)


def _source_artifact(path, *, cache_paths) -> None:
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
            name: {"path": str(value), "sha256": file_sha256_short(value)}
            for name, value in cache_paths.items()
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


def test_builder_enumerates_full_track_observations_without_changing_candidates(
    tmp_path, monkeypatch
) -> None:
    cache_paths = {
        "radio_final_context_cache": tmp_path / "final.npz",
        "radio_intermediate_context_cache": tmp_path / "intermediate.npz",
        "alike_spatial_context_cache": tmp_path / "alike.npz",
    }
    for path in cache_paths.values():
        path.write_bytes(b"fixture-cache")
    source_path = tmp_path / "source.npz"
    _source_artifact(source_path, cache_paths=cache_paths)
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
    image_ids = np.asarray(["a.png", "b.png", "q.png"])
    source = ImageGridFeatureSource(
        name="fixture",
        image_ids=image_ids,
        image_sizes=np.asarray([[5, 5], [5, 5], [5, 5]], dtype=np.int64),
        grid_size=2,
        descriptors=np.asarray(
            [
                [[1.0, 0.0]] * 4,
                [[0.0, 1.0]] * 4,
                [[1.0, 0.0]] * 4,
            ],
            dtype=np.float32,
        ).reshape(3, 4, 2),
        metadata={},
    )
    source_load_kwargs = {}

    def fake_sources(**kwargs):
        source_load_kwargs.update(kwargs)
        return {
            "radio_final": source,
            "radio_intermediate": source,
            "alike": source,
        }

    monkeypatch.setattr(builder, "_as_image_grid_sources", fake_sources)
    output = tmp_path / "fulltrack.npz"
    result = builder.build_frozen_fulltrack_candidate_appearance(
        source_appearance_artifact=source_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=cache_paths["radio_final_context_cache"],
        radio_intermediate_context_cache=cache_paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=cache_paths["alike_spatial_context_cache"],
        profiles="final:radio_final:1",
        template_batch_size=256,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
        device="cpu",
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
        retain_per_view_edges=True,
    )
    assert source_load_kwargs["allow_mismatched_descriptor_dimensions"] is False
    assert result["candidate_entries_expanded_beyond_maplet"] == 192
    with np.load(output, allow_pickle=False) as data:
        assert data["candidate_support_observation_counts"][0, :2].tolist() == [2, 1]
        assert data["source_maplet_support_view_counts"][0, :2].tolist() == [1, 1]
        np.testing.assert_allclose(data["candidate_probabilities"][0, :2], [0.45, 0.45])
        metadata = json.loads(str(data["metadata_json"].item()))
        assert metadata["support_view_source"] == "all_real_sfm_track_observations_v1"
        assert metadata["strict_fulltrack_appearance_contract"]["support_view_count_cap"] is None
        assert metadata["format"] == builder.PER_VIEW_ARTIFACT_FORMAT
        assert metadata["per_view_edges_retained"] is True
        assert data["edge_candidate_offsets"].shape == (192 * 20 + 1,)
        assert data["edge_candidate_offsets"][-1] == len(data["edge_geometry_rows"])
        assert data["edge_profile_scores"].shape == data["edge_profile_valid"].shape
    assert result["format"] == builder.PER_VIEW_ARTIFACT_FORMAT


def _intermediate_pca_cache(path, *, projection_dim: int) -> None:
    metadata = {
        "format": "radio_intermediate_image_context_pca_v1",
        "source_context_sha256": "source-context",
        "source_image_manifest_sha256": "image-manifest",
        "pca_training_manifest_sha256": "pca-manifest",
        "pca_training_image_list_sha256": "pca-images",
        "pca_training_image_count": 2,
        "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
        "radio_checkpoint_sha256": "radio",
        "radio_version": "c-radio",
        "intermediate_index": -6,
        "normalization": "input_row_l2_centered_pca_output_row_l2",
        "spatial_grid_sizes": [16],
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "projection_dim": int(projection_dim),
    }
    np.savez_compressed(
        path,
        image_ids=np.asarray(["a.png", "q.png"]),
        image_sizes=np.asarray([[5, 5], [5, 5]], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_builder_allows_only_a_lineage_matched_higher_dim_intermediate_override(
    tmp_path, monkeypatch
) -> None:
    cache_paths = {
        "radio_final_context_cache": tmp_path / "final.npz",
        "radio_intermediate_context_cache": tmp_path / "intermediate64.npz",
        "alike_spatial_context_cache": tmp_path / "alike.npz",
    }
    cache_paths["radio_final_context_cache"].write_bytes(b"fixture-final")
    cache_paths["alike_spatial_context_cache"].write_bytes(b"fixture-alike")
    _intermediate_pca_cache(cache_paths["radio_intermediate_context_cache"], projection_dim=64)
    override = tmp_path / "intermediate256.npz"
    _intermediate_pca_cache(override, projection_dim=256)
    source_path = tmp_path / "source.npz"
    _source_artifact(source_path, cache_paths=cache_paths)
    geometry_path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png",),
            image_offsets=np.asarray([0, 2], dtype=np.int64),
            source_row_indices=np.asarray([0, 1], dtype=np.int64),
            track_ids=np.asarray([10, 11], dtype=np.int64),
            xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            viewing_rays=np.ones((2, 3), dtype=np.float32),
            reprojection_errors=np.zeros((2,), dtype=np.float32),
        ),
        geometry_path,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    source = ImageGridFeatureSource(
        name="fixture",
        image_ids=np.asarray(["a.png", "q.png"]),
        image_sizes=np.asarray([[5, 5], [5, 5]], dtype=np.int64),
        grid_size=2,
        descriptors=np.asarray([[[1.0, 0.0]] * 4, [[1.0, 0.0]] * 4], dtype=np.float32),
        metadata={},
    )
    source_load_kwargs = {}

    def fake_sources(**kwargs):
        source_load_kwargs.update(kwargs)
        return {
            "radio_final": source,
            "radio_intermediate": source,
            "alike": source,
        }

    monkeypatch.setattr(builder, "_as_image_grid_sources", fake_sources)
    output = tmp_path / "fulltrack_override.npz"
    builder.build_frozen_fulltrack_candidate_appearance(
        source_appearance_artifact=source_path,
        support_geometry_index=geometry_path,
        radio_final_context_cache=cache_paths["radio_final_context_cache"],
        radio_intermediate_context_cache=override,
        alike_spatial_context_cache=cache_paths["alike_spatial_context_cache"],
        profiles="intermediate:radio_intermediate:1",
        template_batch_size=256,
        minimum_support_fraction=0.75,
        minimum_overlap_fraction=0.75,
        device="cpu",
        output=output,
        summary_json=tmp_path / "summary_override.json",
        force=False,
        allow_radio_intermediate_projection_override=True,
    )
    assert source_load_kwargs["allow_mismatched_descriptor_dimensions"] is True
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    override_metadata = metadata["radio_intermediate_projection_override"]
    assert override_metadata["enabled"] is True
    assert override_metadata["source_projection_dim"] == 64
    assert override_metadata["override_projection_dim"] == 256
