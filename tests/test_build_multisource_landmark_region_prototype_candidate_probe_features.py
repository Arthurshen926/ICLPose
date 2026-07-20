from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_multisource_landmark_region_prototype_candidate_probe_features import (
    ARTIFACT_FORMAT,
    build_multisource_landmark_region_prototype_candidate_probe_features,
)
from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT as INTERMEDIATE_PCA_FORMAT,
)
from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    ARTIFACT_FORMAT as ALIKE_SPATIAL_FORMAT,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    MULTISOURCE_LANDMARK_REGION_APPEARANCE_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
)


_IMAGE_SOURCE_CONTRACT = {
    "version": 1,
    "resolved_image_root": "/tmp/images",
    "image_count": 3,
    "image_ids_sha256": "ids-test",
    "sampled_content_manifest_sha256": "image-manifest",
    "source_image_dimensions": {"80x80": 3},
    "sampled_bytes_per_file_end": 32768,
}
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)


def _unit_grid(count: int, size: int, *, axis: int) -> np.ndarray:
    values = np.zeros((count, size * size, 2), dtype=np.float32)
    values[..., axis] = 1.0
    return values


def _write_layout_and_geometry(tmp_path):
    layout = tmp_path / "layout.npz"
    maplet = tmp_path / "maplet.npz"
    geometry = tmp_path / "geometry.npz"
    np.savez(
        layout,
        source_row_indices=np.asarray([3], dtype=np.int64),
        query_ids=np.asarray(["q.png"]),
        split_names=np.asarray(["train"]),
        xy=np.asarray([[40.0, 40.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[11]], dtype=np.int64),
        candidate_canonical_rows=np.asarray([[0]], dtype=np.int64),
        candidate_features=np.asarray(
            [[[[0.25, 0.5, 0.75], [0.25, 0.6, 0.8]]]], dtype=np.float32
        ),
        candidate_view_valid=np.asarray([[[True, True]]]),
        candidate_support_image_ids=np.asarray([[["s1.png", "s2.png"]]]),
        candidate_support_coverage_counts=np.asarray([[[3, 2]]], dtype=np.int32),
        feature_names=np.asarray(
            [
                "radio_final_anchor_cosine",
                "radio_intermediate_anchor_cosine",
                "alike_anchor_cosine",
            ]
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "multiscale_candidate_probe_features_v1",
                    "contains_ground_truth": False,
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "whole_image_summary_or_global_used": False,
                    "proposals_sha256": "test-proposals-sha",
                    "mapper_checkpoint_sha256": "mapper-test",
                }
            )
        ),
    )
    np.savez(
        maplet,
        anchor_track_ids=np.asarray([11], dtype=np.int64),
        support_image_ids=np.asarray(["s1.png", "s2.png"]),
        support_image_indices=np.asarray([[0, 1, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[3, 2, 0]], dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps({"format": "local_maplet_support_index_npz", "max_support_views": 3})
        ),
    )
    index = SupportObservationGeometryIndex(
        image_ids=("s1.png", "s2.png"),
        image_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        source_row_indices=np.asarray([0, 1], dtype=np.int64),
        track_ids=np.asarray([11, 11], dtype=np.int64),
        xy=np.asarray([[40.0, 40.0], [40.0, 40.0]], dtype=np.float32),
        viewing_rays=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        reprojection_errors=np.asarray([0.1, 0.1], dtype=np.float32),
    )
    save_support_observation_geometry_index_npz(
        index,
        geometry,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    return layout, maplet, geometry


def _write_final_cache(path) -> None:
    ids = np.asarray(["q.png", "s1.png", "s2.png"])
    grid4 = _unit_grid(3, 4, axis=0)
    grid8 = _unit_grid(3, 8, axis=0)
    grid16 = _unit_grid(3, 16, axis=0)
    np.savez(
        path,
        image_ids=ids,
        image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
        summary_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        global_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        grid4_descriptors=grid4,
        grid8_descriptors=grid8,
        grid16_descriptors=grid16,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "radio_final_context_pca_v1",
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "pca_fit_scope": "mapping_train_images_only",
                    "spatial_grid_sizes": [4, 8, 16],
                    "radio_checkpoint_sha256": "radio-test",
                    "source_image_manifest_sha256": "image-manifest",
                    "image_source_contract": _IMAGE_SOURCE_CONTRACT,
                }
            )
        ),
    )


def _write_spatial_cache(path, *, format_name: str, grid_size: int, axis: int, metadata: dict) -> None:
    cache = SpatialImageContextCache(
        image_ids=np.asarray(["q.png", "s1.png", "s2.png"]),
        image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
        grids={grid_size: _unit_grid(3, grid_size, axis=axis)},
        metadata={
            "format": format_name,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "spatial_grid_sizes": [grid_size],
            "source_image_manifest_sha256": "image-manifest",
            "image_source_contract": _IMAGE_SOURCE_CONTRACT,
            **metadata,
        },
    )
    save_spatial_image_context_cache(cache, path)


def test_multisource_builder_keeps_all_views_and_cache_lineage(tmp_path) -> None:
    layout, maplet, geometry = _write_layout_and_geometry(tmp_path)
    final = tmp_path / "final.npz"
    intermediate = tmp_path / "intermediate.npz"
    alike = tmp_path / "alike.npz"
    output = tmp_path / "features.npz"
    summary = tmp_path / "summary.json"
    _write_final_cache(final)
    _write_spatial_cache(
        intermediate,
        format_name=INTERMEDIATE_PCA_FORMAT,
        grid_size=16,
        axis=1,
        metadata={
            "pca_fit_scope": "mapping_train_images_only",
            "radio_checkpoint_sha256": "radio-test",
            "intermediate_index": -6,
        },
    )
    _write_spatial_cache(
        alike,
        format_name=ALIKE_SPATIAL_FORMAT,
        grid_size=32,
        axis=0,
        metadata={"alike_checkpoint_sha256": "alike-test"},
    )

    build_multisource_landmark_region_prototype_candidate_probe_features(
        frozen_layout_features=layout,
        maplet_support_index=maplet,
        support_geometry_index=geometry,
        radio_final_context_cache=final,
        radio_intermediate_context_cache=intermediate,
        alike_spatial_context_cache=alike,
        output=output,
        summary_json=summary,
        devices=("cpu",),
        batch_size=2,
        force=False,
    )

    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        features = np.asarray(data["candidate_features"], dtype=np.float32)
        valid = np.asarray(data["candidate_view_valid"], dtype=bool)
        assert data["feature_names"].tolist() == list(
            MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES
        )
        assert features.shape == (1, 1, 3, len(MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES))
        assert valid.tolist() == [[[True, True, False]]]
        np.testing.assert_allclose(features[0, 0, :2, 0], 0.25)
        np.testing.assert_allclose(features[0, 0, :2, 1:], 1.0, atol=1e-3)
        assert np.isnan(features[0, 0, 2]).all()
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["source_image_manifest_sha256"] == "image-manifest"
        assert metadata["appearance_evidence_manifest"]["sources"][1]["intermediate_index"] == -6
        assert metadata["appearance_evidence_manifest"]["sources"][2]["alike_checkpoint_sha256"] == "alike-test"


def test_multisource_appearance_only_family_excludes_nonvisual_coverage() -> None:
    appearance = MULTISOURCE_LANDMARK_REGION_APPEARANCE_CONTEXT_FEATURE_NAMES
    assert len(appearance) == 60
    assert not any(name.endswith("_common_cell_fraction") for name in appearance)
    assert (
        ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES[
            "multisource_landmark_region_candidate_specific_appearance_only"
        ]
        == appearance
    )
    assert ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES[
        "multisource_landmark_region_with_anchor_appearance_only"
    ][0] == "radio_final_anchor_cosine"
