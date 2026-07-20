from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_alike_image_spatial_context_cache import (
    ARTIFACT_FORMAT as ALIKE_SPATIAL_FORMAT,
)
from feature_extract.tools.vfm.build_dense_alike_local_mode_candidate_probe_features import (
    ARTIFACT_FORMAT,
    build_dense_alike_local_mode_candidate_probe_features,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)


def _unit_grid(count: int, size: int, *, axis: int) -> np.ndarray:
    values = np.zeros((count, size * size, 2), dtype=np.float32)
    values[..., axis] = 1.0
    return values


def _write_inputs(tmp_path):
    layout = tmp_path / "layout.npz"
    maplet = tmp_path / "maplet.npz"
    geometry = tmp_path / "geometry.npz"
    final = tmp_path / "final.npz"
    alike = tmp_path / "alike.npz"
    ids = np.asarray(["q.png", "s1.png", "s2.png"])
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
                    "radio_checkpoint_sha256": "radio-test",
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
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("s1.png", "s2.png"),
            image_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            source_row_indices=np.asarray([0, 1], dtype=np.int64),
            track_ids=np.asarray([11, 11], dtype=np.int64),
            xy=np.asarray([[40.0, 40.0], [40.0, 40.0]], dtype=np.float32),
            viewing_rays=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            reprojection_errors=np.asarray([0.1, 0.1], dtype=np.float32),
        ),
        geometry,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    grid4 = _unit_grid(3, 4, axis=0)
    grid8 = _unit_grid(3, 8, axis=0)
    grid16 = _unit_grid(3, 16, axis=0)
    np.savez(
        final,
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
                }
            )
        ),
    )
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=ids,
            image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
            grids={64: _unit_grid(3, 64, axis=1)},
            metadata={
                "format": ALIKE_SPATIAL_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "render": False,
                "spatial_grid_sizes": [64],
                "source_image_manifest_sha256": "image-manifest",
                "alike_checkpoint_sha256": "alike-test",
            },
        ),
        alike,
    )
    return layout, maplet, geometry, final, alike


def test_dense_alike_local_mode_builder_keeps_fixed_view_layout(tmp_path) -> None:
    layout, maplet, geometry, final, alike = _write_inputs(tmp_path)
    output = tmp_path / "features.npz"
    summary = tmp_path / "summary.json"

    build_dense_alike_local_mode_candidate_probe_features(
        frozen_layout_features=layout,
        maplet_support_index=maplet,
        support_geometry_index=geometry,
        radio_final_context_cache=final,
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
        assert data["feature_names"].tolist() == list(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES)
        assert features.shape == (1, 1, 3, len(DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES))
        assert valid.tolist() == [[[True, True, False]]]
        np.testing.assert_allclose(features[0, 0, :2, 0], 0.25)
        assert np.isfinite(features[0, 0, :2]).all()
        assert np.isnan(features[0, 0, 2]).all()
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["dense_local_modes"]["grid_size"] == 64
        assert metadata["dense_local_modes"]["explicit_crop_coverage_feature_used"] is False
