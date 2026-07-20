from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_anchor_aligned_global_layout_candidate_probe_features import (
    ARTIFACT_FORMAT,
    build_anchor_aligned_global_layout_candidate_probe_features,
)
from feature_extract.tools.vfm.fit_multiscale_candidate_probe import _load_features
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
)


def _unit_grid(count: int, size: int) -> np.ndarray:
    values = np.zeros((count, size * size, 2), dtype=np.float32)
    values[..., 0] = 1.0
    return values


def _write_inputs(tmp_path):
    layout = tmp_path / "layout.npz"
    geometry = tmp_path / "geometry.npz"
    final = tmp_path / "final.npz"
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
                    "support_view_selection": "fixed_test_support_views",
                }
            )
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
    np.savez(
        final,
        image_ids=ids,
        image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
        summary_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        global_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        grid4_descriptors=_unit_grid(3, 4),
        grid8_descriptors=_unit_grid(3, 8),
        grid16_descriptors=_unit_grid(3, 16),
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
    return layout, geometry, final


def test_anchor_aligned_global_layout_builder_keeps_per_view_frozen_contract(tmp_path) -> None:
    layout, geometry, final = _write_inputs(tmp_path)
    output = tmp_path / "features.npz"
    summary = tmp_path / "summary.json"

    build_anchor_aligned_global_layout_candidate_probe_features(
        frozen_layout_features=layout,
        support_geometry_index=geometry,
        radio_final_context_cache=final,
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
        assert data["feature_names"].tolist() == list(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES)
        assert features.shape == (1, 1, 2, len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES))
        assert valid.tolist() == [[[True, True]]]
        np.testing.assert_allclose(features[0, 0, :, 0], 0.25)
        assert np.isfinite(features[valid]).all()
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["full_image_spatial_layout_used"] is True
        assert metadata["whole_image_summary_or_global_used"] is False
        assert metadata["anchor_aligned_global_layout"]["grid_size"] == 16

    arrays, fit_metadata = _load_features(output)
    assert arrays["candidate_features"].shape == features.shape
    assert fit_metadata["format"] == ARTIFACT_FORMAT
