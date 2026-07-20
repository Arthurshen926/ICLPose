from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    ARTIFACT_FORMAT,
    build_landmark_region_prototype_candidate_probe_features,
    parse_args,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
)


def _layout_metadata(*, radio_checkpoint_sha256: str = "test-radio-checkpoint") -> str:
    return json.dumps(
        {
            "format": "multiscale_candidate_probe_features_v1",
            "contains_ground_truth": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": False,
            "proposals_sha256": "test-proposals-sha",
            "radio_checkpoint_sha256": radio_checkpoint_sha256,
        }
    )


def _radio_cache(
    path,
    *,
    include_grid16: bool = False,
    radio_checkpoint_sha256: str = "test-radio-checkpoint",
) -> None:
    image_ids = np.asarray(["q.png", "s1.png", "s2.png"])
    grid8 = np.zeros((3, 8, 8, 2), dtype=np.float32)
    grid8[..., 0] = 1.0
    values = {
        "image_ids": image_ids,
        "image_sizes": np.asarray([[80, 80], [80, 80], [80, 80]], dtype=np.int64),
        "summary_descriptors": np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        "global_descriptors": np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        "grid4_descriptors": np.tile(
            np.asarray([[[1.0, 0.0]]], dtype=np.float32), (3, 16, 1)
        ),
        "grid8_descriptors": grid8.reshape(3, 64, 2),
    }
    spatial_grid_sizes = [4, 8]
    if include_grid16:
        grid16 = np.zeros((3, 16, 16, 2), dtype=np.float32)
        grid16[..., 1] = 1.0
        values["grid16_descriptors"] = grid16.reshape(3, 256, 2)
        spatial_grid_sizes.append(16)
    values["metadata_json"] = np.asarray(
        json.dumps(
            {
                "format": "radio_final_context_pca_v1",
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "pca_fit_scope": "mapping_train_images_only",
                "spatial_grid_sizes": spatial_grid_sizes,
                "radio_checkpoint_sha256": radio_checkpoint_sha256,
            }
        )
    )
    np.savez(path, **values)


def _region_fixture(tmp_path):
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
        metadata_json=np.asarray(_layout_metadata()),
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


def test_cli_parser_accepts_region_prototype_profiles() -> None:
    arguments = [
        "--frozen_layout_features",
        "layout.npz",
        "--maplet_support_index",
        "maplet.npz",
        "--support_geometry_index",
        "geometry.npz",
        "--radio_final_context_cache",
        "radio.npz",
        "--output",
        "output.npz",
        "--summary_json",
        "summary.json",
    ]
    default = parse_args(arguments)
    highres = parse_args([*arguments, "--profile", "grid16_multiscale7_11"])

    assert default.devices == "cuda:0,cuda:1"
    assert default.batch_size == 16384
    assert default.profile == "grid8_window7"
    assert highres.profile == "grid16_multiscale7_11"


def test_builder_uses_fixed_support8_observation_crops_without_view_averaging(tmp_path) -> None:
    layout, maplet, geometry = _region_fixture(tmp_path)
    radio = tmp_path / "radio.npz"
    output = tmp_path / "region_features.npz"
    summary = tmp_path / "summary.json"
    _radio_cache(radio)

    result = build_landmark_region_prototype_candidate_probe_features(
        frozen_layout_features=layout,
        maplet_support_index=maplet,
        support_geometry_index=geometry,
        radio_final_context_cache=radio,
        output=output,
        summary_json=summary,
        devices=("cpu",),
        batch_size=2,
        force=False,
    )

    assert result["support_view_count"] == 3
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        features = np.asarray(data["candidate_features"], dtype=np.float32)
        valid = np.asarray(data["candidate_view_valid"], dtype=bool)
        assert data["feature_names"].tolist() == list(LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES)
        assert features.shape == (1, 1, 3, len(LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES))
        assert valid.tolist() == [[[True, True, False]]]
        np.testing.assert_allclose(features[0, 0, :2, 0], 0.25)
        np.testing.assert_allclose(features[0, 0, :2, 1], 1.0, atol=1e-3)
        assert np.isnan(features[0, 0, 2]).all()
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["whole_image_summary_or_global_used"] is False
        assert (
            metadata["support_view_marginalization"]
            == "per_view_features_preserved_for_later_log_mixture_v1"
        )
        assert metadata["region_prototype"]["support_coordinate_source"] == "sfm_observation_xy"


def test_builder_exports_predeclared_grid16_multiscale_profile(tmp_path) -> None:
    layout, maplet, geometry = _region_fixture(tmp_path)
    radio = tmp_path / "radio_grid16.npz"
    output = tmp_path / "grid16_features.npz"
    summary = tmp_path / "grid16_summary.json"
    _radio_cache(radio, include_grid16=True)

    build_landmark_region_prototype_candidate_probe_features(
        frozen_layout_features=layout,
        maplet_support_index=maplet,
        support_geometry_index=geometry,
        radio_final_context_cache=radio,
        output=output,
        summary_json=summary,
        devices=("cpu",),
        batch_size=2,
        force=False,
        profile_name="grid16_multiscale7_11",
    )

    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        features = np.asarray(data["candidate_features"], dtype=np.float32)
        assert data["feature_names"].tolist() == list(
            HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES
        )
        assert features.shape == (1, 1, 3, len(HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES))
        np.testing.assert_allclose(features[0, 0, :2, 0], 0.25)
        np.testing.assert_allclose(features[0, 0, :2, 1], 1.0, atol=1e-3)
        np.testing.assert_allclose(features[0, 0, :2, 12], 1.0, atol=1e-3)
        assert metadata["format"] == HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT
        assert metadata["region_prototype"]["profile"] == "grid16_multiscale7_11"
        assert metadata["region_prototype"]["grid_size"] == 16
        assert metadata["region_prototype"]["window_sizes"] == [7, 11]


def test_builder_refuses_to_fallback_from_grid16_to_grid8(tmp_path) -> None:
    layout, maplet, geometry = _region_fixture(tmp_path)
    radio = tmp_path / "radio_grid8.npz"
    _radio_cache(radio)

    with pytest.raises(ValueError, match="grid16 cache"):
        build_landmark_region_prototype_candidate_probe_features(
            frozen_layout_features=layout,
            maplet_support_index=maplet,
            support_geometry_index=geometry,
            radio_final_context_cache=radio,
            output=tmp_path / "grid16_features.npz",
            summary_json=tmp_path / "grid16_summary.json",
            devices=("cpu",),
            batch_size=2,
            force=False,
            profile_name="grid16_multiscale7_11",
        )


def test_builder_rejects_a_radio_checkpoint_mismatch(tmp_path) -> None:
    layout, maplet, geometry = _region_fixture(tmp_path)
    radio = tmp_path / "mismatched_radio.npz"
    _radio_cache(radio, radio_checkpoint_sha256="other-radio-checkpoint")

    with pytest.raises(ValueError, match="different RADIO checkpoints"):
        build_landmark_region_prototype_candidate_probe_features(
            frozen_layout_features=layout,
            maplet_support_index=maplet,
            support_geometry_index=geometry,
            radio_final_context_cache=radio,
            output=tmp_path / "features.npz",
            summary_json=tmp_path / "summary.json",
            devices=("cpu",),
            batch_size=2,
            force=False,
        )
