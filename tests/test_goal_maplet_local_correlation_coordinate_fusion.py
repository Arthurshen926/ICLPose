import numpy as np
import pytest

from feature_extract.tools.vfm.fuse_goal_maplet_local_correlation_coordinates import (
    _fusion_metadata,
    _paired_contract,
    _pareto_local_coordinate_mask,
)


def _coordinate(image, uv, match):
    count = len(image)
    covariance = np.zeros((count, 3, 3), np.float32)
    covariance[:, 0, 0] = np.asarray(uv, np.float32)
    covariance[:, 1, 1] = np.asarray(uv, np.float32)
    return {
        "query_measurement_variance_px2": np.asarray(image, np.float32),
        "prototype_centroid_covariance_world_m2": covariance,
        "correspondence_match_probability": np.asarray(match, np.float32),
    }


def test_pareto_coordinate_mask_requires_all_three_signals():
    baseline = _coordinate([2, 2, 2], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5])
    local = _coordinate([1, 1, 1], [0.1, 0.3, 0.1], [0.6, 0.6, 0.4])
    np.testing.assert_array_equal(
        _pareto_local_coordinate_mask(baseline, local), [True, False, False],
    )


def test_coordinate_mask_rejects_nonfinite_or_nonpositive_variance():
    baseline = _coordinate([2], [0.2], [0.5])
    local = _coordinate([0], [0.1], [0.6])
    with pytest.raises(ValueError, match="uncertainty inputs"):
        _pareto_local_coordinate_mask(baseline, local)


def test_paired_contract_rejects_candidate_inventory_drift():
    common = {
        "names": np.asarray(["q"]),
        "correspondence_offsets": np.asarray([0, 1]),
        "query_tokens": np.asarray([4]),
        "provenance_region_plane_atlas_row": np.asarray([[0, 1, 2]]),
        "camera_matrices": np.eye(3)[None],
        "radial_k1": np.asarray([0.0]),
        "prototype_atlas_row": np.asarray([2]),
        "query_plane_visible_fraction": np.asarray([1.0]),
        "radio_match_score": np.asarray([0.5]),
        "prototype_world_covariance_m2": np.zeros((1, 3, 3)),
        "prototype_plane_pixel_purity": np.ones(1),
        "prototype_plane_depth_dispersion_m": np.zeros(1),
        "prototype_chart_uv_cell_lower_m": np.zeros((1, 2)),
        "world_points": np.zeros((1, 3)),
        "query_measurements_xy": np.zeros((1, 2)),
        "query_measurement_variance_px2": np.ones(1),
        "correspondence_match_probability": np.ones(1),
        "prototype_centroid_covariance_world_m2": np.eye(3)[None],
        "prototype_chart_uv_measurement_m": np.zeros((1, 2)),
    }
    meta = {
        "artifact_type": "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
    }
    local = {key: value.copy() for key, value in common.items()}
    local["query_tokens"][0] = 5
    with pytest.raises(ValueError, match="query_tokens"):
        _paired_contract(common, meta, local, meta)


def test_fusion_metadata_preserves_both_heads_without_single_head_claim():
    baseline = {"content_sha256": "old", "mapping_subtoken_head_content_sha256": "v11",
                "mapping_chart_uv_measurement_variance_scale": 2.0, "topk_planes": 3}
    local = {"mapping_subtoken_head_content_sha256": "v12",
             "mapping_chart_uv_measurement_variance_scale": 3.0}
    metadata = _fusion_metadata(baseline, local)
    assert "content_sha256" not in metadata
    assert "mapping_subtoken_head_content_sha256" not in metadata
    assert "mapping_chart_uv_measurement_variance_scale" not in metadata
    assert metadata["coordinate_fusion_source_head_metadata"]["1_v12"] == local
    assert metadata["coordinate_fusion_source_head_metadata"]["0_v11"][
        "mapping_subtoken_head_content_sha256"] == "v11"
    assert metadata["topk_planes"] == 3
    assert metadata["coordinate_fusion_match_probability_calibrated"] is False
