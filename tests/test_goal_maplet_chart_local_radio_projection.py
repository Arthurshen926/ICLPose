from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.train_goal_maplet_chart_local_radio_projection import (
    _load_observation_bank,
    _pair_inventory,
    _sample_triplets,
)
from feature_extract.vfm.localization_goal_maplet.chart_local_radio_projection import (
    load_chart_local_radio_projection,
    project_chart_local_radio,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256


def test_projection_normalizes_and_preserves_rows() -> None:
    feature = np.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], np.float32)
    weight = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], np.float32)
    output = project_chart_local_radio(feature, weight)
    assert output.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0)


def test_pair_inventory_requires_two_distinct_mapping_observations() -> None:
    identity = np.asarray([0, 0, 1, 1, 1, 2, 2])
    observation = np.asarray([0, 0, 0, 1, 1, 2, 3])
    route = np.asarray(["seq1", "seq1", "seq9", "seq9"])
    fit = _pair_inventory(identity, observation, route, {"seq1"})
    np.testing.assert_array_equal(fit["eligible_identity"], [1])
    validation = _pair_inventory(identity, observation, route, {"seq9"})
    np.testing.assert_array_equal(validation["eligible_identity"], [2])


def test_triplet_sampler_never_falls_back_to_a_different_plane() -> None:
    inventory = {
        "token_order": np.arange(6),
        "pair_start": np.arange(6),
        "pair_end": np.arange(1, 7),
        "pair_identity": np.asarray([0, 0, 1, 1, 2, 2]),
        "pair_observation": np.arange(6),
        "identity_start": np.asarray([0, 2, 4]),
        "identity_end": np.asarray([2, 4, 6]),
        "eligible_identity": np.asarray([0, 1, 2]),
    }
    # Identity 2 is the only eligible cell on plane 1 and must never be sampled.
    identity_plane = np.asarray([0, 0, 1])
    anchor, positive, negative = _sample_triplets(
        inventory, identity_plane, np.random.default_rng(7), batch_size=2,
    )
    assert set(anchor.tolist()) <= {0, 1, 2, 3}
    assert set(positive.tolist()) <= {0, 1, 2, 3}
    assert set(negative.tolist()) <= {0, 1, 2, 3}


def test_projection_loader_rejects_metadata_tampering(tmp_path) -> None:
    arrays = {"weight": np.eye(2, 3, dtype=np.float32)}
    metadata = {
        "artifact_type": "goal_maplet_chart_local_radio_projection_v1",
        "input_dimension": 3,
        "output_dimension": 2,
        "query_pose_depth_or_ground_truth_read": False,
        "mapping_rgb_stored": False,
        "source_view_identity_retained_at_runtime": False,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    path = tmp_path / "projection.npz"
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    load_chart_local_radio_projection(path)
    metadata["mapping_rgb_stored"] = True
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    with pytest.raises(ValueError, match="contract differs"):
        load_chart_local_radio_projection(path)


def test_observation_bank_loader_rejects_array_tampering(tmp_path) -> None:
    arrays = {
        "observation_offsets": np.asarray([0, 1], np.int64),
        "token_ids": np.asarray([0], np.int16),
        "world_points": np.ones((1, 3), np.float64),
        "radio_features": np.ones((1, 4), np.float16),
        "world_point_covariance_m2": np.zeros((1, 3, 3), np.float32),
        "plane_pixel_purity": np.ones(1, np.float32),
        "plane_depth_dispersion_m": np.zeros(1, np.float32),
    }
    metadata = {
        "artifact_type": "goal_maplet_plane_pnp_observation_bank_v2",
        "plane_specific_geometry": True,
        "uses_query_pose_or_ground_truth": False,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    path = tmp_path / "bank.npz"
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    _load_observation_bank(path)
    arrays["world_points"][0, 0] = 2.0
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    with pytest.raises(ValueError, match="lineage differs"):
        _load_observation_bank(path)
