from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
    fixed_topl_coarse_posterior,
    load_mixed_verification_points,
    select_detector_rows_spatial_quota,
    select_disjoint_lattice_points,
)


def test_fixed_topl_coarse_posterior_preserves_explicit_null_mass() -> None:
    probabilities, null = fixed_topl_coarse_posterior(
        np.asarray([[0.8, 0.4, -np.inf], [0.2, -np.inf, -np.inf]], dtype=np.float32),
        np.asarray([[True, True, False], [True, False, False]]),
        temperature=0.1,
        null_probability=0.2,
    )
    np.testing.assert_allclose(probabilities.sum(axis=1) + null, [1.0, 1.0], atol=1e-6)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[0, 2] == 0.0
    assert probabilities[1, 0] == pytest.approx(0.8)


def test_detector_spatial_quota_excludes_fit_rows_and_cycles_cells() -> None:
    rows = np.arange(12, dtype=np.int64)
    xy = np.asarray(
        [
            [5.0, 5.0], [15.0, 5.0], [25.0, 5.0], [35.0, 5.0],
            [5.0, 15.0], [15.0, 15.0], [25.0, 15.0], [35.0, 15.0],
            [5.0, 25.0], [15.0, 25.0], [25.0, 25.0], [35.0, 25.0],
        ],
        dtype=np.float32,
    )
    chosen = select_detector_rows_spatial_quota(
        source_rows=rows,
        xy=xy,
        scores=np.linspace(1.0, 0.0, len(rows)),
        excluded_rows=np.asarray([0, 5], dtype=np.int64),
        point_count=6,
        image_width=40,
        image_height=30,
        grid_rows=3,
        grid_columns=4,
    )
    assert len(chosen) == 6
    assert len(np.unique(chosen)) == 6
    assert not np.any(np.isin(chosen, [0, 5]))


def test_lattice_uses_alternate_phase_before_disjointness_fallback() -> None:
    points, diagnostics = select_disjoint_lattice_points(
        image_width=100,
        image_height=100,
        grid_rows=1,
        grid_columns=1,
        primary_phase=(0.5, 0.5),
        alternate_phases=((0.25, 0.25),),
        excluded_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        minimum_distance_px=8.0,
    )
    np.testing.assert_allclose(points, [[25.0, 25.0]])
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["minimum_fit_distance_px"] > 8.0


def _write_artifact(path, *, null_probability: float = 0.2) -> None:
    metadata = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    probability = np.asarray([[0.5, 0.3], [0.4, 0.4]], dtype=np.float32)
    probability[0] *= (1.0 - float(null_probability)) / 0.8
    np.savez_compressed(
        path,
        source_point_ids=np.asarray([10, 11], dtype=np.int64),
        query_ids=np.asarray(["q.png", "q.png"]),
        split_names=np.asarray(["validation", "validation"]),
        xy=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        point_sources=np.asarray([POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_INTERMEDIATE]),
        source_detector_rows=np.asarray([7, -1], dtype=np.int64),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        candidate_bank_rows=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        candidate_track_ids=np.asarray([[100, 101], [102, 103]], dtype=np.int64),
        candidate_prototype_ids=np.asarray([[0, 0], [0, 0]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        candidate_prior_probabilities=probability,
        null_probabilities=np.asarray([null_probability, 0.2], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_mixed_verification_loader_enforces_candidate_mass(tmp_path) -> None:
    path = tmp_path / "points.npz"
    _write_artifact(path)
    points = load_mixed_verification_points(path)
    assert points.rows_for_query("q.png").tolist() == [0, 1]
    assert points.point_sources.tolist() == [POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_INTERMEDIATE]

    invalid = tmp_path / "invalid.npz"
    _write_artifact(invalid)
    with np.load(invalid, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays["null_probabilities"] = np.asarray([0.1, 0.2], dtype=np.float32)
    np.savez_compressed(invalid, **arrays)
    with pytest.raises(ValueError, match="mass is not conserved"):
        load_mixed_verification_points(invalid)


def test_mixed_verification_loader_rejects_unknown_source(tmp_path) -> None:
    path = tmp_path / "points.npz"
    _write_artifact(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    arrays["point_sources"] = np.asarray([POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_FINAL + "_bad"])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="arrays are invalid"):
        load_mixed_verification_points(path)
