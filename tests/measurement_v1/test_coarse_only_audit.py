from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.measurement_v1.coarse_only_audit import (
    apply_coarse_feature_control,
    coarse_match_index_summary,
    parse_cell_shift,
    same_cell_identity_keypoint_matches,
)


def _toy_feature(channels: int = 3, height: int = 2, width: int = 3) -> np.ndarray:
    values = np.arange(channels * height * width, dtype=np.float32)
    return values.reshape(channels, height, width)


def test_parse_cell_shift_accepts_negative_csv() -> None:
    assert parse_cell_shift("-2,4") == (-2, 4)
    with pytest.raises(ValueError):
        parse_cell_shift("1,2,3")


def test_same_cell_identity_keypoint_matches_use_grid_cell_centers() -> None:
    query = _toy_feature(height=2, width=3)
    render = _toy_feature(height=2, width=3)

    matches = same_cell_identity_keypoint_matches(
        query,
        render,
        query_image_width=30,
        query_image_height=20,
        render_image_width=30,
        render_image_height=20,
        max_matches=4,
    )

    assert [(m.query_index, m.render_index) for m in matches] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert np.allclose(matches[0].query_xy, [5.0, 5.0])
    assert np.allclose(matches[1].render_xy, [15.0, 5.0])
    assert all(m.dual_softmax_confidence == 1.0 for m in matches)


def test_apply_coarse_feature_control_constant_keeps_nonzero_cells() -> None:
    query, render = apply_coarse_feature_control(_toy_feature(), _toy_feature(), mode="constant")

    qnorm = np.linalg.norm(query.reshape(query.shape[0], -1).T, axis=1)
    rnorm = np.linalg.norm(render.reshape(render.shape[0], -1).T, axis=1)
    assert np.allclose(qnorm, 1.0)
    assert np.allclose(rnorm, 1.0)


def test_apply_coarse_feature_control_spatial_permutation_changes_query_only() -> None:
    query = _toy_feature()
    render = _toy_feature()

    q_perm, r_perm = apply_coarse_feature_control(query, render, mode="spatial_permute_query", seed=7)

    assert not np.allclose(q_perm, query)
    assert np.allclose(r_perm, render)
    assert sorted(q_perm.reshape(query.shape[0], -1)[0].tolist()) == sorted(query.reshape(query.shape[0], -1)[0].tolist())


def test_apply_coarse_feature_control_shift_fills_exposed_cells_with_zero() -> None:
    query = np.ones((1, 2, 3), dtype=np.float32)
    shifted, _render = apply_coarse_feature_control(
        query,
        query,
        mode="shift_query_cells",
        shift_cells=(1, 0),
    )

    assert shifted.shape == query.shape
    assert np.allclose(shifted[:, :, 0], 0.0)
    assert np.allclose(shifted[:, :, 1:], 1.0)


def test_coarse_match_index_summary_reports_same_cell_and_displacement() -> None:
    query = _toy_feature(height=2, width=3)
    render = _toy_feature(height=2, width=3)
    matches = same_cell_identity_keypoint_matches(
        query,
        render,
        query_image_width=30,
        query_image_height=20,
        render_image_width=30,
        render_image_height=20,
    )

    summary = coarse_match_index_summary(matches, query_grid_width=3, render_grid_width=3)

    assert summary["coarse_match_count"] == 6
    assert summary["same_cell_fraction"] == pytest.approx(1.0)
    assert summary["median_cell_delta_x"] == pytest.approx(0.0)
    assert summary["median_cell_delta_y"] == pytest.approx(0.0)
