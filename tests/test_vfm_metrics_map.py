import numpy as np
import pytest

from feature_extract.vfm.map_lifting import (
    TrackObservation,
    aggregate_selected_tracks,
    load_selected_track_bank_npz,
    mapability_summary,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.metrics import (
    basin_recall_at_k,
    hard_false_accept_rate,
    ndcg_at_k,
    oracle_gap,
    ranking_summary,
    spearman_rank,
)


def test_ranking_metrics_capture_pose_cost_ordering():
    scores = np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float32)
    costs = np.array([0.10, 0.90, 0.20, 0.40], dtype=np.float32)
    basin = np.array([True, False, True, False])

    summary = ranking_summary(scores=scores, costs_m=costs, basin_labels=basin)

    assert summary.pred_cost_m == pytest.approx(0.10)
    assert summary.oracle_cost_m == pytest.approx(0.10)
    assert summary.top1_acc == 1.0
    assert summary.basin_recall_at_2 == 1.0
    assert oracle_gap(scores=scores, costs_m=costs) == 0.0
    assert spearman_rank(scores, -costs) > 0.7
    assert ndcg_at_k(scores=scores, relevance=-costs, k=3) > 0.8
    assert basin_recall_at_k(scores=scores, basin_labels=basin, k=1) == 1.0


def test_hard_false_accept_rate_uses_risk_threshold():
    scores = np.array([0.95, 0.60, 0.40, 0.20], dtype=np.float32)
    basin = np.array([False, True, False, True])

    assert hard_false_accept_rate(scores=scores, basin_labels=basin, accept_threshold=0.5) == 0.5


def test_selected_track_aggregation_filters_visibility_and_geometry():
    observations = [
        TrackObservation(
            track_id=7,
            image_id="a",
            feature=np.array([1.0, 2.0], dtype=np.float32),
            visible=True,
            geometry_valid=True,
        ),
        TrackObservation(
            track_id=7,
            image_id="b",
            feature=np.array([3.0, 4.0], dtype=np.float32),
            visible=True,
            geometry_valid=True,
        ),
        TrackObservation(
            track_id=7,
            image_id="c",
            feature=np.array([100.0, 100.0], dtype=np.float32),
            visible=False,
            geometry_valid=True,
        ),
        TrackObservation(
            track_id=8,
            image_id="d",
            feature=np.array([5.0, 5.0], dtype=np.float32),
            visible=True,
            geometry_valid=False,
        ),
    ]

    bank = aggregate_selected_tracks(observations, min_observations=2)

    assert set(bank.tracks) == {7}
    np.testing.assert_allclose(bank.tracks[7].mean_feature, np.array([2.0, 3.0], dtype=np.float32))
    assert bank.tracks[7].observation_count == 2
    assert bank.tracks[7].mean_variance > 0.0


def test_selected_track_bank_round_trip_and_summary(tmp_path):
    observations = [
        TrackObservation(1, "a", np.array([1.0, 0.0], dtype=np.float32), True, True, utility=0.8),
        TrackObservation(1, "b", np.array([0.8, 0.2], dtype=np.float32), True, True, utility=0.6),
        TrackObservation(2, "a", np.array([0.0, 1.0], dtype=np.float32), True, True, utility=0.4),
        TrackObservation(2, "b", np.array([0.1, 0.9], dtype=np.float32), True, True, utility=0.5),
    ]
    bank = aggregate_selected_tracks(observations, min_observations=2)
    path = tmp_path / "selected_tracks.npz"

    save_selected_track_bank_npz(bank, path)
    loaded = load_selected_track_bank_npz(path)
    summary = mapability_summary(loaded)

    assert set(loaded.tracks) == {1, 2}
    assert loaded.feature_dim == 2
    assert summary.track_count == 2
    assert summary.mean_observation_count == pytest.approx(2.0)
    assert summary.mean_track_variance > 0.0
