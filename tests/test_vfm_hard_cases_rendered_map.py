import numpy as np

from feature_extract.vfm.hard_cases import HardCaseCandidate, build_hard_case_splits
from feature_extract.vfm.map_lifting import TrackObservation, aggregate_selected_tracks
from feature_extract.vfm.rendered_map_scoring import (
    render_selected_track_bank,
    score_rendered_selected_features,
)
from feature_extract.vfm.verifier import LinearEvidenceVerifier


def test_hard_case_split_builder_identifies_required_subsets():
    rows = [
        HardCaseCandidate(
            query_id="q0",
            candidate_id="bad_top1",
            cost_m=0.6,
            basin_label=False,
            retrieval_rank=1,
            verifier_score=0.7,
            pnp_score=40.0,
            identity_delta_m=0.02,
        ),
        HardCaseCandidate(
            query_id="q0",
            candidate_id="good_top2",
            cost_m=0.1,
            basin_label=True,
            retrieval_rank=2,
            verifier_score=0.6,
            pnp_score=20.0,
            identity_delta_m=0.3,
        ),
        HardCaseCandidate(
            query_id="q1",
            candidate_id="good_top1",
            cost_m=0.1,
            basin_label=True,
            retrieval_rank=1,
            verifier_score=0.9,
            pnp_score=60.0,
            identity_delta_m=0.01,
        ),
    ]

    splits = build_hard_case_splits(
        rows,
        accept_threshold=0.5,
        pnp_score_threshold=30.0,
        near_identity_threshold_m=0.05,
    )

    assert splits.retrieval_top1_wrong == ("q0",)
    assert splits.near_identity_false_positive == ("q0",)
    assert splits.pnp_high_score_wrong == ("q0",)


def test_rendered_selected_map_scoring_prefers_consistent_features():
    observations = [
        TrackObservation(0, "i0", np.array([1.0, 0.0]), True, True),
        TrackObservation(0, "i1", np.array([1.0, 0.0]), True, True),
        TrackObservation(1, "i0", np.array([0.0, 1.0]), True, True),
        TrackObservation(1, "i1", np.array([0.0, 1.0]), True, True),
    ]
    bank = aggregate_selected_tracks(observations, min_observations=2)
    rendered = render_selected_track_bank(bank, candidate_id="c0", visible_track_ids=[0, 1])
    verifier = LinearEvidenceVerifier(prior_weight=0.0, accept_threshold=0.6)

    matching_query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mismatched_query = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    good = score_rendered_selected_features(
        query_features=matching_query,
        rendered=rendered,
        query_uncertainty=np.zeros(2, dtype=np.float32),
        verifier=verifier,
    )
    bad = score_rendered_selected_features(
        query_features=mismatched_query,
        rendered=rendered,
        query_uncertainty=np.zeros(2, dtype=np.float32),
        verifier=verifier,
    )

    assert good.score > bad.score
    assert good.accepted
    assert not bad.accepted
