from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    concatenate_hypothesis_shard_field,
)
from feature_extract.tools.vfm.fit_audit_calibrated_relation_likelihood import (
    _promotion_gate,
    _relation_scores,
)
from feature_extract.vfm.localization.calibrated_relation_likelihood import (
    fit_relation_density_ratio,
    score_relation_histograms,
)
from feature_extract.vfm.localization.candidate_relation_features import (
    CandidateModeMixture,
    RELATION_CHANNELS,
    RelationResidualHistograms,
    build_query_knn_relation_graph,
)


def _all_topology_channels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros((*values.shape[:-2], len(RELATION_CHANNELS), values.shape[-1]))
    output[..., : values.shape[-2], :] = values
    return output


def test_query_relation_graph_is_deterministic_and_unique() -> None:
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    first = build_query_knn_relation_graph(xy, neighbor_k=1)
    second = build_query_knn_relation_graph(xy, neighbor_k=1)
    assert np.array_equal(first.edges, np.asarray([[0, 1], [1, 2]]))
    assert first.sha256 == second.sha256


def test_variable_relation_edges_are_padded_across_shards() -> None:
    first = np.ones((2, 3, 4), dtype=np.float64)
    second = np.full((1, 5, 4), 2.0, dtype=np.float64)
    merged = concatenate_hypothesis_shard_field(
        "verification_relation_feature_edge_histograms", [first, second]
    )
    assert merged.shape == (3, 5, 4)
    assert np.all(np.isnan(merged[:2, 3:]))
    assert np.all(merged[2] == 2.0)


def test_candidate_mode_mixture_requires_preserved_mass() -> None:
    xy = np.asarray([[[[10.0, 20.0], [11.0, 20.0]]]])
    mixture = CandidateModeMixture(
        xy,
        np.asarray([[[0.7, 0.3]]]),
        np.asarray([[[True, True]]]),
    )
    assert float(np.sum(mixture.probabilities)) == 1.0


def test_calibrated_relation_density_ratio_prefers_positive_residual_bins() -> None:
    bins = np.asarray([0.0, 1.0, 4.0, 16.0])
    calibration = fit_relation_density_ratio(
        _all_topology_channels(np.asarray([[20.0, 4.0, 1.0], [10.0, 2.0, 1.0]])),
        _all_topology_channels(np.asarray([[1.0, 4.0, 20.0], [1.0, 2.0, 10.0]])),
        bin_edges_px=bins,
        source_manifest={"train_query_hash": "frozen"},
    )
    good = RelationResidualHistograms(
        _all_topology_channels(np.asarray([[[0.8, 0.0, 0.0], [0.2, 0.0, 0.0]]])),
        np.asarray([1.0]),
        np.asarray([0.0]),
        bins,
        "edge",
    )
    bad = RelationResidualHistograms(
        _all_topology_channels(np.asarray([[[0.0, 0.0, 0.8], [0.0, 0.0, 0.2]]])),
        np.asarray([1.0]),
        np.asarray([0.0]),
        bins,
        "edge",
    )
    assert score_relation_histograms(good, calibration)[
        "relation_log_likelihood_ratio_mean"
    ] > score_relation_histograms(bad, calibration)[
        "relation_log_likelihood_ratio_mean"
    ]


def test_missing_relation_mass_is_neutral_not_renormalized() -> None:
    bins = np.asarray([0.0, 2.0, 8.0])
    calibration = fit_relation_density_ratio(
        _all_topology_channels(np.asarray([[8.0, 1.0], [8.0, 1.0]])),
        _all_topology_channels(np.asarray([[1.0, 8.0], [1.0, 8.0]])),
        bin_edges_px=bins,
        source_manifest={"train_query_hash": "frozen"},
    )
    features = RelationResidualHistograms(
        _all_topology_channels(np.asarray([[[0.1, 0.0], [0.0, 0.0]]])),
        np.asarray([0.1]),
        np.asarray([0.9]),
        bins,
        "edge",
    )
    score = score_relation_histograms(features, calibration)
    assert score["relation_log_likelihood_ratio_mean"] > 0.0
    assert score["relation_log_likelihood_ratio_mean"] < float(
        calibration.log_density_ratio[0, 0]
    )


def test_null_only_edges_remain_in_fixed_mean_denominator() -> None:
    bins = np.asarray([0.0, 2.0, 8.0])
    calibration = fit_relation_density_ratio(
        _all_topology_channels(np.asarray([[8.0, 1.0], [8.0, 1.0]])),
        _all_topology_channels(np.asarray([[1.0, 8.0], [1.0, 8.0]])),
        bin_edges_px=bins,
        source_manifest={"train_query_hash": "frozen"},
    )
    one_edge = RelationResidualHistograms(
        _all_topology_channels(np.asarray([[[1.0, 0.0], [0.0, 0.0]]])),
        np.asarray([1.0]),
        np.asarray([0.0]),
        bins,
        "edge",
    )
    with_null_edge = RelationResidualHistograms(
        _all_topology_channels(np.asarray(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        bins,
        "edge",
    )
    first = score_relation_histograms(one_edge, calibration)
    second = score_relation_histograms(with_null_edge, calibration)
    assert second["effective_edge_count"] == 1
    assert np.isclose(
        second["relation_log_likelihood_ratio_mean"],
        0.5 * first["relation_log_likelihood_ratio_mean"],
    )


def test_offline_per_edge_score_matches_runtime_factorization() -> None:
    bins = np.asarray([0.0, 2.0, 8.0])
    calibration = fit_relation_density_ratio(
        _all_topology_channels(np.asarray([[8.0, 1.0], [1.0, 8.0]])),
        _all_topology_channels(np.asarray([[1.0, 8.0], [8.0, 1.0]])),
        bin_edges_px=bins,
        source_manifest={"train_query_hash": "frozen"},
    )
    features = RelationResidualHistograms(
        _all_topology_channels(np.asarray(
            [
                [[0.7, 0.0], [0.0, 0.1]],
                [[0.0, 0.2], [0.1, 0.0]],
            ]
        )),
        np.asarray([0.8, 0.3]),
        np.asarray([0.2, 0.7]),
        bins,
        "edge",
    )
    arrays = {
        "verification_relation_feature_edge_histograms": features.histograms[
            None
        ],
        "verification_relation_feature_edge_null_touching_masses": (
            features.null_touching_mass[None]
        ),
        "verification_relation_feature_edge_counts": np.asarray([2]),
    }
    offline = _relation_scores(
        arrays, calibration, tuple(range(len(RELATION_CHANNELS)))
    )[0]
    runtime = score_relation_histograms(features, calibration)[
        "relation_log_likelihood_ratio_mean"
    ]
    assert np.isclose(offline, runtime, atol=1e-12)

    aggregate_then_log = np.log(
        np.sum(
            features.null_touching_mass
            + np.sum(
                features.histograms * np.exp(calibration.log_density_ratio)[None],
                axis=(1, 2),
            )
        )
    ) / 2.0
    assert not np.isclose(offline, aggregate_then_log)


def test_relation_promotion_requires_cross_block_rank_and_pose_improvement() -> None:
    unary = {
        "median_selected_true_rank": 50.0,
        "median_translation_m": 0.20,
        "p90_translation_m": 0.80,
        "median_rotation_deg": 0.5,
        "p90_rotation_deg": 1.5,
        "catastrophic_gt1m_count": 2,
    }
    improved = {
        **unary,
        "median_selected_true_rank": 20.0,
        "median_translation_m": 0.15,
    }
    payload = {
        split: {
            "unary": unary,
            "unary_plus_frozen_relation": improved,
            "paired_vs_unary": {"win_count": 4, "loss_count": 1},
        }
        for split in ("validation", "test")
    }
    assert _promotion_gate(payload)["passed"]
    payload["test"] = {
        "unary": unary,
        "unary_plus_frozen_relation": unary,
        "paired_vs_unary": {"win_count": 0, "loss_count": 0},
    }
    failed = _promotion_gate(payload)
    assert not failed["passed"]
    assert failed["deployment_relation_weight"] == 0.0
