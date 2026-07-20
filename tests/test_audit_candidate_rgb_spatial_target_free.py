from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_rgb_spatial_target_free import (
    _same_path,
    _support_view_oracle_gap,
)


def test_same_path_accepts_relative_and_absolute_spelling(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _same_path("targets.csv", tmp_path / "targets.csv")


def test_support_view_oracle_reports_frozen_mixture_gap() -> None:
    report = _support_view_oracle_gap(
        candidate_records={
            ("query.png", 3, 1): [0, 1],
            ("query.png", 7, 1): [2, 3],
        },
        target_log_likelihood=np.log(np.asarray([0.1, 0.8, 0.2, 0.3])),
        support_view_probabilities=np.asarray([0.9, 0.1, 0.5, 0.5]),
        support_view_ranks=np.asarray([0, 1, 0, 1]),
        correct_labels_by_threshold={
            "2px": np.asarray([True, True, False, False]),
        },
    )

    metrics = report["2px"]
    assert metrics["correct_candidate_count"] == 1
    assert metrics["evaluable_correct_candidate_count"] == 1
    assert metrics["frozen_top1_matches_oracle_best_rate_TARGET_ONLY"] == 0.0
    assert metrics["frozen_top2_contains_oracle_best_rate_TARGET_ONLY"] == 1.0
    assert np.isclose(
        metrics["oracle_over_all_view_log_likelihood_gain_TARGET_ONLY"]["mean"],
        np.log(0.8) - np.log(0.17),
    )


def test_support_view_oracle_rejects_inconsistent_candidate_labels() -> None:
    with pytest.raises(ValueError, match="disagree on target label"):
        _support_view_oracle_gap(
            candidate_records={("query.png", 3, 1): [0, 1]},
            target_log_likelihood=np.log(np.asarray([0.2, 0.8])),
            support_view_probabilities=np.asarray([0.5, 0.5]),
            support_view_ranks=np.asarray([0, 1]),
            correct_labels_by_threshold={"2px": np.asarray([True, False])},
        )
