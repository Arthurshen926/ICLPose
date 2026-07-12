from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.landmark_topl_oracle import (
    TopLOracleObservation,
    summarize_topl_oracle,
)


def test_topl_oracle_reports_positive_pool_growth_and_pnp_eligibility() -> None:
    observations = []
    for query_id, ranks in (("q0", [1, 2, 3, 4, None]), ("q1", [10, 20, None, None, None])):
        for index, rank in enumerate(ranks):
            observations.append(
                TopLOracleObservation(
                    query_id=query_id,
                    track_id=100 * (1 + int(query_id[-1])) + index,
                    correct_rank=rank,
                    xy=np.asarray([10.0 * index, 5.0 * index], dtype=np.float64),
                    xyz=np.asarray([float(index), 0.0, 5.0], dtype=np.float64),
                )
            )

    summary = summarize_topl_oracle(observations, top_ls=(1, 5, 20))

    assert summary["top_l"]["1"]["correct_observation_recall"] == pytest.approx(0.1)
    assert summary["top_l"]["5"]["correct_observation_recall"] == pytest.approx(0.4)
    assert summary["top_l"]["5"]["query_with_at_least_4_positive_rate"] == pytest.approx(0.5)
    assert summary["top_l"]["20"]["mean_positive_proposal_count"] == pytest.approx(3.0)
