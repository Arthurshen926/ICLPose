import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_maplet_prior_overlay import (
    _build_overlay_arrays,
)


def test_overlay_remaps_compact_probabilities_without_losing_null_mass() -> None:
    arrays, audit = _build_overlay_arrays(
        proposal_track_ids=np.asarray([[10, 11, 12], [20, 21, 22]]),
        selected_rows=np.asarray([0, 1]),
        selected_columns=np.asarray([[2, 0], [1, 2]]),
        valid_edges=np.ones((2, 2), dtype=bool),
        compact_candidate_probabilities=np.asarray(
            [[0.2, 0.3], [0.4, 0.1]], dtype=np.float32
        ),
        compact_dustbin_probabilities=np.asarray(
            [[0.5, 0.5], [0.5, 0.5]], dtype=np.float32
        ),
    )

    np.testing.assert_allclose(
        arrays["candidate_probabilities"],
        np.asarray([[0.3, 0.0, 0.2], [0.0, 0.4, 0.1]], dtype=np.float32),
    )
    np.testing.assert_allclose(arrays["null_probabilities"], [0.5, 0.5])
    assert audit["maximum_output_mass_error"] < 1e-6


def test_overlay_rejects_partial_rows_or_nonconstant_dustbin() -> None:
    kwargs = {
        "proposal_track_ids": np.asarray([[10, 11], [20, 21]]),
        "selected_rows": np.asarray([0]),
        "selected_columns": np.asarray([[0, 1]]),
        "valid_edges": np.ones((1, 2), dtype=bool),
        "compact_candidate_probabilities": np.asarray([[0.2, 0.3]]),
        "compact_dustbin_probabilities": np.asarray([[0.5, 0.5]]),
    }
    with pytest.raises(ValueError, match="full proposal-row coverage"):
        _build_overlay_arrays(**kwargs)

    kwargs["selected_rows"] = np.asarray([0, 1])
    kwargs["selected_columns"] = np.asarray([[0, 1], [0, 1]])
    kwargs["valid_edges"] = np.ones((2, 2), dtype=bool)
    kwargs["compact_candidate_probabilities"] = np.asarray(
        [[0.2, 0.3], [0.2, 0.3]]
    )
    kwargs["compact_dustbin_probabilities"] = np.asarray(
        [[0.5, 0.4], [0.5, 0.5]]
    )
    with pytest.raises(ValueError, match="constant across candidate columns"):
        _build_overlay_arrays(**kwargs)
