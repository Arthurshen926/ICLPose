from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.detector_grid_assignment_data import (
    greedy_unique_assignment_targets,
)


def test_greedy_targets_enforce_one_to_one_and_dustbin() -> None:
    residuals = np.asarray(
        [[0.5, 4.0, np.inf], [0.8, 1.5, np.inf], [8.0, 9.0, np.inf]],
        dtype=np.float32,
    )
    edge_mask = np.isfinite(residuals)
    targets = greedy_unique_assignment_targets(residuals, edge_mask, threshold_px=2.0)
    np.testing.assert_array_equal(targets, [0, 1, 3])


def test_greedy_targets_choose_globally_smallest_duplicate_track_residual() -> None:
    residuals = np.asarray([[1.0], [0.5]], dtype=np.float32)
    targets = greedy_unique_assignment_targets(
        residuals,
        np.ones_like(residuals, dtype=bool),
        threshold_px=2.0,
    )
    np.testing.assert_array_equal(targets, [1, 0])
