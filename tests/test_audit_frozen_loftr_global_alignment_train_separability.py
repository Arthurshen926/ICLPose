from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_loftr_global_alignment_train_separability import (
    fixed_view_weighted_negative_log1p_error,
)


def test_fixed_view_weighted_raw_error_preserves_view_mass_and_unknowns() -> None:
    score, mass = fixed_view_weighted_negative_log1p_error(
        values=np.asarray([[[1.0, 3.0], [2.0, np.nan]]], dtype=np.float32),
        usable=np.asarray([[[True, True], [True, False]]]),
        weights=np.asarray([[[0.25, 0.75], [0.4, 0.6]]], dtype=np.float32),
    )
    assert mass[0, 0] == pytest.approx(1.0)
    assert mass[0, 1] == pytest.approx(0.4)
    assert score[0, 0] == pytest.approx(-(0.25 * np.log(2.0) + 0.75 * np.log(4.0)))
    assert score[0, 1] == pytest.approx(-np.log(3.0))


def test_fixed_view_weighted_raw_error_rejects_nonfinite_available_values() -> None:
    with pytest.raises(ValueError, match="invalid"):
        fixed_view_weighted_negative_log1p_error(
            values=np.asarray([[[np.nan]]], dtype=np.float32),
            usable=np.asarray([[[True]]]),
            weights=np.asarray([[[1.0]]], dtype=np.float32),
        )
