import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_acquisition_extrapolation_strata import (
    _bin,
    _forward_world,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_extrapolation_strata import (
    _summary,
)


def test_acquisition_strata_use_fixed_boundary_semantics():
    assert _bin(0.5, (0.5, 2.0), ("a", "b", "c")) == "b"
    assert _bin(1.0, (0.5, 2.0), ("a", "b", "c")) == "b"
    np.testing.assert_allclose(_forward_world(np.eye(3)), [0.0, 0.0, 1.0])


def test_extrapolation_summary_reports_catastrophic_risk():
    rows = [
        {"final_translation_m": 0.2, "final_rotation_deg": 2.0},
        {"final_translation_m": 3.0, "final_rotation_deg": 2.0},
    ]
    summary = _summary(rows)
    assert summary["strict_success"]["count"] == 1
    assert summary["catastrophic_failure"]["count"] == 1
