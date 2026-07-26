from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_surface_pose_oracles import (
    _best,
    _metrics,
)


def test_oracle_uses_fixed_translation_rotation_cost() -> None:
    assert _best([(0.10, 8.0), (0.20, 0.5)]) == (0.20, 0.5)
    assert _best([]) is None


def test_oracle_metrics_keep_failures_in_recall_denominator() -> None:
    summary = _metrics([(0.03, 0.5), (0.08, 2.0), None])
    assert summary["query_count"] == 3
    assert summary["available_count"] == 2
    assert np.isclose(summary["coverage"], 2.0 / 3.0)
    assert np.isclose(summary["recall_4cm_1deg_all_queries"], 1.0 / 3.0)
    assert np.isclose(summary["recall_10cm_5deg_all_queries"], 2.0 / 3.0)
