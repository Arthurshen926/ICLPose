from __future__ import annotations

from feature_extract.vfm.measurement_v1.support_selection import (
    SUPPORT_SELECTOR_FEATURE_NAMES,
    MeasurementSupportSelector,
)


def test_support_selector_probabilities_are_normalized_and_prefer_higher_score() -> None:
    selector = MeasurementSupportSelector(
        mean=tuple(0.0 for _ in SUPPORT_SELECTOR_FEATURE_NAMES),
        scale=tuple(1.0 for _ in SUPPORT_SELECTOR_FEATURE_NAMES),
        weights=(1.0,) + tuple(0.0 for _ in SUPPORT_SELECTOR_FEATURE_NAMES[1:]),
        bias=0.0,
        temperature=0.5,
    )
    rows = [
        {"support_view_probability": "0.2"},
        {"support_view_probability": "0.8"},
    ]

    probabilities = selector.probabilities(rows)

    assert abs(float(probabilities.sum()) - 1.0) < 1e-12
    assert probabilities[1] > probabilities[0]


def test_support_selector_feature_schema_excludes_query_pose_and_targets() -> None:
    forbidden = ("target", "query_reprojection", "view_angle", "gt", "pose")

    assert all(not any(token in name for token in forbidden) for name in SUPPORT_SELECTOR_FEATURE_NAMES)
