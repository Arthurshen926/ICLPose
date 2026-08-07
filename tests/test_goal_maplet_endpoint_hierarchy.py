import numpy as np

from feature_extract.vfm.localization_goal_maplet.endpoint_hierarchy import (
    EndpointHierarchyCalibration,
    conditional_with_tail,
    fit_endpoint_hierarchy_calibration,
)


def test_conditional_with_tail_preserves_omitted_probability_mass():
    result = conditional_with_tail(np.asarray([0.3, 0.2]), 0.8)
    assert np.allclose(result, [0.375, 0.25, 0.375])
    assert np.sum(result) == 1.0


def test_hierarchy_calibration_roundtrip_and_mass_conservation(tmp_path):
    artifact = EndpointHierarchyCalibration(
        1.5, 0.25, 0.8, 1.2, 0.6,
        {"artifact_type": "goal_maplet_endpoint_hierarchy_calibration_v1"},
    )
    path = tmp_path / "hierarchy.json"
    artifact.save_json(path)
    loaded = EndpointHierarchyCalibration.load_json(path)
    assert loaded.content_sha256 == artifact.content_sha256
    for level in ("parent", "child", "mode"):
        value = loaded.calibrate_distribution(np.asarray([0.5, 0.3, 0.2]), level=level)
        assert np.isclose(np.sum(value), 1.0)
        assert np.all(value >= 0.0)


def test_fit_hierarchy_uses_proper_scores_to_sharpen_separable_support():
    distributions = [np.asarray([0.8, 0.2]), np.asarray([0.7, 0.3])]
    artifact = fit_endpoint_hierarchy_calibration(
        np.asarray([0.8, 0.7, 0.3, 0.2]), np.asarray([1, 1, 0, 0]),
        distributions, np.asarray([0, 0]),
        distributions, np.asarray([0, 0]),
        distributions, np.asarray([0, 0]),
    )
    calibrated = artifact.calibrate_support(np.asarray([0.8, 0.2]))
    assert calibrated[0] > 0.8
    assert calibrated[1] < 0.2

