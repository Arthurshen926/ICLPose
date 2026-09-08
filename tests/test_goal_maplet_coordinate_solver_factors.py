import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_coordinate_solver_factors import _factor_inputs


def test_factor_inputs_change_only_requested_source_without_mutation():
    keys = ("world_points", "query_measurements_xy", "query_measurement_variance_px2",
            "prototype_centroid_covariance_world_m2")
    baseline = {key: np.asarray([i]) for i, key in enumerate(keys)}
    local = {key: np.asarray([i + 10]) for i, key in enumerate(keys)}
    for means in (0, 1):
        for covariance in (0, 1):
            result = _factor_inputs(baseline, local, means, covariance)
            for i in range(4):
                np.testing.assert_array_equal(result[i], [i + 10 * (means if i < 2 else covariance)])
    np.testing.assert_array_equal(baseline["world_points"], [0])
