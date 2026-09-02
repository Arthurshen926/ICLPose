import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_large_plane_radio_pose import (
    build_plane_families,
    calibrate_no_match,
)


def _region(row, chart, z, descriptor, x=0.0):
    points = np.asarray([[x, 0, z], [x + .1, 0, z], [x, .1, z], [x + .1, .1, z]])
    return {
        "region_id": row, "chart": chart, "chart_name": f"c{chart}",
        "local_region": 0, "pixel_count": 100,
        "normal": np.asarray([0.0, 0.0, 1.0]), "offset": z,
        "descriptor": np.asarray(descriptor, np.float64),
        "token_points": points,
    }


def test_family_merges_cross_view_overlap_but_not_parallel_offset():
    regions = [
        _region(0, 0, 2.0, [1, 0]),
        _region(1, 1, 2.02, [1, 0], x=.15),
        _region(2, 2, 3.0, [0, 1]),
    ]
    families = build_plane_families(regions)
    assert len(families) == 2
    assert sorted(row["chart_count"] for row in families) == [1, 2]
    assert regions[0]["family_id"] == regions[1]["family_id"]
    assert regions[2]["family_id"] != regions[0]["family_id"]


def test_no_match_calibration_is_source_only_and_bounded():
    regions = [
        _region(0, 0, 2.0, [1.0, 0.0]),
        _region(1, 1, 2.01, [.99, .01], x=.15),
        _region(2, 2, 3.0, [0.0, 1.0]),
        _region(3, 3, 4.0, [-1.0, 0.0]),
    ]
    families = build_plane_families(regions)
    result = calibrate_no_match(regions, families)
    assert 0.4 <= result["score_threshold"] <= .9
    assert 0 <= result["margin_threshold"] <= .2
    assert result["source_unmatched_false_accept_rate"] <= .1
    assert result["source_trial_count"] == 4
