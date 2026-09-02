import math

import pytest

from feature_extract.tools.vfm.build_goal_maplet_moge3_chart_initializers import focal_for_source


def test_focal_is_rescaled_from_chart_canvas_to_source_canvas():
    assert focal_for_source(445.0, 512, 1024) == 890.0
    source_fov = 2.0 * math.atan(1024.0 / (2.0 * focal_for_source(445.0, 512, 1024)))
    chart_fov = 2.0 * math.atan(512.0 / (2.0 * 445.0))
    assert source_fov == pytest.approx(chart_fov)


@pytest.mark.parametrize("focal,canvas,source", [(0, 512, 1024), (445, 0, 1024), (445, 512, 0)])
def test_invalid_focal_canvas_contract_fails(focal, canvas, source):
    with pytest.raises(ValueError):
        focal_for_source(focal, canvas, source)
