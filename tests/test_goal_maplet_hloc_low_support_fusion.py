import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_hloc_low_support import (
    _parse_hloc_results,
    _plane_name_to_hloc_name,
    _select_hloc_fallback,
)


def test_plane_name_to_hloc_name_is_strict():
    assert _plane_name_to_hloc_name("seq8__frame00126.png.npz") == "seq8/frame00126.png"
    try:
        _plane_name_to_hloc_name("frame00126.png")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid planar name was accepted")


def test_hloc_wxyz_pose_parser(tmp_path):
    path = tmp_path / "results.txt"
    path.write_text("seq4/frame00001.png 1 0 0 0 1 2 3\n")
    poses = _parse_hloc_results(path)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(poses["seq4/frame00001.png"], expected)


def test_low_support_fallback_uses_existing_strict_boundary():
    selected = _select_hloc_fallback(
        np.asarray([True, True, True, False]),
        np.asarray([0.39, 0.4, 0.8, 0.9]),
        0.4,
    )
    np.testing.assert_array_equal(selected, [True, False, False, True])
