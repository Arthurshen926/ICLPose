import numpy as np

from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _errors


def test_pose_errors_use_camera_centres_and_relative_rotation(tmp_path):
    gt = np.eye(4)
    np.savez_compressed(tmp_path / "seq0__frame0.npz", pose_w2c=gt)
    pose = np.eye(4)
    pose[0, 3] = -0.2
    arrays = {
        "names": np.asarray(["seq0__frame0.npz"]),
        "pose_w2c": pose[None],
        "usable": np.asarray([True]),
    }
    translation, rotation = _errors(arrays, tmp_path)
    np.testing.assert_allclose(translation, [0.2])
    np.testing.assert_allclose(rotation, [0.0])
def test_threshold_hits_preserve_one_sided_usable_recovery():
    import numpy as np
    from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _threshold_hits
    baseline = _threshold_hits(np.asarray([np.inf, .1]), np.asarray([np.inf, 1.]), .25, 2.)
    method = _threshold_hits(np.asarray([.1, np.inf]), np.asarray([1., np.inf]), .25, 2.)
    assert int(np.sum(~baseline & method)) == 1
    assert int(np.sum(baseline & ~method)) == 1


def test_temporal_blocks_sort_numeric_frames_and_preserve_route_strata():
    import numpy as np
    from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _temporal_block_ci
    names = np.asarray(["seq1__frame10.png.npz", "seq1__frame2.png.npz",
                        "seq2__frame1.png.npz", "seq2__frame2.png.npz"])
    result = _temporal_block_ci(names, np.asarray([1., 1., 3., 3.]), 2, 100, 42)
    assert result["estimate"] == result["lower"] == result["upper"] == 2.
    assert result["route_count"] == 2
    shuffled = _temporal_block_ci(names[[1, 0, 3, 2]], np.asarray([1., 1., 3., 3.]), 2, 100, 42)
    assert result == shuffled


def test_temporal_blocks_reject_ambiguous_frame_identity():
    import numpy as np
    import pytest
    from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _temporal_block_ci
    with pytest.raises(ValueError, match="duplicate"):
        _temporal_block_ci(np.asarray(["seq1__frame1.png.npz"] * 2), np.ones(2), 2, 10, 42)
