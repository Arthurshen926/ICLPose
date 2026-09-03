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
