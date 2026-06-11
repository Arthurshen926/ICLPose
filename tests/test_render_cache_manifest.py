from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _cache_file_sha1,
    _cache_numpy_sha1,
    _cache_pose_hash,
)


def test_cache_hash_helpers_are_stable_and_sensitive(tmp_path) -> None:
    array = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    same = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    changed = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)

    assert _cache_numpy_sha1(array) == _cache_numpy_sha1(same)
    assert _cache_numpy_sha1(array) != _cache_numpy_sha1(changed)
    assert _cache_pose_hash(np.eye(4, dtype=np.float64)) == _cache_pose_hash(np.eye(4, dtype=np.float64))

    path = tmp_path / "cache.bin"
    path.write_bytes(b"abc")
    digest = _cache_file_sha1(path)
    path.write_bytes(b"abcd")

    assert digest != _cache_file_sha1(path)
    assert _cache_file_sha1(tmp_path / "missing.bin") is None
