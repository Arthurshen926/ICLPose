import json

import numpy as np
import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_pose_modes import (
    _load_query_evaluation_payload,
)


def test_actual_query_path_does_not_require_contributor_identity_arrays(tmp_path):
    path = tmp_path / "query.npz"
    np.savez_compressed(
        path,
        pose_w2c=np.eye(4, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"image_id": "seq3/frame00001.png"})),
    )
    metadata, pose, labels = _load_query_evaluation_payload(
        path, load_oracle_labels=False,
    )
    assert metadata["image_id"] == "seq3/frame00001.png"
    np.testing.assert_allclose(pose, np.eye(4))
    assert labels is None

    with pytest.raises(KeyError):
        _load_query_evaluation_payload(path, load_oracle_labels=True)
