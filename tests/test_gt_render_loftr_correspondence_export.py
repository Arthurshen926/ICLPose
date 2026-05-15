from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.export_gt_render_loftr_correspondences import payload_from_loftr_result


class _Result:
    success = True
    extra = {
        "query_keypoints": np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
        "ref_keypoints": np.array([[11.0, 21.0], [31.0, 41.0], [51.0, 61.0]], dtype=np.float32),
        "pts3d_world": np.ones((3, 3), dtype=np.float32),
        "confidence": np.array([0.2, 0.9, 0.6], dtype=np.float32),
        "pnp_inlier_mask": np.array([True, False, True]),
        "loftr_hw": np.array([120, 200], dtype=np.int32),
    }


def test_payload_from_loftr_result_keeps_inlier_matches_sorted_by_confidence():
    payload = payload_from_loftr_result(_Result(), max_points=8, inlier_only=True, source="unit")

    assert payload["query_xy"].shape == (2, 2)
    assert payload["map_xy"].shape == (2, 2)
    assert np.allclose(payload["confidence"], np.array([0.6, 0.2], dtype=np.float32))
    assert payload["query_xy"].tolist() == [[50.0, 60.0], [10.0, 20.0]]
    assert payload["map_xy"].tolist() == [[51.0, 61.0], [11.0, 21.0]]
    assert payload["query_hw"].tolist() == [120, 200]
    assert str(payload["coordinate_space"]) == "image"
