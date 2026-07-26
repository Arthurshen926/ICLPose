from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.select_surface_pose_by_feature_map import (
    _pose_disagreement,
    _select_candidate,
)


def _pose(center_x: float, rotation_deg: float = 0.0) -> list[float]:
    angle = np.radians(rotation_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ np.asarray(
        [center_x, 0.0, 0.0],
        dtype=np.float64,
    )
    return pose.reshape(-1).tolist()


def _evidence(score: float) -> dict[str, object]:
    return {
        "log_likelihood": score,
        "supported_cell_count": 10,
        "supported_anchor_count": 20,
    }


def test_pose_disagreement_uses_camera_centers() -> None:
    translation, rotation = _pose_disagreement(
        np.asarray(_pose(0.0, 0.0)),
        np.asarray(_pose(3.0, 20.0)),
    )
    assert np.isclose(translation, 3.0)
    assert np.isclose(rotation, 20.0)


def test_selector_requires_fixed_feature_margin() -> None:
    selected, reason, disagreement = _select_candidate(
        candidate_rows={
            "new": {"pose_w2c": _pose(0.0)},
            "safe": {"pose_w2c": _pose(0.0)},
        },
        evidence_by_name={
            "new": _evidence(-0.97),
            "safe": _evidence(-1.0),
        },
        safety_candidate="safe",
        minimum_score_margin=0.05,
        maximum_disagreement_translation_m=5.0,
        maximum_disagreement_rotation_deg=10.0,
    )
    assert selected == "safe"
    assert reason == "insufficient_feature_score_margin"
    assert disagreement is None


def test_selector_rejects_large_pose_disagreement() -> None:
    selected, reason, disagreement = _select_candidate(
        candidate_rows={
            "new": {"pose_w2c": _pose(8.0)},
            "safe": {"pose_w2c": _pose(0.0)},
        },
        evidence_by_name={
            "new": _evidence(-0.5),
            "safe": _evidence(-1.0),
        },
        safety_candidate="safe",
        minimum_score_margin=0.05,
        maximum_disagreement_translation_m=5.0,
        maximum_disagreement_rotation_deg=10.0,
    )
    assert selected == "safe"
    assert reason == "candidate_pose_disagreement_exceeds_limit"
    assert disagreement is not None
    assert np.isclose(disagreement[0], 8.0)


def test_selector_accepts_clear_consistent_feature_gain() -> None:
    selected, reason, disagreement = _select_candidate(
        candidate_rows={
            "new": {"pose_w2c": _pose(1.0, 2.0)},
            "safe": {"pose_w2c": _pose(0.0, 0.0)},
        },
        evidence_by_name={
            "new": _evidence(-0.5),
            "safe": _evidence(-1.0),
        },
        safety_candidate="safe",
        minimum_score_margin=0.05,
        maximum_disagreement_translation_m=5.0,
        maximum_disagreement_rotation_deg=10.0,
    )
    assert selected == "new"
    assert reason is None
    assert disagreement is not None
