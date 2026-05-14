from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feature_retrieval.local_lattice_render_loftr_quality_export import (
    build_direction_hard_candidates,
    build_local_lattice_candidate_bank,
    deterministic_initial_delta,
    merge_candidate_banks,
)


def _camera_center(pose: np.ndarray) -> np.ndarray:
    return -pose[:3, :3].T @ pose[:3, 3]


def test_deterministic_initial_delta_has_requested_magnitude():
    delta = deterministic_initial_delta(query_index=0, trans_cm=25.0, rot_deg=5.0)

    assert np.isclose(np.linalg.norm(delta[:3]), 0.25, atol=1e-6)
    assert np.isclose(np.linalg.norm(delta[3:]), np.deg2rad(5.0), atol=1e-6)


def test_direction_hard_candidates_include_toward_and_opposite_translation():
    gt = np.eye(4, dtype=np.float32)
    init_delta = deterministic_initial_delta(query_index=0, trans_cm=25.0, rot_deg=0.0)
    from feature_retrieval.local_lattice_render_loftr_quality_export import apply_delta_np

    init = apply_delta_np(gt, init_delta)
    candidates = build_direction_hard_candidates(init, init_delta, fractions=[1.0], include_identity=True)

    init_center = _camera_center(init)
    gt_center = _camera_center(gt)
    cand_centers = np.stack([_camera_center(p) for p in candidates], axis=0)
    target = gt_center - init_center
    moves = cand_centers - init_center[None]
    cos = (moves @ target) / np.maximum(np.linalg.norm(moves, axis=1) * np.linalg.norm(target), 1e-8)

    assert candidates.shape[0] == 7
    assert cos.max() > 0.95
    assert cos.min() < -0.95


def test_merge_candidate_banks_deduplicates_and_pads():
    pose = np.eye(4, dtype=np.float32)
    shifted = pose.copy()
    shifted[0, 3] = 1.0

    merged, valid = merge_candidate_banks([np.stack([pose, pose]), shifted[None]], max_candidates=4, pad_pose=pose)

    assert merged.shape == (4, 4, 4)
    assert valid.tolist() == [True, True, False, False]
    assert np.allclose(merged[0], pose)
    assert np.allclose(merged[1], shifted)


def test_build_local_lattice_candidate_bank_keeps_init_separate_from_candidates():
    gt = np.eye(4, dtype=np.float32)

    init, candidates, valid, init_delta = build_local_lattice_candidate_bank(
        gt,
        query_index=1,
        center_trans_cm=25.0,
        center_rot_deg=5.0,
        lattice_trans_cm=[0, 5, 25],
        lattice_rot_deg=[0, 1, 5],
        topk=12,
        direction_fractions=[1.0, 0.5],
        include_identity=True,
        append_gt_candidate=False,
    )

    assert init.shape == (4, 4)
    assert candidates.shape == (12, 4, 4)
    assert valid.shape == (12,)
    assert valid.sum() == 12
    assert init_delta.shape == (6,)
    assert np.allclose(candidates[0], init)
    assert not np.allclose(init, gt)
