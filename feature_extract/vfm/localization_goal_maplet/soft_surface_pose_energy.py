"""Correspondence-free RADIO/3DGS energy for a fixed pose candidate.

Child identity remains latent: for each token the energy sums the query's
truncated child evidence at the child rendered by the candidate pose.  The
same fixed query-evidence denominator is used for every pose.  Missing render
support receives the unknown floor -1 and therefore cannot improve a score by
disappearing.  Canonical RADIO cosine is evaluated on the same support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pure_retrieval import PureRadioPhysicalRetrieval


@dataclass(frozen=True)
class SoftSurfacePoseEnergy:
    combined_score: float
    support_layout_score: float
    canonical_radio_score: float
    effective_query_mass: float
    rendered_visible_fraction: float
    rendered_feature_fraction: float


def score_soft_surface_pose_energy(
    query_feature: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    rendered,
    *,
    radio_weight: float = 0.5,
) -> SoftSurfacePoseEnergy:
    """Score one rendered pose without selecting hard 2D--3D correspondences."""

    alpha = float(radio_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("radio_weight must lie in [0,1]")
    query = np.asarray(query_feature, dtype=np.float32)
    render = np.asarray(rendered.feature, dtype=np.float32)
    if query.shape != render.shape or query.ndim != 3:
        raise ValueError("query/rendered features must have shape [C,H,W]")
    height, width = query.shape[1:]
    if retrieval.token_xy.shape[0] != height * width:
        raise ValueError("retrieval tokens and rendered grid differ")
    child = np.asarray(rendered.child_id, dtype=np.int64).reshape(-1)
    visible = np.asarray(rendered.visibility, dtype=bool).reshape(-1)
    feature_valid = np.asarray(rendered.mask, dtype=bool).reshape(-1)
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    probability = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    selected = np.zeros((int(np.max(rows, initial=-1)) + 1,), dtype=bool)
    selected_rows = np.asarray(retrieval.scene_child_rows, dtype=np.int64)
    if selected_rows.size == 0:
        return SoftSurfacePoseEnergy(-1.0, -1.0, -1.0, 0.0, float(visible.mean()), float(feature_valid.mean()))
    if selected_rows.size:
        if int(np.max(selected_rows)) >= selected.size:
            selected = np.pad(selected, (0, int(np.max(selected_rows)) + 1 - selected.size))
        selected[selected_rows] = True
    safe_rows = np.maximum(rows, 0)
    retained = (
        (rows >= 0)
        & (safe_rows < selected.size)
        & selected[np.minimum(safe_rows, selected.size - 1)]
    )
    retained_mass = np.sum(np.where(retained, probability, 0.0), axis=1)
    effective = float(np.sum(retained_mass))
    if effective <= 0.0:
        return SoftSurfacePoseEnergy(-1.0, -1.0, -1.0, 0.0, float(visible.mean()), float(feature_valid.mean()))
    match = retained & (rows == child[:, None]) & visible[:, None]
    matched_mass = np.sum(np.where(match, probability, 0.0), axis=1)
    conditional_match = np.divide(
        matched_mass, retained_mass,
        out=np.zeros_like(matched_mass), where=retained_mass > 0.0,
    )
    support_atom = 2.0 * conditional_match - 1.0
    support_score = float(np.sum(retained_mass * support_atom) / effective)
    query_unit = query / np.maximum(np.linalg.norm(query, axis=0, keepdims=True), 1e-8)
    render_unit = render / np.maximum(np.linalg.norm(render, axis=0, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(query_unit * render_unit, axis=0).reshape(-1), -1.0, 1.0)
    radio_atom = np.where(feature_valid, cosine, -1.0)
    radio_score = float(np.sum(retained_mass * radio_atom) / effective)
    combined = (1.0 - alpha) * support_score + alpha * radio_score
    return SoftSurfacePoseEnergy(
        combined_score=float(combined),
        support_layout_score=float(support_score),
        canonical_radio_score=float(radio_score),
        effective_query_mass=effective,
        rendered_visible_fraction=float(np.mean(visible)),
        rendered_feature_fraction=float(np.mean(feature_valid)),
    )


def translate_camera_world(pose_w2c: np.ndarray, delta_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    center = -pose[:3, :3].T @ pose[:3, 3] + np.asarray(delta_world, dtype=np.float64)
    pose[:3, 3] = -pose[:3, :3] @ center
    return pose


def rotate_camera_local(
    pose_w2c: np.ndarray,
    axis_camera: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    axis = np.asarray(axis_camera, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = np.radians(float(angle_degrees))
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    center = -pose[:3, :3].T @ pose[:3, 3]
    pose[:3, :3] = delta @ pose[:3, :3]
    pose[:3, 3] = -pose[:3, :3] @ center
    return pose
