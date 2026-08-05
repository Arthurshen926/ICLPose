"""Runtime features and lightweight ranker for child-local primitive modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .canonical_field import CanonicalSurfaceField
from .child_local_likelihood import ChildLocalSurfaceLikelihood
from .physical_map import GoalMapletPhysicalMap


FEATURE_NAMES = (
    "log_vfm_mode_probability",
    "vfm_mode_rank_fraction",
    "field_confidence",
    "field_uncertainty",
    "normalized_reprojection_residual",
    "geometry_log_likelihood",
    "front_facing",
    "view_normal_incidence",
    "normalized_child_depth",
    "normalized_child_radius",
    "vfm_geometry_joint_score",
)


def child_local_mode_runtime_features(
    likelihood: ChildLocalSurfaceLikelihood,
    child_rows: np.ndarray,
    query_xy_px: np.ndarray,
    query_scale_px: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
) -> tuple[np.ndarray, np.ndarray]:
    """Build relative, runtime-only features for the VFM Top-M local modes."""

    import cv2

    children = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    primitive = np.asarray(likelihood.mode_primitive_rows, dtype=np.int64)
    probability = np.asarray(likelihood.mode_probabilities, dtype=np.float64)
    count, modes = primitive.shape
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if children.shape != (count,) or xy.shape != (count, 2) or scale.shape != (count,):
        raise ValueError("child-local mode feature inputs differ")
    valid = primitive >= 0
    safe = np.maximum(primitive, 0)
    field_row = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
    selected = field_row[safe]
    valid &= selected >= 0
    safe_selected = np.maximum(selected, 0)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(
        physical.primitive_centers[safe.reshape(-1)].astype(np.float64),
        rotation_vector, pose[:3, 3], matrix, distortion,
    )
    projected = projected.reshape(count, modes, 2)
    residual = np.linalg.norm(projected - xy[:, None], axis=2)
    normalized_residual = residual / np.maximum(scale[:, None], 8.0)
    geometry_log = -0.5 * np.square(normalized_residual)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    points = physical.primitive_centers[safe]
    camera_xyz = points @ pose[:3, :3].T + pose[:3, 3]
    view = camera_center[None, None] - points
    view /= np.maximum(np.linalg.norm(view, axis=2, keepdims=True), 1e-8)
    incidence = np.sum(physical.primitive_normals[safe] * view, axis=2)
    front = (physical.primitive_sidedness[safe] == 2) | (incidence >= 0.02)
    valid &= (camera_xyz[..., 2] > 0.05) & front
    child_center = physical.child_centers[children]
    child_frame = physical.child_frames[children]
    local = np.einsum("nmi,nji->nmj", points - child_center[:, None], child_frame)
    extent = np.maximum(physical.child_extents[children], 1e-4)
    normalized_depth = local[..., 2] / np.maximum(extent[:, None, 2], 0.05)
    normalized_radius = np.linalg.norm(local[..., :2] / extent[:, None, :2], axis=2)
    rank_fraction = np.broadcast_to(
        np.arange(modes, dtype=np.float64)[None] / max(modes - 1, 1), (count, modes)
    )
    log_probability = np.log(np.maximum(probability, 1e-12))
    confidence = field.confidence[safe_selected]
    uncertainty = field.uncertainty[safe_selected]
    output = np.stack([
        log_probability,
        rank_fraction,
        confidence,
        uncertainty,
        normalized_residual,
        geometry_log,
        front.astype(np.float64),
        incidence,
        normalized_depth,
        normalized_radius,
        log_probability + geometry_log,
    ], axis=2)
    output[~valid] = 0.0
    if not np.all(np.isfinite(output)):
        raise ValueError("child-local mode features contain non-finite values")
    return output.astype(np.float32), valid


@dataclass(frozen=True)
class ChildLocalModeRankerArtifact:
    estimator: object
    metadata: Mapping[str, object]

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise ValueError("child-local mode feature dimension differs")
        return np.asarray(self.estimator.predict_proba(value)[:, 1], dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": dict(self.metadata)}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ChildLocalModeRankerArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("child-local mode ranker feature contract differs")
        return cls(payload["estimator"], metadata)


@dataclass(frozen=True)
class ChildLocalPairwiseRankerArtifact:
    """Query-group ranker trained on continuous within-group surface utility.

    The estimator consumes ordered feature differences.  At runtime every
    valid mode is compared with the other modes from the *same* query group;
    the Borda mean is therefore a genuinely group-relative score rather than
    an independently calibrated binary label.
    """

    estimator: object
    metadata: Mapping[str, object]

    def score_modes(
        self,
        features: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(features, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if value.ndim != 3 or value.shape[2] != len(FEATURE_NAMES):
            raise ValueError("child-local pairwise feature dimension differs")
        if mask.shape != value.shape[:2]:
            raise ValueError("child-local pairwise validity shape differs")
        count, modes = mask.shape
        score = np.full((count, modes), -np.inf, dtype=np.float64)
        probability = np.zeros((count, modes), dtype=np.float64)
        temperature = max(float(self.metadata.get("score_temperature", 1.0)), 1e-4)
        pair_features: list[np.ndarray] = []
        pair_groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        offset = 0
        for group in range(count):
            rows = np.flatnonzero(mask[group])
            if rows.size == 0:
                continue
            if rows.size == 1:
                score[group, rows[0]] = 1.0
                probability[group, rows[0]] = 1.0
                continue
            left, right = np.triu_indices(rows.size, 1)
            left, right = rows[left], rows[right]
            difference = value[group, left] - value[group, right]
            pair_features.append(difference)
            pair_groups.append((group, left, right))
            offset += difference.shape[0]
        if pair_features:
            pair_probability = np.asarray(
                self.estimator.predict_proba(np.concatenate(pair_features, axis=0))[:, 1],
                dtype=np.float64,
            )
            offset = 0
            for group, left, right in pair_groups:
                size = left.size
                current = pair_probability[offset : offset + size]
                offset += size
                total = np.zeros((modes,), dtype=np.float64)
                comparisons = np.zeros((modes,), dtype=np.float64)
                np.add.at(total, left, current)
                np.add.at(total, right, 1.0 - current)
                np.add.at(comparisons, left, 1.0)
                np.add.at(comparisons, right, 1.0)
                rows = np.flatnonzero(mask[group])
                score[group, rows] = total[rows] / np.maximum(comparisons[rows], 1.0)
                logits = score[group, rows] / temperature
                logits -= np.max(logits)
                mass = np.exp(np.clip(logits, -60.0, 0.0))
                probability[group, rows] = mass / np.maximum(np.sum(mass), 1e-12)
        return score, probability

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": dict(self.metadata)}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ChildLocalPairwiseRankerArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_child_local_pairwise_ranker_v2":
            raise ValueError("not a child-local pairwise ranker")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("child-local pairwise feature contract differs")
        return cls(payload["estimator"], metadata)
