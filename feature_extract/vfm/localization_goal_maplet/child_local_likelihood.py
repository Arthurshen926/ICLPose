"""Parent-conditioned child-local surface likelihood over one canonical field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical_field import CanonicalSurfaceField
from .physical_map import GoalMapletPhysicalMap


@dataclass(frozen=True)
class ChildLocalSurfaceLikelihood:
    map_points: np.ndarray
    expected_points: np.ndarray
    covariance_local: np.ndarray
    mode_primitive_rows: np.ndarray
    mode_probabilities: np.ndarray
    maximum_similarity: np.ndarray
    entropy: np.ndarray
    feature_coverage: np.ndarray
    null_probabilities: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.map_points, dtype=np.float64)
        count = point.shape[0]
        expected = np.asarray(self.expected_points, dtype=np.float64)
        covariance = np.asarray(self.covariance_local, dtype=np.float64)
        modes = np.asarray(self.mode_primitive_rows, dtype=np.int64)
        probability = np.asarray(self.mode_probabilities, dtype=np.float64)
        if (
            point.shape != (count, 3)
            or expected.shape != point.shape
            or covariance.shape != (count, 2, 2)
            or modes.ndim != 2
            or modes.shape[0] != count
            or probability.shape != modes.shape
        ):
            raise ValueError("invalid child-local surface likelihood arrays")
        for name in (
            "maximum_similarity", "entropy", "feature_coverage", "null_probabilities",
        ):
            if np.asarray(getattr(self, name)).reshape(-1).shape != (count,):
                raise ValueError("invalid child-local likelihood statistic")


def predict_child_local_surface_likelihood(
    query_descriptors: np.ndarray,
    child_rows: np.ndarray,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    *,
    temperature: float = 0.07,
    maximum_modes: int = 8,
    confidence_prior_weight: float = 0.25,
    query_xy_px: np.ndarray | None = None,
    query_scale_px: np.ndarray | None = None,
    pose_w2c: np.ndarray | None = None,
    camera=None,
    geometry_weight: float = 1.0,
) -> ChildLocalSurfaceLikelihood:
    """Predict a multi-modal primitive/UV distribution inside each known child.

    This is deliberately conditioned on child identity.  It does not ask a
    coarse RADIO descriptor to distinguish the whole scene at primitive
    granularity, and it stores no second map embedding.
    """

    query = np.asarray(query_descriptors, dtype=np.float32)
    children = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    if query.ndim != 2 or query.shape[0] != children.size or query.shape[1] != field.feature_dim:
        raise ValueError("query descriptors, children and canonical field differ")
    if np.any((children < 0) | (children >= physical.child_parent_rows.size)):
        raise ValueError("child-local likelihood received an invalid child row")
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    if float(temperature) <= 0.0 or int(maximum_modes) < 1:
        raise ValueError("invalid child-local likelihood configuration")
    geometry_values = (query_xy_px, query_scale_px, pose_w2c, camera)
    geometry_conditioned = any(value is not None for value in geometry_values)
    if geometry_conditioned and not all(value is not None for value in geometry_values):
        raise ValueError("geometry-conditioned likelihood requires xy, scale, pose and camera")
    if geometry_conditioned:
        xy_px = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
        scale_px = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        if xy_px.shape[0] != children.size or scale_px.shape != (children.size,):
            raise ValueError("geometry conditioning and query count differ")
        from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
        import cv2

        camera_matrix, distortion = camera_matrix_and_distortion(camera)
        rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
        camera_center = -pose[:3, :3].T @ pose[:3, 3]
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    field_by_primitive = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_by_primitive[field.primitive_rows] = np.arange(field.primitive_rows.size, dtype=np.int64)
    mode_count = int(maximum_modes)
    map_points = np.zeros((children.size, 3), dtype=np.float64)
    expected_points = np.zeros_like(map_points)
    covariance = np.zeros((children.size, 2, 2), dtype=np.float64)
    mode_rows = np.full((children.size, mode_count), -1, dtype=np.int64)
    mode_probability = np.zeros((children.size, mode_count), dtype=np.float64)
    maximum_similarity = np.full((children.size,), -1.0, dtype=np.float64)
    entropy = np.zeros((children.size,), dtype=np.float64)
    coverage = np.zeros((children.size,), dtype=np.float64)
    null = np.ones((children.size,), dtype=np.float64)
    # Group identical children.  Training/evidence generation evaluates many
    # query groups against the same physical tile; one matrix multiply and one
    # projection per child avoids thousands of tiny Python/BLAS calls.
    for child in np.unique(children).tolist():
        query_rows = np.flatnonzero(children == int(child))
        start, end = int(physical.child_member_offsets[child]), int(physical.child_member_offsets[child + 1])
        primitive = physical.child_member_primitive_rows[start:end]
        membership = np.asarray(physical.child_member_weights[start:end], dtype=np.float64)
        selected = field_by_primitive[primitive]
        valid = selected >= 0
        total_membership = max(float(np.sum(membership)), 1e-12)
        child_coverage = float(np.sum(membership[valid]) / total_membership)
        coverage[query_rows] = child_coverage
        if not np.any(valid):
            map_points[query_rows] = physical.child_centers[child]
            expected_points[query_rows] = physical.child_centers[child]
            continue
        primitive = primitive[valid]
        membership = membership[valid]
        selected = selected[valid]
        similarity = np.asarray(query[query_rows] @ field.codes[selected].T, dtype=np.float64)
        reliability = np.maximum(
            field.confidence[selected] * (1.0 - field.uncertainty[selected]), 1e-4
        )
        logits = similarity / float(temperature)
        logits += float(confidence_prior_weight) * np.log(reliability)[None]
        logits += 0.10 * np.log(np.maximum(membership, 1e-6))[None]
        visible = np.ones_like(logits, dtype=bool)
        if geometry_conditioned:
            projected, _ = cv2.projectPoints(
                physical.primitive_centers[primitive].astype(np.float64),
                rotation_vector, pose[:3, 3], camera_matrix, distortion,
            )
            projected = projected.reshape(-1, 2)
            camera_xyz = physical.primitive_centers[primitive] @ pose[:3, :3].T + pose[:3, 3]
            view = camera_center[None] - physical.primitive_centers[primitive]
            view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
            incidence = np.sum(physical.primitive_normals[primitive] * view, axis=1)
            front = (physical.primitive_sidedness[primitive] == 2) | (incidence >= 0.02)
            residual = np.linalg.norm(projected[None] - xy_px[query_rows, None], axis=2)
            sigma = np.maximum(scale_px[query_rows], 8.0)
            logits += float(geometry_weight) * -0.5 * np.square(residual / sigma[:, None])
            visible &= ((camera_xyz[:, 2] > 0.05) & front)[None]
        row_has_visible = np.any(visible, axis=1)
        safe_logits = np.where(visible, logits, -np.inf)
        maximum = np.max(safe_logits, axis=1, keepdims=True)
        maximum[~np.isfinite(maximum)] = 0.0
        probability = np.exp(np.clip(safe_logits - maximum, -60.0, 0.0)) * visible
        probability /= np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)
        points = physical.primitive_centers[primitive]
        for local_row, index in enumerate(query_rows.tolist()):
            maximum_similarity[index] = float(np.max(similarity[local_row]))
            if not row_has_visible[local_row]:
                map_points[index] = physical.child_centers[child]
                expected_points[index] = physical.child_centers[child]
                null[index] = 1.0
                continue
            order = np.argsort(-probability[local_row], kind="stable")[:mode_count]
            order = order[probability[local_row, order] > 0.0]
            mode_rows[index, : order.size] = primitive[order]
            mode_probability[index, : order.size] = probability[local_row, order]
            map_points[index] = points[int(order[0])]
            expected_points[index] = probability[local_row] @ points
            local = (points - expected_points[index]) @ physical.child_frames[child].T
            covariance[index] = np.einsum(
                "n,ni,nj->ij", probability[local_row], local[:, :2], local[:, :2]
            )
            entropy[index] = float(-np.sum(
                probability[local_row] * np.log(np.maximum(probability[local_row], 1e-12))
            ))
        # Null represents absent canonical evidence, not low visual contrast.
        # Similarity calibration belongs to the query-level validity model.
        null[query_rows[row_has_visible]] = float(np.clip(1.0 - child_coverage, 0.0, 1.0))
    return ChildLocalSurfaceLikelihood(
        map_points=map_points,
        expected_points=expected_points,
        covariance_local=covariance,
        mode_primitive_rows=mode_rows,
        mode_probabilities=mode_probability,
        maximum_similarity=maximum_similarity,
        entropy=entropy,
        feature_coverage=coverage,
        null_probabilities=null,
    )
