"""Coordinate-correct physical-surface metrics for pure RADIO retrieval.

The existing 2DGS contributor cache is rendered on an ideal-pinhole sampling
grid, whereas Cambridge RADIO tokens were extracted from the raw
``SIMPLE_RADIAL`` image.  Evaluation therefore inverse-warps each raw sample
ray into the pinhole contributor grid before aggregating visibility.  This is
a sampled, map-relative visibility target; it is not claimed to be real-world
free-space or dense-pixel ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .pfir import ContributorLabels
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import (
    PureRadioPhysicalRetrieval,
    aggregate_sparse_token_evidence,
)
from .visibility import camera_center_from_w2c


COORDINATE_CONTRACT = (
    "raw_simple_radial_equal_area_samples_inverse_warped_to_ideal_pinhole_"
    "contributor_grid_nearest_center_v1"
)
LEGACY_COORDINATE_CONTRACT = "legacy_pinhole_contributor_grid_treated_as_raw_diagnostic_v1"


def inverse_simple_radial(
    distorted_xy: np.ndarray,
    k1: float,
    *,
    iterations: int = 12,
) -> np.ndarray:
    """Invert ``x_d=x_u(1+k1*||x_u||^2)`` on normalized coordinates."""

    value = np.asarray(distorted_xy, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 2 or np.any(~np.isfinite(value)):
        raise ValueError("distorted coordinates must be finite (...,2)")
    if not np.isfinite(float(k1)) or int(iterations) <= 0:
        raise ValueError("invalid SIMPLE_RADIAL inverse parameters")
    radius_distorted = np.linalg.norm(value, axis=-1)
    radius = radius_distorted.copy()
    for _ in range(int(iterations)):
        function = radius * (1.0 + float(k1) * radius * radius) - radius_distorted
        derivative = 1.0 + 3.0 * float(k1) * radius * radius
        if np.any(derivative <= 1e-8):
            raise ValueError("SIMPLE_RADIAL model is non-monotone on the sampled field")
        radius -= function / derivative
    if np.any(radius < -1e-10) or np.any(~np.isfinite(radius)):
        raise ValueError("SIMPLE_RADIAL inverse did not converge")
    scale = np.divide(
        radius,
        radius_distorted,
        out=np.ones_like(radius),
        where=radius_distorted > 1e-15,
    )
    return value * scale[..., None]


def remap_pinhole_contributors_to_raw_grid(
    labels: ContributorLabels,
    *,
    camera_model_id: int,
    camera_width: int,
    camera_height: int,
    camera_params: np.ndarray,
) -> tuple[ContributorLabels, dict[str, object]]:
    """Sample pinhole contributors at rays of the raw distorted image grid."""

    ids = np.asarray(labels.topk_primitive_ids, dtype=np.int64)
    weights = np.asarray(labels.topk_weights, dtype=np.float32)
    if ids.ndim != 3 or weights.shape != ids.shape:
        raise ValueError("contributor arrays must have shape (H,W,K)")
    height, width, _ = ids.shape
    if int(camera_width) <= 0 or int(camera_height) <= 0:
        raise ValueError("camera canvas must be positive")
    scale_x = float(width) / float(camera_width)
    scale_y = float(height) / float(camera_height)
    if abs(scale_x - scale_y) > 1e-12:
        raise ValueError("contributor grid must preserve camera aspect scale")
    params = np.asarray(camera_params, dtype=np.float64).reshape(-1)
    model = int(camera_model_id)
    if model == 0 and params.size >= 3:
        f, cx, cy = params[:3]
        k1 = 0.0
    elif model == 2 and params.size >= 4:
        f, cx, cy, k1 = params[:4]
    else:
        raise ValueError("retrieval truth requires PINHOLE or SIMPLE_RADIAL camera")
    if float(f) <= 0.0 or np.any(~np.isfinite([f, cx, cy, k1])):
        raise ValueError("invalid camera intrinsics")
    f_grid = float(f) * scale_x
    cx_grid = float(cx) * scale_x
    cy_grid = float(cy) * scale_y
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    distorted = np.stack(
        [(xx - cx_grid) / f_grid, (yy - cy_grid) / f_grid], axis=-1
    )
    undistorted = inverse_simple_radial(distorted, float(k1))
    ideal_x = f_grid * undistorted[..., 0] + cx_grid
    ideal_y = f_grid * undistorted[..., 1] + cy_grid
    source_x = np.floor(ideal_x).astype(np.int64)
    source_y = np.floor(ideal_y).astype(np.int64)
    valid = (
        (source_x >= 0)
        & (source_x < width)
        & (source_y >= 0)
        & (source_y < height)
    )
    safe_x = np.clip(source_x, 0, width - 1)
    safe_y = np.clip(source_y, 0, height - 1)
    remapped_ids = ids[safe_y, safe_x].copy()
    remapped_weights = weights[safe_y, safe_x].copy()
    remapped_ids[~valid] = -1
    remapped_weights[~valid] = 0.0
    radius_squared = np.sum(undistorted * undistorted, axis=-1)
    forward = undistorted * (1.0 + float(k1) * radius_squared)[..., None]
    residual_grid = np.linalg.norm(forward - distorted, axis=-1) * f_grid
    displacement_grid = np.sqrt(
        np.square(ideal_x - xx) + np.square(ideal_y - yy)
    )
    audit = {
        "coordinate_contract": COORDINATE_CONTRACT,
        "camera_model_id": model,
        "camera_width": int(camera_width),
        "camera_height": int(camera_height),
        "contributor_width": int(width),
        "contributor_height": int(height),
        "scale": scale_x,
        "k1": float(k1),
        "valid_raw_sample_fraction": float(np.mean(valid)),
        "maximum_inverse_roundtrip_residual_contributor_px": float(
            np.max(residual_grid, initial=0.0)
        ),
        "mean_pinhole_to_raw_displacement_contributor_px": float(
            np.mean(displacement_grid)
        ),
        "maximum_pinhole_to_raw_displacement_contributor_px": float(
            np.max(displacement_grid, initial=0.0)
        ),
        "nearest_center_sampling_max_error_source_px": float(
            0.5 / max(scale_x, 1e-12)
        ),
        "claim_scope": "sampled_map_relative_visible_2dgs_surface",
    }
    return (
        ContributorLabels(
            topk_primitive_ids=remapped_ids,
            topk_weights=remapped_weights,
            pose_w2c=labels.pose_w2c,
        ),
        audit,
    )


def load_contributors_in_radio_coordinates(
    path: Path,
    *,
    legacy_pinhole_as_raw_diagnostic: bool = False,
) -> tuple[ContributorLabels, dict[str, object]]:
    """Load a contributor cache on the frozen raw-RADIO coordinate grid."""

    with np.load(Path(path), allow_pickle=False) as data:
        labels = ContributorLabels(
            topk_primitive_ids=np.asarray(data["topk_ids"], dtype=np.int64),
            topk_weights=np.asarray(data["topk_weights"], dtype=np.float32),
            pose_w2c=np.asarray(data["pose_w2c"], dtype=np.float64),
        )
        if legacy_pinhole_as_raw_diagnostic:
            return labels, {
                "coordinate_contract": LEGACY_COORDINATE_CONTRACT,
                "claim_scope": "diagnostic_only_coordinate_misaligned_control",
                "camera_model_id": int(np.asarray(data["camera_model_id"]).item()),
            }
        return remap_pinhole_contributors_to_raw_grid(
            labels,
            camera_model_id=int(np.asarray(data["camera_model_id"]).item()),
            camera_width=int(np.asarray(data["camera_width"]).item()),
            camera_height=int(np.asarray(data["camera_height"]).item()),
            camera_params=np.asarray(data["camera_params"], dtype=np.float64),
        )


@dataclass(frozen=True)
class PhysicalIncidence:
    primitive_to_parent: sparse.csr_matrix
    primitive_to_child: sparse.csr_matrix

    @classmethod
    def from_physical_map(
        cls, physical: GoalMapletPhysicalMap
    ) -> "PhysicalIncidence":
        primitive_count = int(physical.primitive_ids.size)

        def build(
            offsets: np.ndarray,
            primitive_rows: np.ndarray,
            weights: np.ndarray,
            entity_count: int,
        ) -> sparse.csr_matrix:
            entity_rows = np.repeat(
                np.arange(int(entity_count), dtype=np.int64),
                np.diff(np.asarray(offsets, dtype=np.int64)),
            )
            primitive = np.asarray(primitive_rows, dtype=np.int64)
            value = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
            if entity_rows.shape != primitive.shape or value.shape != primitive.shape:
                raise ValueError("physical incidence arrays differ")
            total = np.bincount(primitive, weights=value, minlength=primitive_count)
            normalized = value / np.maximum(total[primitive], 1e-12)
            return sparse.coo_matrix(
                (normalized, (primitive, entity_rows)),
                shape=(primitive_count, int(entity_count)),
            ).tocsr()

        return cls(
            primitive_to_parent=build(
                physical.membership_offsets,
                physical.membership_primitive_rows,
                physical.membership_weights,
                physical.maplet_ids.size,
            ),
            primitive_to_child=build(
                physical.child_member_offsets,
                physical.child_member_primitive_rows,
                physical.child_member_weights,
                physical.child_parent_rows.size,
            ),
        )


def token_primitive_visibility(
    labels: ContributorLabels,
    physical: GoalMapletPhysicalMap,
    *,
    token_height: int,
    token_width: int,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, object]]:
    """Aggregate raw-grid contributor samples into token×primitive mass."""

    ids = np.asarray(labels.topk_primitive_ids, dtype=np.int64)
    weights = np.maximum(np.asarray(labels.topk_weights, dtype=np.float64), 0.0)
    if ids.ndim != 3 or weights.shape != ids.shape:
        raise ValueError("contributor arrays must have shape (H,W,K)")
    height, width, topk = ids.shape
    if height % int(token_height) or width % int(token_width):
        raise ValueError("contributor grid must divide exactly into RADIO tokens")
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.int64),
        np.arange(width, dtype=np.int64),
        indexing="ij",
    )
    token = (
        (yy * int(token_height) // height) * int(token_width)
        + (xx * int(token_width) // width)
    )
    token = np.broadcast_to(token[..., None], ids.shape).reshape(-1)
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    primitive_ids = np.asarray(physical.primitive_ids, dtype=np.int64)
    order = np.argsort(primitive_ids, kind="stable")
    sorted_ids = primitive_ids[order]
    position = np.searchsorted(sorted_ids, flat_ids)
    safe = np.minimum(position, max(sorted_ids.size - 1, 0))
    recognized = (
        (flat_ids >= 0)
        & (flat_weights > 0.0)
        & (position < sorted_ids.size)
        & (sorted_ids[safe] == flat_ids)
    )
    primitive_row = order[safe[recognized]]
    samples_per_token = (height // int(token_height)) * (
        width // int(token_width)
    )
    scale = 1.0 / float(samples_per_token)
    matrix = sparse.coo_matrix(
        (flat_weights[recognized] * scale, (token[recognized], primitive_row)),
        shape=(int(token_height) * int(token_width), primitive_ids.size),
    ).tocsr()
    all_mass = np.bincount(
        token,
        weights=flat_weights * scale,
        minlength=int(token_height) * int(token_width),
    ).astype(np.float64)
    recognized_mass = np.asarray(matrix.sum(axis=1)).reshape(-1)
    return matrix, all_mass, {
        "top_k_contributors": int(topk),
        "samples_per_radio_token": int(samples_per_token),
        "retained_topk_alpha_mass_mean_per_token": float(np.mean(all_mass)),
        "recognized_physical_primitive_mass_fraction": float(
            np.sum(recognized_mass) / max(float(np.sum(all_mass)), 1e-12)
        ),
    }


def _candidate_recall(
    truth: sparse.csr_matrix,
    candidate_rows: np.ndarray,
    ks: Iterable[int],
    *,
    absolute_denominator: np.ndarray | None = None,
) -> dict[int, tuple[float, float]]:
    """Return ``K -> (conditional recall, absolute retained-mass coverage)``."""

    matrix = truth.tocsr(copy=True)
    matrix.sort_indices()
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[0] != matrix.shape[0]:
        raise ValueError("truth/candidate token counts differ")
    conditional_denominator = np.asarray(matrix.sum(axis=1)).reshape(-1)
    absolute = (
        conditional_denominator
        if absolute_denominator is None
        else np.asarray(absolute_denominator, dtype=np.float64).reshape(-1)
    )
    if absolute.shape != conditional_denominator.shape:
        raise ValueError("absolute token denominator differs")
    # Resolve all candidate values with one CSR-row search.  The previous
    # implementation constructed a SciPy fancy-index matrix for every
    # token×K pair (2304×5×parent/child), which dominated evaluation time.
    selected = np.zeros(candidates.shape, dtype=np.float64)
    for token in range(candidates.shape[0]):
        start, end = int(matrix.indptr[token]), int(matrix.indptr[token + 1])
        truth_rows = matrix.indices[start:end]
        truth_values = matrix.data[start:end]
        candidate = candidates[token]
        valid = (candidate >= 0) & (candidate < matrix.shape[1])
        if not np.any(valid) or truth_rows.size == 0:
            continue
        position = np.searchsorted(truth_rows, candidate[valid])
        safe = np.minimum(position, truth_rows.size - 1)
        matched = truth_rows[safe] == candidate[valid]
        target = np.flatnonzero(valid)[matched]
        selected[token, target] = truth_values[safe[matched]]
    # Candidate rows should be unique, but fail safely if an upstream artifact
    # repeats one: only its first occurrence contributes to cumulative recall.
    for column in range(1, candidates.shape[1]):
        duplicate = np.any(
            candidates[:, column, None] == candidates[:, :column], axis=1
        )
        selected[duplicate, column] = 0.0
    cumulative = np.cumsum(selected, axis=1)
    result: dict[int, tuple[float, float]] = {}
    for requested in ks:
        count = min(max(int(requested), 0), candidates.shape[1])
        retrieved = (
            0.0 if count == 0 else float(np.sum(cumulative[:, count - 1]))
        )
        result[int(requested)] = (
            float(retrieved / max(float(np.sum(conditional_denominator)), 1e-12)),
            float(retrieved / max(float(np.sum(absolute)), 1e-12)),
        )
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.size == 0 or value.shape != weight.shape or float(np.sum(weight)) <= 0.0:
        return float("inf")
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    target = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    row = min(int(np.searchsorted(cumulative, target, side="left")), value.size - 1)
    return float(value[order[row]])


def _primitive_union_for_parents(
    physical: GoalMapletPhysicalMap, parent_ids: np.ndarray
) -> np.ndarray:
    row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    values: list[np.ndarray] = []
    for parent_id in np.asarray(parent_ids, dtype=np.int64).tolist():
        row = row_by_id.get(int(parent_id))
        if row is None:
            continue
        start = int(physical.membership_offsets[row])
        end = int(physical.membership_offsets[row + 1])
        values.append(
            np.asarray(physical.membership_primitive_rows[start:end], dtype=np.int64)
        )
    return np.unique(np.concatenate(values)) if values else np.zeros(0, dtype=np.int64)


def _primitive_union_for_children(
    physical: GoalMapletPhysicalMap, child_rows: np.ndarray
) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in np.asarray(child_rows, dtype=np.int64).tolist():
        if row < 0 or row >= physical.child_parent_rows.size:
            continue
        start = int(physical.child_member_offsets[row])
        end = int(physical.child_member_offsets[row + 1])
        values.append(
            np.asarray(
                physical.child_member_primitive_rows[start:end], dtype=np.int64
            )
        )
    return np.unique(np.concatenate(values)) if values else np.zeros(0, dtype=np.int64)


def _hierarchical_child_rows(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_parents: int,
    children_per_parent: int,
) -> np.ndarray:
    """Allocate fine-surface budget inside globally retrieved parent regions.

    A single global Top-K over thousands of one-metre children fragments the
    budget across a scene.  The hierarchy instead retrieves coarse physical
    regions globally, then ranks children only inside each retained region.
    Selection is pose/GT-free and uses the same frozen full-token evidence as
    the retrieval artifact.
    """

    if int(maximum_parents) <= 0 or int(children_per_parent) <= 0:
        raise ValueError("hierarchical child budgets must be positive")
    scores = aggregate_sparse_token_evidence(
        retrieval.token_xy,
        retrieval.token_child_rows,
        retrieval.token_child_probabilities,
        entity_count=physical.child_parent_rows.size,
        token_height=int(retrieval.metadata["token_height"]),
        token_width=int(retrieval.metadata["token_width"]),
    )
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    chosen: list[np.ndarray] = []
    for parent_id in retrieval.scene_parent_ids[: int(maximum_parents)].tolist():
        parent_row = parent_row_by_id.get(int(parent_id))
        if parent_row is None:
            continue
        rows = np.flatnonzero(physical.child_parent_rows == int(parent_row))
        if rows.size == 0:
            continue
        order = np.lexsort((rows, -scores[rows]))
        selected = rows[order]
        selected = selected[scores[selected] > 0.0][: int(children_per_parent)]
        if selected.size:
            chosen.append(selected.astype(np.int64))
    return np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)


def _parent_primitive_rows_under_surface_fraction(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_surface_fraction: float,
) -> tuple[np.ndarray, int, float]:
    """Take the longest ranked-parent prefix inside a physical-area budget.

    Fixed Top-K is not a fair comparison when two scorers prefer regions of
    different size.  This control charges every newly covered primitive by its
    2DGS ellipse area and stops before the first parent that would exceed the
    fixed map-surface budget.  Membership overlap is charged only once.
    """

    fraction = float(maximum_surface_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("parent surface fraction must be in (0,1]")
    primitive_area = (
        np.pi
        * np.asarray(physical.primitive_scale1, dtype=np.float64)
        * np.asarray(physical.primitive_scale2, dtype=np.float64)
    )
    maximum_area = fraction * float(np.sum(primitive_area))
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    selected = np.zeros((physical.primitive_ids.size,), dtype=bool)
    selected_area = 0.0
    selected_parent_count = 0
    for parent_id in retrieval.scene_parent_ids.tolist():
        parent_row = parent_row_by_id.get(int(parent_id))
        if parent_row is None:
            continue
        start = int(physical.membership_offsets[parent_row])
        end = int(physical.membership_offsets[parent_row + 1])
        members = np.unique(
            np.asarray(physical.membership_primitive_rows[start:end], dtype=np.int64)
        )
        new_members = members[~selected[members]]
        added_area = float(np.sum(primitive_area[new_members]))
        if selected_area + added_area > maximum_area + 1e-12:
            break
        selected[new_members] = True
        selected_area += added_area
        selected_parent_count += 1
    return np.flatnonzero(selected), selected_parent_count, selected_area


def _surface_set_metrics(
    predicted_rows: np.ndarray,
    scene_mass: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    tolerances_m: tuple[float, ...],
) -> dict[str, object]:
    predicted = np.unique(np.asarray(predicted_rows, dtype=np.int64))
    visible = np.flatnonzero(scene_mass > 0.0)
    total_mass = max(float(np.sum(scene_mass)), 1e-12)
    exact = float(np.sum(scene_mass[predicted]) / total_mass) if predicted.size else 0.0
    primitive_area = (
        np.pi
        * np.asarray(physical.primitive_scale1, dtype=np.float64)
        * np.asarray(physical.primitive_scale2, dtype=np.float64)
    )
    visible_set = np.zeros((scene_mass.size,), dtype=bool)
    visible_set[visible] = True
    area_precision = (
        float(
            np.sum(primitive_area[predicted[visible_set[predicted]]])
            / max(float(np.sum(primitive_area[predicted])), 1e-12)
        )
        if predicted.size
        else 0.0
    )
    if predicted.size and visible.size:
        tree = cKDTree(np.asarray(physical.primitive_centers[predicted], dtype=np.float64))
        distance, _ = tree.query(
            np.asarray(physical.primitive_centers[visible], dtype=np.float64), k=1
        )
    else:
        distance = np.full((visible.size,), np.inf, dtype=np.float64)
    visible_weight = scene_mass[visible]
    result: dict[str, object] = {
        "predicted_primitive_count": int(predicted.size),
        "predicted_surface_area_m2": float(np.sum(primitive_area[predicted])),
        "exact_visible_mass_recall": exact,
        "visible_primitive_area_precision": area_precision,
        "weighted_distance_median_m": _weighted_quantile(
            distance, visible_weight, 0.50
        ),
        "weighted_distance_p90_m": _weighted_quantile(distance, visible_weight, 0.90),
        "weighted_distance_p95_m": _weighted_quantile(distance, visible_weight, 0.95),
    }
    for tolerance in tolerances_m:
        result[f"tolerant_visible_mass_recall_{float(tolerance):g}m"] = float(
            np.sum(visible_weight[distance <= float(tolerance)]) / total_mass
        )
    return result


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    predicted = np.asarray(probability, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape:
        raise ValueError("calibration arrays differ")
    value = 0.0
    for index in range(int(bins)):
        low, high = index / bins, (index + 1) / bins
        chosen = (predicted >= low) & (
            predicted < high if index + 1 < bins else predicted <= high
        )
        if np.any(chosen):
            value += float(np.mean(chosen)) * abs(
                float(np.mean(predicted[chosen])) - float(np.mean(truth[chosen]))
            )
    return float(value)


def evaluate_pure_retrieval_query(
    retrieval: PureRadioPhysicalRetrieval,
    pinhole_labels: ContributorLabels,
    physical: GoalMapletPhysicalMap,
    *,
    camera_model_id: int,
    camera_width: int,
    camera_height: int,
    camera_params: np.ndarray,
    incidence: PhysicalIncidence | None = None,
    ks: tuple[int, ...] = (1, 5, 20, 32, 64),
    surface_ks: tuple[int, ...] | None = None,
    tolerances_m: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    """Evaluate one pose-free result; GT is opened only in this function."""

    if retrieval.physical_map_sha256 != physical.content_sha256:
        raise ValueError("retrieval/physical-map lineage differs")
    raw_labels, coordinate_audit = remap_pinhole_contributors_to_raw_grid(
        pinhole_labels,
        camera_model_id=int(camera_model_id),
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        camera_params=np.asarray(camera_params),
    )
    height = int(retrieval.metadata["token_height"])
    width = int(retrieval.metadata["token_width"])
    token_primitive, token_total_mass, sampling_audit = token_primitive_visibility(
        raw_labels, physical, token_height=height, token_width=width
    )
    physical_incidence = (
        PhysicalIncidence.from_physical_map(physical)
        if incidence is None
        else incidence
    )
    expected_shape = (physical.primitive_ids.size, physical.maplet_ids.size)
    if physical_incidence.primitive_to_parent.shape != expected_shape:
        raise ValueError("parent incidence does not match physical map")
    if physical_incidence.primitive_to_child.shape != (
        physical.primitive_ids.size,
        physical.child_parent_rows.size,
    ):
        raise ValueError("child incidence does not match physical map")
    token_parent = (
        token_primitive @ physical_incidence.primitive_to_parent
    ).tocsr()
    token_child = (
        token_primitive @ physical_incidence.primitive_to_child
    ).tocsr()
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    parent_candidates = np.asarray(
        [
            [parent_row_by_id.get(int(value), -1) for value in row]
            for row in retrieval.token_parent_ids.tolist()
        ],
        dtype=np.int64,
    )
    parent_recall = _candidate_recall(
        token_parent,
        parent_candidates,
        ks,
        absolute_denominator=token_total_mass,
    )
    child_recall = _candidate_recall(
        token_child,
        retrieval.token_child_rows,
        ks,
        absolute_denominator=token_total_mass,
    )
    scene_mass = np.asarray(token_primitive.sum(axis=0)).reshape(-1)
    total_visible_mass = max(float(np.sum(scene_mass)), 1e-12)
    parent_ceiling = float(
        np.sum(np.asarray(token_parent.sum(axis=0)).reshape(-1)) / total_visible_mass
    )
    child_ceiling = float(
        np.sum(np.asarray(token_child.sum(axis=0)).reshape(-1)) / total_visible_mass
    )
    surface: dict[str, object] = {}
    requested_surface_ks = ks if surface_ks is None else surface_ks
    if not requested_surface_ks or any(int(value) <= 0 for value in requested_surface_ks):
        raise ValueError("surface_ks must contain positive retrieval cutoffs")
    for requested in requested_surface_ks:
        parent_rows = _primitive_union_for_parents(
            physical, retrieval.scene_parent_ids[: int(requested)]
        )
        child_rows = _primitive_union_for_children(
            physical, retrieval.scene_child_rows[: int(requested)]
        )
        surface[f"parent_top{int(requested)}"] = _surface_set_metrics(
            parent_rows, scene_mass, physical, tolerances_m=tolerances_m
        )
        surface[f"child_top{int(requested)}"] = _surface_set_metrics(
            child_rows, scene_mass, physical, tolerances_m=tolerances_m
        )
    area_parent_rows, area_parent_count, area_used = (
        _parent_primitive_rows_under_surface_fraction(
            retrieval, physical, maximum_surface_fraction=0.20
        )
    )
    surface["parent_area20pct"] = _surface_set_metrics(
        area_parent_rows, scene_mass, physical, tolerances_m=tolerances_m
    )
    surface["parent_area20pct"]["selected_parent_count"] = int(area_parent_count)
    surface["parent_area20pct"]["charged_surface_area_m2"] = float(area_used)
    surface["parent_area20pct"]["selection_semantics"] = (
        "longest_ranked_parent_prefix_with_unique_2dgs_surface_area_le_20pct_v1"
    )
    hierarchical_child_rows = _hierarchical_child_rows(
        retrieval,
        physical,
        maximum_parents=16,
        children_per_parent=4,
    )
    surface["hierarchical_child_parent16_x4"] = _surface_set_metrics(
        _primitive_union_for_children(physical, hierarchical_child_rows),
        scene_mass,
        physical,
        tolerances_m=tolerances_m,
    )
    surface["hierarchical_child_parent16_x4"]["selected_child_count"] = int(
        hierarchical_child_rows.size
    )
    surface["hierarchical_child_parent16_x4"]["selection_semantics"] = (
        "global_top16_parents_then_top4_children_per_parent_v1"
    )
    representable_mass = np.asarray(token_parent.sum(axis=1)).reshape(-1)
    target_out_of_map = 1.0 - np.divide(
        representable_mass,
        token_total_mass,
        out=np.zeros_like(representable_mass),
        where=token_total_mass > 1e-12,
    )
    target_out_of_map[token_total_mass <= 1e-12] = 1.0
    predicted_out_of_map = np.asarray(
        retrieval.token_out_of_map_probabilities, dtype=np.float64
    )

    selected = retrieval.scene_child_rows[: min(64, retrieval.scene_child_rows.size)]
    scene_child_truth = np.asarray(token_child.sum(axis=0)).reshape(-1)
    correct = selected[scene_child_truth[selected] > 0.0] if selected.size else selected
    centers = np.asarray(physical.child_centers[correct], dtype=np.float64)
    camera_center = camera_center_from_w2c(pinhole_labels.pose_w2c)
    spread = (
        float(np.max(np.linalg.norm(centers[:, None] - centers[None], axis=2)))
        if centers.shape[0] >= 2
        else 0.0
    )
    bearings = centers - camera_center[None] if centers.size else np.zeros((0, 3))
    bearing_rank = (
        int(
            np.linalg.matrix_rank(
                bearings - np.mean(bearings, axis=0), tol=0.10
            )
        )
        if bearings.shape[0] >= 2
        else 0
    )
    normal_rank = (
        int(np.linalg.matrix_rank(physical.child_normals[correct], tol=0.20))
        if correct.size
        else 0
    )
    configuration_sufficient = bool(
        correct.size >= 3 and spread >= 0.5 and (bearing_rank >= 2 or normal_rank >= 2)
    )
    token_parent_metrics = {
        f"conditional_recall_at_{key}": parent_recall[key][0] for key in ks
    }
    token_parent_metrics.update(
        {
            f"absolute_visible_mass_coverage_at_{key}": parent_recall[key][1]
            for key in ks
        }
    )
    token_child_metrics = {
        f"conditional_recall_at_{key}": child_recall[key][0] for key in ks
    }
    token_child_metrics.update(
        {
            f"absolute_visible_mass_coverage_at_{key}": child_recall[key][1]
            for key in ks
        }
    )
    return {
        "image_id": retrieval.image_id,
        "coordinate_audit": coordinate_audit,
        "sampling_audit": sampling_audit,
        "visible_topk_contributor_mass": float(np.sum(scene_mass)),
        "parent_representable_visible_mass_ceiling": parent_ceiling,
        "child_representable_visible_mass_ceiling": child_ceiling,
        "token_parent": token_parent_metrics,
        "token_child": token_child_metrics,
        "surface_sets": surface,
        "null_calibration": {
            "out_of_map_brier": float(
                np.mean(np.square(predicted_out_of_map - target_out_of_map))
            ),
            "out_of_map_ece": _ece(predicted_out_of_map, target_out_of_map),
            "target_out_of_map_mean": float(np.mean(target_out_of_map)),
            "predicted_out_of_map_mean": float(np.mean(predicted_out_of_map)),
        },
        "configuration_at_64": {
            "correct_child_count": int(correct.size),
            "correct_child_center_spread_m": spread,
            "bearing_rank": bearing_rank,
            "normal_rank": normal_rank,
            "sufficient": configuration_sufficient,
        },
        "claim_scope": {
            "retrieval_only": True,
            "uses_gt_only_after_pose_free_retrieval": True,
            "metric_is_sampled_map_relative_visible_surface": True,
            "metric_is_localization_success": False,
            "metric_is_real_world_geometry_truth": False,
        },
    }
