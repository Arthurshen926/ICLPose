"""Low-cost pose-free parent-support layout guide for factorized proposals.

This is deliberately a coarse guide, not a final q_pose estimator.  It uses
only a frozen position/orientation factorization, the query's soft RADIO
parent posterior, physical parent rectangles, and camera intrinsics.  It does
not accept a query pose/ground truth, render Gaussian appearance, establish
hard 2D--3D correspondences, or run PnP.

For each candidate factor pair, visible parent rectangles are projected to a
binary RADIO-token footprint.  The score is the cosine/Bhattacharyya affinity
between ``sqrt(query parent probability)`` and ``sqrt(projected footprint)``.
The query norm is computed once over the complete retained posterior and is
never renormalized by candidate visibility.  The score is symmetric in the two
non-negative layouts and is bounded in ``[0,1]`` by Cauchy--Schwarz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval, all_radio_token_coordinates


SCORE_SEMANTICS = (
    "fixed_query_denominator_symmetric_sqrt_parent_token_footprint_affinity_v1"
)
NORMAL_CONTRACT = (
    "unsigned_parent_normal_absolute_incidence_no_mapping_camera_sign_v1"
)
TOKEN_FOOTPRINT_PHASE = (
    "raw_pixel_index_edge_aligned_u_times_token_width_over_image_width_v1"
)


@dataclass(frozen=True)
class ParentLayoutCamera:
    """Pinhole or SIMPLE_RADIAL intrinsics in the query image coordinates."""

    model_id: int
    width: int
    height: int
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        model = int(self.model_id)
        width, height = int(self.width), int(self.height)
        params = tuple(float(value) for value in self.params)
        expected = {0: 3, 1: 4, 2: 4}.get(model)
        if (
            expected is None or len(params) != expected
            or width <= 0 or height <= 0
            or not np.all(np.isfinite(params))
        ):
            raise ValueError("parent layout guide camera intrinsics are invalid")
        if model in (0, 2):
            focal, cx, cy = params[:3]
            fx, fy = focal, focal
        else:
            fx, fy, cx, cy = params
        if fx <= 0.0 or fy <= 0.0 or not (0.0 <= cx <= width) or not (0.0 <= cy <= height):
            raise ValueError("parent layout guide focal length/principal point is invalid")
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "params", params)

    @property
    def fx_fy_cx_cy_k1(self) -> tuple[float, float, float, float, float]:
        if self.model_id == 0:
            focal, cx, cy = self.params
            return focal, focal, cx, cy, 0.0
        if self.model_id == 1:
            fx, fy, cx, cy = self.params
            return fx, fy, cx, cy, 0.0
        focal, cx, cy, radial = self.params
        return focal, focal, cx, cy, radial


@dataclass(frozen=True)
class ParentSupportLayoutGuideResult:
    score_semantics: str
    total_factor_pair_count: int
    scored_orientation_count: int
    selected_query_parent_ids: np.ndarray
    selected_query_parent_probability_mass: np.ndarray
    complete_query_parent_probability_mass: float
    selected_query_parent_probability_mass_total: float
    top_scores: np.ndarray
    top_position_factor_indices: np.ndarray
    top_position_seed_indices: np.ndarray
    top_position_offset_indices: np.ndarray
    top_orientation_factor_indices: np.ndarray
    top_orientation_source_candidate_ranks: np.ndarray
    top_visible_parent_counts: np.ndarray
    top_front_facing_parent_counts: np.ndarray
    top_positive_depth_parent_counts: np.ndarray
    top_center_in_image_parent_counts: np.ndarray
    top_projected_token_footprint_mass: np.ndarray
    top_sqrt_overlap_mass: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.top_scores).reshape(-1).size
        fields = (
            "top_position_factor_indices", "top_position_seed_indices",
            "top_position_offset_indices", "top_orientation_factor_indices",
            "top_orientation_source_candidate_ranks", "top_visible_parent_counts",
            "top_front_facing_parent_counts", "top_positive_depth_parent_counts",
            "top_center_in_image_parent_counts", "top_projected_token_footprint_mass",
            "top_sqrt_overlap_mass",
        )
        if any(np.asarray(getattr(self, name)).reshape(-1).size != count for name in fields):
            raise ValueError("parent layout guide Top-K arrays differ")
        pairs = np.stack([
            np.asarray(self.top_position_factor_indices, dtype=np.int64),
            np.asarray(self.top_orientation_factor_indices, dtype=np.int64),
        ], axis=1)
        if np.unique(pairs, axis=0).shape[0] != count:
            raise ValueError("parent layout guide Top-K factor pairs are not distinct")
        scores = np.asarray(self.top_scores, dtype=np.float64)
        if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0 + 1e-12)):
            raise ValueError("parent layout guide scores are not bounded")


def adapt_ranked_factor_pairs_for_exact_scorer(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    orientation_valid: np.ndarray,
    orientation_source_candidate_ranks: np.ndarray,
    ranked_position_factor_indices: np.ndarray,
    ranked_orientation_factor_indices: np.ndarray,
    ranked_orientation_source_candidate_ranks: np.ndarray,
    ranked_guide_scores: np.ndarray,
    *,
    maximum_pairs: int,
) -> dict[str, np.ndarray]:
    """Materialize only selected factor pairs for an exact pose scorer.

    The adapter preserves guide order and provenance.  If the position factor
    stores camera center ``C`` and the orientation factor stores ``R_w2c``, the
    only constructed pose is ``[R_w2c | -R_w2c C]``.  No unselected Cartesian
    pair is ever allocated, and there is no query-pose or label input.
    """

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    valid = np.asarray(orientation_valid)
    source = np.asarray(orientation_source_candidate_ranks)
    ranked_position = np.asarray(ranked_position_factor_indices, dtype=np.int64).reshape(-1)
    ranked_orientation = np.asarray(
        ranked_orientation_factor_indices, dtype=np.int64,
    ).reshape(-1)
    ranked_source = np.asarray(
        ranked_orientation_source_candidate_ranks, dtype=np.int64,
    ).reshape(-1)
    ranked_score = np.asarray(ranked_guide_scores, dtype=np.float64).reshape(-1)
    rank_count = ranked_position.size
    if (
        position.ndim != 3 or position.shape[2] != 3
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or valid.dtype != np.bool_ or valid.shape != rotation.shape[:1]
        or source.shape != valid.shape or source.dtype.kind not in "iu"
        or ranked_orientation.shape != (rank_count,)
        or ranked_source.shape != (rank_count,) or ranked_score.shape != (rank_count,)
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or np.any(~np.isfinite(ranked_score))
        or np.any((ranked_score < 0.0) | (ranked_score > 1.0 + 1.0e-12))
        or int(maximum_pairs) <= 0 or rank_count == 0
    ):
        raise ValueError("exact factor-pair adapter inputs differ")
    budget = min(int(maximum_pairs), rank_count)
    position_flat = position.reshape(-1, 3)
    selected_position = ranked_position[:budget]
    selected_orientation = ranked_orientation[:budget]
    if (
        np.any((selected_position < 0) | (selected_position >= position_flat.shape[0]))
        or np.any((selected_orientation < 0) | (selected_orientation >= rotation.shape[0]))
        or np.any(~valid[selected_orientation])
        or not np.array_equal(source[selected_orientation], ranked_source[:budget])
    ):
        raise ValueError("exact factor-pair adapter indices/provenance differ")
    pairs = np.stack([selected_position, selected_orientation], axis=1)
    if np.unique(pairs, axis=0).shape[0] != budget:
        raise ValueError("exact factor-pair adapter requires distinct ranked pairs")
    selected_rotation = rotation[selected_orientation]
    selected_center = position_flat[selected_position]
    poses = np.broadcast_to(np.eye(4, dtype=np.float64), (budget, 4, 4)).copy()
    poses[:, :3, :3] = selected_rotation
    poses[:, :3, 3] = -np.einsum("bij,bj->bi", selected_rotation, selected_center)
    offsets_per_seed = int(position.shape[1])
    return {
        "candidate_poses_w2c": poses,
        "position_factor_indices": selected_position.copy(),
        "position_seed_indices": (selected_position // offsets_per_seed).copy(),
        "position_offset_indices": (selected_position % offsets_per_seed).copy(),
        "orientation_factor_indices": selected_orientation.copy(),
        "orientation_source_candidate_ranks": ranked_source[:budget].copy(),
        "parent_layout_guide_scores": ranked_score[:budget].copy(),
    }


def _query_parent_layout(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_query_parents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height = int(retrieval.metadata["token_height"])
    width = int(retrieval.metadata["token_width"])
    if not np.array_equal(
        retrieval.token_xy, all_radio_token_coordinates(height, width),
    ):
        raise ValueError("parent layout guide requires the complete row-major token grid")
    ids = np.asarray(retrieval.token_parent_ids, dtype=np.int64)
    probabilities = np.asarray(retrieval.token_parent_probabilities, dtype=np.float64)
    if ids.shape != probabilities.shape or np.any(~np.isfinite(probabilities)):
        raise ValueError("query parent posterior arrays differ")
    map_ids = np.asarray(physical.maplet_ids, dtype=np.int64)
    map_order = np.argsort(map_ids, kind="stable")
    sorted_ids = map_ids[map_order]
    flat_ids = ids.reshape(-1)
    flat_probability = probabilities.reshape(-1)
    valid = (flat_ids >= 0) & (flat_probability > 0.0)
    positions = np.searchsorted(sorted_ids, flat_ids[valid])
    safe = np.minimum(positions, max(sorted_ids.size - 1, 0))
    matched = (
        (positions < sorted_ids.size)
        & (sorted_ids.size > 0)
        & (sorted_ids[safe] == flat_ids[valid])
    ) if sorted_ids.size else np.zeros(positions.shape, dtype=bool)
    if np.any(~matched):
        raise ValueError("query posterior contains an unknown physical parent ID")
    # A duplicate ID inside one token would make sparse posterior mass
    # representation-dependent.  Refuse it instead of silently summing.
    for token_ids, token_probability in zip(ids, probabilities):
        positive = token_ids[token_probability > 0.0]
        positive = positive[positive >= 0]
        if np.unique(positive).size != positive.size:
            raise ValueError("query posterior repeats a parent within one token")
    parent_rows = map_order[positions]
    total_by_parent = np.bincount(
        parent_rows, weights=flat_probability[valid], minlength=map_ids.size,
    )
    positive_rows = np.flatnonzero(total_by_parent > 0.0)
    if positive_rows.size == 0:
        raise ValueError("query parent posterior has no physical mass")
    budget = min(int(maximum_query_parents), int(positive_rows.size))
    if budget <= 0:
        raise ValueError("maximum_query_parents must be positive")
    order = np.lexsort((map_ids[positive_rows], -total_by_parent[positive_rows]))
    selected_rows = positive_rows[order[:budget]]
    selected_index = np.full((map_ids.size,), -1, dtype=np.int64)
    selected_index[selected_rows] = np.arange(selected_rows.size, dtype=np.int64)
    token_indices = np.repeat(np.arange(ids.shape[0], dtype=np.int64), ids.shape[1])
    entry_parent_rows = np.full(flat_ids.shape, -1, dtype=np.int64)
    entry_parent_rows[valid] = parent_rows
    selected_entry = valid & (selected_index[np.maximum(entry_parent_rows, 0)] >= 0)
    layout = np.zeros((selected_rows.size, height * width), dtype=np.float64)
    np.add.at(
        layout,
        (
            selected_index[entry_parent_rows[selected_entry]],
            token_indices[selected_entry],
        ),
        flat_probability[selected_entry],
    )
    if np.any(layout > 1.0 + 1.0e-6):
        raise ValueError("query parent probability exceeds one after sparse expansion")
    sqrt_layout = np.sqrt(np.clip(layout, 0.0, 1.0)).reshape(
        selected_rows.size, height, width,
    )
    integral = np.pad(
        np.cumsum(np.cumsum(sqrt_layout, axis=1), axis=2),
        ((0, 0), (1, 0), (1, 0)),
    )
    return (
        selected_rows,
        total_by_parent[selected_rows],
        integral,
        float(np.sum(flat_probability[valid])),
    )


def _parent_rectangles(
    physical: GoalMapletPhysicalMap,
    parent_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(parent_rows, dtype=np.int64)
    center = np.asarray(physical.maplet_centers[rows], dtype=np.float64)
    frames = np.asarray(physical.maplet_frames[rows], dtype=np.float64)
    extents = np.maximum(
        np.asarray(physical.maplet_extents[rows, :2], dtype=np.float64), 1.0e-4,
    )
    signs = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    corners = (
        center[:, None, :]
        + signs[None, :, 0, None] * extents[:, None, 0, None] * frames[:, None, 0, :]
        + signs[None, :, 1, None] * extents[:, None, 1, None] * frames[:, None, 1, :]
    )
    return (
        center, corners,
        np.asarray(physical.maplet_normals[rows], dtype=np.float64),
        np.asarray(physical.maplet_ids[rows], dtype=np.int64),
    )


def _project_camera_points(
    points_camera: np.ndarray,
    camera: ParentLayoutCamera,
) -> np.ndarray:
    value = np.asarray(points_camera, dtype=np.float64)
    depth = value[..., 2]
    safe = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
    x, y = value[..., 0] / safe, value[..., 1] / safe
    fx, fy, cx, cy, radial = camera.fx_fy_cx_cy_k1
    scale = 1.0 + float(radial) * (x * x + y * y)
    return np.stack([fx * x * scale + cx, fy * y * scale + cy], axis=-1)


def projected_image_bounds_to_radio_token_footprint(
    low_xy_pixels: np.ndarray,
    high_xy_pixels: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
    finite: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert projected bounds to the existing PFIR token-cell phase.

    PFIR labels treat image-array coordinate zero as the first pixel-cell edge
    and map token-cell boundaries by ``x * image_width / token_width``.  The
    inverse used here is therefore ``u * token_width / image_width`` with no
    ad-hoc half-token or half-pixel offset.  A footprint contains every token
    cell intersected by the clipped projected rectangle.
    """

    low = np.asarray(low_xy_pixels, dtype=np.float64)
    high = np.asarray(high_xy_pixels, dtype=np.float64)
    if (
        low.shape != high.shape or low.shape[-1:] != (2,)
        or int(image_width) <= 0 or int(image_height) <= 0
        or int(token_width) <= 0 or int(token_height) <= 0
    ):
        raise ValueError("projected image/token footprint bounds differ")
    finite_mask = (
        np.all(np.isfinite(low) & np.isfinite(high), axis=-1)
        if finite is None else np.asarray(finite, dtype=bool)
    )
    if finite_mask.shape != low.shape[:-1]:
        raise ValueError("projected image/token footprint finite mask differs")
    scale = np.asarray([
        int(token_width) / float(image_width),
        int(token_height) / float(image_height),
    ])
    safe_low = np.where(
        finite_mask[..., None], np.clip(low * scale, -1.0e12, 1.0e12), 0.0,
    )
    safe_high = np.where(
        finite_mask[..., None], np.clip(high * scale, -1.0e12, 1.0e12), 0.0,
    )
    x0 = np.clip(np.floor(safe_low[..., 0]).astype(np.int64), 0, int(token_width))
    y0 = np.clip(np.floor(safe_low[..., 1]).astype(np.int64), 0, int(token_height))
    x1 = np.clip(np.ceil(safe_high[..., 0]).astype(np.int64), 0, int(token_width))
    y1 = np.clip(np.ceil(safe_high[..., 1]).astype(np.int64), 0, int(token_height))
    return x0, y0, x1, y1


def score_parent_support_layout_guide(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    orientation_valid: np.ndarray,
    orientation_source_candidate_ranks: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    camera: ParentLayoutCamera,
    *,
    maximum_query_parents: int = 64,
    topk: int = 128,
    candidate_batch_size: int = 256,
    minimum_depth_m: float = 0.05,
    minimum_front_incidence: float = 0.02,
) -> ParentSupportLayoutGuideResult:
    """Rank distinct position/orientation factors by soft parent layout."""

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    valid = np.asarray(orientation_valid)
    source_rank = np.asarray(orientation_source_candidate_ranks)
    if (
        position.ndim != 3 or position.shape[2] != 3
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or valid.dtype != np.bool_ or valid.shape != rotation.shape[:1]
        or source_rank.shape != valid.shape or source_rank.dtype.kind not in "iu"
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or int(topk) <= 0 or int(candidate_batch_size) <= 0
        or float(minimum_depth_m) <= 0.0
    ):
        raise ValueError("parent layout guide factor arrays/configuration differ")
    orientation_slots = np.flatnonzero(valid)
    if orientation_slots.size == 0:
        raise ValueError("parent layout guide has no valid orientation")
    if np.unique(source_rank[orientation_slots]).size != orientation_slots.size:
        raise ValueError("parent layout guide orientation source ranks repeat")
    if retrieval.physical_map_sha256 != physical.content_sha256:
        raise ValueError("query retrieval and physical map lineage differ")
    height = int(retrieval.metadata["token_height"])
    width = int(retrieval.metadata["token_width"])
    selected_rows, query_parent_mass, query_integral, query_mass = _query_parent_layout(
        retrieval, physical, maximum_query_parents=int(maximum_query_parents),
    )
    parent_center, parent_corners, parent_normal, parent_ids = (
        _parent_rectangles(physical, selected_rows)
    )
    parent_count = int(parent_ids.size)
    position_flat = position.reshape(-1, 3)
    offsets_per_seed = int(position.shape[1])
    orientation_count = int(orientation_slots.size)
    total = int(position_flat.shape[0] * orientation_count)
    scores = np.zeros((total,), dtype=np.float64)
    visible_counts = np.zeros((total,), dtype=np.int16)
    front_counts = np.zeros((total,), dtype=np.int16)
    depth_counts = np.zeros((total,), dtype=np.int16)
    center_image_counts = np.zeros((total,), dtype=np.int16)
    footprint_mass = np.zeros((total,), dtype=np.float64)
    overlap_mass = np.zeros((total,), dtype=np.float64)
    parent_index = np.arange(parent_count, dtype=np.int64)[None]

    for start in range(0, total, int(candidate_batch_size)):
        end = min(start + int(candidate_batch_size), total)
        pair_index = np.arange(start, end, dtype=np.int64)
        position_index = pair_index // orientation_count
        orientation_slot = orientation_slots[pair_index % orientation_count]
        candidate_center = position_flat[position_index]
        candidate_rotation = rotation[orientation_slot]
        center_delta = parent_center[None] - candidate_center[:, None]
        center_camera = np.einsum("bij,bpj->bpi", candidate_rotation, center_delta)
        corner_delta = parent_corners[None] - candidate_center[:, None, None]
        corner_camera = np.einsum("bij,bpkj->bpki", candidate_rotation, corner_delta)
        view = candidate_center[:, None] - parent_center[None]
        view /= np.maximum(np.linalg.norm(view, axis=2, keepdims=True), 1.0e-12)
        incidence = np.sum(view * parent_normal[None], axis=2)
        # Normal sign must not carry mapping-camera orientation into a held
        # route.  The physical map's normal is an unsigned tangent-plane axis;
        # only grazing incidence is rejected.
        front = np.abs(incidence) >= float(minimum_front_incidence)
        positive_depth = (
            (center_camera[..., 2] > float(minimum_depth_m))
            & np.all(corner_camera[..., 2] > float(minimum_depth_m), axis=2)
        )
        center_xy = _project_camera_points(center_camera, camera)
        corner_xy = _project_camera_points(corner_camera, camera)
        all_xy = np.concatenate([center_xy[:, :, None, :], corner_xy], axis=2)
        finite = np.all(np.isfinite(all_xy), axis=(2, 3))
        low = np.min(all_xy, axis=2)
        high = np.max(all_xy, axis=2)
        x0, y0, x1, y1 = projected_image_bounds_to_radio_token_footprint(
            low, high,
            image_width=camera.width, image_height=camera.height,
            token_width=width, token_height=height, finite=finite,
        )
        nonempty = (x1 > x0) & (y1 > y0)
        center_in_image = (
            (center_xy[..., 0] >= 0.0) & (center_xy[..., 0] < camera.width)
            & (center_xy[..., 1] >= 0.0) & (center_xy[..., 1] < camera.height)
        )
        visible = front & positive_depth & finite & nonempty
        area = ((x1 - x0) * (y1 - y0)).astype(np.float64)
        area = np.where(visible, area, 0.0)
        local_overlap = (
            query_integral[parent_index, y1, x1]
            - query_integral[parent_index, y0, x1]
            - query_integral[parent_index, y1, x0]
            + query_integral[parent_index, y0, x0]
        )
        local_overlap = np.where(visible, np.maximum(local_overlap, 0.0), 0.0)
        map_mass = np.sum(area, axis=1)
        overlap = np.sum(local_overlap, axis=1)
        denominator = np.sqrt(float(query_mass) * map_mass)
        scores[start:end] = np.divide(
            overlap, denominator, out=np.zeros_like(overlap),
            where=denominator > 1.0e-12,
        )
        scores[start:end] = np.clip(scores[start:end], 0.0, 1.0)
        visible_counts[start:end] = np.sum(visible, axis=1).astype(np.int16)
        front_counts[start:end] = np.sum(front, axis=1).astype(np.int16)
        depth_counts[start:end] = np.sum(positive_depth, axis=1).astype(np.int16)
        center_image_counts[start:end] = np.sum(
            visible & center_in_image, axis=1,
        ).astype(np.int16)
        footprint_mass[start:end] = map_mass
        overlap_mass[start:end] = overlap

    all_pair = np.arange(total, dtype=np.int64)
    all_position = all_pair // orientation_count
    all_orientation_slot = orientation_slots[all_pair % orientation_count]
    order = np.lexsort((all_orientation_slot, all_position, -scores))
    selected = order[: min(int(topk), total)]
    selected_position = all_position[selected]
    selected_orientation = all_orientation_slot[selected]
    return ParentSupportLayoutGuideResult(
        score_semantics=SCORE_SEMANTICS,
        total_factor_pair_count=total,
        scored_orientation_count=orientation_count,
        selected_query_parent_ids=parent_ids,
        selected_query_parent_probability_mass=query_parent_mass,
        complete_query_parent_probability_mass=float(query_mass),
        selected_query_parent_probability_mass_total=float(np.sum(query_parent_mass)),
        top_scores=scores[selected],
        top_position_factor_indices=selected_position,
        top_position_seed_indices=selected_position // offsets_per_seed,
        top_position_offset_indices=selected_position % offsets_per_seed,
        top_orientation_factor_indices=selected_orientation,
        top_orientation_source_candidate_ranks=source_rank[selected_orientation],
        top_visible_parent_counts=visible_counts[selected],
        top_front_facing_parent_counts=front_counts[selected],
        top_positive_depth_parent_counts=depth_counts[selected],
        top_center_in_image_parent_counts=center_image_counts[selected],
        top_projected_token_footprint_mass=footprint_mass[selected],
        top_sqrt_overlap_mass=overlap_mass[selected],
    )
