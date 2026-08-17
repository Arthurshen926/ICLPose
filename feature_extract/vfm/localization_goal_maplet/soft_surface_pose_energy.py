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

from .pure_retrieval import PureRadioPhysicalRetrieval, all_radio_token_coordinates


@dataclass(frozen=True)
class SoftSurfacePoseEnergy:
    combined_score: float
    support_layout_score: float
    canonical_radio_score: float
    effective_query_mass: float
    rendered_visible_fraction: float
    rendered_feature_fraction: float


@dataclass(frozen=True)
class BidirectionalSoftSurfacePoseEnergy:
    combined_score: float
    child_overlap_score: float
    child_coupled_radio_score: float
    mean_child_overlap: float
    query_null_mass: float
    rendered_null_mass: float
    coupled_feature_mass: float
    coupled_feature_agreement: float
    rendered_child_tail_mass: float
    rendered_unassigned_geometry_mass: float
    rendered_background_mass: float
    rendered_canonical_field_missing_mass: float
    rendered_payload_excluded_mass: float


@dataclass(frozen=True)
class HierarchicalSpatialSoftSurfacePoseEnergy:
    combined_score: float
    hierarchical_identity_score: float
    parent_support_score: float
    child_precision_score: float
    child_coupled_radio_score: float
    mean_parent_overlap: float
    mean_child_overlap: float
    coupled_feature_mass: float
    coupled_feature_agreement: float
    spatial_kernel: str
    query_reliability_semantics: str
    effective_query_reliability: float


@dataclass(frozen=True)
class SoftSurfaceOverlapLadder:
    parent_overlap: float
    child_overlap_radius2: float
    child_overlap_radius1: float
    child_overlap_radius0: float
    query_reliability_semantics: str


def query_only_pose_reliability_weights(
    retrieval: PureRadioPhysicalRetrieval,
    *,
    semantics: str = "child_mass_entropy_background_v1",
) -> np.ndarray:
    """Return non-negative pose-independent token reliability.

    The default combines retained child mass, posterior concentration and
    out-of-map probability.  Every input is fixed by the query retrieval; no
    rendered visibility or candidate pose can change a token's weight.
    """

    child = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    if child.ndim != 2 or np.any(~np.isfinite(child)) or np.any(child < 0.0):
        raise ValueError("query child probabilities are invalid")
    count = int(child.shape[0])
    if semantics == "uniform_v1":
        return np.ones((count,), dtype=np.float64)
    if semantics != "child_mass_entropy_background_v1":
        raise ValueError("unknown query-only reliability semantics")
    support = np.sum(child, axis=1)
    if np.any(support > 1.0 + 2e-5):
        raise ValueError("query child probabilities exceed unit mass")
    conditional = np.divide(
        child, np.maximum(support[:, None], 1e-12),
        out=np.zeros_like(child), where=support[:, None] > 0.0,
    )
    entropy = -np.sum(
        np.where(conditional > 0.0, conditional * np.log(np.maximum(conditional, 1e-12)), 0.0),
        axis=1,
    )
    entropy /= np.log(max(int(child.shape[1]), 2))
    concentration = np.clip(1.0 - entropy, 0.0, 1.0)
    background = np.asarray(
        retrieval.token_out_of_map_probabilities, dtype=np.float64
    ).reshape(-1)
    if (
        background.shape != (count,) or np.any(~np.isfinite(background))
        or np.any(background < 0.0) or np.any(background > 1.0 + 2e-5)
    ):
        raise ValueError("query out-of-map probabilities are invalid")
    return np.clip(
        support * (0.25 + 0.75 * concentration) * (1.0 - np.clip(background, 0.0, 1.0)),
        0.0, 1.0,
    )


def _fixed_spatial_kernel(radius: int) -> tuple[tuple[int, int, float], ...]:
    value = int(radius)
    if value < 0 or value > 3:
        raise ValueError("spatial kernel radius must lie in [0,3]")
    if value == 0:
        return ((0, 0, 1.0),)
    if value == 1:
        return (
            (0, 0, 0.5), (-1, 0, 0.125), (1, 0, 0.125),
            (0, -1, 0.125), (0, 1, 0.125),
        )
    sigma = float(value) / 1.5
    rows = []
    for dy in range(-value, value + 1):
        for dx in range(-value, value + 1):
            rows.append((dx, dy, float(np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)))))
    normalizer = sum(row[2] for row in rows)
    return tuple((dx, dy, weight / normalizer) for dx, dy, weight in rows)


def _query_weighted_mean(value: np.ndarray, reliability: np.ndarray) -> float:
    atom = np.asarray(value, dtype=np.float64).reshape(-1)
    weight = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if atom.shape != weight.shape:
        raise ValueError("query reliability differs from score atoms")
    total = float(np.sum(weight))
    if total <= 1e-12:
        return -1.0
    return float(np.sum(weight * atom) / total)


def score_hierarchical_spatial_soft_surface_pose_energy(
    query_feature: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    rendered,
    *,
    child_to_parent_ids: np.ndarray,
    radio_weight: float = 0.5,
    spatial_kernel_radius: int = 1,
    query_reliability_semantics: str = "child_mass_entropy_background_v1",
) -> HierarchicalSpatialSoftSurfacePoseEnergy:
    """Parent→child score with a fixed, pose-independent spatial kernel.

    The kernel is *not* renormalized at borders: center has weight 1/2 and the
    four cardinal neighbours 1/8 each.  Off-grid/missing support stays at the
    failure floor, so disappearing evidence cannot improve the score.
    """

    alpha = float(radio_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("radio_weight must lie in [0,1]")
    query = np.asarray(query_feature, dtype=np.float32)
    if query.ndim != 3:
        raise ValueError("query feature must have shape [C,H,W]")
    channels, height, width = query.shape
    if not np.array_equal(
        np.asarray(retrieval.token_xy), all_radio_token_coordinates(height, width)
    ):
        raise ValueError("retrieval token order is not exact row-major (x,y)")
    child_parent = np.asarray(child_to_parent_ids, dtype=np.int64).reshape(-1)
    map_rows = np.asarray(rendered.child_rows, dtype=np.int64).reshape(height * width, -1)
    map_mass = np.asarray(rendered.child_weights, dtype=np.float64).reshape(height * width, -1)
    map_feature = np.asarray(rendered.child_features, dtype=np.float32).reshape(
        height * width, map_rows.shape[1], channels
    )
    map_valid = np.asarray(rendered.child_feature_valid, dtype=bool).reshape(map_rows.shape)
    if (
        map_mass.shape != map_rows.shape or map_valid.shape != map_rows.shape
        or np.any((map_rows >= child_parent.size) | (map_rows < -1))
    ):
        raise ValueError("rendered child arrays or parent lineage differ")
    if hasattr(rendered, "parent_rows") and hasattr(rendered, "parent_weights"):
        map_parent = np.asarray(rendered.parent_rows, dtype=np.int64).reshape(height * width, -1)
        map_parent_mass = np.asarray(rendered.parent_weights, dtype=np.float64).reshape(height * width, -1)
        if map_parent.shape != map_parent_mass.shape:
            raise ValueError("direct rendered parent arrays differ")
    else:
        # Compatibility path for old diagnostics only.  New renderer outputs
        # direct parent segments so child Top-L tail cannot erase parent mass.
        map_parent = np.full(map_rows.shape, -1, dtype=np.int64)
        valid_map_row = map_rows >= 0
        map_parent[valid_map_row] = child_parent[map_rows[valid_map_row]]
        map_parent_mass = map_mass
    query_parent = np.asarray(retrieval.token_parent_ids, dtype=np.int64)
    query_parent_mass = np.asarray(retrieval.token_parent_probabilities, dtype=np.float64)
    query_child = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    query_child_mass = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    if (
        query_parent.shape != query_parent_mass.shape
        or query_child.shape != query_child_mass.shape
        or query_parent.shape[0] != height * width
        or query_child.shape[0] != height * width
    ):
        raise ValueError("query hierarchy arrays differ from token grid")
    query_unit = query / np.maximum(np.linalg.norm(query, axis=0, keepdims=True), 1e-8)
    query_token = query_unit.transpose(1, 2, 0).reshape(height * width, channels)
    map_feature = map_feature / np.maximum(
        np.linalg.norm(map_feature, axis=2, keepdims=True), 1e-8
    )
    parent_overlap = np.zeros((height * width,), dtype=np.float64)
    child_overlap = np.zeros_like(parent_overlap)
    coupled = np.zeros_like(parent_overlap)
    coupled_mass = np.zeros_like(parent_overlap)
    coupled_cosine = np.zeros_like(parent_overlap)
    offsets = _fixed_spatial_kernel(int(spatial_kernel_radius))
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    query_index_full = (yy * width + xx).reshape(-1)
    for dx, dy, kernel_weight in offsets:
        target_x, target_y = xx + dx, yy + dy
        inside = (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
        query_index = query_index_full[inside.reshape(-1)]
        map_index = (target_y[inside] * width + target_x[inside]).astype(np.int64)
        parent_match = query_parent[query_index, :, None] == map_parent[map_index, None, :]
        parent_match &= (query_parent[query_index, :, None] >= 0) & (map_parent[map_index, None, :] >= 0)
        parent_overlap[query_index] += float(kernel_weight) * np.sum(
            query_parent_mass[query_index, :, None] * map_parent_mass[map_index, None, :] * parent_match,
            axis=(1, 2),
        )
        child_match = query_child[query_index, :, None] == map_rows[map_index, None, :]
        child_match &= (query_child[query_index, :, None] >= 0) & (map_rows[map_index, None, :] >= 0)
        gamma = query_child_mass[query_index, :, None] * map_mass[map_index, None, :] * child_match
        child_overlap[query_index] += float(kernel_weight) * np.sum(gamma, axis=(1, 2))
        cosine = np.clip(
            np.einsum("tc,tlc->tl", query_token[query_index], map_feature[map_index]),
            -1.0, 1.0,
        )
        valid_gamma = gamma * map_valid[map_index, None, :]
        local_mass = np.sum(valid_gamma, axis=(1, 2))
        local_cosine = np.sum(valid_gamma * cosine[:, None, :], axis=(1, 2))
        coupled_mass[query_index] += float(kernel_weight) * local_mass
        coupled_cosine[query_index] += float(kernel_weight) * local_cosine
        coupled[query_index] += float(kernel_weight) * (local_mass + local_cosine)
    parent_overlap = np.clip(parent_overlap, 0.0, 1.0)
    child_overlap = np.clip(child_overlap, 0.0, 1.0)
    reliability = query_only_pose_reliability_weights(
        retrieval, semantics=str(query_reliability_semantics)
    )
    parent_score = _query_weighted_mean(2.0 * parent_overlap - 1.0, reliability)
    child_score = _query_weighted_mean(2.0 * child_overlap - 1.0, reliability)
    # Factorized coarse-to-fine evidence with a fixed 1/2 coarse floor:
    # P(parent) * [1/2 + 1/2 P(child|parent)] = (parent+child)/2.
    hierarchical_atom = np.clip(-1.0 + parent_overlap + child_overlap, -1.0, 1.0)
    hierarchical_score = _query_weighted_mean(hierarchical_atom, reliability)
    feature_score = _query_weighted_mean(
        np.clip(-1.0 + coupled, -1.0, 1.0), reliability
    )
    reliability_total = max(float(np.sum(reliability)), 1e-12)
    return HierarchicalSpatialSoftSurfacePoseEnergy(
        combined_score=float((1.0 - alpha) * hierarchical_score + alpha * feature_score),
        hierarchical_identity_score=hierarchical_score,
        parent_support_score=parent_score,
        child_precision_score=child_score,
        child_coupled_radio_score=feature_score,
        mean_parent_overlap=float(np.sum(reliability * parent_overlap) / reliability_total),
        mean_child_overlap=float(np.sum(reliability * child_overlap) / reliability_total),
        coupled_feature_mass=float(np.sum(reliability * coupled_mass) / reliability_total),
        coupled_feature_agreement=float(
            np.sum(reliability * coupled_cosine)
            / max(float(np.sum(reliability * coupled_mass)), 1e-12)
        ),
        spatial_kernel=(
            "fixed_center_half_cardinal_four_eighths_no_border_renormalization_v1"
            if int(spatial_kernel_radius) == 1 else
            f"fixed_gaussian_radius{int(spatial_kernel_radius)}_no_border_renormalization_v1"
        ),
        query_reliability_semantics=str(query_reliability_semantics),
        effective_query_reliability=float(np.mean(reliability)),
    )


def score_soft_surface_overlap_ladder(
    query_feature: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    rendered,
    *,
    child_to_parent_ids: np.ndarray,
    query_reliability_semantics: str = "child_mass_entropy_background_v1",
) -> SoftSurfaceOverlapLadder:
    """Report parent and child overlap from tolerant to exact support."""

    scores = [
        score_hierarchical_spatial_soft_surface_pose_energy(
            query_feature, retrieval, rendered,
            child_to_parent_ids=child_to_parent_ids, radio_weight=0.0,
            spatial_kernel_radius=radius,
            query_reliability_semantics=query_reliability_semantics,
        )
        for radius in (2, 1, 0)
    ]
    return SoftSurfaceOverlapLadder(
        parent_overlap=scores[0].mean_parent_overlap,
        child_overlap_radius2=scores[0].mean_child_overlap,
        child_overlap_radius1=scores[1].mean_child_overlap,
        child_overlap_radius0=scores[2].mean_child_overlap,
        query_reliability_semantics=str(query_reliability_semantics),
    )


def score_bidirectional_soft_surface_pose_energy(
    query_feature: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    rendered,
    *,
    radio_weight: float = 0.5,
) -> BidirectionalSoftSurfacePoseEnergy:
    """Score query/map Top-L child distributions without dominant identities.

    Both bounded terms use the same latent child overlap.  RADIO evidence is
    counted only for the child contributor that simultaneously has query mass
    and rendered alpha; every unmatched unit remains at the fixed ``-1`` floor.
    """

    alpha = float(radio_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("radio_weight must lie in [0,1]")
    query = np.asarray(query_feature, dtype=np.float32)
    if query.ndim != 3:
        raise ValueError("query feature must have shape [C,H,W]")
    height, width = query.shape[1:]
    map_rows = np.asarray(rendered.child_rows, dtype=np.int64)
    map_weight = np.asarray(rendered.child_weights, dtype=np.float64)
    map_feature = np.asarray(rendered.child_features, dtype=np.float32)
    map_feature_valid = np.asarray(rendered.child_feature_valid, dtype=bool)
    map_null = np.asarray(rendered.null_weight, dtype=np.float64).reshape(-1)
    if (
        map_rows.ndim != 3
        or map_rows.shape[:2] != (height, width)
        or map_weight.shape != map_rows.shape
        or map_feature.shape != map_rows.shape + (query.shape[0],)
        or map_feature_valid.shape != map_rows.shape
        or map_null.shape != (height * width,)
    ):
        raise ValueError("rendered soft-child arrays differ from query grid")
    query_rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    query_weight = np.asarray(
        retrieval.token_child_probabilities, dtype=np.float64
    )
    if query_rows.shape != query_weight.shape or query_rows.shape[0] != height * width:
        raise ValueError("query child evidence differs from query grid")
    expected_xy = all_radio_token_coordinates(height, width)
    if not np.array_equal(np.asarray(retrieval.token_xy), expected_xy):
        raise ValueError("retrieval token order is not exact row-major (x,y)")
    flat_map_weight = map_weight.reshape(height * width, -1)
    typed_names = (
        "child_tail_weight", "unassigned_geometry_weight", "background_weight",
    )
    if all(hasattr(rendered, name) for name in typed_names):
        typed = [
            np.asarray(getattr(rendered, name), dtype=np.float64).reshape(-1)
            for name in typed_names
        ]
        conservation = np.sum(flat_map_weight, axis=1) + sum(typed)
        if (
            any(value.shape != (height * width,) for value in typed)
            or any(np.any(~np.isfinite(value)) or np.any(value < -1e-7) for value in typed)
            or np.max(np.abs(conservation - 1.0), initial=0.0) > 2e-5
        ):
            raise ValueError("rendered typed child mass is not conserved")
        typed_null = sum(typed)
        if np.max(np.abs(typed_null - map_null), initial=0.0) > 2e-5:
            raise ValueError("rendered null mass differs from typed residual mass")
    query_sum = np.sum(query_weight, axis=1)
    if np.any(query_sum > 1.0 + 2e-5):
        raise ValueError("query child evidence exceeds unit mass")
    query_null = np.maximum(1.0 - query_sum, 0.0)
    flat_map_rows = map_rows.reshape(height * width, -1)
    match = query_rows[:, :, None] == flat_map_rows[:, None, :]
    match &= (query_rows[:, :, None] >= 0) & (flat_map_rows[:, None, :] >= 0)
    gamma = (
        query_weight[:, :, None] * flat_map_weight[:, None, :] * match
    )
    child_overlap = np.sum(gamma, axis=(1, 2))
    # Unknown is a fixed failure floor, not a semantic class.  In particular,
    # query-null and render-null must never earn a positive "null-null match".
    overlap = np.clip(child_overlap, 0.0, 1.0)
    identity_score = float(np.mean(2.0 * overlap - 1.0))

    query_unit = query / np.maximum(
        np.linalg.norm(query, axis=0, keepdims=True), 1e-8
    )
    query_token = query_unit.transpose(1, 2, 0).reshape(height * width, -1)
    flat_feature = map_feature.reshape(height * width, -1, query.shape[0])
    flat_feature /= np.maximum(
        np.linalg.norm(flat_feature, axis=2, keepdims=True), 1e-8
    )
    cosine = np.clip(
        np.einsum("tc,tlc->tl", query_token, flat_feature), -1.0, 1.0
    )
    valid_gamma = gamma * map_feature_valid.reshape(height * width, 1, -1)
    coupled = np.sum(valid_gamma * (cosine[:, None, :] + 1.0), axis=(1, 2))
    feature_atom = np.clip(-1.0 + coupled, -1.0, 1.0)
    feature_score = float(np.mean(feature_atom))
    combined = (1.0 - alpha) * identity_score + alpha * feature_score
    feature_mass = float(np.mean(np.sum(valid_gamma, axis=(1, 2))))
    feature_agreement = float(
        np.sum(valid_gamma * cosine[:, None, :])
        / max(float(np.sum(valid_gamma)), 1e-12)
    )
    typed_mean = {
        name: float(np.mean(np.asarray(getattr(rendered, name), dtype=np.float64)))
        if hasattr(rendered, name) else float("nan")
        for name in (
            "child_tail_weight", "unassigned_geometry_weight", "background_weight",
            "canonical_field_missing_weight", "payload_excluded_weight",
        )
    }
    return BidirectionalSoftSurfacePoseEnergy(
        combined_score=float(combined),
        child_overlap_score=identity_score,
        child_coupled_radio_score=feature_score,
        mean_child_overlap=float(np.mean(overlap)),
        query_null_mass=float(np.mean(query_null)),
        rendered_null_mass=float(np.mean(map_null)),
        coupled_feature_mass=feature_mass,
        coupled_feature_agreement=feature_agreement,
        rendered_child_tail_mass=typed_mean["child_tail_weight"],
        rendered_unassigned_geometry_mass=typed_mean["unassigned_geometry_weight"],
        rendered_background_mass=typed_mean["background_weight"],
        rendered_canonical_field_missing_mass=typed_mean["canonical_field_missing_weight"],
        rendered_payload_excluded_mass=typed_mean["payload_excluded_weight"],
    )


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
    # The caller may pass a view into a probe-design matrix.  Normalizing that
    # view in place silently changed the coordinates later used to fit the 6D
    # Hessian, while the rendered pose still used the pre-normalization angle.
    axis = np.array(axis_camera, dtype=np.float64, copy=True).reshape(3)
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
