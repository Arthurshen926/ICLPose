"""Multi-modal coarse pose proposals from region-to-child surface factors.

The observations are medium-scale query regions and physical child tiles.  A
minimal PnP solver is used only as a numerical initializer for a region-factor
proposal; the map does not contain keypoints, tracks, or point descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

from .child_retrieval import ChildTilePosterior
from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap
from .typed_graph import TypedParentGraph


@dataclass(frozen=True)
class CoarsePoseModes:
    poses_w2c: np.ndarray
    scores: np.ndarray
    supporting_region_count: np.ndarray
    configuration_parent_rows: np.ndarray | None = None
    configuration_child_rows: np.ndarray | None = None
    proposal_seed_parent_rows: np.ndarray | None = None
    proposal_seed_support_rows: np.ndarray | None = None
    mapping_view_anchor_labels: np.ndarray | None = None
    mapping_view_prior_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.poses_w2c, dtype=np.float64)
        score = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        support = np.asarray(self.supporting_region_count, dtype=np.int64).reshape(-1)
        if pose.shape != (score.size, 4, 4) or support.shape != score.shape:
            raise ValueError("invalid coarse pose modes")
        parent = self.configuration_parent_rows
        child = self.configuration_child_rows
        seed = self.proposal_seed_parent_rows
        seed_support = self.proposal_seed_support_rows
        mapping_label = self.mapping_view_anchor_labels
        mapping_score = self.mapping_view_prior_scores
        if (parent is None) != (child is None):
            raise ValueError("coarse pose configuration provenance is incomplete")
        if parent is not None:
            parent = np.asarray(parent, dtype=np.int64)
            child = np.asarray(child, dtype=np.int64)
            if parent.ndim != 2 or child.shape != parent.shape or parent.shape[0] != score.size:
                raise ValueError("coarse pose configuration provenance differs")
            object.__setattr__(self, "configuration_parent_rows", parent)
            object.__setattr__(self, "configuration_child_rows", child)
        if seed is not None:
            seed = np.asarray(seed, dtype=np.int64)
            if seed.ndim != 2 or seed.shape[0] != score.size:
                raise ValueError("coarse pose seed provenance differs")
            object.__setattr__(self, "proposal_seed_parent_rows", seed)
        if (seed is None) != (seed_support is None):
            raise ValueError("coarse pose seed parent/support provenance is incomplete")
        if seed_support is not None:
            seed_support = np.asarray(seed_support, dtype=np.int64)
            if seed_support.shape != seed.shape:
                raise ValueError("coarse pose seed support provenance differs")
            object.__setattr__(self, "proposal_seed_support_rows", seed_support)
        if (mapping_label is None) != (mapping_score is None):
            raise ValueError("coarse pose mapping-view provenance is incomplete")
        if mapping_label is not None:
            mapping_label = np.asarray(mapping_label, dtype=np.int64).reshape(-1)
            mapping_score = np.asarray(mapping_score, dtype=np.float64).reshape(-1)
            if mapping_label.shape != score.shape or mapping_score.shape != score.shape:
                raise ValueError("coarse pose mapping-view provenance differs")
            if np.any((mapping_label < 0) & np.isfinite(mapping_score)):
                raise ValueError("unanchored pose cannot carry a mapping-view prior")
            object.__setattr__(self, "mapping_view_anchor_labels", mapping_label)
            object.__setattr__(self, "mapping_view_prior_scores", mapping_score)
        object.__setattr__(self, "poses_w2c", pose)
        object.__setattr__(self, "scores", score)
        object.__setattr__(self, "supporting_region_count", support)


def _rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left)[:3, :3] @ np.asarray(right)[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _select_child_for_parent(
    posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    *,
    support: int,
    parent_slot: int,
    parent_row: int,
    parent_probability: float,
) -> tuple[int, float]:
    """Enumerate a parent-conditioned child independently of score weights."""

    if posterior.best_child_rows_by_parent is not None:
        child = int(posterior.best_child_rows_by_parent[support, parent_slot])
        if child < 0:
            return -1, 0.0
        if int(physical.child_parent_rows[child]) != int(parent_row):
            raise ValueError("parent-conditioned child belongs to another parent")
        return child, float(
            parent_probability
            * posterior.best_child_probabilities_by_parent[support, parent_slot]
        )
    rows = posterior.candidate_child_rows[support]
    probability = posterior.candidate_probabilities[support]
    valid = rows >= 0
    selected = np.flatnonzero(
        valid & (physical.child_parent_rows[np.maximum(rows, 0)] == int(parent_row))
    )
    if selected.size == 0:
        return -1, 0.0
    slot = int(selected[np.argmax(probability[selected])])
    return int(rows[slot]), float(probability[slot])


def _project(points: np.ndarray, pose: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(np.asarray(pose[:3, :3], dtype=np.float64))
    xy, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64), rotation, np.asarray(pose[:3, 3], dtype=np.float64),
        matrix, distortion,
    )
    camera_xyz = np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
    return xy.reshape(-1, 2), camera_xyz[:, 2]


def _score_pose(
    pose: np.ndarray,
    xy: np.ndarray,
    extent: np.ndarray,
    posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    maximum_regions: int = 96,
    maximum_children: int = 32,
) -> tuple[float, int, np.ndarray]:
    confidence = 1.0 - posterior.null_probabilities
    selected_regions = np.argsort(-confidence, kind="stable")[: int(maximum_regions)]
    rows = posterior.candidate_child_rows[selected_regions, : int(maximum_children)]
    probability = posterior.candidate_probabilities[selected_regions, : int(maximum_children)].astype(np.float64)
    valid = (rows >= 0) & (probability > 0.0)
    if not np.any(valid):
        return float("-inf"), 0, np.full((xy.shape[0],), -1, dtype=np.int64)
    safe_rows = np.maximum(rows, 0)
    flat_xy, flat_depth = _project(physical.child_centers[safe_rows.reshape(-1)], pose, camera)
    projected = flat_xy.reshape(rows.shape + (2,))
    depth = flat_depth.reshape(rows.shape)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    view = camera_center[None, None] - physical.child_centers[safe_rows]
    view /= np.maximum(np.linalg.norm(view, axis=2, keepdims=True), 1e-8)
    incidence = np.sum(physical.child_normals[safe_rows] * view, axis=2)
    parent = physical.child_parent_rows[safe_rows]
    front = (physical.maplet_sidedness[parent] == DOUBLE_SIDED) | (incidence >= 0.02)
    valid &= (depth > 0.05) & front
    residual = np.linalg.norm(projected - xy[selected_regions, None], axis=2)
    # Region support, not a point keypoint, defines the uncertainty scale.
    sigma = np.maximum(10.0, 0.75 * np.linalg.norm(extent[selected_regions], axis=1))
    likelihood = probability * np.exp(-0.5 * np.square(residual / sigma[:, None])) * valid
    region_likelihood = np.sum(likelihood, axis=1)
    # Explicit null branch prevents low-coverage modes from winning by scoring
    # only their easiest observations.
    floor = np.maximum(0.02 * posterior.null_probabilities[selected_regions], 1e-8)
    weight = np.maximum(confidence[selected_regions], 0.05)
    score = float(np.sum(weight * np.log(np.maximum(region_likelihood, floor))) / np.sum(weight))
    best_slot = np.argmax(likelihood, axis=1)
    best_rows = rows[np.arange(rows.shape[0]), best_slot]
    best_rows[region_likelihood <= np.maximum(floor, 1e-4)] = -1
    aligned_best_rows = np.full((xy.shape[0],), -1, dtype=np.int64)
    aligned_best_rows[selected_regions] = best_rows
    support = int(np.sum(region_likelihood > np.maximum(floor, 1e-4)))
    return score, support, aligned_best_rows


def _refine_region_mode(
    pose: np.ndarray,
    xy: np.ndarray,
    posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    chosen_region_rows: np.ndarray,
    *,
    minimum_regions: int = 6,
) -> np.ndarray:
    selected = np.flatnonzero(chosen_region_rows >= 0)
    if selected.size < int(minimum_regions):
        return pose
    points = physical.child_centers[chosen_region_rows[selected]]
    projected, depth = _project(points, pose, camera)
    error = np.linalg.norm(projected - xy[selected], axis=1)
    order = np.argsort(error, kind="stable")
    retain = order[: max(int(minimum_regions), int(np.ceil(0.6 * order.size)))]
    selected = selected[retain]
    points = physical.child_centers[chosen_region_rows[selected]]
    if np.unique(np.round(points, 5), axis=0).shape[0] < int(minimum_regions):
        return pose
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(pose[:3, :3])
    translation = pose[:3, 3].reshape(3, 1)
    try:
        rotation, translation = cv2.solvePnPRefineLM(
            points.astype(np.float64), xy[selected].astype(np.float64), matrix, distortion,
            rotation, translation,
        )
    except cv2.error:
        return pose
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(rotation)[0]
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def generate_region_pose_modes(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    camera: ColmapCamera,
    *,
    maximum_modes: int = 32,
    proposal_trials: int = 2048,
    seed_region_count: int = 48,
    seed_child_count: int = 24,
    minimal_region_count: int = 6,
    probability_power: float = 0.5,
    translation_nms_m: float = 0.20,
    rotation_nms_deg: float = 3.0,
    random_seed: int = 194917,
) -> CoarsePoseModes:
    """Generate diverse Top-N poses from uncertain region/child assignments."""

    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    if xy.shape != extent.shape or xy.shape[0] != posterior.candidate_child_rows.shape[0]:
        raise ValueError("query regions and child posterior differ")
    confidence = 1.0 - posterior.null_probabilities
    eligible = np.flatnonzero(
        (confidence > 0.02)
        & np.any((posterior.candidate_child_rows[:, : int(seed_child_count)] >= 0)
                 & (posterior.candidate_probabilities[:, : int(seed_child_count)] > 0.0), axis=1)
    )
    if eligible.size < int(minimal_region_count):
        return CoarsePoseModes(np.zeros((0, 4, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))
    # Preserve strong evidence while retaining image-wide geometric leverage.
    image_center = np.asarray([0.5 * camera.width, 0.5 * camera.height])
    leverage = 0.5 + np.linalg.norm((xy - image_center) / [camera.width, camera.height], axis=1)
    priority = confidence * leverage
    eligible = eligible[np.argsort(-priority[eligible], kind="stable")[: int(seed_region_count)]]
    region_probability = priority[eligible]
    region_probability /= np.sum(region_probability)
    rng = np.random.default_rng(int(random_seed))
    matrix, distortion = camera_matrix_and_distortion(camera)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(int(proposal_trials)):
        region = rng.choice(
            eligible, size=int(minimal_region_count), replace=False,
            p=region_probability,
        )
        child_rows = []
        for support in region.tolist():
            rows = posterior.candidate_child_rows[support, : int(seed_child_count)]
            probability = posterior.candidate_probabilities[support, : int(seed_child_count)].astype(np.float64)
            valid = (rows >= 0) & (probability > 0.0)
            rows, probability = rows[valid], probability[valid]
            if rows.size == 0:
                child_rows = []
                break
            tempered = np.power(probability, float(probability_power))
            tempered /= np.sum(tempered)
            child_rows.append(int(rng.choice(rows, p=tempered)))
        if len(child_rows) != int(minimal_region_count):
            continue
        xyz = physical.child_centers[np.asarray(child_rows, dtype=np.int64)]
        if np.unique(np.round(xyz, 4), axis=0).shape[0] < int(minimal_region_count):
            continue
        try:
            success, rotation, translation = cv2.solvePnP(
                xyz.astype(np.float64), xy[region].astype(np.float64), matrix, distortion,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            continue
        if not success:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = np.asarray(translation).reshape(3)
        if not np.all(np.isfinite(pose)):
            continue
        score, support, chosen = _score_pose(pose, xy, extent, posterior, physical, camera=camera)
        if not np.isfinite(score):
            continue
        pose = _refine_region_mode(pose, xy, posterior, physical, camera, chosen)
        score, support, chosen = _score_pose(pose, xy, extent, posterior, physical, camera=camera)
        parent_rows = np.full(chosen.shape, -1, dtype=np.int64)
        valid_chosen = chosen >= 0
        parent_rows[valid_chosen] = physical.child_parent_rows[chosen[valid_chosen]]
        candidates.append((
            score, support, pose, parent_rows, chosen,
            np.full((2,), -1, dtype=np.int64),
        ))
    candidates.sort(key=lambda value: (-value[0], -value[1]))
    retained: list[tuple[float, int, np.ndarray]] = []
    for candidate in candidates:
        camera_center = -candidate[2][:3, :3].T @ candidate[2][:3, 3]
        duplicate = False
        for existing in retained:
            existing_center = -existing[2][:3, :3].T @ existing[2][:3, 3]
            if (
                np.linalg.norm(camera_center - existing_center) < float(translation_nms_m)
                and _rotation_distance_degrees(candidate[2], existing[2]) < float(rotation_nms_deg)
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
        if len(retained) >= int(maximum_modes):
            break
    return CoarsePoseModes(
        poses_w2c=np.asarray([value[2] for value in retained], dtype=np.float64).reshape(-1, 4, 4),
        scores=np.asarray([value[0] for value in retained], dtype=np.float64),
        supporting_region_count=np.asarray([value[1] for value in retained], dtype=np.int64),
        configuration_parent_rows=np.asarray(
            [value[3] for value in retained], dtype=np.int64,
        ).reshape(-1, xy.shape[0]),
        configuration_child_rows=np.asarray(
            [value[4] for value in retained], dtype=np.int64,
        ).reshape(-1, xy.shape[0]),
        proposal_seed_parent_rows=np.asarray(
            [value[5] for value in retained], dtype=np.int64,
        ).reshape(-1, 2),
        proposal_seed_support_rows=np.full((len(retained), 2), -1, dtype=np.int64),
    )


def generate_graph_conditioned_pose_modes(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    graph: TypedParentGraph,
    camera: ColmapCamera,
    *,
    maximum_modes: int = 32,
    seed_parent_count: int = 64,
    pair_anchor_count: int = 16,
    seed_parent_pair_count: int = 0,
    support_anchor_count: int = 0,
    support_anchor_pair_count: int = 0,
    support_anchor_candidate_count: int = 8,
    covisibility_weight: float = 0.75,
    local_evidence_weight: float = 1.0,
    ransac_reprojection_px: float = 32.0,
    ransac_iterations: int = 6000,
    translation_nms_m: float = 0.20,
    rotation_nms_deg: float = 3.0,
    random_seed: int = 194917,
) -> CoarsePoseModes:
    """Build coherent parent modes, then solve robust child-region factors.

    Each seed is a physical parent identity with image-wide posterior support.
    Co-visibility conditions all region assignments jointly before any pose is
    solved.  Same-parent evidence is neutral (compatibility one), never zero.
    """

    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("typed graph and physical map lineage differ")
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if (
        xy.shape != extent.shape or parent_ids.shape != parent_probability.shape
        or parent_ids.shape[0] != xy.shape[0] or parent_null.shape != (xy.shape[0],)
        or child_posterior.candidate_child_rows.shape[0] != xy.shape[0]
    ):
        raise ValueError("graph-conditioned pose inputs differ")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    candidate_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    for support in range(parent_ids.shape[0]):
        for slot in range(parent_ids.shape[1]):
            candidate_rows[support, slot] = row_by_id.get(int(parent_ids[support, slot]), -1)
    valid_parent = (candidate_rows >= 0) & (parent_probability > 0.0)
    conditioned_child = child_posterior.conditional_parent_ids is not None
    uses_local_evidence_score = conditioned_child and float(local_evidence_weight) > 0.0
    if conditioned_child:
        conditional_parent_ids = np.asarray(child_posterior.conditional_parent_ids, dtype=np.int64)
        if conditional_parent_ids.shape != parent_ids.shape or not np.array_equal(
            conditional_parent_ids, parent_ids
        ):
            raise ValueError("parent-conditioned child evidence is not aligned to parent candidates")
        if uses_local_evidence_score:
            local_evidence = np.asarray(
                child_posterior.conditional_parent_log_evidence, dtype=np.float64
            ).copy()
            finite = np.isfinite(local_evidence) & valid_parent
            for support in range(local_evidence.shape[0]):
                values = local_evidence[support, finite[support]]
                if values.size:
                    center = float(np.mean(values))
                    scale = max(float(np.std(values)), 1e-4)
                    local_evidence[support, finite[support]] = (values - center) / scale
                local_evidence[support, ~finite[support]] = -1e4
    global_mass = np.zeros((physical.maplet_ids.size,), dtype=np.float64)
    np.add.at(global_mass, candidate_rows[valid_parent], parent_probability[valid_parent])
    seeds = np.argsort(-global_mass, kind="stable")[: int(seed_parent_count)]
    seeds = seeds[global_mass[seeds] > 0.0]
    covis = graph.covisibility_matrix().astype(np.float64)
    matrix, distortion = camera_matrix_and_distortion(camera)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    cv2.setRNGSeed(int(random_seed))
    # The unconditioned mode is retained as a regression reference.  Pair
    # seeds encode a jointly visible context configuration; this is essential
    # on periodic facades where any one window/parent is an ambiguous phase.
    # Each seed item is (query support, physical parent, candidate slot).  A
    # support of -1 denotes the historical image-wide parent seed.  Explicit
    # support anchors retain the missing query-to-map association without
    # pretending that a parent centre is a point measurement.
    seed_modes: list[tuple[tuple[int, int, int], ...]] = [((-1, -1, -1),)]
    seed_modes.extend([((-1, int(seed), -1),) for seed in seeds.tolist()])
    pair_anchors = seeds[: int(pair_anchor_count)]
    pair_modes = []
    for left_offset, left in enumerate(pair_anchors.tolist()):
        for right in pair_anchors[left_offset + 1 :].tolist():
            relation = float(covis[int(left), int(right)])
            if relation <= 0.0:
                continue
            priority = (
                np.log1p(global_mass[int(left)]) + np.log1p(global_mass[int(right)])
                + np.log(0.05 + 0.95 * relation)
                + 0.25 * float(graph.parent_distinctiveness[int(left)] + graph.parent_distinctiveness[int(right)])
            )
            pair_modes.append((priority, (int(left), int(right))))
    pair_modes.sort(key=lambda item: (-item[0], item[1]))
    seed_modes.extend([
        tuple((-1, int(parent), -1) for parent in item[1])
        for item in pair_modes[: int(seed_parent_pair_count)]
    ])

    anchor_candidates: list[tuple[float, int, int, int]] = []
    normalized_xy = xy / np.asarray([float(camera.width), float(camera.height)])
    image_leverage = 0.5 + np.linalg.norm(normalized_xy - 0.5, axis=1)
    candidate_limit = min(int(support_anchor_candidate_count), parent_probability.shape[1])
    for support in range(xy.shape[0]):
        if parent_null[support] >= 0.98:
            continue
        for slot in range(candidate_limit):
            parent = int(candidate_rows[support, slot])
            probability = float(parent_probability[support, slot])
            if parent < 0 or probability <= 0.0:
                continue
            priority = (
                np.log(max(probability, 1e-12))
                + np.log(max(1.0 - float(parent_null[support]), 1e-4))
                + np.log(float(image_leverage[support]))
                + 0.25 * float(graph.parent_distinctiveness[parent])
            )
            anchor_candidates.append((priority, int(support), parent, int(slot)))
    anchor_candidates.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))
    retained_anchors: list[tuple[float, int, int, int]] = []
    per_support_count: dict[int, int] = {}
    for value in anchor_candidates:
        support = int(value[1])
        # Keep alternatives across the image instead of spending the complete
        # budget on one large high-confidence facade region.
        if per_support_count.get(support, 0) >= 2:
            continue
        retained_anchors.append(value)
        per_support_count[support] = per_support_count.get(support, 0) + 1
        if len(retained_anchors) >= int(support_anchor_count):
            break
    seed_modes.extend([((support, parent, slot),) for _, support, parent, slot in retained_anchors])

    anchor_pair_modes: list[tuple[float, tuple[tuple[int, int, int], tuple[int, int, int]]]] = []
    # Pair hypotheses need identity alternatives, not just Top-1 hypotheses
    # from many near-duplicate supports.  Select spatially spread supports and
    # retain their first ``support_anchor_candidate_count`` alternatives.
    support_priority = np.clip(1.0 - parent_null, 0.0, 1.0) * image_leverage
    support_order = np.argsort(-support_priority, kind="stable")
    selected_pair_supports: list[int] = []
    spatial_count: dict[int, int] = {}
    bins = np.clip(np.floor(3.0 * normalized_xy).astype(np.int64), 0, 2)
    for support in support_order.tolist():
        if not np.any(valid_parent[support, :candidate_limit]):
            continue
        spatial_bin = int(3 * bins[support, 1] + bins[support, 0])
        if spatial_count.get(spatial_bin, 0) >= 4:
            continue
        selected_pair_supports.append(int(support))
        spatial_count[spatial_bin] = spatial_count.get(spatial_bin, 0) + 1
        if len(selected_pair_supports) >= 32:
            break
    selected_pair_support_set = set(selected_pair_supports)
    pair_pool = [
        value for value in anchor_candidates
        if int(value[1]) in selected_pair_support_set
    ]
    for left_offset, left_value in enumerate(pair_pool):
        _, left_support, left_parent, left_slot = left_value
        for right_value in pair_pool[left_offset + 1 :]:
            _, right_support, right_parent, right_slot = right_value
            if left_support == right_support:
                continue
            displacement = float(np.linalg.norm(normalized_xy[left_support] - normalized_xy[right_support]))
            if displacement < 0.20:
                continue
            relation = float(covis[left_parent, right_parent])
            if relation <= 0.0:
                continue
            priority = (
                float(left_value[0]) + float(right_value[0])
                + np.log(0.05 + 0.95 * relation)
                + np.log(0.5 + displacement)
            )
            anchors = (
                (int(left_support), int(left_parent), int(left_slot)),
                (int(right_support), int(right_parent), int(right_slot)),
            )
            anchor_pair_modes.append((priority, anchors))
    anchor_pair_modes.sort(key=lambda value: (-value[0], value[1]))
    seed_modes.extend([value[1] for value in anchor_pair_modes[: int(support_anchor_pair_count)]])
    for seed_mode in seed_modes:
        compatibility = np.ones_like(parent_probability)
        seed_parents = [int(value[1]) for value in seed_mode if int(value[1]) >= 0]
        if seed_parents:
            safe = np.maximum(candidate_rows, 0)
            log_compatibility = np.zeros_like(parent_probability)
            for seed in seed_parents:
                value = 0.05 + 0.95 * covis[safe, int(seed)]
                value[candidate_rows < 0] = 0.05
                log_compatibility += np.log(np.maximum(value, 1e-8))
            compatibility = np.exp(log_compatibility / float(len(seed_parents)))
            # Unseen co-visibility is uncertainty, not an impossible edge.
            compatibility[candidate_rows < 0] = 0.0
        assignment_score = np.log(np.maximum(parent_probability, 1e-12))
        assignment_score += float(covisibility_weight) * np.log(np.maximum(compatibility, 1e-8))
        if uses_local_evidence_score:
            assignment_score += float(local_evidence_weight) * local_evidence
        assignment_score[~valid_parent] = -np.inf
        chosen_slot = np.argmax(assignment_score, axis=1)
        for anchor_support, anchor_parent, anchor_slot in seed_mode:
            if anchor_support < 0:
                continue
            if int(candidate_rows[anchor_support, anchor_slot]) != int(anchor_parent):
                raise ValueError("support anchor provenance differs from candidates")
            chosen_slot[anchor_support] = int(anchor_slot)
        chosen_parent = candidate_rows[np.arange(candidate_rows.shape[0]), chosen_slot]
        chosen_parent_score = assignment_score[np.arange(candidate_rows.shape[0]), chosen_slot]
        child_rows = np.full((xy.shape[0],), -1, dtype=np.int64)
        child_probability = np.zeros((xy.shape[0],), dtype=np.float64)
        for support in range(xy.shape[0]):
            parent = int(chosen_parent[support])
            if parent < 0 or not np.isfinite(chosen_parent_score[support]):
                continue
            slot = int(chosen_slot[support])
            child, probability = _select_child_for_parent(
                child_posterior,
                physical,
                support=support,
                parent_slot=slot,
                parent_row=parent,
                parent_probability=float(parent_probability[support, slot]),
            )
            child_rows[support] = child
            child_probability[support] = probability
        selected = np.flatnonzero(
            (child_rows >= 0) & (child_probability > 0.0) & (parent_null < 0.98)
        )
        if selected.size < 6:
            continue
        xyz = physical.child_centers[child_rows[selected]]
        if np.unique(np.round(xyz, 5), axis=0).shape[0] < 6:
            continue
        try:
            success, rotation, translation, inliers = cv2.solvePnPRansac(
                xyz.astype(np.float64), xy[selected].astype(np.float64), matrix, distortion,
                iterationsCount=int(ransac_iterations),
                reprojectionError=float(ransac_reprojection_px), confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            continue
        if not success or inliers is None or len(inliers) < 6:
            continue
        inlier_rows = np.asarray(inliers, dtype=np.int64).reshape(-1)
        try:
            rotation, translation = cv2.solvePnPRefineLM(
                xyz[inlier_rows].astype(np.float64), xy[selected[inlier_rows]].astype(np.float64),
                matrix, distortion, rotation, translation,
            )
        except cv2.error:
            pass
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = np.asarray(translation).reshape(3)
        region_score, support, _ = _score_pose(
            pose, xy, extent, child_posterior, physical, camera=camera
        )
        inlier_fraction = float(len(inlier_rows) / max(selected.size, 1))
        seed_mass = float(np.mean([global_mass[parent] for parent in seed_parents])) if seed_parents else 0.0
        score = float(region_score + 2.0 * inlier_fraction + 0.1 * np.log1p(seed_mass))
        padded_seed = np.full((2,), -1, dtype=np.int64)
        padded_support = np.full((2,), -1, dtype=np.int64)
        for index, (anchor_support, anchor_parent, _anchor_slot) in enumerate(seed_mode[:2]):
            padded_seed[index] = int(anchor_parent)
            padded_support[index] = int(anchor_support)
        candidates.append((
            score, int(len(inlier_rows)), pose,
            chosen_parent.copy(), child_rows.copy(), padded_seed, padded_support,
        ))
    candidates.sort(key=lambda value: (-value[0], -value[1]))
    retained: list[tuple[float, int, np.ndarray]] = []
    for candidate in candidates:
        center = -candidate[2][:3, :3].T @ candidate[2][:3, 3]
        if any(
            np.linalg.norm(center - (-other[2][:3, :3].T @ other[2][:3, 3]))
            < float(translation_nms_m)
            and _rotation_distance_degrees(candidate[2], other[2])
            < float(rotation_nms_deg)
            for other in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= int(maximum_modes):
            break
    return CoarsePoseModes(
        np.asarray([value[2] for value in retained], dtype=np.float64).reshape(-1, 4, 4),
        np.asarray([value[0] for value in retained], dtype=np.float64),
        np.asarray([value[1] for value in retained], dtype=np.int64),
        np.asarray([value[3] for value in retained], dtype=np.int64).reshape(-1, xy.shape[0]),
        np.asarray([value[4] for value in retained], dtype=np.int64).reshape(-1, xy.shape[0]),
        np.asarray([value[5] for value in retained], dtype=np.int64).reshape(-1, 2),
        np.asarray([value[6] for value in retained], dtype=np.int64).reshape(-1, 2),
    )


def generate_parent_then_child_pose_modes(
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    graph: TypedParentGraph,
    camera: ColmapCamera,
    *,
    maximum_modes: int = 32,
    seed_parent_count: int = 64,
    pair_anchor_count: int = 16,
    seed_parent_pair_count: int = 32,
    covisibility_weight: float = 0.75,
    parent_ransac_reprojection_px: float = 96.0,
    child_ransac_reprojection_px: float = 32.0,
    ransac_iterations: int = 6000,
    refinement_iterations: int = 2,
    random_seed: int = 194917,
) -> CoarsePoseModes:
    """Generate configurations without selecting a child before pose exists.

    Context and graph factors first select a physical-parent configuration.
    Parent centers provide only a coarse numerical pose.  Child surface tiles
    are then read out by projection under that pose and refine it iteratively.
    This removes the circular ``unknown pose -> visual child -> pose`` decision
    that creates self-consistent aliases on repeated facades.
    """

    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("typed graph and physical map lineage differ")
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if (
        xy.shape != extent.shape
        or parent_ids.shape != parent_probability.shape
        or parent_ids.shape[0] != xy.shape[0]
        or parent_null.shape != (xy.shape[0],)
    ):
        raise ValueError("parent-then-child pose inputs differ")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    candidate_rows = np.full(parent_ids.shape, -1, dtype=np.int64)
    for support in range(parent_ids.shape[0]):
        for slot in range(parent_ids.shape[1]):
            candidate_rows[support, slot] = row_by_id.get(int(parent_ids[support, slot]), -1)
    valid_parent = (candidate_rows >= 0) & (parent_probability > 0.0)
    global_mass = np.zeros((physical.maplet_ids.size,), dtype=np.float64)
    np.add.at(global_mass, candidate_rows[valid_parent], parent_probability[valid_parent])
    seeds = np.argsort(-global_mass, kind="stable")[: int(seed_parent_count)]
    seeds = seeds[global_mass[seeds] > 0.0]
    covis = graph.covisibility_matrix().astype(np.float64)
    seed_modes: list[tuple[int, ...]] = [(-1,)] + [(int(seed),) for seed in seeds.tolist()]
    pair_anchors = seeds[: int(pair_anchor_count)]
    pairs: list[tuple[float, tuple[int, int]]] = []
    for offset, left in enumerate(pair_anchors.tolist()):
        for right in pair_anchors[offset + 1 :].tolist():
            relation = float(covis[int(left), int(right)])
            if relation <= 0.0:
                continue
            priority = (
                np.log1p(global_mass[int(left)])
                + np.log1p(global_mass[int(right)])
                + np.log(0.05 + 0.95 * relation)
                + 0.25 * float(
                    graph.parent_distinctiveness[int(left)]
                    + graph.parent_distinctiveness[int(right)]
                )
            )
            pairs.append((priority, (int(left), int(right))))
    pairs.sort(key=lambda value: (-value[0], value[1]))
    seed_modes.extend([value[1] for value in pairs[: int(seed_parent_pair_count)]])
    matrix, distortion = camera_matrix_and_distortion(camera)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    cv2.setRNGSeed(int(random_seed))

    for seed_mode in seed_modes:
        compatibility = np.ones_like(parent_probability)
        if seed_mode[0] >= 0:
            safe = np.maximum(candidate_rows, 0)
            log_compatibility = np.zeros_like(parent_probability)
            for seed in seed_mode:
                value = 0.05 + 0.95 * covis[safe, int(seed)]
                value[candidate_rows < 0] = 0.05
                log_compatibility += np.log(np.maximum(value, 1e-8))
            compatibility = np.exp(log_compatibility / float(len(seed_mode)))
            compatibility[candidate_rows < 0] = 0.0
        assignment_score = np.log(np.maximum(parent_probability, 1e-12))
        assignment_score += float(covisibility_weight) * np.log(np.maximum(compatibility, 1e-8))
        assignment_score[~valid_parent] = -np.inf
        chosen_slot = np.argmax(assignment_score, axis=1)
        chosen_parent = candidate_rows[np.arange(xy.shape[0]), chosen_slot]
        chosen_score = assignment_score[np.arange(xy.shape[0]), chosen_slot]
        valid_support = (
            (chosen_parent >= 0) & np.isfinite(chosen_score) & (parent_null < 0.98)
        )

        # A context parent is a region, not a point observation.  Collapse all
        # correlated supports assigned to it before the coarse center solve.
        parent_xyz, parent_xy, parent_weight = [], [], []
        for parent in np.unique(chosen_parent[valid_support]).tolist():
            rows = np.flatnonzero(valid_support & (chosen_parent == int(parent)))
            if rows.size == 0:
                continue
            probability = parent_probability[rows, chosen_slot[rows]]
            probability = np.maximum(probability, 1e-8)
            parent_xyz.append(physical.maplet_centers[int(parent)])
            parent_xy.append(np.sum(xy[rows] * probability[:, None], axis=0) / np.sum(probability))
            parent_weight.append(float(np.sum(probability)))
        if len(parent_xyz) < 6:
            continue
        parent_xyz_array = np.asarray(parent_xyz, dtype=np.float64)
        parent_xy_array = np.asarray(parent_xy, dtype=np.float64)
        if np.unique(np.round(parent_xyz_array, 5), axis=0).shape[0] < 6:
            continue
        try:
            success, rotation, translation, inliers = cv2.solvePnPRansac(
                parent_xyz_array,
                parent_xy_array,
                matrix,
                distortion,
                iterationsCount=int(ransac_iterations),
                reprojectionError=float(parent_ransac_reprojection_px),
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            continue
        if not success or inliers is None or len(inliers) < 6:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = np.asarray(translation).reshape(3)

        child_rows = np.full((xy.shape[0],), -1, dtype=np.int64)
        child_residual = np.full((xy.shape[0],), np.inf, dtype=np.float64)
        for _ in range(int(refinement_iterations)):
            camera_center = -pose[:3, :3].T @ pose[:3, 3]
            for support in np.flatnonzero(valid_support).tolist():
                parent = int(chosen_parent[support])
                start = int(physical.maplet_child_offsets[parent])
                end = int(physical.maplet_child_offsets[parent + 1])
                rows = np.arange(start, end, dtype=np.int64)
                if rows.size == 0:
                    continue
                projected, depth = _project(physical.child_centers[rows], pose, camera)
                view = camera_center[None] - physical.child_centers[rows]
                view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
                incidence = np.sum(physical.child_normals[rows] * view, axis=1)
                front = (physical.maplet_sidedness[parent] == DOUBLE_SIDED) | (incidence >= 0.02)
                residual = np.linalg.norm(projected - xy[support], axis=1)
                residual[(depth <= 0.05) | ~front] = np.inf
                best = int(np.argmin(residual))
                if np.isfinite(residual[best]):
                    child_rows[support] = int(rows[best])
                    child_residual[support] = float(residual[best])

            # Collapse duplicate surface regions so repeated query evidence is
            # not counted as multiple independent 2D-3D constraints.
            child_xyz, child_xy = [], []
            for child in np.unique(child_rows[child_rows >= 0]).tolist():
                supports = np.flatnonzero(child_rows == int(child))
                if supports.size == 0:
                    continue
                weight = np.maximum(parent_probability[supports, chosen_slot[supports]], 1e-8)
                child_xyz.append(physical.child_centers[int(child)])
                child_xy.append(np.sum(xy[supports] * weight[:, None], axis=0) / np.sum(weight))
            if len(child_xyz) < 6:
                break
            child_xyz_array = np.asarray(child_xyz, dtype=np.float64)
            child_xy_array = np.asarray(child_xy, dtype=np.float64)
            try:
                success, rotation, translation, child_inliers = cv2.solvePnPRansac(
                    child_xyz_array,
                    child_xy_array,
                    matrix,
                    distortion,
                    iterationsCount=int(ransac_iterations),
                    reprojectionError=float(child_ransac_reprojection_px),
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_EPNP,
                )
            except cv2.error:
                break
            if not success or child_inliers is None or len(child_inliers) < 6:
                break
            pose[:3, :3] = cv2.Rodrigues(rotation)[0]
            pose[:3, 3] = np.asarray(translation).reshape(3)
            try:
                rotation, translation = cv2.solvePnPRefineLM(
                    child_xyz_array[np.asarray(child_inliers).reshape(-1)],
                    child_xy_array[np.asarray(child_inliers).reshape(-1)],
                    matrix,
                    distortion,
                    rotation,
                    translation,
                )
                pose[:3, :3] = cv2.Rodrigues(rotation)[0]
                pose[:3, 3] = np.asarray(translation).reshape(3)
            except cv2.error:
                pass

        selected = np.flatnonzero(np.isfinite(child_residual) & valid_support)
        if selected.size < 6 or not np.all(np.isfinite(pose)):
            continue
        sigma = np.maximum(10.0, 0.75 * np.linalg.norm(extent[selected], axis=1))
        geometric = -0.5 * np.square(np.minimum(child_residual[selected] / sigma, 6.0))
        identity = np.clip(chosen_score[selected], -12.0, 0.0)
        weight = np.maximum(1.0 - parent_null[selected], 0.05)
        score = float(np.sum(weight * (geometric + 0.25 * identity)) / np.sum(weight))
        support_count = int(np.sum(child_residual[selected] <= 2.0 * sigma))
        padded_seed = np.full((2,), -1, dtype=np.int64)
        padded_seed[: min(2, len(seed_mode))] = np.asarray(seed_mode[:2], dtype=np.int64)
        candidates.append((
            score, support_count, pose.copy(), chosen_parent.copy(),
            child_rows.copy(), padded_seed,
        ))

    candidates.sort(key=lambda value: (-value[0], -value[1]))
    retained: list[tuple[float, int, np.ndarray]] = []
    for candidate in candidates:
        center = -candidate[2][:3, :3].T @ candidate[2][:3, 3]
        if any(
            np.linalg.norm(center - (-other[2][:3, :3].T @ other[2][:3, 3])) < 0.20
            and _rotation_distance_degrees(candidate[2], other[2]) < 3.0
            for other in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= int(maximum_modes):
            break
    return CoarsePoseModes(
        np.asarray([value[2] for value in retained], dtype=np.float64).reshape(-1, 4, 4),
        np.asarray([value[0] for value in retained], dtype=np.float64),
        np.asarray([value[1] for value in retained], dtype=np.int64),
        np.asarray([value[3] for value in retained], dtype=np.int64).reshape(-1, xy.shape[0]),
        np.asarray([value[4] for value in retained], dtype=np.int64).reshape(-1, xy.shape[0]),
        np.asarray([value[5] for value in retained], dtype=np.int64).reshape(-1, 2),
    )
