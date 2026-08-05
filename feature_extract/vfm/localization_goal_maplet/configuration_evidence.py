"""Complete runtime evidence for one Goal-Maplet pose-configuration set."""

from __future__ import annotations

import numpy as np

from .canonical_field import CanonicalSurfaceField
from .child_eligibility import ChildGeometryEligibility
from .child_local_factor import (
    NULL_TYPES,
    ChildLocalFactorCalibratorArtifact,
    child_local_factor_runtime_features,
)
from .child_local_likelihood import predict_child_local_surface_likelihood
from .child_local_mode_ranker import child_local_mode_runtime_features
from .child_retrieval import ChildTilePosterior
from .latent_configuration import (
    capacitated_assignment,
    correlated_support_clusters,
    effective_group_weights,
    independent_assignment,
    weighted_mean,
    weighted_quantile,
)
from .physical_map import GoalMapletPhysicalMap
from .typed_graph import TypedParentGraph


FEATURE_NAMES = (
    "configuration_assigned_group_fraction",
    "configuration_valid_factor_mass_mean",
    "configuration_valid_factor_log_mean",
    "configuration_wrong_child_mass_mean",
    "configuration_pose_incompatible_mass_mean",
    "configuration_unresolved_mass_mean",
    "configuration_field_missing_mass_mean",
    "configuration_selected_mode_mass_mean",
    "configuration_mode_entropy_mean",
    "configuration_normalized_reprojection_median",
    "configuration_normalized_reprojection_p90",
    "configuration_unique_child_fraction",
    "configuration_child_collision_fraction",
    "configuration_unique_parent_fraction",
    "configuration_refinement_eligible_fraction",
    "configuration_graph_covisibility_mean",
    "configuration_retrieval_joint_probability_mean",
    "configuration_retrieval_joint_log_mean",
)


LATENT_FEATURE_NAMES = (
    "configuration_fixed_eligible_group_fraction",
    "configuration_legacy_valid_factor_mass_mean",
    "configuration_fixed_valid_factor_mass_unweighted_mean",
    "configuration_fixed_valid_factor_mass_mean",
    "configuration_fixed_valid_factor_mass_q10",
    "configuration_fixed_valid_factor_mass_median",
    "configuration_fixed_valid_factor_mass_q90",
    "configuration_fixed_wrong_child_mass_mean",
    "configuration_fixed_pose_incompatible_mass_mean",
    "configuration_fixed_unresolved_mass_mean",
    "configuration_fixed_field_missing_mass_mean",
    "configuration_latent_uncap_assigned_group_fraction",
    "configuration_latent_uncap_valid_mass_mean",
    "configuration_latent_cap_assigned_group_fraction",
    "configuration_latent_cap_valid_mass_mean",
    "configuration_latent_cap_valid_mass_q10",
    "configuration_latent_cap_valid_mass_median",
    "configuration_latent_cap_valid_mass_q90",
    "configuration_latent_typed_consistency_mass_mean",
    "configuration_latent_pose_incompatible_mass_mean",
    "configuration_latent_assignment_margin_mean",
    "configuration_latent_normalized_reprojection_median",
    "configuration_latent_normalized_reprojection_p90",
    "configuration_latent_unique_child_fraction",
    "configuration_latent_unique_parent_fraction",
    "configuration_latent_graph_covisibility_mean",
    "configuration_latent_child_capacity_utilization_mean",
    "configuration_latent_child_capacity_utilization_max",
    "configuration_latent_spatial_bin_coverage",
    "configuration_effective_group_fraction",
    "configuration_latent_uncap_primitive_collision_fraction",
    "configuration_latent_uncap_child_overflow_fraction",
)


def configuration_candidate_evidence(
    poses_w2c: np.ndarray,
    query_descriptors: np.ndarray,
    query_xy_px: np.ndarray,
    query_scale_px: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    graph: TypedParentGraph,
    eligibility: ChildGeometryEligibility,
    factor_calibrator: ChildLocalFactorCalibratorArtifact,
    camera,
    *,
    maximum_groups: int = 64,
    maximum_children: int = 4,
    temperature: float = 0.07,
    maximum_modes: int = 8,
) -> np.ndarray:
    """Aggregate latent child/mode evidence without fixing an assignment first."""

    poses = np.asarray(poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    descriptor = np.asarray(query_descriptors, dtype=np.float32)
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if descriptor.shape[0] != xy.shape[0] or xy.shape[0] != scale.size:
        raise ValueError("configuration query evidence differs")
    if parent_ids.shape != parent_probability.shape or parent_ids.shape[0] != xy.shape[0]:
        raise ValueError("configuration parent evidence differs")
    priority = 1.0 - parent_null
    groups = np.argsort(-priority, kind="stable")[: int(maximum_groups)]
    groups = groups[priority[groups] > 0.02]
    output = np.zeros((poses.shape[0], len(FEATURE_NAMES)), dtype=np.float64)
    if groups.size == 0:
        return output.astype(np.float32)
    child_rows = child_posterior.candidate_child_rows[groups, : int(maximum_children)]
    child_probability = child_posterior.candidate_probabilities[groups, : int(maximum_children)]
    option_valid = (child_rows >= 0) & (child_probability > 0.0)
    option_valid &= eligibility.proposal_qualified[np.maximum(child_rows, 0)]
    group_index, option_index = np.nonzero(option_valid)
    if group_index.size == 0:
        return output.astype(np.float32)
    selected_groups = groups[group_index]
    selected_children = child_rows[group_index, option_index]
    selected_joint = child_probability[group_index, option_index]
    selected_parent_rows = physical.child_parent_rows[selected_children]
    selected_parent_ids = physical.maplet_ids[selected_parent_rows]
    selected_parent_probability = np.asarray([
        np.sum(parent_probability[group][parent_ids[group] == parent_id])
        for group, parent_id in zip(selected_groups.tolist(), selected_parent_ids.tolist())
    ], dtype=np.float64)
    selected_parent_null = parent_null[selected_groups]
    repeated_descriptor = descriptor[selected_groups]
    repeated_xy = xy[selected_groups]
    repeated_scale = scale[selected_groups]
    covisibility = graph.covisibility_matrix().astype(np.float64)
    valid_index = NULL_TYPES.index("valid")
    wrong_index = NULL_TYPES.index("wrong_child")
    pose_index = NULL_TYPES.index("pose_incompatible")
    unresolved_index = NULL_TYPES.index("unresolved")
    missing_index = NULL_TYPES.index("field_missing")
    # Fixed denominator: VFM proposes local modes once.  Candidate geometry
    # may evaluate those modes but must not change their identity/probability,
    # otherwise a bad pose can manufacture evidence that validates itself.
    likelihood = predict_child_local_surface_likelihood(
        repeated_descriptor, selected_children, physical, field,
        temperature=float(temperature), maximum_modes=int(maximum_modes),
    )
    raw_mode_probability = np.asarray(likelihood.mode_probabilities, dtype=np.float64)
    raw_mode_mass = np.sum(raw_mode_probability, axis=1)
    normalized_mode_probability = raw_mode_probability / np.maximum(raw_mode_mass[:, None], 1e-12)
    mode_mass = np.max(normalized_mode_probability, axis=1)
    mode_entropy = -np.sum(
        normalized_mode_probability * np.log(np.maximum(normalized_mode_probability, 1e-12)), axis=1
    )
    for pose_row, pose in enumerate(poses):
        mode_feature, mode_valid = child_local_mode_runtime_features(
            likelihood, selected_children, repeated_xy, repeated_scale,
            pose, camera, physical, field,
        )
        factor_feature = child_local_factor_runtime_features(
            likelihood, mode_feature, mode_valid, selected_children, physical,
            parent_probability=selected_parent_probability,
            child_probability=selected_joint,
            parent_null_probability=selected_parent_null,
            query_scale_px=repeated_scale,
            image_diagonal_px=float(np.hypot(camera.width, camera.height)),
        )
        typed = factor_calibrator.predict_typed_probabilities(factor_feature)
        assignment_score = np.log(np.maximum(selected_joint, 1e-12)) + np.log(np.maximum(typed[:, valid_index], 1e-12))
        chosen_flat = []
        for local_group in range(groups.size):
            rows = np.flatnonzero(group_index == local_group)
            if rows.size:
                chosen_flat.append(int(rows[np.argmax(assignment_score[rows])]))
        if not chosen_flat:
            continue
        chosen = np.asarray(chosen_flat, dtype=np.int64)
        valid_mass = typed[chosen, valid_index]
        assigned = valid_mass >= 0.20
        chosen_children = selected_children[chosen]
        chosen_parents = selected_parent_rows[chosen]
        assigned_count = max(int(np.sum(assigned)), 1)
        unique_children = np.unique(chosen_children[assigned]).size if np.any(assigned) else 0
        unique_parents = np.unique(chosen_parents[assigned]).size if np.any(assigned) else 0
        pair_covis = []
        assigned_parents = chosen_parents[assigned]
        if assigned_parents.size > 1:
            left, right = np.triu_indices(assigned_parents.size, 1)
            pair_covis = covisibility[assigned_parents[left], assigned_parents[right]].tolist()
        residual = factor_feature[chosen, 7]
        output[pose_row] = np.asarray([
            float(np.mean(assigned)),
            float(np.mean(valid_mass)),
            float(np.mean(np.log(np.maximum(valid_mass, 1e-12)))),
            float(np.mean(typed[chosen, wrong_index])),
            float(np.mean(typed[chosen, pose_index])),
            float(np.mean(typed[chosen, unresolved_index])),
            float(np.mean(typed[chosen, missing_index])),
            float(np.mean(mode_mass[chosen])),
            float(np.mean(mode_entropy[chosen])),
            float(np.median(residual)),
            float(np.percentile(residual, 90.0)),
            float(unique_children / assigned_count),
            float(1.0 - unique_children / assigned_count),
            float(unique_parents / assigned_count),
            float(np.mean(eligibility.refinement_qualified[chosen_children])),
            float(np.mean(pair_covis)) if pair_covis else 0.0,
            float(np.mean(selected_joint[chosen])),
            float(np.mean(np.log(np.maximum(selected_joint[chosen], 1e-12)))),
        ], dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("configuration factor evidence contains non-finite values")
    return output.astype(np.float32)


def _projected_child_capacities(
    pose_w2c: np.ndarray,
    child_rows: np.ndarray,
    group_extent_px: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
) -> dict[int, int]:
    """Convert projected child area into a conservative support-cluster cap."""

    import cv2

    from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion

    children = np.unique(np.asarray(child_rows, dtype=np.int64).reshape(-1))
    children = children[children >= 0]
    if children.size == 0:
        return {}
    sign = np.asarray([
        [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    local = sign[None] * np.maximum(physical.child_extents[children, None], 1e-3)
    world = np.einsum("nki,nij->nkj", local, physical.child_frames[children])
    world += physical.child_centers[children, None]
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(
        world.reshape(-1, 3), rotation_vector, pose[:3, 3], matrix, distortion,
    )
    projected = projected.reshape(children.size, 8, 2)
    camera_xyz = world @ pose[:3, :3].T + pose[:3, 3]
    low = np.maximum(np.min(projected, axis=1), [0.0, 0.0])
    high = np.minimum(np.max(projected, axis=1), [float(camera.width), float(camera.height)])
    projected_area = np.prod(np.maximum(high - low, 0.0), axis=1)
    projected_area[np.max(camera_xyz[..., 2], axis=1) <= 0.05] = 0.0
    extent = np.asarray(group_extent_px, dtype=np.float64).reshape(-1, 2)
    group_area = 4.0 * np.prod(np.maximum(extent, 1.0), axis=1)
    reference_area = max(float(np.median(group_area)), 64.0)
    # One independent group per comparable projected support, with a small
    # tolerance for boundary overlap.  Correlated groups count only once in
    # the assignment itself.  The cap is deliberately bounded: a single tile
    # must never absorb an entire repeated facade.
    capacity = np.ceil(projected_area / (0.75 * reference_area)).astype(np.int64)
    capacity = np.clip(capacity, 1, 6)
    return {int(child): int(value) for child, value in zip(children.tolist(), capacity.tolist())}


def _spatial_bin_coverage(xy_px: np.ndarray, selected: np.ndarray, camera) -> float:
    xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(selected, dtype=bool).reshape(-1)
    normalized = xy / np.asarray([max(camera.width, 1), max(camera.height, 1)], dtype=np.float64)
    bins = np.clip(np.floor(3.0 * normalized).astype(np.int64), 0, 2)
    occupied = set((3 * bins[:, 1] + bins[:, 0]).tolist())
    assigned = set((3 * bins[mask, 1] + bins[mask, 0]).tolist())
    return float(len(assigned) / max(len(occupied), 1))


def configuration_candidate_latent_evidence(
    poses_w2c: np.ndarray,
    query_descriptors: np.ndarray,
    query_xy_px: np.ndarray,
    query_extent_px: np.ndarray,
    query_scale_px: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_posterior: ChildTilePosterior,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
    graph: TypedParentGraph,
    eligibility: ChildGeometryEligibility,
    factor_calibrator: ChildLocalFactorCalibratorArtifact,
    camera,
    *,
    maximum_groups: int = 64,
    maximum_children: int = 4,
    temperature: float = 0.07,
    maximum_modes: int = 8,
) -> np.ndarray:
    """Score fixed poses with one-mode-per-group and physical capacities.

    Unlike :func:`configuration_candidate_evidence`, every selected query
    group remains in every candidate's denominator.  Groups without an
    eligible child are explicit unresolved nulls, never silently discarded.
    """

    poses = np.asarray(poses_w2c, dtype=np.float64).reshape(-1, 4, 4)
    descriptor = np.asarray(query_descriptors, dtype=np.float32)
    xy = np.asarray(query_xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(query_extent_px, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(query_scale_px, dtype=np.float64).reshape(-1)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    if descriptor.shape[0] != xy.shape[0] or xy.shape != extent.shape or xy.shape[0] != scale.size:
        raise ValueError("latent configuration query evidence differs")
    if parent_ids.shape != parent_probability.shape or parent_ids.shape[0] != xy.shape[0]:
        raise ValueError("latent configuration parent evidence differs")
    priority = 1.0 - parent_null
    groups = np.argsort(-priority, kind="stable")[: int(maximum_groups)]
    groups = groups[priority[groups] > 0.02]
    output = np.zeros((poses.shape[0], len(LATENT_FEATURE_NAMES)), dtype=np.float64)
    if groups.size == 0:
        return output.astype(np.float32)

    group_xy, group_extent = xy[groups], extent[groups]
    cluster = correlated_support_clusters(group_xy, group_extent)
    group_weight = effective_group_weights(cluster)
    child_rows = child_posterior.candidate_child_rows[groups, : int(maximum_children)]
    child_probability = child_posterior.candidate_probabilities[groups, : int(maximum_children)]
    option_valid = (child_rows >= 0) & (child_probability > 0.0)
    option_valid &= eligibility.proposal_qualified[np.maximum(child_rows, 0)]
    group_index, option_index = np.nonzero(option_valid)
    effective_fraction = float(np.unique(cluster).size / groups.size)
    if group_index.size == 0:
        # All groups are explicit unresolved nulls.  The effective-group
        # fraction remains a query property and is therefore still emitted.
        output[:, LATENT_FEATURE_NAMES.index("configuration_fixed_unresolved_mass_mean")] = 1.0
        output[:, LATENT_FEATURE_NAMES.index("configuration_effective_group_fraction")] = effective_fraction
        return output.astype(np.float32)

    selected_groups = groups[group_index]
    selected_children = child_rows[group_index, option_index]
    selected_joint = np.asarray(child_probability[group_index, option_index], dtype=np.float64)
    selected_parent_rows = physical.child_parent_rows[selected_children]
    selected_parent_ids = physical.maplet_ids[selected_parent_rows]
    selected_parent_probability = np.asarray([
        np.sum(parent_probability[group][parent_ids[group] == parent_id])
        for group, parent_id in zip(selected_groups.tolist(), selected_parent_ids.tolist())
    ], dtype=np.float64)
    selected_parent_null = parent_null[selected_groups]
    repeated_descriptor = descriptor[selected_groups]
    repeated_xy = xy[selected_groups]
    repeated_scale = scale[selected_groups]
    likelihood = predict_child_local_surface_likelihood(
        repeated_descriptor, selected_children, physical, field,
        temperature=float(temperature), maximum_modes=int(maximum_modes),
    )
    raw_mode_probability = np.asarray(likelihood.mode_probabilities, dtype=np.float64)
    retained_mode_mass = np.sum(raw_mode_probability, axis=1)
    conditional_mode_probability = raw_mode_probability / np.maximum(retained_mode_mass[:, None], 1e-12)
    # Child probabilities are only used conditionally within the fixed Top-C
    # set.  Absolute retrieval null remains represented by the typed factor;
    # otherwise the very diffuse Top-64 parent posterior would force every
    # fine mode into null before pose compatibility can be tested.
    child_denominator = np.bincount(group_index, weights=selected_joint, minlength=groups.size)
    conditional_child_probability = selected_joint / np.maximum(child_denominator[group_index], 1e-12)
    covisibility = graph.covisibility_matrix().astype(np.float64)
    valid_index = NULL_TYPES.index("valid")
    wrong_index = NULL_TYPES.index("wrong_child")
    pose_index = NULL_TYPES.index("pose_incompatible")
    unresolved_index = NULL_TYPES.index("unresolved")
    missing_index = NULL_TYPES.index("field_missing")

    for pose_row, pose in enumerate(poses):
        mode_feature, mode_valid = child_local_mode_runtime_features(
            likelihood, selected_children, repeated_xy, repeated_scale,
            pose, camera, physical, field,
        )
        factor_feature = child_local_factor_runtime_features(
            likelihood, mode_feature, mode_valid, selected_children, physical,
            parent_probability=selected_parent_probability,
            child_probability=selected_joint,
            parent_null_probability=selected_parent_null,
            query_scale_px=repeated_scale,
            image_diagonal_px=float(np.hypot(camera.width, camera.height)),
        )
        typed = factor_calibrator.predict_typed_probabilities(factor_feature)

        # Corrected independent-factor baseline.  The child is selected as in
        # G11, but missing groups contribute zero valid mass and unresolved
        # null mass to a candidate-independent denominator.
        factor_score = np.log(np.maximum(selected_joint, 1e-12)) + np.log(
            np.maximum(typed[:, valid_index], 1e-12)
        )
        fixed_factor = np.full((groups.size,), -1, dtype=np.int64)
        for local_group in range(groups.size):
            rows = np.flatnonzero(group_index == local_group)
            if rows.size:
                fixed_factor[local_group] = int(rows[np.argmax(factor_score[rows])])
        fixed_eligible = fixed_factor >= 0
        safe_fixed = np.maximum(fixed_factor, 0)
        fixed_typed = np.zeros((groups.size, len(NULL_TYPES)), dtype=np.float64)
        fixed_typed[:, unresolved_index] = 1.0
        fixed_typed[fixed_eligible] = typed[safe_fixed[fixed_eligible]]
        fixed_valid = fixed_typed[:, valid_index]

        factor_row, mode_row = np.nonzero(mode_valid)
        option_group = group_index[factor_row]
        option_child = selected_children[factor_row]
        option_primitive = np.asarray(likelihood.mode_primitive_rows, dtype=np.int64)[factor_row, mode_row]
        option_score = (
            np.log(np.maximum(conditional_child_probability[factor_row], 1e-12))
            + np.log(np.maximum(typed[factor_row, valid_index], 1e-12))
            + np.log(np.maximum(conditional_mode_probability[factor_row, mode_row], 1e-12))
            + np.clip(np.asarray(mode_feature, dtype=np.float64)[factor_row, mode_row, 5], -12.0, 0.0)
        )
        option_valid_mass = typed[factor_row, valid_index]
        uncap = independent_assignment(
            group_count=groups.size,
            option_group_rows=option_group,
            option_child_rows=option_child,
            option_primitive_rows=option_primitive,
            option_factor_rows=factor_row,
            option_scores=option_score,
            option_valid_mass=option_valid_mass,
        )
        capacities = _projected_child_capacities(
            pose, selected_children, group_extent, physical, camera,
        )
        cap = capacitated_assignment(
            group_count=groups.size,
            option_group_rows=option_group,
            option_child_rows=option_child,
            option_primitive_rows=option_primitive,
            option_factor_rows=factor_row,
            option_scores=option_score,
            option_valid_mass=option_valid_mass,
            group_cluster_rows=cluster,
            child_capacities=capacities,
        )

        cap_typed = fixed_typed.copy()
        rejected_valid = cap_typed[:, valid_index].copy()
        cap_typed[:, valid_index] = 0.0
        cap_typed[:, unresolved_index] += rejected_valid
        cap_typed[cap.assigned] = typed[cap.factor_rows[cap.assigned]]
        typed_consistency = (
            cap.valid_mass
            + 0.5 * cap_typed[:, unresolved_index]
            + cap_typed[:, missing_index]
        )

        assigned_count = max(int(np.sum(cap.assigned)), 1)
        assigned_children = cap.child_rows[cap.assigned]
        assigned_parents = physical.child_parent_rows[assigned_children]
        unique_children = np.unique(assigned_children).size if assigned_children.size else 0
        unique_parents = np.unique(assigned_parents).size if assigned_parents.size else 0
        pair_covis = []
        if assigned_parents.size > 1:
            left, right = np.triu_indices(assigned_parents.size, 1)
            pair_covis = covisibility[assigned_parents[left], assigned_parents[right]].tolist()

        utilization = []
        for child in np.unique(assigned_children).tolist():
            owner_clusters = np.unique(cluster[(cap.child_rows == int(child)) & cap.assigned]).size
            utilization.append(owner_clusters / max(int(capacities.get(int(child), 1)), 1))
        utilization_array = np.asarray(utilization, dtype=np.float64)

        collision = np.zeros((groups.size,), dtype=bool)
        for primitive in np.unique(uncap.primitive_rows[uncap.assigned]).tolist():
            rows = np.flatnonzero(uncap.assigned & (uncap.primitive_rows == int(primitive)))
            if np.unique(cluster[rows]).size > 1:
                collision[rows] = True
        overflow_excess, uncap_cluster_count = 0, 0
        for child in np.unique(uncap.child_rows[uncap.assigned]).tolist():
            owners = np.unique(cluster[(uncap.child_rows == int(child)) & uncap.assigned]).size
            uncap_cluster_count += owners
            overflow_excess += max(owners - int(capacities.get(int(child), 1)), 0)

        margin_value = np.zeros((groups.size,), dtype=np.float64)
        for local_group in np.flatnonzero(cap.assigned).tolist():
            values = option_score[option_group == local_group]
            values = np.sort(values)[::-1]
            margin_value[local_group] = float(values[0] - values[1]) if values.size > 1 else 0.0
        residual = np.zeros((groups.size,), dtype=np.float64)
        if np.any(cap.assigned):
            chosen = cap.option_rows[cap.assigned]
            residual[cap.assigned] = np.asarray(mode_feature, dtype=np.float64)[
                factor_row[chosen], mode_row[chosen], 4
            ]
        assigned_weight = group_weight * cap.assigned
        residual_weight = assigned_weight if np.sum(assigned_weight) > 0.0 else group_weight

        output[pose_row] = np.asarray([
            weighted_mean(fixed_eligible, group_weight),
            float(np.mean(fixed_valid[fixed_eligible])) if np.any(fixed_eligible) else 0.0,
            float(np.mean(fixed_valid)),
            weighted_mean(fixed_valid, group_weight),
            weighted_quantile(fixed_valid, group_weight, 0.10),
            weighted_quantile(fixed_valid, group_weight, 0.50),
            weighted_quantile(fixed_valid, group_weight, 0.90),
            weighted_mean(fixed_typed[:, wrong_index], group_weight),
            weighted_mean(fixed_typed[:, pose_index], group_weight),
            weighted_mean(fixed_typed[:, unresolved_index], group_weight),
            weighted_mean(fixed_typed[:, missing_index], group_weight),
            weighted_mean(uncap.assigned, group_weight),
            weighted_mean(uncap.valid_mass, group_weight),
            weighted_mean(cap.assigned, group_weight),
            weighted_mean(cap.valid_mass, group_weight),
            weighted_quantile(cap.valid_mass, group_weight, 0.10),
            weighted_quantile(cap.valid_mass, group_weight, 0.50),
            weighted_quantile(cap.valid_mass, group_weight, 0.90),
            weighted_mean(typed_consistency, group_weight),
            weighted_mean(cap_typed[:, pose_index], group_weight),
            weighted_mean(margin_value, assigned_weight) if np.sum(assigned_weight) > 0.0 else 0.0,
            weighted_quantile(residual, residual_weight, 0.50),
            weighted_quantile(residual, residual_weight, 0.90),
            float(unique_children / assigned_count),
            float(unique_parents / assigned_count),
            float(np.mean(pair_covis)) if pair_covis else 0.0,
            float(np.mean(utilization_array)) if utilization_array.size else 0.0,
            float(np.max(utilization_array)) if utilization_array.size else 0.0,
            _spatial_bin_coverage(group_xy, cap.assigned, camera),
            effective_fraction,
            weighted_mean(collision, group_weight),
            float(overflow_excess / max(uncap_cluster_count, 1)),
        ], dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("latent configuration evidence contains non-finite values")
    return output.astype(np.float32)
