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
