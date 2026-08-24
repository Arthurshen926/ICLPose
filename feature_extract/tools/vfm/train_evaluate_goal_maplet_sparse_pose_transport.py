"""Train and one-shot evaluate the real map-disjoint sparse transport backend.

The split is a single deterministic mapping-route train/dev split, not a
five-fold experiment.  Candidate 0 is a diagnostic GT anchor used only to
shape the local energy; all end-to-end-like reranking metrics exclude it.
The dev labels are consumed only after the fixed final epoch is serialized.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
    TORCH_TRANSPORT_SEMANTICS,
    FrozenSparseTransportEdges,
    build_frozen_sparse_transport_edges,
    differentiable_fixed_kernel_capacity_pose_transport,
    differentiable_sparse_pose_transport,
)
from feature_extract.vfm.localization_goal_maplet.dense_fixed_identity_transport import (
    DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
    DensePoseTransportHierarchyGPU,
    dense_fixed_identity_transport,
)
from feature_extract.vfm.localization_goal_maplet.controlled_pose_stencil import (
    CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
    pose_transport_hierarchy_content_sha256,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
    build_pose_transport_hierarchy,
)
from feature_extract.vfm.localization_goal_maplet.pose_transport_training import (
    PoseTransportTrainingConfig,
    pose_transport_energy_landscape_loss,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    MinimalPoseTransportConfig,
    MinimalPoseTransportReadout,
    QueryPoseHeadOutput,
    pose_transport_model_content_sha256,
    save_minimal_pose_transport_readout,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    ViewConditionedPrimitiveField,
)


REPORT_SCHEMA = "goal_maplet_real_sparse_pose_transport_train_dev_report_v2"
CANONICAL_MAP_FIELD_SEMANTICS = "single_view_independent_canonical_field_control_v1"
VIEW_CONDITIONED_MAP_FIELD_SEMANTICS = (
    "candidate_pose_evaluated_low_rank_view_conditioned_canonical_field_v1"
)
IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS = (
    "shared_radio128_identity_feature_hierarchy_layout_reference_gauge_two_dof_v3"
)
FULL_MINIMAL_READOUT_CONTROL_SEMANTICS = "full_minimal_pose_readout_control_v1"
IDENTITY_TRAINED_EDGE_COMPONENTS = (0, 5)
IDENTITY_REFERENCE_EDGE_COMPONENT = 4
IDENTITY_DISABLED_EDGE_COMPONENTS = (1, 2, 3)

# Candidate rows emitted by ``_controlled_pose_candidates`` are six independent
# radial paths through SE(3), not one total ordering.  In particular, the + and
# - perturbations at a common radius have equal supervision error and must never
# be forced into an arbitrary ordering.  Keeping this contract here makes the
# only GT-relative structure consumed by the trainer explicit and testable.
CONTROLLED_RADIAL_PATHS = (
    (0, 1, 3, 5),
    (0, 2, 4, 6),
    (0, 7, 9, 11),
    (0, 8, 10, 12),
    (0, 13, 15, 17),
    (0, 14, 16, 18),
)
CONTROLLED_CANDIDATE_SEMANTICS = {
    "controlled_local_oracle_v1",
    CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS,
}
_STAGE_RADII = {
    "coarse": (2.0, 45.0),
    "medium": (1.0, 15.0),
    "fine": (0.5, 5.0),
}
_TRANSPORT_DEPTH_SEMANTICS = {
    "coarse": "ordinal_depth_v1",
    "medium": "centered_log_depth_v1",
    "fine": "metric_log_depth_with_uncertainty_v1",
}
_TRANSPORT_SPATIAL_RADIUS = {"coarse": 3, "medium": 2, "fine": 0}
FIXED_IDENTITY_SUFFICIENT_STATISTICS_SEMANTICS = (
    "fixed_kernel_identity_linear_edge_component_sufficient_statistics_v1"
)
DENSE_FIXED_IDENTITY_STATISTICS_BACKEND = "dense_exact"
SPARSE_FIXED_IDENTITY_STATISTICS_BACKEND = "sparse_authority"


def _feature_only_query(
    value: QueryPoseHeadOutput, *, transport_stage: str = "medium",
) -> QueryPoseHeadOutput:
    """Remove randomly initialized geometry/confidence from the P1 readout.

    The frozen surface mapper already aligns the query with the canonical
    128-D RADIO map.  Six pose-labelled images cannot identify a new dense
    normal/depth/confidence head.  The primary small-data experiment therefore
    preserves the mapped RADIO code exactly and exposes only its validity as
    feature confidence.  Geometry heads remain available in the explicit
    full-readout negative control.
    """

    if str(transport_stage) not in _TRANSPORT_DEPTH_SEMANTICS:
        raise ValueError("feature-only query transport stage differs")
    confidence = torch.zeros_like(value.confidence)
    confidence[:, 0] = value.pose_code_valid.to(dtype=confidence.dtype)
    return QueryPoseHeadOutput(
        pose_code=value.pose_code,
        normal_camera=value.normal_camera,
        relative_depth=value.relative_depth,
        boundary=value.boundary,
        confidence=confidence,
        pose_code_valid=value.pose_code_valid,
        normal_valid=torch.zeros_like(value.normal_valid),
        depth_valid=torch.zeros_like(value.depth_valid),
        boundary_valid=torch.zeros_like(value.boundary_valid),
        normal_frame=value.normal_frame,
        # The depth value and validity are both disabled above.  This tag only
        # satisfies the explicit stage schema so coarse/fine *structural*
        # relation/radius controls can reuse the RADIO-only observation.  It
        # must not be interpreted as a coarse/fine depth observation.
        depth_semantics=_TRANSPORT_DEPTH_SEMANTICS[str(transport_stage)],
    )


def _identity_edge_gradient_mask(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the identifiable RADIO/layout degrees of freedom.

    Hierarchy is held at softplus(0) as the reference gauge.  Normal, depth,
    and boundary are absent observations and are fixed near zero.  This avoids
    letting inactive weights act only through the compatibility denominator.
    """

    mask = torch.zeros(6, device=device, dtype=dtype)
    mask[list(IDENTITY_TRAINED_EDGE_COMPONENTS)] = 1.0
    return mask


def _rankdata(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    rank = np.empty(order.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and values[order[stop]] == values[order[start]]:
            stop += 1
        rank[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return rank


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _rankdata(left), _rankdata(right)
    if a.size < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _image_route(image_id: str) -> str:
    route, separator, relative = str(image_id).partition("/")
    if not separator or not route or not relative:
        raise ValueError("transport image ID lacks route-relative structure")
    return route


def _validate_scientific_dataset_contract(
    arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> dict[str, object]:
    """Validate the claims needed by this controlled local-energy experiment.

    The expensive builder already recomputed errors and rendered evidence.  The
    trainer must nevertheless fail closed on the map/query split and GT anchor,
    rather than trusting descriptive JSON fields that do not match the arrays.
    Candidate zero can only be validated through its stored recomputed error;
    the dataset does not contain a second independent GT pose copy.  That
    limitation is recorded in the returned audit instead of being hidden.
    """

    required = {
        "image_ids", "radio_final", "source_child_rows",
        "source_child_probabilities", "query_reliability", "token_xy",
        "candidate_poses_w2c", "translation_m", "rotation_deg",
        "candidate_valid", "target_child_rows", "target_child_weights",
        "target_canonical_features", "target_normals_camera",
        "target_double_sided", "target_relative_depth", "target_boundary",
        "target_modality_valid", "target_modality_confidence",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("transport dataset lacks trainer arrays: " + ",".join(missing))
    image_ids = np.asarray(arrays["image_ids"])
    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    translation = np.asarray(arrays["translation_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_deg"], dtype=np.float64)
    if image_ids.ndim != 1 or image_ids.size < 2 or valid.ndim != 2:
        raise ValueError("transport dataset query/candidate axes differ")
    query_count, candidate_count = valid.shape
    if query_count != image_ids.size or candidate_count < 2:
        raise ValueError("transport dataset requires multiple candidates per query")
    if translation.shape != valid.shape or rotation.shape != valid.shape:
        raise ValueError("transport pose errors differ from candidate validity")
    if (
        np.asarray(arrays["candidate_poses_w2c"]).shape != (query_count, candidate_count, 4, 4)
        or np.any(~np.isfinite(np.asarray(arrays["candidate_poses_w2c"])))
        or np.any(~np.isfinite(translation)) or np.any(~np.isfinite(rotation))
        or np.any(translation < 0.0) or np.any(rotation < 0.0)
    ):
        raise ValueError("transport candidate poses/errors are invalid")
    expected_prefixes = {
        "radio_final": (query_count,),
        "source_child_rows": (query_count,),
        "source_child_probabilities": (query_count,),
        "query_reliability": (query_count,),
        "token_xy": (query_count,),
        "target_child_rows": (query_count, candidate_count),
        "target_child_weights": (query_count, candidate_count),
        "target_canonical_features": (query_count, candidate_count),
        "target_normals_camera": (query_count, candidate_count),
        "target_double_sided": (query_count, candidate_count),
        "target_relative_depth": (query_count, candidate_count),
        "target_boundary": (query_count, candidate_count),
        "target_modality_valid": (query_count, candidate_count),
        "target_modality_confidence": (query_count, candidate_count),
    }
    for name, prefix in expected_prefixes.items():
        if np.asarray(arrays[name]).shape[:len(prefix)] != prefix:
            raise ValueError(f"transport array {name} has the wrong leading axes")
    identifiers = [str(value) for value in image_ids.tolist()]
    if len(set(identifiers)) != len(identifiers) or identifiers != sorted(identifiers):
        raise ValueError("transport query IDs must be unique and deterministically sorted")
    routes = {_image_route(value) for value in identifiers}
    query_route = str(metadata.get("query_route", ""))
    mapping_routes = {str(value) for value in metadata.get("map_training_routes", ())}
    if routes != {query_route} or not query_route:
        raise ValueError("transport query route metadata differs from image IDs")
    if query_route in mapping_routes:
        raise ValueError("transport query route occurs in canonical map training routes")
    if (
        int(metadata.get("query_count", -1)) != query_count
        or int(metadata.get("candidate_count", -1)) != candidate_count
    ):
        raise ValueError("transport metadata counts differ from arrays")
    if (
        metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
        or metadata.get("pose_errors_recomputed") is not True
        or not np.all(valid[:, 0])
        or np.any(np.abs(translation[:, 0]) > 1.0e-5)
        or np.any(np.abs(rotation[:, 0]) > 1.0e-4)
    ):
        raise ValueError("transport candidate zero is not a valid zero-error GT anchor")
    if not np.all(np.sum(valid[:, 1:], axis=1) >= 1):
        raise ValueError("every transport query requires a valid non-anchor candidate")
    candidate_semantics = str(metadata.get("candidate_semantics", ""))
    controlled = candidate_semantics in CONTROLLED_CANDIDATE_SEMANTICS
    if controlled and metadata.get(
        "controlled_candidates_are_gt_relative_oracle_diagnostic"
    ) is not True:
        raise ValueError("controlled transport candidates lack GT-relative diagnostic contract")
    embedded_paths = arrays.get("controlled_radial_paths")
    if candidate_semantics == CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS:
        if embedded_paths is None:
            raise ValueError("quadratic controlled dataset lacks embedded radial paths")
        path_array = np.asarray(embedded_paths, dtype=np.int64)
        stencil_audit = metadata.get("controlled_pose_stencil_audit")
        if (
            path_array.ndim != 2 or path_array.shape[0] <= 0 or path_array.shape[1] < 2
            or np.any(path_array < 0) or np.any(path_array >= candidate_count)
            or not np.all(path_array[:, 0] == 0)
            or not isinstance(stencil_audit, dict)
            or stencil_audit.get("local_quadratic_6dof_identifiable") is not True
            or metadata.get("full_6dof_observability_stencil") is not True
        ):
            raise ValueError("quadratic controlled radial-path contract differs")
    return {
        "query_route": query_route,
        "map_training_routes": sorted(mapping_routes),
        "map_query_route_intersection": sorted(routes & mapping_routes),
        "query_ids_unique_and_sorted": True,
        "candidate_zero_valid_for_every_query": True,
        "candidate_zero_max_translation_error_m": float(np.max(translation[:, 0])),
        "candidate_zero_max_rotation_error_deg": float(np.max(rotation[:, 0])),
        "candidate_zero_pose_independently_recoverable_from_dataset": False,
        "candidate_zero_validation_semantics": (
            "builder_recomputed_zero_error_and_lineage_contract_not_independent_gt_copy_v1"
        ),
        "controlled_radial_stencil_complete": bool(
            (candidate_semantics == "controlled_local_oracle_v1" and candidate_count >= 19)
            or (
                candidate_semantics == CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS
                and embedded_paths is not None
            )
        ),
        "controlled_radial_paths_embedded": embedded_paths is not None,
        "full_6dof_observability_stencil": bool(
            metadata.get("full_6dof_observability_stencil", False)
        ),
        "local_quadratic_6dof_identifiable": bool(
            metadata.get("local_quadratic_6dof_identifiable", False)
        ),
    }


def _controlled_paths_for_row(
    candidate_valid: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    *,
    candidate_semantics: str,
    radial_paths: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    semantics = str(candidate_semantics)
    if semantics not in CONTROLLED_CANDIDATE_SEMANTICS:
        return ()
    valid = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    translation = np.asarray(translation_m, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_deg, dtype=np.float64).reshape(-1)
    if translation.shape != valid.shape or rotation.shape != valid.shape:
        raise ValueError("controlled path pose errors differ")
    if semantics == CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS:
        if radial_paths is None:
            raise ValueError("quadratic controlled candidates require embedded radial paths")
        source_paths = np.asarray(radial_paths, dtype=np.int64)
        if source_paths.ndim != 2 or source_paths.shape[1] < 2:
            raise ValueError("embedded controlled radial paths must be a matrix")
        path_values = [tuple(int(value) for value in row) for row in source_paths]
    else:
        if radial_paths is not None:
            raise ValueError("legacy controlled stencil cannot carry unrelated radial paths")
        path_values = list(CONTROLLED_RADIAL_PATHS)
    result: list[tuple[int, ...]] = []
    for full_path in path_values:
        if semantics == CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS:
            if (
                full_path[0] != 0 or len(set(full_path)) != len(full_path)
                or min(full_path) < 0 or max(full_path) >= valid.size
                or not np.all(valid[list(full_path)])
            ):
                raise ValueError("embedded controlled radial path references invalid candidates")
            path = full_path
        else:
            path = tuple(index for index in full_path if index < valid.size and valid[index])
        if len(path) < 2:
            continue
        joint = np.maximum(translation[list(path)] / 1.0, rotation[list(path)] / 15.0)
        if np.any(np.diff(joint) <= 1.0e-8):
            raise ValueError("controlled radial path does not move strictly away from GT")
        result.append(path)
    return tuple(result)


def _controlled_monotonic_pair_rows(
    candidate_valid: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    *,
    candidate_semantics: str,
    radial_paths: np.ndarray | None = None,
) -> np.ndarray:
    paths = _controlled_paths_for_row(
        candidate_valid, translation_m, rotation_deg,
        candidate_semantics=str(candidate_semantics),
        radial_paths=radial_paths,
    )
    return np.asarray(
        [[0, first, second] for path in paths for first, second in zip(path[:-1], path[1:])],
        dtype=np.int64,
    ).reshape(-1, 3)


def _mapper_supervision_audit(
    metadata: dict[str, object], *, query_image_ids: list[str], query_route: str,
) -> dict[str, object]:
    supervised: set[str] = set()

    def collect(value: object) -> None:
        if not isinstance(value, dict):
            return
        for key in ("training_images", "validation_images"):
            rows = value.get(key, ())
            if isinstance(rows, list):
                supervised.update(str(row) for row in rows)
        collect(value.get("initial_checkpoint_metadata"))

    collect(metadata)
    routes = sorted({_image_route(value) for value in supervised})
    exact_overlap = sorted(set(query_image_ids) & supervised)
    return {
        "supervised_image_count_recursive": len(supervised),
        "supervised_routes_recursive": routes,
        "query_route": str(query_route),
        "query_route_in_mapper_supervision": str(query_route) in routes,
        "exact_query_image_overlap_count": len(exact_overlap),
        "exact_query_image_overlap": exact_overlap,
        "strict_query_representation_route_disjoint": str(query_route) not in routes,
    }


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("transport dataset lacks metadata")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    if metadata.get("artifact_type") != "goal_maplet_real_sparse_pose_transport_dataset_v2":
        raise ValueError("not a real sparse pose transport dataset")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("transport dataset content hash differs")
    required_false = ("uses_alike", "uses_point_correspondences", "uses_pnp", "uses_absolute_pose_regression")
    if any(metadata.get(key) is not False for key in required_false):
        raise ValueError("transport dataset violates method boundary")
    if metadata.get("canonical_map_excludes_query_route") is not True:
        raise ValueError("transport dataset is not map-disjoint")
    map_field_semantics = str(metadata.get("map_pose_field_semantics", ""))
    if map_field_semantics not in {
        CANONICAL_MAP_FIELD_SEMANTICS, VIEW_CONDITIONED_MAP_FIELD_SEMANTICS,
    }:
        raise ValueError("transport dataset map-pose field semantics differ")
    view_hash = metadata.get("view_conditioned_field_sha256")
    view_file_hash = metadata.get("view_conditioned_field_file_sha256")
    if map_field_semantics == VIEW_CONDITIONED_MAP_FIELD_SEMANTICS:
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (view_hash, view_file_hash)
        ):
            raise ValueError("view-conditioned transport dataset lacks field lineage")
    elif view_hash is not None or view_file_hash is not None:
        raise ValueError("canonical-field control carries view-conditioned lineage")
    hierarchy_names = (
        "hierarchy_child_parent_ids",
        "hierarchy_child_support_ids",
        "hierarchy_adjacency_offsets",
        "hierarchy_adjacency_child_rows",
    )
    if any(name not in arrays for name in hierarchy_names):
        raise ValueError("transport dataset lacks replayable hierarchy arrays")
    hierarchy_hash = pose_transport_hierarchy_content_sha256(
        arrays[hierarchy_names[0]], arrays[hierarchy_names[1]],
        arrays[hierarchy_names[2]], arrays[hierarchy_names[3]],
    )
    if (
        metadata.get("hierarchy_content_sha256") != hierarchy_hash
        or metadata.get("hierarchy_semantics") != HIERARCHY_SEMANTICS
    ):
        raise ValueError("transport dataset hierarchy lineage differs")
    return arrays, metadata


def _ray_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height) * 2.0 - 1.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width) * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)[None]


def _edge_for_candidate(
    arrays: dict[str, np.ndarray], query: int, candidate: int, hierarchy, *, stage: str,
) -> FrozenSparseTransportEdges:
    if not bool(arrays["candidate_valid"][query, candidate]):
        return FrozenSparseTransportEdges(
            source_index=np.zeros(0, dtype=np.int64),
            target_index=np.zeros(0, dtype=np.int64),
            hierarchy_score=np.zeros(0, dtype=np.float32),
            layout_score=np.zeros(0, dtype=np.float32),
            source_count=int(arrays["source_child_rows"][query].size),
            target_count=int(arrays["target_child_rows"][query, candidate].size),
            stage=str(stage),
        )
    return build_frozen_sparse_transport_edges(
        arrays["source_child_rows"][query],
        arrays["source_child_probabilities"][query],
        arrays["token_xy"][query],
        arrays["target_child_rows"][query, candidate],
        arrays["target_child_weights"][query, candidate],
        hierarchy, stage=str(stage),
        minimum_source_probability=1.0e-6,
        minimum_target_weight=1.0e-6,
    )


def _edges_for_query(
    arrays: dict[str, np.ndarray], query: int, hierarchy, *, stage: str,
) -> list[FrozenSparseTransportEdges]:
    return [
        _edge_for_candidate(arrays, query, candidate, hierarchy, stage=str(stage))
        for candidate in range(arrays["candidate_valid"].shape[1])
    ]


def _query_scores(
    model: MinimalPoseTransportReadout,
    arrays: dict[str, np.ndarray],
    query_index: int,
    edge_rows: list[FrozenSparseTransportEdges],
    *,
    device: torch.device,
    gradient: bool,
    transport_semantics: str,
    readout_training_semantics: str,
) -> torch.Tensor:
    edge_stages = {str(value.stage) for value in edge_rows}
    if len(edge_stages) != 1:
        raise ValueError("one query score batch must use one transport stage")
    transport_stage = next(iter(edge_stages))
    context = torch.enable_grad() if gradient else torch.no_grad()
    with context:
        query_feature_key = (
            "pose_query_features" if "pose_query_features" in arrays else "radio_final"
        )
        radio = torch.as_tensor(
            arrays[query_feature_key][query_index], device=device, dtype=torch.float32
        )[None]
        query = model(radio, _ray_grid(36, 64, device=device))
        if str(readout_training_semantics) == IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS:
            query = _feature_only_query(query, transport_stage=transport_stage)
        elif str(readout_training_semantics) != FULL_MINIMAL_READOUT_CONTROL_SEMANTICS:
            raise ValueError("unknown pose readout training semantics")
        source = torch.as_tensor(
            arrays["source_child_probabilities"][query_index], device=device, dtype=torch.float32
        )
        reliability = torch.as_tensor(
            arrays["query_reliability"][query_index], device=device, dtype=torch.float32
        )
        scores = []
        for candidate, edges in enumerate(edge_rows):
            if not bool(arrays["candidate_valid"][query_index, candidate]):
                scores.append(torch.full((), -1.0, device=device))
                continue
            arguments = (
                model, query, source, reliability,
                torch.as_tensor(
                    arrays["target_child_weights"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_canonical_features"][query_index, candidate],
                    device=device, dtype=torch.float32,
                ),
                torch.as_tensor(
                    arrays["target_normals_camera"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_double_sided"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_relative_depth"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_boundary"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_modality_valid"][query_index, candidate], device=device
                ),
                torch.as_tensor(
                    arrays["target_modality_confidence"][query_index, candidate], device=device
                ),
                edges,
            )
            if str(transport_semantics) == FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS:
                value = differentiable_fixed_kernel_capacity_pose_transport(*arguments)
            elif str(transport_semantics) == TORCH_TRANSPORT_SEMANTICS:
                value = differentiable_sparse_pose_transport(*arguments)
            else:
                raise ValueError("unknown pose transport semantics")
            scores.append(value.combined_score)
        return torch.stack(scores)


def _fixed_identity_scores_from_component_statistics(
    model: MinimalPoseTransportReadout, component_statistics: torch.Tensor,
) -> torch.Tensor:
    """Recover exact fixed-kernel identity scores from linear edge statistics."""

    value = torch.as_tensor(component_statistics)
    if value.ndim != 2 or value.shape[1] != 6 or not torch.isfinite(value).all():
        raise ValueError("fixed identity component statistics must have shape [candidate,6]")
    weights = model.edge_weights().to(device=value.device, dtype=value.dtype)
    return -1.0 + torch.sum(value * weights[None], dim=1) / weights.sum().clamp_min(
        torch.finfo(value.dtype).tiny
    )


def _fixed_identity_query_component_statistics(
    model: MinimalPoseTransportReadout,
    arrays: dict[str, np.ndarray],
    query_index: int,
    hierarchy: PoseTransportHierarchy,
    *,
    stage: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    """Stream one query and retain only six scalar statistics per candidate.

    Under fixed-kernel transport, edge allocation is linear in each evidence
    component and the only shared nonlinearity is division by the sum of six
    positive weights.  The sufficient statistic below performs the exact
    source/token/reliability reductions before discarding the large frozen edge
    arrays.  It turns an otherwise unbounded 85-candidate graph cache into a
    bounded one-candidate working set without changing the trained objective.
    """

    if str(stage) not in _TRANSPORT_SPATIAL_RADIUS:
        raise ValueError("fixed identity statistics stage differs")
    candidate_count = int(arrays["candidate_valid"].shape[1])
    started = time.monotonic()
    edge_counts: list[int] = []
    output = np.zeros((candidate_count, 6), dtype=np.float32)
    with torch.no_grad():
        radio = torch.as_tensor(
            arrays["pose_query_features"][query_index], device=device, dtype=torch.float32
        )[None]
        query = _feature_only_query(
            model(radio, _ray_grid(36, 64, device=device)),
            transport_stage=str(stage),
        )
        query_code = query.pose_code[0].permute(1, 2, 0).reshape(-1, 128)
        query_valid = query.pose_code_valid[0].reshape(-1)
        query_confidence = query.confidence[0, 0].reshape(-1)
        source = torch.as_tensor(
            arrays["source_child_probabilities"][query_index],
            device=device, dtype=torch.float32,
        )
        source_flat = source.reshape(-1)
        source_slots = int(source.shape[1])
        reliability = torch.as_tensor(
            arrays["query_reliability"][query_index], device=device, dtype=torch.float32
        ).reshape(-1)
        reliability_sum = reliability.sum().clamp_min(1.0e-12)
        if query_code.shape[0] != source.shape[0] or reliability.shape[0] != source.shape[0]:
            raise ValueError("fixed identity query/token grids differ")
        confidence_epsilon = torch.as_tensor(1.0e-12, device=device, dtype=torch.float32)
        kernel = 1.0 / float((2 * _TRANSPORT_SPATIAL_RADIUS[str(stage)] + 1) ** 2)
        for candidate in range(candidate_count):
            edges = _edge_for_candidate(
                arrays, query_index, candidate, hierarchy, stage=str(stage)
            )
            edge_count = int(edges.source_index.size)
            edge_counts.append(edge_count)
            if not bool(arrays["candidate_valid"][query_index, candidate]):
                continue
            edge_source = torch.as_tensor(
                edges.source_index, device=device, dtype=torch.long
            )
            edge_target = torch.as_tensor(
                edges.target_index, device=device, dtype=torch.long
            )
            source_token = torch.div(edge_source, source_slots, rounding_mode="floor")
            target_mass = torch.as_tensor(
                arrays["target_child_weights"][query_index, candidate],
                device=device, dtype=torch.float32,
            ).reshape(-1)
            map_code, map_code_valid = model.project_map_code(torch.as_tensor(
                arrays["target_canonical_features"][query_index, candidate],
                device=device, dtype=torch.float32,
            ))
            map_code = map_code.reshape(-1, map_code.shape[-1])
            map_code_valid = map_code_valid.reshape(-1)
            map_modality_valid = torch.as_tensor(
                arrays["target_modality_valid"][query_index, candidate, ..., 0],
                device=device, dtype=torch.bool,
            ).reshape(-1)
            map_confidence = torch.as_tensor(
                arrays["target_modality_confidence"][query_index, candidate, ..., 0],
                device=device, dtype=torch.float32,
            ).reshape(-1)
            feature_cosine = torch.sum(
                query_code[source_token] * map_code[edge_target], dim=1
            )
            feature_similarity = 0.5 * (1.0 + feature_cosine.clamp(-1.0, 1.0))
            confidence_product = (
                query_confidence[source_token] * map_confidence[edge_target]
            )
            confidence = torch.sqrt(
                confidence_product + confidence_epsilon
            ) - torch.sqrt(confidence_epsilon)
            feature_active = (
                query_valid[source_token]
                & map_code_valid[edge_target]
                & map_modality_valid[edge_target]
            )
            feature_component = torch.where(
                feature_active, confidence * feature_similarity, 0.0
            )
            components = torch.stack([
                feature_component,
                torch.as_tensor(
                    edges.hierarchy_score, device=device, dtype=torch.float32
                ).clamp(0.0, 1.0),
                torch.as_tensor(
                    edges.layout_score, device=device, dtype=torch.float32
                ).clamp(0.0, 1.0),
            ], dim=1)
            base = (
                source_flat[edge_source]
                * target_mass[edge_target]
                * float(kernel)
            )
            matched_source = torch.zeros(
                (source_flat.numel(), 3), device=device, dtype=torch.float32
            )
            if edge_source.numel():
                matched_source.index_add_(0, edge_source, base[:, None] * components)
            token_matched = matched_source.reshape(
                source.shape[0], source_slots, 3
            ).sum(dim=1)
            reduced = 2.0 * torch.sum(
                reliability[:, None] * token_matched, dim=0
            ) / reliability_sum
            output[candidate, [0, 4, 5]] = reduced.detach().cpu().numpy()
            del (
                edges, edge_source, edge_target, source_token, target_mass,
                map_code, map_code_valid, map_modality_valid, map_confidence,
                feature_cosine, feature_similarity, confidence_product,
                confidence, feature_active, feature_component, components,
                base, matched_source, token_matched, reduced,
            )
    return output, {
        "query_index": int(query_index),
        "stage": str(stage),
        "candidate_count": candidate_count,
        "total_edge_count": int(sum(edge_counts)),
        "maximum_candidate_edge_count": int(max(edge_counts, default=0)),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_process_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "peak_cuda_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            if device.type == "cuda" else 0.0
        ),
        "peak_cuda_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
            if device.type == "cuda" else 0.0
        ),
        "semantics": FIXED_IDENTITY_SUFFICIENT_STATISTICS_SEMANTICS,
    }


def _dense_fixed_identity_query_component_statistics(
    model: MinimalPoseTransportReadout,
    arrays: dict[str, np.ndarray],
    query_index: int,
    hierarchy: DensePoseTransportHierarchyGPU,
    *,
    stage: str,
    device: torch.device,
    candidate_batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate the exact fixed/identity sufficient statistics in dense batches."""

    if str(stage) not in _TRANSPORT_SPATIAL_RADIUS:
        raise ValueError("dense fixed identity statistics stage differs")
    batch_size = int(candidate_batch_size)
    if batch_size <= 0:
        raise ValueError("dense fixed identity candidate batch size must be positive")
    candidate_valid = np.asarray(arrays["candidate_valid"][query_index], dtype=bool)
    candidate_count = int(candidate_valid.size)
    output = np.zeros((candidate_count, 6), dtype=np.float32)
    valid_indices = np.flatnonzero(candidate_valid)
    started = time.monotonic()
    maximum_source_excess = 0.0
    maximum_target_excess = 0.0
    processed_batches = 0
    with torch.no_grad():
        radio = torch.as_tensor(
            arrays["pose_query_features"][query_index],
            device=device, dtype=torch.float32,
        )[None]
        query = _feature_only_query(
            model(radio, _ray_grid(36, 64, device=device)),
            transport_stage=str(stage),
        )
        for begin in range(0, valid_indices.size, batch_size):
            indices = valid_indices[begin : begin + batch_size]
            result = dense_fixed_identity_transport(
                model,
                query,
                arrays["source_child_rows"][query_index],
                arrays["source_child_probabilities"][query_index],
                arrays["query_reliability"][query_index],
                arrays["token_xy"][query_index],
                arrays["target_child_rows"][query_index, indices],
                arrays["target_child_weights"][query_index, indices],
                arrays["target_canonical_features"][query_index, indices],
                arrays["target_modality_confidence"][query_index, indices, ..., 0],
                arrays["target_modality_valid"][query_index, indices, ..., 0],
                hierarchy,
                stage=str(stage),
                height=36,
                width=64,
            )
            output[indices] = result.component_statistics.detach().cpu().numpy()
            maximum_source_excess = max(
                maximum_source_excess, float(result.maximum_source_capacity_excess)
            )
            maximum_target_excess = max(
                maximum_target_excess, float(result.maximum_target_capacity_excess)
            )
            processed_batches += 1
    return output, {
        "query_index": int(query_index),
        "stage": str(stage),
        "candidate_count": candidate_count,
        "valid_candidate_count": int(valid_indices.size),
        "candidate_batch_size": batch_size,
        "processed_candidate_batch_count": processed_batches,
        "maximum_source_capacity_excess": maximum_source_excess,
        "maximum_target_capacity_excess": maximum_target_excess,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_process_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "peak_cuda_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            if device.type == "cuda" else 0.0
        ),
        "peak_cuda_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
            if device.type == "cuda" else 0.0
        ),
        "semantics": DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
    }


def _release_streaming_cuda_cache(
    device: torch.device, audit: dict[str, object],
) -> None:
    """Release allocator fragments after a streamed query without changing values."""

    if device.type != "cuda":
        audit.update({
            "cuda_cache_release_at_query_boundary": False,
            "cuda_allocated_after_query_mib": 0.0,
            "cuda_reserved_before_empty_cache_mib": 0.0,
            "cuda_reserved_after_empty_cache_mib": 0.0,
        })
        return
    # All retained statistics are already CPU numpy arrays at this boundary.
    # Collecting unreachable tensor wrappers before empty_cache prevents shape-
    # fragmented blocks from accumulating over many independently sized views.
    gc.collect()
    torch.cuda.synchronize(device)
    audit["cuda_allocated_after_query_mib"] = float(
        torch.cuda.memory_allocated(device) / (1024.0 ** 2)
    )
    audit["cuda_reserved_before_empty_cache_mib"] = float(
        torch.cuda.memory_reserved(device) / (1024.0 ** 2)
    )
    torch.cuda.empty_cache()
    audit["cuda_reserved_after_empty_cache_mib"] = float(
        torch.cuda.memory_reserved(device) / (1024.0 ** 2)
    )
    audit["cuda_cache_release_at_query_boundary"] = True


def _full_6dof_direction_metrics(
    score_rows: np.ndarray,
    valid_rows: np.ndarray,
    radial_paths: np.ndarray,
    candidate_direction_ids: np.ndarray,
    candidate_signs: np.ndarray,
    direction_axis_pairs: np.ndarray,
    *,
    twist_order: tuple[str, ...],
) -> dict[str, object]:
    """Measure capture independently along every signed 6-DoF stencil ray."""

    scores = np.asarray(score_rows, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    paths = np.asarray(radial_paths, dtype=np.int64)
    direction_ids = np.asarray(candidate_direction_ids, dtype=np.int64).reshape(-1)
    signs = np.asarray(candidate_signs, dtype=np.int64).reshape(-1)
    axis_pairs = np.asarray(direction_axis_pairs, dtype=np.int64)
    if (
        scores.ndim != 2 or valid.shape != scores.shape
        or paths.ndim != 2 or paths.shape[1] < 2
        or direction_ids.shape != (scores.shape[1],)
        or signs.shape != direction_ids.shape
        or axis_pairs.ndim != 2 or axis_pairs.shape[1] != 2
        or len(twist_order) != 6
    ):
        raise ValueError("full-6dof direction metric arrays differ")
    direction_count = int(axis_pairs.shape[0])
    counters: dict[tuple[int, int], dict[str, int]] = {
        (direction, sign): {
            "pair_correct": 0, "pair_total": 0,
            "complete": 0, "path_total": 0,
        }
        for direction in range(direction_count) for sign in (-1, 1)
    }
    for query_score, query_valid in zip(scores, valid):
        for path in paths:
            indices = [int(value) for value in path.tolist()]
            if indices[0] != 0 or not np.all(query_valid[indices]):
                raise ValueError("full-6dof metric path is not a valid GT-outward ray")
            ray_direction_ids = np.unique(direction_ids[indices[1:]])
            ray_signs = np.unique(signs[indices[1:]])
            if (
                ray_direction_ids.size != 1 or ray_signs.size != 1
                or int(ray_direction_ids[0]) not in range(direction_count)
                or int(ray_signs[0]) not in (-1, 1)
            ):
                raise ValueError("full-6dof metric path mixes direction or sign")
            counter = counters[(int(ray_direction_ids[0]), int(ray_signs[0]))]
            comparisons = [
                bool(query_score[first] > query_score[second])
                for first, second in zip(indices[:-1], indices[1:])
            ]
            counter["pair_correct"] += int(sum(comparisons))
            counter["pair_total"] += len(comparisons)
            counter["complete"] += int(all(comparisons))
            counter["path_total"] += 1

    def summarize(counter_rows: list[dict[str, int]]) -> dict[str, object]:
        pair_correct = int(sum(row["pair_correct"] for row in counter_rows))
        pair_total = int(sum(row["pair_total"] for row in counter_rows))
        complete = int(sum(row["complete"] for row in counter_rows))
        path_total = int(sum(row["path_total"] for row in counter_rows))
        return {
            "radial_pair_correct": pair_correct,
            "radial_pair_count": pair_total,
            "radial_pair_accuracy": float(pair_correct / max(pair_total, 1)),
            "complete_path_count": complete,
            "signed_path_count": path_total,
            "complete_path_rate": float(complete / max(path_total, 1)),
        }

    per_direction = []
    for direction, (first_axis, second_axis) in enumerate(axis_pairs.tolist()):
        if first_axis not in range(6) or second_axis not in {-1, *range(6)}:
            raise ValueError("full-6dof direction axis index differs")
        kind = "coordinate_axis" if second_axis == -1 else "pair_coupling"
        label = str(twist_order[first_axis]) if second_axis == -1 else (
            f"{twist_order[first_axis]}+{twist_order[second_axis]}"
        )
        negative = counters[(direction, -1)]
        positive = counters[(direction, 1)]
        per_direction.append({
            "direction_id": int(direction),
            "label": label,
            "kind": kind,
            "axis_pair": [int(first_axis), int(second_axis)],
            **summarize([negative, positive]),
            "negative_sign": summarize([negative]),
            "positive_sign": summarize([positive]),
        })
    coordinate_rows = [
        counters[(direction, sign)]
        for direction, pair in enumerate(axis_pairs.tolist()) if pair[1] == -1
        for sign in (-1, 1)
    ]
    coupling_rows = [
        counters[(direction, sign)]
        for direction, pair in enumerate(axis_pairs.tolist()) if pair[1] != -1
        for sign in (-1, 1)
    ]
    all_rows = list(counters.values())
    return {
        "semantics": "signed_gt_outward_medium_6dof_radial_capture_v1",
        "twist_order": list(twist_order),
        "direction_count": direction_count,
        "signed_direction_count": len(counters),
        "aggregate": summarize(all_rows),
        "coordinate_axes": summarize(coordinate_rows),
        "pair_couplings": summarize(coupling_rows),
        "minimum_direction_radial_pair_accuracy": float(min(
            row["radial_pair_accuracy"] for row in per_direction
        )),
        "minimum_direction_complete_path_rate": float(min(
            row["complete_path_rate"] for row in per_direction
        )),
        "per_direction": per_direction,
    }


def _metrics(
    score_rows: np.ndarray,
    translation_rows: np.ndarray,
    rotation_rows: np.ndarray,
    valid_rows: np.ndarray,
    image_ids: np.ndarray,
    *,
    stage: str,
    candidate_semantics: str,
    radial_paths: np.ndarray | None = None,
    candidate_direction_ids: np.ndarray | None = None,
    candidate_signs: np.ndarray | None = None,
    direction_axis_pairs: np.ndarray | None = None,
    twist_order: tuple[str, ...] = ("r_x", "r_y", "r_z", "t_x", "t_y", "t_z"),
) -> dict[str, object]:
    if str(stage) not in _STAGE_RADII:
        raise ValueError("metric stage must be coarse, medium, or fine")
    translation_radius, rotation_radius = _STAGE_RADII[str(stage)]
    rows = []
    for image_id, score, translation, rotation, valid in zip(
        image_ids.tolist(), score_rows, translation_rows, rotation_rows, valid_rows
    ):
        indices = np.flatnonzero(valid)
        nonanchor = indices[indices != 0]
        joint = np.maximum(
            translation / float(translation_radius),
            rotation / float(rotation_radius),
        )
        selected = int(nonanchor[np.argmax(score[nonanchor])])
        oracle = int(nonanchor[np.argmin(joint[nonanchor])])
        proposal = int(nonanchor[0])
        ordered_pairs = 0
        correct_pairs = 0
        for a in indices.tolist():
            for b in indices.tolist():
                if joint[a] + 1.0e-8 < joint[b]:
                    ordered_pairs += 1
                    correct_pairs += int(score[a] > score[b])
        row_paths = _controlled_paths_for_row(
            valid, translation, rotation,
            candidate_semantics=str(candidate_semantics),
            radial_paths=radial_paths,
        )
        radial_pair_total = 0
        radial_pair_correct = 0
        radial_complete = 0
        for path in row_paths:
            comparisons = [
                bool(score[first] > score[second])
                for first, second in zip(path[:-1], path[1:])
            ]
            radial_pair_total += len(comparisons)
            radial_pair_correct += sum(comparisons)
            radial_complete += int(all(comparisons))
        rows.append({
            "image_id": str(image_id),
            "gt_anchor_rank": int(1 + np.sum(score[indices] > score[0])),
            "gt_anchor_margin_over_best_nonanchor": float(
                score[0] - np.max(score[nonanchor])
            ),
            "score_error_spearman": _spearman(score[indices], -joint[indices]),
            "pairwise_correct": correct_pairs,
            "pairwise_total": ordered_pairs,
            "controlled_radial_pair_correct": radial_pair_correct,
            "controlled_radial_pair_total": radial_pair_total,
            "controlled_radial_complete_paths": radial_complete,
            "controlled_radial_path_total": len(row_paths),
            "selected_index_excluding_gt_anchor": selected,
            "proposal_index": proposal,
            "oracle_index_excluding_gt_anchor": oracle,
            "selected_translation_m": float(translation[selected]),
            "selected_rotation_deg": float(rotation[selected]),
            "proposal_translation_m": float(translation[proposal]),
            "proposal_rotation_deg": float(rotation[proposal]),
            "oracle_translation_m": float(translation[oracle]),
            "oracle_rotation_deg": float(rotation[oracle]),
            "selected_strict_0_5m_5deg": bool(translation[selected] <= 0.5 and rotation[selected] <= 5.0),
            "proposal_strict_0_5m_5deg": bool(translation[proposal] <= 0.5 and rotation[proposal] <= 5.0),
            "selected_loose_1m_10deg": bool(translation[selected] <= 1.0 and rotation[selected] <= 10.0),
            "proposal_loose_1m_10deg": bool(translation[proposal] <= 1.0 and rotation[proposal] <= 10.0),
        })
    def mean(key):
        return float(np.mean([float(row[key]) for row in rows]))
    radial_pair_count = int(sum(row["controlled_radial_pair_total"] for row in rows))
    radial_pair_correct = int(sum(
        row["controlled_radial_pair_correct"] for row in rows
    ))
    radial_path_count = int(sum(row["controlled_radial_path_total"] for row in rows))
    radial_complete = int(sum(
        row["controlled_radial_complete_paths"] for row in rows
    ))
    full_6dof_metrics = None
    supplied_direction_arrays = (
        candidate_direction_ids is not None,
        candidate_signs is not None,
        direction_axis_pairs is not None,
    )
    if any(supplied_direction_arrays):
        if not all(supplied_direction_arrays) or radial_paths is None:
            raise ValueError("full-6dof metrics require all direction arrays and radial paths")
        full_6dof_metrics = _full_6dof_direction_metrics(
            score_rows, valid_rows, radial_paths,
            candidate_direction_ids, candidate_signs, direction_axis_pairs,
            twist_order=twist_order,
        )
    return {
        "stage": str(stage),
        "joint_error_translation_radius_m": float(translation_radius),
        "joint_error_rotation_radius_deg": float(rotation_radius),
        "query_count": len(rows),
        "gt_anchor_top1_rate": float(np.mean([row["gt_anchor_rank"] == 1 for row in rows])),
        "mean_gt_anchor_margin_over_best_nonanchor": mean(
            "gt_anchor_margin_over_best_nonanchor"
        ),
        "mean_score_error_spearman": mean("score_error_spearman"),
        "pairwise_order_accuracy": float(sum(row["pairwise_correct"] for row in rows) / max(sum(row["pairwise_total"] for row in rows), 1)),
        "controlled_radial_pair_accuracy": (
            None if radial_pair_count == 0
            else float(radial_pair_correct / radial_pair_count)
        ),
        "controlled_radial_complete_path_rate": (
            None if radial_path_count == 0
            else float(radial_complete / radial_path_count)
        ),
        "directional_outward_drift_violation_rate": (
            None if radial_pair_count == 0
            else float(1.0 - radial_pair_correct / radial_pair_count)
        ),
        "controlled_radial_pair_count": radial_pair_count,
        "controlled_radial_path_count": radial_path_count,
        "full_6dof_directional_capture": full_6dof_metrics,
        "selected_strict_0_5m_5deg": mean("selected_strict_0_5m_5deg"),
        "proposal_strict_0_5m_5deg": mean("proposal_strict_0_5m_5deg"),
        "selected_loose_1m_10deg": mean("selected_loose_1m_10deg"),
        "proposal_loose_1m_10deg": mean("proposal_loose_1m_10deg"),
        "selected_median_translation_m": float(np.median([row["selected_translation_m"] for row in rows])),
        "selected_median_rotation_deg": float(np.median([row["selected_rotation_deg"] for row in rows])),
        "proposal_median_translation_m": float(np.median([row["proposal_translation_m"] for row in rows])),
        "proposal_median_rotation_deg": float(np.median([row["proposal_rotation_deg"] for row in rows])),
        "oracle_median_translation_m": float(np.median([row["oracle_translation_m"] for row in rows])),
        "oracle_median_rotation_deg": float(np.median([row["oracle_rotation_deg"] for row in rows])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--train_queries", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surface_mapper", default="")
    parser.add_argument("--field_feature_contract", default="")
    parser.add_argument(
        "--allow_query_route_pretrained_mapper_control", action="store_true",
        help=(
            "explicitly allow a diagnostic whose canonical map is route-disjoint "
            "but whose frozen surface mapper saw the query route; the report is "
            "then marked ineligible for a strict route-disjoint representation claim"
        ),
    )
    parser.add_argument(
        "--view_conditioned_field", default="",
        help=(
            "required live replay artifact when the rendered dataset declares "
            "candidate-pose-conditioned map codes"
        ),
    )
    parser.add_argument(
        "--transport_semantics",
        choices=(
            FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
            TORCH_TRANSPORT_SEMANTICS,
        ),
        default=FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
        help=(
            "capacity-bounded fixed-kernel transport is the main v2 path; "
            "the source-softmax implementation is retained only as a negative control"
        ),
    )
    parser.add_argument(
        "--readout_training_semantics",
        choices=(
            IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
            FULL_MINIMAL_READOUT_CONTROL_SEMANTICS,
        ),
        default=IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
        help=(
            "the primary small-data path preserves the frozen shared 128-D "
            "RADIO space, fixes absent geometry weights and a hierarchy gauge, "
            "and trains only feature/layout relative evidence weights"
        ),
    )
    parser.add_argument(
        "--fixed_identity_statistics_backend",
        choices=(
            DENSE_FIXED_IDENTITY_STATISTICS_BACKEND,
            SPARSE_FIXED_IDENTITY_STATISTICS_BACKEND,
        ),
        default=DENSE_FIXED_IDENTITY_STATISTICS_BACKEND,
        help=(
            "exact batched dense reducer for the fixed identity path, or the "
            "much slower explicit sparse authority used for equivalence audits"
        ),
    )
    parser.add_argument("--dense_candidate_batch_size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_path, report_path = Path(args.output_model), Path(args.output_report)
    if (model_path.exists() or report_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite pose transport experiment")
    if int(args.epochs) <= 0 or float(args.learning_rate) <= 0.0:
        raise ValueError("training schedule must be positive")
    if int(args.dense_candidate_batch_size) <= 0:
        raise ValueError("dense_candidate_batch_size must be positive")
    arrays, dataset_metadata = _load_dataset(Path(args.dataset))
    scientific_dataset_audit = _validate_scientific_dataset_contract(
        arrays, dataset_metadata
    )
    query_count = int(arrays["image_ids"].size)
    train_count = int(args.train_queries)
    if not 1 <= train_count < query_count:
        raise ValueError("train_queries must leave at least one dev query")
    train_image_ids = [str(value) for value in arrays["image_ids"][:train_count].tolist()]
    dev_image_ids = [str(value) for value in arrays["image_ids"][train_count:].tolist()]
    if set(train_image_ids) & set(dev_image_ids):
        raise ValueError("transport train/dev query identities overlap")
    query_route = str(dataset_metadata["query_route"])
    split_audit = {
        "semantics": f"single_{query_route}_sorted_prefix_train_dev_no_fivefold_v2",
        "train_query_count": len(train_image_ids),
        "dev_query_count": len(dev_image_ids),
        "train_image_ids": train_image_ids,
        "dev_image_ids": dev_image_ids,
        "query_identity_overlap": [],
        "train_and_dev_share_query_route": True,
        "dev_is_independent_route_holdout": False,
        "all_query_rows_are_disjoint_from_map_training_routes": True,
        "dev_labels_used_for_gradient": False,
        "dev_labels_used_for_model_selection": False,
        "fixed_epoch_count_declared_before_dev_scoring": True,
    }
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    if (
        physical.content_sha256 != dataset_metadata.get("physical_map_sha256")
        or file_sha256(Path(args.physical_map))
        != dataset_metadata.get("physical_map_file_sha256")
    ):
        raise ValueError("dataset and physical map differ")
    map_field_semantics = str(dataset_metadata["map_pose_field_semantics"])
    if map_field_semantics == VIEW_CONDITIONED_MAP_FIELD_SEMANTICS:
        if not str(args.view_conditioned_field):
            raise ValueError("view-conditioned dataset requires its live field artifact")
        live_view_path = Path(args.view_conditioned_field)
        live_view = ViewConditionedPrimitiveField.load_npz(live_view_path)
        if (
            live_view.content_sha256
            != dataset_metadata.get("view_conditioned_field_sha256")
            or file_sha256(live_view_path)
            != dataset_metadata.get("view_conditioned_field_file_sha256")
            or live_view.physical_map_sha256 != physical.content_sha256
            or live_view.canonical_field_sha256
            != dataset_metadata.get("canonical_field_sha256")
            or live_view.metadata.get("coordinate_correct") is not True
        ):
            raise ValueError("live view-conditioned field and rendered dataset differ")
    elif str(args.view_conditioned_field):
        raise ValueError("canonical-field control cannot consume a view-conditioned field")
    live_hierarchy = build_pose_transport_hierarchy(physical)
    hierarchy = PoseTransportHierarchy(
        child_parent_ids=np.asarray(arrays["hierarchy_child_parent_ids"], dtype=np.int64),
        child_support_ids=np.asarray(arrays["hierarchy_child_support_ids"], dtype=np.int64),
        adjacency_offsets=np.asarray(arrays["hierarchy_adjacency_offsets"], dtype=np.int64),
        adjacency_child_rows=np.asarray(
            arrays["hierarchy_adjacency_child_rows"], dtype=np.int64
        ),
        content_sha256=str(dataset_metadata["hierarchy_content_sha256"]),
    )
    for name in (
        "child_parent_ids", "child_support_ids", "adjacency_offsets",
        "adjacency_child_rows",
    ):
        if not np.array_equal(getattr(hierarchy, name), getattr(live_hierarchy, name)):
            raise ValueError("embedded and live transport hierarchy arrays differ")
    shared_surface_space = bool(str(args.surface_mapper))
    mapper_sha = None
    mapper_supervision_audit = None
    if shared_surface_space:
        if not str(args.field_feature_contract):
            raise ValueError("surface_mapper requires field_feature_contract")
        contract_path = Path(args.field_feature_contract)
        contract = json.loads(contract_path.read_text())
        mapper_path = Path(args.surface_mapper)
        mapper_sha = file_sha256(mapper_path)
        if (
            contract.get("artifact_type") != "goal_maplet_field_feature_contract_v1"
            or contract.get("canonical_field_sha256") != dataset_metadata.get("canonical_field_sha256")
            or contract.get("query_readout_type") != "surface_maplet_mapper"
            or contract.get("query_readout_sha256") != mapper_sha
        ):
            raise ValueError("surface mapper/canonical field contract differs from dataset")
        mapper, mapper_metadata = load_surface_maplet_mapper(mapper_path, device=str(device))
        mapper_supervision_audit = _mapper_supervision_audit(
            mapper_metadata,
            query_image_ids=train_image_ids + dev_image_ids,
            query_route=query_route,
        )
        if (
            mapper_supervision_audit["query_route_in_mapper_supervision"]
            and not bool(args.allow_query_route_pretrained_mapper_control)
        ):
            raise ValueError(
                "surface mapper saw the query route; pass the explicit diagnostic "
                "control flag or provide a route-disjoint mapper/field contract"
            )
        mapped_rows = []
        mapper.model.to(device).eval()
        with torch.no_grad():
            for query_index in range(query_count):
                raw = torch.as_tensor(
                    arrays["radio_final"][query_index], device=device, dtype=torch.float32
                )[None]
                mapped_rows.append(
                    mapper.model(raw)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                )
        arrays["pose_query_features"] = np.stack(mapped_rows, axis=0)
    identity_feature_only = (
        str(args.readout_training_semantics) == IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS
    )
    if identity_feature_only and not shared_surface_space:
        raise ValueError("identity feature-only readout requires the shared surface mapper")
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=128 if shared_surface_space else 1280,
        pose_code_dim=128 if identity_feature_only else 32,
        shared_query_map_projection=shared_surface_space,
    )).to(device)
    if identity_feature_only:
        with torch.no_grad():
            model.map_projection.weight.copy_(
                torch.eye(128, dtype=model.map_projection.weight.dtype, device=device)
            )
            model.query_pose_residual_scale.zero_()
            model.edge_weight_unconstrained[list(IDENTITY_DISABLED_EDGE_COMPONENTS)] = -30.0
            model.edge_weight_unconstrained[IDENTITY_REFERENCE_EDGE_COMPONENT] = 0.0
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.edge_weight_unconstrained.requires_grad_(True)
        identity_gradient_mask = _identity_edge_gradient_mask(
            device=device, dtype=model.edge_weight_unconstrained.dtype,
        )
        model.edge_weight_unconstrained.register_hook(
            lambda gradient: gradient * identity_gradient_mask
        )
        identity_frozen_edge_values = model.edge_weight_unconstrained.detach().clone()
    elif shared_surface_space:
        # P1 deliberately trains only the already aligned feature space plus
        # fixed hierarchy/layout evidence. Random, unsupervised query normal,
        # depth and boundary heads must not inject noise into this first gate.
        with torch.no_grad():
            model.edge_weight_unconstrained[1:4].fill_(-8.0)

    use_fixed_identity_statistics = bool(
        identity_feature_only
        and str(args.transport_semantics) == FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_edge_weights = model.edge_weights().detach().cpu().numpy().astype(np.float64)
    initial_reference_edge_weight = (
        float(initial_edge_weights[IDENTITY_REFERENCE_EDGE_COMPONENT])
        if identity_feature_only else None
    )
    initial_edge_weight_ratios = (
        (initial_edge_weights / initial_reference_edge_weight).tolist()
        if identity_feature_only else None
    )
    candidate_semantics = str(dataset_metadata.get("candidate_semantics", ""))
    dataset_radial_paths = (
        np.asarray(arrays["controlled_radial_paths"], dtype=np.int64)
        if "controlled_radial_paths" in arrays else None
    )
    direction_ids = (
        np.asarray(arrays["controlled_candidate_direction_ids"], dtype=np.int64)
        if "controlled_candidate_direction_ids" in arrays else None
    )
    direction_signs = (
        np.asarray(arrays["controlled_candidate_signs"], dtype=np.int64)
        if "controlled_candidate_signs" in arrays else None
    )
    direction_axis_pairs = (
        np.asarray(arrays["controlled_direction_axis_pairs"], dtype=np.int64)
        if "controlled_direction_axis_pairs" in arrays else None
    )
    twist_order = tuple(
        str(value) for value in (
            dataset_metadata.get("controlled_pose_stencil_audit", {}) or {}
        ).get("twist_order", ("r_x", "r_y", "r_z", "t_x", "t_y", "t_z"))
    )
    monotonic_pair_semantics = (
        "embedded_dataset_controlled_radial_paths_v1"
        if dataset_radial_paths is not None
        else "six_independent_gt_outward_controlled_stencil_paths_v1"
    )
    monotonic_pair_rows = [
        _controlled_monotonic_pair_rows(
            arrays["candidate_valid"][query_index],
            arrays["translation_m"][query_index],
            arrays["rotation_deg"][query_index],
            candidate_semantics=candidate_semantics,
            radial_paths=dataset_radial_paths,
        )
        for query_index in range(query_count)
    ]
    train_monotonic_pair_count = int(sum(
        value.shape[0] for value in monotonic_pair_rows[:train_count]
    ))

    # The 85-candidate stencil contains millions of sparse edges per query.
    # Fixed/identity scoring is exactly linear in six evidence components, so
    # retain six sufficient statistics per candidate and release each graph.
    # Other semantics fail closed on large pools instead of risking an OOM.
    medium_edges: list[list[FrozenSparseTransportEdges]] | None = None
    component_statistics_by_stage: dict[str, list[np.ndarray | None]] = {}
    streaming_resource_audit: dict[str, list[dict[str, object]]] = {}
    statistics_backend = (
        str(args.fixed_identity_statistics_backend)
        if use_fixed_identity_statistics else None
    )
    dense_hierarchy = (
        DensePoseTransportHierarchyGPU(hierarchy, device=device)
        if statistics_backend == DENSE_FIXED_IDENTITY_STATISTICS_BACKEND else None
    )
    statistics_semantics = (
        DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS
        if statistics_backend == DENSE_FIXED_IDENTITY_STATISTICS_BACKEND
        else FIXED_IDENTITY_SUFFICIENT_STATISTICS_SEMANTICS
    )

    def build_component_statistics(
        query_index: int, *, stage: str,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if statistics_backend == DENSE_FIXED_IDENTITY_STATISTICS_BACKEND:
            if dense_hierarchy is None:
                raise AssertionError("dense hierarchy was not initialized")
            return _dense_fixed_identity_query_component_statistics(
                model, arrays, query_index, dense_hierarchy,
                stage=str(stage), device=device,
                candidate_batch_size=int(args.dense_candidate_batch_size),
            )
        if statistics_backend == SPARSE_FIXED_IDENTITY_STATISTICS_BACKEND:
            return _fixed_identity_query_component_statistics(
                model, arrays, query_index, hierarchy,
                stage=str(stage), device=device,
            )
        raise ValueError("fixed identity statistics backend differs")

    print(json.dumps({
        "hierarchy_sha256": hierarchy.content_sha256,
        "fixed_identity_sufficient_statistics": use_fixed_identity_statistics,
        "fixed_identity_statistics_backend": statistics_backend,
        "fixed_identity_statistics_semantics": statistics_semantics,
        "candidate_count": int(arrays["candidate_valid"].shape[1]),
    }), flush=True)
    if use_fixed_identity_statistics:
        component_statistics_by_stage["medium"] = [None] * query_count
        streaming_resource_audit["medium"] = []
        for query in range(query_count):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            statistics, audit = build_component_statistics(query, stage="medium")
            _release_streaming_cuda_cache(device, audit)
            component_statistics_by_stage["medium"][query] = statistics
            streaming_resource_audit["medium"].append(audit)
            print(json.dumps({
                "query": str(arrays["image_ids"][query]),
                "fixed_identity_statistics": audit,
            }), flush=True)
    else:
        if int(arrays["candidate_valid"].shape[1]) > 32:
            raise ValueError(
                "large candidate pools require fixed-kernel identity sufficient statistics"
            )
        medium_edges = []
        for query in range(query_count):
            medium_edges.append(_edges_for_query(
                arrays, query, hierarchy, stage="medium"
            ))
            print(json.dumps({
                "query": str(arrays["image_ids"][query]),
                "medium_edge_count": int(sum(
                    value.source_index.size for value in medium_edges[-1]
                )),
            }), flush=True)

    def scores_for_query(
        query_index: int, *, stage: str, gradient: bool,
        edge_rows: list[FrozenSparseTransportEdges] | None = None,
    ) -> torch.Tensor:
        if use_fixed_identity_statistics:
            statistics = component_statistics_by_stage[str(stage)][query_index]
            if statistics is None:
                raise ValueError("requested fixed identity statistics were not streamed")
            context = torch.enable_grad() if gradient else torch.no_grad()
            with context:
                return _fixed_identity_scores_from_component_statistics(
                    model,
                    torch.as_tensor(statistics, device=device, dtype=torch.float32),
                )
        if edge_rows is None:
            raise ValueError("explicit sparse transport requires frozen edges")
        return _query_scores(
            model, arrays, query_index, edge_rows, device=device,
            gradient=gradient, transport_semantics=str(args.transport_semantics),
            readout_training_semantics=str(args.readout_training_semantics),
        )

    def evaluate_stage(indices: range, *, stage: str, edges_by_query=None) -> np.ndarray:
        return np.stack([
            scores_for_query(
                q, stage=str(stage), gradient=False,
                edge_rows=(None if edges_by_query is None else edges_by_query[q]),
            ).detach().cpu().numpy()
            for q in indices
        ])

    train_indices = range(0, train_count)
    dev_indices = range(train_count, query_count)
    # True frozen equal-weight baseline, before the first optimizer step.
    # Dev is diagnostic only and never participates in a choice or update.
    initial_train_medium_score = evaluate_stage(
        train_indices, stage="medium", edges_by_query=medium_edges,
    )
    initial_dev_medium_score = evaluate_stage(
        dev_indices, stage="medium", edges_by_query=medium_edges,
    )
    trainable_parameters = [value for value in model.parameters() if value.requires_grad]
    effective_trainable_scalar_count = (
        len(IDENTITY_TRAINED_EDGE_COMPONENTS)
        if identity_feature_only
        else int(sum(value.numel() for value in trainable_parameters))
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=float(args.learning_rate),
        # Decoupled decay would move masked entries even with zero gradients.
        weight_decay=0.0 if identity_feature_only else 1.0e-5,
    )
    loss_config = PoseTransportTrainingConfig(stage="medium", attribution_weight=0.0)
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        epoch_loss = []
        epoch_objective: dict[str, list[float]] = {}
        order = np.roll(np.arange(train_count), epoch % train_count)
        for query_index in order.tolist():
            optimizer.zero_grad(set_to_none=True)
            scores = scores_for_query(
                query_index, stage="medium", gradient=True,
                edge_rows=(
                    None if medium_edges is None else medium_edges[query_index]
                ),
            )[None]
            valid = torch.as_tensor(arrays["candidate_valid"][query_index], device=device)[None]
            translation = torch.as_tensor(arrays["translation_m"][query_index], device=device)[None]
            rotation = torch.as_tensor(arrays["rotation_deg"][query_index], device=device)[None]
            monotonic = torch.as_tensor(
                monotonic_pair_rows[query_index],
                device=device, dtype=torch.long,
            )
            loss, stats = pose_transport_energy_landscape_loss(
                scores, translation, rotation, valid,
                config=loss_config, monotonic_pairs=monotonic,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite pose-transport loss at epoch={epoch + 1}, query={query_index}"
                )
            loss.backward()
            nonfinite_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite_gradients:
                raise FloatingPointError(
                    "nonfinite pose-transport gradients: " + ",".join(nonfinite_gradients)
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("nonfinite pose-transport gradient norm")
            optimizer.step()
            if identity_feature_only:
                frozen_indices = (
                    *IDENTITY_DISABLED_EDGE_COMPONENTS,
                    IDENTITY_REFERENCE_EDGE_COMPONENT,
                )
                if not torch.equal(
                    model.edge_weight_unconstrained.detach()[list(frozen_indices)],
                    identity_frozen_edge_values[list(frozen_indices)],
                ):
                    raise AssertionError("identity readout moved a fixed edge-weight gauge")
            nonfinite_parameters = [
                name for name, parameter in model.named_parameters()
                if not torch.isfinite(parameter).all()
            ]
            if nonfinite_parameters:
                raise FloatingPointError(
                    "nonfinite pose-transport parameters: " + ",".join(nonfinite_parameters)
                )
            epoch_loss.append(float(loss.detach().cpu()))
            for name, value in stats.items():
                if isinstance(value, (float, int)):
                    epoch_objective.setdefault(name, []).append(float(value))
        if epoch in {0, 1, 4, 9, 19, 39, int(args.epochs) - 1}:
            row = {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(epoch_loss)),
                "mean_objective_terms": {
                    name: float(np.mean(values))
                    for name, values in sorted(epoch_objective.items())
                },
            }
            history.append(row); print(json.dumps(row), flush=True)

    # Freeze the fixed final epoch before any dev scoring.  Dev errors were
    # already opened for fail-closed artifact/anchor validation, but never
    # entered a gradient, hyperparameter decision, or model-selection branch.
    model.eval()
    learned_edge_weights = model.edge_weights().detach().cpu().numpy().astype(np.float64)
    reference_edge_weight = float(
        learned_edge_weights[IDENTITY_REFERENCE_EDGE_COMPONENT]
    ) if identity_feature_only else None
    learned_edge_weight_ratios = (
        (learned_edge_weights / reference_edge_weight).tolist()
        if identity_feature_only else None
    )
    model_sha = pose_transport_model_content_sha256(model)
    save_minimal_pose_transport_readout(model, model_path, metadata={
        "dataset_content_sha256": dataset_metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "surface_mapper_file_sha256": mapper_sha,
        "mapper_supervision_audit": mapper_supervision_audit,
        "strict_query_representation_route_disjoint": bool(
            mapper_supervision_audit is None
            or mapper_supervision_audit["strict_query_representation_route_disjoint"]
        ),
        "map_pose_field_semantics": map_field_semantics,
        "view_conditioned_field_sha256": dataset_metadata.get(
            "view_conditioned_field_sha256"
        ),
        "shared_query_map_projection": shared_surface_space,
        "transport_semantics": str(args.transport_semantics),
        "readout_training_semantics": str(args.readout_training_semantics),
        "optimizer_parameter_storage_count": int(sum(
            value.numel() for value in trainable_parameters
        )),
        "effective_trainable_scalar_count": effective_trainable_scalar_count,
        "trainable_parameter_count": effective_trainable_scalar_count,
        "identity_trained_edge_components": (
            list(IDENTITY_TRAINED_EDGE_COMPONENTS) if identity_feature_only else None
        ),
        "identity_reference_edge_component": (
            IDENTITY_REFERENCE_EDGE_COMPONENT if identity_feature_only else None
        ),
        "identity_disabled_edge_components": (
            list(IDENTITY_DISABLED_EDGE_COMPONENTS) if identity_feature_only else None
        ),
        "learned_edge_weights": learned_edge_weights.tolist(),
        "learned_edge_weight_ratios_to_hierarchy_reference": learned_edge_weight_ratios,
        "initial_edge_weights": initial_edge_weights.tolist(),
        "initial_edge_weight_ratios_to_hierarchy_reference": initial_edge_weight_ratios,
        "fixed_identity_sufficient_statistics": use_fixed_identity_statistics,
        "fixed_identity_sufficient_statistics_semantics": (
            statistics_semantics
            if use_fixed_identity_statistics else None
        ),
        "fixed_identity_statistics_backend": statistics_backend,
        "dense_candidate_batch_size": (
            int(args.dense_candidate_batch_size)
            if statistics_backend == DENSE_FIXED_IDENTITY_STATISTICS_BACKEND else None
        ),
        "p1_random_query_geometry_modalities_disabled_at_initialization": shared_surface_space,
        "train_image_ids": train_image_ids,
        "dev_image_ids": dev_image_ids,
        "fixed_final_epoch": int(args.epochs),
        "dev_labels_used_for_gradient": False,
        "dev_labels_used_for_model_selection": False,
        "dev_contract_labels_validated_before_model_freeze": True,
        "training_monotonic_pair_semantics": monotonic_pair_semantics,
        "training_monotonic_pair_count": train_monotonic_pair_count,
        "optimizer_trajectory_drift_negative_pair_count": 0,
    })

    train_score = evaluate_stage(
        train_indices, stage="medium", edges_by_query=medium_edges,
    )
    dev_medium_score = evaluate_stage(
        dev_indices, stage="medium", edges_by_query=medium_edges,
    )
    # Coarse and exact-child controls are constructed only after the trained
    # model is frozen; neither can change weights or hyperparameters.
    coarse_edges = None
    fine_edges = None
    dev_coarse_score = None
    dev_fine_score = None
    if identity_feature_only:
        if use_fixed_identity_statistics:
            for stage in ("coarse", "fine"):
                component_statistics_by_stage[stage] = [None] * query_count
                streaming_resource_audit[stage] = []
                for q in dev_indices:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    statistics, audit = build_component_statistics(q, stage=stage)
                    _release_streaming_cuda_cache(device, audit)
                    component_statistics_by_stage[stage][q] = statistics
                    streaming_resource_audit[stage].append(audit)
                    print(json.dumps({
                        "query": str(arrays["image_ids"][q]),
                        "fixed_identity_statistics": audit,
                    }), flush=True)
            dev_coarse_score = evaluate_stage(dev_indices, stage="coarse")
            dev_fine_score = evaluate_stage(dev_indices, stage="fine")
        else:
            coarse_edges = [None] * query_count
            fine_edges = [None] * query_count
            for q in dev_indices:
                coarse_edges[q] = _edges_for_query(
                    arrays, q, hierarchy, stage="coarse"
                )
                fine_edges[q] = _edges_for_query(
                    arrays, q, hierarchy, stage="fine"
                )
            dev_coarse_score = evaluate_stage(
                dev_indices, stage="coarse", edges_by_query=coarse_edges,
            )
            dev_fine_score = evaluate_stage(
                dev_indices, stage="fine", edges_by_query=fine_edges,
            )
    # Frozen post-hoc diagnosis only: the serialized model above is never
    # changed.  These fixed component masks identify whether held ranking is
    # carried by RADIO appearance, physical hierarchy, or layout.  They are
    # not searched hyperparameters and cannot authorize a promoted model.
    frozen_edge_weight = model.edge_weight_unconstrained.detach().clone()
    component_ablation_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    component_masks = {
        "feature_only": (0,),
        "hierarchy_only": (4,),
        "layout_only": (5,),
        "feature_hierarchy_layout": (0, 4, 5),
    }
    with torch.no_grad():
        for name, active_components in component_masks.items():
            model.edge_weight_unconstrained.fill_(-30.0)
            model.edge_weight_unconstrained[list(active_components)] = 0.0
            component_ablation_scores[name] = (
                evaluate_stage(
                    train_indices, stage="medium", edges_by_query=medium_edges,
                ),
                evaluate_stage(
                    dev_indices, stage="medium", edges_by_query=medium_edges,
                ),
            )
        model.edge_weight_unconstrained.copy_(frozen_edge_weight)
    if pose_transport_model_content_sha256(model) != model_sha:
        raise AssertionError("post-hoc component diagnosis changed the frozen model")
    def metrics_for(score: np.ndarray, begin: int, end: int, *, stage: str):
        return _metrics(
            score,
            arrays["translation_m"][begin:end],
            arrays["rotation_deg"][begin:end],
            arrays["candidate_valid"][begin:end],
            arrays["image_ids"][begin:end],
            stage=str(stage), candidate_semantics=candidate_semantics,
            radial_paths=dataset_radial_paths,
            candidate_direction_ids=direction_ids,
            candidate_signs=direction_signs,
            direction_axis_pairs=direction_axis_pairs,
            twist_order=twist_order,
        )

    initial_train_medium_metrics = metrics_for(
        initial_train_medium_score, 0, train_count, stage="medium",
    )
    initial_dev_medium_metrics = metrics_for(
        initial_dev_medium_score, train_count, query_count, stage="medium",
    )
    train_metrics = metrics_for(train_score, 0, train_count, stage="medium")
    dev_coarse_metrics = (
        metrics_for(dev_coarse_score, train_count, query_count, stage="coarse")
        if dev_coarse_score is not None else None
    )
    dev_medium_metrics = metrics_for(
        dev_medium_score, train_count, query_count, stage="medium",
    )
    dev_fine_metrics = (
        metrics_for(dev_fine_score, train_count, query_count, stage="fine")
        if dev_fine_score is not None else None
    )
    component_ablation_metrics = {
        name: {
            "active_component_indices": list(component_masks[name]),
            "train": metrics_for(score_pair[0], 0, train_count, stage="medium"),
            "dev": metrics_for(
                score_pair[1], train_count, query_count, stage="medium",
            ),
        }
        for name, score_pair in component_ablation_scores.items()
    }
    training_gain_keys = (
        "gt_anchor_top1_rate",
        "mean_gt_anchor_margin_over_best_nonanchor",
        "mean_score_error_spearman",
        "pairwise_order_accuracy",
        "controlled_radial_pair_accuracy",
        "controlled_radial_complete_path_rate",
        "directional_outward_drift_violation_rate",
    )

    def metric_delta(after: dict[str, object], before: dict[str, object]):
        return {
            key: (
                None if after.get(key) is None or before.get(key) is None
                else float(after[key]) - float(before[key])
            )
            for key in training_gain_keys
        }

    training_gain = {
        "semantics": "trained_minus_preoptimizer_equal_weight_initialization_v1",
        "train_medium": metric_delta(train_metrics, initial_train_medium_metrics),
        "dev_medium": metric_delta(dev_medium_metrics, initial_dev_medium_metrics),
    }
    final_resource_audit = {
        "semantics": (
            statistics_semantics
            if use_fixed_identity_statistics else "explicit_frozen_edge_cache_v1"
        ),
        "statistics_backend": statistics_backend,
        "per_query_stage_audits": streaming_resource_audit,
        "peak_process_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "peak_cuda_allocated_mib_over_streamed_queries": float(max(
            (
                float(audit.get("peak_cuda_allocated_mib", 0.0))
                for audits in streaming_resource_audit.values()
                for audit in audits
            ),
            default=0.0,
        )),
        "peak_cuda_reserved_mib_over_streamed_queries": float(max(
            (
                float(audit.get("peak_cuda_reserved_mib", 0.0))
                for audits in streaming_resource_audit.values()
                for audit in audits
            ),
            default=0.0,
        )),
        "query_boundary_empty_cache_enabled": bool(use_fixed_identity_statistics),
        "retained_component_statistic_bytes": int(sum(
            value.nbytes
            for rows in component_statistics_by_stage.values()
            for value in rows if value is not None
        )),
        "all_candidate_edge_graphs_retained": not use_fixed_identity_statistics,
    }
    report = {
        "artifact_type": REPORT_SCHEMA,
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "dataset_content_sha256": dataset_metadata["content_sha256"],
        "physical_map_sha256": physical.content_sha256,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "surface_mapper_file_sha256": mapper_sha,
        "mapper_supervision_audit": mapper_supervision_audit,
        "strict_query_representation_route_disjoint": bool(
            mapper_supervision_audit is None
            or mapper_supervision_audit["strict_query_representation_route_disjoint"]
        ),
        "map_pose_field_semantics": map_field_semantics,
        "view_conditioned_field_sha256": dataset_metadata.get(
            "view_conditioned_field_sha256"
        ),
        "shared_query_map_projection": shared_surface_space,
        "p1_random_query_geometry_modalities_disabled_at_initialization": shared_surface_space,
        "model_file_sha256": file_sha256(model_path),
        "model_content_sha256": model_sha,
        "seed": int(args.seed), "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "transport_semantics": str(args.transport_semantics),
        "readout_training_semantics": str(args.readout_training_semantics),
        "fixed_identity_sufficient_statistics": use_fixed_identity_statistics,
        "fixed_identity_sufficient_statistics_semantics": (
            statistics_semantics
            if use_fixed_identity_statistics else None
        ),
        "fixed_identity_statistics_backend": statistics_backend,
        "dense_candidate_batch_size": (
            int(args.dense_candidate_batch_size)
            if statistics_backend == DENSE_FIXED_IDENTITY_STATISTICS_BACKEND else None
        ),
        "streaming_resource_audit": final_resource_audit,
        "optimizer_parameter_storage_count": int(sum(
            value.numel() for value in trainable_parameters
        )),
        "effective_trainable_scalar_count": effective_trainable_scalar_count,
        "trainable_parameter_count": effective_trainable_scalar_count,
        "identity_weight_identifiability_semantics": (
            "hierarchy_softplus_zero_reference_feature_and_layout_two_relative_dof_v1"
            if identity_feature_only else None
        ),
        "identity_trained_edge_components": (
            list(IDENTITY_TRAINED_EDGE_COMPONENTS) if identity_feature_only else None
        ),
        "identity_reference_edge_component": (
            IDENTITY_REFERENCE_EDGE_COMPONENT if identity_feature_only else None
        ),
        "identity_disabled_edge_components": (
            list(IDENTITY_DISABLED_EDGE_COMPONENTS) if identity_feature_only else None
        ),
        "learned_edge_weights": learned_edge_weights.tolist(),
        "learned_edge_weight_ratios_to_hierarchy_reference": learned_edge_weight_ratios,
        "initial_edge_weights": initial_edge_weights.tolist(),
        "initial_edge_weight_ratios_to_hierarchy_reference": initial_edge_weight_ratios,
        "scientific_dataset_contract_audit": scientific_dataset_audit,
        "split_contract_audit": split_audit,
        "split_semantics": split_audit["semantics"],
        "train_image_ids": train_image_ids,
        "dev_image_ids": dev_image_ids,
        "dev_labels_opened_before_model_freeze": True,
        "dev_labels_opened_before_model_freeze_reason": (
            "loaded_and_validated_as_dataset_contract_but_never_used_for_gradient_or_selection"
        ),
        "dev_labels_used_for_gradient": False,
        "dev_labels_used_for_model_selection": False,
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "controlled_candidates_are_gt_relative_oracle_diagnostic": bool(
            dataset_metadata.get(
                "controlled_candidates_are_gt_relative_oracle_diagnostic", False
            )
        ),
        "end_to_end_metrics_exclude_gt_anchor": True,
        "selected_candidate_threshold_metrics_are_core_claim": False,
        "training_stage": "medium",
        "dev_transport_stage_control_observation_contract": (
            {
                "coarse": (
                    "RADIO_feature_only_with_coarse_edge_relations_and_radius;"
                    "normal_depth_boundary_query_modalities_inactive"
                ),
                "medium": (
                    "RADIO_feature_only_with_medium_edge_relations_and_radius;"
                    "normal_depth_boundary_query_modalities_inactive"
                ),
                "fine": (
                    "RADIO_feature_only_with_exact_child_edges_and_fine_radius;"
                    "normal_depth_boundary_query_modalities_inactive"
                ),
            }
            if identity_feature_only else {
                "medium": "full_typed_centered_log_depth_observation",
                "coarse": "unavailable_without_ordinal_depth_query_observation",
                "fine": "unavailable_without_metric_depth_uncertainty_query_observation",
            }
        ),
        "coarse_fine_typed_depth_observation_claim_supported": False,
        "training_monotonic_pair_semantics": monotonic_pair_semantics,
        "training_monotonic_pair_count": train_monotonic_pair_count,
        "optimizer_trajectory_drift_supervision_available": False,
        "optimizer_trajectory_drift_negative_pair_count": 0,
        "objective_drift_claim_supported": False,
        "objective_drift_limitation": (
            "controlled_radial_violations_are_measured_but_no_frozen_optimizer_trajectory_is_present"
        ),
        "experiment_role": (
            "primary_fixed_kernel_identity_readout"
            if str(args.transport_semantics) == FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
            else "explicit_legacy_source_softmax_negative_control"
        ),
        "preoptimizer_equal_weight_frozen_baseline": {
            "semantics": (
                "feature_hierarchy_layout_equal_softplus_weights_with_geometry_disabled_v1"
                if identity_feature_only else "model_seed_initialization_v1"
            ),
            "train_medium": initial_train_medium_metrics,
            "dev_medium": initial_dev_medium_metrics,
            "computed_before_any_optimizer_step": True,
            "dev_used_for_selection": False,
        },
        "training_gain_over_preoptimizer_baseline": training_gain,
        "train_medium_transport": train_metrics,
        "dev_coarse_feature_only_structural_transport": dev_coarse_metrics,
        "dev_coarse_sparse_transport": dev_coarse_metrics,
        "dev_medium_sparse_transport": dev_medium_metrics,
        "dev_medium_full_6dof_directional_capture": dev_medium_metrics.get(
            "full_6dof_directional_capture"
        ),
        "dev_fine_feature_only_exact_child_transport": dev_fine_metrics,
        "dev_fine_exact_child_edge_control": dev_fine_metrics,
        "dev_exact_child_edge_control": dev_fine_metrics,
        "posthoc_fixed_component_ablation_diagnostic": component_ablation_metrics,
        "history": history,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "strict_route_disjoint_backend_claim_supported": bool(
            mapper_supervision_audit is None
            or mapper_supervision_audit["strict_query_representation_route_disjoint"]
        ),
        "claim": (
            "canonical_map_disjoint_local_backend_diagnostic_with_mapper_pretraining_audit;"
            "not_end_to_end_localization"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
