"""Build a map-disjoint real RADIO sparse-pose-transport dataset.

This is a local-backend experiment, not an end-to-end localization metric.
It reuses a frozen pose-free proposal list only for candidate SE(3) values,
adds one diagnostic GT anchor, recomputes every pose error, and rerenders all
candidate evidence from the requested frozen physical/canonical map.  Query
RADIO and retrieval artifacts are never recomputed from pose labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import _pose_errors
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.controlled_pose_stencil import (
    CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS,
    build_medium_quadratic_complete_6dof_stencil,
    controlled_pose_stencil_audit,
    stencil_candidate_poses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    DOUBLE_SIDED,
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
    build_pose_transport_hierarchy,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import FrozenSoftSurfaceSceneGPU
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    query_only_pose_reliability_weights,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    ViewConditionedPrimitiveField,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp


SCHEMA = "goal_maplet_real_sparse_pose_transport_dataset_v2"
_CANDIDATE_OBSERVATION_ARRAYS = (
    "target_child_rows",
    "target_child_weights",
    "target_canonical_features",
    "target_normals_camera",
    "target_double_sided",
    "target_relative_depth",
    "target_boundary",
    "target_modality_valid",
    "target_modality_confidence",
)


def _dataset_budget_audit(
    arrays: dict[str, np.ndarray],
    *,
    query_count: int,
    candidate_count: int,
    render_batch_size: int,
    total_render_seconds: float,
) -> dict[str, object]:
    """Return exact storage/render counts without claiming future runtime."""

    queries = int(query_count)
    candidates = int(candidate_count)
    batch = int(render_batch_size)
    seconds = float(total_render_seconds)
    if queries <= 0 or candidates <= 0 or batch <= 0 or not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError("invalid sparse transport dataset budget")
    missing = sorted(set(_CANDIDATE_OBSERVATION_ARRAYS) - set(arrays))
    if missing:
        raise ValueError("candidate observation budget lacks arrays: " + ",".join(missing))
    candidate_bytes = 0
    for name in _CANDIDATE_OBSERVATION_ARRAYS:
        value = np.asarray(arrays[name])
        if value.shape[:2] != (queries, candidates):
            raise ValueError(f"candidate observation {name} has the wrong leading axes")
        candidate_bytes += int(value.nbytes)
    return {
        "rendered_pose_count": int(queries * candidates),
        "render_batch_count": int(queries * ((candidates + batch - 1) // batch)),
        "render_batch_size": batch,
        "actual_total_render_seconds": seconds,
        "actual_mean_render_seconds_per_pose": float(
            seconds / max(queries * candidates, 1)
        ),
        "uncompressed_candidate_observation_bytes": int(candidate_bytes),
        "uncompressed_candidate_observation_bytes_per_query": int(
            candidate_bytes // queries
        ),
        "uncompressed_all_array_bytes": int(
            sum(np.asarray(value).nbytes for value in arrays.values())
        ),
        "compressed_output_size_is_not_predicted": True,
    }


def _controlled_pose_candidates(target_pose: np.ndarray) -> list[np.ndarray]:
    """Return a frozen symmetric local SE(3) diagnostic stencil.

    Twists use the repository's left-multiplicative camera-frame convention
    ``Exp(delta) @ T_w2c`` with coordinate order ``[r_xyz,t_xyz]``.  This is
    an oracle landscape diagnostic only; it is never used as a deployable
    proposal source.
    """

    degree = np.pi / 180.0
    deltas = [np.zeros(6, dtype=np.float64)]
    deltas += [
        np.asarray([0, 0, 0, sign * distance, 0, 0], dtype=np.float64)
        for distance in (0.25, 0.50, 1.00) for sign in (-1.0, 1.0)
    ]
    deltas += [
        np.asarray([0, sign * angle * degree, 0, 0, 0, 0], dtype=np.float64)
        for angle in (5.0, 10.0, 15.0) for sign in (-1.0, 1.0)
    ]
    deltas += [
        np.asarray([0, sign * angle * degree, 0, sign * distance, 0, 0], dtype=np.float64)
        for distance, angle in ((0.50, 5.0), (1.00, 10.0), (2.00, 20.0))
        for sign in (-1.0, 1.0)
    ]
    return [se3_exp(delta) @ np.asarray(target_pose, dtype=np.float64) for delta in deltas]


def _pose_key(value: np.ndarray) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=np.float64).round(10).reshape(-1).tolist())


def _load_token_inventory(paths: list[Path], *, artifact_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError("RADIO token manifest lacks records")
        for row in rows:
            image_id = str(row.get("image_id", ""))
            token_path = Path(str(row.get("token_path", "")))
            if not token_path.is_absolute():
                token_path = Path(artifact_root) / token_path
            if not image_id or image_id in result or not token_path.is_file():
                raise ValueError("RADIO token inventory is incomplete or duplicated")
            result[image_id] = token_path.resolve()
    return result


def _load_contributors(
    path: Path, *, required_image_ids: list[str] | None = None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if required_image_ids is None:
        sources = sorted(Path(path).glob("*.npz"))
    else:
        if len(set(required_image_ids)) != len(required_image_ids):
            raise ValueError("required contributor IDs are duplicated")
        sources = [
            Path(path) / (image_id.replace("/", "__") + ".npz")
            for image_id in required_image_ids
        ]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"missing contributor artifact: {source}")
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata.get("image_id", ""))
        if not image_id or image_id in result:
            raise ValueError("contributor inventory is incomplete or duplicated")
        result[image_id] = source.resolve()
    return result


def _candidate_map_geometry(
    physical: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    child_rows: np.ndarray,
    child_weights: np.ndarray,
    child_feature_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(child_rows, dtype=np.int64)
    weight = np.asarray(child_weights, dtype=np.float32)
    valid = (rows >= 0) & (weight > 1.0e-6)
    safe = np.maximum(rows, 0)
    rotation = np.asarray(pose_w2c, dtype=np.float64)[:3, :3]
    translation = np.asarray(pose_w2c, dtype=np.float64)[:3, 3]
    normal = np.einsum("ij,tsj->tsi", rotation, physical.child_normals[safe])
    normal_norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(normal_norm, 1.0e-8)
    center_camera = np.einsum("ij,tsj->tsi", rotation, physical.child_centers[safe]) + translation
    depth_valid = valid & (center_camera[..., 2] > 1.0e-6)
    log_depth = np.zeros_like(weight, dtype=np.float64)
    log_depth[depth_valid] = np.log(center_camera[..., 2][depth_valid])
    denominator = float(np.sum(weight[depth_valid], dtype=np.float64))
    mean_log_depth = 0.0 if denominator <= 1.0e-12 else float(
        np.sum(weight[depth_valid] * log_depth[depth_valid], dtype=np.float64) / denominator
    )
    relative_depth = (log_depth - mean_log_depth).astype(np.float32)
    boundary = np.clip(1.0 - weight, 0.0, 1.0).astype(np.float32)
    parent = physical.child_parent_rows[safe]
    double_sided = physical.maplet_sidedness[parent] == DOUBLE_SIDED
    modality_valid = np.stack([
        valid & np.asarray(child_feature_valid, dtype=bool),
        valid & (normal_norm[..., 0] >= 1.0e-8),
        depth_valid,
        valid,
    ], axis=2)
    confidence = np.repeat(weight[..., None], 4, axis=2).astype(np.float32)
    confidence[~modality_valid] = 0.0
    normal[~valid] = 0.0
    relative_depth[~depth_valid] = 0.0
    boundary[~valid] = 0.0
    return (
        normal.astype(np.float32), double_sided.astype(bool), relative_depth,
        boundary, modality_valid, confidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--token_manifest", action="append", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--canonical_field_audit", required=True)
    parser.add_argument("--view_conditioned_field", default="")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--artifact_root", default=".")
    parser.add_argument("--query_route", default="seq11")
    parser.add_argument(
        "--candidate_semantics",
        choices=(
            "proposal_pool_v1",
            "controlled_local_oracle_v1",
            CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS,
        ),
        default="proposal_pool_v1",
    )
    parser.add_argument("--maximum_candidates", type=int, default=8)
    parser.add_argument(
        "--maximum_queries", type=int, default=0,
        help=(
            "deterministic sorted query prefix for bounded diagnostic builds; "
            "zero consumes the complete requested route"
        ),
    )
    parser.add_argument("--source_slots", type=int, default=16)
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite sparse pose transport dataset")
    if (
        int(args.maximum_candidates) < 2 or int(args.source_slots) <= 0
        or int(args.maximum_queries) < 0
    ):
        raise ValueError("dataset requires at least two candidates and positive source slots")
    if int(args.render_batch_size) <= 0:
        raise ValueError("render_batch_size must be positive")
    controlled_6dof_stencil = (
        build_medium_quadratic_complete_6dof_stencil()
        if str(args.candidate_semantics)
        == CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS
        else None
    )
    if (
        controlled_6dof_stencil is not None
        and int(args.maximum_candidates) != controlled_6dof_stencil.candidate_count
    ):
        raise ValueError(
            "the complete medium 6DoF stencil requires exactly "
            f"{controlled_6dof_stencil.candidate_count} candidates"
        )

    physical_path = Path(args.physical_map)
    field_path = Path(args.canonical_field)
    field_audit_path = Path(args.canonical_field_audit)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    hierarchy = build_pose_transport_hierarchy(physical)
    field = CanonicalSurfaceField.load_npz(field_path)
    view_field_path = Path(args.view_conditioned_field) if args.view_conditioned_field else None
    view_field = (
        ViewConditionedPrimitiveField.load_npz(view_field_path)
        if view_field_path is not None else None
    )
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    if view_field is not None:
        view_field.validate_alignment(
            physical_map_sha256=physical.content_sha256,
            canonical_field_sha256=field.content_sha256,
            canonical_primitive_rows=field.primitive_rows,
            canonical_feature_dim=field.feature_dim,
        )
        if view_field.metadata.get("coordinate_correct") is not True:
            raise ValueError("view-conditioned field is not coordinate-correct")
    field_audit = json.loads(field_audit_path.read_text())
    routes = set(str(value) for value in field_audit.get("mapping_trajectory_ids", ()))
    if str(args.query_route) in routes:
        raise ValueError("canonical map contains the query route; self-copy is forbidden")
    if field_audit.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("canonical field audit and NPZ differ")
    if field_audit.get("storage_contract", {}).get("coordinate_correct") is not True:
        raise ValueError("canonical field is not coordinate-correct")

    pool_path = Path(args.candidate_pool)
    pool = json.loads(pool_path.read_text())
    pool_rows = {
        str(row["image_id"]): row for row in pool.get("rows", [])
        if str(row.get("image_id", "")).startswith(str(args.query_route) + "/")
    }
    if not pool_rows:
        raise ValueError("candidate pool has no requested query route")
    all_image_ids = sorted(pool_rows)
    full_query_count = len(all_image_ids)
    image_ids = all_image_ids
    if int(args.maximum_queries) > 0:
        image_ids = image_ids[:int(args.maximum_queries)]
    if len(image_ids) < 2:
        raise ValueError("bounded transport dataset requires at least two queries")
    contributors = _load_contributors(
        Path(args.contributors), required_image_ids=image_ids,
    )
    token_manifest_paths = [Path(value) for value in args.token_manifest]
    tokens = _load_token_inventory(
        token_manifest_paths, artifact_root=Path(args.artifact_root).resolve()
    )
    retrieval_dir = Path(args.retrieval_dir)
    if not set(image_ids).issubset(tokens) or set(contributors) != set(image_ids):
        raise ValueError("candidate queries lack contributor or RADIO evidence")

    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(args.device))
    radio_rows: list[np.ndarray] = []
    source_child_rows: list[np.ndarray] = []
    source_probability_rows: list[np.ndarray] = []
    reliability_rows: list[np.ndarray] = []
    token_xy_rows: list[np.ndarray] = []
    candidate_pose_rows: list[np.ndarray] = []
    translation_rows: list[np.ndarray] = []
    rotation_rows: list[np.ndarray] = []
    candidate_valid_rows: list[np.ndarray] = []
    child_rows_out: list[np.ndarray] = []
    child_weight_out: list[np.ndarray] = []
    feature_out: list[np.ndarray] = []
    normal_out: list[np.ndarray] = []
    double_out: list[np.ndarray] = []
    depth_out: list[np.ndarray] = []
    boundary_out: list[np.ndarray] = []
    modality_valid_out: list[np.ndarray] = []
    confidence_out: list[np.ndarray] = []
    retrieval_hashes: list[str] = []
    token_hashes: list[str] = []
    contributor_hashes: list[str] = []
    render_seconds: list[float] = []
    source_mass_fraction: list[float] = []

    maximum_candidates = int(args.maximum_candidates)
    source_slots = int(args.source_slots)
    for image_id in image_ids:
        query_render_start = len(render_seconds)
        contributor_path = contributors[image_id]
        with np.load(contributor_path, allow_pickle=False) as data:
            target_pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        if controlled_6dof_stencil is not None:
            candidate_values = list(
                stencil_candidate_poses(target_pose, controlled_6dof_stencil)
            )
        elif str(args.candidate_semantics) == "controlled_local_oracle_v1":
            candidate_values = _controlled_pose_candidates(target_pose)[:maximum_candidates]
        else:
            candidate_values = [target_pose]
            seen = {_pose_key(target_pose)}
            details = pool_rows[image_id].get("mode_details", {}).get("actual_parent_actual_child", [])
            for detail in details:
                pose = np.asarray(detail["pose_w2c"], dtype=np.float64)
                key = _pose_key(pose)
                if key in seen:
                    continue
                seen.add(key)
                candidate_values.append(pose)
                if len(candidate_values) >= maximum_candidates:
                    break
        if len(candidate_values) < 2:
            raise ValueError(f"query lacks multiple candidate poses: {image_id}")
        valid_count = len(candidate_values)
        while len(candidate_values) < maximum_candidates:
            candidate_values.append(candidate_values[-1].copy())
        candidate_pose = np.stack(candidate_values)
        translation, rotation = _pose_errors(candidate_pose, target_pose)
        candidate_valid = np.arange(maximum_candidates) < valid_count

        retrieval_path = retrieval_dir / (image_id.replace("/", "__") + ".npz")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.image_id != image_id or retrieval.physical_map_sha256 != physical.content_sha256:
            raise ValueError("retrieval query or physical lineage differs")
        if source_slots > retrieval.token_child_rows.shape[1]:
            raise ValueError("requested source slots exceed retrieval artifact")
        token_path = tokens[image_id]
        with np.load(token_path, allow_pickle=False) as data:
            if set(data.files) != {"radio_final"}:
                raise ValueError("RADIO token NPZ members differ")
            radio = np.asarray(data["radio_final"], dtype=np.float32)
        if radio.shape != (1280, 36, 64) or np.any(~np.isfinite(radio)):
            raise ValueError("RADIO query tensor differs")

        rendered_rows = []
        for start in range(0, maximum_candidates, int(args.render_batch_size)):
            stop = min(start + int(args.render_batch_size), maximum_candidates)
            batch = scene.render_exact_batch(
                candidate_pose[start:stop], _camera(contributor_path),
                width=64, height=36, top_l=4,
                view_conditioned_field=view_field,
            )
            rendered_rows.extend(batch.rendered)
            render_seconds.append(float(batch.audit.total_seconds))
        q_child, q_weight, q_feature = [], [], []
        q_normal, q_double, q_depth, q_boundary = [], [], [], []
        q_validity, q_confidence = [], []
        for pose, rendered in zip(candidate_pose, rendered_rows):
            rows = np.asarray(rendered.child_rows, dtype=np.int64).reshape(2304, 4)
            weight = np.asarray(rendered.child_weights, dtype=np.float32).reshape(2304, 4)
            feature = np.asarray(rendered.child_features, dtype=np.float32).reshape(2304, 4, 128)
            feature_valid = np.asarray(rendered.child_feature_valid, dtype=bool).reshape(2304, 4)
            normal, double, depth, boundary, validity, confidence = _candidate_map_geometry(
                physical, pose, rows, weight, feature_valid
            )
            q_child.append(rows.astype(np.int32)); q_weight.append(weight)
            q_feature.append(feature.astype(np.float16)); q_normal.append(normal.astype(np.float16))
            q_double.append(double); q_depth.append(depth.astype(np.float16))
            q_boundary.append(boundary.astype(np.float16)); q_validity.append(validity)
            q_confidence.append(confidence.astype(np.float16))

        full_mass = np.sum(retrieval.token_child_probabilities, dtype=np.float64)
        retained_mass = np.sum(retrieval.token_child_probabilities[:, :source_slots], dtype=np.float64)
        source_mass_fraction.append(float(retained_mass / max(full_mass, 1.0e-12)))
        radio_rows.append(radio.astype(np.float16))
        source_child_rows.append(retrieval.token_child_rows[:, :source_slots].astype(np.int32))
        source_probability_rows.append(retrieval.token_child_probabilities[:, :source_slots].astype(np.float32))
        reliability_rows.append(query_only_pose_reliability_weights(retrieval).astype(np.float32))
        token_xy_rows.append(retrieval.token_xy.astype(np.int16))
        candidate_pose_rows.append(candidate_pose)
        translation_rows.append(translation.astype(np.float32)); rotation_rows.append(rotation.astype(np.float32))
        candidate_valid_rows.append(candidate_valid)
        child_rows_out.append(np.stack(q_child)); child_weight_out.append(np.stack(q_weight))
        feature_out.append(np.stack(q_feature)); normal_out.append(np.stack(q_normal))
        double_out.append(np.stack(q_double)); depth_out.append(np.stack(q_depth))
        boundary_out.append(np.stack(q_boundary)); modality_valid_out.append(np.stack(q_validity))
        confidence_out.append(np.stack(q_confidence))
        retrieval_hashes.append(retrieval.content_sha256)
        token_hashes.append(file_sha256(token_path)); contributor_hashes.append(file_sha256(contributor_path))
        print(json.dumps({
            "image_id": image_id, "candidate_count": valid_count,
            "render_seconds": float(sum(render_seconds[query_render_start:])),
            "source_mass_fraction": source_mass_fraction[-1],
        }), flush=True)

    arrays = {
        "image_ids": np.asarray(image_ids),
        "radio_final": np.stack(radio_rows),
        "source_child_rows": np.stack(source_child_rows),
        "source_child_probabilities": np.stack(source_probability_rows),
        "query_reliability": np.stack(reliability_rows),
        "token_xy": np.stack(token_xy_rows),
        "candidate_poses_w2c": np.stack(candidate_pose_rows),
        "translation_m": np.stack(translation_rows),
        "rotation_deg": np.stack(rotation_rows),
        "candidate_valid": np.stack(candidate_valid_rows),
        "target_child_rows": np.stack(child_rows_out),
        "target_child_weights": np.stack(child_weight_out),
        "target_canonical_features": np.stack(feature_out),
        "target_normals_camera": np.stack(normal_out),
        "target_double_sided": np.stack(double_out),
        "target_relative_depth": np.stack(depth_out),
        "target_boundary": np.stack(boundary_out),
        "target_modality_valid": np.stack(modality_valid_out),
        "target_modality_confidence": np.stack(confidence_out),
        "retrieval_content_sha256": np.asarray(retrieval_hashes),
        "radio_file_sha256": np.asarray(token_hashes),
        "contributor_file_sha256": np.asarray(contributor_hashes),
        # The hierarchy participates directly in edge construction.  A file
        # hash in JSON is insufficient because the source physical map may be
        # moved or replaced after this expensive rendered dataset is built.
        # Store the exact immutable arrays needed to replay transport, while
        # still requiring any live physical map supplied by a trainer to
        # match the original file/content lineage.
        "hierarchy_child_parent_ids": np.asarray(
            hierarchy.child_parent_ids, dtype=np.int32
        ),
        "hierarchy_child_support_ids": np.asarray(
            hierarchy.child_support_ids, dtype=np.int32
        ),
        "hierarchy_adjacency_offsets": np.asarray(
            hierarchy.adjacency_offsets, dtype=np.int64
        ),
        "hierarchy_adjacency_child_rows": np.asarray(
            hierarchy.adjacency_child_rows, dtype=np.int32
        ),
    }
    if controlled_6dof_stencil is not None:
        arrays.update({
            "controlled_candidate_twists_left_camera": np.asarray(
                controlled_6dof_stencil.twists_left_camera, dtype=np.float64
            ),
            "controlled_candidate_direction_ids": np.asarray(
                controlled_6dof_stencil.candidate_direction_ids, dtype=np.int16
            ),
            "controlled_candidate_signs": np.asarray(
                controlled_6dof_stencil.candidate_signs, dtype=np.int8
            ),
            "controlled_candidate_radius_fractions": np.asarray(
                controlled_6dof_stencil.candidate_radius_fractions, dtype=np.float32
            ),
            "controlled_normalized_directions": np.asarray(
                controlled_6dof_stencil.normalized_directions, dtype=np.float64
            ),
            "controlled_direction_axis_pairs": np.asarray(
                controlled_6dof_stencil.direction_axis_pairs, dtype=np.int8
            ),
            "controlled_radial_paths": np.asarray(
                controlled_6dof_stencil.radial_paths, dtype=np.int16
            ),
        })
    stencil_audit = (
        controlled_pose_stencil_audit(controlled_6dof_stencil)
        if controlled_6dof_stencil is not None else None
    )
    dataset_budget = _dataset_budget_audit(
        arrays,
        query_count=len(image_ids),
        candidate_count=maximum_candidates,
        render_batch_size=int(args.render_batch_size),
        total_render_seconds=float(sum(render_seconds)),
    )
    metadata = {
        "artifact_type": SCHEMA,
        "content_sha256": arrays_sha256(arrays),
        "query_route": str(args.query_route),
        "query_count": len(image_ids),
        "full_route_query_count": int(full_query_count),
        "maximum_queries": int(args.maximum_queries),
        "bounded_sorted_query_prefix_diagnostic": bool(
            int(args.maximum_queries) > 0 and len(image_ids) < full_query_count
        ),
        "candidate_count": maximum_candidates,
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "candidate_pose_source_only": True,
        "candidate_semantics": str(args.candidate_semantics),
        "controlled_candidates_are_gt_relative_oracle_diagnostic": bool(
            str(args.candidate_semantics) in {
                "controlled_local_oracle_v1",
                CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS,
            }
        ),
        "controlled_pose_stencil_audit": stencil_audit,
        "full_6dof_observability_stencil": bool(
            controlled_6dof_stencil is not None
        ),
        "local_quadratic_6dof_identifiable": bool(
            stencil_audit is not None
            and stencil_audit["local_quadratic_6dof_identifiable"]
        ),
        "dataset_budget_audit": dataset_budget,
        "old_candidate_scores_consumed": False,
        "pose_errors_recomputed": True,
        "canonical_map_excludes_query_route": True,
        "map_training_routes": sorted(routes),
        "physical_map_sha256": physical.content_sha256,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "canonical_field_sha256": field.content_sha256,
        "physical_map_file_sha256": file_sha256(physical_path),
        "canonical_field_file_sha256": file_sha256(field_path),
        "view_conditioned_field_sha256": (
            view_field.content_sha256 if view_field is not None else None
        ),
        "view_conditioned_field_file_sha256": (
            file_sha256(view_field_path) if view_field_path is not None else None
        ),
        "map_pose_field_semantics": (
            "candidate_pose_evaluated_low_rank_view_conditioned_canonical_field_v1"
            if view_field is not None else "single_view_independent_canonical_field_control_v1"
        ),
        "canonical_field_audit_file_sha256": file_sha256(field_audit_path),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "retrieval_directory": str(retrieval_dir.resolve()),
        "token_manifest_file_sha256": (
            file_sha256(token_manifest_paths[0]) if len(token_manifest_paths) == 1 else None
        ),
        "token_manifest_files": [
            {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
            for path in token_manifest_paths
        ],
        "source_slots": source_slots,
        "minimum_retained_source_mass_fraction": float(min(source_mass_fraction)),
        "total_render_seconds": float(sum(render_seconds)),
        "normal_frame": "camera",
        "depth_semantics": "centered_log_depth_v1",
        "boundary_semantics": "one_minus_rendered_child_alpha_v1",
        "map_feature_semantics": "rendered_coordinate_correct_canonical_radio_128d_v1",
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "end_to_end_localization_claim_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(output)
    audit_path = output.with_suffix(".json")
    audit_path.write_text(json.dumps({**metadata, "output_npz": str(output.resolve())}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
