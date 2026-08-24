"""Build phase-separated RADIO source and GT-render footprint teacher artifacts.

The source artifact is deployable input only and deliberately stores no pose,
candidate, or teacher field.  Fit and held-route teacher labels are written to
different files so an evaluator can freeze all held predictions before opening
held labels.  GT camera poses are consumed only by this offline teacher build.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
    _load_token_inventory,
)
from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import (
    _camera,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
    build_pose_transport_hierarchy,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    query_only_pose_reliability_weights,
)


SOURCE_SCHEMA_V1 = "goal_maplet_query_footprint_predictability_source_v1"
SOURCE_SCHEMA = "goal_maplet_query_footprint_predictability_source_v2"
TEACHER_SCHEMA = "goal_maplet_query_footprint_GT_render_teacher_v1"


_RETRIEVAL_LINEAGE_KEYS = (
    "physical_map_file_sha256",
    "canonical_field_file_sha256",
    "canonical_field_coordinate_correct",
    "canonical_field_coordinate_contract",
    "surface_mapper_file_sha256",
    "field_feature_contract_file_sha256",
    "validity_calibration_file_sha256",
    "parent_score_semantics",
    "parent_scene_ranking_semantics",
    "parent_mode_temperature",
    "child_probability_semantics",
)


def _atomic_save(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream, **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parse_route_directories(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        route, separator, directory = str(value).partition("=")
        path = Path(directory)
        if not separator or not route or route in result or not path.is_dir():
            raise ValueError("retrieval_dir must be unique ROUTE=DIRECTORY entries")
        result[route] = path
    return result


def _metric_log_depth(
    child_rows: np.ndarray, pose_w2c: np.ndarray, child_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(child_rows, dtype=np.int64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    centers = np.asarray(child_centers, dtype=np.float64)
    if rows.ndim != 2 or pose.shape != (4, 4) or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("teacher metric-depth inputs differ")
    valid_row = rows >= 0
    if np.any(rows[valid_row] >= centers.shape[0]):
        raise ValueError("teacher child row is out of range")
    safe = np.maximum(rows, 0)
    camera = np.einsum("ij,tsj->tsi", pose[:3, :3], centers[safe]) + pose[:3, 3]
    valid = valid_row & np.isfinite(camera[..., 2]) & (camera[..., 2] > 1.0e-6)
    depth = np.zeros(rows.shape, dtype=np.float32)
    depth[valid] = np.log(camera[..., 2][valid]).astype(np.float32)
    return depth, valid


def _route(image_id: str) -> str:
    return str(image_id).split("/", 1)[0]


def _select_direct_route_ids(
    token_image_ids: list[str], contributor_image_ids: set[str], route: str, count: int,
) -> list[str]:
    """Select exactly the manifest-order prefix consumed by retrieval ``max_queries``."""

    if int(count) <= 0 or len(token_image_ids) != len(set(token_image_ids)):
        raise ValueError("direct teacher token IDs/count differ")
    eligible = [
        str(image_id) for image_id in token_image_ids
        if str(image_id) in contributor_image_ids and _route(str(image_id)) == str(route)
    ]
    if len(eligible) < int(count):
        raise ValueError(f"direct teacher route {route} lacks {count} views")
    return eligible[:int(count)]


def _retrieval_lineage(metadata: dict[str, object]) -> dict[str, object]:
    lineage = {key: metadata.get(key) for key in _RETRIEVAL_LINEAGE_KEYS}
    if (
        lineage["canonical_field_coordinate_correct"] is not True
        or any(lineage[key] in (None, "") for key in _RETRIEVAL_LINEAGE_KEYS)
    ):
        raise ValueError("teacher retrieval lineage is incomplete or non-coordinate-correct")
    return lineage


def _estimated_source_uncompressed_bytes(query_count: int, source_slots: int) -> int:
    """Conservative bound for the v2 numeric source payload before compression."""

    count = int(query_count); slots = int(source_slots)
    if count <= 0 or slots <= 0:
        raise ValueError("source resource estimate dimensions must be positive")
    per_query = (
        2304 * 32 * np.dtype(np.float32).itemsize  # exact frozen RADIO channel groups
        + 2304 * slots * np.dtype(np.int32).itemsize  # q_ret child rows
        + 2304 * slots * np.dtype(np.float32).itemsize  # q_ret probability
        + 2304 * np.dtype(np.float32).itemsize  # query reliability
        + 3 * 64 * np.dtype("U1").itemsize  # three SHA256 strings
        + 256 * np.dtype("U1").itemsize  # image ID and role safety allowance
    )
    # Hierarchy arrays and NPZ headers are query-independent; 8 MiB is a strict
    # practical safety reserve for this map family rather than a hidden query term.
    return int(count * per_query + 8 * 1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="")
    parser.add_argument("--contributors", default="")
    parser.add_argument("--token_manifest", action="append", default=[])
    parser.add_argument("--artifact_root", default=".")
    parser.add_argument("--fit_count", type=int, default=32)
    parser.add_argument("--held_count", type=int, default=16)
    parser.add_argument("--allow_mapping_route_teacher_control", action="store_true")
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--canonical_field_audit", required=True)
    parser.add_argument("--retrieval_dir", action="append", required=True)
    parser.add_argument("--fit_route", required=True)
    parser.add_argument("--held_route", required=True)
    parser.add_argument("--source_slots", type=int, default=16)
    parser.add_argument("--max_source_uncompressed_mib", type=float, default=1024.0)
    parser.add_argument("--output_source", required=True)
    parser.add_argument("--output_fit_teacher", required=True)
    parser.add_argument("--output_held_teacher", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    outputs = [Path(args.output_source), Path(args.output_fit_teacher), Path(args.output_held_teacher)]
    sidecars = [path.with_suffix(".json") for path in outputs]
    if any(path.exists() for path in outputs + sidecars):
        raise FileExistsError("refusing to overwrite query footprint teacher artifacts")
    if args.fit_route == args.held_route or int(args.source_slots) <= 0:
        raise ValueError("fit and held routes must differ and source slots must be positive")
    if not np.isfinite(float(args.max_source_uncompressed_mib)) or float(
        args.max_source_uncompressed_mib
    ) <= 0.0:
        raise ValueError("source uncompressed memory budget must be positive")
    retrieval_dirs = _parse_route_directories(args.retrieval_dir)
    if set(retrieval_dirs) != {str(args.fit_route), str(args.held_route)}:
        raise ValueError("retrieval directories must exactly cover fit and held routes")

    inventory_path = Path(args.inventory) if args.inventory else None
    if inventory_path is not None:
        if args.contributors or args.token_manifest:
            raise ValueError("inventory and direct contributor/token modes are exclusive")
        with np.load(inventory_path, allow_pickle=False) as source:
            inventory_metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
            inventory = {
                name: np.asarray(source[name]) for name in source.files if name != "metadata_json"
            }
        if (
            inventory_metadata.get("artifact_type")
            != "goal_maplet_controlled_pose_training_inventory_v1"
            or inventory_metadata.get("content_sha256") != arrays_sha256(inventory)
            or inventory_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
            or inventory_metadata.get("contains_standard_test_queries") is not False
        ):
            raise ValueError("controlled teacher inventory contract differs")
    else:
        if (
            not args.contributors or not args.token_manifest
            or int(args.fit_count) <= 0 or int(args.held_count) <= 0
        ):
            raise ValueError("direct mode requires contributors, token manifest, and positive counts")
        contributors = _load_contributors(Path(args.contributors))
        tokens = _load_token_inventory(
            [Path(value) for value in args.token_manifest],
            artifact_root=Path(args.artifact_root).resolve(),
        )
        direct_ids = []
        token_image_ids = list(tokens)
        contributor_image_ids = set(contributors)
        for route, count in (
            (str(args.fit_route), int(args.fit_count)),
            (str(args.held_route), int(args.held_count)),
        ):
            direct_ids.extend(_select_direct_route_ids(
                token_image_ids, contributor_image_ids, route, count,
            ))
        poses, token_paths, contributor_paths = [], [], []
        for image_id in direct_ids:
            contributor_path = contributors[image_id]
            with np.load(contributor_path, allow_pickle=False) as contributor:
                poses.append(np.asarray(contributor["pose_w2c"], dtype=np.float64))
            token_paths.append(str(tokens[image_id])); contributor_paths.append(str(contributor_path))
        inventory = {
            "image_ids": np.asarray(direct_ids),
            "radio_token_paths": np.asarray(token_paths),
            "radio_file_sha256": np.asarray([file_sha256(Path(value)) for value in token_paths]),
            "contributor_paths": np.asarray(contributor_paths),
            "contributor_file_sha256": np.asarray([
                file_sha256(Path(value)) for value in contributor_paths
            ]),
            "candidate_poses_w2c": np.asarray(poses)[:, None],
        }
        inventory_metadata = {
            "artifact_type": "goal_maplet_query_footprint_direct_teacher_inventory_v1",
            "content_sha256": arrays_sha256(inventory),
            "candidate_zero_is_diagnostic_gt_anchor": True,
            "contains_standard_test_queries": False,
            "selection_semantics": (
                "per_route_RADIO_manifest_order_prefix_matching_retrieval_max_queries_v1"
            ),
        }
    image_ids = np.asarray(inventory["image_ids"]).reshape(-1)
    routes = np.asarray([_route(str(value)) for value in image_ids])
    if set(routes.tolist()) != {str(args.fit_route), str(args.held_route)}:
        raise ValueError("teacher inventory route set differs")
    if np.asarray(inventory["candidate_poses_w2c"]).shape != (image_ids.size, 1, 4, 4):
        raise ValueError("teacher inventory must contain one GT render pose per image")
    estimated_source_bytes = _estimated_source_uncompressed_bytes(
        int(image_ids.size), int(args.source_slots),
    )
    source_budget_bytes = int(float(args.max_source_uncompressed_mib) * 1024.0 * 1024.0)
    if estimated_source_bytes > source_budget_bytes:
        raise MemoryError(
            "query footprint source estimate exceeds max_source_uncompressed_mib; "
            "use route-balanced shards rather than one unbounded artifact"
        )

    physical_path = Path(args.physical_map); field_path = Path(args.canonical_field)
    audit_path = Path(args.canonical_field_audit)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    audit = json.loads(audit_path.read_text())
    map_routes = set(str(value) for value in audit.get("mapping_trajectory_ids", ()))
    teacher_map_overlap = sorted(set(routes.tolist()) & map_routes)
    if teacher_map_overlap and not bool(args.allow_mapping_route_teacher_control):
        raise ValueError("mapping-route teacher overlap requires explicit control opt-in")
    if (
        field.physical_map_sha256 != physical.content_sha256
        or audit.get("canonical_field_sha256") != field.content_sha256
        or audit.get("storage_contract", {}).get("coordinate_correct") is not True
    ):
        raise ValueError("teacher map is not coordinate-correct")
    hierarchy = build_pose_transport_hierarchy(physical)
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(args.device))

    radio_group_rows, source_rows, source_probabilities, reliability_rows = [], [], [], []
    retrieval_hashes, retrieval_file_hashes, radio_hashes = [], [], []
    retrieval_lineage: dict[str, object] | None = None
    retrieval_promotion_eligible: list[bool] = []
    retrieval_control_only: list[bool] = []
    teacher_rows, teacher_weights, teacher_depths, teacher_depth_valid = [], [], [], []
    render_seconds = []
    started = time.monotonic()
    for index, image_value in enumerate(image_ids.tolist()):
        image_id = str(image_value); route = _route(image_id)
        token_path = Path(str(inventory["radio_token_paths"][index]))
        contributor_path = Path(str(inventory["contributor_paths"][index]))
        if (
            file_sha256(token_path) != str(inventory["radio_file_sha256"][index])
            or file_sha256(contributor_path) != str(inventory["contributor_file_sha256"][index])
        ):
            raise ValueError("teacher source file lineage differs")
        with np.load(token_path, allow_pickle=False) as token:
            if set(token.files) != {"radio_final"}:
                raise ValueError("teacher RADIO token members differ")
            radio = np.asarray(token["radio_final"], dtype=np.float32)
        if radio.shape != (1280, 36, 64) or np.any(~np.isfinite(radio)):
            raise ValueError("teacher RADIO tensor differs")
        retrieval_path = retrieval_dirs[route] / (image_id.replace("/", "__") + ".npz")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.image_id != image_id or retrieval.physical_map_sha256 != physical.content_sha256:
            raise ValueError("teacher retrieval lineage differs")
        current_lineage = _retrieval_lineage(dict(retrieval.metadata))
        if retrieval_lineage is None:
            retrieval_lineage = current_lineage
        elif current_lineage != retrieval_lineage:
            raise ValueError("teacher retrieval artifacts do not share one frozen lineage")
        if current_lineage["canonical_field_file_sha256"] != file_sha256(field_path):
            raise ValueError("teacher retrieval/canonical field file lineage differs")
        if int(args.source_slots) > retrieval.token_child_rows.shape[1]:
            raise ValueError("teacher source slots exceed q_ret")
        pose = np.asarray(inventory["candidate_poses_w2c"][index, 0], dtype=np.float64)
        rendered_batch = scene.render_exact_batch(
            pose[None], _camera(contributor_path), width=64, height=36, top_l=4,
        )
        rendered = rendered_batch.rendered[0]
        child_rows = np.asarray(rendered.child_rows, dtype=np.int32).reshape(2304, 4)
        child_weights = np.asarray(rendered.child_weights, dtype=np.float32).reshape(2304, 4)
        log_depth, depth_valid = _metric_log_depth(child_rows, pose, physical.child_centers)
        radio_group_rows.append(
            radio.reshape(32, 1280 // 32, 2304).mean(axis=1).T.astype(np.float32)
        )
        source_rows.append(retrieval.token_child_rows[:, :int(args.source_slots)].astype(np.int32))
        source_probabilities.append(
            retrieval.token_child_probabilities[:, :int(args.source_slots)].astype(np.float32)
        )
        reliability_rows.append(query_only_pose_reliability_weights(retrieval).astype(np.float32))
        retrieval_hashes.append(retrieval.content_sha256)
        retrieval_file_hashes.append(file_sha256(retrieval_path))
        radio_hashes.append(file_sha256(token_path))
        retrieval_promotion_eligible.append(bool(retrieval.metadata.get("promotion_eligible", False)))
        retrieval_control_only.append(bool(retrieval.metadata.get("control_only", True)))
        teacher_rows.append(child_rows); teacher_weights.append(child_weights)
        teacher_depths.append(log_depth); teacher_depth_valid.append(depth_valid)
        render_seconds.append(float(rendered_batch.audit.total_seconds))
        print(json.dumps({
            "image_id": image_id, "route_role": (
                "fit" if route == str(args.fit_route) else "held"
            ), "render_seconds": render_seconds[-1],
        }), flush=True)

    source_arrays = {
        "image_ids": image_ids,
        "route_roles": np.asarray([
            "fit" if route == str(args.fit_route) else "held" for route in routes.tolist()
        ]),
        "radio_group_means": np.stack(radio_group_rows),
        "source_child_rows": np.stack(source_rows),
        "source_child_probabilities": np.stack(source_probabilities),
        "query_reliability": np.stack(reliability_rows),
        "retrieval_content_sha256": np.asarray(retrieval_hashes),
        "retrieval_file_sha256": np.asarray(retrieval_file_hashes),
        "radio_file_sha256": np.asarray(radio_hashes),
        "hierarchy_child_parent_ids": hierarchy.child_parent_ids.astype(np.int32),
        "hierarchy_child_support_ids": hierarchy.child_support_ids.astype(np.int32),
    }
    forbidden_source_names = {
        "pose_w2c", "candidate_poses_w2c", "teacher_child_rows",
        "teacher_child_weights", "teacher_metric_log_depth",
    }
    if forbidden_source_names & set(source_arrays):
        raise AssertionError("deployable source artifact contains pose or teacher labels")
    source_uncompressed_bytes = int(sum(value.nbytes for value in source_arrays.values()))
    if source_uncompressed_bytes > source_budget_bytes:
        raise MemoryError("actual query footprint source exceeds declared resource budget")
    source_metadata = {
        "artifact_type": SOURCE_SCHEMA,
        "content_sha256": arrays_sha256(source_arrays),
        "inventory_file_sha256": (
            file_sha256(inventory_path) if inventory_path is not None else None
        ),
        "inventory_content_sha256": inventory_metadata["content_sha256"],
        "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_file_sha256": file_sha256(field_path),
        "canonical_field_sha256": field.content_sha256,
        "canonical_field_audit_file_sha256": file_sha256(audit_path),
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "fit_route": str(args.fit_route), "held_route": str(args.held_route),
        "fit_count": int(np.sum(routes == str(args.fit_route))),
        "held_count": int(np.sum(routes == str(args.held_route))),
        "teacher_routes_overlapping_canonical_map": teacher_map_overlap,
        "mapping_route_overlap_control": bool(teacher_map_overlap),
        "map_storage_disjoint_from_teacher_routes": not bool(teacher_map_overlap),
        "query_readout_and_calibration_route_disjoint": bool(
            retrieval_promotion_eligible and all(retrieval_promotion_eligible)
        ),
        "strict_route_disjoint_predictability_claim_supported": bool(
            not teacher_map_overlap
            and retrieval_promotion_eligible
            and all(retrieval_promotion_eligible)
        ),
        "retrieval_lineage": retrieval_lineage,
        "all_retrieval_artifacts_promotion_eligible": bool(
            retrieval_promotion_eligible and all(retrieval_promotion_eligible)
        ),
        "any_retrieval_artifact_control_only": bool(
            any(retrieval_control_only) or not retrieval_control_only
        ),
        "source_slots": int(args.source_slots),
        "source_feature_representation": (
            "query_only_RADIO_32_fixed_channel_group_means_float32_v1"
        ),
        "raw_RADIO_persisted": False,
        "estimated_source_uncompressed_bytes": int(estimated_source_bytes),
        "actual_source_uncompressed_bytes": int(source_uncompressed_bytes),
        "max_source_uncompressed_mib": float(args.max_source_uncompressed_mib),
        "route_balanced_sharding_required_above_budget": True,
        "contains_pose_or_GT_teacher_labels": False,
        "inference_inputs": [
            "fixed_RADIO_channel_group_means", "pose_free_q_ret", "query_only_reliability",
            "query_independent_physical_hierarchy",
        ],
        "uses_alike": False, "uses_point_correspondences": False,
        "uses_pnp": False, "uses_absolute_pose_regression": False,
    }
    _atomic_save(outputs[0], source_arrays, source_metadata)
    sidecars[0].write_text(json.dumps(source_metadata, indent=2, sort_keys=True) + "\n")

    all_teacher = {
        "image_ids": image_ids,
        "teacher_child_rows": np.stack(teacher_rows),
        "teacher_child_weights": np.stack(teacher_weights),
        "teacher_metric_log_depth": np.stack(teacher_depths),
        "teacher_metric_depth_valid": np.stack(teacher_depth_valid),
    }
    for role, route, output, sidecar in (
        ("fit", str(args.fit_route), outputs[1], sidecars[1]),
        ("held", str(args.held_route), outputs[2], sidecars[2]),
    ):
        selected = np.flatnonzero(routes == route)
        teacher_arrays = {
            name: np.asarray(value)[selected] for name, value in all_teacher.items()
        }
        teacher_metadata = {
            "artifact_type": TEACHER_SCHEMA,
            "content_sha256": arrays_sha256(teacher_arrays),
            "source_content_sha256": source_metadata["content_sha256"],
            "source_file_sha256": file_sha256(outputs[0]),
            "role": role, "route": route, "query_count": int(selected.size),
            "target_semantics": (
                "offline_GT_pose_3DGS_exact_render_token_x_physical_child_projected_mass_"
                "plus_absolute_metric_log_depth_v1"
            ),
            "primary_target": "projected_token_x_connected_support_mass",
            "metric_depth_is_optional_diagnostic_not_default_score": True,
            "GT_pose_consumed_offline": True,
            "GT_pose_stored": False,
            "render_seconds": float(sum(render_seconds[index] for index in selected.tolist())),
            "physical_map_sha256": physical.content_sha256,
            "canonical_field_sha256": field.content_sha256,
            "uses_alike": False, "uses_point_correspondences": False,
            "uses_pnp": False, "uses_absolute_pose_regression": False,
            "production_eligible": False,
        }
        _atomic_save(output, teacher_arrays, teacher_metadata)
        sidecar.write_text(json.dumps(teacher_metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source": source_metadata,
        "elapsed_seconds": float(time.monotonic() - started),
        "output_source": str(outputs[0]),
        "output_fit_teacher": str(outputs[1]),
        "output_held_teacher": str(outputs[2]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
