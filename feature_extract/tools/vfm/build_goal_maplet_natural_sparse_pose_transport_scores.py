"""Score a frozen natural pose-free pool with the fixed q_pose backend.

This command intentionally has no direct-candidate/ground-truth input.  It
loads only the pose-free K-candidate pool, RADIO tensors, frozen retrieval
posteriors, camera intrinsics and the frozen map/readouts.  The compact score
artifact can be joined with post-freeze pose-error labels by the separate
evaluation command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.dense_fixed_identity_transport import (
    DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
    DensePoseTransportHierarchyGPU,
    dense_fixed_identity_transport,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    NATURAL_SCORE_SCHEMA,
    load_pose_free_camera_binding,
    load_pose_free_candidate_pool,
    pose_free_pool_arrays,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_transport_hierarchy import (
    HIERARCHY_SEMANTICS,
    build_pose_transport_hierarchy,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    query_only_pose_reliability_weights,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    QueryPoseHeadOutput,
    load_minimal_pose_transport_readout,
    pose_transport_model_content_sha256,
)


FIXED_KERNEL_SEMANTICS = (
    "differentiable_fixed_local_kernel_source_target_capacity_pose_transport_v2"
)
IDENTITY_READOUT_SEMANTICS = (
    "shared_radio128_identity_feature_hierarchy_layout_reference_gauge_two_dof_v3"
)
CANONICAL_FIELD_SEMANTICS = "single_view_independent_canonical_field_control_v1"


def _load_token_inventory(
    paths: list[Path], *, artifact_root: Path,
) -> tuple[dict[str, tuple[Path, str]], list[dict[str, str]]]:
    result: dict[str, tuple[Path, str]] = {}
    manifests: list[dict[str, str]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError("RADIO token manifest lacks records")
        for row in rows:
            image_id = str(row.get("image_id", ""))
            token_path = Path(str(row.get("token_path", "")))
            if not token_path.is_absolute():
                token_path = artifact_root / token_path
            checksum = str(row.get("checksum", ""))
            if (
                not image_id or image_id in result or not token_path.is_file()
                or len(checksum) != 64
            ):
                raise ValueError("RADIO token inventory is incomplete or duplicated")
            result[image_id] = (token_path.resolve(), checksum)
        manifests.append({"path": str(path.resolve()), "file_sha256": file_sha256(path)})
    return result, manifests


def _camera_without_pose(
    path: Path, *, expected_image_id: str,
) -> tuple[ColmapCamera, str]:
    """Read only whitelisted intrinsics; never inspect/hash the GT pose member."""

    model_id, width, height, params, binding = load_pose_free_camera_binding(
        path, image_id=expected_image_id,
    )
    return ColmapCamera(0, model_id, width, height, params), binding


def _ray_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height) * 2.0 - 1.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width) * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)[None]


def _feature_only_query(
    value: QueryPoseHeadOutput, *, stage: str,
) -> QueryPoseHeadOutput:
    depth = {
        "coarse": "ordinal_depth_v1",
        "medium": "centered_log_depth_v1",
        "fine": "metric_log_depth_with_uncertainty_v1",
    }
    if str(stage) not in depth:
        raise ValueError("natural score transport stage differs")
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
        depth_semantics=depth[str(stage)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--token_manifest", action="append", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--canonical_field_audit", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--training_dataset_metadata", required=True,
        help=(
            "JSON-only sidecar binding the model to the map; the rendered/labelled "
            "training NPZ and GT-derived training report are deliberately unopened"
        ),
    )
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--artifact_root", default=".")
    parser.add_argument("--stage", choices=("coarse", "medium", "fine"), default="coarse")
    parser.add_argument("--maximum_candidates", type=int, default=64)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--source_slots", type=int, default=16)
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow_non_strict_mapper_control", action="store_true",
        help="allow an explicitly non-production mapper-pretraining control",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite natural pose-transport scores")
    if (
        int(args.maximum_candidates) <= 0 or int(args.maximum_queries) < 0
        or int(args.query_start) < 0 or int(args.source_slots) <= 0
        or int(args.render_batch_size) <= 0
    ):
        raise ValueError("natural score bounds must be positive")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    pool_path = Path(args.candidate_pool).resolve()
    pool = load_pose_free_candidate_pool(pool_path)
    pool_arrays = pose_free_pool_arrays(
        pool,
        maximum_candidates=int(args.maximum_candidates),
        query_start=int(args.query_start),
        maximum_queries=int(args.maximum_queries),
    )
    image_ids = [str(value) for value in pool_arrays["image_ids"].tolist()]
    query_routes = sorted({value.split("/", 1)[0] for value in image_ids})

    physical_path = Path(args.physical_map).resolve()
    field_path = Path(args.canonical_field).resolve()
    field_audit_path = Path(args.canonical_field_audit).resolve()
    mapper_path = Path(args.surface_mapper).resolve()
    contract_path = Path(args.field_feature_contract).resolve()
    model_path = Path(args.model).resolve()
    training_metadata_path = Path(args.training_dataset_metadata).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    field_audit = json.loads(field_audit_path.read_text())
    if (
        field_audit.get("canonical_field_sha256") != field.content_sha256
        or field_audit.get("storage_contract", {}).get("coordinate_correct") is not True
        or set(query_routes) & set(str(value) for value in field_audit.get("mapping_trajectory_ids", ()))
    ):
        raise ValueError("canonical field audit is not coordinate-correct/map-disjoint")
    hierarchy = build_pose_transport_hierarchy(physical)

    training_metadata = json.loads(training_metadata_path.read_text())
    model, model_metadata = load_minimal_pose_transport_readout(model_path, device=str(device))
    if (
        training_metadata.get("artifact_type")
        != "goal_maplet_real_sparse_pose_transport_dataset_v2"
        or model_metadata.get("dataset_content_sha256") != training_metadata.get("content_sha256")
        or model_metadata.get("physical_map_sha256") != physical.content_sha256
        or training_metadata.get("physical_map_sha256") != physical.content_sha256
        or training_metadata.get("physical_map_file_sha256") != file_sha256(physical_path)
        or training_metadata.get("canonical_field_sha256") != field.content_sha256
        or training_metadata.get("canonical_field_file_sha256") != file_sha256(field_path)
        or training_metadata.get("canonical_field_audit_file_sha256") != file_sha256(field_audit_path)
        or model_metadata.get("hierarchy_content_sha256") != hierarchy.content_sha256
        or model_metadata.get("hierarchy_semantics") != HIERARCHY_SEMANTICS
        or model_metadata.get("transport_semantics") != FIXED_KERNEL_SEMANTICS
        or model_metadata.get("readout_training_semantics") != IDENTITY_READOUT_SEMANTICS
        or model_metadata.get("map_pose_field_semantics") != CANONICAL_FIELD_SEMANTICS
        or training_metadata.get("map_pose_field_semantics") != CANONICAL_FIELD_SEMANTICS
        or model_metadata.get("shared_query_map_projection") is not True
        or model_metadata.get("view_conditioned_field_sha256") is not None
    ):
        raise ValueError("natural score model/map/training lineage differs")
    strict_mapper = bool(model_metadata.get("strict_query_representation_route_disjoint"))
    if not strict_mapper and not bool(args.allow_non_strict_mapper_control):
        raise ValueError(
            "pose readout uses a non-strict pretrained mapper; pass the explicit control flag"
        )
    contract = json.loads(contract_path.read_text())
    mapper_file_hash = file_sha256(mapper_path)
    if (
        contract.get("artifact_type") != "goal_maplet_field_feature_contract_v1"
        or contract.get("canonical_field_sha256") != field.content_sha256
        or contract.get("query_readout_type") != "surface_maplet_mapper"
        or contract.get("query_readout_sha256") != mapper_file_hash
        or model_metadata.get("surface_mapper_file_sha256") != mapper_file_hash
    ):
        raise ValueError("surface mapper/field/model contract differs")
    mapper, mapper_metadata = load_surface_maplet_mapper(mapper_path, device=str(device))
    mapper.model.to(device).eval()
    model.eval()
    if (
        int(model.config.radio_channels) != 128
        or int(model.config.map_feature_dim) != 128
        or int(model.config.pose_code_dim) != 128
        or not bool(model.config.shared_query_map_projection)
    ):
        raise ValueError("natural dense scorer requires the shared RADIO128 identity readout")

    token_inventory, token_manifest_bindings = _load_token_inventory(
        [Path(value).resolve() for value in args.token_manifest],
        artifact_root=Path(args.artifact_root).resolve(),
    )
    if not set(image_ids).issubset(token_inventory):
        raise ValueError("pose-free candidate pool lacks RADIO tensors")
    contributors = Path(args.contributors).resolve()
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(device))
    dense_hierarchy = DensePoseTransportHierarchyGPU(hierarchy, device=device)
    candidate_pose = np.asarray(pool_arrays["candidate_poses_w2c"], dtype=np.float64)
    candidate_valid = np.asarray(pool_arrays["candidate_valid"], dtype=bool)
    query_count, candidate_count = candidate_valid.shape
    scores = np.full((query_count, candidate_count), -1.0, dtype=np.float32)
    component_statistics = np.zeros((query_count, candidate_count, 6), dtype=np.float32)
    target_mean_mass = np.zeros((query_count, candidate_count), dtype=np.float32)
    target_feature_fraction = np.zeros((query_count, candidate_count), dtype=np.float32)
    target_visible_fraction = np.zeros((query_count, candidate_count), dtype=np.float32)
    source_retained_fraction = np.zeros(query_count, dtype=np.float32)
    retrieval_hashes: list[str] = []
    radio_hashes: list[str] = []
    camera_intrinsics_hashes: list[str] = []
    render_seconds = np.zeros(query_count, dtype=np.float64)
    score_seconds = np.zeros(query_count, dtype=np.float64)
    maximum_source_excess = 0.0
    maximum_target_excess = 0.0

    for query_index, image_id in enumerate(image_ids):
        retrieval_path = Path(str(pool_arrays["retrieval_paths"][query_index]))
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if (
            retrieval.image_id != image_id
            or retrieval.content_sha256 != str(pool_arrays["retrieval_content_sha256"][query_index])
            or retrieval.physical_map_sha256 != physical.content_sha256
        ):
            raise ValueError("pose-free retrieval evidence lineage differs")
        if int(args.source_slots) > retrieval.token_child_rows.shape[1]:
            raise ValueError("natural score source slot count exceeds retrieval")
        token_path, declared_radio_hash = token_inventory[image_id]
        actual_radio_hash = file_sha256(token_path)
        if actual_radio_hash != declared_radio_hash:
            raise ValueError("RADIO token manifest checksum differs")
        with np.load(token_path, allow_pickle=False) as data:
            if set(data.files) != {"radio_final"}:
                raise ValueError("RADIO token NPZ members differ")
            radio = np.asarray(data["radio_final"], dtype=np.float32)
        if radio.shape != (1280, 36, 64) or np.any(~np.isfinite(radio)):
            raise ValueError("natural score RADIO tensor differs")
        contributor_path = contributors / (image_id.replace("/", "__") + ".npz")
        if not contributor_path.is_file():
            raise ValueError("natural score camera contributor is missing")
        camera, camera_intrinsics_hash = _camera_without_pose(
            contributor_path, expected_image_id=image_id,
        )
        with torch.no_grad():
            mapped = mapper.model(torch.as_tensor(radio, device=device)[None])
            query = _feature_only_query(
                model(mapped, _ray_grid(36, 64, device=device)), stage=str(args.stage)
            )
        full_mass = float(np.sum(retrieval.token_child_probabilities, dtype=np.float64))
        source_rows = retrieval.token_child_rows[:, : int(args.source_slots)]
        source_probability = retrieval.token_child_probabilities[:, : int(args.source_slots)]
        retained_mass = float(np.sum(source_probability, dtype=np.float64))
        source_retained_fraction[query_index] = retained_mass / max(full_mass, 1.0e-12)
        reliability = query_only_pose_reliability_weights(retrieval)
        for begin in range(0, candidate_count, int(args.render_batch_size)):
            end = min(begin + int(args.render_batch_size), candidate_count)
            batch_indices = begin + np.flatnonzero(candidate_valid[query_index, begin:end])
            if batch_indices.size == 0:
                continue
            rendered = scene.render_exact_batch(
                candidate_pose[query_index, batch_indices], camera,
                width=64, height=36, top_l=4,
            )
            render_seconds[query_index] += float(rendered.audit.total_seconds)
            rows = np.stack([
                np.asarray(value.child_rows, dtype=np.int32).reshape(2304, 4)
                for value in rendered.rendered
            ])
            mass = np.stack([
                np.asarray(value.child_weights, dtype=np.float32).reshape(2304, 4)
                for value in rendered.rendered
            ])
            feature = np.stack([
                np.asarray(value.child_features, dtype=np.float32).reshape(2304, 4, 128)
                for value in rendered.rendered
            ])
            feature_valid = np.stack([
                np.asarray(value.child_feature_valid, dtype=bool).reshape(2304, 4)
                for value in rendered.rendered
            ])
            feature_confidence = mass * feature_valid.astype(np.float32)
            batch_mass = mass.sum(axis=2)
            target_mean_mass[query_index, batch_indices] = batch_mass.mean(axis=1)
            target_visible_fraction[query_index, batch_indices] = np.mean(
                batch_mass > 1.0e-6, axis=1
            )
            valid_feature_mass = np.sum(mass * feature_valid, axis=(1, 2), dtype=np.float64)
            total_feature_mass = np.sum(mass, axis=(1, 2), dtype=np.float64)
            target_feature_fraction[query_index, batch_indices] = (
                valid_feature_mass / np.maximum(total_feature_mass, 1.0e-12)
            ).astype(np.float32)
            score_started = time.monotonic()
            with torch.no_grad():
                result = dense_fixed_identity_transport(
                    model, query, source_rows, source_probability, reliability,
                    retrieval.token_xy, rows, mass, feature, feature_confidence,
                    feature_valid,
                    dense_hierarchy, stage=str(args.stage), height=36, width=64,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            score_seconds[query_index] += time.monotonic() - score_started
            scores[query_index, batch_indices] = result.scores.detach().cpu().numpy()
            component_statistics[query_index, batch_indices] = (
                result.component_statistics.detach().cpu().numpy()
            )
            maximum_source_excess = max(
                maximum_source_excess, result.maximum_source_capacity_excess
            )
            maximum_target_excess = max(
                maximum_target_excess, result.maximum_target_capacity_excess
            )
        retrieval_hashes.append(retrieval.content_sha256)
        radio_hashes.append(actual_radio_hash)
        camera_intrinsics_hashes.append(camera_intrinsics_hash)
        print(json.dumps({
            "query_index": int(args.query_start) + query_index,
            "image_id": image_id,
            "candidate_count": candidate_count,
            "render_seconds": float(render_seconds[query_index]),
            "score_seconds": float(score_seconds[query_index]),
            "score_min": float(np.min(scores[query_index])),
            "score_max": float(np.max(scores[query_index])),
        }), flush=True)

    arrays = {
        "image_ids": np.asarray(image_ids),
        "candidate_poses_w2c": candidate_pose,
        "candidate_valid": candidate_valid,
        "scores": scores,
        "component_statistics": component_statistics,
        "retrieval_content_sha256": np.asarray(retrieval_hashes),
        "radio_file_sha256": np.asarray(radio_hashes),
        "camera_intrinsics_content_sha256": np.asarray(camera_intrinsics_hashes),
        "target_mean_rendered_mass": target_mean_mass,
        "target_feature_valid_mass_fraction": target_feature_fraction,
        "target_visible_token_fraction": target_visible_fraction,
        "source_retained_mass_fraction": source_retained_fraction,
        "query_render_seconds": render_seconds,
        "query_score_seconds": score_seconds,
    }
    metadata: dict[str, object] = {
        "artifact_type": NATURAL_SCORE_SCHEMA,
        "content_sha256": arrays_sha256(arrays),
        "query_count": query_count,
        "candidate_count": candidate_count,
        "query_start": int(args.query_start),
        "maximum_queries": int(args.maximum_queries),
        "candidate_pool_content_sha256": str(pool["content_sha256"]),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_pool_semantics": str(pool["candidate_semantics"]),
        "candidate_pose_arrays_sha256": arrays_sha256({
            "candidate_poses_w2c": candidate_pose,
            "candidate_valid": candidate_valid,
        }),
        "candidate_zero_is_present": False,
        "pose_error_labels_opened_during_scoring": False,
        "direct_candidate_dataset_opened_during_scoring": False,
        "contributor_file_bytes_hashed_during_scoring": False,
        "contributor_pose_member_opened_during_scoring": False,
        "candidate_pool_scores_consumed": False,
        "transport_stage": str(args.stage),
        "transport_semantics": FIXED_KERNEL_SEMANTICS,
        "dense_replay_semantics": DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
        "readout_training_semantics": IDENTITY_READOUT_SEMANTICS,
        "model_content_sha256": pose_transport_model_content_sha256(model),
        "model_file_sha256": file_sha256(model_path),
        "training_dataset_content_sha256": str(training_metadata["content_sha256"]),
        "training_dataset_metadata_file_sha256": file_sha256(training_metadata_path),
        "rendered_training_dataset_opened_during_scoring": False,
        "gt_derived_model_report_opened_during_scoring": False,
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": file_sha256(physical_path),
        "hierarchy_content_sha256": hierarchy.content_sha256,
        "hierarchy_semantics": HIERARCHY_SEMANTICS,
        "canonical_field_sha256": field.content_sha256,
        "canonical_field_file_sha256": file_sha256(field_path),
        "canonical_field_audit_file_sha256": file_sha256(field_audit_path),
        "map_pose_field_semantics": CANONICAL_FIELD_SEMANTICS,
        "surface_mapper_file_sha256": mapper_file_hash,
        "field_feature_contract_file_sha256": file_sha256(contract_path),
        "mapper_supervision_audit": model_metadata.get("mapper_supervision_audit"),
        "strict_query_representation_route_disjoint": strict_mapper,
        "non_strict_pretrained_mapper_control": not strict_mapper,
        "known_v1_mapper_coordinate_status": (
            "control_only_not_strict_coordinate_claim" if not strict_mapper else "strict_artifact_contract"
        ),
        "token_manifest_files": token_manifest_bindings,
        "source_slots": int(args.source_slots),
        "render_batch_size": int(args.render_batch_size),
        "total_render_seconds": float(np.sum(render_seconds)),
        "total_dense_score_seconds": float(np.sum(score_seconds)),
        "maximum_source_capacity_excess": float(maximum_source_excess),
        "maximum_target_capacity_excess": float(maximum_target_excess),
        "mapper_metadata_present": bool(mapper_metadata),
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "uses_absolute_pose_regression": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "production_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve())
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "query_count": query_count,
        "candidate_count": candidate_count,
        "total_render_seconds": metadata["total_render_seconds"],
        "total_dense_score_seconds": metadata["total_dense_score_seconds"],
        "strict_query_representation_route_disjoint": strict_mapper,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
