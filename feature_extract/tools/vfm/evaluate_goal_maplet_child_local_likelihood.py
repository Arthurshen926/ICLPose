"""Evaluate parent-conditioned child-local surface measurements from exact truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField, readout_canonical_field
from feature_extract.vfm.localization_goal_maplet.child_eligibility import ChildGeometryEligibility
from feature_extract.vfm.localization_goal_maplet.child_local_likelihood import predict_child_local_surface_likelihood
from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import (
    ChildLocalModeRankerArtifact,
    child_local_mode_runtime_features,
)
from feature_extract.vfm.localization_goal_maplet.child_local_pose import solve_joint_child_local_pose
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.oracle_pose import token_oracle_evidence
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_support import (
    aggregate_group_descriptors,
    all_token_coordinates,
    group_tokens_after_retrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import ValidityCalibration, retrieve_maplet_posterior
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion, pnp_pose_error
from feature_extract.vfm.surface_maplet_bank import RadioFinalRegionConfig, encode_radio_final_regions


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(
            camera_id=0,
            model_id=int(data["camera_model_id"]),
            width=int(data["camera_width"]),
            height=int(data["camera_height"]),
            params=tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def _solve_pose(xy: np.ndarray, xyz: np.ndarray, camera: ColmapCamera, image_id: str):
    if xy.shape[0] < 6 or np.unique(np.round(xyz, 5), axis=0).shape[0] < 6:
        return None
    matrix, distortion = camera_matrix_and_distortion(camera)
    seed = int.from_bytes(hashlib.sha256(image_id.encode("utf8")).digest()[:4], "little") & 0x7FFFFFFF
    cv2.setRNGSeed(seed)
    try:
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz.astype(np.float64), xy.astype(np.float64), matrix, distortion,
            iterationsCount=6000, reprojectionError=32.0, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not success or inliers is None or len(inliers) < 6:
        return None
    rows = np.asarray(inliers, dtype=np.int64).reshape(-1)
    try:
        rotation, translation = cv2.solvePnPRefineLM(
            xyz[rows].astype(np.float64), xy[rows].astype(np.float64),
            matrix, distortion, rotation, translation,
        )
    except cv2.error:
        pass
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rotation)[0]
    pose[:3, 3] = np.asarray(translation).reshape(3)
    return pose, int(len(rows))


def _metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--child_eligibility", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sample_output_npz", default="")
    parser.add_argument("--child_local_mode_ranker", default="")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--joint_pose_trials", type=int, default=512)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite child-local likelihood report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    contract = FieldFeatureContract.load_json(Path(args.field_feature_contract))
    if contract.query_readout_type != "surface_maplet_mapper":
        raise ValueError("child-local likelihood requires the frozen surface mapper readout")
    contract.validate(field, query_readout_path=Path(args.surface_mapper))
    eligibility = ChildGeometryEligibility.load_npz(Path(args.child_eligibility))
    if (
        eligibility.physical_map_sha256 != physical.content_sha256
        or eligibility.canonical_field_sha256 != field.content_sha256
    ):
        raise ValueError("child eligibility lineage differs")
    readout = readout_canonical_field(field, physical)
    calibration = ValidityCalibration.load_json(Path(args.validity_calibration))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    mode_ranker = None
    if str(args.child_local_mode_ranker):
        mode_ranker = ChildLocalModeRankerArtifact.load(Path(args.child_local_mode_ranker))
        for key, expected in (
            ("physical_map_sha256", physical.content_sha256),
            ("canonical_field_sha256", field.content_sha256),
            ("child_eligibility_sha256", eligibility.content_sha256),
            ("field_feature_contract_sha256", contract.content_sha256),
        ):
            if mode_ranker.metadata.get(key) != expected:
                raise ValueError(f"child-local mode ranker lineage differs: {key}")
    paths = sorted(Path(args.contributors).glob("*.npz"))[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        paths = paths[: int(args.maximum_queries)]
    context_config = RadioFinalRegionConfig()
    local_config = RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,))
    rows = []
    direct = {name: [] for name in (
        "child_center", "primitive_map", "posterior_expected", "oracle_top_modes",
        "coarse_geometry_map", "coarse_geometry_expected",
        "learned_mode",
    )}
    pose_values = {name: {"translation": [], "rotation": []} for name in (
        "child_center", "primitive_map", "posterior_expected",
        "coarse_geometry_map", "coarse_geometry_expected", "oracle_child_local",
        "learned_mode", "joint_multimodal",
    )}
    sample_features, sample_errors, sample_valid = [], [], []
    sample_image_ids, sample_trajectories = [], []
    for path in paths:
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        _, token_xy = all_token_coordinates(int(raw.shape[1]), int(raw.shape[2]))
        context = encode_radio_final_regions(mapped, token_xy, context_config)
        local = encode_radio_final_regions(mapped, token_xy, local_config)
        parent_ids, parent_probability, parent_null, _ = retrieve_maplet_posterior(
            context, readout.parent_descriptors, physical.maplet_ids,
            readout.parent_coverage > 0.0,
            maximum_candidates=64, temperature=0.07,
            null_similarity_center=float(calibration.center),
            null_similarity_scale=float(calibration.scale),
        )
        grouped = group_tokens_after_retrieval(
            token_xy, context, parent_ids[:, 0],
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_width=int(camera.width), image_height=int(camera.height),
            descriptor_half_size_tokens=2.0, minimum_descriptor_cosine=0.96,
        )
        grouped_local = aggregate_group_descriptors(
            local, grouped.member_offsets, grouped.member_token_indices
        )
        evidence = token_oracle_evidence(
            labels, physical, token_xy,
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_height=int(camera.height), image_width=int(camera.width), camera=camera,
        )
        child_rows, gt_xyz, gt_xy, group_rows = [], [], [], []
        for group in range(grouped.member_offsets.size - 1):
            members = grouped.member_token_indices[
                int(grouped.member_offsets[group]) : int(grouped.member_offsets[group + 1])
            ]
            members = members[evidence.child_rows[members] >= 0]
            if members.size == 0:
                continue
            score: dict[int, float] = {}
            for token in members.tolist():
                child = int(evidence.child_rows[token])
                score[child] = score.get(child, 0.0) + float(evidence.child_mass[token])
            child = min(score, key=lambda value: (-score[value], value))
            if not bool(eligibility.refinement_qualified[child]):
                continue
            selected = members[evidence.child_rows[members] == child]
            weight = evidence.child_mass[selected]
            if float(np.sum(weight)) <= 0.0:
                continue
            child_rows.append(child)
            gt_xyz.append(np.average(evidence.child_local_xyz[selected], axis=0, weights=weight))
            gt_xy.append(np.average(evidence.xy_px[selected], axis=0, weights=weight))
            group_rows.append(group)
        if child_rows:
            child_array = np.asarray(child_rows, dtype=np.int64)
            truth = np.asarray(gt_xyz, dtype=np.float64)
            xy = np.asarray(gt_xy, dtype=np.float64)
            prediction = predict_child_local_surface_likelihood(
                grouped_local[np.asarray(group_rows, dtype=np.int64)], child_array,
                physical, field, temperature=float(args.temperature),
                maximum_modes=int(args.maximum_modes),
            )
            candidate_points = physical.primitive_centers[np.maximum(prediction.mode_primitive_rows, 0)]
            candidate_error = np.linalg.norm(candidate_points - truth[:, None], axis=2)
            metric = {
                "child_center": np.linalg.norm(physical.child_centers[child_array] - truth, axis=1),
                "primitive_map": np.linalg.norm(prediction.map_points - truth, axis=1),
                "posterior_expected": np.linalg.norm(prediction.expected_points - truth, axis=1),
                "oracle_top_modes": np.min(np.where(prediction.mode_primitive_rows >= 0, candidate_error, np.inf), axis=1),
            }
            center_solution = _solve_pose(
                xy, physical.child_centers[child_array], camera,
                f"{metadata['image_id']}:child_center",
            )
            conditioned = None
            joint_solution = None
            if center_solution is not None:
                center_pose, _ = center_solution
                extent_px = grouped.extent[np.asarray(group_rows, dtype=np.int64)] * np.asarray(
                    [camera.width, camera.height], dtype=np.float64
                )
                conditioned = predict_child_local_surface_likelihood(
                    grouped_local[np.asarray(group_rows, dtype=np.int64)], child_array,
                    physical, field, temperature=float(args.temperature),
                    maximum_modes=int(args.maximum_modes),
                    query_xy_px=xy,
                    query_scale_px=np.maximum(0.5 * np.linalg.norm(extent_px, axis=1), 8.0),
                    pose_w2c=center_pose,
                    camera=camera,
                )
                metric["coarse_geometry_map"] = np.linalg.norm(conditioned.map_points - truth, axis=1)
                metric["coarse_geometry_expected"] = np.linalg.norm(conditioned.expected_points - truth, axis=1)
                mode_features, mode_valid = child_local_mode_runtime_features(
                    prediction, child_array, xy,
                    np.maximum(0.5 * np.linalg.norm(extent_px, axis=1), 8.0),
                    center_pose, camera, physical, field,
                )
                mode_points = physical.primitive_centers[
                    np.maximum(prediction.mode_primitive_rows, 0)
                ]
                mode_error = np.linalg.norm(mode_points - truth[:, None], axis=2)
                sample_features.append(mode_features)
                sample_errors.append(mode_error.astype(np.float32))
                sample_valid.append(mode_valid)
                sample_image_ids.extend([str(metadata["image_id"])] * child_array.size)
                sample_trajectories.extend([
                    str(metadata["image_id"]).split("/", 1)[0]
                ] * child_array.size)
                if mode_ranker is not None:
                    probability = mode_ranker.predict_probability(
                        mode_features.reshape(-1, mode_features.shape[2])
                    ).reshape(mode_valid.shape)
                    probability[~mode_valid] = -np.inf
                    selected_mode = np.argmax(probability, axis=1)
                    learned_points = mode_points[np.arange(child_array.size), selected_mode]
                    metric["learned_mode"] = np.linalg.norm(learned_points - truth, axis=1)
                    if int(args.joint_pose_trials) > 0:
                        joint_solution = solve_joint_child_local_pose(
                            xy,
                            np.maximum(0.5 * np.linalg.norm(extent_px, axis=1), 8.0),
                            mode_points,
                            probability,
                            mode_valid,
                            camera,
                            initial_pose_w2c=center_pose,
                            random_key=str(metadata["image_id"]),
                            trials=int(args.joint_pose_trials),
                        )
            for name, value in metric.items():
                direct[name].extend(value.tolist())
            pose_inputs = {
                "child_center": physical.child_centers[child_array],
                "primitive_map": prediction.map_points,
                "posterior_expected": prediction.expected_points,
                "oracle_child_local": truth,
            }
            if conditioned is not None:
                pose_inputs["coarse_geometry_map"] = conditioned.map_points
                pose_inputs["coarse_geometry_expected"] = conditioned.expected_points
                if mode_ranker is not None:
                    pose_inputs["learned_mode"] = learned_points
            pose_report = {}
            for name, xyz in pose_inputs.items():
                solved = _solve_pose(xy, xyz, camera, f"{metadata['image_id']}:{name}")
                if solved is None:
                    pose_report[name] = {"success": False}
                    continue
                pose, inliers = solved
                error = pnp_pose_error(pose, labels.pose_w2c)
                pose_values[name]["translation"].append(float(error.translation_m))
                pose_values[name]["rotation"].append(float(error.rotation_deg))
                pose_report[name] = {
                    "success": True,
                    "inliers": inliers,
                    "translation_m": float(error.translation_m),
                    "rotation_deg": float(error.rotation_deg),
                }
            if joint_solution is not None and joint_solution.success:
                error = pnp_pose_error(joint_solution.pose_w2c, labels.pose_w2c)
                pose_values["joint_multimodal"]["translation"].append(float(error.translation_m))
                pose_values["joint_multimodal"]["rotation"].append(float(error.rotation_deg))
                pose_report["joint_multimodal"] = {
                    "success": True,
                    "support_count": int(joint_solution.support_count),
                    "factor_score": float(joint_solution.score),
                    "translation_m": float(error.translation_m),
                    "rotation_deg": float(error.rotation_deg),
                }
            row = {
                "image_id": str(metadata["image_id"]),
                "eligible_group_count": int(child_array.size),
                "direct_error_median_m": {name: float(np.median(value)) for name, value in metric.items()},
                "feature_coverage_median": float(np.median(prediction.feature_coverage)),
                "maximum_similarity_median": float(np.median(prediction.maximum_similarity)),
                "entropy_median": float(np.median(prediction.entropy)),
                "pose": pose_report,
            }
        else:
            row = {"image_id": str(metadata["image_id"]), "eligible_group_count": 0, "pose": {}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "stage": "evaluate_goal_maplet_child_local_surface_likelihood",
        "protocol": "oracle_child_identity_runtime_vfm_local_measurement",
        "query_count": len(rows),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "field_feature_contract_sha256": contract.content_sha256,
        "child_eligibility_sha256": eligibility.content_sha256,
        "temperature": float(args.temperature),
        "maximum_modes": int(args.maximum_modes),
        "joint_pose_trials": int(args.joint_pose_trials),
        "child_local_mode_ranker": str(args.child_local_mode_ranker) if mode_ranker is not None else None,
        "child_local_mode_ranker_sha256": (
            file_sha256(Path(args.child_local_mode_ranker)) if mode_ranker is not None else None
        ),
        "direct_point_error_m": {name: _metrics(values) for name, values in direct.items()},
        "pose_error": {
            name: {
                "translation_m": _metrics(values["translation"]),
                "rotation_deg": _metrics(values["rotation"]),
            }
            for name, values in pose_values.items()
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if str(args.sample_output_npz):
        sample_output = Path(args.sample_output_npz)
        if sample_output.exists() and not args.force:
            raise FileExistsError("refusing to overwrite child-local rank samples")
        if not sample_features:
            raise ValueError("no child-local mode samples were generated")
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        sample_metadata = {
            "artifact_type": "goal_maplet_child_local_mode_rank_samples_v1",
            "physical_map_sha256": physical.content_sha256,
            "canonical_field_sha256": field.content_sha256,
            "field_feature_contract_sha256": contract.content_sha256,
            "child_eligibility_sha256": eligibility.content_sha256,
            "temperature": float(args.temperature),
            "maximum_modes": int(args.maximum_modes),
            "uses_gt_only_for_training_target": True,
            "uses_mapping_rgb": False,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": True,
            "deployment_artifact": False,
            "stored_downstream_embedding_count": 0,
        }
        np.savez_compressed(
            sample_output,
            features=np.concatenate(sample_features, axis=0).astype(np.float32),
            target_errors_m=np.concatenate(sample_errors, axis=0).astype(np.float32),
            valid=np.concatenate(sample_valid, axis=0).astype(bool),
            group_image_ids=np.asarray(sample_image_ids),
            group_trajectories=np.asarray(sample_trajectories),
            metadata_json=np.asarray(json.dumps(sample_metadata, sort_keys=True)),
        )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
