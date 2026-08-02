"""Evaluate V7 Top-64 whole-set pose signatures on held-out queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _retrieve,
)
from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)
from feature_extract.vfm.localization_v7.pose_signature import (
    geometry_only_pose_modes,
    local_linear_pose_modes,
    PoseMode,
    PoseSignatureBank,
    file_sha256,
    pose_modes_from_signature_scores,
    pose_signature_from_retrieval,
    rotation_angle_deg,
    score_pose_signatures,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_signature_bank", required=True)
    parser.add_argument("--retrieval_regions", required=True)
    parser.add_argument("--spatial_maplets", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--probability_calibration", required=True)
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_query_signatures_npz", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregation", default="topq_nms")
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _pose_error(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation = float(
        np.linalg.norm(
            camera_center_from_pose_w2c(pose)
            - camera_center_from_pose_w2c(target)
        )
    )
    rotation = float(
        rotation_angle_deg(
            np.asarray(pose, dtype=np.float64)[None, :3, :3],
            np.asarray(target, dtype=np.float64)[:3, :3],
        )[0]
    )
    return translation, rotation


def _rank_errors(
    poses: Sequence[np.ndarray], target: np.ndarray
) -> dict[str, object]:
    errors = [_pose_error(value, target) for value in poses]
    result: dict[str, object] = {
        "translation_m": [value[0] for value in errors],
        "rotation_deg": [value[1] for value in errors],
    }
    for rank in (1, 5, 16, 64):
        local = errors[: min(rank, len(errors))]
        if not local:
            continue
        best = min(local, key=lambda value: value[0] + 0.1 * value[1])
        result[f"oracle_top{rank}_translation_m"] = float(best[0])
        result[f"oracle_top{rank}_rotation_deg"] = float(best[1])
        result[f"oracle_top{rank}_within_30cm_3deg"] = bool(
            any(value[0] <= 0.30 and value[1] <= 3.0 for value in local)
        )
    return result


def _prototype_geometry_diagnostic(
    poses: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    centers = np.stack(
        [camera_center_from_pose_w2c(value) for value in poses]
    )
    target_center = camera_center_from_pose_w2c(target)
    translation = np.linalg.norm(centers - target_center[None], axis=1)
    rotation = rotation_angle_deg(
        np.asarray(poses, dtype=np.float64)[:, :3, :3],
        np.asarray(target, dtype=np.float64)[:3, :3],
    )
    count = centers.shape[0]
    initial = np.full(count, 1.0 / max(count, 1), dtype=np.float64)
    objective = lambda weight: float(
        np.sum((weight @ centers - target_center) ** 2)
    )
    solution = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weight: np.sum(weight) - 1.0},
        options={"maxiter": 200, "ftol": 1e-12},
    )
    convex_distance = (
        float(np.sqrt(max(float(solution.fun), 0.0)))
        if bool(solution.success)
        else None
    )
    result: dict[str, object] = {
        "nearest_center_translation_m": float(np.min(translation)),
        "nearest_rotation_deg": float(np.min(rotation)),
        "center_convex_hull_distance_m": convex_distance,
    }
    for threshold in (3.0, 5.0, 10.0, 20.0):
        keep = rotation <= threshold
        result[f"nearest_center_within_{threshold:g}deg_m"] = (
            float(np.min(translation[keep])) if np.any(keep) else None
        )
    return result


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {"query_count": len(rows)}
    for method in (
        "identity_prototypes",
        "full_prototypes",
        "full_modes",
        "full_local_linear_modes",
        "full_geometry_only_modes",
    ):
        translations = np.asarray(
            [row[method]["translation_m"][0] for row in rows],
            dtype=np.float64,
        )
        rotations = np.asarray(
            [row[method]["rotation_deg"][0] for row in rows],
            dtype=np.float64,
        )
        summary: dict[str, object] = {
            "top1_translation_median_m": float(np.median(translations)),
            "top1_translation_p90_m": float(np.quantile(translations, 0.9)),
            "top1_rotation_median_deg": float(np.median(rotations)),
            "top1_rotation_p90_deg": float(np.quantile(rotations, 0.9)),
            "top1_within_30cm_3deg": int(
                np.sum((translations <= 0.30) & (rotations <= 3.0))
            ),
        }
        for rank in (1, 5, 16, 64):
            key = f"oracle_top{rank}_translation_m"
            if key not in rows[0][method]:
                continue
            local_translation = np.asarray(
                [row[method][key] for row in rows], dtype=np.float64
            )
            local_rotation = np.asarray(
                [
                    row[method][f"oracle_top{rank}_rotation_deg"]
                    for row in rows
                ],
                dtype=np.float64,
            )
            summary[f"oracle_top{rank}_translation_median_m"] = float(
                np.median(local_translation)
            )
            summary[f"oracle_top{rank}_rotation_median_deg"] = float(
                np.median(local_rotation)
            )
            summary[f"oracle_top{rank}_within_30cm_3deg"] = int(
                sum(
                    bool(row[method][f"oracle_top{rank}_within_30cm_3deg"])
                    for row in rows
                )
            )
        output[method] = summary
    nearest_translation = np.asarray(
        [row["nearest_mapping_pose"]["translation_m"][0] for row in rows]
    )
    nearest_rotation = np.asarray(
        [row["nearest_mapping_pose"]["rotation_deg"][0] for row in rows]
    )
    output["nearest_mapping_pose"] = {
        "translation_median_m": float(np.median(nearest_translation)),
        "translation_p90_m": float(np.quantile(nearest_translation, 0.9)),
        "rotation_median_deg": float(np.median(nearest_rotation)),
        "within_30cm_3deg": int(
            np.sum((nearest_translation <= 0.30) & (nearest_rotation <= 3.0))
        ),
    }
    for name in ("identity_top64_geometry", "full_top64_geometry"):
        keys = (
            "nearest_center_translation_m",
            "nearest_rotation_deg",
            "center_convex_hull_distance_m",
            "nearest_center_within_3deg_m",
            "nearest_center_within_5deg_m",
            "nearest_center_within_10deg_m",
            "nearest_center_within_20deg_m",
        )
        output[name] = {}
        for key in keys:
            values = [row[name].get(key) for row in rows]
            finite = np.asarray(
                [float(value) for value in values if value is not None],
                dtype=np.float64,
            )
            output[name][f"{key}_median"] = (
                float(np.median(finite)) if finite.size else None
            )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V7 retrieval report")
    signature_path = Path(args.pose_signature_bank)
    identity_path = Path(args.retrieval_regions)
    spatial_path = Path(args.spatial_maplets)
    mapper_path = Path(args.surface_mapper_checkpoint)
    calibration_path = Path(args.probability_calibration)
    signature_bank = PoseSignatureBank.load_npz(signature_path)
    lineage = {
        "retrieval_regions_sha256": (identity_path, "retrieval regions"),
        "spatial_maplets_sha256": (spatial_path, "spatial maplets"),
        "surface_mapper_checkpoint_sha256": (mapper_path, "surface mapper"),
        "probability_calibration_sha256": (calibration_path, "calibration"),
    }
    for key, (path, label) in lineage.items():
        if signature_bank.metadata.get(key) != file_sha256(path):
            raise ValueError(f"pose-signature/{label} lineage differs")
    identity_bank = SurfaceRetrievalMapletBank.load_npz(identity_path)
    spatial_bank = SurfaceRetrievalMapletBank.load_npz(spatial_path)
    if not np.array_equal(signature_bank.maplet_ids, identity_bank.maplet_ids):
        raise ValueError("pose-signature and retrieval maplet bases differ")
    mapper, mapper_metadata = load_surface_maplet_mapper(
        mapper_path, device=str(args.device)
    )
    calibration = V6ProbabilityCalibration.load_json(calibration_path)
    contributor_paths = sorted(Path(args.query_contributor_dir).glob("*.npz"))
    if int(args.maximum_queries) > 0:
        contributor_paths = contributor_paths[: int(args.maximum_queries)]
    if not contributor_paths:
        raise ValueError("no held-out contributor caches found")
    query_trajectory_ids = []
    rows: list[dict[str, object]] = []
    query_signatures = []
    query_target_poses = []
    for contributor_path in contributor_paths:
        with np.load(contributor_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            target_pose = np.asarray(data["pose_w2c"], dtype=np.float64)
            width = int(data["camera_width"].item())
            height = int(data["camera_height"].item())
        image_id = str(metadata["image_id"])
        query_trajectory_ids.append(image_id.split("/", 1)[0])
        token_path = Path(str(metadata["token_path"]))
        with np.load(token_path, allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        retrieval = _retrieve(
            raw,
            identity_bank,
            spatial_bank,
            mapper,
            mapper_metadata,
            calibration,
            (width, height),
            str(args.aggregation),
        )
        signature = pose_signature_from_retrieval(
            retrieval, signature_bank.maplet_ids, (width, height)
        )
        query_signatures.append(signature)
        query_target_poses.append(target_pose)
        identity_scores, _identity_components = score_pose_signatures(
            signature,
            signature_bank,
            layout_weight=0.0,
            extent_weight=0.0,
        )
        full_scores, full_components = score_pose_signatures(
            signature, signature_bank
        )
        identity_order = np.argsort(-identity_scores, kind="mergesort")[:64]
        full_order = np.argsort(-full_scores, kind="mergesort")[:64]
        modes: tuple[PoseMode, ...] = pose_modes_from_signature_scores(
            signature_bank, full_scores, maximum_prototypes=64, maximum_modes=16
        )
        linear_modes: tuple[PoseMode, ...] = local_linear_pose_modes(
            signature,
            signature_bank,
            full_scores,
            maximum_prototypes=64,
            maximum_modes=16,
        )
        geometry_modes: tuple[PoseMode, ...] = geometry_only_pose_modes(
            signature_bank,
            full_scores,
            maximum_prototypes=64,
            maximum_modes=16,
        )
        target_center = camera_center_from_pose_w2c(target_pose)
        mapping_center = np.stack(
            [camera_center_from_pose_w2c(value) for value in signature_bank.poses_w2c]
        )
        nearest = int(np.argmin(np.linalg.norm(mapping_center - target_center, axis=1)))
        row = {
            "image_id": image_id,
            "retrieval_top64_count": int(retrieval.ranked_maplet_ids.size),
            "identity_prototypes": _rank_errors(
                signature_bank.poses_w2c[identity_order], target_pose
            ),
            "full_prototypes": _rank_errors(
                signature_bank.poses_w2c[full_order], target_pose
            ),
            "full_modes": _rank_errors(
                [mode.pose_w2c for mode in modes], target_pose
            ),
            "full_local_linear_modes": _rank_errors(
                [mode.pose_w2c for mode in linear_modes], target_pose
            ),
            "full_geometry_only_modes": _rank_errors(
                [mode.pose_w2c for mode in geometry_modes], target_pose
            ),
            "full_mode_sources": [mode.source for mode in modes],
            "full_local_linear_mode_sources": [
                mode.source for mode in linear_modes
            ],
            "full_geometry_only_mode_sources": [
                mode.source for mode in geometry_modes
            ],
            "identity_top64_geometry": _prototype_geometry_diagnostic(
                signature_bank.poses_w2c[identity_order], target_pose
            ),
            "full_top64_geometry": _prototype_geometry_diagnostic(
                signature_bank.poses_w2c[full_order], target_pose
            ),
            "nearest_mapping_pose": _rank_errors(
                [signature_bank.poses_w2c[nearest]], target_pose
            ),
            "score_diagnostics": {
                "top1_identity": float(full_components["identity"][full_order[0]]),
                "top1_layout": float(full_components["layout"][full_order[0]]),
                "top1_extent": float(full_components["extent"][full_order[0]]),
                "top1_overlap": float(full_components["overlap"][full_order[0]]),
            },
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "image_id": image_id,
                    "processed": len(rows),
                    "top1_translation_m": row["full_modes"]["translation_m"][0],
                    "top1_rotation_deg": row["full_modes"]["rotation_deg"][0],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    overlap = sorted(
        set(query_trajectory_ids)
        & set(signature_bank.metadata.get("mapping_trajectory_ids", []))
    )
    if overlap:
        raise ValueError(f"strict V7 query/mapping trajectory overlap: {overlap}")
    report = {
        "artifact_type": "v7_pose_aware_retrieval_report",
        "method_contract": {
            "query_map_feature_interactions": 1,
            "interaction_stage": "stage_a_maplet_retrieval_only",
            "stage_b_global_chart_correlation_used": False,
            "point_correspondence_pnp_used": False,
            "mapping_rgb_stored": False,
            "mapping_image_ids_stored": False,
            "mapping_image_paths_stored": False,
            "mapping_observation_descriptors_stored": False,
            "pose_estimation": "maplet_sufficient_statistic_pose_modes",
        },
        "configuration": {
            "retrieval_top_k": 64,
            "identity_weight": 1.0,
            "layout_weight": 0.35,
            "extent_weight": 0.10,
            "pose_cluster_translation_radius_m": 1.5,
            "pose_cluster_rotation_radius_deg": 18.0,
            "score_temperature": 0.04,
            "local_linear_ridge_fraction": 0.05,
            "local_linear_maximum_translation_extrapolation_m": 0.75,
            "local_linear_maximum_rotation_extrapolation_deg": 8.0,
        },
        "pose_signature_bank_sha256": file_sha256(signature_path),
        "query_trajectory_ids": sorted(set(query_trajectory_ids)),
        "mapping_trajectory_ids": signature_bank.metadata.get(
            "mapping_trajectory_ids", []
        ),
        "strict_query_mapping_overlap": overlap,
        "summary": _summarize_rows(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if str(args.output_query_signatures_npz):
        query_output = Path(args.output_query_signatures_npz)
        query_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            query_output,
            maplet_ids=signature_bank.maplet_ids,
            target_poses_w2c=np.stack(query_target_poses),
            identity=np.stack([value.identity for value in query_signatures]),
            layout_mean_xy=np.stack(
                [value.layout_mean_xy for value in query_signatures]
            ),
            layout_extent_xy=np.stack(
                [value.layout_extent_xy for value in query_signatures]
            ),
            layout_variance_xy=np.stack(
                [value.layout_variance_xy for value in query_signatures]
            ),
            layout_mass=np.stack(
                [value.layout_mass for value in query_signatures]
            ),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "artifact_type": "v7_query_pose_signature_diagnostic",
                        "query_trajectory_ids": sorted(
                            set(query_trajectory_ids)
                        ),
                        "contains_ground_truth_for_evaluation": True,
                        "stores_query_rgb": False,
                        "stores_query_image_paths": False,
                    },
                    sort_keys=True,
                )
            ),
        )


if __name__ == "__main__":
    main()
