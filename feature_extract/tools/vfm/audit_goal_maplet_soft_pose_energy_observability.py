"""Audit local SE(3) observability of correspondence-free RADIO/3DGS energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    score_bidirectional_soft_surface_pose_energy,
    score_hierarchical_spatial_soft_surface_pose_energy,
    score_soft_surface_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    complete_quadratic_probe_coordinates,
    fit_complete_local_se3_quadratic,
    fit_local_se3_quadratic_least_squares,
    left_retract_pose_w2c,
    minimal_quadratic_probe_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_canonical_surface_field,
    render_soft_child_surface_field,
)


def _camera(contributor: Path) -> ColmapCamera:
    with np.load(contributor, allow_pickle=False) as data:
        return ColmapCamera(
            0,
            int(data["camera_model_id"]),
            int(data["camera_width"]),
            int(data["camera_height"]),
            tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--retrieval_run", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_ids", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_step_m", type=float, default=0.5)
    parser.add_argument("--rotation_step_deg", type=float, default=5.0)
    parser.add_argument("--radio_weight", type=float, default=0.5)
    parser.add_argument(
        "--energy_semantics",
        choices=(
            "dominant_child_v1", "bidirectional_soft_child_v2",
            "hierarchical_spatial_soft_child_v3",
        ),
        default="bidirectional_soft_child_v2",
    )
    parser.add_argument("--render_top_l", type=int, default=4)
    parser.add_argument("--coordinate_supersample_factor", type=int, default=4)
    parser.add_argument(
        "--quadratic_model",
        choices=("axis_only", "minimal_full_6d", "complete_symmetric_6d"),
        default="axis_only",
    )
    parser.add_argument("--quadratic_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite pose-energy observability audit")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    run = json.loads(Path(args.retrieval_run).read_text())
    artifact_by_id = {str(row["image_id"]): Path(row["artifact"]) for row in run["rows"]}
    gt = {record.image_id: record.pose_w2c for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    axis_names = ("tx", "ty", "tz", "rx", "ry", "rz")
    rows_out = []
    for image_id in args.image_ids:
        if image_id not in artifact_by_id or image_id not in gt:
            raise KeyError(f"missing retrieval/GT for {image_id}")
        contributor = Path(args.contributors) / image_id.replace("/", "__")
        contributor = contributor.with_suffix(contributor.suffix + ".npz")
        if not contributor.exists():
            raise FileNotFoundError(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        query = np.asarray(
            mapper.project(_load_raw_final(Path(str(metadata["token_path"])), "radio_final")).measurement_context,
            dtype=np.float32,
        )
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact_by_id[image_id])
        if float(args.quadratic_scale) <= 0.0:
            raise ValueError("quadratic_scale must be positive")
        if args.quadratic_model == "minimal_full_6d":
            quadratic_coordinates = minimal_quadratic_probe_coordinates()
        elif args.quadratic_model == "complete_symmetric_6d":
            quadratic_coordinates = complete_quadratic_probe_coordinates()
        else:
            quadratic_coordinates = {"center": np.zeros((6,), dtype=np.float64)}
            eye = np.eye(6, dtype=np.float64)
            for index, name in enumerate(axis_names):
                quadratic_coordinates[name + "+"] = eye[index]
                quadratic_coordinates[name + "-"] = -eye[index]
        scaled_coordinates = {
            name: np.asarray(coordinate, dtype=np.float64) * float(args.quadratic_scale)
            for name, coordinate in quadratic_coordinates.items()
        }
        poses = [
            (
                name,
                left_retract_pose_w2c(
                    gt[image_id], coordinate,
                    translation_step_m=float(args.translation_step_m),
                    rotation_step_degrees=float(args.rotation_step_deg),
                ),
            )
            for name, coordinate in scaled_coordinates.items()
        ]
        scores = []
        camera = _camera(contributor)
        for name, pose in poses:
            if args.energy_semantics in (
                "bidirectional_soft_child_v2", "hierarchical_spatial_soft_child_v3",
            ):
                rendered = render_soft_child_surface_field(
                    physical, field, pose, camera,
                    width=query.shape[2], height=query.shape[1],
                    selected_child_rows=retrieval.scene_child_rows,
                    top_l=int(args.render_top_l), device=str(args.device),
                    coordinate_supersample_factor=int(args.coordinate_supersample_factor),
                )
                if args.energy_semantics == "hierarchical_spatial_soft_child_v3":
                    energy = score_hierarchical_spatial_soft_surface_pose_energy(
                        query, retrieval, rendered,
                        child_to_parent_ids=physical.maplet_ids[physical.child_parent_rows],
                        radio_weight=float(args.radio_weight),
                    )
                else:
                    energy = score_bidirectional_soft_surface_pose_energy(
                        query, retrieval, rendered, radio_weight=float(args.radio_weight)
                    )
            else:
                rendered = render_canonical_surface_field(
                    physical, field, pose, camera,
                    width=query.shape[2], height=query.shape[1],
                    selected_child_rows=retrieval.scene_child_rows,
                    device=str(args.device),
                )
                energy = score_soft_surface_pose_energy(
                    query, retrieval, rendered, radio_weight=float(args.radio_weight)
                )
            scores.append({"probe": name, **energy.__dict__})
        by_name = {row["probe"]: row for row in scores}
        curvature = {}
        for axis_name in ("tx", "ty", "tz"):
            curvature[axis_name] = float(
                2.0 * by_name["center"]["combined_score"]
                - by_name[axis_name + "+"]["combined_score"]
                - by_name[axis_name + "-"]["combined_score"]
            ) / (float(args.translation_step_m) * float(args.quadratic_scale)) ** 2
        for axis_name in ("rx", "ry", "rz"):
            key = "r" + axis_name[1]
            curvature[axis_name] = float(
                2.0 * by_name["center"]["combined_score"]
                - by_name[key + "+"]["combined_score"]
                - by_name[key + "-"]["combined_score"]
            ) / (float(args.rotation_step_deg) * float(args.quadratic_scale)) ** 2
        ranked = sorted(scores, key=lambda row: (-float(row["combined_score"]), str(row["probe"])))
        quadratic = None
        eigenvector_probes = []
        if args.quadratic_model != "axis_only":
            normalized_scores = {name: float(row["combined_score"]) for name, row in by_name.items()}
            if args.quadratic_model == "complete_symmetric_6d":
                # Coordinates were uniformly scaled physically; the normalized
                # finite-difference design remains the canonical +/-1 design.
                fitted = fit_complete_local_se3_quadratic(normalized_scores)
                regression = fit_local_se3_quadratic_least_squares(
                    quadratic_coordinates, normalized_scores,
                )
            else:
                fitted = fit_local_se3_quadratic_least_squares(
                    quadratic_coordinates,
                    normalized_scores,
                )
                regression = fitted
            eigenvalues, eigenvectors = np.linalg.eigh(fitted.loss_hessian)
            minimum_direction = eigenvectors[:, 0]
            if float(eigenvalues[0]) < 0.0:
                for multiplier in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                    coordinate = minimum_direction * float(multiplier) * float(args.quadratic_scale)
                    pose = left_retract_pose_w2c(
                        gt[image_id], coordinate,
                        translation_step_m=float(args.translation_step_m),
                        rotation_step_degrees=float(args.rotation_step_deg),
                    )
                    if args.energy_semantics in (
                        "bidirectional_soft_child_v2", "hierarchical_spatial_soft_child_v3",
                    ):
                        rendered = render_soft_child_surface_field(
                            physical, field, pose, camera,
                            width=query.shape[2], height=query.shape[1],
                            selected_child_rows=retrieval.scene_child_rows,
                            top_l=int(args.render_top_l), device=str(args.device),
                            coordinate_supersample_factor=int(args.coordinate_supersample_factor),
                        )
                        if args.energy_semantics == "hierarchical_spatial_soft_child_v3":
                            energy = score_hierarchical_spatial_soft_surface_pose_energy(
                                query, retrieval, rendered,
                                child_to_parent_ids=physical.maplet_ids[physical.child_parent_rows],
                                radio_weight=float(args.radio_weight),
                            )
                        else:
                            energy = score_bidirectional_soft_surface_pose_energy(
                                query, retrieval, rendered, radio_weight=float(args.radio_weight)
                            )
                    else:
                        rendered = render_canonical_surface_field(
                            physical, field, pose, camera,
                            width=query.shape[2], height=query.shape[1],
                            selected_child_rows=retrieval.scene_child_rows,
                            device=str(args.device),
                        )
                        energy = score_soft_surface_pose_energy(
                            query, retrieval, rendered, radio_weight=float(args.radio_weight)
                        )
                    eigenvector_probes.append({
                        "multiplier": float(multiplier),
                        "normalized_coordinate": coordinate.tolist(),
                        **energy.__dict__,
                    })
            fitted = fitted
            quadratic = {
                "design": (
                    "complete_73_probe_symmetric_single_left_se3_retraction_v2"
                    if args.quadratic_model == "complete_symmetric_6d"
                    else "minimal_28_probe_single_left_se3_retraction_v2"
                ),
                "loss_gradient": fitted.loss_gradient.tolist(),
                "loss_hessian": fitted.loss_hessian.tolist(),
                "hessian_eigenvalues": fitted.hessian_eigenvalues.tolist(),
                "minimum_eigenvector": minimum_direction.tolist(),
                "fit_root_mean_square_error": regression.fit_root_mean_square_error,
                "fit_maximum_absolute_error": regression.fit_maximum_absolute_error,
                "hessian_condition_number": (
                    fitted.hessian_condition_number
                    if np.isfinite(fitted.hessian_condition_number) else None
                ),
                "predicted_bias_normalized": (
                    fitted.predicted_bias_normalized.tolist()
                    if np.all(np.isfinite(fitted.predicted_bias_normalized)) else None
                ),
                "positive_definite": bool(fitted.positive_definite),
                "direct_minimum_eigenvector_probes": eigenvector_probes,
            }
        rows_out.append({
            "image_id": image_id,
            "center_probe_rank": 1 + next(i for i, row in enumerate(ranked) if row["probe"] == "center"),
            "positive_negative_score_curvature_axis_count": int(
                sum(value > 0.0 for value in curvature.values())
            ),
            "negative_score_axis_curvature": curvature,
            "local_quadratic_6d": quadratic,
            "scores": scores,
        })
    report = {
        "artifact_type": "goal_maplet_soft_pose_energy_observability_audit_v3",
        "query_count": len(rows_out),
        "translation_step_m": float(args.translation_step_m),
        "rotation_step_deg": float(args.rotation_step_deg),
        "radio_weight": float(args.radio_weight),
        "energy_semantics": str(args.energy_semantics),
        "render_top_l": int(args.render_top_l),
        "coordinate_supersample_factor": int(args.coordinate_supersample_factor),
        "render_coordinate_semantics": "ideal_pinhole_highres_hits_inverse_simple_radial_to_raw_grid_then_token_average_v1",
        "quadratic_model": str(args.quadratic_model),
        "quadratic_scale": float(args.quadratic_scale),
        "pose_retraction_semantics": "left_se3_exponential_T_xi_equals_Exp_xi_times_T0_v1",
        "curvature_sign_semantics": "reported values approximate Hessian(-combined_score); positive means a 1D local score maximum",
        "mean_positive_negative_score_curvature_axis_count": float(np.mean([
            row["positive_negative_score_curvature_axis_count"] for row in rows_out
        ])),
        "center_top1_fraction": float(np.mean([row["center_probe_rank"] == 1 for row in rows_out])),
        "claims": {
            "uses_alike": False,
            "uses_pnp": False,
            "uses_hard_correspondences": False,
            "map_side_child_distribution_is_soft": args.energy_semantics != "dominant_child_v1",
            "radio_feature_is_coupled_to_matching_child": args.energy_semantics != "dominant_child_v1",
            "uses_fixed_parent_child_spatial_hierarchy": args.energy_semantics == "hierarchical_spatial_soft_child_v3",
            "uses_gt_only_to_place_observability_probes": True,
            "is_global_localization_result": False,
        },
        "rows": rows_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
