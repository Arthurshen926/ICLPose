"""Measure the local convergence basin of anchor-free 2DGS feature alignment.

Ground-truth is used only to construct controlled perturbations and oracle
visible maplets.  This isolates the fine alignment field from retrieval/coarse
initialization and is therefore a gate, not a deployable localization result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
    _load_raw_final,
)
from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
    pose_w2c_from_center_rotation,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    ContinuousSurfaceAlignmentConfig,
    align_surface_feature_field,
    build_detector_heatmap,
    select_visible_maplets,
)
from feature_extract.vfm.localization.alike_detector_only import AlikeDetectorOnly
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--query_camera_manifest", required=True)
    parser.add_argument("--query_image_root", default="")
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--surface_feature_field", required=True)
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=16)
    parser.add_argument("--perturbations_per_radius", type=int, default=2)
    parser.add_argument("--translation_radii_m", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--rotation_radii_deg", default="1,2,4,6")
    parser.add_argument("--maximum_maplets", type=int, default=12)
    parser.add_argument("--maximum_samples", type=int, default=2048)
    parser.add_argument("--detector_top_k", type=int, default=1024)
    parser.add_argument("--detector_candidate_top_k", type=int, default=4096)
    parser.add_argument("--detector_heatmap_width", type=int, default=256)
    parser.add_argument("--detector_heatmap_height", type=int, default=144)
    parser.add_argument("--lbfgs_iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis.tolist()
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return (
        np.eye(3)
        + np.sin(float(angle_rad)) * skew
        + (1.0 - np.cos(float(angle_rad))) * (skew @ skew)
    )


def _perturb_pose(
    pose_w2c: np.ndarray,
    translation_m: float,
    rotation_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    center = -pose[:3, :3].T @ pose[:3, 3]
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    axis = rng.normal(size=3)
    perturbed_center = center + float(translation_m) * direction
    perturbed_rotation = _axis_angle_rotation(
        axis, np.deg2rad(float(rotation_deg))
    ) @ pose[:3, :3]
    return pose_w2c_from_center_rotation(perturbed_center, perturbed_rotation)


def _pose_errors(estimate_w2c: np.ndarray, target_w2c: np.ndarray) -> tuple[float, float]:
    estimate = np.asarray(estimate_w2c, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_w2c, dtype=np.float64).reshape(4, 4)
    estimate_center = -estimate[:3, :3].T @ estimate[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation = float(np.linalg.norm(estimate_center - target_center))
    relative = estimate[:3, :3] @ target[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return translation, float(np.rad2deg(np.arccos(cosine)))


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(np.mean(array)),
    }


def _compact_mapper_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: metadata[key]
        for key in (
            "supervision",
            "vfm_layer",
            "uses_radio_intermediate",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "best_epoch",
            "best_validation",
        )
        if key in metadata
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite basin report")
    translation_radii = [
        float(value) for value in str(args.translation_radii_m).split(",")
    ]
    rotation_radii = [
        float(value) for value in str(args.rotation_radii_deg).split(",")
    ]
    if len(translation_radii) != len(rotation_radii):
        raise ValueError("translation and rotation basin lists must align")

    field = SurfaceFeatureField.load_npz(Path(args.surface_feature_field))
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper_checkpoint), device=str(args.device)
    )
    metric_mapper = (
        load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device=str(args.device)
        )
        if str(args.metric_mapper_checkpoint)
        else None
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    pose_by_image = {
        record.image_id: record
        for record in parse_cambridge_pose_file(Path(args.query_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.query_camera_manifest)
    )
    records = [
        record
        for record in manifest.records
        if record.image_id in pose_by_image and record.image_id in camera_by_image
    ][: int(args.max_queries)]
    config = ContinuousSurfaceAlignmentConfig(
        maximum_maplets=int(args.maximum_maplets),
        maximum_samples=int(args.maximum_samples),
        lbfgs_iterations=int(args.lbfgs_iterations),
    )
    rng = np.random.default_rng(int(args.seed))
    detector = None
    if str(args.query_image_root):
        detector = AlikeDetectorOnly(
            device=str(args.device),
            matcha_repo=Path("/root/matcha"),
            model_name="alike-t",
        )
    rows: list[dict[str, object]] = []
    for record in records:
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        query_feature = mapper.project(raw).measurement_context
        if metric_mapper is not None:
            query_feature = metric_mapper.project_map(query_feature)
        target = pose_by_image[record.image_id].pose_w2c
        camera = camera_by_image[record.image_id]
        detector_heatmap = None
        if detector is not None:
            detections = detector.detect(
                Path(args.query_image_root) / record.image_id,
                image_width=int(camera.width),
                image_height=int(camera.height),
                top_k=int(args.detector_top_k),
                candidate_top_k=int(args.detector_candidate_top_k),
                nms_radius_px=2.0,
                grid_rows=8,
                grid_cols=8,
            )
            # The detector-only forward returns only xy and scalar scores.
            detector_heatmap = build_detector_heatmap(
                detections.xy,
                detections.scores,
                image_width=int(camera.width),
                image_height=int(camera.height),
                output_width=int(args.detector_heatmap_width),
                output_height=int(args.detector_heatmap_height),
                sigma_px=4.0,
            )
        visible_maplets = select_visible_maplets(
            field,
            target,
            camera,
            maximum_maplets=int(args.maximum_maplets),
        )
        for radius, rotation in zip(translation_radii, rotation_radii):
            for repeat in range(int(args.perturbations_per_radius)):
                initial = _perturb_pose(target, radius, rotation, rng)
                initial_t, initial_r = _pose_errors(initial, target)
                result = align_surface_feature_field(
                    field,
                    query_feature,
                    initial,
                    camera,
                    visible_maplets,
                    config=config,
                    device=str(args.device),
                    detector_heatmap=detector_heatmap,
                )
                final_t, final_r = _pose_errors(result.pose_w2c, target)
                rows.append(
                    {
                        "image_id": record.image_id,
                        "radius_m": float(radius),
                        "rotation_radius_deg": float(rotation),
                        "repeat": int(repeat),
                        "initial_translation_m": initial_t,
                        "initial_rotation_deg": initial_r,
                        "final_translation_m": final_t,
                        "final_rotation_deg": final_r,
                        "initial_score": result.initial_score,
                        "final_score": result.final_score,
                        "sample_count": result.sample_count,
                        "converged": result.converged,
                        "selected_maplet_count": int(visible_maplets.size),
                        "trace": result.diagnostics["optimization_trace"],
                    }
                )
                print(
                    f"{record.image_id} {radius:.2f}m: "
                    f"{initial_t:.3f}->{final_t:.3f}m "
                    f"{initial_r:.2f}->{final_r:.2f}deg"
                )

    by_radius: dict[str, object] = {}
    for radius in translation_radii:
        selected = [row for row in rows if row["radius_m"] == float(radius)]
        translations = [float(row["final_translation_m"]) for row in selected]
        rotations = [float(row["final_rotation_deg"]) for row in selected]
        by_radius[f"{radius:.2f}"] = {
            "count": len(selected),
            "translation_m": _quantiles(translations),
            "rotation_deg": _quantiles(rotations),
            "improved_fraction": float(
                np.mean(
                    [
                        float(row["final_translation_m"])
                        < float(row["initial_translation_m"])
                        for row in selected
                    ]
                )
            ),
            "within_4cm_1deg": float(
                np.mean(
                    [
                        float(row["final_translation_m"]) <= 0.04
                        and float(row["final_rotation_deg"]) <= 1.0
                        for row in selected
                    ]
                )
            ),
        }
    report = {
        "stage": "anchor_free_2dgs_surface_alignment_basin",
        "query_count": len(records),
        "trial_count": len(rows),
        "by_initial_translation_radius_m": by_radius,
        "gate": {
            "criterion": "<=20cm: >=80% to 4cm/1deg and <=30cm: >=60%",
            "pass": bool(
                by_radius.get("0.20", {}).get("within_4cm_1deg", 0.0) >= 0.80
                and by_radius.get("0.30", {}).get("within_4cm_1deg", 0.0) >= 0.60
            ),
        },
        "production_contract": {
            "map_representation": "one_radio_final_distribution_per_clean_2dgs_surfel",
            "fine_pose": "continuous_se3_disconnected_maplet_feature_render_alignment",
            "computes_alike_descriptor_map": False,
            "uses_alike_descriptors": False,
            "uses_alike_detector_weights": bool(detector is not None),
            "uses_surface_metric_mapper": bool(metric_mapper is not None),
            "uses_stable_anchor_identity": False,
            "uses_point_correspondence_pnp": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
        "mapper_metadata": _compact_mapper_metadata(mapper_metadata),
        "camera_audit": camera_audit,
        "config": {
            "maximum_maplets": int(args.maximum_maplets),
            "maximum_samples": int(args.maximum_samples),
            "lbfgs_iterations": int(args.lbfgs_iterations),
            "translation_radii_m": translation_radii,
            "rotation_radii_deg": rotation_radii,
            "perturbations_per_radius": int(args.perturbations_per_radius),
            "detector_heatmap": (
                None
                if detector is None
                else [
                    int(args.detector_heatmap_height),
                    int(args.detector_heatmap_width),
                ]
            ),
        },
        "trials": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
