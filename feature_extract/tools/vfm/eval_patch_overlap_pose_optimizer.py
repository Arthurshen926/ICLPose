"""Evaluate patch-overlap SE(3) pose optimization with GT patch positives."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.patch_overlap_pose import (
    PatchOverlapConfig,
    build_vfm_2dgs_patch_overlap_supports,
    oracle_patch_overlap_matches,
    patch_overlap_energy_numpy,
    patch_overlap_pose_optimize,
)
from feature_extract.vfm.patch_to_3d_matching import (
    build_patch_positive_sets,
    filter_landmarks_by_projected_visibility,
    oracle_patch_positive_matches,
    patch_uncertainty_pnp_threshold,
)
from feature_extract.vfm.query_to_3d_matching import estimate_pose_pnp_ransac, pnp_pose_error
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap


def _finite_median(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.median(present))


def _finite_mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not present else float(np.mean(present))


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else float(np.mean([1.0 if value else 0.0 for value in values]))


def _metric_summary(rows: Sequence[dict[str, object]], prefix: str) -> dict[str, object]:
    solve_key = f"{prefix}_success"
    t_key = f"{prefix}_translation_error_m"
    r_key = f"{prefix}_rotation_error_deg"
    solved = [bool(row.get(solve_key, False)) for row in rows]
    t_values = [row.get(t_key) for row in rows if bool(row.get(solve_key, False))]
    r_values = [row.get(r_key) for row in rows if bool(row.get(solve_key, False))]
    return {
        "solve_rate": _rate(solved),
        "median_translation_error_m": _finite_median(t_values),
        "median_rotation_error_deg": _finite_median(r_values),
        "success_10cm_5deg": _rate(
            [
                bool(row.get(solve_key, False))
                and row.get(t_key) is not None
                and float(row[t_key]) <= 0.10
                and row.get(r_key) is not None
                and float(row[r_key]) <= 5.0
                for row in rows
            ]
        ),
        "success_25cm_10deg": _rate(
            [
                bool(row.get(solve_key, False))
                and row.get(t_key) is not None
                and float(row[t_key]) <= 0.25
                and row.get(r_key) is not None
                and float(row[r_key]) <= 10.0
                for row in rows
            ]
        ),
        "success_50cm_10deg": _rate(
            [
                bool(row.get(solve_key, False))
                and row.get(t_key) is not None
                and float(row[t_key]) <= 0.50
                and row.get(r_key) is not None
                and float(row[r_key]) <= 10.0
                for row in rows
            ]
        ),
        "mean_final_energy": _finite_mean([row.get(f"{prefix}_final_energy") for row in rows]),
        "mean_gt_energy": _finite_mean([row.get(f"{prefix}_gt_energy") for row in rows]),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pose_error_fields(prefix: str, pose_w2c: np.ndarray | None, gt_pose_w2c: np.ndarray) -> dict[str, object]:
    if pose_w2c is None:
        return {
            f"{prefix}_success": False,
            f"{prefix}_translation_error_m": None,
            f"{prefix}_rotation_error_deg": None,
        }
    error = pnp_pose_error(pose_w2c, gt_pose_w2c)
    return {
        f"{prefix}_success": bool(np.isfinite(error.translation_m) and np.isfinite(error.rotation_deg)),
        f"{prefix}_translation_error_m": float(error.translation_m),
        f"{prefix}_rotation_error_deg": float(error.rotation_deg),
    }


def _run_patch_overlap_mode(
    prefix: str,
    matches,
    patch_boxes,
    supports,
    camera,
    init_pose_w2c: np.ndarray | None,
    gt_pose_w2c: np.ndarray,
    config: PatchOverlapConfig,
) -> dict[str, object]:
    if init_pose_w2c is None or len(matches) < 4:
        return {
            f"{prefix}_success": False,
            f"{prefix}_translation_error_m": None,
            f"{prefix}_rotation_error_deg": None,
            f"{prefix}_initial_energy": None,
            f"{prefix}_final_energy": None,
            f"{prefix}_gt_energy": None,
            f"{prefix}_mean_containment": None,
        }
    result = patch_overlap_pose_optimize(
        matches,
        patch_boxes,
        supports,
        camera,
        init_pose_w2c,
        config=config,
        gt_pose_w2c=gt_pose_w2c,
    )
    fields = _pose_error_fields(prefix, result.pose_w2c if result.success else None, gt_pose_w2c)
    fields.update(
        {
            f"{prefix}_initial_energy": float(result.initial_energy),
            f"{prefix}_final_energy": float(result.final_energy),
            f"{prefix}_gt_energy": None if result.gt_energy is None else float(result.gt_energy),
            f"{prefix}_mean_containment": float(result.mean_containment),
        }
    )
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--semidense_anchor_npz", required=True)
    parser.add_argument("--anchor_map_npz", required=True)
    parser.add_argument("--surface_elements_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--max_queries", type=int, default=20)
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--oracle_association_mode", default="support", choices=("support", "centroid"))
    parser.add_argument("--min_support_containment", type=float, default=0.5)
    parser.add_argument("--oracle_max_positives_per_token", type=int, default=1)
    parser.add_argument("--support_samples_per_anchor", type=int, default=32)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=1.25)
    parser.add_argument("--pnp_iterations", type=int, default=2000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--pnp_method", default="EPNP")
    parser.add_argument("--pnp_refine_method", default="LM")
    parser.add_argument("--overlap_iterations", type=int, default=80)
    parser.add_argument("--overlap_lr", type=float, default=0.03)
    parser.add_argument("--overlap_sigma", type=float, default=0.5)
    parser.add_argument("--overlap_max_anchors_per_patch", type=int, default=10)
    parser.add_argument("--moment_center_weight", type=float, default=0.1)
    parser.add_argument("--moment_init_regularization_weight", type=float, default=1e-3)
    parser.add_argument("--moment_optimize_rotation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    semidense_map = SemiDenseAnchorMap.load_npz(Path(args.semidense_anchor_npz))
    landmark_index = semidense_map.to_landmark_index()
    anchor_map = Vfm2DgsAnchorMap.load_npz(Path(args.anchor_map_npz))
    surface_elements = SurfaceElementMap.load_npz(Path(args.surface_elements_npz))
    supports = build_vfm_2dgs_patch_overlap_supports(
        anchor_map,
        surface_elements,
        max_samples_per_anchor=int(args.support_samples_per_anchor),
    )
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}

    rows: list[dict[str, object]] = []
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    overlap_config = PatchOverlapConfig(
        iterations=int(args.overlap_iterations),
        lr=float(args.overlap_lr),
        sigma=float(args.overlap_sigma),
        max_anchors_per_patch=int(args.overlap_max_anchors_per_patch),
        depth_weight=1.0,
        center_weight=0.0,
        init_regularization_weight=0.0,
        optimize_rotation=True,
    )
    moment_config = PatchOverlapConfig(
        iterations=int(args.overlap_iterations),
        lr=float(args.overlap_lr),
        sigma=float(args.overlap_sigma),
        max_anchors_per_patch=int(args.overlap_max_anchors_per_patch),
        depth_weight=1.0,
        center_weight=float(args.moment_center_weight),
        init_regularization_weight=float(args.moment_init_regularization_weight),
        optimize_rotation=bool(args.moment_optimize_rotation),
    )

    for record in records:
        gt_pose = gt_by_query.get(record.image_id)
        if gt_pose is None:
            continue
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
        stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
        stride = float(max(stride_x, stride_y))
        submap = filter_landmarks_by_projected_visibility(landmark_index, gt_pose.pose_w2c, camera)
        if args.oracle_association_mode == "support":
            matches, patch_boxes = oracle_patch_overlap_matches(
                submap,
                supports,
                gt_pose.pose_w2c,
                camera,
                token_width,
                token_height,
                patch_scale=float(args.patch_scale),
                min_support_containment=float(args.min_support_containment),
                max_per_token=int(args.oracle_max_positives_per_token),
            )
        else:
            positives = build_patch_positive_sets(
                submap,
                gt_pose.pose_w2c,
                camera,
                token_width,
                token_height,
                patch_scale=float(args.patch_scale),
            )
            matches = oracle_patch_positive_matches(
                submap,
                positives,
                max_per_token=int(args.oracle_max_positives_per_token),
            )
            patch_boxes = {
                int(token_index): positive.patch_box
                for token_index, positive in positives.by_token.items()
                if positive.track_ids
            }
        pnp_threshold = patch_uncertainty_pnp_threshold(stride, float(args.pnp_threshold_stride_multiplier))
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=pnp_threshold,
            confidence=float(args.pnp_confidence),
            iterations=int(args.pnp_iterations),
            min_inliers=int(args.pnp_min_inliers),
            refine_lm=str(args.pnp_refine_method).upper() != "NONE",
            pnp_method=args.pnp_method,
            refine_method=args.pnp_refine_method,
        )
        row: dict[str, object] = {
            "query_id": record.image_id,
            "visible_landmarks": int(len(submap)),
            "match_count": int(len(matches)),
            "pnp_inlier_count": int(pnp.inlier_count),
            "stride_px": float(stride),
        }
        row.update(_pose_error_fields("pnp", pnp.pose_w2c if pnp.success else None, gt_pose.pose_w2c))
        if pnp.success and pnp.pose_w2c is not None:
            row["overlap_energy_gt"] = patch_overlap_energy_numpy(matches, patch_boxes, supports, camera, gt_pose.pose_w2c, overlap_config)
            row["overlap_energy_pnp"] = patch_overlap_energy_numpy(matches, patch_boxes, supports, camera, pnp.pose_w2c, overlap_config)
        else:
            row["overlap_energy_gt"] = None
            row["overlap_energy_pnp"] = None
        row.update(
            _run_patch_overlap_mode(
                "overlap",
                matches,
                patch_boxes,
                supports,
                camera,
                pnp.pose_w2c if pnp.success else None,
                gt_pose.pose_w2c,
                overlap_config,
            )
        )
        row.update(
            _run_patch_overlap_mode(
                "moment",
                matches,
                patch_boxes,
                supports,
                camera,
                pnp.pose_w2c if pnp.success else None,
                gt_pose.pose_w2c,
                moment_config,
            )
        )
        rows.append(row)

    output_dir = Path(args.output_dir)
    rows_path = output_dir / "patch_overlap_pose_rows.csv"
    summary_path = output_dir / "patch_overlap_pose_summary.json"
    _write_csv(rows_path, rows)
    summary = {
        "stage": "patch_overlap_se3_optimizer",
        "elapsed_sec": float(time.perf_counter() - started),
        "query_count": int(len(rows)),
        "camera": {
            "source": camera_source,
            "width": int(camera.width),
            "height": int(camera.height),
            "model_id": int(camera.model_id),
            "params": [float(value) for value in camera.params],
        },
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "semidense_anchor_npz": str(args.semidense_anchor_npz),
            "anchor_map_npz": str(args.anchor_map_npz),
            "surface_elements_npz": str(args.surface_elements_npz),
        },
        "config": {
            "patch_scale": float(args.patch_scale),
            "oracle_association_mode": str(args.oracle_association_mode),
            "min_support_containment": float(args.min_support_containment),
            "oracle_max_positives_per_token": int(args.oracle_max_positives_per_token),
            "support_samples_per_anchor": int(args.support_samples_per_anchor),
            "pnp_threshold_stride_multiplier": float(args.pnp_threshold_stride_multiplier),
            "overlap": overlap_config.__dict__,
            "moment": moment_config.__dict__,
        },
        "pnp": _metric_summary(rows, "pnp"),
        "overlap": _metric_summary(rows, "overlap"),
        "moment": _metric_summary(rows, "moment"),
        "energy": {
            "mean_overlap_gt": _finite_mean([row.get("overlap_energy_gt") for row in rows]),
            "mean_overlap_pnp": _finite_mean([row.get("overlap_energy_pnp") for row in rows]),
            "gt_lower_than_pnp_rate": _rate(
                [
                    row.get("overlap_energy_gt") is not None
                    and row.get("overlap_energy_pnp") is not None
                    and float(row["overlap_energy_gt"]) < float(row["overlap_energy_pnp"])
                    for row in rows
                ]
            ),
        },
        "outputs": {"rows": str(rows_path), "summary": str(summary_path)},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
