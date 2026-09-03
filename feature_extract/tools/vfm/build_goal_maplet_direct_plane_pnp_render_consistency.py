"""Score frozen direct-plane poses by metric and scale-free MoGe3/map agreement.

The input poses must have been sealed before any query pose/GT member was
opened.  The query MoGe3 geometry is used only after pose estimation and a
single robust depth scale is fitted independently for every query.  No query
depth value enters PnP and no pose label is read by this program.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.visibility import signed_surface_visibility
from feature_extract.vfm.localization_v6.primitive_contributors import (
    render_primitive_contributors,
)
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _camera_inventory,
)


def _load_frozen_poses(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        all_arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    required = ("names", "pose_w2c", "usable")
    if any(name not in all_arrays for name in required):
        raise ValueError("frozen direct-plane pose arrays are incomplete")
    candidate_key = (
        "candidate_correspondence_count"
        if "candidate_correspondence_count" in all_arrays
        else "selected_candidate_correspondence_count"
    )
    inlier_key = (
        "pnp_inlier_count" if "pnp_inlier_count" in all_arrays
        else "selected_pnp_inlier_count"
    )
    if candidate_key not in all_arrays or inlier_key not in all_arrays:
        raise ValueError("frozen direct-plane pose support arrays are incomplete")
    arrays = {
        "names": all_arrays["names"],
        "pose_w2c": all_arrays["pose_w2c"],
        "usable": all_arrays["usable"],
        "candidate_correspondence_count": all_arrays[candidate_key],
        "pnp_inlier_count": all_arrays[inlier_key],
    }
    count = len(arrays["names"])
    artifact_type = metadata.get("artifact_type")
    depth_used = metadata.get("query_depth_or_scale_used_by_pose_solver")
    if (
        artifact_type not in (
            "goal_maplet_frozen_direct_plane_pnp_pose_inventory_v1",
            "goal_maplet_direct_plane_pnp_map_density_inlier_selected_v1",
            "goal_maplet_direct_plane_pnp_top5_top10_pose_consensus_v1",
            "goal_maplet_direct_plane_pnp_pose_medoid_fallback_v1",
            "goal_maplet_direct_plane_pnp_damped_union_closure_v1",
            "goal_maplet_direct_plane_pnp_reliability_damped_refinement_v1",
            "goal_maplet_direct_plane_pnp_reliability_tiered_damped_refinement_v2",
            "goal_maplet_direct_plane_pnp_plane_and_depth_reliability_tiered_damped_refinement_v3",
            "goal_maplet_direct_plane_pnp_spatial_plane_reliability_tiered_damped_refinement_v4",
            "goal_maplet_direct_plane_pnp_reliability_cascade_v5",
            "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v6",
            "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v7",
            "goal_maplet_moge3_plane_scale_surface_refinement_v1",
            "goal_maplet_canonical_plane_uv_view_geometry_refined_pose_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read", False) is not False
        or (
            depth_used is not (
                True if artifact_type == "goal_maplet_moge3_plane_scale_surface_refinement_v1"
                else False
            )
        )
        or arrays_sha256(all_arrays) != metadata.get("arrays_sha256")
        or arrays["pose_w2c"].shape != (count, 4, 4)
        or arrays["usable"].shape != (count,)
        or len(set(arrays["names"].astype(str).tolist())) != count
    ):
        raise ValueError("frozen direct-plane PnP pose inventory differs")
    usable = np.asarray(arrays["usable"], bool)
    if not np.isfinite(arrays["pose_w2c"][usable]).all():
        raise ValueError("usable frozen PnP pose is nonfinite")
    return arrays, metadata


def _metric_depth_normal_agreement(
    rendered_depth: np.ndarray,
    rendered_normal_camera: np.ndarray,
    query_depth: np.ndarray,
    query_normal_camera: np.ndarray,
    query_valid: np.ndarray,
) -> dict[str, object]:
    """Compute metric, scale-only, and affine relative-depth diagnostics.

    MoGe3's raw depth is retained as a metric cue, while a fitted scale and a
    bounded affine log-depth model expose what remains if its global scale or
    depth compression drifts.  The fitted variables never enter the pose that
    produced the render; this function is a frozen-candidate verifier.
    """

    rendered = np.asarray(rendered_depth, np.float64)
    query = np.asarray(query_depth, np.float64)
    rendered_normal = np.asarray(rendered_normal_camera, np.float64)
    query_normal = np.asarray(query_normal_camera, np.float64)
    valid_query = (
        np.asarray(query_valid, bool) & np.isfinite(query) & (query > 0.0)
        & np.isfinite(query_normal).all(axis=2)
    )
    common = valid_query & np.isfinite(rendered) & (rendered > 0.0)
    query_count = int(np.sum(valid_query))
    common_count = int(np.sum(common))
    if query_count == 0 or common_count < 16:
        return {
            "query_valid_pixel_count": query_count,
            "common_valid_pixel_count": common_count,
            "render_coverage_of_query_valid": 0.0 if query_count == 0 else common_count / query_count,
            "fitted_query_depth_scale": None,
            "metric_query_depth_scale_log_bias": None,
            "metric_absrel_median": None,
            "metric_absrel_p90": None,
            "absolute_log_depth_median": None,
            "absolute_log_depth_p90": None,
            "affine_log_depth_slope": None,
            "affine_log_depth_intercept": None,
            "affine_log_depth_median": None,
            "affine_log_depth_p90": None,
            "relative_log_depth_correlation": None,
            "depth_ratio_within_10pct": 0.0,
            "depth_ratio_within_20pct": 0.0,
            "depth_ratio_within_50pct": 0.0,
            "absolute_normal_cosine_median": None,
            "absolute_normal_cosine_p10": None,
            "normal_within_20deg": 0.0,
            "normal_within_30deg": 0.0,
        }
    scale = float(np.median(rendered[common] / query[common]))
    rendered_common = rendered[common]
    query_common = query[common]
    log_query = np.log(query_common)
    log_rendered = np.log(rendered_common)
    log_error = np.abs(log_rendered - log_query - np.log(scale))
    metric_absrel = np.abs(rendered_common - query_common) / np.maximum(
        rendered_common, 1e-12,
    )
    centered_query = log_query - float(np.mean(log_query))
    centered_rendered = log_rendered - float(np.mean(log_rendered))
    denominator = float(np.sum(centered_query * centered_query))
    slope = 1.0 if denominator <= 1e-12 else float(
        np.sum(centered_query * centered_rendered) / denominator
    )
    slope = float(np.clip(slope, 0.5, 1.5))
    intercept = float(np.median(log_rendered - slope * log_query))
    signed_affine = log_rendered - (slope * log_query + intercept)
    median_signed = float(np.median(signed_affine))
    mad = float(np.median(np.abs(signed_affine - median_signed)))
    trim = np.abs(signed_affine - median_signed) <= max(3.0 * 1.4826 * mad, 1e-6)
    if int(np.sum(trim)) >= 16 and float(np.var(log_query[trim])) > 1e-12:
        design = np.c_[log_query[trim], np.ones(int(np.sum(trim)), np.float64)]
        fitted = np.linalg.lstsq(design, log_rendered[trim], rcond=None)[0]
        slope = float(np.clip(fitted[0], 0.5, 1.5))
        intercept = float(np.median(log_rendered[trim] - slope * log_query[trim]))
        signed_affine = log_rendered - (slope * log_query + intercept)
    affine_error = np.abs(signed_affine)
    correlation_denominator = float(
        np.linalg.norm(centered_query) * np.linalg.norm(centered_rendered)
    )
    correlation = (
        0.0 if correlation_denominator <= 1e-12
        else float(np.sum(centered_query * centered_rendered) / correlation_denominator)
    )
    rn = rendered_normal[common]
    qn = query_normal[common]
    cosine = np.abs(np.sum(rn * qn, axis=1)) / np.maximum(
        np.linalg.norm(rn, axis=1) * np.linalg.norm(qn, axis=1), 1e-12,
    )
    cosine = np.clip(cosine, 0.0, 1.0)
    return {
        "query_valid_pixel_count": query_count,
        "common_valid_pixel_count": common_count,
        "render_coverage_of_query_valid": float(common_count / query_count),
        "fitted_query_depth_scale": scale,
        "metric_query_depth_scale_log_bias": float(abs(np.log(scale))),
        "metric_absrel_median": float(np.median(metric_absrel)),
        "metric_absrel_p90": float(np.quantile(metric_absrel, 0.9)),
        "absolute_log_depth_median": float(np.median(log_error)),
        "absolute_log_depth_p90": float(np.quantile(log_error, 0.9)),
        "affine_log_depth_slope": slope,
        "affine_log_depth_intercept": intercept,
        "affine_log_depth_median": float(np.median(affine_error)),
        "affine_log_depth_p90": float(np.quantile(affine_error, 0.9)),
        "relative_log_depth_correlation": correlation,
        "depth_ratio_within_10pct": float(np.mean(log_error <= np.log(1.1))),
        "depth_ratio_within_20pct": float(np.mean(log_error <= np.log(1.2))),
        "depth_ratio_within_50pct": float(np.mean(log_error <= np.log(1.5))),
        "absolute_normal_cosine_median": float(np.median(cosine)),
        "absolute_normal_cosine_p10": float(np.quantile(cosine, 0.1)),
        "normal_within_20deg": float(np.mean(cosine >= np.cos(np.deg2rad(20.0)))),
        "normal_within_30deg": float(np.mean(cosine >= np.cos(np.deg2rad(30.0)))),
    }


def _elements(physical: GoalMapletPhysicalMap, rows: np.ndarray) -> SurfaceElementMap:
    selected = np.asarray(rows, np.int64)
    return SurfaceElementMap(
        element_ids=physical.primitive_ids[selected],
        parent_gaussian_indices=physical.primitive_ids[selected],
        centers=physical.primitive_centers[selected],
        tangent1=physical.primitive_tangent1[selected],
        tangent2=physical.primitive_tangent2[selected],
        normals=physical.primitive_normals[selected],
        scale1=physical.primitive_scale1[selected],
        scale2=physical.primitive_scale2[selected],
        opacity=physical.primitive_opacity[selected],
        area=np.pi * physical.primitive_scale1[selected] * physical.primitive_scale2[selected],
        adjacency=tuple(),
        metadata={"representation": "frozen_full_clean_2dgs_pose_verification"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--physical_map", type=Path, required=True)
    parser.add_argument("--query_camera_inventory", type=Path, required=True)
    parser.add_argument("--moge3_query", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite render-consistency artifact")

    poses, pose_meta = _load_frozen_poses(args.frozen_pose_inventory)
    camera_rows, camera_meta = _camera_inventory(args.query_camera_inventory)
    if pose_meta.get("query_camera_only_inventory_file_sha256") != file_sha256(
        args.query_camera_inventory
    ):
        raise ValueError("frozen poses and query camera inventory differ")
    manifests = []
    manifest_rows: dict[str, tuple[Path, dict[str, object]]] = {}
    for root in args.moge3_query:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rows = {str(row["image_id"]).replace("/", "__") + ".npz": row for row in manifest["rows"]}
        if (
            manifest.get("artifact_type") != "goal_maplet_moge_query_geometry_v2_manifest"
            or manifest.get("output_height") != 144
            or manifest.get("output_width") != 256
            or len(rows) != len(manifest["rows"])
            or set(rows) & set(manifest_rows)
        ):
            raise ValueError("MoGe3 query manifest differs")
        manifest_rows.update({name: (root, row) for name, row in rows.items()})
        manifests.append((manifest_path, manifest))

    physical = GoalMapletPhysicalMap.load_npz(args.physical_map)
    max_id = int(np.max(physical.primitive_ids))
    row_by_id = np.full(max_id + 1, -1, np.int32)
    row_by_id[physical.primitive_ids] = np.arange(len(physical.primitive_ids), dtype=np.int32)
    rows = []
    start = time.perf_counter()
    for index, name in enumerate(poses["names"].astype(str).tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(poses["usable"][index])}
        if not row["usable"]:
            rows.append(row)
            continue
        if name not in camera_rows or name not in manifest_rows:
            raise ValueError("render-consistency input lacks a query")
        moge_root, moge_row = manifest_rows[name]
        moge_path = moge_root / name
        if file_sha256(moge_path) != moge_row.get("file_sha256"):
            raise ValueError("MoGe3 query bytes differ")
        with np.load(moge_path, allow_pickle=False) as data:
            query_depth = np.asarray(data["depth_camera"], np.float64)
            query_normal = np.asarray(data["normal_camera"], np.float64)
            query_valid = np.asarray(data["valid"], bool)
            moge_meta = json.loads(str(data["metadata_json"].item()))
        if moge_meta.get("content_sha256") != moge_row.get("content_sha256"):
            raise ValueError("MoGe3 embedded metadata differs")

        pose = np.asarray(poses["pose_w2c"][index], np.float64)
        front, _ = signed_surface_visibility(
            physical.primitive_centers,
            physical.primitive_normals,
            physical.primitive_sidedness,
            pose,
            minimum_incidence=0.05,
        )
        scene_rows = np.flatnonzero(front)
        elements = _elements(physical, scene_rows)
        model_id, width, height, params = camera_rows[name]
        camera = ColmapCamera(
            camera_id=0,
            model_id=int(model_id),
            width=int(width),
            height=int(height),
            params=tuple(float(value) for value in np.asarray(params).tolist()),
        )
        view = GaussianVFMFeatureView(
            image_id=name,
            feature_map=np.zeros((1, 144, 256), np.float32),
            pose_w2c=pose,
            camera=camera,
        )
        rendered = render_primitive_contributors(
            elements, view, width=256, height=144, top_k=1, device=args.device,
        )
        dominant = np.asarray(rendered.dominant_ids, np.int64)
        valid_id = (dominant >= 0) & (dominant <= max_id)
        primitive_rows = np.full(dominant.shape, -1, np.int32)
        primitive_rows[valid_id] = row_by_id[dominant[valid_id]]
        valid_id &= primitive_rows >= 0
        normal_camera = np.zeros((*dominant.shape, 3), np.float64)
        normal_camera[valid_id] = (
            physical.primitive_normals[primitive_rows[valid_id]] @ pose[:3, :3].T
        )
        score = _metric_depth_normal_agreement(
            rendered.primitive_depth,
            normal_camera,
            query_depth,
            query_normal,
            query_valid,
        )
        row.update(score)
        row["front_facing_primitive_count"] = int(len(scene_rows))
        rows.append(row)
        print(json.dumps({"completed": index + 1, "total": len(poses["names"]), "name": name}))

    report = {
        "artifact_type": "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1",
        "query_count": int(len(rows)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_role": "post_pose_metric_scale_relative_depth_and_normal_verification",
        "depth_scale_fit": "per_query_median_rendered_depth_over_moge3_depth",
        "affine_log_depth_fit": "per_query_MAD_trimmed_bounded_slope_[0.5,1.5]",
        "raw_metric_depth_retained": True,
        "query_depth_changes_frozen_pose": False,
        "frozen_pose_inventory_file_sha256": file_sha256(args.frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": pose_meta.get("content_sha256"),
        "physical_map_file_sha256": file_sha256(args.physical_map),
        "physical_map_content_sha256": physical.content_sha256,
        "query_camera_inventory_file_sha256": file_sha256(args.query_camera_inventory),
        "query_camera_inventory_content_sha256": camera_meta.get("content_sha256"),
        "moge3_manifest_file_sha256_in_order": [file_sha256(path) for path, _ in manifests],
        "moge3_manifest_content_sha256_in_order": [
            manifest.get("content_sha256") for _, manifest in manifests
        ],
        "renderer": "clean_2dgs_disks_alpha_transmittance_dominant_depth",
        "minimum_front_incidence": 0.05,
        "elapsed_seconds": float(time.perf_counter() - start),
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
