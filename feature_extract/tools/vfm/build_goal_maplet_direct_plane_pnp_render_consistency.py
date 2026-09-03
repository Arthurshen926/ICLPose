"""Score frozen direct-plane poses by metric and scale-free MoGe3/map agreement.

The input poses must have been sealed before any query pose/GT member was
opened.  The query MoGe3 geometry is used only after pose estimation and a
single robust depth scale is fitted independently for every query.  No query
depth value enters PnP and no pose label is read by this program.
"""

from __future__ import annotations

import argparse
import json
import math
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
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.visibility import signed_surface_visibility
from feature_extract.vfm.localization_v6.primitive_contributors import (
    render_primitive_contributors,
)
from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    _composite_sorted_packed_hits,
    _intrinsic_matrix,
    _surface_element_quaternions_and_scales,
)

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _camera_inventory,
)
from feature_extract.tools.vfm.build_goal_maplet_sparse_first_render_plan import (
    _load_sparse_first_plan,
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
            "goal_maplet_moge3_plane_scale_surface_refinement_v2",
            "goal_maplet_canonical_plane_uv_view_geometry_refined_pose_v1",
            "goal_maplet_uncertainty_weighted_plane_pose_refinement_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read", False) is not False
        or (
            depth_used is not (
                True if artifact_type in {
                    "goal_maplet_moge3_plane_scale_surface_refinement_v1",
                    "goal_maplet_moge3_plane_scale_surface_refinement_v2",
                }
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


def _validate_query_camera_lineage(
    pose_metadata: dict[str, object],
    query_camera_inventory: Path,
    frozen_correspondence: Path | None,
) -> dict[str, object] | None:
    """Replay the pose -> correspondence -> camera chain when it is indirect.

    Early pose inventories copied the query-camera hash into their own
    metadata.  The current uncertainty-refined endpoints instead bind the
    complete frozen-correspondence artifact, whose metadata owns that hash.
    Accepting the latter without replaying the intermediate bytes would turn a
    useful lineage compression into a camera-substitution hole.
    """

    camera_sha = file_sha256(query_camera_inventory)
    if pose_metadata.get("query_camera_only_inventory_file_sha256") == camera_sha:
        if frozen_correspondence is not None:
            raise ValueError("direct camera lineage must not supply an unrelated correspondence")
        return None
    if frozen_correspondence is None:
        raise ValueError(
            "frozen poses use indirect camera lineage; --frozen_correspondence is required"
        )
    expected_file = pose_metadata.get("frozen_correspondence_file_sha256")
    expected_content = pose_metadata.get("frozen_correspondence_content_sha256")
    if file_sha256(frozen_correspondence) != expected_file:
        raise ValueError("frozen pose does not bind the supplied correspondence bytes")
    with np.load(frozen_correspondence, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("frozen correspondence metadata is missing")
        correspondence_metadata = json.loads(str(data["metadata_json"].item()))
        correspondence_arrays = {
            name: np.asarray(data[name]) for name in data.files if name != "metadata_json"
        }
    if (
        arrays_sha256(correspondence_arrays)
        != correspondence_metadata.get("arrays_sha256")
        or correspondence_metadata.get("content_sha256") != expected_content
        or correspondence_metadata.get("query_camera_only_inventory_file_sha256")
        != camera_sha
        or correspondence_metadata.get("pose_or_ground_truth_opened") is not False
    ):
        raise ValueError("frozen correspondence camera lineage differs")
    return correspondence_metadata


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


def _resident_render_primitive_batches(
    physical: GoalMapletPhysicalMap,
    poses_w2c: np.ndarray,
    cameras: list[ColmapCamera],
    usable: np.ndarray,
    *,
    width: int,
    height: int,
    batch_size: int,
    device: str,
    minimum_incidence: float,
    gpu_composite: bool,
) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    """Render exact dominant primitive/depth buffers with resident geometry.

    This is algebraically the same gsplat projection, tile rasterization and
    deterministic alpha compositor as ``render_primitive_contributors``.  The
    only optimization is keeping static map tensors resident and evaluating a
    small batch of independently frozen camera hypotheses together.
    """

    try:
        import torch
        from gsplat.cuda._wrapper import (
            fully_fused_projection_2dgs,
            isect_offset_encode,
            isect_tiles,
            rasterize_to_indices_in_range_2dgs,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch and gsplat CUDA wrappers are required") from exc
    torch_device = torch.device(str(device))
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("resident primitive rendering requires CUDA")
    pose_array = np.asarray(poses_w2c, np.float64)
    valid = np.asarray(usable, bool)
    if (
        pose_array.ndim != 3
        or pose_array.shape[1:] != (4, 4)
        or len(cameras) != len(pose_array)
        or valid.shape != (len(pose_array),)
        or int(batch_size) <= 0
    ):
        raise ValueError("resident primitive render inventory differs")
    all_rows = np.arange(len(physical.primitive_ids), dtype=np.int64)
    elements = _elements(physical, all_rows)
    quaternions, scales = _surface_element_quaternions_and_scales(elements)
    means = torch.as_tensor(
        physical.primitive_centers, dtype=torch.float32, device=torch_device,
    ).contiguous()
    quats = torch.as_tensor(
        quaternions, dtype=torch.float32, device=torch_device,
    ).contiguous()
    scale = torch.as_tensor(
        scales, dtype=torch.float32, device=torch_device,
    ).contiguous()
    opacity = torch.as_tensor(
        physical.primitive_opacity, dtype=torch.float32, device=torch_device,
    ).reshape(-1).clamp(0.0, 1.0).contiguous()
    stable_ids = np.asarray(physical.primitive_ids, np.int64)
    output: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    usable_indices = np.flatnonzero(valid)
    tile_size = 16
    tile_width = math.ceil(int(width) / tile_size)
    tile_height = math.ceil(int(height) / tile_size)
    for begin in range(0, len(usable_indices), int(batch_size)):
        indices = usable_indices[begin:begin + int(batch_size)]
        batch = len(indices)
        pose = torch.as_tensor(
            pose_array[indices].astype(np.float32),
            dtype=torch.float32, device=torch_device,
        ).contiguous()
        intrinsic = torch.as_tensor(
            np.stack([
                _intrinsic_matrix(cameras[int(index)], int(width), int(height))
                for index in indices.tolist()
            ]).astype(np.float32),
            dtype=torch.float32, device=torch_device,
        ).contiguous()
        radii, means2d, depths, transforms, _normals = fully_fused_projection_2dgs(
            means, quats, scale, pose, intrinsic, int(width), int(height), packed=False,
        )
        front_numpy = np.stack([
            signed_surface_visibility(
                physical.primitive_centers,
                physical.primitive_normals,
                physical.primitive_sidedness,
                pose_array[int(index)],
                minimum_incidence=float(minimum_incidence),
            )[0]
            for index in indices.tolist()
        ])
        front = torch.as_tensor(
            front_numpy, dtype=torch.bool, device=torch_device,
        ).contiguous()
        radii = torch.where(front, radii, torch.zeros_like(radii)).contiguous()
        per_camera_opacity = (
            opacity[None, :].expand(batch, -1) * front.to(torch.float32)
        ).contiguous()
        _tiles, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_cameras=batch,
        )
        offsets = isect_offset_encode(
            isect_ids, batch, tile_width, tile_height,
        )
        transmittance = torch.ones(
            (batch, int(height), int(width)),
            dtype=torch.float32, device=torch_device,
        )
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
            0, 1_000_000_000, transmittance, means2d, transforms,
            per_camera_opacity, int(width), int(height), tile_size,
            offsets, flatten_ids,
        )
        if gs_ids.numel() == 0:
            for local, index in enumerate(indices.tolist()):
                output[int(index)] = (
                    np.full((int(height), int(width)), -1, np.int64),
                    np.zeros((int(height), int(width)), np.float32),
                    int(np.sum(front_numpy[local])),
                )
            continue
        pixel_x = (pixel_ids % int(width)).to(torch.float32) + 0.5
        pixel_y = (pixel_ids // int(width)).to(torch.float32) + 0.5
        pixel_coordinates = torch.stack([pixel_x, pixel_y], dim=-1)
        deltas = pixel_coordinates - means2d[camera_ids, gs_ids]
        transform = transforms[camera_ids, gs_ids]
        h_u = -transform[..., 0, :3] + transform[..., 2, :3] * pixel_x[..., None]
        h_v = -transform[..., 1, :3] + transform[..., 2, :3] * pixel_y[..., None]
        temporary = torch.cross(h_u, h_v, dim=-1)
        denominator = temporary[..., 2]
        denominator = torch.where(
            torch.abs(denominator) < 1e-12,
            torch.where(denominator >= 0.0, 1e-12, -1e-12),
            denominator,
        )
        local_u = temporary[..., 0] / denominator
        local_v = temporary[..., 1] / denominator
        sigma_3d = local_u * local_u + local_v * local_v
        sigma_2d = 2.0 * torch.sum(deltas * deltas, dim=1)
        sigma = 0.5 * torch.minimum(sigma_3d, sigma_2d)
        alpha = torch.clamp(
            per_camera_opacity[camera_ids, gs_ids] * torch.exp(-sigma), max=0.999,
        )
        global_pixels = camera_ids.to(torch.int64) * (int(width) * int(height)) + pixel_ids
        if bool(gpu_composite):
            from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
                _composite_packed_hits_torch,
            )
            hit_pixel_t, hit_row_t, hit_weight_t = _composite_packed_hits_torch(
                global_pixels.to(torch.int64).contiguous(),
                depths[camera_ids, gs_ids].to(torch.float32).contiguous(),
                gs_ids.to(torch.int64).contiguous(),
                alpha.to(torch.float32).contiguous(),
            )
            hit_pixel = hit_pixel_t.detach().cpu().numpy()
            hit_row = hit_row_t.detach().cpu().numpy()
            hit_weight = hit_weight_t.detach().cpu().numpy()
        else:
            packed = torch.stack([
                global_pixels.to(torch.float64),
                depths[camera_ids, gs_ids].to(torch.float64),
                gs_ids.to(torch.float64),
                alpha.to(torch.float64),
            ], dim=1).detach().cpu().numpy()
            order = np.lexsort((packed[:, 2], packed[:, 1], packed[:, 0]))
            hit_pixel, hit_row, hit_weight = _composite_sorted_packed_hits(packed[order])
        pixels_per_camera = int(width) * int(height)
        depth_numpy = depths.detach().cpu().numpy()
        for local, index in enumerate(indices.tolist()):
            lower = local * pixels_per_camera
            mask = (hit_pixel >= lower) & (hit_pixel < lower + pixels_per_camera)
            local_pixel = hit_pixel[mask] - lower
            local_row = hit_row[mask]
            local_weight = hit_weight[mask]
            dominant = np.full((pixels_per_camera,), -1, np.int64)
            depth_image = np.zeros((pixels_per_camera,), np.float32)
            if local_pixel.size:
                ranking = np.lexsort((stable_ids[local_row], -local_weight, local_pixel))
                sorted_pixel = local_pixel[ranking]
                keep = np.r_[True, sorted_pixel[1:] != sorted_pixel[:-1]]
                chosen = ranking[keep]
                chosen_pixel = local_pixel[chosen]
                chosen_row = local_row[chosen]
                dominant[chosen_pixel] = stable_ids[chosen_row]
                depth_image[chosen_pixel] = depth_numpy[local, chosen_row]
            output[int(index)] = (
                dominant.reshape(int(height), int(width)),
                depth_image.reshape(int(height), int(width)),
                int(np.sum(front_numpy[local])),
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--physical_map", type=Path, required=True)
    parser.add_argument("--query_camera_inventory", type=Path, required=True)
    parser.add_argument(
        "--query_plane_regions", type=Path,
        help=(
            "Optional pose-free MoGe3 plane-region cache. When supplied, the report also "
            "scores the render on observed finite planar pixels only."
        ),
    )
    parser.add_argument(
        "--frozen_correspondence",
        type=Path,
        help="Required when the final pose binds its camera through a correspondence artifact.",
    )
    parser.add_argument(
        "--sparse_first_render_plan", type=Path,
        help=(
            "Optional pose/label-free plan that renders only queries where the frozen "
            "sparse geometry test leaves the dense score able to change the decision."
        ),
    )
    parser.add_argument("--moge3_query", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resident_batch_size", type=int, default=4,
        help="Keep full map geometry resident and render this many frozen poses per CUDA batch.",
    )
    parser.add_argument(
        "--resident_gpu_composite", dest="resident_gpu_composite",
        action="store_true", help="Use the deterministic GPU compositor (default).",
    )
    parser.add_argument(
        "--no_resident_gpu_composite", dest="resident_gpu_composite",
        action="store_false",
        help="Disable the GPU compositor; required with batch size one for the legacy path.",
    )
    parser.set_defaults(resident_gpu_composite=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite render-consistency artifact")
    if bool(args.resident_gpu_composite) and int(args.resident_batch_size) <= 1:
        raise ValueError("resident GPU compositing requires resident_batch_size > 1")
    poses, pose_meta = _load_frozen_poses(args.frozen_pose_inventory)
    sparse_plan = None
    sparse_plan_meta = None
    render_required = np.ones(len(poses["names"]), bool)
    if args.sparse_first_render_plan is not None:
        sparse_plan, sparse_plan_meta = _load_sparse_first_plan(args.sparse_first_render_plan)
        pose_sha = file_sha256(args.frozen_pose_inventory)
        pose_content = pose_meta.get("content_sha256")
        if (
            not np.array_equal(
                poses["names"].astype(str), sparse_plan["names"].astype(str),
            )
            or not (
                (
                    sparse_plan_meta.get("primary_pose_file_sha256") == pose_sha
                    and sparse_plan_meta.get("primary_pose_content_sha256") == pose_content
                )
                or (
                    sparse_plan_meta.get("alternate_pose_file_sha256") == pose_sha
                    and sparse_plan_meta.get("alternate_pose_content_sha256") == pose_content
                )
            )
        ):
            raise ValueError("sparse-first render plan does not bind this pose inventory")
        render_required = np.asarray(sparse_plan["dense_render_required"], bool)
    camera_rows, camera_meta = _camera_inventory(args.query_camera_inventory)
    correspondence_meta = _validate_query_camera_lineage(
        pose_meta, args.query_camera_inventory, args.frozen_correspondence,
    )
    query_plane_manifest_path = None
    query_plane_manifest = None
    query_plane_rows: dict[str, tuple[int, int, str]] = {}
    if args.query_plane_regions is not None:
        query_plane_manifest_path = args.query_plane_regions / "manifest.json"
        query_plane_manifest = json.loads(query_plane_manifest_path.read_text())
        plane_manifest_rows = query_plane_manifest.get("rows")
        if (
            query_plane_manifest.get("artifact_type") not in {
                "goal_maplet_query_plane_region_cache_run_v1",
                "goal_maplet_query_plane_region_cache_run_v2",
                "goal_maplet_query_plane_region_cache_run_v3",
            }
            or query_plane_manifest.get("uses_pose_or_ground_truth") is not False
            or not isinstance(plane_manifest_rows, list)
        ):
            raise ValueError("query-plane manifest differs")
        for plane_manifest_row in plane_manifest_rows:
            if not isinstance(plane_manifest_row, list) or len(plane_manifest_row) != 4:
                raise ValueError("query-plane manifest row differs")
            plane_name = str(plane_manifest_row[0])
            if plane_name in query_plane_rows:
                raise ValueError("query-plane manifest contains duplicate names")
            query_plane_rows[plane_name] = (
                int(plane_manifest_row[1]),
                int(plane_manifest_row[2]),
                str(plane_manifest_row[3]),
            )
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
    resident_rendered = None
    if int(args.resident_batch_size) > 1:
        cameras = []
        for name in poses["names"].astype(str).tolist():
            if name not in camera_rows:
                raise ValueError("resident render input lacks a query camera")
            model_id, camera_width, camera_height, camera_params = camera_rows[name]
            cameras.append(ColmapCamera(
                camera_id=0,
                model_id=int(model_id),
                width=int(camera_width),
                height=int(camera_height),
                params=tuple(float(value) for value in np.asarray(camera_params).tolist()),
            ))
        resident_rendered = _resident_render_primitive_batches(
            physical,
            poses["pose_w2c"],
            cameras,
            np.asarray(poses["usable"], bool) & render_required,
            width=256,
            height=144,
            batch_size=int(args.resident_batch_size),
            device=args.device,
            minimum_incidence=0.05,
            gpu_composite=bool(args.resident_gpu_composite),
        )
    for index, name in enumerate(poses["names"].astype(str).tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(poses["usable"][index])}
        if sparse_plan is not None:
            row["dense_score_evaluated"] = bool(row["usable"] and render_required[index])
        if not row["usable"]:
            rows.append(row)
            continue
        if sparse_plan is not None and not render_required[index]:
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
        query_plane = None
        if args.query_plane_regions is not None:
            if name not in query_plane_rows:
                raise ValueError("query-plane cache lacks a rendered query")
            query_plane, query_plane_meta = QueryPlaneRegions.load_npz(
                args.query_plane_regions / name,
            )
            plane_count, plane_pixels, plane_content = query_plane_rows[name]
            if (
                query_plane_meta.get("content_sha256") != plane_content
                or query_plane_meta.get("uses_pose_or_ground_truth") is not False
                or query_plane_meta.get("source_file_sha256") != file_sha256(moge_path)
                or query_plane_meta.get("source_name") != name
                or len(query_plane.normals_camera) != plane_count
                or int(np.sum(query_plane.pixel_counts)) != plane_pixels
                or query_plane.labels.shape != query_valid.shape
            ):
                raise ValueError("query-plane cache does not replay MoGe3 geometry")

        pose = np.asarray(poses["pose_w2c"][index], np.float64)
        model_id, width, height, params = camera_rows[name]
        camera = ColmapCamera(
            camera_id=0,
            model_id=int(model_id),
            width=int(width),
            height=int(height),
            params=tuple(float(value) for value in np.asarray(params).tolist()),
        )
        if resident_rendered is None:
            front, _ = signed_surface_visibility(
                physical.primitive_centers,
                physical.primitive_normals,
                physical.primitive_sidedness,
                pose,
                minimum_incidence=0.05,
            )
            scene_rows = np.flatnonzero(front)
            elements = _elements(physical, scene_rows)
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
            rendered_depth = rendered.primitive_depth
            front_facing_count = int(len(scene_rows))
        else:
            dominant, rendered_depth, front_facing_count = resident_rendered[index]
        valid_id = (dominant >= 0) & (dominant <= max_id)
        primitive_rows = np.full(dominant.shape, -1, np.int32)
        primitive_rows[valid_id] = row_by_id[dominant[valid_id]]
        valid_id &= primitive_rows >= 0
        normal_camera = np.zeros((*dominant.shape, 3), np.float64)
        normal_camera[valid_id] = (
            physical.primitive_normals[primitive_rows[valid_id]] @ pose[:3, :3].T
        )
        score = _metric_depth_normal_agreement(
            rendered_depth,
            normal_camera,
            query_depth,
            query_normal,
            query_valid,
        )
        row.update(score)
        if query_plane is not None:
            planar_score = _metric_depth_normal_agreement(
                rendered_depth,
                normal_camera,
                query_depth,
                query_normal,
                query_valid & (query_plane.labels >= 0),
            )
            row.update({f"planar_{key}": value for key, value in planar_score.items()})
        row["front_facing_primitive_count"] = front_facing_count
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
        "query_plane_manifest_file_sha256": (
            None if query_plane_manifest_path is None
            else file_sha256(query_plane_manifest_path)
        ),
        "query_plane_manifest_artifact_type": (
            None if query_plane_manifest is None
            else query_plane_manifest.get("artifact_type")
        ),
        "planar_pixel_score_role": (
            None if query_plane_manifest is None
            else "observed_finite_MoGe3_plane_pixels_only_missing_render_is_failure"
        ),
        "frozen_correspondence_file_sha256": (
            None if args.frozen_correspondence is None else file_sha256(args.frozen_correspondence)
        ),
        "frozen_correspondence_content_sha256": (
            None if correspondence_meta is None else correspondence_meta.get("content_sha256")
        ),
        "moge3_manifest_file_sha256_in_order": [file_sha256(path) for path, _ in manifests],
        "moge3_manifest_content_sha256_in_order": [
            manifest.get("content_sha256") for _, manifest in manifests
        ],
        "renderer": (
            "clean_2dgs_disks_alpha_transmittance_dominant_depth"
            if resident_rendered is None
            else "resident_batched_clean_2dgs_disks_alpha_transmittance_dominant_depth"
        ),
        "resident_batch_size": int(args.resident_batch_size),
        "resident_gpu_composite": bool(args.resident_gpu_composite),
        "minimum_front_incidence": 0.05,
        "elapsed_seconds": float(time.perf_counter() - start),
        "production_eligible": False,
        "rows": rows,
    }
    if sparse_plan is not None:
        report.update({
            "sparse_first_render_plan_file_sha256": file_sha256(
                args.sparse_first_render_plan,
            ),
            "sparse_first_render_plan_content_sha256": sparse_plan_meta.get(
                "content_sha256",
            ),
            "dense_render_query_count": int(np.sum(
                np.asarray(poses["usable"], bool) & render_required,
            )),
        })
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
