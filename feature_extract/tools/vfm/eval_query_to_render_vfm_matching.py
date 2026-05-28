"""Evaluate query-token to rendered Gaussian VFM map matching with PnP-RANSAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianVFMRenderConfig,
    _project_gaussians_to_image,
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
)
from feature_extract.vfm.query_to_3d_matching import (
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    reprojection_error_stats,
    reprojection_precision,
)
from feature_extract.vfm.query_to_render_matching import (
    QueryToRenderMatchingConfig,
    match_query_tokens_to_rendered_map,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_default_camera(text: str) -> ColmapCamera:
    values = [float(item) for item in text.split(",")]
    if len(values) < 7:
        raise ValueError("--default_camera must be 'model_id,width,height,param0,param1,...'")
    return ColmapCamera(
        camera_id=-1,
        model_id=int(values[0]),
        width=int(values[1]),
        height=int(values[2]),
        params=tuple(float(item) for item in values[3:]),
    )


def _load_default_camera(camera_model_dir: str, fallback: ColmapCamera) -> ColmapCamera:
    if not camera_model_dir:
        return fallback
    cameras = read_colmap_cameras_binary(Path(camera_model_dir) / "cameras.bin")
    if not cameras:
        return fallback
    ordered = sorted(cameras.values(), key=lambda camera: camera.camera_id)
    return ordered[len(ordered) // 2]


def _infer_camera_model_dir(query_pose_file: str, explicit: str) -> str:
    if explicit:
        return explicit
    if not query_pose_file:
        return ""
    scene_dir = Path(query_pose_file).resolve(strict=False).parent
    candidate = scene_dir / "sparse" / "0"
    if (candidate / "cameras.bin").exists():
        return str(candidate)
    return ""


def _load_camera_with_source(camera_model_dir: str, fallback: ColmapCamera) -> tuple[ColmapCamera, str]:
    if camera_model_dir and (Path(camera_model_dir) / "cameras.bin").exists():
        return _load_default_camera(camera_model_dir, fallback), str(Path(camera_model_dir) / "cameras.bin")
    return fallback, "fallback_default_camera"


def _load_query_feature(path: Path, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        feature_map = np.asarray(data[layer_name], dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query token feature map must have shape (C, H, W)")
    return feature_map


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_candidate_poses(candidate_bank: str, candidate_top_n: int) -> dict[str, list[dict[str, object]]]:
    if candidate_top_n <= 0:
        raise ValueError("candidate_top_n must be positive")
    by_query: dict[str, list[tuple[float, int, dict[str, object]]]] = {}
    order = 0
    for line in Path(candidate_bank).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") == "header":
            continue
        if item.get("record_type", "candidate") != "candidate":
            continue
        query_id = item.get("query_id")
        pose = item.get("pose")
        if query_id is None or pose is None:
            continue
        metadata = dict(item.get("metadata") or {})
        rank = metadata.get("retrieval_rank", metadata.get("reference_rank", order + 1))
        try:
            rank_value = float(rank)
        except (TypeError, ValueError):
            rank_value = float(order + 1)
        row = {
            "candidate_id": str(item.get("candidate_id", f"{query_id}:{order}")),
            "query_id": str(query_id),
            "reference_image": None if item.get("reference_image") is None else str(item.get("reference_image")),
            "rank": int(rank_value) if rank_value.is_integer() else rank_value,
            "pose_w2c": np.asarray(pose, dtype=np.float64).reshape(4, 4),
            "metadata": metadata,
        }
        by_query.setdefault(str(query_id), []).append((rank_value, order, row))
        order += 1
    return {
        query_id: [row for _rank, _order, row in sorted(rows)[:candidate_top_n]]
        for query_id, rows in by_query.items()
    }


def _dense_render_contributor_diagnostics(
    field: GaussianVFMField,
    matches,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    render_config: GaussianVFMRenderConfig,
) -> dict[int, dict[str, float | int | None]]:
    if not matches or len(field) == 0:
        return {}
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for dense contributor diagnostics") from exc
    uv, depths = _project_gaussians_to_image(
        field.xyz,
        pose_w2c,
        camera,
        int(render_config.width),
        int(render_config.height),
    )
    valid = (
        (depths > 1e-6)
        & (uv[:, 0] >= -float(render_config.radius_px))
        & (uv[:, 0] < float(render_config.width) + float(render_config.radius_px))
        & (uv[:, 1] >= -float(render_config.radius_px))
        & (uv[:, 1] < float(render_config.height) + float(render_config.radius_px))
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return {}
    tree = cKDTree(uv[valid_indices])
    radius = float(render_config.radius_px)
    radius_sq = max(radius * radius, 1e-6)
    diagnostics: dict[int, dict[str, float | int | None]] = {}
    for match in matches:
        pixel_index = int(match.track_id)
        if pixel_index < 0:
            continue
        px = float(pixel_index % int(render_config.width))
        py = float(pixel_index // int(render_config.width))
        local = tree.query_ball_point([px, py], r=radius)
        if not local:
            diagnostics[pixel_index] = {
                "contributor_count": 0,
                "alpha_entropy": None,
                "top1_alpha_contribution": None,
                "depth_variance_along_ray": None,
                "depth_range_along_ray": None,
                "depth_to_rendered_delta": None,
            }
            continue
        candidate_indices = valid_indices[np.asarray(local, dtype=np.int64)]
        delta = uv[candidate_indices] - np.asarray([[px, py]], dtype=np.float64)
        dist_sq = np.sum(delta * delta, axis=1)
        spatial = np.exp(-0.5 * dist_sq / max(radius_sq * 0.25, 1e-6))
        weights = np.asarray(field.opacity[candidate_indices], dtype=np.float64) * spatial
        positive = weights > 1e-12
        weights = weights[positive]
        candidate_indices = candidate_indices[positive]
        if weights.size == 0:
            diagnostics[pixel_index] = {
                "contributor_count": 0,
                "alpha_entropy": None,
                "top1_alpha_contribution": None,
                "depth_variance_along_ray": None,
                "depth_range_along_ray": None,
                "depth_to_rendered_delta": None,
            }
            continue
        candidate_depths = depths[candidate_indices].astype(np.float64)
        probs = weights / max(float(np.sum(weights)), 1e-12)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        normalized_entropy = 0.0 if probs.size <= 1 else float(entropy / np.log(float(probs.size)))
        depth_mean = float(np.sum(probs * candidate_depths))
        depth_var = float(np.sum(probs * np.square(candidate_depths - depth_mean)))
        diagnostics[pixel_index] = {
            "contributor_count": int(probs.size),
            "alpha_entropy": normalized_entropy,
            "top1_alpha_contribution": float(np.max(probs)),
            "depth_variance_along_ray": depth_var,
            "depth_range_along_ray": float(np.max(candidate_depths) - np.min(candidate_depths)),
            "depth_to_rendered_delta": None
            if match.render_depth is None
            else float(abs(float(match.render_depth) - depth_mean)),
        }
    return diagnostics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate query VFM token to rendered Gaussian VFM map matching")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--query_pose_file", default="")
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--candidate_top_n", type=int, default=5)
    parser.add_argument("--render_width", type=int, default=160)
    parser.add_argument("--render_height", type=int, default=90)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--query_token_step", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--precision_reprojection_threshold_px", type=float, default=16.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_matches_jsonl", default="")
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    field = GaussianVFMField.load_npz(Path(args.field))
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    candidate_poses = _load_candidate_poses(args.candidate_bank, args.candidate_top_n)
    gt_by_query = {}
    if args.query_pose_file:
        gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    match_config = QueryToRenderMatchingConfig(
        top_k=args.top_k,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity=args.min_similarity,
        mutual=bool(args.mutual),
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
    )
    render_config = GaussianVFMRenderConfig(
        width=int(args.render_width),
        height=int(args.render_height),
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )

    rows = []
    match_rows = []
    records = list(manifest.records)
    if args.max_queries > 0:
        records = records[: args.max_queries]
    for record in records:
        query_id = record.image_id
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        _channels, token_height, token_width = query_feature.shape
        token_scale_x = 0.0 if token_width <= 1 else float(camera.width - 1) / float(token_width - 1)
        token_scale_y = 0.0 if token_height <= 1 else float(camera.height - 1) / float(token_height - 1)
        candidates = candidate_poses.get(query_id, [])
        gt_pose = gt_by_query.get(query_id)
        best_row = None
        for candidate in candidates:
            pose_w2c = np.asarray(candidate["pose_w2c"], dtype=np.float64).reshape(4, 4)
            if args.renderer == "gsplat":
                rendered = render_gaussian_vfm_feature_map_gsplat(
                    field,
                    pose_w2c=pose_w2c,
                    camera=camera,
                    config=render_config,
                    device=args.device,
                    channel_chunk=args.channel_chunk,
                )
            else:
                rendered = render_gaussian_vfm_feature_map(
                    field,
                    pose_w2c=pose_w2c,
                    camera=camera,
                    config=render_config,
                )
            matches = match_query_tokens_to_rendered_map(
                query_feature,
                rendered.feature_map,
                rendered.xyz_map,
                rendered.visibility_mask,
                match_config,
                image_width=int(camera.width),
                image_height=int(camera.height),
                alpha_map=rendered.weight_sum,
                depth_map=rendered.depth,
            )
            contributor_stats = _dense_render_contributor_diagnostics(
                field,
                matches,
                pose_w2c=pose_w2c,
                camera=camera,
                render_config=render_config,
            )
            pnp = estimate_pose_pnp_ransac(
                matches,
                camera,
                reprojection_error_px=args.pnp_reprojection_error_px,
                iterations=args.pnp_iterations,
            )
            pose_error = pnp_pose_error(pnp.pose_w2c, gt_pose.pose_w2c) if gt_pose is not None else None
            precision = None
            false_match_rate = None
            geometry_stats = {}
            if gt_pose is not None and matches:
                precision, false_match_rate = reprojection_precision(
                    matches,
                    gt_pose.pose_w2c,
                    camera,
                    threshold_px=args.precision_reprojection_threshold_px,
                )
                geometry_stats = reprojection_error_stats(
                    matches,
                    gt_pose.pose_w2c,
                    camera,
                    thresholds_px=(5.0, 10.0, 16.0, 32.0),
                    pnp_inlier_mask=pnp.inlier_mask,
                )
                gt_errors = match_reprojection_errors(matches, gt_pose.pose_w2c, camera)
            else:
                gt_errors = np.zeros((len(matches),), dtype=np.float64)
            pnp_residual_stats = pnp_reprojection_residual_stats(
                matches,
                pnp.pose_w2c,
                camera,
                inlier_mask=pnp.inlier_mask,
            )
            all_spatial_stats = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height))
            inlier_spatial_stats = match_spatial_distribution_stats(
                matches,
                int(camera.width),
                int(camera.height),
                mask=pnp.inlier_mask,
            )
            for match_idx, match in enumerate(matches):
                render_diag = contributor_stats.get(int(match.track_id), {})
                match_rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate["candidate_id"],
                        "selected_candidate_rank": candidate["rank"],
                        "match_index": int(match_idx),
                        "source": match.source,
                        "token_index": int(match.token_index),
                        "xy": [float(match.xy[0]), float(match.xy[1])],
                        "xyz": [float(v) for v in match.xyz.tolist()],
                        "similarity": float(match.similarity),
                        "ratio": float(match.ratio),
                        "similarity_margin": match.similarity_margin,
                        "distance_to_boundary_px": match.distance_to_boundary_px,
                        "gt_reproj_error_px": None
                        if gt_errors.shape[0] <= match_idx
                        else float(gt_errors[match_idx]),
                        "gt_inlier_5px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 5.0),
                        "gt_inlier_16px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 16.0),
                        "gt_inlier_32px": bool(gt_errors.shape[0] > match_idx and gt_errors[match_idx] <= 32.0),
                        "pnp_inlier": bool(
                            pnp.inlier_mask.shape[0] > match_idx and bool(pnp.inlier_mask[match_idx])
                        ),
                        "render_alpha": match.render_alpha,
                        "render_depth": match.render_depth,
                        "contributor_count": render_diag.get("contributor_count"),
                        "alpha_entropy": render_diag.get("alpha_entropy"),
                        "top1_alpha_contribution": render_diag.get("top1_alpha_contribution"),
                        "depth_variance_along_ray": render_diag.get("depth_variance_along_ray"),
                        "depth_range_along_ray": render_diag.get("depth_range_along_ray"),
                        "depth_to_rendered_delta": render_diag.get("depth_to_rendered_delta"),
                    }
                )
            diagnostic_values = list(contributor_stats.values())
            translation_error = None if pose_error is None else float(pose_error.translation_m)
            rotation_error = None if pose_error is None else float(pose_error.rotation_deg)
            row = {
                "query_id": query_id,
                "candidate_id": candidate["candidate_id"],
                "reference_image": candidate["reference_image"],
                "selected_candidate_rank": candidate["rank"],
                "candidate_count": len(candidates),
                "coordinate_audit": {
                    "query_feature_shape_chw": [int(v) for v in query_feature.shape],
                    "camera_model_id": int(camera.model_id),
                    "camera_width": int(camera.width),
                    "camera_height": int(camera.height),
                    "token_grid_scale_x_px": token_scale_x,
                    "token_grid_scale_y_px": token_scale_y,
                    "query_token_step": int(args.query_token_step),
                    "render_width": int(render_config.width),
                    "render_height": int(render_config.height),
                    "render_to_image_scale_x_px": float(camera.width) / max(float(render_config.width), 1.0),
                    "render_to_image_scale_y_px": float(camera.height) / max(float(render_config.height), 1.0),
                },
                "render_visible_pixel_count": int(np.sum(rendered.visibility_mask)),
                "render_visible_fraction": float(np.mean(rendered.visibility_mask)),
                "match_count": len(matches),
                "mean_similarity": _mean([match.similarity for match in matches]),
                "feature_precision_at_px": precision,
                "hard_false_match_rate": false_match_rate,
                "match_geometry": geometry_stats,
                "match_sources": {
                    "source": "dense_render",
                    "mean_render_alpha": _mean(
                        [float(match.render_alpha) for match in matches if match.render_alpha is not None]
                    ),
                    "median_render_alpha": None
                    if not [match.render_alpha for match in matches if match.render_alpha is not None]
                    else float(np.median([float(match.render_alpha) for match in matches if match.render_alpha is not None])),
                    "mean_render_depth": _mean(
                        [float(match.render_depth) for match in matches if match.render_depth is not None]
                    ),
                    "mean_similarity_margin": _mean(
                        [float(match.similarity_margin) for match in matches if match.similarity_margin is not None]
                    ),
                    "median_distance_to_boundary_px": None
                    if not [match.distance_to_boundary_px for match in matches if match.distance_to_boundary_px is not None]
                    else float(
                        np.median(
                            [
                                float(match.distance_to_boundary_px)
                                for match in matches
                                if match.distance_to_boundary_px is not None
                            ]
                        )
                    ),
                },
                "dense_render_diagnostics": {
                    "mean_alpha": _mean(
                        [float(match.render_alpha) for match in matches if match.render_alpha is not None]
                    ),
                    "median_alpha": None
                    if not [match.render_alpha for match in matches if match.render_alpha is not None]
                    else float(np.median([float(match.render_alpha) for match in matches if match.render_alpha is not None])),
                    "low_alpha_match_fraction": _mean(
                        [
                            1.0 if float(match.render_alpha) < 0.25 else 0.0
                            for match in matches
                            if match.render_alpha is not None
                        ]
                    ),
                    "mean_contributor_count": _mean(
                        [
                            float(item["contributor_count"])
                            for item in diagnostic_values
                            if item.get("contributor_count") is not None
                        ]
                    ),
                    "mean_alpha_entropy": _mean(
                        [float(item["alpha_entropy"]) for item in diagnostic_values if item.get("alpha_entropy") is not None]
                    ),
                    "mean_top1_alpha_contribution": _mean(
                        [
                            float(item["top1_alpha_contribution"])
                            for item in diagnostic_values
                            if item.get("top1_alpha_contribution") is not None
                        ]
                    ),
                    "mean_depth_variance_along_ray": _mean(
                        [
                            float(item["depth_variance_along_ray"])
                            for item in diagnostic_values
                            if item.get("depth_variance_along_ray") is not None
                        ]
                    ),
                    "mean_depth_range_along_ray": _mean(
                        [
                            float(item["depth_range_along_ray"])
                            for item in diagnostic_values
                            if item.get("depth_range_along_ray") is not None
                        ]
                    ),
                    "mean_depth_to_rendered_delta": _mean(
                        [
                            float(item["depth_to_rendered_delta"])
                            for item in diagnostic_values
                            if item.get("depth_to_rendered_delta") is not None
                        ]
                    ),
                    "alpha_entropy_available": True,
                    "top1_alpha_contribution_available": True,
                    "depth_variance_available": True,
                },
                "all_match_spatial": all_spatial_stats,
                "pnp_inlier_spatial": inlier_spatial_stats,
                "pnp_reprojection": pnp_residual_stats,
                "pnp_success": bool(pnp.success),
                "pnp_inlier_count": int(pnp.inlier_count),
                "pnp_inlier_ratio": float(pnp.inlier_ratio),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "success_10cm_5deg": bool(
                    pnp.success
                    and translation_error is not None
                    and translation_error <= 0.10
                    and rotation_error is not None
                    and rotation_error <= 5.0
                ),
                "success_25cm_10deg": bool(
                    pnp.success
                    and translation_error is not None
                    and translation_error <= 0.25
                    and rotation_error is not None
                    and rotation_error <= 10.0
                ),
                "candidate_metadata": candidate["metadata"],
            }
            score = (
                1 if row["pnp_success"] else 0,
                float(row["pnp_inlier_count"]),
                float(row["pnp_inlier_ratio"]),
                float(row["match_count"]),
                float(row["mean_similarity"]),
                -float(candidate["rank"]),
            )
            if best_row is None or score > best_row["_selection_score"]:
                row["_selection_score"] = score
                best_row = row
        if best_row is None:
            best_row = {
                "query_id": query_id,
                "candidate_id": None,
                "reference_image": None,
                "selected_candidate_rank": None,
                "candidate_count": 0,
                "render_visible_pixel_count": 0,
                "render_visible_fraction": 0.0,
                "match_count": 0,
                "mean_similarity": 0.0,
                "feature_precision_at_px": None,
                "hard_false_match_rate": None,
                "pnp_success": False,
                "pnp_inlier_count": 0,
                "pnp_inlier_ratio": 0.0,
                "translation_error_m": None,
                "rotation_error_deg": None,
                "success_10cm_5deg": False,
                "success_25cm_10deg": False,
                "candidate_metadata": {},
                "_selection_score": (0, 0.0, 0.0, 0.0, 0.0, 0.0),
            }
        best_row.pop("_selection_score", None)
        rows.append(best_row)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))
    if args.output_matches_jsonl:
        matches_path = Path(args.output_matches_jsonl)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        matches_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in match_rows) + ("\n" if match_rows else "")
        )

    labeled_rows = [row for row in rows if row["translation_error_m"] is not None]
    summary = {
        "stage": "query_to_render_vfm_matching_baseline",
        "query_count": len(rows),
        "labeled_query_count": len(labeled_rows),
        "field_count": len(field),
        "feature_dim": int(field.feature_dim),
        "matching_config": {
            "top_k": args.top_k,
            "ratio_threshold": None if args.disable_ratio_test else args.ratio_threshold,
            "min_similarity": args.min_similarity,
            "mutual": bool(args.mutual),
            "query_token_step": args.query_token_step,
            "max_matches": args.max_matches,
        },
        "camera": {
            "source": camera_source,
            "model_id": int(camera.model_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        },
        "render_config": {
            **render_config.to_dict(),
            "renderer": args.renderer,
            "device": args.device,
            "candidate_top_n": args.candidate_top_n,
        },
        "mean_candidate_count": _mean([float(row["candidate_count"]) for row in rows]),
        "mean_render_visible_fraction": _mean([float(row["render_visible_fraction"]) for row in rows]),
        "mean_match_count": _mean([float(row["match_count"]) for row in rows]),
        "mean_feature_precision": _mean(
            [float(row["feature_precision_at_px"]) for row in rows if row["feature_precision_at_px"] is not None]
        ),
        "mean_hard_false_match_rate": _mean(
            [float(row["hard_false_match_rate"]) for row in rows if row["hard_false_match_rate"] is not None]
        ),
        "mean_gt_precision_5px": _mean(
            [
                float(row["match_geometry"]["gt_precision_5px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_5px") is not None
            ]
        ),
        "mean_gt_precision_16px": _mean(
            [
                float(row["match_geometry"]["gt_precision_16px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_16px") is not None
            ]
        ),
        "mean_gt_precision_32px": _mean(
            [
                float(row["match_geometry"]["gt_precision_32px"])
                for row in rows
                if row["match_geometry"].get("gt_precision_32px") is not None
            ]
        ),
        "median_gt_reproj_median_px": None
        if not [row for row in rows if row["match_geometry"].get("gt_reproj_median_px") is not None]
        else float(
            np.median(
                [
                    float(row["match_geometry"]["gt_reproj_median_px"])
                    for row in rows
                    if row["match_geometry"].get("gt_reproj_median_px") is not None
                ]
            )
        ),
        "mean_pnp_inlier_gt_precision_5px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_5px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_5px") is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_16px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_16px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_16px") is not None
            ]
        ),
        "mean_pnp_inlier_gt_precision_32px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_32px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_32px") is not None
            ]
        ),
        "mean_pnp_reproj_inlier_median_px": _mean(
            [
                float(row["pnp_reprojection"]["pnp_reproj_inlier_median_px"])
                for row in rows
                if row["pnp_reprojection"].get("pnp_reproj_inlier_median_px") is not None
            ]
        ),
        "mean_pnp_inlier_bbox_area_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["bbox_area_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("bbox_area_frac") is not None
            ]
        ),
        "mean_pnp_inlier_grid_4x4_occupancy_frac": _mean(
            [
                float(row["pnp_inlier_spatial"]["grid_4x4_occupancy_frac"])
                for row in rows
                if row["pnp_inlier_spatial"].get("grid_4x4_occupancy_frac") is not None
            ]
        ),
        "mean_pnp_inlier_xy_pca_minor_major_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xy_pca_minor_major_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xy_pca_minor_major_ratio") is not None
            ]
        ),
        "mean_pnp_inlier_xyz_planarity_ratio": _mean(
            [
                float(row["pnp_inlier_spatial"]["xyz_planarity_ratio"])
                for row in rows
                if row["pnp_inlier_spatial"].get("xyz_planarity_ratio") is not None
            ]
        ),
        "pnp_success_rate": _mean([1.0 if row["pnp_success"] else 0.0 for row in rows]),
        "success_10cm_5deg": _mean([1.0 if row["success_10cm_5deg"] else 0.0 for row in labeled_rows]),
        "success_25cm_10deg": _mean([1.0 if row["success_25cm_10deg"] else 0.0 for row in labeled_rows]),
        "median_translation_error_m": None
        if not labeled_rows
        else float(np.median([float(row["translation_error_m"]) for row in labeled_rows])),
        "median_rotation_error_deg": None
        if not labeled_rows
        else float(np.median([float(row["rotation_error_deg"]) for row in labeled_rows])),
        "inputs": {
            "query_manifest": args.query_manifest,
            "field": args.field,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
        },
        "outputs": {"rows": str(output_jsonl), "matches": args.output_matches_jsonl},
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
