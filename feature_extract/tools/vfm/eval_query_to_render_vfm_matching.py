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
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
)
from feature_extract.vfm.query_to_3d_matching import (
    estimate_pose_pnp_ransac,
    pnp_pose_error,
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
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    field = GaussianVFMField.load_npz(Path(args.field))
    camera = _load_default_camera(args.camera_model_dir, _parse_default_camera(args.default_camera))
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
    records = list(manifest.records)
    if args.max_queries > 0:
        records = records[: args.max_queries]
    for record in records:
        query_id = record.image_id
        query_feature = _load_query_feature(record.token_path, args.layer_name)
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
            translation_error = None if pose_error is None else float(pose_error.translation_m)
            rotation_error = None if pose_error is None else float(pose_error.rotation_deg)
            row = {
                "query_id": query_id,
                "candidate_id": candidate["candidate_id"],
                "reference_image": candidate["reference_image"],
                "selected_candidate_rank": candidate["rank"],
                "candidate_count": len(candidates),
                "render_visible_pixel_count": int(np.sum(rendered.visibility_mask)),
                "render_visible_fraction": float(np.mean(rendered.visibility_mask)),
                "match_count": len(matches),
                "mean_similarity": _mean([match.similarity for match in matches]),
                "feature_precision_at_px": precision,
                "hard_false_match_rate": false_match_rate,
                "match_geometry": geometry_stats,
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
        "mean_pnp_inlier_gt_precision_16px": _mean(
            [
                float(row["match_geometry"]["pnp_inlier_gt_precision_16px"])
                for row in rows
                if row["match_geometry"].get("pnp_inlier_gt_precision_16px") is not None
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
        "outputs": {"rows": str(output_jsonl)},
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
