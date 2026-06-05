"""Diagnostics for Stage H2 raw-VFM Gaussian landmark sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianRGBSource, load_gaussian_rgb_source_from_ply
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _intrinsic_matrix(camera, width: int, height: int) -> np.ndarray:
    params = tuple(float(value) for value in camera.params)
    if int(camera.model_id) == 1 and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif int(camera.model_id) in {0, 2, 8} and len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx = fy = params[0] if params else float(max(width, height))
        cx, cy = float(width) * 0.5, float(height) * 0.5
    sx = float(width) / max(float(camera.width), 1.0)
    sy = float(height) / max(float(camera.height), 1.0)
    return np.asarray([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_xyz(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    xyz_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    cam_xyz = (pose @ xyz_h.T).T[:, :3]
    depth = cam_xyz[:, 2]
    uvw = (_intrinsic_matrix(camera, width, height) @ cam_xyz.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    valid = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & np.isfinite(depth)
        & (depth > 1e-8)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    return uv.astype(np.float64), depth.astype(np.float64), valid


def _read_image_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _resize_for_output(image: Image.Image, max_width: int) -> Image.Image:
    if int(max_width) <= 0 or image.width <= int(max_width):
        return image
    scale = float(max_width) / float(image.width)
    return image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.BILINEAR)


def _draw_caption(image: Image.Image, lines: Sequence[str]) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    height = 20 + 18 * len(lines)
    draw.rectangle((8, 8, min(image.width - 8, 760), height), fill=(0, 0, 0, 165))
    for idx, line in enumerate(lines):
        draw.text((16, 14 + 18 * idx), line, fill=(255, 255, 255, 255))
    return image


def _pseudo_render_rgb(
    source: GaussianRGBSource,
    pose_w2c: np.ndarray,
    camera,
    width: int,
    height: int,
    max_gaussians: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(max_gaussians), int(source.xyz.shape[0])) if int(max_gaussians) > 0 else int(source.xyz.shape[0])
    uv, depth, valid = _project_xyz(source.xyz[:count], pose_w2c, camera, width, height)
    rows = np.flatnonzero(valid)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    if rows.size == 0:
        return canvas, np.zeros((height, width), dtype=bool)
    order = rows[np.argsort(depth[rows], kind="mergesort")]
    for row in order.tolist():
        x = int(round(float(uv[row, 0])))
        y = int(round(float(uv[row, 1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        z = float(depth[row])
        if z >= depth_buffer[y, x]:
            continue
        alpha = float(np.clip(source.opacity[row], 0.05, 1.0))
        color = np.asarray(source.rgb[row] * 255.0, dtype=np.float32)
        canvas[y, x] = np.asarray(color * alpha + canvas[y, x].astype(np.float32) * (1.0 - alpha), dtype=np.uint8)
        depth_buffer[y, x] = z
    mask = np.isfinite(depth_buffer)
    return canvas, mask


def _draw_anchor_overlay(
    image_rgb: np.ndarray,
    all_uv: np.ndarray,
    all_valid: np.ndarray,
    anchor_uv: np.ndarray,
    anchor_valid: np.ndarray,
    anchor_scores: np.ndarray,
    max_all_points: int,
    point_radius: int,
) -> np.ndarray:
    image = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    all_rows = np.flatnonzero(all_valid)
    if all_rows.size > int(max_all_points) > 0:
        rng = np.random.default_rng(17)
        all_rows = rng.choice(all_rows, size=int(max_all_points), replace=False)
    for row in all_rows.tolist():
        x, y = float(all_uv[row, 0]), float(all_uv[row, 1])
        draw.point((x, y), fill=(150, 150, 150, 65))
    valid_anchor_rows = np.flatnonzero(anchor_valid)
    scores = np.asarray(anchor_scores, dtype=np.float32).reshape(-1)
    finite_scores = scores[np.isfinite(scores)]
    lo = float(np.percentile(finite_scores, 5)) if finite_scores.size else 0.0
    hi = float(np.percentile(finite_scores, 95)) if finite_scores.size else 1.0
    for row in valid_anchor_rows.tolist():
        x, y = float(anchor_uv[row, 0]), float(anchor_uv[row, 1])
        t = 0.0 if hi <= lo else float(np.clip((scores[row] - lo) / (hi - lo), 0.0, 1.0))
        color = (int(255 * t), int(220 * (1.0 - 0.4 * t)), int(30 * (1.0 - t)), 225)
        r = int(point_radius)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0, 160))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _make_training_montage(input_paths: Sequence[Path], output_path: Path, max_width: int) -> list[str]:
    images = []
    used = []
    for path in input_paths:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img = _resize_for_output(img, max_width=max(1, int(max_width) // max(1, len(input_paths))))
        _draw_caption(img, (path.parent.name + "/" + path.name,))
        images.append(img)
        used.append(str(path))
    if not images:
        return []
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return used


def _write_distribution_plot(
    output_path: Path,
    source: GaussianRGBSource,
    vote_counts: np.ndarray,
    selected_indices: np.ndarray,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < source.xyz.shape[0])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    axes[0].hist(source.opacity, bins=80, alpha=0.55, density=True, label="all")
    if selected_indices.size:
        axes[0].hist(source.opacity[selected_indices], bins=80, alpha=0.55, density=True, label="selected")
    axes[0].set_title("Opacity distribution")
    axes[0].legend()
    axes[1].hist(source.scale, bins=80, alpha=0.55, density=True, label="all")
    if selected_indices.size:
        axes[1].hist(source.scale[selected_indices], bins=80, alpha=0.55, density=True, label="selected")
    axes[1].set_title("Mean exp(scale) distribution")
    axes[1].legend()
    nonzero = vote_counts[vote_counts > 0]
    axes[2].hist(nonzero, bins=np.arange(1, int(nonzero.max()) + 2) if nonzero.size else 10, alpha=0.8)
    axes[2].set_title("Nonzero VFM token-saliency vote counts")
    axes[2].set_xlabel("votes")
    axes[2].set_ylabel("count")
    if selected_indices.size:
        axes[3].scatter(
            source.opacity[selected_indices],
            vote_counts[selected_indices],
            s=3,
            alpha=0.25,
            linewidths=0,
        )
    axes[3].set_title("Selected anchors: opacity vs votes")
    axes[3].set_xlabel("opacity")
    axes[3].set_ylabel("votes")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _nearest_neighbor_stats(xyz: np.ndarray, max_points: int = 50000) -> dict[str, float | int]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2:
        return {"count": int(points.shape[0])}
    if points.shape[0] > int(max_points):
        rng = np.random.default_rng(23)
        points = points[rng.choice(points.shape[0], size=int(max_points), replace=False)]
    tree = cKDTree(points)
    distances, _indices = tree.query(points, k=2)
    return _stats(np.asarray(distances[:, 1], dtype=np.float32))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze trained 3DGS and Stage H2 Gaussian landmark sampling")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--anchor_npz", required=True)
    parser.add_argument("--votes_npz", required=True)
    parser.add_argument("--stage_h2_summary_json", default="")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--pseudo_render_max_gaussians", type=int, default=250000)
    parser.add_argument("--overlay_max_all_points", type=int, default=120000)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--max_output_width", type=int, default=1536)
    parser.add_argument("--training_vis", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    source = load_gaussian_rgb_source_from_ply(Path(args.gaussian_ply), max_gaussians=0)
    anchors = SemiDenseAnchorMap.load_npz(Path(args.anchor_npz))
    with np.load(Path(args.votes_npz)) as data:
        vote_counts = np.asarray(data["vote_counts"], dtype=np.int64)
        sampled_indices = np.asarray(data["sampled_indices"], dtype=np.int64) if "sampled_indices" in data else np.zeros((0,), dtype=np.int64)

    if vote_counts.shape[0] != source.xyz.shape[0]:
        raise ValueError("vote_counts length does not match Gaussian PLY row count")
    anchor_source_indices = anchors.source_gaussian_indices.astype(np.int64)
    anchor_source_indices = anchor_source_indices[(anchor_source_indices >= 0) & (anchor_source_indices < source.xyz.shape[0])]
    selected_mask = np.zeros((source.xyz.shape[0],), dtype=bool)
    selected_mask[anchor_source_indices] = True
    sampled_mask = np.zeros((source.xyz.shape[0],), dtype=bool)
    valid_sampled = sampled_indices[(sampled_indices >= 0) & (sampled_indices < source.xyz.shape[0])]
    sampled_mask[valid_sampled] = True
    voted_mask = vote_counts > 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_summary = {}
    if args.stage_h2_summary_json and Path(args.stage_h2_summary_json).exists():
        stage_summary = json.loads(Path(args.stage_h2_summary_json).read_text())

    training_vis_paths = [Path(item.strip()) for item in str(args.training_vis).split(",") if item.strip()]
    montage_path = output_dir / "training_3dgs_visualization_montage.png"
    montage_inputs = _make_training_montage(training_vis_paths, montage_path, max_width=int(args.max_output_width))
    distribution_plot = _write_distribution_plot(
        output_dir / "gaussian_sampling_distributions.png",
        source,
        vote_counts,
        anchor_source_indices,
    )

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    width, height = int(camera.width), int(camera.height)
    requested = {item.strip() for item in str(args.query_ids).split(",") if item.strip()}
    records = [
        record
        for record in manifest.records
        if record.image_id in poses and (not requested or record.image_id in requested)
    ]
    if not requested and int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]

    rows = []
    for record in records:
        rgb = _read_image_rgb(Path(args.image_root) / record.image_id, (width, height))
        pseudo_rgb, pseudo_mask = _pseudo_render_rgb(
            source,
            poses[record.image_id].pose_w2c,
            camera,
            width,
            height,
            max_gaussians=int(args.pseudo_render_max_gaussians),
        )
        all_uv, _all_depth, all_valid = _project_xyz(source.xyz, poses[record.image_id].pose_w2c, camera, width, height)
        anchor_uv, _anchor_depth, anchor_valid = _project_xyz(anchors.xyz, poses[record.image_id].pose_w2c, camera, width, height)
        anchor_scores = np.zeros((len(anchors),), dtype=np.float32)
        valid_idx = anchors.source_gaussian_indices.astype(np.int64)
        in_bounds = (valid_idx >= 0) & (valid_idx < vote_counts.shape[0])
        anchor_scores[in_bounds] = vote_counts[valid_idx[in_bounds]].astype(np.float32)
        overlay = _draw_anchor_overlay(
            rgb,
            all_uv,
            all_valid,
            anchor_uv,
            anchor_valid,
            anchor_scores,
            max_all_points=int(args.overlay_max_all_points),
            point_radius=int(args.point_radius),
        )
        pseudo_overlay = Image.blend(Image.fromarray(rgb), Image.fromarray(pseudo_rgb), alpha=0.65)
        _draw_caption(
            pseudo_overlay,
            (
                "3DGS pseudo RGB z-buffer projection",
                f"visible pixels={float(np.mean(pseudo_mask)):.3f}, gaussians={source.xyz.shape[0]}",
            ),
        )
        overlay_img = Image.fromarray(overlay)
        _draw_caption(
            overlay_img,
            (
                "Stage H2 Gaussian landmark sampling",
                "gray=projected all Gaussians, green/red=selected anchors by VFM vote",
                f"visible all={int(np.sum(all_valid))}, visible anchors={int(np.sum(anchor_valid))}",
            ),
        )
        safe = _safe_name(record.image_id)
        pseudo_path = output_dir / f"{safe}_3dgs_pseudo_rgb.png"
        overlay_path = output_dir / f"{safe}_h2_gaussian_landmark_overlay.png"
        _resize_for_output(pseudo_overlay.convert("RGB"), int(args.max_output_width)).save(pseudo_path)
        _resize_for_output(overlay_img.convert("RGB"), int(args.max_output_width)).save(overlay_path)
        rows.append(
            {
                "query_id": record.image_id,
                "visible_all_gaussians": int(np.sum(all_valid)),
                "visible_selected_anchors": int(np.sum(anchor_valid)),
                "selected_visible_ratio_vs_all": float(np.sum(anchor_valid) / max(float(np.sum(all_valid)), 1.0)),
                "pseudo_render_visible_pixel_fraction": float(np.mean(pseudo_mask)),
                "pseudo_rgb_png": str(pseudo_path),
                "anchor_overlay_png": str(overlay_path),
            }
        )

    all_stats = {
        "count": int(source.xyz.shape[0]),
        "bbox_min": [float(v) for v in np.min(source.xyz, axis=0).tolist()],
        "bbox_max": [float(v) for v in np.max(source.xyz, axis=0).tolist()],
        "opacity": _stats(source.opacity),
        "scale": _stats(source.scale),
        "nearest_neighbor_distance": _nearest_neighbor_stats(source.xyz),
    }
    selected_source_indices = anchor_source_indices
    selected_stats = {
        "count": int(selected_source_indices.size),
        "fraction_of_all_gaussians": float(selected_source_indices.size / max(source.xyz.shape[0], 1)),
        "fraction_of_voted_gaussians": float(selected_source_indices.size / max(int(np.sum(voted_mask)), 1)),
        "bbox_min": [float(v) for v in np.min(source.xyz[selected_source_indices], axis=0).tolist()] if selected_source_indices.size else [],
        "bbox_max": [float(v) for v in np.max(source.xyz[selected_source_indices], axis=0).tolist()] if selected_source_indices.size else [],
        "vote_counts": _stats(vote_counts[selected_source_indices]) if selected_source_indices.size else {},
        "opacity": _stats(source.opacity[selected_source_indices]) if selected_source_indices.size else {},
        "scale": _stats(source.scale[selected_source_indices]) if selected_source_indices.size else {},
        "anchor_observation_counts": _stats(anchors.observation_counts),
        "anchor_quality_scores": _stats(anchors.quality_scores),
        "nearest_neighbor_distance": _nearest_neighbor_stats(anchors.xyz),
    }
    voted_stats = {
        "count": int(np.sum(voted_mask)),
        "fraction_of_all_gaussians": float(np.sum(voted_mask) / max(source.xyz.shape[0], 1)),
        "vote_counts_nonzero": _stats(vote_counts[voted_mask]),
    }
    sampled_stats = {
        "count": int(np.sum(sampled_mask)),
        "fraction_of_all_gaussians": float(np.sum(sampled_mask) / max(source.xyz.shape[0], 1)),
        "survived_raw_aggregation_fraction": float(selected_source_indices.size / max(int(np.sum(sampled_mask)), 1)),
    }
    summary = {
        "stage": "stage_h2_gaussian_landmark_diagnostics",
        "inputs": {
            "gaussian_ply": args.gaussian_ply,
            "anchor_npz": args.anchor_npz,
            "votes_npz": args.votes_npz,
            "stage_h2_summary_json": args.stage_h2_summary_json,
            "query_manifest": args.query_manifest,
            "query_pose_file": args.query_pose_file,
            "image_root": args.image_root,
            "training_vis": [str(path) for path in training_vis_paths],
        },
        "camera_source": camera_source,
        "stage_h2_summary": {
            key: stage_summary.get(key)
            for key in ("source_gaussian_count", "sampled_gaussian_count", "anchor_count", "feature_dim", "view_count")
            if key in stage_summary
        },
        "all_gaussians": all_stats,
        "voted_gaussians": voted_stats,
        "sampled_gaussians": sampled_stats,
        "selected_anchors": selected_stats,
        "camera_rows": rows,
        "outputs": {
            "output_dir": str(output_dir),
            "training_montage_png": str(montage_path) if montage_inputs else "",
            "distribution_plot_png": distribution_plot,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
