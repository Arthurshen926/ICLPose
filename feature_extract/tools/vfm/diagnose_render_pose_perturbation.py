"""Visualize and quantify GT-vs-perturbed render inputs for localization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _render_rgb_and_depth
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
    _read_rgb,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.render_pose_protocol import translate_pose_world


def _load_manifest_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [dict(item) for item in records]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"unsupported query manifest format: {path}")


def _parse_offsets(text: str) -> list[np.ndarray]:
    offsets = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        values = [float(part.strip()) for part in item.split(",")]
        if len(values) != 3:
            raise ValueError("offsets must be formatted as dx,dy,dz;dx,dy,dz")
        offsets.append(np.asarray(values, dtype=np.float64))
    if not offsets:
        raise ValueError("at least one offset is required")
    return offsets


def _safe_stem(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__")


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for render diagnostics") from exc
    return cv2.resize(np.asarray(image, dtype=np.uint8), (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _depth_to_rgb(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for render diagnostics") from exc
    values = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(values) & (values > 0.0)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return np.zeros((*values.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(values[mask], [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values[mask])), float(np.max(values[mask]) + 1e-6)
    normalized = np.clip((values - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap(np.asarray(np.rint(normalized * 255.0), dtype=np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~mask] = 0
    return colored.astype(np.uint8, copy=False)


def _alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    return np.repeat(np.asarray(np.rint(a * 255.0), dtype=np.uint8)[..., None], 3, axis=2)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for render diagnostics") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise ValueError(f"failed to write image: {path}")


def _project_flow_stats(depth: np.ndarray, camera, gt_pose_w2c: np.ndarray, render_pose_w2c: np.ndarray, *, step: int = 8) -> dict[str, float | None]:
    """Project GT-render depth points into a perturbed camera and summarize pixel displacement."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for render diagnostics") from exc
    depth_values = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth_values) & (depth_values > 0.0)
    ys, xs = np.nonzero(valid[:: int(step), :: int(step)])
    if xs.size == 0:
        return {"flow_median_px": None, "flow_p90_px": None, "flow_p95_px": None, "flow_visible_fraction": 0.0}
    xs = xs.astype(np.float64) * float(step)
    ys = ys.astype(np.float64) * float(step)
    z = depth_values[ys.astype(np.int64), xs.astype(np.int64)]
    matrix, distortion = camera_matrix_and_distortion(camera)
    if np.any(np.abs(distortion) > 1e-12):
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(points, matrix, distortion).reshape(-1, 2)
        xnorm, ynorm = undist[:, 0], undist[:, 1]
    else:
        fx, fy = matrix[0, 0], matrix[1, 1]
        cx, cy = matrix[0, 2], matrix[1, 2]
        xnorm = (xs - cx) / fx
        ynorm = (ys - cy) / fy
    points_cam = np.stack([xnorm * z, ynorm * z, z], axis=1)
    gt_c2w = np.linalg.inv(np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4))
    points_world = (gt_c2w[:3, :3] @ points_cam.T).T + gt_c2w[:3, 3]
    perturb = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4)
    points_render = (perturb[:3, :3] @ points_world.T).T + perturb[:3, 3]
    in_front = points_render[:, 2] > 1e-6
    if not np.any(in_front):
        return {"flow_median_px": None, "flow_p90_px": None, "flow_p95_px": None, "flow_visible_fraction": 0.0}
    projected, _jac = cv2.projectPoints(
        points_world[in_front],
        cv2.Rodrigues(perturb[:3, :3])[0],
        perturb[:3, 3],
        matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    source = np.stack([xs[in_front], ys[in_front]], axis=1)
    inside = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(camera.width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(camera.height - 1))
    )
    if not np.any(inside):
        return {"flow_median_px": None, "flow_p90_px": None, "flow_p95_px": None, "flow_visible_fraction": 0.0}
    flow = np.linalg.norm(projected[inside] - source[inside], axis=1)
    return {
        "flow_median_px": float(np.median(flow)),
        "flow_p90_px": float(np.percentile(flow, 90.0)),
        "flow_p95_px": float(np.percentile(flow, 95.0)),
        "flow_visible_fraction": float(np.mean(inside)),
    }


def _render_stats(rgb: np.ndarray, depth: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
    valid_depth = np.isfinite(depth) & (np.asarray(depth) > 0.0)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    return {
        "alpha_mean": float(np.mean(alpha_arr)),
        "alpha_gt_0p05": float(np.mean(alpha_arr > 0.05)),
        "alpha_gt_0p5": float(np.mean(alpha_arr > 0.5)),
        "depth_valid_fraction": float(np.mean(valid_depth)),
        "depth_median_m": None if not np.any(valid_depth) else float(np.median(np.asarray(depth)[valid_depth])),
        "rgb_nonzero_fraction": float(np.mean(np.any(np.asarray(rgb, dtype=np.uint8) > 0, axis=2))),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--offsets", default="0,0,0;0.05,0,0;0.10,0,0;0.25,0,0")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest_records(Path(args.query_manifest))
    records = manifest[: int(args.max_queries) if int(args.max_queries) > 0 else len(manifest)]
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(str(args.query_pose_file), str(args.camera_model_dir))
    camera, _camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(str(args.default_camera)))
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    config = GaussianVFMRenderConfig(width=int(args.render_width), height=int(args.render_height))
    source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
    offsets = _parse_offsets(args.offsets)
    rows: list[dict[str, object]] = []
    for record in records:
        image_id = str(record["image_id"])
        if image_id not in poses:
            raise KeyError(f"pose not found for {image_id}")
        query_rgb = _resize_rgb(_read_rgb(Path(args.image_root) / image_id), int(args.render_width), int(args.render_height))
        gt_pose = poses[image_id].pose_w2c
        panels = [query_rgb]
        panel_labels = ["query"]
        gt_depth = None
        gt_rgb = None
        gt_alpha = None
        for offset in offsets:
            pose = translate_pose_world(gt_pose, offset)
            rgb, depth, alpha = _render_rgb_and_depth(
                source,
                None,
                pose_w2c=pose,
                camera=render_camera,
                config=config,
                renderer="official_2dgs",
                device=str(args.device),
            )
            stem = _safe_stem(image_id)
            label = f"dx{offset[0]:+.2f}_dy{offset[1]:+.2f}_dz{offset[2]:+.2f}"
            _write_rgb(output_dir / "renders" / f"{stem}_{label}_rgb.png", rgb)
            _write_rgb(output_dir / "renders" / f"{stem}_{label}_depth.png", _depth_to_rgb(depth))
            _write_rgb(output_dir / "renders" / f"{stem}_{label}_alpha.png", _alpha_to_rgb(alpha))
            if np.allclose(offset, 0.0):
                gt_rgb, gt_depth, gt_alpha = rgb, depth, alpha
            panels.append(rgb)
            panel_labels.append(label)
            stats = _render_stats(rgb, depth, alpha)
            flow = (
                {"flow_median_px": 0.0, "flow_p90_px": 0.0, "flow_p95_px": 0.0, "flow_visible_fraction": 1.0}
                if np.allclose(offset, 0.0)
                else _project_flow_stats(gt_depth if gt_depth is not None else depth, render_camera, gt_pose, pose)
            )
            row = {
                "query_id": image_id,
                "offset_x_m": float(offset[0]),
                "offset_y_m": float(offset[1]),
                "offset_z_m": float(offset[2]),
                **stats,
                **flow,
            }
            if gt_rgb is not None and not np.allclose(offset, 0.0):
                common_alpha = (np.asarray(alpha) > 0.05) & (np.asarray(gt_alpha) > 0.05)
                rgb_delta = np.mean(np.abs(np.asarray(rgb, dtype=np.float32) - np.asarray(gt_rgb, dtype=np.float32)), axis=2)
                row["rgb_mae_vs_gt_common_alpha"] = None if not np.any(common_alpha) else float(np.mean(rgb_delta[common_alpha]))
                depth_delta = np.abs(np.asarray(depth, dtype=np.float32) - np.asarray(gt_depth, dtype=np.float32))
                valid_depth = common_alpha & np.isfinite(depth_delta)
                row["depth_mae_vs_gt_common_alpha_m"] = None if not np.any(valid_depth) else float(np.mean(depth_delta[valid_depth]))
            rows.append(row)
        resized_panels = []
        for label, panel in zip(panel_labels, panels):
            canvas = np.asarray(panel, dtype=np.uint8).copy()
            try:
                import cv2

                cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            except Exception:
                pass
            resized_panels.append(canvas)
        _write_rgb(output_dir / "composites" / f"{_safe_stem(image_id)}_rgb_offsets.png", np.concatenate(resized_panels, axis=1))
    rows_path = output_dir / "render_perturbation_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "query_count": len(records),
        "offsets": [offset.tolist() for offset in offsets],
        "rows": str(rows_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
