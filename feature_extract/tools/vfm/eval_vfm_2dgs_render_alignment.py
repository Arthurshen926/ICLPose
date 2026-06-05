"""Evaluate GT-aligned raw VFM-2DGS feature rendering quality."""

from __future__ import annotations

import argparse
import json
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
from feature_extract.vfm.query_to_3d_visualization import _fit_pca_projection, _pca_colors
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap
from feature_extract.vfm.vfm_2dgs_render_diagnostic import (
    Vfm2DgsRenderConfig,
    evaluate_gt_aligned_render_features,
    render_vfm_2dgs_anchor_features,
)


def _safe_query_name(query_id: str) -> str:
    return query_id.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _mean(values: list[float | None]) -> float | None:
    arr = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _median(values: list[float | None]) -> float | None:
    arr = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _read_image_rgb(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _render_feature_pca(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    channels, height, width = feature_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    flat = np.asarray(feature_map, dtype=np.float32).reshape(channels, -1).T
    visible = flat[np.asarray(mask, dtype=bool).reshape(-1)]
    if visible.shape[0] == 0:
        return rgb
    mean, components, low, high = _fit_pca_projection(visible)
    rgb.reshape(-1, 3)[np.asarray(mask, dtype=bool).reshape(-1)] = _pca_colors(visible, mean, components, low, high)
    return rgb


def _heatmap(values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    import cv2

    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[mask] = np.clip((values[mask] - float(vmin)) / max(float(vmax) - float(vmin), 1e-8), 0.0, 1.0)
    colors = cv2.applyColorMap(np.asarray(normalized * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    colors[~mask] = 0
    return colors


def _write_visualizations(
    output_dir: Path,
    query_id: str,
    query_feature: np.ndarray,
    rendered,
    metrics: dict[str, object],
    image_root: str,
) -> dict[str, str]:
    channels, height, width = query_feature.shape
    query_valid = np.linalg.norm(query_feature.reshape(channels, -1).T, axis=1).reshape(height, width) > 1e-8
    query_pca = _render_feature_pca(query_feature, query_valid)
    render_pca = _render_feature_pca(rendered.feature_map, rendered.visibility_mask)
    query_flat = query_feature.reshape(channels, -1).T
    render_flat = rendered.feature_map.reshape(channels, -1).T
    qn = np.linalg.norm(query_flat, axis=1, keepdims=True)
    rn = np.linalg.norm(render_flat, axis=1, keepdims=True)
    cosine = np.zeros((height * width,), dtype=np.float32)
    valid = (qn.reshape(-1) > 1e-8) & (rn.reshape(-1) > 1e-8) & rendered.visibility_mask.reshape(-1)
    cosine[valid] = np.sum(
        (query_flat[valid] / np.maximum(qn[valid], 1e-8)) * (render_flat[valid] / np.maximum(rn[valid], 1e-8)),
        axis=1,
    )
    cosine = cosine.reshape(height, width)
    cosine_heat = _heatmap(cosine, rendered.visibility_mask, -0.2, 0.8)
    alpha_heat = _heatmap(rendered.alpha_map, rendered.visibility_mask, 0.0, max(float(np.percentile(rendered.alpha_map[rendered.visibility_mask], 95.0)) if np.any(rendered.visibility_mask) else 1.0, 1e-6))
    variance_heat = _heatmap(rendered.variance_map, rendered.visibility_mask, 0.0, max(float(np.percentile(rendered.variance_map[rendered.visibility_mask], 95.0)) if np.any(rendered.visibility_mask) else 1.0, 1e-6))
    safe = _safe_query_name(query_id)
    outputs = {
        "query_feature_pca_png": str(output_dir / f"{safe}_query_feature_pca.png"),
        "render_feature_pca_png": str(output_dir / f"{safe}_render_feature_pca.png"),
        "cosine_heatmap_png": str(output_dir / f"{safe}_same_token_cosine.png"),
        "alpha_heatmap_png": str(output_dir / f"{safe}_render_alpha.png"),
        "variance_heatmap_png": str(output_dir / f"{safe}_render_variance.png"),
    }
    for key, image in (
        ("query_feature_pca_png", query_pca),
        ("render_feature_pca_png", render_pca),
        ("cosine_heatmap_png", cosine_heat),
        ("alpha_heatmap_png", alpha_heat),
        ("variance_heatmap_png", variance_heat),
    ):
        _write_image_rgb(Path(outputs[key]), image)
    if image_root:
        rgb = _read_image_rgb(Path(image_root) / query_id)
        if rgb is not None:
            outputs["query_rgb_png"] = str(output_dir / f"{safe}_query_rgb.png")
            _write_image_rgb(Path(outputs["query_rgb_png"]), rgb)
    (output_dir / f"{safe}_render_alignment_metrics.json").write_text(
        json.dumps({"query_id": query_id, "metrics": metrics, "outputs": outputs}, indent=2, sort_keys=True) + "\n"
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw VFM-2DGS GT-pose feature render alignment")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--surface_elements", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--query_id", action="append", default=[])
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--min_radius_px", type=float, default=0.5)
    parser.add_argument("--max_radius_px", type=float, default=4.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_visualizations", type=int, default=4)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = {record.image_id: record for record in manifest.records}
    pose_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    requested = list(args.query_id) if args.query_id else [record.image_id for record in manifest.records]
    if int(args.max_queries) > 0:
        requested = requested[: int(args.max_queries)]
    missing = [query_id for query_id in requested if query_id not in records or query_id not in pose_by_query]
    if missing:
        raise ValueError(f"query ids missing from manifest or pose file: {missing[:8]}")

    anchor_map = Vfm2DgsAnchorMap.load_npz(Path(args.anchor_map))
    surface_elements = SurfaceElementMap.load_npz(Path(args.surface_elements))
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else output_jsonl.parent / "render_alignment_vis"
    rows = []
    with output_jsonl.open("w") as handle:
        for idx, query_id in enumerate(requested):
            query_feature = _load_query_feature(records[query_id].token_path, args.layer_name)
            _channels, height, width = query_feature.shape
            render_config = Vfm2DgsRenderConfig(
                width=int(width),
                height=int(height),
                radius_scale=float(args.radius_scale),
                min_radius_px=float(args.min_radius_px),
                max_radius_px=float(args.max_radius_px),
                depth_epsilon=float(args.depth_epsilon),
            )
            rendered = render_vfm_2dgs_anchor_features(
                anchor_map,
                surface_elements,
                pose_w2c=pose_by_query[query_id].pose_w2c,
                camera=camera,
                config=render_config,
            )
            metrics = evaluate_gt_aligned_render_features(query_feature, rendered, top_k=int(args.top_k))
            row = {
                "query_id": query_id,
                **metrics,
                "render_config": {
                    "width": int(width),
                    "height": int(height),
                    "radius_scale": float(args.radius_scale),
                    "min_radius_px": float(args.min_radius_px),
                    "max_radius_px": float(args.max_radius_px),
                    "depth_epsilon": float(args.depth_epsilon),
                },
            }
            if idx < int(args.max_visualizations):
                row["visualizations"] = _write_visualizations(output_dir, query_id, query_feature, rendered, metrics, args.image_root)
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metric_keys = [
        "visible_fraction",
        "same_cosine_mean",
        "same_cosine_median",
        "negative_cosine_mean",
        "cosine_margin_mean",
        "same_gt_negative_win_rate",
        "top1_exact",
        "top1_within1",
        f"top{int(args.top_k)}_exact",
        f"top{int(args.top_k)}_within1",
    ]
    summary = {
        "stage": "vfm_2dgs_raw_render_alignment",
        "selector_free": True,
        "query_count": int(len(rows)),
        "anchor_count": int(len(anchor_map)),
        "surface_element_count": int(len(surface_elements)),
        "camera_source": camera_source,
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "anchor_map": str(args.anchor_map),
            "surface_elements": str(args.surface_elements),
            "query_pose_file": str(args.query_pose_file),
        },
        "outputs": {
            "rows": str(output_jsonl),
            "visualization_dir": str(output_dir),
        },
        "metrics_mean": {key: _mean([row.get(key) for row in rows]) for key in metric_keys},
        "metrics_median": {key: _median([row.get(key) for row in rows]) for key in metric_keys},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
