"""Camera-view diagnostics for scene-level dense Gaussian VFM fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.dense_gaussian_field_diagnostics import pca_feature_rgb, visibility_overlay_rgb
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianVFMRenderConfig,
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _read_image_rgb(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _draw_caption(image_rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas, "RGBA")
    height = 18 + 18 * len(lines)
    draw.rectangle((8, 8, 520, height), fill=(0, 0, 0, 155))
    for idx, line in enumerate(lines):
        draw.text((16, 14 + 18 * idx), line, fill=(255, 255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize dense Gaussian VFM fields from query camera views")
    parser.add_argument("--field", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--radius_px", type=float, default=2.0)
    parser.add_argument("--depth_epsilon", type=float, default=0.03)
    parser.add_argument("--renderer", choices=("soft", "gsplat"), default="soft")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    field = GaussianVFMField.load_npz(Path(args.field))
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    width, height = int(camera.width), int(camera.height)
    config = GaussianVFMRenderConfig(
        width=width,
        height=height,
        radius_px=float(args.radius_px),
        depth_epsilon=float(args.depth_epsilon),
        l2_normalize_pixels=True,
    )

    requested = {item.strip() for item in str(args.query_ids).split(",") if item.strip()}
    records = [
        record
        for record in manifest.records
        if record.image_id in poses and (not requested or record.image_id in requested)
    ]
    if not requested and int(args.max_images) > 0:
        records = records[: int(args.max_images)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    rows = []
    for record in records:
        image_rgb = _read_image_rgb(image_root / record.image_id, (width, height))
        if image_rgb is None:
            continue
        if args.renderer == "gsplat":
            result = render_gaussian_vfm_feature_map_gsplat(
                field,
                pose_w2c=poses[record.image_id].pose_w2c,
                camera=camera,
                config=config,
                device=args.device,
                channel_chunk=int(args.channel_chunk),
            )
        else:
            result = render_gaussian_vfm_feature_map(
                field,
                pose_w2c=poses[record.image_id].pose_w2c,
                camera=camera,
                config=config,
            )
        visible_count = int(np.sum(result.visibility_mask))
        visible_fraction = float(visible_count / max(width * height, 1))
        overlay = visibility_overlay_rgb(image_rgb, result.visibility_mask, color=(0, 255, 255), alpha=0.45)
        overlay = _draw_caption(
            overlay,
            (
                "Dense Gaussian VFM field camera-view visibility",
                f"cyan=rendered feature support, visible={visible_fraction:.3f}",
            ),
        )
        feature_rgb = pca_feature_rgb(result.feature_map, result.visibility_mask)
        feature_rgb = _draw_caption(
            feature_rgb,
            (
                "Dense Gaussian VFM field PCA feature color",
                f"visible={visible_fraction:.3f}, dim={result.feature_map.shape[0]}",
            ),
        )
        safe = _safe_name(record.image_id)
        overlay_path = output_dir / f"{safe}_dense_visibility_overlay.png"
        feature_path = output_dir / f"{safe}_dense_feature_pca.png"
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(feature_rgb).save(feature_path)
        rows.append(
            {
                "query_id": record.image_id,
                "visible_pixel_count": visible_count,
                "visible_fraction": visible_fraction,
                "mean_weight_sum": 0.0
                if visible_count == 0
                else float(np.mean(result.weight_sum[result.visibility_mask])),
                "overlay_png": str(overlay_path),
                "feature_pca_png": str(feature_path),
            }
        )

    summary = {
        "stage": "dense_gaussian_field_camera_visualization",
        "field": args.field,
        "query_manifest": args.query_manifest,
        "query_pose_file": args.query_pose_file,
        "image_root": args.image_root,
        "camera_source": camera_source,
        "renderer": args.renderer,
        "config": config.to_dict(),
        "image_count": int(len(rows)),
        "mean_visible_fraction": 0.0 if not rows else float(np.mean([row["visible_fraction"] for row in rows])),
        "rows": rows,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
