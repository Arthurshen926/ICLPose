"""Camera-view visualization for Dense Patch Map around sparse anchors."""

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
from feature_extract.vfm.dense_patch_context import DensePatchContextBank
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _project(points: np.ndarray, pose_w2c: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = (pose[:3, :3] @ pts.T + pose[:3, 3:4]).T
    z = cam[:, 2]
    visible = z > 1e-8
    safe_z = np.where(visible, z, 1.0)
    x = cam[:, 0] / safe_z
    y = cam[:, 1] / safe_z
    if int(camera.model_id) == 0:
        f, cx, cy = camera.params[:3]
        u = float(f) * x + float(cx)
        v = float(f) * y + float(cy)
    elif int(camera.model_id) == 1:
        fx, fy, cx, cy = camera.params[:4]
        u = float(fx) * x + float(cx)
        v = float(fy) * y + float(cy)
    elif int(camera.model_id) == 2:
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x * x + y * y)
        u = float(f) * x * radial + float(cx)
        v = float(f) * y * radial + float(cy)
    else:
        raise ValueError(f"unsupported camera model id: {camera.model_id}")
    xy = np.stack([u, v], axis=1)
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy.astype(np.float64), visible


def _draw_circle(draw: ImageDraw.ImageDraw, xy: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, fill=color)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize dense patch context support in camera view")
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--dense_context_bank", required=True)
    parser.add_argument("--semidense_npz", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--max_anchors", type=int, default=120)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    context_bank = DensePatchContextBank.load_npz(Path(args.dense_context_bank))
    semidense = SemiDenseAnchorMap.load_npz(Path(args.semidense_npz))
    poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    records = list(manifest.records)[: int(args.max_images)]
    for record in records:
        pose = poses.get(record.image_id)
        image_path = Path(args.image_root) / record.image_id
        if pose is None or not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        if image.size != (int(camera.width), int(camera.height)):
            image = image.resize((int(camera.width), int(camera.height)))
        draw = ImageDraw.Draw(image, "RGBA")
        anchor_xy, anchor_visible = _project(context_bank.anchor_xyz, pose.pose_w2c, camera)
        candidate_rows = np.flatnonzero(anchor_visible & (context_bank.support_counts > 0))
        if candidate_rows.size:
            order = candidate_rows[np.argsort(-context_bank.reliability_scores[candidate_rows])]
            candidate_rows = order[: int(args.max_anchors)]
        support_drawn = 0
        for row in candidate_rows.tolist():
            anchor = anchor_xy[int(row)]
            support_rows = context_bank.support_indices[int(row)]
            support_rows = support_rows[support_rows >= 0]
            if support_rows.size:
                support_xy, support_visible = _project(semidense.xyz[support_rows], pose.pose_w2c, camera)
                for support_point, visible in zip(support_xy, support_visible):
                    if not bool(visible):
                        continue
                    draw.line((float(anchor[0]), float(anchor[1]), float(support_point[0]), float(support_point[1])), fill=(255, 210, 0, 100), width=1)
                    _draw_circle(draw, support_point, 2, (0, 210, 255))
                    support_drawn += 1
            _draw_circle(draw, anchor, 4, (0, 255, 80))
        draw.rectangle((8, 8, 395, 58), fill=(0, 0, 0, 150))
        draw.text((16, 14), "Dense Patch Context around Sparse Anchors", fill=(255, 255, 255, 255))
        draw.text((16, 34), "green=sparse anchor, cyan=dense support, yellow=support link", fill=(255, 255, 255, 255))
        safe_name = record.image_id.replace("/", "__")
        out = output_dir / f"{safe_name}_dense_patch_context.png"
        image.save(out)
        rows.append(
            {
                "query_id": record.image_id,
                "output": str(out),
                "visible_context_anchors": int(candidate_rows.size),
                "support_points_drawn": int(support_drawn),
            }
        )
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "stage": "dense_patch_context_camera_visualization",
                "camera_source": camera_source,
                "image_count": int(len(rows)),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
