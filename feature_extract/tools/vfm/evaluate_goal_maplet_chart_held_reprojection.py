"""Pose-free-map held-view depth audit for an explicit chart atlas.

The reference is the frozen MASt3R point map, not sensor depth ground truth.
It is therefore a comparative mapping diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import ExplicitChartAtlas


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reference(pointmap: Path, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(pointmap.read_text())
    points = np.asarray(payload["points"], np.float64).reshape(288, 512, 3)
    conf = np.asarray(payload["confs"], np.float64)
    depth = ((points - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]
    depth = cv2.resize(depth, (256, 144), interpolation=cv2.INTER_AREA)
    conf = cv2.resize(conf, (256, 144), interpolation=cv2.INTER_AREA)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(conf) & (conf > 0.25)
    return depth, valid


def _project(vertices: np.ndarray, c2w: np.ndarray, focal_512: float):
    height, width = 144, 256
    camera = (vertices - c2w[:3, 3]) @ c2w[:3, :3]
    z = camera[:, 2]
    focal = float(focal_512) * 0.5
    u = focal * camera[:, 0] / z + (width - 1) / 2
    v = focal * camera[:, 1] / z + (height - 1) / 2
    return np.stack((u, v), axis=1), z, np.isfinite(camera).all(1) & (z > 1e-4)


def _render_vertices(vertices: np.ndarray, c2w: np.ndarray, focal_512: float) -> np.ndarray:
    height, width = 144, 256
    pixels, z, valid = _project(vertices, c2w, focal_512)
    u = np.rint(pixels[valid, 0]).astype(np.int64)
    v = np.rint(pixels[valid, 1]).astype(np.int64)
    z = z[valid]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = v[inside] * width + u[inside]
    rendered = np.full(height * width, np.inf, np.float64)
    np.minimum.at(rendered, flat, z[inside])
    return rendered.reshape(height, width)


def _render_triangles(vertices: np.ndarray, faces: np.ndarray, c2w: np.ndarray, focal_512: float) -> np.ndarray:
    """CPU reference rasterizer with perspective-correct z interpolation."""
    height, width = 144, 256
    pixels, z, valid_vertex = _project(vertices, c2w, focal_512)
    rendered = np.full((height, width), np.inf, np.float64)
    for face in faces:
        if not valid_vertex[face].all():
            continue
        triangle = pixels[face]
        z_triangle = z[face]
        min_u = max(0, int(np.floor(triangle[:, 0].min())))
        max_u = min(width - 1, int(np.ceil(triangle[:, 0].max())))
        min_v = max(0, int(np.floor(triangle[:, 1].min())))
        max_v = min(height - 1, int(np.ceil(triangle[:, 1].max())))
        if min_u > max_u or min_v > max_v:
            continue
        # Discontinuity-filtered source faces should remain bounded after
        # reprojection.  Reject pathological near-camera projections rather
        # than spending unbounded time on a diagnostic CPU rasterizer.
        if (max_u - min_u + 1) * (max_v - min_v + 1) > height * width // 2:
            continue
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if not np.isfinite(denominator) or abs(denominator) < 1e-10:
            continue
        yy, xx = np.mgrid[min_v:max_v + 1, min_u:max_u + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        if not inside.any():
            continue
        inverse_z = w0 / z_triangle[0] + w1 / z_triangle[1] + w2 / z_triangle[2]
        depth = np.where(inside & (inverse_z > 0), 1.0 / inverse_z, np.inf)
        region = rendered[min_v:max_v + 1, min_u:max_u + 1]
        np.minimum(region, depth, out=region)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--all_cameras", type=Path, required=True)
    parser.add_argument("--pointmaps_dir", type=Path, required=True)
    parser.add_argument("--route", default="seq4")
    parser.add_argument("--disjoint_upstream_authority", type=Path)
    parser.add_argument("--render_mode", choices=("vertex", "triangle"), default="triangle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite held-view chart audit")
    atlas = ExplicitChartAtlas.load_npz(args.atlas)
    cameras = json.loads(args.all_cameras.read_text())
    upstream_authority = None
    if args.disjoint_upstream_authority is not None:
        upstream_authority = json.loads(args.disjoint_upstream_authority.read_text())
        claimed = upstream_authority.pop("content_sha256", None)
        if claimed != _canonical(upstream_authority):
            raise ValueError("disjoint upstream authority content hash differs")
        upstream_authority["content_sha256"] = claimed
        if (
            upstream_authority.get("artifact_type")
            not in {
                "goal_maplet_disjoint_chart_upstream_authority_v1",
                "goal_maplet_disjoint_chart_upstream_authority_v2",
            }
            or not upstream_authority.get("strict_disjoint_upstream")
        ):
            raise ValueError("invalid disjoint upstream authority")
        held_authority = upstream_authority["held"]
        if args.all_cameras.resolve() != (
            Path(held_authority["root"]) / "cameras.json"
        ).resolve():
            raise ValueError("held cameras are not from the certified held-only run")
        if args.pointmaps_dir.resolve() != (
            Path(held_authority["root"]) / "pointmaps"
        ).resolve():
            raise ValueError("held pointmaps are not from the certified held-only run")
        if _sha(args.all_cameras) != held_authority["cameras_file_sha256"]:
            raise ValueError("held camera bytes differ from authority")
        rows_for_hash = [
            {
                "name": name,
                "file_sha256": _sha(
                    args.pointmaps_dir / f"{Path(name).stem}.json"
                ),
            }
            for name in sorted(held_authority["ordered_names"])
        ]
        if _canonical(rows_for_hash) != held_authority["pointmap_inventory_sha256"]:
            raise ValueError("held point-map bytes differ from authority")
        if args.route not in held_authority["routes"]:
            raise ValueError("requested held route is outside the certified held run")
        source_authority = atlas.metadata.get("source_authority")
        if source_authority is None or Path(source_authority).resolve() != args.disjoint_upstream_authority.resolve():
            raise ValueError("atlas is not bound to this disjoint upstream authority")
    rows = {Path(path).name: i for i, path in enumerate(cameras["filepaths"])}
    source = set(atlas.chart_names.tolist())
    held = sorted(
        name for name in rows
        if name.split("__", 1)[0] == args.route and name not in source
    )
    if not held:
        raise ValueError("no held chart cameras remain on the selected route")
    per_view = []
    all_error = []
    all_relative_error = []
    reference_count = 0
    compared_count = 0
    for name in held:
        row = rows[name]
        c2w = np.asarray(cameras["cams2world"][row], np.float64)
        reference, valid_reference = _reference(
            args.pointmaps_dir / f"{Path(name).stem}.json", c2w
        )
        rendered = (
            _render_triangles(atlas.vertices_world, atlas.faces, c2w, float(cameras["focals"][row]))
            if args.render_mode == "triangle"
            else _render_vertices(atlas.vertices_world, c2w, float(cameras["focals"][row]))
        )
        valid = valid_reference & np.isfinite(rendered)
        error = np.abs(rendered[valid] - reference[valid])
        relative_error = error / np.maximum(reference[valid], 1e-6)
        reference_count += int(valid_reference.sum())
        compared_count += int(valid.sum())
        all_error.append(error)
        all_relative_error.append(relative_error)
        per_view.append({
            "name": name,
            "reference_valid_pixels": int(valid_reference.sum()),
            "compared_pixels": int(valid.sum()),
            "coverage": float(valid.sum() / max(1, valid_reference.sum())),
            "absolute_depth_median_m": float(np.median(error)) if len(error) else None,
            "absolute_depth_p90_m": float(np.quantile(error, 0.9)) if len(error) else None,
            "relative_depth_median": float(np.median(relative_error)) if len(error) else None,
            "relative_depth_p90": float(np.quantile(relative_error, 0.9)) if len(error) else None,
        })
    errors = np.concatenate(all_error) if any(len(x) for x in all_error) else np.zeros(0)
    relative_errors = np.concatenate(all_relative_error) if any(len(x) for x in all_relative_error) else np.zeros(0)
    report = {
        "artifact_type": "goal_maplet_explicit_chart_held_reprojection_audit_v2",
        "reference_semantics": (
            "source_disjoint_frozen_MASt3R_mapping_pointmap_not_sensor_depth_GT"
            if upstream_authority
            else "frozen_MASt3R_mapping_pointmap_not_sensor_depth_GT"
        ),
        "render_semantics": (
            "CPU_triangle_zbuffer_perspective_correct_depth_at_256x144"
            if args.render_mode == "triangle"
            else "nearest_z_vertex_splat_at_256x144_no_triangle_fill"
        ),
        "atlas_file_sha256": _sha(args.atlas),
        "all_cameras_file_sha256": _sha(args.all_cameras),
        "disjoint_upstream_authority_file_sha256": (
            _sha(args.disjoint_upstream_authority)
            if args.disjoint_upstream_authority
            else None
        ),
        "strict_source_held_disjoint_upstream": bool(upstream_authority),
        "route": args.route,
        "source_chart_count": len(source),
        "held_view_count": len(held),
        "held_names": held,
        "coverage": float(compared_count / max(1, reference_count)),
        "compared_pixels": compared_count,
        "absolute_depth_median_m": float(np.median(errors)) if len(errors) else None,
        "absolute_depth_p90_m": float(np.quantile(errors, 0.9)) if len(errors) else None,
        "relative_depth_median": float(np.median(relative_errors)) if len(relative_errors) else None,
        "relative_depth_p90": float(np.quantile(relative_errors, 0.9)) if len(relative_errors) else None,
        "per_view": per_view,
        "uses_query_or_ground_truth": False,
        "production_eligible": False,
    }
    report["content_sha256"] = _canonical(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "per_view"}, indent=2))


if __name__ == "__main__":
    main()
