"""Visualize historical held MoGe3 geometry, plane regions, and atlas support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_moge_reference_held_render_control import _normals
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    StrictHeldRayInventory,
    _SurfaceMesh,
    _render_mesh,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions


def _plane_rgb(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(260830)
    count = max(int(labels.max()) + 1, 1)
    palette = rng.uniform(0.15, 0.95, size=(count, 3))
    output = np.zeros((*labels.shape, 3), np.float64)
    assigned = labels >= 0
    output[assigned] = palette[labels[assigned]]
    output[~valid] = 0.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--held_cameras", type=Path, required=True)
    parser.add_argument("--moge3_query", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite held visualization")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    vertices = _load_aligned_vertices(args.alignment, authority)
    mesh = _SurfaceMesh(
        names=authority.chart_names,
        vertex_offsets=authority.chart_vertex_offsets,
        vertices=vertices,
        normals=_normals(vertices, authority.faces),
        face_offsets=authority.chart_face_offsets,
        faces=authority.faces,
    )
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    cameras = json.loads(args.held_cameras.read_text())
    image_by_name = {Path(path).name: Path(path) for path in cameras["filepaths"]}
    row_by_name = {name: row for row, name in enumerate(held.view_names.astype(str))}
    if any(name not in row_by_name or name not in image_by_name for name in args.names):
        raise ValueError("visualization name is absent from held inventory")
    figure, axes = plt.subplots(len(args.names), 6, figsize=(20, 3.6 * len(args.names)), squeeze=False)
    titles = ("query RGB", "reference depth", "MoGe3 scaled depth", "scaled |depth error|", "MoGe3 plane regions", "atlas depth / support")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11)
    for row_index, name in enumerate(args.names):
        view = row_by_name[name]
        bgr = cv2.imread(str(image_by_name[name]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 144), interpolation=cv2.INTER_AREA) / 255.0
        with np.load(args.moge3_query / f"{name}.npz", allow_pickle=False) as data:
            points = np.asarray(data["points_camera"], np.float64)
            normal = np.asarray(data["normal_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        reference = held.reference_depth_m[view]
        common = held.reference_valid[view] & valid
        scale = float(np.median(reference[common] / points[..., 2][common]))
        scaled = scale * points[..., 2]
        error = np.abs(scaled - reference)
        planes = extract_query_plane_regions(points, normal, valid)
        render, _ = _render_mesh(
            mesh,
            held.camera_to_world[view],
            held.focal_xy[view],
            held.principal_xy[view],
            144,
            256,
        )
        depth_values = reference[held.reference_valid[view]]
        vmax = float(np.quantile(depth_values, 0.98))
        vmin = float(np.quantile(depth_values, 0.02))
        support = np.isfinite(render)
        support_image = np.where(support, render, np.nan)
        displays = (
            (rgb, None),
            (np.where(held.reference_valid[view], reference, np.nan), "turbo"),
            (np.where(valid, scaled, np.nan), "turbo"),
            (np.where(common, error, np.nan), "magma"),
            (_plane_rgb(planes.labels, valid), None),
            (support_image, "turbo"),
        )
        for column, (value, cmap) in enumerate(displays):
            axis = axes[row_index, column]
            if cmap is None:
                axis.imshow(value)
            elif column == 3:
                axis.imshow(value, cmap=cmap, vmin=0.0, vmax=0.5)
            else:
                axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_axis_off()
        associated = np.sum(held.reference_valid[view] & support) / max(int(held.reference_valid[view].sum()), 1)
        axes[row_index, 0].set_ylabel(
            f"{name.split('frame')[-1].split('.')[0]}\nscale={scale:.3f}\nrender={associated:.1%}",
            fontsize=10,
        )
    figure.suptitle(
        "Historical held diagnostic: MoGe3 scale, finite planes, and MoGe/reference atlas support",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
