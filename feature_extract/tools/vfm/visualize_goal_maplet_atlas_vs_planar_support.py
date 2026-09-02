"""Compare sparse chart-atlas support with exact finite-plane 2DGS support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_moge_reference_held_render_control import (
    _normals,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import (
    _load_aligned_vertices,
)
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    StrictHeldRayInventory,
    _SurfaceMesh,
    _render_mesh,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
)


def _good(reference: np.ndarray, reference_valid: np.ndarray, depth: np.ndarray) -> np.ndarray:
    tolerance = np.maximum(0.5, 0.05 * reference)
    return reference_valid & np.isfinite(depth) & (np.abs(depth - reference) <= tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--held_cameras", type=Path, required=True)
    parser.add_argument("--planar_render_cache", type=Path, required=True)
    parser.add_argument("--planar_report", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite support comparison")

    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    vertices = _load_aligned_vertices(args.alignment, authority)
    atlas = _SurfaceMesh(
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
    row_by_name = {str(name): row for row, name in enumerate(held.view_names)}
    if any(name not in row_by_name or name not in image_by_name for name in args.names):
        raise ValueError("requested view is absent from the frozen held inventory")

    figure, axes = plt.subplots(
        len(args.names), 7, figsize=(23, 3.45 * len(args.names)), squeeze=False,
    )
    titles = (
        "query RGB", "reference depth", "16-chart atlas depth",
        "finite-plane member depth", "atlas |dz|", "finite-plane |dz|",
        "support: atlas red / plane green",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    rows = []
    for display_row, name in enumerate(args.names):
        view = row_by_name[name]
        bgr = cv2.imread(str(image_by_name[name]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_by_name[name])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 144), interpolation=cv2.INTER_AREA) / 255.0
        reference = held.reference_depth_m[view]
        reference_valid = held.reference_valid[view]
        atlas_depth, _ = _render_mesh(
            atlas,
            held.camera_to_world[view],
            held.focal_xy[view],
            held.principal_xy[view],
            144,
            256,
        )
        cache_path = args.planar_render_cache / f"{name}.npz"
        with np.load(cache_path, allow_pickle=False) as data:
            plane_depth = np.asarray(data["depth"], np.float64)
            plane_valid = np.asarray(data["valid"], bool)
        plane_depth = np.where(plane_valid, plane_depth, np.inf)
        atlas_valid = np.isfinite(atlas_depth)
        atlas_good = _good(reference, reference_valid, atlas_depth)
        plane_good = _good(reference, reference_valid, plane_depth)
        denominator = max(int(reference_valid.sum()), 1)
        atlas_coverage = float(np.sum(reference_valid & atlas_valid) / denominator)
        plane_coverage = float(np.sum(reference_valid & plane_valid) / denominator)
        atlas_good_recall = float(atlas_good.sum() / denominator)
        plane_good_recall = float(plane_good.sum() / denominator)
        rows.append({
            "name": name,
            "reference_ray_count": int(reference_valid.sum()),
            "atlas_rendered_recall": atlas_coverage,
            "finite_plane_rendered_recall": plane_coverage,
            "atlas_good_ray_recall": atlas_good_recall,
            "finite_plane_good_ray_recall": plane_good_recall,
        })
        valid_depth = reference[reference_valid]
        vmin, vmax = np.quantile(valid_depth, [0.02, 0.98])
        overlay = np.zeros((*reference.shape, 3), np.float64)
        overlay[..., 0] = reference_valid & atlas_valid
        overlay[..., 1] = reference_valid & plane_valid
        values = (
            (rgb, None, None),
            (np.where(reference_valid, reference, np.nan), "turbo", (vmin, vmax)),
            (np.where(atlas_valid, atlas_depth, np.nan), "turbo", (vmin, vmax)),
            (np.where(plane_valid, plane_depth, np.nan), "turbo", (vmin, vmax)),
            (np.where(reference_valid & atlas_valid, np.abs(atlas_depth - reference), np.nan), "magma", (0.0, 1.0)),
            (np.where(reference_valid & plane_valid, np.abs(plane_depth - reference), np.nan), "magma", (0.0, 1.0)),
            (overlay, None, None),
        )
        for column, (value, cmap, limits) in enumerate(values):
            axis = axes[display_row, column]
            if cmap is None:
                axis.imshow(value)
            else:
                axis.imshow(value, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_axis_off()
        axes[display_row, 0].set_ylabel(
            f"{name.split('frame')[-1].split('.')[0]}\n"
            f"atlas {atlas_good_recall:.1%}\nplane {plane_good_recall:.1%}",
            fontsize=9,
        )
    figure.suptitle(
        "Historical unequal-budget diagnostic: chart compression vs exact finite plane members",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)

    report = {
        "artifact_type": "goal_maplet_atlas_vs_exact_finite_plane_support_visualization_v1",
        "historical_unequal_budget_diagnostic_only": True,
        "full_train_plane_map_may_have_consumed_held_mapping_images": True,
        "convex_plane_boundary_consumed": False,
        "finite_plane_support": "exact member 2DGS Gaussian inventory",
        "authority_file_sha256": file_sha256(args.authority),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "planar_report_file_sha256": file_sha256(args.planar_report),
        "image": str(args.output.resolve()),
        "image_file_sha256": file_sha256(args.output),
        "rows": rows,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"image": str(args.output), "report": str(report_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
