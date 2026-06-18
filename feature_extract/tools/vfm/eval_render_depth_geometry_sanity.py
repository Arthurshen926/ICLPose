"""Evaluate the ideal rendered-depth geometry/PnP upper bound.

This diagnostic bypasses feature matching. For each query pose it renders depth
at the GT pose, samples a regular image grid, backprojects those pixels to 3D,
then solves PnP from the same 2D pixels. If this does not recover the GT pose
to centimeter-level accuracy, the bottleneck is in camera/depth/backprojection/
PnP geometry rather than MATCHA matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import (
    _depth_field_from_rgb_source,
    _load_or_render_rgb_depth_cache,
    _render_rgb_and_depth,
    _resolve_render_size,
    _safe_image_stem,
    _select_records,
)
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _parse_default_camera,
    _scale_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMRenderConfig, load_gaussian_rgb_source_from_ply
from feature_extract.vfm.official_2dgs_renderer import load_official_2dgs_source_from_ply
from feature_extract.vfm.render_pose_diagnostics import (
    ideal_depth_correspondences,
    render_depth_roundtrip_stats,
    run_pnp_solver_ablation,
)
from feature_extract.vfm.rendered_keypoint_matching import _sample_scalar_map
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_sample_grid(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(",", "x")
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("sample grid must be formatted as COLSxROWS, e.g. 32x18")
    cols, rows = int(parts[0]), int(parts[1])
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("sample grid dimensions must be positive")
    return cols, rows


def _grid_xy(width: int, height: int, grid: tuple[int, int], *, margin_px: float = 0.5) -> np.ndarray:
    cols, rows = int(grid[0]), int(grid[1])
    margin = max(float(margin_px), 0.0)
    xs = np.linspace(margin, max(float(width - 1) - margin, margin), cols, dtype=np.float64)
    ys = np.linspace(margin, max(float(height - 1) - margin, margin), rows, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=bool)))


def _summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_solver: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_solver.setdefault(str(row.get("solver")), []).append(dict(row))
    solvers: dict[str, object] = {}
    for solver, solver_rows in sorted(by_solver.items()):
        translations = [float(row["translation_error_m"]) for row in solver_rows if row.get("translation_error_m") is not None]
        rotations = [float(row["rotation_error_deg"]) for row in solver_rows if row.get("rotation_error_deg") is not None]
        residuals = [float(row["residual_median_px"]) for row in solver_rows if row.get("residual_median_px") is not None]
        success_flags = [bool(row.get("success")) for row in solver_rows]
        solvers[solver] = {
            "query_count": int(len(solver_rows)),
            "solve_rate": _rate(success_flags),
            "median_translation_error_m": _median(translations),
            "median_rotation_error_deg": _median(rotations),
            "median_residual_median_px": _median(residuals),
            "success_3cm_1deg": _rate(
                [
                    bool(row.get("success"))
                    and row.get("translation_error_m") is not None
                    and float(row["translation_error_m"]) <= 0.03
                    and row.get("rotation_error_deg") is not None
                    and float(row["rotation_error_deg"]) <= 1.0
                    for row in solver_rows
                ]
            ),
        }
    valid_counts = [float(row["valid_match_count"]) for row in rows if row.get("valid_match_count") is not None]
    return {
        "query_solver_row_count": int(len(rows)),
        "solver_count": int(len(by_solver)),
        "median_valid_match_count": _median(valid_counts),
        "solvers": solvers,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--rows_csv", default="")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=0)
    parser.add_argument("--render_height", type=int, default=0)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="official_2dgs", choices=("soft", "gsplat", "official_2dgs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--skip_existing_render_rgb_depth", action="store_true")
    parser.add_argument("--sample_grid", type=_parse_sample_grid, default=_parse_sample_grid("32x18"))
    parser.add_argument("--sample_margin_px", type=float, default=4.0)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument(
        "--solvers",
        default="plain,ransac,magsac,weighted,covariance,oracle_uncertainty",
        help="Comma-separated solver matrix.",
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=4.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--max_queries", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    solvers = tuple(part.strip() for part in str(args.solvers).split(",") if part.strip())
    grid_text = f"{int(args.sample_grid[0])}x{int(args.sample_grid[1])}"
    if bool(args.dry_run):
        print(
            "eval_render_depth_geometry_sanity "
            f"--render_pose_mode gt --sample_grid {grid_text} --solvers {','.join(solvers)} "
            f"--max_queries {int(args.max_queries)} --start_index {int(args.start_index)}"
        )
        return

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "summary.json"
    rows_path = Path(args.rows_csv) if args.rows_csv else output_dir / "rows.csv"

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = _select_records(
        manifest.records,
        int(args.max_queries),
        str(args.view_selection),
        start_index=int(args.start_index),
    )
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_width, render_height = _resolve_render_size(camera, int(args.render_width), int(args.render_height))
    render_camera = _scale_camera(camera, render_width, render_height)
    render_config = GaussianVFMRenderConfig(
        width=render_width,
        height=render_height,
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )
    if str(args.renderer) == "official_2dgs":
        rgb_source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = None
    else:
        rgb_source = load_gaussian_rgb_source_from_ply(Path(args.gaussian_rgb_ply))
        depth_field = _depth_field_from_rgb_source(rgb_source)

    render_cache_dir = Path(args.render_rgb_depth_cache_dir) if args.render_rgb_depth_cache_dir else None
    rows: list[dict[str, object]] = []
    skipped = {"missing_pose": 0, "no_valid_depth": 0}
    xy = _grid_xy(render_width, render_height, args.sample_grid, margin_px=float(args.sample_margin_px))
    for record_index, record in enumerate(records):
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            skipped["missing_pose"] += 1
            continue
        cache_path = None
        if render_cache_dir is not None:
            cache_path = render_cache_dir / f"{_safe_image_stem(record.image_id)}_{render_width}x{render_height}.npz"
        _rgb, depth, alpha = _load_or_render_rgb_depth_cache(
            cache_path=cache_path,
            render_fn=lambda gt_pose=gt.pose_w2c: _render_rgb_and_depth(
                rgb_source,
                depth_field,
                pose_w2c=gt_pose,
                camera=camera,
                config=render_config,
                renderer=str(args.renderer),
                device=str(args.device),
            ),
            skip_existing=bool(args.skip_existing_render_rgb_depth),
        )
        depth_samples, depth_valid = _sample_scalar_map(depth, xy, render_width, render_height)
        alpha_samples, alpha_valid = _sample_scalar_map(alpha, xy, render_width, render_height)
        valid_xy = xy[depth_valid & alpha_valid]
        valid_depth = depth_samples[depth_valid & alpha_valid]
        valid_alpha = alpha_samples[depth_valid & alpha_valid]
        matches, match_summary = ideal_depth_correspondences(
            valid_xy,
            valid_depth,
            render_camera,
            gt.pose_w2c,
            alpha=valid_alpha,
            min_alpha=float(args.min_alpha),
            source="ideal_2dgs_depth",
            measurement_sigma_px=1.0,
        )
        if not matches:
            skipped["no_valid_depth"] += 1
            continue
        roundtrip = render_depth_roundtrip_stats(valid_xy, valid_depth, render_camera, gt.pose_w2c)
        ablation = run_pnp_solver_ablation(
            matches,
            render_camera,
            gt_pose_w2c=gt.pose_w2c,
            solvers=solvers,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            confidence=float(args.pnp_confidence),
            iterations=int(args.pnp_iterations),
        )
        for solver, solver_row in ablation.items():
            rows.append(
                {
                    "query_id": record.image_id,
                    "query_index": int(record_index),
                    "solver": str(solver),
                    **match_summary,
                    "roundtrip_median_error_px": roundtrip.get("median_roundtrip_error_px"),
                    "roundtrip_p95_error_px": roundtrip.get("p95_roundtrip_error_px"),
                    **solver_row,
                }
            )

    _write_csv(rows_path, rows)
    summary = {
        "stage": "render_depth_geometry_sanity",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        },
        "camera": {
            "source": str(camera_source),
            "width": int(camera.width),
            "height": int(camera.height),
            "render_width": int(render_width),
            "render_height": int(render_height),
        },
        "config": {
            "renderer": str(args.renderer),
            "sample_grid": grid_text,
            "min_alpha": float(args.min_alpha),
            "solvers": list(solvers),
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
        },
        "skipped": skipped,
        "metrics": _summarize_rows(rows),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
