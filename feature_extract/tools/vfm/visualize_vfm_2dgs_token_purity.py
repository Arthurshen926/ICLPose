"""Visualize VFM-2DGS token purity and surface ambiguity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import _load_feature, _select_records
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_diagnostics import token_purity_diagnostic_grid
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsContributionBuffer


def _safe_name(image_id: str) -> str:
    return str(image_id).replace("/", "__").replace("\\", "__").replace(" ", "_")


def _read_image_rgb(path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _colorize_grid(values: np.ndarray, valid: np.ndarray, invert: bool = False) -> np.ndarray:
    import cv2

    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.asarray(valid, dtype=bool) & np.isfinite(arr)]
    if finite.size == 0:
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo, hi = float(np.percentile(finite, 5.0)), float(np.percentile(finite, 95.0))
        norm = (arr - lo) / max(hi - lo, 1e-8)
        if invert:
            norm = 1.0 - norm
        norm = np.where(np.isfinite(norm), norm, 0.0)
        scaled = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~np.asarray(valid, dtype=bool)] = np.asarray([20, 20, 20], dtype=np.uint8)
    return colored


def _draw_token_overlay(
    image_rgb: np.ndarray,
    token_count_h: int,
    token_count_w: int,
    component_grid: np.ndarray,
    entropy_grid: np.ndarray,
    support_grid: np.ndarray,
    suspect_grid: np.ndarray,
) -> np.ndarray:
    import cv2

    output = np.asarray(image_rgb, dtype=np.uint8).copy()
    image_h, image_w = output.shape[:2]
    cell_w = float(image_w) / float(token_count_w)
    cell_h = float(image_h) / float(token_count_h)
    for y in range(token_count_h):
        for x in range(token_count_w):
            if not np.isfinite(component_grid[y, x]):
                continue
            component = float(component_grid[y, x])
            entropy = float(entropy_grid[y, x]) if np.isfinite(entropy_grid[y, x]) else 0.0
            support = int(support_grid[y, x])
            suspect = bool(suspect_grid[y, x])
            color = (40, 230, 70) if not suspect else (255, 70, 40)
            thickness = 1 if not suspect else 2
            x0, y0 = int(round(x * cell_w)), int(round(y * cell_h))
            x1, y1 = int(round((x + 1) * cell_w)), int(round((y + 1) * cell_h))
            cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness=thickness, lineType=cv2.LINE_AA)
            if suspect:
                label = f"c{component:.2f} e{entropy:.2f} s{support}"
                cv2.putText(
                    output,
                    label,
                    (x0 + 2, min(y1 - 3, y0 + 11)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize VFM-2DGS token purity diagnostics")
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--contribution_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--query_ids", default="")
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--component_threshold", type=float, default=0.6)
    parser.add_argument("--entropy_threshold", type=float, default=0.7)
    parser.add_argument("--support_threshold", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    requested = {item.strip() for item in str(args.query_ids).split(",") if item.strip()}
    records = [record for record in manifest.records if not requested or record.image_id in requested]
    if not requested and int(args.max_queries) > 0:
        records = _select_records(records, int(args.max_queries), "uniform")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        contribution_path = Path(args.contribution_dir) / f"{_safe_name(record.image_id)}.npz"
        if not contribution_path.exists():
            continue
        feature_map = _load_feature(Path(record.token_path), str(args.layer_name))
        _channels, grid_h, grid_w = feature_map.shape
        buffer = Vfm2DgsContributionBuffer.load_npz(contribution_path)
        diagnostic = token_purity_diagnostic_grid(
            buffer,
            grid_shape=(int(grid_h), int(grid_w)),
            component_threshold=float(args.component_threshold),
            entropy_threshold=float(args.entropy_threshold),
            support_threshold=int(args.support_threshold),
        )
        image = _read_image_rgb(Path(args.image_root) / record.image_id)
        valid = np.isfinite(diagnostic["component_concentration_grid"])
        overlay = _draw_token_overlay(
            image,
            int(grid_h),
            int(grid_w),
            diagnostic["component_concentration_grid"],
            diagnostic["alpha_entropy_grid"],
            diagnostic["support_count_grid"],
            diagnostic["cross_surface_suspect_grid"],
        )
        component_heatmap = _colorize_grid(diagnostic["component_concentration_grid"], valid, invert=False)
        entropy_heatmap = _colorize_grid(diagnostic["alpha_entropy_grid"], valid, invert=False)
        support_heatmap = _colorize_grid(diagnostic["support_count_grid"].astype(np.float32), valid, invert=False)
        import cv2

        image_h, image_w = image.shape[:2]
        component_heatmap = cv2.resize(component_heatmap, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        entropy_heatmap = cv2.resize(entropy_heatmap, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        support_heatmap = cv2.resize(support_heatmap, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        stem = _safe_name(record.image_id)
        overlay_path = output_dir / f"{stem}_token_purity_overlay.png"
        component_path = output_dir / f"{stem}_component_concentration.png"
        entropy_path = output_dir / f"{stem}_alpha_entropy.png"
        support_path = output_dir / f"{stem}_support_count.png"
        _write_image_rgb(overlay_path, overlay)
        _write_image_rgb(component_path, component_heatmap)
        _write_image_rgb(entropy_path, entropy_heatmap)
        _write_image_rgb(support_path, support_heatmap)
        rows.append(
            {
                "image_id": record.image_id,
                "contribution_npz": str(contribution_path),
                "overlay_png": str(overlay_path),
                "component_concentration_png": str(component_path),
                "alpha_entropy_png": str(entropy_path),
                "support_count_png": str(support_path),
                "token_count": int(diagnostic["token_count"]),
                "cross_surface_suspect_count": int(diagnostic["cross_surface_suspect_count"]),
                "cross_surface_suspect_fraction": float(diagnostic["cross_surface_suspect_fraction"]),
                "low_component_count": int(diagnostic["low_component_count"]),
                "low_component_fraction": float(diagnostic["low_component_fraction"]),
                "high_entropy_count": int(diagnostic["high_entropy_count"]),
                "high_entropy_fraction": float(diagnostic["high_entropy_fraction"]),
                "large_support_count": int(diagnostic["large_support_count"]),
                "large_support_fraction": float(diagnostic["large_support_fraction"]),
                "mean_purity": float(diagnostic["mean_purity"]),
                "mean_component_concentration": float(diagnostic["mean_component_concentration"]),
                "mean_alpha_entropy": float(diagnostic["mean_alpha_entropy"]),
                "mean_support_count": float(diagnostic["mean_support_count"]),
            }
        )

    token_total = sum(int(row["token_count"]) for row in rows)
    suspect_total = sum(int(row["cross_surface_suspect_count"]) for row in rows)
    low_component_total = sum(int(row["low_component_count"]) for row in rows)
    high_entropy_total = sum(int(row["high_entropy_count"]) for row in rows)
    large_support_total = sum(int(row["large_support_count"]) for row in rows)
    summary = {
        "stage": "vfm_2dgs_token_purity_visualization",
        "visualized_view_count": int(len(rows)),
        "token_count": int(token_total),
        "cross_surface_suspect_count": int(suspect_total),
        "cross_surface_suspect_fraction": float(suspect_total / max(token_total, 1)),
        "low_component_count": int(low_component_total),
        "low_component_fraction": float(low_component_total / max(token_total, 1)),
        "high_entropy_count": int(high_entropy_total),
        "high_entropy_fraction": float(high_entropy_total / max(token_total, 1)),
        "large_support_count": int(large_support_total),
        "large_support_fraction": float(large_support_total / max(token_total, 1)),
        "thresholds": {
            "component": float(args.component_threshold),
            "entropy": float(args.entropy_threshold),
            "support": int(args.support_threshold),
        },
        "outputs": rows,
        "inputs": {
            "reference_manifest": str(args.reference_manifest),
            "contribution_dir": str(args.contribution_dir),
            "image_root": str(args.image_root),
            "layer_name": str(args.layer_name),
        },
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
