"""Visual comparison of two explicit chart-atlas geometry gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import ExplicitChartAtlas


def _scatter(ax, atlas: ExplicitChartAtlas, axes: tuple[int, int], title: str) -> None:
    colors = plt.get_cmap("tab10")
    for chart in range(len(atlas.chart_names)):
        lo, hi = map(int, atlas.chart_vertex_offsets[chart:chart + 2])
        points = atlas.vertices_world[lo:hi]
        ax.scatter(points[:, axes[0]], points[:, axes[1]], s=1.2, alpha=0.6,
                   color=colors(chart % 10), rasterized=True)
    ax.set_title(title)
    labels = ("X", "Y", "Z")
    ax.set_xlabel(f"world {labels[axes[0]]} (m)")
    ax.set_ylabel(f"world {labels[axes[1]]} (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dav2", type=Path, required=True)
    parser.add_argument("--moge3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite chart visualization")
    dav2 = ExplicitChartAtlas.load_npz(args.dav2)
    moge3 = ExplicitChartAtlas.load_npz(args.moge3)
    if dav2.chart_names.tolist() != moge3.chart_names.tolist():
        raise ValueError("chart inventories differ")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _scatter(axes[0, 0], dav2, (0, 2), "DAV2 + masked MAtCha alignment (top view X–Z)")
    _scatter(axes[0, 1], moge3, (0, 2), "MoGe-3 + masked MAtCha alignment (top view X–Z)")
    _scatter(axes[1, 0], dav2, (0, 1), "DAV2 (X–Y)")
    _scatter(axes[1, 1], moge3, (0, 1), "MoGe-3 (X–Y)")
    all_points = np.concatenate((dav2.vertices_world, moge3.vertices_world))
    for row, dimensions in enumerate(((0, 2), (0, 1))):
        for dimension, axis_index in enumerate(dimensions):
            low, high = np.quantile(all_points[:, axis_index], (0.01, 0.99))
            for column in range(2):
                (axes[row, column].set_xlim if dimension == 0 else axes[row, column].set_ylim)(low, high)
    figure.suptitle("Explicit chart atlas: identical 8 seq4 keyframes, 1000 alignment iterations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    sidecar = {
        "artifact_type": "goal_maplet_chart_alignment_visualization_v1",
        "dav2_atlas": str(args.dav2),
        "moge3_atlas": str(args.moge3),
        "chart_count": len(dav2.chart_names),
        "plot_limits": "joint_1_to_99_percentile_for_outlier_robust_visual_comparison",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
