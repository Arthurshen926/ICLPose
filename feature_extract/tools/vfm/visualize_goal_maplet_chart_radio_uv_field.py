"""Visualize coordinate and multi-view evidence in a chart RADIO UV smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_radio_uv_field import (
    CanonicalChartRadioField,
    SourceViewChartRadioField,
)


def visualize_chart_radio_uv_field(
    source: SourceViewChartRadioField,
    canonical: CanonicalChartRadioField,
    output_png: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    view = 0
    t0, t1 = map(int, source.view_token_offsets[view:view + 2])
    height, width = map(int, source.view_token_shapes[view])
    raw_uv = source.token_xy[t0:t1].astype(np.float64)
    raw_uv /= np.asarray([width - 1.0, height - 1.0])
    ideal_uv = source.token_chart_uv[t0:t1]
    axes[0, 0].quiver(
        raw_uv[:, 0], raw_uv[:, 1], ideal_uv[:, 0] - raw_uv[:, 0],
        ideal_uv[:, 1] - raw_uv[:, 1], angles="xy", scale_units="xy", scale=1,
        width=0.0015, color="#3366aa", alpha=0.8,
    )
    axes[0, 0].set(
        title=f"raw RADIO grid → ideal chart UV\n{source.view_names[view]}",
        xlabel="u", ylabel="v", xlim=(-0.02, 1.02), ylim=(1.02, -0.02), aspect="equal",
    )

    v0, v1 = map(int, source.chart_vertex_offsets[view:view + 2])
    camera = source.raw_camera_parameters[view]
    chart_uv = source.vertex_raw_xy[v0:v1] / np.asarray([camera[0] - 1.0, camera[1] - 1.0])
    scatter = axes[0, 1].scatter(
        chart_uv[:, 0], chart_uv[:, 1], c=source.vertex_observation_weight[v0:v1],
        cmap="viridis", vmin=0.0, vmax=1.0, s=14,
    )
    figure.colorbar(scatter, ax=axes[0, 1], label="geometry × facing × coordinate weight")
    axes[0, 1].set(
        title="chart vertices in raw-RADIO coordinates", xlabel="raw u", ylabel="raw v",
        xlim=(-0.02, 1.02), ylim=(1.02, -0.02), aspect="equal",
    )

    same = []
    for node in np.flatnonzero(np.diff(canonical.prototype_offsets) >= 2):
        begin = int(canonical.prototype_offsets[node])
        same.append(float(canonical.prototype_codes[begin] @ canonical.prototype_codes[begin + 1]))
    observed = np.flatnonzero(canonical.view_count >= 1)
    rng = np.random.default_rng(20260830)
    pair = rng.choice(observed, size=(4096, 2), replace=True)
    pair = pair[pair[:, 0] != pair[:, 1]]
    random = np.sum(canonical.codes[pair[:, 0]] * canonical.codes[pair[:, 1]], axis=1)
    bins = np.linspace(-0.2, 1.0, 40)
    axes[1, 0].hist(random, bins=bins, density=True, alpha=0.65, label=f"random nodes (median {np.median(random):.3f})")
    axes[1, 0].hist(same, bins=bins, density=True, alpha=0.75, label=f"same node/cross view (median {np.median(same):.3f})")
    axes[1, 0].set(title="RADIO identity sanity check", xlabel="cosine", ylabel="density")
    axes[1, 0].legend()

    values, counts = np.unique(canonical.view_count, return_counts=True)
    axes[1, 1].bar(values.astype(str), counts, color="#3a7d44")
    axes[1, 1].set(
        title="canonical node source-view support", xlabel="unique source views", ylabel="node count",
    )
    axes[1, 1].text(
        0.98, 0.98,
        f"nodes: {len(canonical.view_count):,}\n"
        f"multi-view: {np.sum(canonical.view_count >= 2):,}\n"
        f"source tokens: {len(source.token_codes):,} × {source.feature_dim}\n"
        "diagnostic family layout; not deployment GO",
        transform=axes[1, 1].transAxes, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_field", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()
    source = SourceViewChartRadioField.load_npz(Path(args.source_field))
    canonical = CanonicalChartRadioField.load_npz(Path(args.canonical_field))
    visualize_chart_radio_uv_field(source, canonical, Path(args.output_png))


if __name__ == "__main__":
    main()
