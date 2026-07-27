"""Build the V6 many-to-many retrieval-region ↔ metric-chart index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.map_entities import (
    MetricSurfaceChartBank,
    RetrievalRegionBank,
    build_region_chart_index,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_regions", required=True)
    parser.add_argument("--metric_charts", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--minimum_valid_fraction", type=float, default=0.01)
    parser.add_argument("--maximum_charts_per_region", type=int, default=12)
    parser.add_argument("--region_extent_multiplier", type=float, default=1.5)
    parser.add_argument("--minimum_normal_cosine", type=float, default=0.65)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_npz)
    summary = Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("output exists; pass --force to replace it")
    region_path = Path(args.retrieval_regions)
    chart_path = Path(args.metric_charts)
    regions = RetrievalRegionBank(
        SurfaceRetrievalMapletBank.load_npz(region_path)
    )
    charts = MetricSurfaceChartBank(
        MapletFeatureAtlasBank.load_npz(chart_path),
        minimum_valid_fraction=float(args.minimum_valid_fraction),
    )
    index = build_region_chart_index(
        regions,
        charts,
        maximum_charts_per_region=int(args.maximum_charts_per_region),
        region_extent_multiplier=float(args.region_extent_multiplier),
        minimum_normal_cosine=float(args.minimum_normal_cosine),
        metadata={
            "retrieval_region_sha256": _sha256(region_path),
            "metric_chart_sha256": _sha256(chart_path),
        },
    )
    index.save_npz(output)
    degrees = np.diff(index.region_offsets)
    inverse_degrees = np.diff(index.chart_offsets)
    texel = charts.physical_texel_size_m[charts.usable_rows]
    payload = {
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "retrieval_region_count": int(index.region_ids.size),
        "metric_chart_count": int(index.chart_ids.size),
        "edge_count": int(index.edge_count),
        "regions_with_multiple_charts": int(np.sum(degrees > 1)),
        "charts_recalled_by_multiple_regions": int(
            np.sum(inverse_degrees > 1)
        ),
        "charts_per_region_percentiles": np.percentile(
            degrees, [10, 50, 90]
        ).tolist(),
        "regions_per_chart_percentiles": np.percentile(
            inverse_degrees, [10, 50, 90]
        ).tolist(),
        "valid_fraction_percentiles": np.percentile(
            charts.valid_fraction[charts.usable_rows], [10, 50, 90]
        ).tolist(),
        "physical_texel_size_cm_percentiles": np.percentile(
            texel.reshape(-1) * 100.0, [10, 50, 90]
        ).tolist(),
        "contract": dict(index.metadata or {}),
        "representation_warning": (
            "geometry-overlap adjacency is not a learned context-rich "
            "retrieval-region descriptor"
        ),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
