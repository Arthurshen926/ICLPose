"""Build one strict, target-free neighbour-topology cache for sparse maplets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_multiscale_translation_mode import (
    _load_translation_sources,
)
from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_sparse_maplet_transport import (
    _maplet_neighbor_topologies,
    write_sparse_maplet_neighbor_topology_cache,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-pca256-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def build_sparse_maplet_topology_cache(
    *,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
) -> dict[str, object]:
    geometry_path = Path(support_geometry_index)
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    sources = _load_translation_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
    )
    topologies = _maplet_neighbor_topologies(geometry=geometry, sources=sources)
    return write_sparse_maplet_neighbor_topology_cache(
        output=Path(output),
        geometry_path=geometry_path,
        geometry_metadata=geometry_metadata,
        sources=sources,
        neighbour_topologies=topologies,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_sparse_maplet_topology_cache(
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
