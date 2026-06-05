"""Export a VFM-2DGS anchor map as a SemiDenseAnchorMap for localization evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.semidense_anchor_map import semidense_anchor_map_stats
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsAnchorMap, vfm_2dgs_anchor_map_to_semidense


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export VFM-2DGS anchors to SemiDenseAnchorMap NPZ")
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--descriptor_mode",
        choices=("mean", "prototypes", "view_bins"),
        default="mean",
        help="Descriptor representation exported for localization: mean anchor feature, feature prototypes, or view bins.",
    )
    args = parser.parse_args(argv)

    anchor_map = Vfm2DgsAnchorMap.load_npz(Path(args.anchor_map))
    semidense = vfm_2dgs_anchor_map_to_semidense(anchor_map, descriptor_mode=str(args.descriptor_mode))
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semidense.save_npz(output_path)
    summary = {
        "stage": "vfm_2dgs_to_semidense_export",
        "descriptor_mode": str(args.descriptor_mode),
        "inputs": {"anchor_map": str(args.anchor_map)},
        "outputs": {"semidense_anchor_npz": str(args.output_npz), "summary": str(args.summary_json)},
        "anchor_count": int(len(semidense)),
        "source_anchor_count": int(len(anchor_map)),
        "feature_dim": int(semidense.feature_dim),
        "stats": semidense_anchor_map_stats(
            semidense,
            sparse_landmark_count=max(int(len(anchor_map)), 1),
            source_gaussian_count=max(int(len(anchor_map)), 1),
        ),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
