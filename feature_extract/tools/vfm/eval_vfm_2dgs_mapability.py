"""Evaluate VFM-2DGS maps without running localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.vfm_2dgs_mapability import evaluate_vfm_2dgs_mapability
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap, Vfm2DgsObservationBank


def _parse_grid(value: str) -> tuple[int, int] | None:
    text = str(value).strip()
    if not text:
        return None
    if "x" in text:
        left, right = text.lower().split("x", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("token grid must be HxW or H,W")
    height, width = int(left), int(right)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("token grid dimensions must be positive")
    return height, width


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mapping-only VFM-2DGS diagnostics")
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--observation_bank", required=True)
    parser.add_argument("--surface_elements", default="")
    parser.add_argument("--surface_seed_min_views", type=int, default=2)
    parser.add_argument("--token_grid", default="", help="optional HxW token grid shape")
    parser.add_argument("--min_support_iou", type=float, default=1e-6)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)

    anchor_map = Vfm2DgsAnchorMap.load_npz(Path(args.anchor_map))
    observation_bank = Vfm2DgsObservationBank.load_npz(Path(args.observation_bank))
    surface_elements = SurfaceElementMap.load_npz(Path(args.surface_elements)) if args.surface_elements else None
    result = evaluate_vfm_2dgs_mapability(
        anchor_map,
        observation_bank,
        token_grid_shape=_parse_grid(args.token_grid),
        min_support_iou=float(args.min_support_iou),
        surface_elements=surface_elements,
        surface_seed_min_views=int(args.surface_seed_min_views),
    )
    result["inputs"] = {
        "anchor_map": str(args.anchor_map),
        "observation_bank": str(args.observation_bank),
        "surface_elements": str(args.surface_elements),
        "token_grid": str(args.token_grid),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
