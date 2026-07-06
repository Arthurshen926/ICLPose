"""Build measurement-v1 RGB patch residual-delta training rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_training_rows import build_rgb_patch_measurement_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_rows_csv", required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--search_radius_px", type=float, default=2.0)
    parser.add_argument("--context_radius_px", type=float, default=8.0)
    parser.add_argument("--offsets_per_anchor", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_quality", type=float, default=0.0)
    parser.add_argument("--exclude_identity", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_rgb_patch_measurement_rows(
        anchor_rows_csv=Path(args.anchor_rows_csv),
        output_rows_csv=Path(args.output_rows_csv),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        offsets_per_anchor=int(args.offsets_per_anchor),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        seed=int(args.seed),
        min_quality=float(args.min_quality),
        include_identity=not bool(args.exclude_identity),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
