#!/usr/bin/env python3
"""Apply a measurement utility gate without loading pose or ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_measurement_utility import (
    apply_candidate_measurement_utility,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--spatial_likelihood",
        action="append",
        required=True,
        help="Repeat only when the split was inferred in deterministic query shards.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_candidate_measurement_utility(
        model_path=Path(args.model),
        spatial_paths=[Path(path) for path in args.spatial_likelihood],
        output_path=Path(args.output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
