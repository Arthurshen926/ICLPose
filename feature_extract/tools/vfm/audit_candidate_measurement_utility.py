#!/usr/bin/env python3
"""Externally audit frozen target-free measurement utility predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_measurement_utility import (
    audit_candidate_measurement_utility,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--spatial_likelihood", action="append", required=True
    )
    parser.add_argument("--supervision_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_candidate_measurement_utility(
        predictions_path=Path(args.predictions),
        spatial_paths=[Path(path) for path in args.spatial_likelihood],
        supervision_rows_csv=Path(args.supervision_rows_csv),
        output_path=Path(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
