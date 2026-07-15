"""Materialize the strict target-free input boundary for RGB measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_rgb_inference import (
    materialize_candidate_rgb_inference_inputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--test_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = materialize_candidate_rgb_inference_inputs(
        candidate_evidence=Path(args.candidate_evidence),
        rows_by_split={
            "train": Path(args.train_rows_csv),
            "validation": Path(args.validation_rows_csv),
            "test": Path(args.test_rows_csv),
        },
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
