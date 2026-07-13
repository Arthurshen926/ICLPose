"""Apply a frozen candidate-specific RGB coordinate update verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_update_verifier import (
    apply_candidate_update_verifier,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model_path",
        "candidate_geometry_verifier_json",
        "diagnostic_rows_csv",
        "output_path",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_candidate_update_verifier(
        model_path=Path(args.model_path),
        candidate_geometry_verifier_json=Path(
            args.candidate_geometry_verifier_json
        ),
        diagnostic_rows_csv=Path(args.diagnostic_rows_csv),
        output_path=Path(args.output_path),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
