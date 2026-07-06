"""Audit measurement-v1 protocol invariants from existing CSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.protocol_audit import audit_zero_perturbation_consistency


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_rows", required=True, help="P1 GT-render rows.csv")
    parser.add_argument("--zero_rows", required=True, help="P3 zero-perturbation rows.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_zero_perturbation_consistency(
        reference_rows_csv=Path(args.reference_rows),
        zero_rows_csv=Path(args.zero_rows),
        output_dir=Path(args.output_dir),
        tolerance=float(args.tolerance),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
