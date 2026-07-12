"""Fit a no-GT-at-inference support-view selector for RGB measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.support_selection import fit_measurement_support_selector


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_diagnostics_csv", required=True)
    parser.add_argument("--validation_diagnostics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--c_value", type=float, default=0.25)
    args = parser.parse_args(argv)
    summary = fit_measurement_support_selector(
        train_diagnostics_csv=Path(args.train_diagnostics_csv),
        validation_diagnostics_csv=Path(args.validation_diagnostics_csv),
        output_dir=Path(args.output_dir),
        c_value=float(args.c_value),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
