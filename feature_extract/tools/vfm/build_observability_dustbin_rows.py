"""Build measurement training rows with observability negatives marked as dustbin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.observability_training_rows import export_observability_dustbin_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--positive_classes", nargs="*", default=["observable_coarse_only", "observable_subpixel"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_observability_dustbin_rows(
        rows_csv=Path(args.rows_csv),
        output_csv=Path(args.output_csv),
        positive_classes=list(args.positive_classes),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
