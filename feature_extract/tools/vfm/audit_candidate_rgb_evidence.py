"""Audit candidate rescue from already exported RGB diagnostic rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.rgb_patch_diagnostics import (
    _candidate_geometry_evidence_metrics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.diagnostic_rows_csv)
    with source.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    report = {
        "stage": "candidate_rgb_evidence_audit",
        "diagnostic_rows_csv": str(source),
        "diagnostic_rows_sha256": file_sha256_short(source),
        "row_count": int(len(rows)),
        "candidate_geometry_evidence": _candidate_geometry_evidence_metrics(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
