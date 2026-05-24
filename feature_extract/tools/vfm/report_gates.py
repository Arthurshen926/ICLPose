"""Build a VFM gate report from metric rows.

This module intentionally keeps the reporting surface simple: experiment
runners should write normalized JSON/CSV rows, then this command formats one
gate at a time without mixing protocol kinds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.reporting import build_gate_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a VFM-MapLoc gate table")
    parser.add_argument("--gate", required=True, help="Gate name, e.g. Feature Utility")
    parser.add_argument("--rows", required=True, help="JSON file containing a list of row objects")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    rows = json.loads(Path(args.rows).read_text())
    table = build_gate_table(args.gate, rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(table + "\n")


if __name__ == "__main__":
    main()
