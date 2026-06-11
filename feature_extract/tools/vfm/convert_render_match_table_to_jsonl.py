#!/usr/bin/env python3
"""Convert render-RGB match-table CSV diagnostics into typed JSONL rows."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Sequence


def _parse_scalar(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null", "nan"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    try:
        integer = int(text)
    except ValueError:
        integer = None
    if integer is not None and str(integer) == text:
        return integer
    try:
        floating = float(text)
    except ValueError:
        return text
    return floating if math.isfinite(floating) else None


def convert_match_table_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    typed_rows: list[dict[str, object]] = []
    for row in rows:
        typed_rows.append({str(key): _parse_scalar(value) for key, value in row.items()})
    return typed_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_csv", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args(argv)

    input_path = Path(args.match_csv)
    output_path = Path(args.output_jsonl)
    with input_path.open("r", newline="") as handle:
        rows = convert_match_table_rows(list(csv.DictReader(handle)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"wrote {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
