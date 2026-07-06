"""Split measurement rows and optional render manifests by query id."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _query_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row.get("query_id", "")).strip() for row in rows if str(row.get("query_id", "")).strip()})


def _parse_query_id_file(path: Path | None) -> list[str]:
    if path is None or not str(path):
        return []
    values: list[str] = []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            values.append(text)
    return values


def split_measurement_rows_by_query(
    *,
    rows_csv: Path,
    output_dir: Path,
    val_query_count: int = 0,
    val_query_id_file: Path | None = None,
    render_cache_manifest_csv: Path | None = None,
) -> dict[str, Any]:
    row_fields, rows = _read_csv(Path(rows_csv))
    if "query_id" not in row_fields:
        raise ValueError("rows_csv must contain query_id")
    all_query_ids = _query_ids(rows)
    explicit_val = _parse_query_id_file(val_query_id_file)
    if explicit_val:
        val_query_ids = [query_id for query_id in explicit_val if query_id in set(all_query_ids)]
    else:
        count = int(val_query_count)
        if count <= 0:
            raise ValueError("val_query_count must be positive when val_query_id_file is not provided")
        if count >= len(all_query_ids):
            raise ValueError("val_query_count must be smaller than the number of unique queries")
        val_query_ids = all_query_ids[-count:]
    val_set = set(val_query_ids)
    train_set = set(all_query_ids) - val_set
    train_rows = [row for row in rows if str(row.get("query_id", "")).strip() in train_set]
    val_rows = [row for row in rows if str(row.get("query_id", "")).strip() in val_set]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows_csv = output / "train_rows.csv"
    val_rows_csv = output / "val_rows.csv"
    _write_csv(train_rows_csv, row_fields, train_rows)
    _write_csv(val_rows_csv, row_fields, val_rows)

    outputs: dict[str, str] = {
        "train_rows_csv": str(train_rows_csv),
        "val_rows_csv": str(val_rows_csv),
        "summary": str(output / "summary.json"),
    }
    manifest_train_count = None
    manifest_val_count = None
    if render_cache_manifest_csv is not None and str(render_cache_manifest_csv):
        manifest_fields, manifest_rows = _read_csv(Path(render_cache_manifest_csv))
        if "query_id" not in manifest_fields:
            raise ValueError("render_cache_manifest_csv must contain query_id")
        train_manifest = [row for row in manifest_rows if str(row.get("query_id", "")).strip() in train_set]
        val_manifest = [row for row in manifest_rows if str(row.get("query_id", "")).strip() in val_set]
        train_manifest_csv = output / "train_render_cache_manifest.csv"
        val_manifest_csv = output / "val_render_cache_manifest.csv"
        _write_csv(train_manifest_csv, manifest_fields, train_manifest)
        _write_csv(val_manifest_csv, manifest_fields, val_manifest)
        outputs["train_render_cache_manifest_csv"] = str(train_manifest_csv)
        outputs["val_render_cache_manifest_csv"] = str(val_manifest_csv)
        manifest_train_count = len(train_manifest)
        manifest_val_count = len(val_manifest)

    summary = {
        "stage": "split_measurement_rows_by_query",
        "rows_csv": str(rows_csv),
        "render_cache_manifest_csv": "" if render_cache_manifest_csv is None else str(render_cache_manifest_csv),
        "query_count": len(all_query_ids),
        "row_count": len(rows),
        "train_query_count": len(train_set),
        "val_query_count": len(val_set),
        "train_row_count": len(train_rows),
        "val_row_count": len(val_rows),
        "query_overlap_count": len(train_set & val_set),
        "train_query_ids": sorted(train_set),
        "val_query_ids": sorted(val_set),
        "manifest_train_count": manifest_train_count,
        "manifest_val_count": manifest_val_count,
        "outputs": outputs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_query_count", type=int, default=0)
    parser.add_argument("--val_query_id_file", default="")
    parser.add_argument("--render_cache_manifest_csv", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = split_measurement_rows_by_query(
        rows_csv=Path(args.rows_csv),
        output_dir=Path(args.output_dir),
        val_query_count=int(args.val_query_count),
        val_query_id_file=Path(args.val_query_id_file) if str(args.val_query_id_file) else None,
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv) if str(args.render_cache_manifest_csv) else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
