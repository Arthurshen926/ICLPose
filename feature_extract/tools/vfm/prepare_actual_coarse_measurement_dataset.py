"""Prepare cache-consistent actual-coarse measurement train/val splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _read_csv(path: Path, max_rows: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return fieldnames, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, preferred_fieldnames: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    fieldnames: list[str] = []
    for key in preferred_fieldnames:
        if str(key) not in seen:
            seen.add(str(key))
            fieldnames.append(str(key))
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _parse_pose_query_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or text.startswith("Visual ") or text.startswith("ImageFile"):
            continue
        out.add(text.split()[0])
    return out


def _manifest_by_query(
    manifest_csv: Path,
    *,
    base_dir: Path,
    inspect_npz_shapes: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    fieldnames, rows = _read_csv(Path(manifest_csv))
    by_query: dict[str, dict[str, str]] = {}
    duplicate_count = 0
    blank_path_count = 0
    missing_path_count = 0
    invalid_npz_count = 0
    shape_counts: dict[str, int] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        path_text = str(row.get("rgb_depth_cache_path", "")).strip()
        if not path_text:
            blank_path_count += 1
            continue
        resolved = _resolve_path(path_text, base_dir=base_dir)
        if not resolved.exists():
            missing_path_count += 1
            continue
        if inspect_npz_shapes:
            try:
                with np.load(resolved) as data:
                    rgb = np.asarray(data["rgb"])
                    depth = np.asarray(data["depth"])
                shape_key = f"{int(rgb.shape[1])}x{int(rgb.shape[0])}/depth:{int(depth.shape[1])}x{int(depth.shape[0])}"
                shape_counts[shape_key] = int(shape_counts.get(shape_key, 0) + 1)
            except Exception:
                invalid_npz_count += 1
                continue
        if query_id in by_query:
            duplicate_count += 1
        by_query[query_id] = dict(row)
    return by_query, {
        "manifest_csv": str(manifest_csv),
        "manifest_fieldnames": fieldnames,
        "manifest_row_count": int(len(rows)),
        "available_cache_count": int(len(by_query)),
        "blank_path_count": int(blank_path_count),
        "missing_path_count": int(missing_path_count),
        "duplicate_query_count": int(duplicate_count),
        "invalid_npz_count": int(invalid_npz_count),
        "npz_shape_counts": shape_counts,
    }


def _query_split(query_ids: Sequence[str], *, val_fraction: float, seed: int, val_query_count: int | None = None) -> tuple[set[str], set[str]]:
    queries = sorted(str(item) for item in query_ids)
    if not queries:
        return set(), set()
    rng = random.Random(int(seed))
    rng.shuffle(queries)
    if val_query_count is None:
        fraction = max(0.0, min(1.0, float(val_fraction)))
        count = int(round(float(len(queries)) * fraction))
    else:
        count = int(val_query_count)
    if len(queries) == 1:
        count = 0
    else:
        count = max(1, count) if (float(val_fraction) > 0.0 or val_query_count is not None) else 0
        count = min(count, len(queries) - 1)
    val = set(queries[:count])
    train = set(queries[count:])
    return train, val


def _residual_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    values = []
    for row in rows:
        text = str(row.get("gt_reproj_error_px", "")).strip()
        if text:
            try:
                value = float(text)
            except ValueError:
                continue
            if np.isfinite(value):
                values.append(value)
    if not values:
        return {"count": 0, "median_px": None, "p90_px": None, "le_2px_rate": None, "le_5px_rate": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "median_px": float(np.percentile(arr, 50.0)),
        "p90_px": float(np.percentile(arr, 90.0)),
        "le_2px_rate": float(np.mean(arr <= 2.0)),
        "le_5px_rate": float(np.mean(arr <= 5.0)),
    }


def _candidate_group_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], int] = {}
    missing_candidate_count = 0
    for index, row in enumerate(rows):
        query_id = str(row.get("query_id", "")).strip()
        candidate = str(row.get("pose_candidate_id", "") or row.get("render_pose_id", "") or row.get("initial_render_pose_label", "")).strip()
        if not candidate:
            candidate = str(row.get("candidate_id", "")).strip()
        if not candidate:
            missing_candidate_count += 1
            candidate = "single_pose"
        groups[(query_id, candidate)] = int(groups.get((query_id, candidate), 0) + 1)
    group_sizes = np.asarray(list(groups.values()), dtype=np.float64) if groups else np.asarray([], dtype=np.float64)
    return {
        "group_count": int(len(groups)),
        "missing_candidate_count": int(missing_candidate_count),
        "group_size_median": None if group_sizes.size == 0 else float(np.percentile(group_sizes, 50.0)),
        "group_size_p10": None if group_sizes.size == 0 else float(np.percentile(group_sizes, 10.0)),
        "group_size_p90": None if group_sizes.size == 0 else float(np.percentile(group_sizes, 90.0)),
    }


def prepare_actual_coarse_measurement_dataset(
    *,
    match_table_csv: Path,
    render_cache_manifest_csv: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    val_query_count: int | None = None,
    max_rows: int | None = None,
    max_rows_per_query: int | None = None,
    seed: int = 0,
    forbidden_query_ids_file: Path | None = None,
    base_dir: Path | None = None,
    inspect_npz_shapes: bool = False,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    match_fieldnames, rows = _read_csv(Path(match_table_csv), max_rows=max_rows)
    if not rows:
        raise ValueError("match_table_csv contains no rows")
    raw_row_count = int(len(rows))
    if max_rows_per_query is not None and int(max_rows_per_query) > 0:
        grouped_rows: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            query_id = str(row.get("query_id", "")).strip()
            if query_id:
                grouped_rows.setdefault(query_id, []).append(dict(row))
        rng = random.Random(int(seed))
        sampled_rows: list[dict[str, str]] = []
        for query_id in sorted(grouped_rows):
            query_rows = list(grouped_rows[query_id])
            rng.shuffle(query_rows)
            sampled_rows.extend(query_rows[: int(max_rows_per_query)])
        rows = sampled_rows
    query_ids = sorted({str(row.get("query_id", "")).strip() for row in rows if str(row.get("query_id", "")).strip()})
    if not query_ids:
        raise ValueError("match_table_csv contains no query_id values")
    forbidden_query_ids: set[str] = set()
    if forbidden_query_ids_file is not None and str(forbidden_query_ids_file):
        forbidden_query_ids = _parse_pose_query_ids(Path(forbidden_query_ids_file))
    forbidden_overlap = sorted(set(query_ids).intersection(forbidden_query_ids))
    if forbidden_overlap:
        summary = {
            "stage": "actual_coarse_measurement_dataset_prepare",
            "status": "blocked_forbidden_query_ids",
            "forbidden_query_id_count": int(len(forbidden_overlap)),
            "forbidden_query_id_examples": forbidden_overlap[:10],
            "outputs": {"summary_json": str(output / "summary.json")},
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        raise ValueError(f"match table contains forbidden query ids: {forbidden_overlap[:5]}")

    manifest_by_query, manifest_audit = _manifest_by_query(
        Path(render_cache_manifest_csv),
        base_dir=base,
        inspect_npz_shapes=bool(inspect_npz_shapes),
    )
    missing_required = sorted(query_id for query_id in query_ids if query_id not in manifest_by_query)
    cache_audit = dict(manifest_audit)
    cache_audit.update(
        {
            "required_query_count": int(len(query_ids)),
            "missing_required_query_count": int(len(missing_required)),
            "missing_required_query_examples": missing_required[:10],
        }
    )
    if missing_required:
        summary = {
            "stage": "actual_coarse_measurement_dataset_prepare",
            "status": "blocked_missing_render_cache",
            "match_table_csv": str(match_table_csv),
            "render_cache_manifest_csv": str(render_cache_manifest_csv),
            "cache_audit": cache_audit,
            "outputs": {"summary_json": str(output / "summary.json")},
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        raise ValueError("render cache manifest does not cover required query ids")

    train_queries, val_queries = _query_split(
        query_ids,
        val_fraction=float(val_fraction),
        seed=int(seed),
        val_query_count=val_query_count,
    )
    train_rows = [dict(row) for row in rows if str(row.get("query_id", "")).strip() in train_queries]
    val_rows = [dict(row) for row in rows if str(row.get("query_id", "")).strip() in val_queries]
    train_manifest_rows = [manifest_by_query[query_id] for query_id in sorted(train_queries)]
    val_manifest_rows = [manifest_by_query[query_id] for query_id in sorted(val_queries)]

    train_match = output / "train_match_table.csv"
    val_match = output / "val_match_table.csv"
    train_manifest = output / "train_render_cache_manifest.csv"
    val_manifest = output / "val_render_cache_manifest.csv"
    _write_csv(train_match, train_rows, preferred_fieldnames=match_fieldnames)
    _write_csv(val_match, val_rows, preferred_fieldnames=match_fieldnames)
    _write_csv(train_manifest, train_manifest_rows, preferred_fieldnames=manifest_audit.get("manifest_fieldnames", []))
    _write_csv(val_manifest, val_manifest_rows, preferred_fieldnames=manifest_audit.get("manifest_fieldnames", []))

    summary = {
        "stage": "actual_coarse_measurement_dataset_prepare",
        "status": "complete",
        "match_table_csv": str(match_table_csv),
        "render_cache_manifest_csv": str(render_cache_manifest_csv),
        "forbidden_query_ids_file": "" if forbidden_query_ids_file is None else str(forbidden_query_ids_file),
        "raw_row_count": int(raw_row_count),
        "sampled_row_count": int(len(rows)),
        "max_rows": None if max_rows is None else int(max_rows),
        "max_rows_per_query": None if max_rows_per_query is None else int(max_rows_per_query),
        "query_count": int(len(query_ids)),
        "train_row_count": int(len(train_rows)),
        "val_row_count": int(len(val_rows)),
        "train_query_count": int(len(train_queries)),
        "val_query_count": int(len(val_queries)),
        "val_fraction": float(val_fraction),
        "requested_val_query_count": None if val_query_count is None else int(val_query_count),
        "seed": int(seed),
        "cache_audit": cache_audit,
        "candidate_group_stats": _candidate_group_stats(rows),
        "train_candidate_group_stats": _candidate_group_stats(train_rows),
        "val_candidate_group_stats": _candidate_group_stats(val_rows),
        "residual_stats": _residual_stats(rows),
        "train_residual_stats": _residual_stats(train_rows),
        "val_residual_stats": _residual_stats(val_rows),
        "outputs": {
            "train_match_table_csv": str(train_match),
            "val_match_table_csv": str(val_match),
            "train_render_cache_manifest_csv": str(train_manifest),
            "val_render_cache_manifest_csv": str(val_manifest),
            "summary_json": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_query_count", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_rows_per_query", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forbidden_query_ids_file", default="")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--inspect_npz_shapes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = prepare_actual_coarse_measurement_dataset(
        match_table_csv=Path(args.match_table_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv),
        output_dir=Path(args.output_dir),
        val_fraction=float(args.val_fraction),
        val_query_count=int(args.val_query_count) if int(args.val_query_count) > 0 else None,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        max_rows_per_query=int(args.max_rows_per_query) if int(args.max_rows_per_query) > 0 else None,
        seed=int(args.seed),
        forbidden_query_ids_file=Path(args.forbidden_query_ids_file) if str(args.forbidden_query_ids_file) else None,
        base_dir=Path(args.base_dir),
        inspect_npz_shapes=bool(args.inspect_npz_shapes),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
