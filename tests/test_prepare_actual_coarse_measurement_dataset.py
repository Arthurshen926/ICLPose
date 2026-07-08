from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.prepare_actual_coarse_measurement_dataset import (
    prepare_actual_coarse_measurement_dataset,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_render_cache(path: Path, *, width: int = 8, height: int = 6) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.ones((height, width), dtype=np.float32)
    alpha = np.ones((height, width), dtype=np.float32)
    np.savez_compressed(path, rgb=rgb, depth=depth, alpha=alpha)


def test_prepare_actual_coarse_dataset_splits_by_query_and_filters_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for query_id in ("q0.png", "q1.png", "q2.png"):
        _write_render_cache(cache_dir / f"{query_id.replace('/', '__')}.npz")
    match_table = tmp_path / "match_table.csv"
    _write_csv(
        match_table,
        [
            {"query_id": "q0.png", "candidate_id": "0", "query_x": "1", "query_y": "2", "render_x": "1", "render_y": "2"},
            {"query_id": "q0.png", "candidate_id": "1", "query_x": "2", "query_y": "2", "render_x": "1", "render_y": "2"},
            {"query_id": "q1.png", "candidate_id": "0", "query_x": "3", "query_y": "4", "render_x": "1", "render_y": "2"},
            {"query_id": "q2.png", "candidate_id": "0", "query_x": "5", "query_y": "6", "render_x": "1", "render_y": "2"},
        ],
    )
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(
        manifest,
        [
            {"query_id": query_id, "rgb_depth_cache_path": str(cache_dir / f"{query_id.replace('/', '__')}.npz")}
            for query_id in ("q0.png", "q1.png", "q2.png")
        ],
    )

    summary = prepare_actual_coarse_measurement_dataset(
        match_table_csv=match_table,
        render_cache_manifest_csv=manifest,
        output_dir=tmp_path / "prepared",
        val_fraction=1.0 / 3.0,
        seed=3,
        base_dir=tmp_path,
    )

    assert summary["status"] == "complete"
    assert summary["raw_row_count"] == 4
    assert summary["query_count"] == 3
    assert summary["train_query_count"] == 2
    assert summary["val_query_count"] == 1
    train_rows = list(csv.DictReader(Path(summary["outputs"]["train_match_table_csv"]).open()))
    val_rows = list(csv.DictReader(Path(summary["outputs"]["val_match_table_csv"]).open()))
    assert {row["query_id"] for row in train_rows}.isdisjoint({row["query_id"] for row in val_rows})
    train_manifest = list(csv.DictReader(Path(summary["outputs"]["train_render_cache_manifest_csv"]).open()))
    val_manifest = list(csv.DictReader(Path(summary["outputs"]["val_render_cache_manifest_csv"]).open()))
    assert {row["query_id"] for row in train_manifest} == {row["query_id"] for row in train_rows}
    assert {row["query_id"] for row in val_manifest} == {row["query_id"] for row in val_rows}
    written_summary = json.loads((tmp_path / "prepared" / "summary.json").read_text())
    assert written_summary["cache_audit"]["missing_required_query_count"] == 0


def test_prepare_actual_coarse_dataset_rejects_missing_cache(tmp_path: Path) -> None:
    match_table = tmp_path / "match_table.csv"
    _write_csv(match_table, [{"query_id": "q0.png", "query_x": "1", "query_y": "2"}])
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(manifest, [{"query_id": "q0.png", "rgb_depth_cache_path": str(tmp_path / "missing.npz")}])

    with pytest.raises(ValueError, match="render cache manifest does not cover"):
        prepare_actual_coarse_measurement_dataset(
            match_table_csv=match_table,
            render_cache_manifest_csv=manifest,
            output_dir=tmp_path / "prepared",
            base_dir=tmp_path,
        )


def test_prepare_actual_coarse_dataset_rejects_forbidden_query_ids(tmp_path: Path) -> None:
    cache_path = tmp_path / "q0.npz"
    _write_render_cache(cache_path)
    match_table = tmp_path / "match_table.csv"
    _write_csv(match_table, [{"query_id": "q0.png", "query_x": "1", "query_y": "2"}])
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(manifest, [{"query_id": "q0.png", "rgb_depth_cache_path": str(cache_path)}])
    forbidden = tmp_path / "dataset_test.txt"
    forbidden.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q0.png 0 0 0 1 0 0 0\n"
    )

    with pytest.raises(ValueError, match="forbidden query ids"):
        prepare_actual_coarse_measurement_dataset(
            match_table_csv=match_table,
            render_cache_manifest_csv=manifest,
            output_dir=tmp_path / "prepared",
            forbidden_query_ids_file=forbidden,
            base_dir=tmp_path,
        )


def test_prepare_actual_coarse_dataset_limits_rows_per_query_before_split(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for query_id in ("q0.png", "q1.png"):
        _write_render_cache(cache_dir / f"{query_id}.npz")
    match_table = tmp_path / "match_table.csv"
    rows = []
    for query_id in ("q0.png", "q1.png"):
        for index in range(5):
            rows.append({"query_id": query_id, "match_index": index, "query_x": index, "query_y": index})
    _write_csv(match_table, rows)
    manifest = tmp_path / "render_cache_manifest.csv"
    _write_csv(
        manifest,
        [{"query_id": query_id, "rgb_depth_cache_path": str(cache_dir / f"{query_id}.npz")} for query_id in ("q0.png", "q1.png")],
    )

    summary = prepare_actual_coarse_measurement_dataset(
        match_table_csv=match_table,
        render_cache_manifest_csv=manifest,
        output_dir=tmp_path / "prepared_limited",
        val_query_count=1,
        max_rows_per_query=2,
        seed=7,
        base_dir=tmp_path,
    )

    assert summary["raw_row_count"] == 10
    assert summary["sampled_row_count"] == 4
    assert summary["max_rows_per_query"] == 2
    train_rows = list(csv.DictReader(Path(summary["outputs"]["train_match_table_csv"]).open()))
    val_rows = list(csv.DictReader(Path(summary["outputs"]["val_match_table_csv"]).open()))
    assert len(train_rows) == 2
    assert len(val_rows) == 2
