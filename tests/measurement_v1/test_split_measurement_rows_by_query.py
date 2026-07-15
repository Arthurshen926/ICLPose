from __future__ import annotations

import csv
from pathlib import Path

from feature_extract.tools.vfm.split_measurement_rows_by_query import split_measurement_rows_by_query


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_split_measurement_rows_by_query_writes_disjoint_rows_and_manifests(tmp_path: Path) -> None:
    rows_csv = tmp_path / "measurement_rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "match_index", "center_x"])
        writer.writeheader()
        for query_id in ["q0.png", "q1.png", "q2.png", "q3.png"]:
            writer.writerow({"query_id": query_id, "match_index": "0", "center_x": "1.0"})
            writer.writerow({"query_id": query_id, "match_index": "1", "center_x": "2.0"})

    manifest_csv = tmp_path / "render_cache_manifest.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rgb_depth_cache_path"])
        writer.writeheader()
        for query_id in ["q0.png", "q1.png", "q2.png", "q3.png"]:
            writer.writerow({"query_id": query_id, "rgb_depth_cache_path": f"{query_id}.npz"})

    summary = split_measurement_rows_by_query(
        rows_csv=rows_csv,
        output_dir=tmp_path / "split",
        val_query_count=1,
        render_cache_manifest_csv=manifest_csv,
    )

    assert summary["train_query_count"] == 3
    assert summary["val_query_count"] == 1
    assert summary["query_overlap_count"] == 0
    assert summary["val_query_ids"] == ["q3.png"]

    train_rows = _read_csv(Path(summary["outputs"]["train_rows_csv"]))
    val_rows = _read_csv(Path(summary["outputs"]["val_rows_csv"]))
    assert {row["query_id"] for row in train_rows} == {"q0.png", "q1.png", "q2.png"}
    assert {row["query_id"] for row in val_rows} == {"q3.png"}
    assert len(train_rows) == 6
    assert len(val_rows) == 2

    train_manifest = _read_csv(Path(summary["outputs"]["train_render_cache_manifest_csv"]))
    val_manifest = _read_csv(Path(summary["outputs"]["val_render_cache_manifest_csv"]))
    assert {row["query_id"] for row in train_manifest} == {"q0.png", "q1.png", "q2.png"}
    assert {row["query_id"] for row in val_manifest} == {"q3.png"}


def test_hash_query_split_is_stable_and_query_disjoint(tmp_path: Path) -> None:
    rows_csv = tmp_path / "measurement_rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "match_index"])
        writer.writeheader()
        for query_index in range(30):
            for match_index in range(2):
                writer.writerow(
                    {
                        "query_id": f"q{query_index:02d}.png",
                        "match_index": str(match_index),
                    }
                )

    first = split_measurement_rows_by_query(
        rows_csv=rows_csv,
        output_dir=tmp_path / "first",
        hash_folds=3,
        hash_val_fold=2,
        hash_salt="production-calibration",
    )
    second = split_measurement_rows_by_query(
        rows_csv=rows_csv,
        output_dir=tmp_path / "second",
        hash_folds=3,
        hash_val_fold=2,
        hash_salt="production-calibration",
    )

    assert first["split_strategy"] == "stable_sha256_query_fold"
    assert first["val_query_ids"] == second["val_query_ids"]
    assert set(first["train_query_ids"]).isdisjoint(first["val_query_ids"])
    assert first["train_row_count"] + first["val_row_count"] == 60
