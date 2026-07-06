from __future__ import annotations

import csv
from pathlib import Path

import pytest

from feature_extract.vfm.measurement_v1.observability_gate import (
    DEFAULT_OBSERVABILITY_GATE_FEATURES,
    FORBIDDEN_OBSERVABILITY_GATE_FEATURES,
    load_observability_gate_dataset,
    train_observability_gate,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(label: str, *, mode_probability: float, entropy_norm: float, peak_gap_z: float, mode_epe_px: float, center_residual_px: float) -> dict[str, object]:
    return {
        "query_id": "seq/frame.png",
        "observability_class": label,
        "mode_probability": mode_probability,
        "entropy_norm": entropy_norm,
        "peak_gap_z": peak_gap_z,
        "logit_std": peak_gap_z + 0.5,
        "render_texture": 0.4 if "observable" in label else 0.02,
        "radio_match_score": "",
        "mode_dx": 0.25 if "observable" in label else 12.0,
        "mode_dy": -0.25 if "observable" in label else -8.0,
        "mode_epe_px": mode_epe_px,
        "center_residual_px": center_residual_px,
        "gt_rank": 1 if "observable" in label else 99,
        "target_dx": 0.0,
        "target_dy": 0.0,
    }


def test_load_observability_gate_dataset_uses_only_inference_available_features(tmp_path: Path) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            _row("observable_subpixel", mode_probability=0.95, entropy_norm=0.05, peak_gap_z=5.0, mode_epe_px=0.2, center_residual_px=2.0),
            _row("ambiguous", mode_probability=0.15, entropy_norm=0.98, peak_gap_z=0.1, mode_epe_px=12.0, center_residual_px=2.0),
        ],
    )

    dataset = load_observability_gate_dataset(rows_csv)

    assert dataset.feature_names == list(DEFAULT_OBSERVABILITY_GATE_FEATURES)
    assert dataset.features.shape == (2, len(DEFAULT_OBSERVABILITY_GATE_FEATURES))
    assert dataset.labels.tolist() == [1.0, 0.0]
    assert float(dataset.features[0, dataset.feature_names.index("radio_match_score")]) == 0.0
    assert float(dataset.features[0, dataset.feature_names.index("mode_abs_dx")]) == pytest.approx(0.25)
    assert not (set(dataset.feature_names) & FORBIDDEN_OBSERVABILITY_GATE_FEATURES)

    with pytest.raises(ValueError, match="GT-derived"):
        load_observability_gate_dataset(rows_csv, feature_names=["mode_probability", "mode_epe_px"])


def test_train_observability_gate_scores_and_exports_accepted_rows(tmp_path: Path) -> None:
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for index in range(30):
        train_rows.append(
            _row(
                "observable_subpixel" if index % 2 == 0 else "observable_coarse_only",
                mode_probability=0.92 - 0.001 * index,
                entropy_norm=0.08,
                peak_gap_z=5.0,
                mode_epe_px=0.4,
                center_residual_px=4.0,
            )
        )
        train_rows.append(
            _row(
                "ambiguous" if index % 2 == 0 else "window_out",
                mode_probability=0.12 + 0.001 * index,
                entropy_norm=0.96,
                peak_gap_z=0.1,
                mode_epe_px=10.0,
                center_residual_px=4.0,
            )
        )
    for index in range(8):
        val_rows.append(
            _row(
                "observable_subpixel",
                mode_probability=0.90 - 0.001 * index,
                entropy_norm=0.10,
                peak_gap_z=4.0,
                mode_epe_px=0.3,
                center_residual_px=3.0,
            )
        )
        val_rows.append(
            _row(
                "domain_mismatch",
                mode_probability=0.10 + 0.001 * index,
                entropy_norm=0.30,
                peak_gap_z=3.0,
                mode_epe_px=9.0,
                center_residual_px=3.0,
            )
        )
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(val_csv, val_rows)

    summary = train_observability_gate(
        train_rows_csv=train_csv,
        val_rows_csv=val_csv,
        output_dir=tmp_path / "gate",
        steps=120,
        batch_size=16,
        lr=0.05,
        hidden_dim=0,
        target_precision=0.90,
        device="cpu",
        seed=3,
    )

    assert summary["threshold_selection"]["target_precision"] == pytest.approx(0.90)
    assert summary["val_metrics"]["auroc"] == pytest.approx(1.0)
    assert summary["val_metrics"]["accepted_precision"] >= 0.90
    assert summary["val_metrics"]["accepted_mode_epe_median_px"] <= 0.5
    assert summary["val_metrics"]["accepted_mode_improve_ratio"] == pytest.approx(1.0)
    assert Path(summary["outputs"]["checkpoint"]).exists()
    accepted_rows = list(csv.DictReader(Path(summary["outputs"]["accepted_val_rows_csv"]).open()))
    scored_rows = list(csv.DictReader(Path(summary["outputs"]["scored_val_rows_csv"]).open()))
    assert len(accepted_rows) == summary["val_metrics"]["accepted_count"]
    assert len(scored_rows) == len(val_rows)
    assert {"observability_prob", "observability_accept"}.issubset(scored_rows[0].keys())
