from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.measurement_v1.dense_alignment_teacher import (
    _center_preserving_hybrid_epe,
    _oracle_gated_epe,
    _threshold_gated_epe,
    patch_observability,
    export_dense_alignment_teacher_diagnostics,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gaussian_image(size: int, center_xy: tuple[float, float], *, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = center_xy
    blob = np.exp(-((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / (2.0 * float(sigma) ** 2))
    texture = 0.25 * np.sin(xx * 0.45) + 0.15 * np.cos(yy * 0.35)
    values = np.clip(0.25 + 0.65 * blob + 0.10 * texture, 0.0, 1.0)
    rgb = np.stack([values, values, values], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def test_dense_alignment_teacher_recovers_subpixel_translation(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    support_xy = (16.0, 16.0)
    query_gt = (18.35, 14.80)
    center = (17.35, 15.80)
    Image.fromarray(_gaussian_image(40, support_xy)).save(image_root / "seq0" / "support.png")
    Image.fromarray(_gaussian_image(40, query_gt)).save(image_root / "seq0" / "query.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": support_xy[0],
                "support_y": support_xy[1],
                "center_x": center[0],
                "center_y": center[1],
                "query_gt_x": query_gt[0],
                "query_gt_y": query_gt[1],
                "requested_residual_px": 1.414,
                "target_is_dustbin": "False",
            }
        ],
    )

    summary = export_dense_alignment_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher",
        win_size_px=15,
        max_level=0,
        max_lk_error=20.0,
        max_flow_from_center_px=4.0,
        fb_max_error_px=0.75,
    )

    assert summary["applied_count"] == 1
    assert summary["metrics"]["lk_applied_median_px"] < 0.35
    assert summary["metrics"]["lk_fallback_improve_ratio"] == 1.0
    row = next(csv.DictReader((tmp_path / "teacher" / "teacher_rows.csv").open()))
    assert row["lk_applied"] == "True"
    assert float(row["lk_epe_px"]) < 0.35
    assert float(row["baseline_epe_px"]) > 1.0


def test_dense_alignment_teacher_rejects_flat_texture_and_falls_back(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    Image.fromarray(flat).save(image_root / "seq0" / "support.png")
    Image.fromarray(flat).save(image_root / "seq0" / "query.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": 16.0,
                "support_y": 16.0,
                "center_x": 15.0,
                "center_y": 16.0,
                "query_gt_x": 16.0,
                "query_gt_y": 16.0,
                "requested_residual_px": 1.0,
                "target_is_dustbin": "False",
            }
        ],
    )

    summary = export_dense_alignment_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher",
        win_size_px=15,
        max_level=0,
        min_eig_threshold=1e-3,
    )

    assert summary["applied_count"] == 0
    assert summary["metrics"]["lk_applied_median_px"] is None
    assert summary["metrics"]["lk_fallback_median_px"] == 1.0
    row = next(csv.DictReader((tmp_path / "teacher" / "teacher_rows.csv").open()))
    assert row["lk_applied"] == "False"
    assert row["lk_reason"] != "applied"


def test_oracle_gated_epe_uses_lk_only_when_it_improves_center() -> None:
    values = _oracle_gated_epe(
        baseline_values=[0.5, 1.0, 2.0, 3.0],
        applied_values=[0.8, None, 0.7, 4.0],
    )

    np.testing.assert_allclose(values, [0.5, 1.0, 0.7, 3.0], atol=1e-6)


def test_center_preserving_hybrid_keeps_micro_residuals_at_center() -> None:
    values = _center_preserving_hybrid_epe(
        baseline_values=[0.5, 1.0, 2.0, 3.0],
        applied_values=[0.2, 0.4, 0.7, None],
        requested_residual_values=[0.5, 1.0, 2.0, 3.0],
        preserve_below_px=1.0,
    )

    np.testing.assert_allclose(values, [0.5, 1.0, 0.7, 3.0], atol=1e-6)


def test_patch_observability_reports_texture_and_ncc() -> None:
    support = np.zeros((33, 33), dtype=np.uint8)
    query = np.zeros((33, 33), dtype=np.uint8)
    support[10:23, 14:19] = 255
    query[10:23, 14:19] = 255

    textured = patch_observability(
        support_gray=support,
        query_gray=query,
        support_xy=np.asarray([16.0, 16.0]),
        query_xy=np.asarray([16.0, 16.0]),
        radius_px=8,
    )
    flat = patch_observability(
        support_gray=np.full((33, 33), 128, dtype=np.uint8),
        query_gray=np.full((33, 33), 128, dtype=np.uint8),
        support_xy=np.asarray([16.0, 16.0]),
        query_xy=np.asarray([16.0, 16.0]),
        radius_px=8,
    )

    assert textured["support_grad_mean"] > flat["support_grad_mean"]
    assert textured["query_grad_mean"] > flat["query_grad_mean"]
    assert textured["patch_ncc"] > 0.99
    assert textured["photometric_mae"] == 0.0
    assert flat["patch_ncc"] is None


def test_threshold_gated_epe_can_use_greater_equal_texture_scores() -> None:
    values = _threshold_gated_epe(
        baseline_values=[1.0, 1.0, 1.0],
        applied_values=[0.2, 0.3, 0.4],
        score_values=[4.0, 8.0, None],
        threshold=5.0,
        direction="ge",
    )

    np.testing.assert_allclose(values, [1.0, 0.3, 1.0], atol=1e-6)
