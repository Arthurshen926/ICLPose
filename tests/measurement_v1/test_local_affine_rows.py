from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.measurement_v1.local_affine_rows import (
    augment_measurement_rows_with_local_affine_from_observations,
    augment_measurement_rows_with_local_homography_from_observations,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_augment_rows_with_local_affine_estimates_support_to_query_matrix(tmp_path: Path) -> None:
    support_id = "seq1/frame00010.png"
    query_id = "seq1/frame00011.png"
    support_center = np.asarray([20.0, 20.0], dtype=np.float64)
    query_center = np.asarray([30.0, 28.0], dtype=np.float64)
    matrix = np.asarray([[1.2, 0.1], [-0.2, 0.9]], dtype=np.float64)
    offsets = [
        (0.0, 0.0),
        (2.0, 0.0),
        (0.0, 2.0),
        (-2.0, 0.0),
        (0.0, -2.0),
        (1.5, 1.0),
    ]
    observations: list[ColmapTrackObservation] = []
    for idx, offset in enumerate(offsets):
        support_xy = support_center + np.asarray(offset, dtype=np.float64)
        query_xy = query_center + matrix @ np.asarray(offset, dtype=np.float64)
        observations.append(
            ColmapTrackObservation(idx, support_id, 0, tuple(support_xy), np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
        observations.append(
            ColmapTrackObservation(idx, query_id, 0, tuple(query_xy), np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
    rows_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "affine_rows.csv"
    _write_rows(
        rows_csv,
        [
            {
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": 0,
                "support_x": support_center[0],
                "support_y": support_center[1],
                "render_x": support_center[0],
                "render_y": support_center[1],
                "center_x": query_center[0] - 1.0,
                "center_y": query_center[1],
                "query_gt_x": query_center[0],
                "query_gt_y": query_center[1],
            }
        ],
    )

    summary = augment_measurement_rows_with_local_affine_from_observations(
        rows_csv=rows_csv,
        observations=observations,
        output_rows_csv=output_csv,
        local_radius_px=5.0,
        min_points=4,
    )

    out_rows = list(csv.DictReader(output_csv.open()))
    assert summary["output_rows"] == 1
    assert out_rows[0]["local_affine_valid"] == "True"
    assert int(out_rows[0]["local_affine_points"]) >= 4
    assert float(out_rows[0]["local_affine_rmse_px"]) < 1e-5
    estimated = np.asarray(
        [
            [float(out_rows[0]["support_to_query_a00"]), float(out_rows[0]["support_to_query_a01"])],
            [float(out_rows[0]["support_to_query_a10"]), float(out_rows[0]["support_to_query_a11"])],
        ],
        dtype=np.float64,
    )
    assert np.allclose(estimated, matrix, atol=1e-5)


def test_augment_rows_with_local_affine_scales_observations_to_target_image_size(tmp_path: Path) -> None:
    support_id = "seq1/frame00010.png"
    query_id = "seq1/frame00011.png"
    observations = [
        ColmapTrackObservation(0, support_id, 0, (10.0, 10.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
        ColmapTrackObservation(0, query_id, 0, (20.0, 10.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
        ColmapTrackObservation(1, support_id, 0, (12.0, 10.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
        ColmapTrackObservation(1, query_id, 0, (22.0, 10.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
        ColmapTrackObservation(2, support_id, 0, (10.0, 12.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
        ColmapTrackObservation(2, query_id, 0, (20.0, 12.0), np.zeros(3), 3, 0.1, image_width=100, image_height=50),
    ]
    rows_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "affine_rows.csv"
    _write_rows(
        rows_csv,
        [
            {
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": 0,
                "support_x": 20.0,
                "support_y": 20.0,
                "render_x": 20.0,
                "render_y": 20.0,
                "center_x": 40.0,
                "center_y": 20.0,
                "query_gt_x": 40.0,
                "query_gt_y": 20.0,
            }
        ],
    )

    summary = augment_measurement_rows_with_local_affine_from_observations(
        rows_csv=rows_csv,
        observations=observations,
        output_rows_csv=output_csv,
        image_width=200,
        image_height=100,
        local_radius_px=8.0,
        min_points=3,
    )

    out_row = next(csv.DictReader(output_csv.open()))
    assert summary["output_rows"] == 1
    assert out_row["local_affine_valid"] == "True"


def test_augment_rows_with_local_homography_estimates_support_to_query_matrix(tmp_path: Path) -> None:
    support_id = "seq1/frame00010.png"
    query_id = "seq1/frame00011.png"
    matrix = np.asarray([[1.0, 0.05, 8.0], [-0.08, 1.1, 5.0], [0.001, -0.0015, 1.0]], dtype=np.float64)
    support_points = [
        (18.0, 18.0),
        (22.0, 18.0),
        (18.0, 22.0),
        (22.0, 22.0),
        (20.0, 16.0),
        (24.0, 20.0),
        (16.0, 20.0),
        (20.0, 24.0),
    ]
    observations: list[ColmapTrackObservation] = []
    for idx, point in enumerate(support_points):
        support_xy = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
        query_h = matrix @ support_xy
        query_xy = query_h[:2] / query_h[2]
        observations.append(
            ColmapTrackObservation(idx, support_id, 0, point, np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
        observations.append(
            ColmapTrackObservation(idx, query_id, 0, tuple(query_xy), np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
    rows_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "homography_rows.csv"
    support_anchor = np.asarray(support_points[0], dtype=np.float64)
    query_anchor_h = matrix @ np.asarray([support_anchor[0], support_anchor[1], 1.0], dtype=np.float64)
    query_anchor = query_anchor_h[:2] / query_anchor_h[2]
    _write_rows(
        rows_csv,
        [
            {
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": 0,
                "support_x": support_anchor[0],
                "support_y": support_anchor[1],
                "render_x": support_anchor[0],
                "render_y": support_anchor[1],
                "center_x": query_anchor[0] - 1.0,
                "center_y": query_anchor[1],
                "query_gt_x": query_anchor[0],
                "query_gt_y": query_anchor[1],
            }
        ],
    )

    summary = augment_measurement_rows_with_local_homography_from_observations(
        rows_csv=rows_csv,
        observations=observations,
        output_rows_csv=output_csv,
        local_radius_px=8.0,
        min_points=4,
    )

    out_row = next(csv.DictReader(output_csv.open()))
    estimated = np.asarray(
        [
            [float(out_row["support_to_query_h00"]), float(out_row["support_to_query_h01"]), float(out_row["support_to_query_h02"])],
            [float(out_row["support_to_query_h10"]), float(out_row["support_to_query_h11"]), float(out_row["support_to_query_h12"])],
            [float(out_row["support_to_query_h20"]), float(out_row["support_to_query_h21"]), float(out_row["support_to_query_h22"])],
        ],
        dtype=np.float64,
    )
    estimated = estimated / estimated[2, 2]
    expected = matrix / matrix[2, 2]
    assert summary["output_rows"] == 1
    assert out_row["local_homography_valid"] == "True"
    assert int(out_row["local_homography_points"]) >= 4
    assert float(out_row["local_homography_rmse_px"]) < 1e-5
    assert np.allclose(estimated, expected, atol=1e-5)


def test_augment_rows_with_local_homography_keeps_anchor_fixed_under_noisy_support(tmp_path: Path) -> None:
    support_id = "seq1/frame00010.png"
    query_id = "seq1/frame00011.png"
    support_center = np.asarray([20.0, 20.0], dtype=np.float64)
    query_center = np.asarray([31.0, 27.0], dtype=np.float64)
    support_points = [
        (16.0, 16.0),
        (24.0, 16.0),
        (16.0, 24.0),
        (24.0, 24.0),
        (20.0, 14.0),
        (26.0, 20.0),
        (14.0, 20.0),
        (20.0, 26.0),
    ]
    noise = np.asarray(
        [
            [0.2, -0.1],
            [-0.3, 0.2],
            [0.1, 0.3],
            [0.4, -0.2],
            [-0.2, -0.3],
            [0.3, 0.1],
            [-0.1, 0.2],
            [0.2, -0.4],
        ],
        dtype=np.float64,
    )
    observations: list[ColmapTrackObservation] = []
    for idx, point in enumerate(support_points):
        support_xy = np.asarray(point, dtype=np.float64)
        query_xy = query_center + (support_xy - support_center) + noise[idx]
        observations.append(
            ColmapTrackObservation(idx, support_id, 0, point, np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
        observations.append(
            ColmapTrackObservation(idx, query_id, 0, tuple(query_xy), np.zeros(3), 2, 0.1, image_width=64, image_height=64)
        )
    rows_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "homography_rows.csv"
    _write_rows(
        rows_csv,
        [
            {
                "query_id": query_id,
                "support_image_id": support_id,
                "track_id": 0,
                "support_x": support_center[0],
                "support_y": support_center[1],
                "render_x": support_center[0],
                "render_y": support_center[1],
                "center_x": query_center[0] - 1.0,
                "center_y": query_center[1],
                "query_gt_x": query_center[0],
                "query_gt_y": query_center[1],
            }
        ],
    )

    summary = augment_measurement_rows_with_local_homography_from_observations(
        rows_csv=rows_csv,
        observations=observations,
        output_rows_csv=output_csv,
        local_radius_px=10.0,
        min_points=4,
    )

    out_row = next(csv.DictReader(output_csv.open()))
    h = np.asarray(
        [
            [float(out_row["support_to_query_h00"]), float(out_row["support_to_query_h01"]), float(out_row["support_to_query_h02"])],
            [float(out_row["support_to_query_h10"]), float(out_row["support_to_query_h11"]), float(out_row["support_to_query_h12"])],
            [float(out_row["support_to_query_h20"]), float(out_row["support_to_query_h21"]), float(out_row["support_to_query_h22"])],
        ],
        dtype=np.float64,
    )
    projected = h @ np.asarray([support_center[0], support_center[1], 1.0], dtype=np.float64)
    projected_xy = projected[:2] / projected[2]
    assert np.allclose(projected_xy, query_center, atol=1e-6)
    assert float(summary["anchor_reprojection_max_px"]) < 1e-6
