from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.lowlevel_offset_sidecar import SuperPointKeypointSet
from feature_extract.vfm.measurement_v1.superpoint_teacher_diagnostics import (
    export_superpoint_teacher_diagnostics,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_export_superpoint_teacher_uses_descriptor_match_and_support_delta(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_root / "seq0" / "query.png")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_root / "seq0" / "support.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": 10.0,
                "support_y": 10.0,
                "center_x": 16.0,
                "center_y": 16.0,
                "query_gt_x": 12.0,
                "query_gt_y": 18.0,
                "requested_residual_px": 4.5,
                "target_is_dustbin": "False",
            }
        ],
    )
    keypoints = {
        "seq0/support.png": SuperPointKeypointSet(
            xy=np.asarray([[11.0, 10.0], [20.0, 20.0]], dtype=np.float32),
            scores=np.asarray([0.8, 0.95], dtype=np.float32),
            descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
        "seq0/query.png": SuperPointKeypointSet(
            xy=np.asarray([[13.0, 18.0], [15.0, 15.0]], dtype=np.float32),
            scores=np.asarray([0.2, 0.9], dtype=np.float32),
            descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    }

    summary = export_superpoint_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher",
        keypoints_by_image=keypoints,
        candidate_radius_px=8.0,
        support_radius_px=4.0,
        min_query_score=0.0,
        min_support_score=0.0,
        descriptor_weight=2.0,
        query_score_weight=0.1,
        center_penalty_weight=0.0,
        support_distance_penalty_weight=0.0,
        score_threshold=0.1,
    )

    assert summary["row_count"] == 1
    assert summary["applied_count"] == 1
    assert summary["metrics"]["baseline_median_px"] > 4.0
    assert summary["metrics"]["teacher_applied_median_px"] == 0.0
    assert summary["metrics"]["teacher_fallback_improve_ratio"] == 1.0
    rows = list(csv.DictReader((tmp_path / "teacher" / "teacher_rows.csv").open()))
    assert len(rows) == 1
    assert rows[0]["teacher_applied"] == "True"
    assert float(rows[0]["teacher_pred_x"]) == 12.0
    assert float(rows[0]["teacher_pred_y"]) == 18.0
    assert float(rows[0]["support_delta_x"]) == -1.0
    assert float(rows[0]["support_delta_y"]) == 0.0


def test_export_superpoint_teacher_reports_unapplied_fallback_and_availability(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_root / "seq0" / "query.png")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_root / "seq0" / "support.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": 10.0,
                "support_y": 10.0,
                "center_x": 16.0,
                "center_y": 16.0,
                "query_gt_x": 13.0,
                "query_gt_y": 18.0,
                "requested_residual_px": 3.6,
                "target_is_dustbin": "False",
            }
        ],
    )
    keypoints = {
        "seq0/support.png": SuperPointKeypointSet(
            xy=np.asarray([[10.0, 10.0]], dtype=np.float32),
            scores=np.asarray([0.8], dtype=np.float32),
            descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        ),
        "seq0/query.png": SuperPointKeypointSet(
            xy=np.asarray([[13.2, 18.1]], dtype=np.float32),
            scores=np.asarray([0.9], dtype=np.float32),
            descriptors=np.asarray([[0.0, 1.0]], dtype=np.float32),
        ),
    }

    summary = export_superpoint_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher",
        keypoints_by_image=keypoints,
        candidate_radius_px=8.0,
        support_radius_px=4.0,
        min_query_score=0.0,
        min_support_score=0.0,
        descriptor_weight=2.0,
        query_score_weight=0.0,
        center_penalty_weight=0.0,
        support_distance_penalty_weight=0.0,
        score_threshold=0.5,
    )

    assert summary["applied_count"] == 0
    assert summary["metrics"]["teacher_applied_median_px"] is None
    assert summary["metrics"]["teacher_fallback_improve_ratio"] == 0.0
    assert summary["metrics"]["oracle_query_sp_availability_at_0p5px"] == 1.0
    rows = list(csv.DictReader((tmp_path / "teacher" / "teacher_rows.csv").open()))
    assert rows[0]["teacher_applied"] == "False"
    assert rows[0]["teacher_reason"] == "selector_below_threshold"


def test_export_superpoint_teacher_reuses_persistent_keypoint_cache(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query = np.zeros((32, 32, 3), dtype=np.uint8)
    query[0, 0, 0] = 2
    support = np.zeros((32, 32, 3), dtype=np.uint8)
    support[0, 0, 0] = 1
    Image.fromarray(query).save(image_root / "seq0" / "query.png")
    Image.fromarray(support).save(image_root / "seq0" / "support.png")
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "track_id": 7,
                "support_track_id": 7,
                "support_x": 10.0,
                "support_y": 10.0,
                "center_x": 16.0,
                "center_y": 16.0,
                "query_gt_x": 13.0,
                "query_gt_y": 18.0,
                "requested_residual_px": 3.6,
                "target_is_dustbin": "False",
            }
        ],
    )

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, image: np.ndarray) -> SuperPointKeypointSet:
            self.calls += 1
            marker = int(image[0, 0, 0])
            if marker == 1:
                return SuperPointKeypointSet(
                    xy=np.asarray([[10.0, 10.0]], dtype=np.float32),
                    scores=np.asarray([0.8], dtype=np.float32),
                    descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
                )
            return SuperPointKeypointSet(
                xy=np.asarray([[13.0, 18.0]], dtype=np.float32),
                scores=np.asarray([0.8], dtype=np.float32),
                descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

    class FailingDetector:
        def detect(self, image: np.ndarray) -> SuperPointKeypointSet:
            raise AssertionError("persistent cache was not used")

    cache_dir = tmp_path / "sp_cache"
    detector = FakeDetector()
    first = export_superpoint_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher_first",
        detector=detector,
        keypoint_cache_dir=cache_dir,
        min_query_score=0.0,
        min_support_score=0.0,
        score_threshold=0.1,
    )
    second = export_superpoint_teacher_diagnostics(
        rows_csv=rows_csv,
        image_root=image_root,
        output_dir=tmp_path / "teacher_second",
        detector=FailingDetector(),
        keypoint_cache_dir=cache_dir,
        min_query_score=0.0,
        min_support_score=0.0,
        score_threshold=0.1,
    )

    assert detector.calls == 2
    assert len(list(cache_dir.glob("*.npz"))) == 2
    assert first["metrics"]["teacher_applied_median_px"] == second["metrics"]["teacher_applied_median_px"] == 0.0
