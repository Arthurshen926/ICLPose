from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.tools.vfm.build_real_real_measurement_rows import _load_image_id_allowlist
from feature_extract.vfm.measurement_v1.real_real_tracks import (
    build_real_real_measurement_rows_from_observations,
    build_real_real_query_measurement_rows_from_observations,
    build_real_same_image_measurement_rows_from_observations,
)


def test_build_real_real_rows_pairs_only_shared_tracks_and_writes_synthetic_query_residual(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/a.png", 0, (10.0, 11.0), np.zeros(3), 2, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(7, "seq0/b.png", 1, (13.0, 12.0), np.zeros(3), 2, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(8, "seq0/c.png", 0, (20.0, 20.0), np.zeros(3), 1, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(9, "seq0/d.png", 0, (1.0, 1.0), np.zeros(3), 2, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(9, "seq0/e.png", 1, (15.0, 15.0), np.zeros(3), 2, 0.2, image_width=32, image_height=32),
    ]
    output_rows = tmp_path / "real_real_rows.csv"

    summary = build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=32,
        image_height=32,
        search_radius_px=4.0,
        context_radius_px=2.0,
        residual_bins_px=(1.5,),
        max_rows=8,
        seed=3,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert summary["output_rows"] == len(rows)
    assert {row["track_id"] for row in rows} == {"7"}
    assert rows
    for row in rows:
        assert row["query_id"] != row["support_image_id"]
        assert float(row["render_x"]) == float(row["support_x"])
        assert float(row["render_y"]) == float(row["support_y"])
        delta = np.asarray(
            [
                float(row["query_gt_x"]) - float(row["center_x"]),
                float(row["query_gt_y"]) - float(row["center_y"]),
            ],
            dtype=np.float64,
        )
        assert np.linalg.norm(delta) <= 4.0
        assert np.linalg.norm(delta) > 0.1
        assert float(row["requested_residual_px"]) == 1.5


def test_build_real_real_rows_can_filter_observations_by_image_allowlist(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/train_a.png", 0, (10.0, 11.0), np.zeros(3), 3, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(7, "seq0/train_b.png", 1, (13.0, 12.0), np.zeros(3), 3, 0.2, image_width=32, image_height=32),
        ColmapTrackObservation(7, "seq0/test_c.png", 2, (14.0, 12.0), np.zeros(3), 3, 0.2, image_width=32, image_height=32),
    ]
    output_rows = tmp_path / "real_real_train_only_rows.csv"

    summary = build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=32,
        image_height=32,
        search_radius_px=4.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        image_id_allowlist={"seq0/train_a.png", "seq0/train_b.png"},
        seed=3,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert rows
    assert summary["image_allowlist_count"] == 2
    assert summary["input_observations"] == 2
    assert {row["query_id"] for row in rows} | {row["support_image_id"] for row in rows} == {
        "seq0/train_a.png",
        "seq0/train_b.png",
    }


def test_build_real_real_query_rows_can_filter_observations_by_image_allowlist(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/train_a.png", 0, (20.0, 20.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq0/train_b.png", 1, (23.0, 21.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq0/test_c.png", 2, (24.0, 21.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_real_query_train_only_rows.csv"

    summary = build_real_real_query_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=4.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        image_id_allowlist={"seq0/train_a.png", "seq0/train_b.png"},
        seed=3,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert rows
    assert summary["image_allowlist_count"] == 2
    assert summary["input_observations"] == 2
    assert {row["query_id"] for row in rows} | {row["support_image_id"] for row in rows} == {
        "seq0/train_a.png",
        "seq0/train_b.png",
    }


def test_load_image_id_allowlist_skips_cambridge_dataset_headers(tmp_path: Path) -> None:
    image_id_file = tmp_path / "dataset_train.txt"
    image_id_file.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "seq9/frame00001.png -14.83 0.10 13.25 0.1 0.8 0.0 0.5",
                "seq9/frame00002.png -14.84 0.19 13.30 0.1 0.8 0.0 0.5",
            ]
        )
        + "\n"
    )

    assert _load_image_id_allowlist(image_id_file) == {"seq9/frame00001.png", "seq9/frame00002.png"}


def test_build_real_real_rows_scales_observation_coordinates_to_requested_image_size(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/a.png", 0, (10.0, 11.0), np.zeros(3), 2, 0.2, image_width=100, image_height=50),
        ColmapTrackObservation(7, "seq0/b.png", 1, (20.0, 12.0), np.zeros(3), 2, 0.2, image_width=100, image_height=50),
    ]
    output_rows = tmp_path / "real_real_rows.csv"

    build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=200,
        image_height=100,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        seed=5,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert rows
    row = rows[0]
    assert float(row["support_x"]) in {20.0, 40.0}
    assert float(row["query_gt_x"]) in {20.0, 40.0}
    assert float(row["support_y"]) in {22.0, 24.0}
    assert float(row["query_gt_y"]) in {22.0, 24.0}


def test_build_real_real_rows_can_filter_large_view_angle_pairs(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(
            7,
            "seq0/a.png",
            0,
            (20.0, 20.0),
            np.zeros(3),
            3,
            0.2,
            image_width=64,
            image_height=64,
            viewing_ray=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ),
        ColmapTrackObservation(
            7,
            "seq0/b.png",
            1,
            (22.0, 20.0),
            np.zeros(3),
            3,
            0.2,
            image_width=64,
            image_height=64,
            viewing_ray=np.asarray([0.0, 0.01, 0.99995], dtype=np.float64),
        ),
        ColmapTrackObservation(
            7,
            "seq0/c.png",
            2,
            (40.0, 40.0),
            np.zeros(3),
            3,
            0.2,
            image_width=64,
            image_height=64,
            viewing_ray=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]
    output_rows = tmp_path / "real_real_rows.csv"

    summary = build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        seed=5,
        max_view_angle_deg=5.0,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert rows
    assert {row["support_image_id"] for row in rows} | {row["query_id"] for row in rows} == {"seq0/a.png", "seq0/b.png"}
    assert summary["skipped"]["large_view_angle"] > 0


def test_build_real_real_rows_can_filter_to_same_sequence_and_frame_gap(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq1/frame00010.png", 0, (20.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq1/frame00012.png", 1, (22.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq1/frame00030.png", 2, (24.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq2/frame00011.png", 3, (26.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_real_rows.csv"

    summary = build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        seed=5,
        same_sequence_only=True,
        max_frame_gap=3,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert rows
    assert {row["query_id"] for row in rows} | {row["support_image_id"] for row in rows} == {
        "seq1/frame00010.png",
        "seq1/frame00012.png",
    }
    assert summary["skipped"]["different_sequence"] > 0
    assert summary["skipped"]["large_frame_gap"] > 0


def test_build_real_real_rows_can_add_wrong_support_dustbin_targets(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq1/frame00010.png", 0, (20.0, 20.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq1/frame00011.png", 1, (22.0, 20.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00010.png", 2, (34.0, 20.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00011.png", 3, (36.0, 20.0), np.zeros(3), 3, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_real_rows.csv"

    summary = build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        wrong_support_rows_per_positive=1,
        same_sequence_only=True,
        max_frame_gap=1,
        seed=5,
    )

    rows = list(csv.DictReader(output_rows.open()))
    wrong_rows = [row for row in rows if row["target_is_dustbin"] == "True"]
    positive_rows = [row for row in rows if row["target_is_dustbin"] == "False"]
    assert positive_rows
    assert summary["wrong_support_rows"] == len(wrong_rows) == len(positive_rows)
    for row in wrong_rows:
        assert row["query_id"] != row["support_image_id"]
        assert row["support_track_id"] != row["track_id"]
        delta = np.asarray(
            [
                float(row["query_gt_x"]) - float(row["center_x"]),
                float(row["query_gt_y"]) - float(row["center_y"]),
            ],
            dtype=np.float64,
        )
        assert np.linalg.norm(delta) <= 2.0


def test_build_real_real_query_rows_caps_unique_tracks_per_query_and_reports_coverage(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq1/frame00010.png", 0, (20.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq1/frame00011.png", 1, (21.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00010.png", 2, (30.0, 20.0), np.zeros(3), 5, 0.1, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00011.png", 3, (31.0, 20.0), np.zeros(3), 5, 0.1, image_width=64, image_height=64),
        ColmapTrackObservation(9, "seq1/frame00010.png", 4, (40.0, 20.0), np.zeros(3), 6, 0.4, image_width=64, image_height=64),
        ColmapTrackObservation(9, "seq1/frame00011.png", 5, (41.0, 20.0), np.zeros(3), 6, 0.4, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "query_rows.csv"

    summary = build_real_real_query_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        same_sequence_only=True,
        max_frame_gap=1,
        max_tracks_per_query=2,
        seed=11,
    )

    rows = list(csv.DictReader(output_rows.open()))
    rows_by_query: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_query.setdefault(row["query_id"], []).append(row)
    assert summary["query_count"] == 2
    assert summary["output_rows"] == 4
    assert summary["rows_per_query_min"] == 2
    assert summary["rows_per_query_median"] == 2
    assert summary["rows_per_query_max"] == 2
    assert set(rows_by_query) == {"seq1/frame00010.png", "seq1/frame00011.png"}
    for query_rows in rows_by_query.values():
        assert len(query_rows) == 2
        assert len({row["track_id"] for row in query_rows}) == 2
        for row in query_rows:
            assert row["query_id"] != row["support_image_id"]
            assert row["track_id"] == row["support_track_id"]
            assert row["target_is_dustbin"] == "False"
            delta = np.asarray(
                [
                    float(row["query_gt_x"]) - float(row["center_x"]),
                    float(row["query_gt_y"]) - float(row["center_y"]),
                ],
                dtype=np.float64,
            )
            assert 0.1 < np.linalg.norm(delta) <= 2.0


def test_build_real_real_query_rows_can_emit_all_residual_bins_per_unique_track(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq1/frame00010.png", 0, (20.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(7, "seq1/frame00011.png", 1, (21.0, 20.0), np.zeros(3), 4, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00010.png", 2, (30.0, 20.0), np.zeros(3), 5, 0.1, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq1/frame00011.png", 3, (31.0, 20.0), np.zeros(3), 5, 0.1, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "query_rows_allbins.csv"

    summary = build_real_real_query_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=3.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0, 2.0),
        same_sequence_only=True,
        max_frame_gap=1,
        max_tracks_per_query=2,
        emit_all_residual_bins=True,
        seed=11,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert summary["emit_all_residual_bins"] is True
    assert summary["unique_tracks_per_query_median"] == 2.0
    assert summary["rows_per_query_median"] == 4.0
    assert len(rows) == 8
    by_query_track: dict[tuple[str, str], set[float]] = {}
    for row in rows:
        by_query_track.setdefault((row["query_id"], row["track_id"]), set()).add(float(row["requested_residual_px"]))
    assert all(values == {1.0, 2.0} for values in by_query_track.values())


def test_build_real_same_image_rows_use_same_observation_as_template_and_query_target(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/a.png", 0, (20.0, 20.0), np.zeros(3), 2, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq0/b.png", 0, (2.0, 2.0), np.zeros(3), 2, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_same_rows.csv"

    summary = build_real_same_image_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        seed=5,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert summary["output_rows"] == 1
    assert rows[0]["query_id"] == "seq0/a.png"
    assert rows[0]["support_image_id"] == "seq0/a.png"
    assert float(rows[0]["support_x"]) == float(rows[0]["query_gt_x"])
    assert float(rows[0]["support_y"]) == float(rows[0]["query_gt_y"])
    delta = np.asarray(
        [
            float(rows[0]["query_gt_x"]) - float(rows[0]["center_x"]),
            float(rows[0]["query_gt_y"]) - float(rows[0]["center_y"]),
        ],
        dtype=np.float64,
    )
    assert np.linalg.norm(delta) > 0.1


def test_build_real_same_image_rows_can_add_dustbin_targets(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/a.png", 0, (20.0, 20.0), np.zeros(3), 2, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_same_rows.csv"

    summary = build_real_same_image_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        dustbin_residual_bins_px=(3.0,),
        seed=5,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert summary["dustbin_rows"] == 1
    assert {row["target_is_dustbin"] for row in rows} == {"False", "True"}
    dustbin = next(row for row in rows if row["target_is_dustbin"] == "True")
    delta = np.asarray(
        [
            float(dustbin["query_gt_x"]) - float(dustbin["center_x"]),
            float(dustbin["query_gt_y"]) - float(dustbin["center_y"]),
        ],
        dtype=np.float64,
    )
    assert np.linalg.norm(delta) > 2.0


def test_build_real_same_image_rows_can_add_wrong_support_dustbin_targets_inside_search_window(tmp_path: Path) -> None:
    observations = [
        ColmapTrackObservation(7, "seq0/a.png", 0, (20.0, 20.0), np.zeros(3), 2, 0.2, image_width=64, image_height=64),
        ColmapTrackObservation(8, "seq0/a.png", 1, (28.0, 20.0), np.zeros(3), 2, 0.2, image_width=64, image_height=64),
    ]
    output_rows = tmp_path / "real_same_rows.csv"

    summary = build_real_same_image_measurement_rows_from_observations(
        observations,
        output_rows_csv=output_rows,
        image_width=64,
        image_height=64,
        search_radius_px=2.0,
        context_radius_px=2.0,
        residual_bins_px=(1.0,),
        wrong_support_rows_per_positive=1,
        seed=5,
    )

    rows = list(csv.DictReader(output_rows.open()))
    wrong_rows = [row for row in rows if row["target_is_dustbin"] == "True"]
    assert summary["wrong_support_rows"] == len(wrong_rows) == 2
    assert wrong_rows
    for row in wrong_rows:
        assert row["support_image_id"] == row["query_id"]
        assert row["support_track_id"] != row["track_id"]
        delta = np.asarray(
            [
                float(row["query_gt_x"]) - float(row["center_x"]),
                float(row["query_gt_y"]) - float(row["center_y"]),
            ],
            dtype=np.float64,
        )
        assert np.linalg.norm(delta) <= 2.0
