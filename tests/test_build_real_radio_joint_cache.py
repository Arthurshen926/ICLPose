from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from feature_extract.tools.vfm import build_real_radio_joint_cache
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_training_set_npz


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), int(value), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_feature(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2) + float(offset)
    np.savez(path, radio_final=feature)


def test_build_real_radio_joint_cache_materializes_fine_supervised_real_pairs(tmp_path: Path, monkeypatch) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "seq/q.png", value=32)
    _write_rgb(image_root / "seq/r.png", value=64)
    _write_feature(feature_root / "seq_q.npz", offset=0.0)
    _write_feature(feature_root / "seq_r.npz", offset=10.0)
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        fieldnames = [
            "query_id",
            "support_image_id",
            "track_id",
            "support_track_id",
            "query_gt_x",
            "query_gt_y",
            "support_x",
            "support_y",
            "query_reprojection_error",
            "support_reprojection_error",
            "target_is_dustbin",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "track_id": "1",
                "support_track_id": "1",
                "query_gt_x": "3",
                "query_gt_y": "4",
                "support_x": "12",
                "support_y": "12",
                "query_reprojection_error": "0.25",
                "support_reprojection_error": "0.25",
                "target_is_dustbin": "False",
            }
        )
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "track_id": "2",
                "support_track_id": "3",
                "query_gt_x": "4",
                "query_gt_y": "4",
                "support_x": "10",
                "support_y": "10",
                "query_reprojection_error": "3.0",
                "support_reprojection_error": "3.0",
                "target_is_dustbin": "True",
            }
        )

    monkeypatch.chdir(tmp_path)
    manifest = Path("out/real_joint_manifest.json")
    summary_json = Path("out/summary.json")
    build_real_radio_joint_cache.main(
        [
            "--rows_csv",
            str(rows_csv),
            "--image_root",
            str(image_root),
            "--feature_root",
            str(feature_root),
            "--feature_path_template",
            "{image_stem}.npz",
            "--feature_key",
            "radio_final",
            "--output_manifest",
            str(manifest),
            "--summary_json",
            str(summary_json),
            "--split_name",
            "train",
            "--hard_negatives_per_match",
            "1",
        ]
    )

    metadata = json.loads((tmp_path / manifest).read_text())
    assert metadata["shards"][0]["path"] == "shards/shard_00000.npz"
    shard_path = manifest.parent / metadata["shards"][0]["path"]
    shard_path = tmp_path / shard_path
    samples, _sample_metadata = load_matcha_joint_training_set_npz(shard_path)
    summary = json.loads((tmp_path / summary_json).read_text())

    assert metadata["format"] == "vfm_matcha_joint_training_manifest_v1"
    assert summary["stage"] == "real_radio_joint_cache_builder"
    assert summary["feature_key"] == "radio_final"
    assert summary["built_pair_count"] == 1
    assert samples.query_feature_maps.shape == (1, 4, 2, 2)
    assert samples.render_feature_maps.shape == (1, 4, 2, 2)
    assert samples.query_rgb_images.shape == (1, 3, 16, 16)
    assert samples.render_rgb_images.shape == (1, 3, 16, 16)
    assert samples.fine_sample_pair_indices is not None
    assert samples.fine_query_cell_indices is not None
    assert samples.fine_render_cell_indices is not None
    assert samples.fine_query_xy is not None
    assert samples.fine_render_xy is not None
    assert samples.sample_no_match_labels is not None
    assert int(np.sum(samples.sample_no_match_labels)) >= 1
    assert samples.sample_track_ids is not None
    assert samples.sample_track_ids.tolist() == [1, -1, -1]
    assert samples.landmark_track_ids.tolist() == [1]


def test_build_real_radio_joint_cache_supports_radio_token_feature_template(tmp_path: Path) -> None:
    assert build_real_radio_joint_cache._feature_path_from_template(
        "{image_token}.npz",
        image_id="seq9/frame00010.png",
    ) == Path("seq9__frame00010.png.npz")


def test_build_real_radio_joint_cache_writes_referenced_manifest_without_shards(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "seq/q.png", value=16)
    _write_rgb(image_root / "seq/r.png", value=48)
    _write_feature(feature_root / "seq_q.npz", offset=1.0)
    _write_feature(feature_root / "seq_r.npz", offset=2.0)
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        fieldnames = ["query_id", "support_image_id", "track_id", "support_track_id", "query_gt_x", "query_gt_y", "support_x", "support_y"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "track_id": "1",
                "support_track_id": "1",
                "query_gt_x": "3",
                "query_gt_y": "4",
                "support_x": "12",
                "support_y": "12",
            }
        )
    manifest = tmp_path / "referenced_manifest.json"
    summary_json = tmp_path / "summary.json"

    build_real_radio_joint_cache.main(
        [
            "--rows_csv",
            str(rows_csv),
            "--image_root",
            str(image_root),
            "--feature_root",
            str(feature_root),
            "--feature_path_template",
            "{image_stem}.npz",
            "--feature_key",
            "radio_final",
            "--output_manifest",
            str(manifest),
            "--summary_json",
            str(summary_json),
            "--manifest_mode",
            "referenced",
            "--hard_negatives_per_match",
            "1",
        ]
    )

    metadata = json.loads(manifest.read_text())
    summary = json.loads(summary_json.read_text())
    provider = build_real_radio_joint_cache.RealRadioReferencedJointSampleProvider(manifest)
    samples = provider.get(0)
    landmark_audit = provider.landmark_retrieval_audit()

    assert metadata["format"] == "vfm_real_radio_joint_referenced_manifest_v1"
    assert "shards" not in metadata
    assert not (tmp_path / "shards").exists()
    assert metadata["records"][0]["query_id"] == "seq/q.png"
    assert metadata["records"][0]["reference_image_id"] == "seq/r.png"
    assert metadata["records"][0]["row_indices"] == [0]
    assert summary["cache_format"] == "referenced"
    assert summary["built_pair_count"] == 1
    assert len(provider) == 1
    assert samples.query_feature_maps.shape == (1, 4, 2, 2)
    assert samples.fine_sample_pair_indices is not None
    assert samples.sample_track_ids is not None
    assert samples.sample_track_ids.tolist() == [1, -1]
    assert samples.landmark_track_ids.tolist() == [1]
    assert landmark_audit["unique_track_count"] == 1
    assert landmark_audit["expected_first_epoch_history_hit_fraction_without_eviction"] == 0.0

    observations = tmp_path / "tracks.jsonl"
    observations.write_text(
        "\n".join(
            json.dumps(
                {
                    "image_id": image_id,
                    "track_id": 1,
                    "xy": xy,
                    "image_width": 16,
                    "image_height": 16,
                    "xyz": [1.0, 2.0, 3.0],
                    "track_length": 2,
                    "reprojection_error": 0.1,
                }
            )
            for image_id, xy in (("seq/q.png", [3.0, 4.0]), ("seq/r.png", [12.0, 12.0]))
        )
        + "\n"
    )
    feature_only_provider = build_real_radio_joint_cache.RealRadioReferencedJointSampleProvider(
        manifest,
        load_rgb=False,
        track_observation_index=build_real_radio_joint_cache.load_track_observation_index(observations),
    )
    feature_only = feature_only_provider.get(0)

    assert feature_only.query_rgb_images is None
    assert feature_only.render_rgb_images is None
    assert feature_only.pair_query_image_sizes.tolist() == [[16, 16]]
    assert feature_only.pair_reference_image_sizes.tolist() == [[16, 16]]
    assert not feature_only_provider._rgb_cache


def test_feature_only_provider_uses_referenced_rgb_size_not_resized_sfm_size(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "seq/q.png", value=16)
    _write_rgb(image_root / "seq/r.png", value=48)
    _write_feature(feature_root / "seq_q.npz", offset=1.0)
    _write_feature(feature_root / "seq_r.npz", offset=2.0)
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "support_image_id",
                "track_id",
                "support_track_id",
                "query_gt_x",
                "query_gt_y",
                "support_x",
                "support_y",
            ],
        )
        writer.writeheader()
        # These 16x16 RGB coordinates would be out of bounds in the 8x8 SfM frame.
        writer.writerow(
            {
                "query_id": "seq/q.png",
                "support_image_id": "seq/r.png",
                "track_id": "1",
                "support_track_id": "1",
                "query_gt_x": "12",
                "query_gt_y": "12",
                "support_x": "12",
                "support_y": "12",
            }
        )
    manifest = tmp_path / "referenced_manifest.json"
    build_real_radio_joint_cache.build_real_radio_joint_cache(
        rows_csv=rows_csv,
        image_root=image_root,
        feature_root=feature_root,
        feature_path_template="{image_stem}.npz",
        feature_key="radio_final",
        output_manifest=manifest,
        split_name="train",
        manifest_mode="referenced",
        hard_negatives_per_match=1,
    )
    observations = tmp_path / "tracks.jsonl"
    observations.write_text(
        "\n".join(
            json.dumps(
                {
                    "image_id": image_id,
                    "track_id": 1,
                    "xy": [6.0, 6.0],
                    "image_width": 8,
                    "image_height": 8,
                    "xyz": [1.0, 2.0, 3.0],
                    "track_length": 2,
                    "reprojection_error": 0.1,
                }
            )
            for image_id in ("seq/q.png", "seq/r.png")
        )
        + "\n"
    )
    provider = build_real_radio_joint_cache.RealRadioReferencedJointSampleProvider(
        manifest,
        load_rgb=False,
        track_observation_index=build_real_radio_joint_cache.load_track_observation_index(observations),
    )

    sample = provider.get(0)

    assert provider._image_size("seq/q.png") == (16, 16)
    assert sample.pair_query_image_sizes.tolist() == [[16, 16]]
    assert sample.pair_reference_image_sizes.tolist() == [[16, 16]]
    assert int(sample.coarse_fine_samples.sample_count) > 0


def test_referenced_cache_preserves_cell_colliding_tracks_for_landmark_retrieval(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "seq/q.png", value=16)
    _write_rgb(image_root / "seq/r.png", value=48)
    _write_feature(feature_root / "seq_q.npz", offset=1.0)
    _write_feature(feature_root / "seq_r.npz", offset=2.0)
    rows_csv = tmp_path / "rows.csv"
    with rows_csv.open("w", newline="") as handle:
        fieldnames = [
            "query_id",
            "support_image_id",
            "track_id",
            "support_track_id",
            "query_gt_x",
            "query_gt_y",
            "support_x",
            "support_y",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track_id, delta in ((1, 0.0), (2, 0.25)):
            writer.writerow(
                {
                    "query_id": "seq/q.png",
                    "support_image_id": "seq/r.png",
                    "track_id": str(track_id),
                    "support_track_id": str(track_id),
                    "query_gt_x": str(3.0 + delta),
                    "query_gt_y": str(4.0 + delta),
                    "support_x": str(11.0 + delta),
                    "support_y": str(12.0 + delta),
                }
            )
    manifest = tmp_path / "referenced_manifest.json"
    build_real_radio_joint_cache.build_real_radio_joint_cache(
        rows_csv=rows_csv,
        image_root=image_root,
        feature_root=feature_root,
        feature_path_template="{image_stem}.npz",
        feature_key="radio_final",
        output_manifest=manifest,
        split_name="train",
        manifest_mode="referenced",
        hard_negatives_per_match=1,
    )

    samples = build_real_radio_joint_cache.RealRadioReferencedJointSampleProvider(manifest).get(0)

    assert int(np.count_nonzero(samples.sample_no_match_labels == 0)) == 1
    assert samples.landmark_track_ids.tolist() == [1, 2]
    assert samples.landmark_query_xy.shape == (2, 2)


def test_sfm_observation_index_builds_scaled_common_track_supervision(tmp_path: Path) -> None:
    observations = tmp_path / "tracks.jsonl"
    rows = [
        {
            "image_id": "q.png",
            "track_id": 7,
            "xy": [511.5, 287.5],
            "image_width": 1024,
            "image_height": 576,
            "xyz": [1.0, 2.0, 3.0],
            "track_length": 4,
            "reprojection_error": 0.1,
        },
        {
            "image_id": "q.png",
            "track_id": 7,
            "xy": [400.0, 200.0],
            "image_width": 1024,
            "image_height": 576,
            "xyz": [1.0, 2.0, 3.0],
            "track_length": 4,
            "reprojection_error": 1.0,
        },
        {
            "image_id": "r.png",
            "track_id": 7,
            "xy": [255.75, 143.75],
            "image_width": 1024,
            "image_height": 576,
            "xyz": [1.0, 2.0, 3.0],
            "track_length": 4,
            "reprojection_error": 0.2,
        },
        {
            "image_id": "q.png",
            "track_id": 8,
            "xy": [100.0, 100.0],
            "image_width": 1024,
            "image_height": 576,
            "xyz": [4.0, 5.0, 6.0],
            "track_length": 2,
            "reprojection_error": 0.3,
        },
    ]
    observations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    index = build_real_radio_joint_cache.load_track_observation_index(observations)
    common = index.common_tracks(
        "q.png",
        "r.png",
        query_source_size=(1920, 1080),
        reference_source_size=(1920, 1080),
    )

    assert common["track_ids"].tolist() == [7]
    np.testing.assert_allclose(common["query_xy"], [[959.5, 539.5]], atol=1e-6)
    np.testing.assert_allclose(common["reference_xy"], [[479.75, 269.75]], atol=1e-6)
    np.testing.assert_allclose(common["track_xyz"], [[1.0, 2.0, 3.0]])

    offsets, track_ids = index.query_cell_positive_csr(
        "q.png",
        np.asarray([[959.5, 539.5], [187.6, 187.7]], dtype=np.float64),
        target_size=(1920, 1080),
        grid_hw=(68, 120),
    )
    assert offsets.tolist() == [0, 1, 2]
    assert track_ids.tolist() == [7, 8]


def test_sfm_observation_index_strict_radius_crosses_coarse_cell_boundary(tmp_path: Path) -> None:
    observations = tmp_path / "tracks.jsonl"
    rows = [
        {
            "image_id": "q.png",
            "track_id": track_id,
            "xy": xy,
            "image_width": 100,
            "image_height": 100,
            "xyz": [float(track_id), 0.0, 1.0],
            "track_length": 2,
            "reprojection_error": 0.1,
        }
        for track_id, xy in (
            (1, [14.2, 20.0]),
            (2, [14.4, 20.0]),
            (3, [16.4, 20.0]),
        )
    ]
    observations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    index = build_real_radio_joint_cache.load_track_observation_index(observations)

    cell_offsets, cell_tracks = index.query_cell_positive_csr(
        "q.png",
        np.asarray([[14.2, 20.0]], dtype=np.float64),
        target_size=(100, 100),
        grid_hw=(7, 7),
    )
    strict_offsets, strict_tracks = index.query_radius_positive_csr(
        "q.png",
        np.asarray([[14.2, 20.0]], dtype=np.float64),
        target_size=(100, 100),
        radius_px=2.0,
    )

    assert cell_offsets.tolist() == [0, 1]
    assert cell_tracks.tolist() == [1]
    assert strict_offsets.tolist() == [0, 2]
    assert strict_tracks.tolist() == [1, 2]


def test_sfm_track_observation_index_cache_roundtrip_and_stale_rejection(tmp_path: Path) -> None:
    observations = tmp_path / "tracks.jsonl"
    rows = [
        {
            "image_id": image_id,
            "track_id": 7,
            "xy": [10.0, 20.0],
            "image_width": 100,
            "image_height": 80,
            "xyz": [1.0, 2.0, 3.0],
            "track_length": 2,
            "reprojection_error": 0.1,
        }
        for image_id in ("q.png", "r.png")
    ]
    observations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    cache = tmp_path / "tracks.index.npz"

    first = build_real_radio_joint_cache.load_track_observation_index(observations, cache_path=cache)
    second = build_real_radio_joint_cache.load_track_observation_index(observations, cache_path=cache)

    assert cache.exists()
    assert not list(tmp_path.glob(".*.tmp.npz"))
    np.testing.assert_array_equal(first.by_image["q.png"].track_ids, second.by_image["q.png"].track_ids)
    observations.write_text(observations.read_text() + json.dumps({**rows[0], "track_id": 8}) + "\n")
    with pytest.raises(ValueError, match="stale track observation index cache"):
        build_real_radio_joint_cache.load_track_observation_index(observations, cache_path=cache)
