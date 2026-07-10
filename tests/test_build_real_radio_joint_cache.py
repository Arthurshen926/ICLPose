from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
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
