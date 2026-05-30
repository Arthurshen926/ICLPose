import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import build_patch_positive_sets
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def test_patch_selector_training_learns_linear_transform_on_hard_negative_toy():
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingConfig,
        PatchSelectorTrainingSet,
        train_linear_patch_selector,
    )

    queries = np.asarray(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    positives = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    negatives = np.asarray(
        [
            [[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, -2.0, 0.0]],
            [[1.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, -2.0]],
            [[0.0, 1.0, -2.0, 0.0], [0.0, 1.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.0, -2.0], [1.0, 0.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    samples = PatchSelectorTrainingSet(
        query_features=queries,
        positive_features=positives,
        positive_mask=np.ones((4, 1), dtype=bool),
        negative_features=negatives,
        metadata={"scene": "toy"},
    )

    run = train_linear_patch_selector(
        samples,
        PatchSelectorTrainingConfig(
            output_dim=2,
            steps=120,
            batch_size=4,
            lr=0.08,
            seed=7,
            device="cpu",
            eval_split_fraction=0.0,
            temperature=0.1,
        ),
    )

    assert run.summary.final_loss < run.summary.initial_loss
    assert run.summary.train_top1_acc >= 0.75
    assert run.transform.method == "learned_linear_patch"
    assert run.transform.matrix.shape == (4, 2)
    assert run.transform.l2_normalize is True
    projected = run.transform.apply_rows(queries)
    assert projected.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), np.ones(4), atol=1e-5)


def test_patch_selector_training_can_hard_gate_low_energy_channel_groups():
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingConfig,
        PatchSelectorTrainingSet,
        train_linear_patch_selector,
    )

    queries = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    positives = queries[:, None, :].copy()
    negatives = np.asarray(
        [
            [[0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    samples = PatchSelectorTrainingSet(
        query_features=queries,
        positive_features=positives,
        positive_mask=np.ones((4, 1), dtype=bool),
        negative_features=negatives,
    )

    run = train_linear_patch_selector(
        samples,
        PatchSelectorTrainingConfig(
            output_dim=2,
            steps=30,
            batch_size=4,
            lr=0.05,
            seed=2,
            device="cpu",
            eval_split_fraction=0.0,
            group_size=2,
            hard_gate_keep_fraction=0.5,
        ),
    )

    group_energy = run.transform.channel_scores.reshape(2, 2).sum(axis=1)
    assert run.summary.group_count == 2
    assert run.summary.active_group_count == 1
    assert 0.0 <= run.summary.gated_train_top1_acc <= 1.0
    assert np.count_nonzero(group_energy > 0.0) == 1
    assert np.allclose(run.transform.matrix[2:4], 0.0) or np.allclose(run.transform.matrix[0:2], 0.0)


def test_stage_c1_sample_builder_uses_patch_positives_and_hard_raw_negatives():
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorSampleConfig,
        build_patch_selector_samples_for_query,
    )

    camera = ColmapCamera(camera_id=1, model_id=0, width=5, height=5, params=(4.0, 2.0, 2.0))
    pose_w2c = np.eye(4, dtype=np.float64)
    landmark_index = LandmarkMapIndex(
        track_ids=np.asarray([10, 20, 30], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [-0.6, 0.0, 4.0], [-0.8, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.95, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64) * 3,
        observation_image_ids=(("ref_a.png",), ("ref_b.png",), ("ref_c.png",)),
    )
    query_feature = np.zeros((4, 2, 2), dtype=np.float32)
    query_feature[:, 1, 1] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    positives = build_patch_positive_sets(landmark_index, pose_w2c, camera, token_width=2, token_height=2)

    samples = build_patch_selector_samples_for_query(
        query_feature,
        landmark_index,
        positives,
        PatchSelectorSampleConfig(
            max_tokens_per_query=1,
            max_positives_per_token=1,
            hard_negatives_per_token=1,
            query_token_step=1,
            seed=3,
        ),
    )

    assert samples.query_features.shape == (1, 4)
    assert samples.positive_features.shape == (1, 1, 4)
    assert samples.negative_features.shape == (1, 1, 4)
    np.testing.assert_allclose(samples.positive_features[0, 0], landmark_index.features[0])
    np.testing.assert_allclose(samples.negative_features[0, 0], landmark_index.features[1])
    assert samples.metadata["sample_count"] == 1
    assert samples.metadata["raw_false_nearest_negative_count"] == 1


def test_stage_c1_sample_builder_can_mine_negatives_with_alternate_descriptor_space():
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorSampleConfig,
        build_patch_selector_samples_for_query,
    )

    camera = ColmapCamera(camera_id=1, model_id=0, width=5, height=5, params=(4.0, 2.0, 2.0))
    landmark_index = LandmarkMapIndex(
        track_ids=np.asarray([10, 20, 30], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [-0.6, 0.0, 4.0], [-0.8, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.95, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.ones((3,), dtype=np.int64) * 3,
        observation_image_ids=(("ref_a.png",), ("ref_b.png",), ("ref_c.png",)),
    )
    query_feature = np.zeros((4, 2, 2), dtype=np.float32)
    query_feature[:, 1, 1] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mining_query_feature = np.zeros((2, 2, 2), dtype=np.float32)
    mining_query_feature[:, 1, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    mining_landmark_features = np.asarray(
        [
            [1.0, 0.0],
            [0.2, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positives = build_patch_positive_sets(landmark_index, np.eye(4, dtype=np.float64), camera, token_width=2, token_height=2)

    samples = build_patch_selector_samples_for_query(
        query_feature,
        landmark_index,
        positives,
        PatchSelectorSampleConfig(
            max_tokens_per_query=1,
            max_positives_per_token=1,
            hard_negatives_per_token=1,
            query_token_step=1,
            seed=3,
        ),
        negative_mining_query_feature_map=mining_query_feature,
        negative_mining_landmark_features=mining_landmark_features,
    )

    np.testing.assert_allclose(samples.negative_features[0, 0], landmark_index.features[2])
    assert samples.metadata["negative_mining_descriptor"] == "alternate"


def test_stage_c1_sample_builder_batches_hard_negative_search_after_token_limit(monkeypatch):
    import feature_extract.vfm.patch_selector_training as training
    from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSet, PatchPositiveSets, TokenPatchBox

    landmark_index = LandmarkMapIndex(
        track_ids=np.asarray([10, 20], dtype=np.int64),
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64) * 3,
        observation_image_ids=(("ref_a.png",), ("ref_b.png",)),
    )
    positives = PatchPositiveSets(
        by_token={
            token_index: PatchPositiveSet(
                token_index=token_index,
                patch_box=TokenPatchBox(token_index, np.zeros(2), 0.0, 0.0, 1.0, 1.0),
                track_ids={10},
            )
            for token_index in range(4)
        },
        stride_x_px=1.0,
        stride_y_px=1.0,
        visible_track_ids={10, 20},
    )
    calls = {"query_count": 0}

    def fake_negative_search(query_features, *_args, **_kwargs):
        calls["query_count"] += int(query_features.shape[0])
        return [[1] for _ in range(query_features.shape[0])]

    monkeypatch.setattr(training, "_sample_negative_indices_batch", fake_negative_search)
    samples = training.build_patch_selector_samples_for_query(
        np.ones((2, 2, 2), dtype=np.float32),
        landmark_index,
        positives,
        training.PatchSelectorSampleConfig(
            max_tokens_per_query=2,
            max_positives_per_token=1,
            hard_negatives_per_token=1,
            seed=0,
        ),
    )

    assert samples.sample_count == 2
    assert calls["query_count"] == 2


def test_stage_c1_cli_trains_and_exports_transform_and_compressed_banks(tmp_path: Path):
    from feature_extract.tools.vfm.train_stage_c1_patch_selector import main
    from feature_extract.vfm.feature_compression import FeatureCompressionTransform
    from feature_extract.vfm.map_lifting import load_selected_track_bank_npz

    token_path = tmp_path / "tokens" / "query.npz"
    token_path.parent.mkdir()
    feature_map = np.zeros((4, 2, 2), dtype=np.float32)
    feature_map[:, 1, 1] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.savez_compressed(token_path, radio_final=feature_map)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="query.png",
                token_path=token_path,
                layers=(TokenLayerSpec(name="radio_final", model="toy", layer="final", channels=4, stride=2),),
                split="train",
                scene="toy",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)

    bank = SelectedTrackFeatureBank(
        tracks={
            10: TrackFeature(
                track_id=10,
                mean_feature=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                variance=np.zeros((4,), dtype=np.float32),
                observation_count=3,
                mean_utility=1.0,
                observation_image_ids=("ref_a.png",),
            ),
            20: TrackFeature(
                track_id=20,
                mean_feature=np.asarray([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
                variance=np.zeros((4,), dtype=np.float32),
                observation_count=3,
                mean_utility=1.0,
                observation_image_ids=("ref_b.png",),
            ),
        },
        feature_dim=4,
    )
    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(bank, bank_path)

    track_jsonl = tmp_path / "tracks.jsonl"
    rows = [
        {"track_id": 10, "image_id": "ref_a.png", "point2d_idx": 0, "xy": [2.0, 2.0], "xyz": [0.0, 0.0, 4.0], "track_length": 3, "reprojection_error": 0.1},
        {"track_id": 20, "image_id": "ref_b.png", "point2d_idx": 0, "xy": [2.0, 2.0], "xyz": [-0.6, 0.0, 4.0], "track_length": 3, "reprojection_error": 0.1},
    ]
    track_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "query.png 0 0 0 1 0 0 0\n"
    )

    output_dir = tmp_path / "out"
    main(
        [
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--track_observations",
            str(track_jsonl),
            "--query_pose_file",
            str(pose_file),
            "--submap_mode",
            "gt_visible",
            "--default_camera",
            "2,5,5,4,2,2,0",
            "--output_dim",
            "2",
            "--steps",
            "5",
            "--batch_size",
            "1",
            "--group_size",
            "2",
            "--hard_gate_keep_fraction",
            "0.5",
            "--max_train_samples",
            "4",
            "--max_tokens_per_query",
            "4",
            "--hard_negatives_per_token",
            "1",
            "--output_transform",
            str(output_dir / "transform.npz"),
            "--summary_json",
            str(output_dir / "summary.json"),
            "--output_query_dir",
            str(output_dir / "tokens"),
            "--output_query_manifest",
            str(output_dir / "manifest.json"),
            "--output_landmark_bank",
            str(output_dir / "bank.npz"),
        ]
    )

    transform = FeatureCompressionTransform.from_npz(output_dir / "transform.npz")
    assert transform.method == "learned_linear_patch"
    assert transform.output_dim == 2
    compressed_manifest = TokenBankManifest.from_json(output_dir / "manifest.json")
    assert compressed_manifest.records[0].layers[0].channels == 2
    compressed_bank = load_selected_track_bank_npz(output_dir / "bank.npz")
    assert compressed_bank.feature_dim == 2
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["sample_summary"]["sample_count"] >= 1
    assert summary["training"]["output_dim"] == 2
    assert summary["training"]["group_size"] == 2
    assert summary["training"]["active_group_count"] == 1


def test_stage_c1_training_set_npz_round_trip(tmp_path: Path):
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingSet,
        load_patch_selector_training_set_npz,
        save_patch_selector_training_set_npz,
    )

    samples = PatchSelectorTrainingSet(
        query_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        positive_features=np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        positive_mask=np.asarray([[True], [True]], dtype=bool),
        negative_features=np.asarray([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32),
        metadata={"scene": "toy", "seed": 3},
    )

    path = tmp_path / "samples.npz"
    save_patch_selector_training_set_npz(samples, path)
    loaded, metadata = load_patch_selector_training_set_npz(path)

    np.testing.assert_allclose(loaded.query_features, samples.query_features)
    np.testing.assert_allclose(loaded.positive_features, samples.positive_features)
    np.testing.assert_array_equal(loaded.positive_mask, samples.positive_mask)
    np.testing.assert_allclose(loaded.negative_features, samples.negative_features)
    assert metadata["scene"] == "toy"
    assert metadata["seed"] == 3


def test_stage_c1_incremental_sample_merge_caps_and_preserves_distances():
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingSet,
        append_patch_selector_training_set_capped,
    )

    def make_samples(offset: int, count: int) -> PatchSelectorTrainingSet:
        query = np.arange(offset, offset + count * 2, dtype=np.float32).reshape(count, 2)
        positives = np.repeat(query[:, None, :], repeats=1, axis=1)
        negatives = np.repeat(query[:, None, :], repeats=2, axis=1) * -1.0
        return PatchSelectorTrainingSet(
            query_features=query,
            positive_features=positives,
            positive_mask=np.ones((count, 1), dtype=bool),
            negative_features=negatives,
            positive_reprojection_distances=np.full((count, 1), float(offset), dtype=np.float32),
            negative_reprojection_distances=np.full((count, 2), float(offset + 1), dtype=np.float32),
            metadata={"sample_count": count},
        )

    merged = append_patch_selector_training_set_capped(None, make_samples(0, 3), max_samples=5, seed=7)
    merged = append_patch_selector_training_set_capped(merged, make_samples(100, 4), max_samples=5, seed=7)

    assert merged.sample_count == 5
    assert merged.metadata["source_set_count"] == 2
    assert merged.metadata["has_reprojection_distances"] is True
    assert merged.positive_reprojection_distances is not None
    assert merged.negative_reprojection_distances is not None
    assert merged.positive_reprojection_distances.shape == (5, 1)
    assert merged.negative_reprojection_distances.shape == (5, 2)


def test_stage_c1_cli_can_build_sample_cache_only_without_training(tmp_path: Path, monkeypatch):
    import feature_extract.tools.vfm.train_stage_c1_patch_selector as cli
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingSet,
        load_patch_selector_training_set_npz,
        save_patch_selector_training_set_npz,
    )

    samples = PatchSelectorTrainingSet(
        query_features=np.eye(4, dtype=np.float32),
        positive_features=np.eye(4, dtype=np.float32)[:, None, :],
        positive_mask=np.ones((4, 1), dtype=bool),
        negative_features=np.flipud(np.eye(4, dtype=np.float32))[:, None, :],
        metadata={"source": "cached"},
    )
    sample_cache = tmp_path / "samples.npz"
    save_patch_selector_training_set_npz(samples, sample_cache)

    def fail_if_training_runs(*_args, **_kwargs):
        raise AssertionError("training should not run in --build_sample_cache_only mode")

    monkeypatch.setattr(cli, "train_linear_patch_selector", fail_if_training_runs)
    output_dir = tmp_path / "out"
    written_cache = output_dir / "limited_samples.npz"
    cli.main(
        [
            "--query_manifest",
            str(tmp_path / "unused_manifest.json"),
            "--landmark_bank",
            str(tmp_path / "unused_bank.npz"),
            "--track_observations",
            str(tmp_path / "unused_tracks.jsonl"),
            "--query_pose_file",
            str(tmp_path / "unused_poses.txt"),
            "--sample_cache",
            str(sample_cache),
            "--write_sample_cache",
            str(written_cache),
            "--build_sample_cache_only",
            "--max_train_samples",
            "2",
            "--output_dim",
            "2",
            "--steps",
            "2",
            "--batch_size",
            "1",
            "--output_transform",
            str(output_dir / "transform.npz"),
            "--summary_json",
            str(output_dir / "summary.json"),
        ]
    )

    limited, _metadata = load_patch_selector_training_set_npz(written_cache)
    assert limited.sample_count == 2
    assert not (output_dir / "transform.npz").exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["cache_only"] is True
    assert summary["sample_summary"]["sample_count"] == 2


def test_stage_c1_cli_reuses_sample_cache(tmp_path: Path, monkeypatch):
    import feature_extract.tools.vfm.train_stage_c1_patch_selector as cli
    from feature_extract.vfm.patch_selector_training import (
        PatchSelectorTrainingSet,
        save_patch_selector_training_set_npz,
    )

    samples = PatchSelectorTrainingSet(
        query_features=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        positive_features=np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        positive_mask=np.asarray([[True]], dtype=bool),
        negative_features=np.asarray([[[0.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        metadata={"source": "cached"},
    )
    sample_cache = tmp_path / "samples.npz"
    save_patch_selector_training_set_npz(samples, sample_cache)

    def fail_if_builder_runs(*_args, **_kwargs):
        raise AssertionError("sample builder should not run when --sample_cache is provided")

    monkeypatch.setattr(cli, "build_patch_selector_samples_for_query", fail_if_builder_runs)
    output_dir = tmp_path / "out"
    cli.main(
        [
            "--query_manifest",
            str(tmp_path / "unused_manifest.json"),
            "--landmark_bank",
            str(tmp_path / "unused_bank.npz"),
            "--track_observations",
            str(tmp_path / "unused_tracks.jsonl"),
            "--query_pose_file",
            str(tmp_path / "unused_poses.txt"),
            "--sample_cache",
            str(sample_cache),
            "--output_dim",
            "2",
            "--steps",
            "2",
            "--batch_size",
            "1",
            "--output_transform",
            str(output_dir / "transform.npz"),
            "--summary_json",
            str(output_dir / "summary.json"),
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["sample_summary"]["sample_count"] == 1
    assert summary["sample_summary"]["sample_cache"] == str(sample_cache)
    assert summary["training"]["output_dim"] == 2
