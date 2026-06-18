from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_matcha_streaming_manifest import main as build_streaming_manifest_main
from feature_extract.tools.vfm.build_matcha_streaming_manifest import parse_args as parse_streaming_args
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import (
    _build_streaming_pair_with_retries,
    _builder_class_for_manifest,
    _label_entropy_bits,
    _is_head_parameter,
    _parse_pair_type_sampling_weights,
    _render_subcell_seed_xy,
    _records_for_curriculum,
    _select_records_by_pair_type_weights,
    _select_multiview_support_pose_records,
    _supervision_balance_summary,
    _synthetic_supervision_overlap_fraction,
    _synthetic_pose_bin_counts,
    _load_synthetic_training_sample_cache,
    _remember_synthetic_training_sample_memory_cache,
    _save_synthetic_training_sample_cache,
    _synthetic_training_sample_memory_cache_get,
    _synthetic_training_sample_cache_key,
    _select_training_source_for_step,
    _share_builder_runtime_state,
    _split_optional_manifest_paths,
    TrainingSource,
    parse_args as parse_streaming_train_args,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineTrainingSet
from feature_extract.vfm.matcha_joint_training import MatchaJointTrainingSet
from feature_extract.vfm.matcha_coarse_supervision import cell_offset_labels
from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _StreamingCsvWriter,
    _fine_confidence_diagnostics_from_rows,
    _fine_offset_diagnostics,
    _matcha_query_token_cache_path,
    _parse_render_rgb_eval_args,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord, pose_w2c_from_center_rotation
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.matcha_streaming_manifest import (
    MatchaStreamingPairManifest,
    MatchaStreamingPairRecord,
    build_streaming_pair_records,
)


def test_streaming_pair_manifest_round_trip_is_tensor_free(tmp_path: Path) -> None:
    records = build_streaming_pair_records(
        ["seq1/frame00001.png", "seq1/frame00002.png"],
        split="train",
        pair_types=["A_gt", "B_trans025", "D_reference"],
        pair_type_ids={"A_gt": 0, "B_trans025": 1, "D_reference": 3},
        seed=17,
        candidate_ids={"seq1/frame00001.png": "cand0", "seq1/frame00002.png": "cand1"},
    )
    manifest = MatchaStreamingPairManifest(
        records=records,
        metadata={"source_query_manifest": "train_manifest.json"},
    )
    path = tmp_path / "streaming_manifest.json"
    manifest.to_json(path)

    payload = json.loads(path.read_text())
    assert payload["format"] == "vfm_matcha_streaming_pair_manifest_v1"
    assert payload["query_count"] == 2
    assert payload["pair_count"] == 6
    assert payload["pair_type_counts"] == {"A_gt": 2, "B_trans025": 2, "D_reference": 2}
    assert "query_feature_maps" not in payload
    assert "render_feature_maps" not in payload

    loaded = MatchaStreamingPairManifest.from_json(path)
    assert loaded.query_count == 2
    assert loaded.records[-1].candidate_id == "cand1"


def test_build_streaming_manifest_cli_uses_full_dataset_by_default() -> None:
    args = parse_streaming_args(
        [
            "--query_manifest",
            "train_manifest.json",
            "--output_manifest",
            "streaming.json",
            "--summary_json",
            "summary.json",
            "--split_name",
            "train",
            "--pair_types",
            "A_gt,B_trans025,C_trans050",
        ]
    )

    assert args.max_queries == 0
    assert args.view_selection == "prefix"


def test_build_streaming_manifest_accepts_curriculum_perturbation_pair_types() -> None:
    args = parse_streaming_args(
        [
            "--query_manifest",
            "train_manifest.json",
            "--output_manifest",
            "streaming.json",
            "--summary_json",
            "summary.json",
            "--split_name",
            "train",
            "--pair_types",
            "A_gt,B_trans005,B_trans010,B_trans025,D_reference",
            "--candidate_bank",
            "candidates.jsonl",
        ]
    )

    assert args.pair_types == "A_gt,B_trans005,B_trans010,B_trans025,D_reference"


def test_build_streaming_manifest_accepts_real_render_query_metadata_flags() -> None:
    args = parse_streaming_args(
        [
            "--query_manifest",
            "train_manifest.json",
            "--output_manifest",
            "streaming.json",
            "--summary_json",
            "summary.json",
            "--split_name",
            "train",
            "--pair_types",
            "A_gt,B_trans025",
            "--pair_source",
            "real_perturbed_render",
            "--query_pose_file",
            "dataset_train.txt",
            "--validation_manifest",
            "val_manifest.json",
        ]
    )

    assert args.pair_source == "real_perturbed_render"
    assert args.query_pose_file == "dataset_train.txt"
    assert args.validation_manifest == "val_manifest.json"


def test_build_streaming_manifest_requires_candidate_bank_for_reference_pairs() -> None:
    with pytest.raises(ValueError, match="candidate_bank"):
        parse_streaming_args(
            [
                "--query_manifest",
                "train_manifest.json",
                "--output_manifest",
                "streaming.json",
                "--summary_json",
                "summary.json",
                "--split_name",
                "train",
                "--pair_types",
                "A_gt,D_reference",
            ]
        )


def test_build_streaming_manifest_rejects_reference_pairs_without_candidate_coverage(tmp_path: Path) -> None:
    query_manifest = tmp_path / "train_manifest.json"
    candidate_bank = tmp_path / "candidates.jsonl"
    query_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"image_id": "q0.png", "token_path": "q0.npz", "layers": [], "split": "train", "scene": "OldHospital"},
                    {"image_id": "q1.png", "token_path": "q1.npz", "layers": [], "split": "train", "scene": "OldHospital"},
                ]
            }
        )
        + "\n"
    )
    candidate_bank.write_text(
        json.dumps(
            {
                "record_type": "header",
                "protocol_name": "test_reference_bank",
                "protocol_kind": "reference_pose",
                "protocol_fingerprint": "",
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "candidate",
                "query_id": "q0.png",
                "candidate_id": "q0:reference:000",
                "candidate_type": "reference_pose",
                "pose": np.eye(4).tolist(),
                "prior_score": 1.0,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="missing D_reference candidates"):
        build_streaming_manifest_main(
            [
                "--query_manifest",
                str(query_manifest),
                "--output_manifest",
                str(tmp_path / "streaming.json"),
                "--summary_json",
                str(tmp_path / "summary.json"),
                "--split_name",
                "train",
                "--pair_types",
                "A_gt,D_reference",
                "--candidate_bank",
                str(candidate_bank),
            ]
        )


def test_supervision_balance_summary_reports_positive_and_no_match_counts() -> None:
    rows = [
        {"sample_count": 10, "supervision_no_match_count": 5},
        {"sample_count": 20, "supervision_no_match_count": 0},
    ]

    summary = _supervision_balance_summary(rows)

    assert summary["row_count"] == 2
    assert summary["positive_match_count"] == 30
    assert summary["no_match_count"] == 5
    assert summary["rows_with_no_match"] == 1
    assert summary["rows_without_no_match"] == 1
    assert summary["no_match_to_positive_ratio"] == pytest.approx(5.0 / 30.0)


def test_streaming_training_cli_does_not_require_dense_joint_cache() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--steps",
            "3",
        ]
    )

    assert args.streaming_manifest == "streaming_train.json"
    assert args.steps == 3
    assert not hasattr(args, "joint_cache")
    assert args.query_feature_cache_dtype == "float16"
    assert args.synthetic_pair_cache_dir == ""
    assert args.synthetic_pair_cache_format == "npz"


def test_streaming_training_cli_accepts_synthetic_pair_cache_dir() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--synthetic_pair_cache_dir",
            "cache/synthetic_pairs",
            "--synthetic_pair_cache_format",
            "compressed_npz",
        ]
    )

    assert args.synthetic_pair_cache_dir == "cache/synthetic_pairs"
    assert args.synthetic_pair_cache_format == "compressed_npz"


def test_streaming_training_cli_accepts_secondary_synthetic_manifest() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "real_train.json",
            "--synthetic_streaming_manifest",
            "synthetic_train.json",
            "--synthetic_sampling_weight",
            "2.5",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
        ]
    )

    assert args.streaming_manifest == "real_train.json"
    assert args.synthetic_streaming_manifest == "synthetic_train.json"
    assert args.synthetic_sampling_weight == 2.5


def test_streaming_training_splits_multiple_synthetic_manifests() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "real_train.json",
            "--synthetic_streaming_manifest",
            "synth_micro.json,synth_small.json,synth_medium.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
        ]
    )

    assert _split_optional_manifest_paths(args.synthetic_streaming_manifest) == [
        "synth_micro.json",
        "synth_small.json",
        "synth_medium.json",
    ]
    assert _split_optional_manifest_paths("") == []


def test_streaming_training_cli_defaults_to_real_heavy_sampling_and_pair_weights() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "real_train.json",
            "--synthetic_streaming_manifest",
            "synthetic_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
        ]
    )

    weights = _parse_pair_type_sampling_weights(args.real_pair_type_sampling_weights)
    assert args.real_sampling_weight == pytest.approx(0.70)
    assert args.synthetic_sampling_weight == pytest.approx(0.30)
    assert weights["A_gt"] == pytest.approx(0.30)
    assert weights["B_trans005"] == pytest.approx(0.25)
    assert weights["D_reference"] == pytest.approx(0.10)


def test_training_source_selection_keeps_real_and_synthetic_builders_separate() -> None:
    real_manifest = MatchaStreamingPairManifest(
        records=(
            MatchaStreamingPairRecord(
                query_id="seq0/frame00001.png",
                split="train",
                pair_type="A_gt",
                pair_type_id=0,
                record_index=0,
                pair_index=0,
                seed=7,
            ),
        ),
        metadata={"source_query_manifest": "train_manifest.json"},
    )
    synthetic_manifest = MatchaStreamingPairManifest(
        records=(
            MatchaStreamingPairRecord(
                query_id="seq0/frame00001.png",
                split="train",
                pair_type="S2DGS_RANDOM",
                pair_type_id=100,
                record_index=0,
                pair_index=0,
                seed=11,
            ),
        ),
        metadata={
            "pair_source": "2dgs_synthetic",
            "source_query_manifest": "train_manifest.json",
        },
    )

    selection = _select_training_source_for_step(
        [
            TrainingSource(name="real", manifest=real_manifest, builder="real_builder", sampling_weight=0.0),
            TrainingSource(name="synthetic", manifest=synthetic_manifest, builder="synthetic_builder", sampling_weight=1.0),
        ],
        rng=np.random.default_rng(123),
        step=0,
        total_steps=10,
        curriculum="robust_default",
    )

    assert selection.name == "synthetic"
    assert selection.builder == "synthetic_builder"
    assert selection.records[0].pair_type == "S2DGS_RANDOM"


def test_pair_type_weighted_record_selection_prefers_configured_real_pair_type() -> None:
    records = build_streaming_pair_records(
        ["q0"],
        split="train",
        pair_types=["A_gt", "B_trans005", "D_reference"],
        pair_type_ids={"A_gt": 0, "B_trans005": 4, "D_reference": 3},
        seed=5,
        candidate_ids={"q0": "cand"},
    )

    selected = _select_records_by_pair_type_weights(
        records,
        weights={"A_gt": 0.0, "B_trans005": 0.0, "D_reference": 1.0},
    )

    assert [record.pair_type for record in selected] == ["D_reference"]


def test_share_builder_runtime_state_reuses_heavy_objects() -> None:
    class Builder:
        pass

    primary = Builder()
    secondary = Builder()
    primary.radio = object()
    primary.rgb_source = object()
    primary.keypoint_extractor = object()
    secondary.radio = object()
    secondary.rgb_source = object()
    secondary.keypoint_extractor = object()

    _share_builder_runtime_state(primary, secondary)

    assert secondary.radio is primary.radio
    assert secondary.rgb_source is primary.rgb_source
    assert secondary.keypoint_extractor is primary.keypoint_extractor


def _toy_joint_training_set() -> MatchaJointTrainingSet:
    features = np.eye(4, dtype=np.float32)
    coarse = MatchaCoarseFineTrainingSet(
        query_features=features,
        render_features=features.copy(),
        query_offset_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
        render_offset_labels=np.asarray([3, 2, 1, 0], dtype=np.int64),
        negative_render_features=np.roll(features, shift=1, axis=0)[:, None, :],
        roundtrip_errors_px=np.zeros((4,), dtype=np.float32),
        metadata={"toy": True},
    )
    return MatchaJointTrainingSet(
        coarse_fine_samples=coarse,
        query_feature_maps=features.T.reshape(1, 4, 2, 2),
        render_feature_maps=features.T.reshape(1, 4, 2, 2).copy(),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        pair_type_ids=np.asarray([100], dtype=np.int64),
        pair_type_names=np.asarray(["S2DGS_RANDOM"], dtype=object),
        pair_query_ids=np.asarray(["seq0/frame00001.png"], dtype=object),
        pair_split_names=np.asarray(["train"], dtype=object),
    )


def test_synthetic_training_sample_cache_round_trips_sample_and_row(tmp_path: Path) -> None:
    sample = _toy_joint_training_set()
    row = {
        "query_id": "seq0/frame00001.png",
        "pair_type": "S2DGS_RANDOM",
        "synthetic_pose_bin": "micro",
        "synthetic_overlap_fraction": 1.0,
    }

    _save_synthetic_training_sample_cache(tmp_path, "abc123", sample, row, compressed=False)
    loaded = _load_synthetic_training_sample_cache(tmp_path, "abc123")

    assert loaded is not None
    loaded_sample, loaded_row = loaded
    assert loaded_sample.coarse_fine_samples.sample_count == sample.coarse_fine_samples.sample_count
    assert loaded_sample.query_feature_maps is not None
    assert loaded_sample.query_feature_maps.shape == (1, 4, 2, 2)
    assert loaded_row["query_id"] == "seq0/frame00001.png"
    assert loaded_row["synthetic_pose_bin"] == "micro"


def test_synthetic_training_sample_cache_key_changes_with_relevant_settings() -> None:
    record = MatchaStreamingPairRecord(
        query_id="seq0/frame00001.png",
        split="train",
        pair_type="S2DGS_RANDOM",
        pair_type_id=100,
        record_index=0,
        pair_index=0,
        seed=17,
    )
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--render_width",
            "640",
            "--render_height",
            "480",
        ]
    )
    changed_args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--render_width",
            "640",
            "--render_height",
            "480",
            "--feature_fusion_mode",
            "local_attention",
        ]
    )
    changed_format_args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--render_width",
            "640",
            "--render_height",
            "480",
            "--synthetic_pair_cache_format",
            "compressed_npz",
        ]
    )
    metadata = {"pair_source": "2dgs_synthetic", "synthetic_target_translation_range_m": [0.0, 0.03]}
    camera = ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(520.0, 520.0, 320.0, 240.0))
    source_pose = np.eye(4, dtype=np.float64)
    target_pose = pose_w2c_from_center_rotation(np.asarray([0.01, 0.0, 0.0]), np.eye(3, dtype=np.float64))

    first = _synthetic_training_sample_cache_key(
        record,
        manifest_metadata=metadata,
        args=args,
        query_camera=camera,
        render_camera=camera,
        source_pose_w2c=source_pose,
        target_pose_w2c=target_pose,
    )
    second = _synthetic_training_sample_cache_key(
        record,
        manifest_metadata=metadata,
        args=changed_args,
        query_camera=camera,
        render_camera=camera,
        source_pose_w2c=source_pose,
        target_pose_w2c=target_pose,
    )
    third = _synthetic_training_sample_cache_key(
        record,
        manifest_metadata=metadata,
        args=changed_format_args,
        query_camera=camera,
        render_camera=camera,
        source_pose_w2c=source_pose,
        target_pose_w2c=target_pose,
    )

    assert len(first) == 64
    assert first != second
    assert first != third


def test_synthetic_training_sample_memory_cache_keeps_most_recent_items() -> None:
    cache = {}
    first = _toy_joint_training_set()
    second = _toy_joint_training_set()
    _remember_synthetic_training_sample_memory_cache(cache, "first", first, {"query_id": "a"}, max_items=1)
    _remember_synthetic_training_sample_memory_cache(cache, "second", second, {"query_id": "b"}, max_items=1)

    assert _synthetic_training_sample_memory_cache_get(cache, "first") is None
    cached = _synthetic_training_sample_memory_cache_get(cache, "second")
    assert cached is not None
    cached_sample, cached_row = cached
    assert cached_sample.coarse_fine_samples.sample_count == second.coarse_fine_samples.sample_count
    assert cached_row["query_id"] == "b"


def test_streaming_training_cli_defaults_to_matcha_original_attention() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
        ]
    )

    assert args.model_type == "radio_dual_attention"
    assert args.attention_fusion_mode == "matcha_original"
    assert args.fine_supervision_source == "render_subcell_stratified"
    assert args.pair_fine_loss_weight == 0.0
    assert args.query_pair_fine_loss_weight == 0.0
    assert args.local_window_fine_loss_weight > 0.0
    assert args.patch_correlation_loss_weight == 0.0
    assert not args.merge_fine_labels_into_coarse


def test_streaming_training_cli_exposes_multiview_supervision_controls() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--multiview_supervision_support_views",
            "3",
            "--multiview_supervision_min_support_views",
            "2",
            "--multiview_supervision_depth_tolerance_m",
            "0.08",
        ]
    )

    assert args.multiview_supervision_support_views == 3
    assert args.multiview_supervision_min_support_views == 2
    assert args.multiview_supervision_depth_tolerance_m == 0.08

    defaults = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
        ]
    )
    assert defaults.multiview_supervision_support_views == 3
    assert defaults.multiview_supervision_min_support_views == 1
    assert defaults.multiview_supervision_depth_tolerance_m > 0.0


def test_streaming_training_radio_matcha_patch_corr_preset_selects_new_fine_path() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--matcha_train_preset",
            "radio_matcha_patch_corr",
            "--builder_device",
            "cuda:0",
        ]
    )

    assert args.builder_device == "cuda:0"
    assert args.fine_supervision_source == "render_subcell_stratified"
    assert args.multiview_supervision_support_views == 3
    assert args.multiview_supervision_min_support_views == 1
    assert args.offset_loss_weight == 0.0
    assert args.pair_fine_loss_weight == 0.0
    assert args.query_pair_fine_loss_weight == 0.0
    assert args.local_window_fine_loss_weight == 0.0
    assert args.patch_correlation_loss_weight == 0.0
    assert args.patch_corr_fine_loss_weight == 1.0
    assert args.patch_corr_fine_batch_size == 256
    assert args.patch_corr_fine_max_samples_per_pair == 512
    assert not args.patch_corr_fine_backprop_context


def test_streaming_training_radio_matcha_2dgs_synthetic_preset_uses_local_window_path() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "synthetic_train.json",
            "--validation_streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--matcha_train_preset",
            "radio_matcha_2dgs_synthetic",
        ]
    )

    assert args.offset_loss_weight == 0.25
    assert args.pair_confidence_loss_weight == 0.1
    assert args.dense_heatmap_loss_weight == 0.25
    assert args.local_window_fine_loss_weight == 1.0
    assert args.patch_corr_fine_loss_weight == 0.0
    assert args.multiview_supervision_support_views == 0


def test_synthetic_manifest_routes_to_synthetic_builder() -> None:
    manifest = MatchaStreamingPairManifest(
        records=(
            MatchaStreamingPairRecord(
                query_id="seq0/frame00001.png",
                split="train",
                pair_type="S2DGS_RANDOM",
                pair_type_id=100,
                record_index=0,
                pair_index=0,
                seed=7,
            ),
        ),
        metadata={
            "pair_source": "2dgs_synthetic",
            "source_query_manifest": "train_manifest.json",
        },
    )

    assert _builder_class_for_manifest(manifest).__name__ == "SyntheticStreamingPairBuilder"


def test_synthetic_pose_bin_counts_summarizes_training_rows() -> None:
    rows = [
        {"synthetic_pose_bin": "micro"},
        {"synthetic_pose_bin": "small"},
        {"synthetic_pose_bin": "small"},
        {"synthetic_pose_bin": ""},
        {"pair_type": "A_gt"},
    ]

    assert _synthetic_pose_bin_counts(rows) == {"micro": 1, "small": 2}


def test_synthetic_supervision_overlap_fraction_uses_smaller_grid_area() -> None:
    assert _synthetic_supervision_overlap_fraction(25, (10, 10), (5, 10)) == pytest.approx(0.5)
    assert _synthetic_supervision_overlap_fraction(0, (10, 10), (5, 10)) == 0.0
    assert _synthetic_supervision_overlap_fraction(3, (0, 10), (5, 10)) == 3.0


def test_build_streaming_pair_with_retries_skips_transient_builder_failures() -> None:
    record = MatchaStreamingPairRecord(
        query_id="seq0/frame00001.png",
        split="train",
        pair_type="S2DGS_RANDOM",
        pair_type_id=100,
        record_index=0,
        pair_index=0,
        seed=7,
    )

    class FailsOnceBuilder:
        def __init__(self) -> None:
            self.calls = 0

        def build(self, item: MatchaStreamingPairRecord):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("synthetic pair overlap fraction below minimum")
            return object(), {"query_id": item.query_id}

    builder = FailsOnceBuilder()

    _samples, row, skipped = _build_streaming_pair_with_retries([record], builder, seed=123)

    assert row == {"query_id": "seq0/frame00001.png"}
    assert skipped == 1
    assert builder.calls == 2


def test_patch_corr_fine_head_is_optimized_with_head_lr_group() -> None:
    assert _is_head_parameter("patch_corr_fine_head.score.0.weight")


def test_render_subcell_seed_xy_is_deterministic_and_non_center() -> None:
    seeds_a = _render_subcell_seed_xy(
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        seed=123,
    )
    seeds_b = _render_subcell_seed_xy(
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        seed=123,
    )
    labels, valid = cell_offset_labels(
        seeds_a,
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
    )

    assert seeds_a.shape == (4, 2)
    assert np.allclose(seeds_a, seeds_b)
    assert np.all((seeds_a[:, 0] >= 0.0) & (seeds_a[:, 0] < 16.0))
    assert np.all((seeds_a[:, 1] >= 0.0) & (seeds_a[:, 1] < 16.0))
    assert valid.tolist() == [True, True, True, True]
    assert np.unique(labels).size > 1
    assert not np.all(labels == 36)
    assert _label_entropy_bits(labels) > 0.0


def test_multiview_support_pose_selection_is_nearest_and_excludes_query() -> None:
    rotation = np.eye(3, dtype=np.float64)

    def pose(image_id: str, center_x: float) -> CambridgePoseRecord:
        center = np.asarray([center_x, 0.0, 0.0], dtype=np.float64)
        return CambridgePoseRecord(
            image_id=image_id,
            camera_center=center,
            rotation_w2c=rotation,
            pose_w2c=pose_w2c_from_center_rotation(center, rotation),
        )

    selected = _select_multiview_support_pose_records(
        {
            "query.png": pose("query.png", 0.0),
            "near.png": pose("near.png", 0.2),
            "far.png": pose("far.png", 2.0),
            "nearer.png": pose("nearer.png", -0.1),
        },
        query_id="query.png",
        target_pose_w2c=pose_w2c_from_center_rotation(np.asarray([0.0, 0.0, 0.0], dtype=np.float64), rotation),
        max_count=2,
    )

    assert [record.image_id for record in selected] == ["nearer.png", "near.png"]


def test_streaming_training_cli_exposes_robust_curriculum_and_freeze_controls() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "streaming_train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--steps",
            "3",
            "--pair_type_curriculum",
            "robust_default",
            "--freeze_descriptor_steps",
            "2",
            "--descriptor_lr_scale",
            "0.05",
            "--visibility_alpha_threshold",
            "0.25",
            "--depth_edge_threshold_m",
            "0.4",
            "--coarse_candidate_rank_loss_weight",
            "0.3",
            "--coarse_candidate_rank_margin",
            "0.4",
        ]
    )

    assert args.pair_type_curriculum == "robust_default"
    assert args.freeze_descriptor_steps == 2
    assert args.descriptor_lr_scale == 0.05
    assert args.visibility_alpha_threshold == 0.25
    assert args.depth_edge_threshold_m == 0.4
    assert args.coarse_candidate_rank_loss_weight == 0.3
    assert args.coarse_candidate_rank_margin == 0.4


def test_streaming_csv_writer_writes_rows_without_retaining_them(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    writer = _StreamingCsvWriter(path)
    writer.writerows(
        [
            {"query_id": "q0", "patch_correct": True},
            {"query_id": "q1", "patch_correct": False},
        ]
    )
    writer.writerows([{"query_id": "q2", "patch_correct": True}])
    writer.close()

    assert writer.row_count == 3
    assert path.read_text().splitlines() == [
        "patch_correct,query_id",
        "True,q0",
        "False,q1",
        "True,q2",
    ]


def test_matcha_query_token_cache_path_reuses_streaming_feature_cache_name(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    streaming_style = cache_dir / "seq1__frame00001.png_1920x1080_radio_dual.npz"
    streaming_style.write_bytes(b"cache")

    path = _matcha_query_token_cache_path(
        cache_dir,
        "seq1/frame00001.png",
        render_width=1920,
        render_height=1080,
        layer_name="radio_dual",
    )

    assert path == streaming_style


def test_conservative_curriculum_never_samples_far_or_reference_pairs() -> None:
    records = build_streaming_pair_records(
        ["q0"],
        split="train",
        pair_types=["A_gt", "B_trans005", "B_trans010", "B_trans025", "C_trans050", "D_reference"],
        pair_type_ids={"A_gt": 0, "B_trans005": 4, "B_trans010": 5, "B_trans025": 1, "C_trans050": 2, "D_reference": 3},
        seed=5,
        candidate_ids={"q0": "cand"},
    )

    early = _records_for_curriculum(records, step=0, total_steps=100, mode="conservative_25cm")
    mid = _records_for_curriculum(records, step=55, total_steps=100, mode="conservative_25cm")
    late = _records_for_curriculum(records, step=90, total_steps=100, mode="conservative_25cm")

    assert {record.pair_type for record in early} == {"A_gt"}
    assert {record.pair_type for record in mid} == {"A_gt", "B_trans005", "B_trans010"}
    assert {record.pair_type for record in late} == {"A_gt", "B_trans005", "B_trans010", "B_trans025"}


def test_render_rgb_eval_conservative_confidence_preset_enables_pnp_confidence() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_eval_preset",
            "conservative_confidence",
        ]
    )

    assert args.matcha_confidence_mode == "learned"
    assert args.pnp_soft_order_mode == "confidence"
    assert args.coverage_filter_min_confidence > 0.0
    assert args.measurement_sigma_px > 0.0


def test_render_rgb_eval_radio_matcha_local_search_preset_avoids_pair_mlp_fine() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_eval_preset",
            "radio_matcha_local_search",
        ]
    )

    assert not args.matcha_use_pair_fine_head
    assert args.matcha_fine_mode == "fine_attention_argmax"
    assert args.fine_render_search_radius_px > 0.0
    assert args.matcha_confidence_mode == "learned"
    assert args.pnp_soft_order_mode == "confidence"
    assert args.coverage_filter_min_confidence > 0.0
    assert args.measurement_sigma_px > 0.0


def test_render_rgb_eval_radio_matcha_local_window_preset_uses_learned_render_fine() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_eval_preset",
            "radio_matcha_local_window",
        ]
    )

    assert args.matcha_use_local_window_fine_head
    assert not args.matcha_use_pair_fine_head
    assert args.matcha_local_window_confidence_blend > 0.0
    assert args.fine_render_search_radius_px == 0.0
    assert args.render_side_local_offset_radius_cells == 1
    assert args.render_side_local_offset_top_k_per_query == 3
    assert args.matcha_confidence_mode == "learned"
    assert args.pnp_soft_order_mode == "confidence"


def test_render_rgb_eval_radio_matcha_patch_corr_preset_uses_render_side_patch_fine() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_eval_preset",
            "radio_matcha_patch_corr",
        ]
    )

    assert args.matcha_use_patch_corr_fine_head
    assert args.matcha_patch_corr_target_side == "render"
    assert not args.matcha_use_pair_fine_head
    assert not args.matcha_use_local_window_fine_head
    assert args.fine_render_search_radius_px == 0.0
    assert args.post_pair_render_refine_radius_px == 0.0
    assert args.matcha_confidence_mode == "learned"
    assert args.pnp_soft_order_mode == "confidence"


def test_render_rgb_eval_accepts_coarse_candidate_ranker_args() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_coarse_top_k_per_query",
            "5",
            "--matcha_coarse_mutual_mode",
            "annotate",
            "--coarse_candidate_ranker_model",
            "ranker.json",
            "--coarse_candidate_ranker_feature_set",
            "coarse",
            "--coarse_candidate_ranker_blend",
            "0.75",
        ]
    )

    assert args.matcha_coarse_top_k_per_query == 5
    assert args.matcha_coarse_mutual_mode == "annotate"
    assert args.coarse_candidate_ranker_model == "ranker.json"
    assert args.coarse_candidate_ranker_feature_set == "coarse"
    assert args.coarse_candidate_ranker_blend == 0.75


def test_render_rgb_eval_accepts_local_window_fine_head_flag() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "test.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--matcha_use_local_window_fine_head",
        ]
    )

    assert args.matcha_use_local_window_fine_head is True


def test_render_rgb_eval_accepts_query_start_index_for_sharded_exports() -> None:
    args = _parse_render_rgb_eval_args(
        [
            "--query_manifest",
            "train.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_dir",
            "eval",
            "--start_index",
            "64",
            "--max_queries",
            "32",
        ]
    )

    assert args.start_index == 64
    assert args.max_queries == 32


def test_fine_confidence_diagnostics_from_rows_reports_ece_and_offset_stats() -> None:
    rows = [
        {
            "match_count": 10,
            "fine_pair_applied_count": 8,
            "fine_pair_mean_confidence": 0.8,
            "fine_pair_mean_entropy": 1.0,
            "mean_dual_softmax_confidence": 0.9,
            "pnp_inlier_gt_precision_16px": 1.0,
            "pnp_match_confidence_mean": 0.8,
            "pnp_inlier_confidence_mean": 0.9,
            "pnp_outlier_confidence_mean": 0.2,
            "pnp_match_measurement_sigma_mean": 12.0,
            "pnp_match_patch_offset_applied_ratio": 1.0,
            "pnp_match_patch_offset_norm_mean_px": 2.0,
        },
        {
            "match_count": 10,
            "fine_pair_applied_count": 5,
            "fine_pair_mean_confidence": 0.2,
            "fine_pair_mean_entropy": 3.0,
            "mean_dual_softmax_confidence": 0.2,
            "pnp_inlier_gt_precision_16px": 0.0,
            "pnp_match_confidence_mean": 0.2,
            "pnp_inlier_confidence_mean": 0.1,
            "pnp_outlier_confidence_mean": 0.3,
            "pnp_match_measurement_sigma_mean": 24.0,
            "pnp_match_patch_offset_applied_ratio": 0.0,
            "pnp_match_patch_offset_norm_mean_px": 0.0,
        },
    ]

    diagnostics = _fine_confidence_diagnostics_from_rows(rows)

    assert diagnostics["mean_fine_pair_applied_ratio"] == 0.65
    assert diagnostics["mean_pnp_confidence_inlier_gap"] == 0.25
    assert diagnostics["confidence_ece_16px"] >= 0.0
    assert diagnostics["mean_pnp_measurement_sigma_px"] == 18.0


def test_fine_offset_diagnostics_measure_refined_query_measurement_accuracy() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=64, height=64, params=(100.0, 100.0, 0.0, 0.0))
    pose = np.eye(4, dtype=np.float64)
    match = QueryTo3DMatch(
        token_index=5,
        xy=np.asarray([20.0, 20.0], dtype=np.float64),
        track_id=7,
        xyz=np.asarray([0.2, 0.2, 1.0], dtype=np.float64),
        similarity=0.5,
        ratio=0.0,
        landmark_variance=0.0,
        pnp_soft_score=0.9,
        measurement_sigma_px=4.0,
    )

    diagnostics = _fine_offset_diagnostics(
        [match],
        pose,
        camera,
        query_grid_width=4,
        query_grid_height=4,
    )

    assert diagnostics["fine_offset_eval_count"] == 1
    assert diagnostics["fine_offset_before_median_px"] > 0.0
    assert diagnostics["fine_offset_after_median_px"] == pytest.approx(0.0)
    assert diagnostics["fine_offset_improved_ratio"] == 1.0
    assert diagnostics["fine_offset_gt16_after"] == 1.0
    assert diagnostics["fine_offset_uncertainty_within_1sigma"] == 1.0
