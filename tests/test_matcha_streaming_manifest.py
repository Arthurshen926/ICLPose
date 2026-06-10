from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_matcha_streaming_manifest import parse_args as parse_streaming_args
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import (
    _records_for_curriculum,
    parse_args as parse_streaming_train_args,
)
from feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose import (
    _fine_confidence_diagnostics_from_rows,
    _fine_offset_diagnostics,
    _parse_render_rgb_eval_args,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.matcha_streaming_manifest import (
    MatchaStreamingPairManifest,
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
        ]
    )

    assert args.pair_type_curriculum == "robust_default"
    assert args.freeze_descriptor_steps == 2
    assert args.descriptor_lr_scale == 0.05
    assert args.visibility_alpha_threshold == 0.25
    assert args.depth_edge_threshold_m == 0.4


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
