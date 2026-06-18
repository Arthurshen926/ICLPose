from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest import main as build_synthetic_manifest_main
from feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest import (
    _format_range_default,
    parse_args as parse_synthetic_manifest_args,
)
from feature_extract.vfm.cambridge_pose_lattice import pose_w2c_from_center_rotation
from feature_extract.vfm.matcha_synthetic_pairs import (
    DEFAULT_TARGET_ROTATION_RANGE_DEG,
    DEFAULT_TARGET_TRANSLATION_RANGE_M,
    SYNTHETIC_PAIR_SOURCE,
    SyntheticPairSamplingConfig,
    relative_pose_bin,
    sample_synthetic_pair_poses,
    synthetic_config_from_metadata,
)
from feature_extract.vfm.render_pose_protocol import render_pose_error_fields


def _pose(center_x: float = 0.0) -> np.ndarray:
    return pose_w2c_from_center_rotation(
        np.asarray([center_x, 0.0, 0.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )


def _write_token_manifest(path: Path, *, record_count: int = 2) -> None:
    records = []
    for index in range(1, record_count + 1):
        records.append(
            {
                "image_id": f"seq0/frame{index:05d}.png",
                "token_path": f"seq0__frame{index:05d}.npz",
                "layers": [
                    {
                        "name": "radio_dual",
                        "model": "radio",
                        "layer": "dual",
                        "channels": 1280,
                        "stride": 16,
                    }
                ],
                "split": "train",
                "scene": "OldHospital",
            }
        )
    payload = {
        "records": records,
    }
    path.write_text(json.dumps(payload) + "\n")


def _build_synthetic_manifest(tmp_path: Path, *, record_count: int = 2, extra_args: list[str] | None = None) -> dict:
    query_manifest = tmp_path / "train_manifest.json"
    output_manifest = tmp_path / "synthetic_train.json"
    summary_json = tmp_path / "summary.json"
    _write_token_manifest(query_manifest, record_count=record_count)

    build_synthetic_manifest_main(
        [
            "--query_manifest",
            str(query_manifest),
            "--output_manifest",
            str(output_manifest),
            "--summary_json",
            str(summary_json),
            "--split_name",
            "train",
            *(extra_args or []),
        ]
    )

    return json.loads(output_manifest.read_text())


def test_synthetic_config_from_metadata_parses_ranges() -> None:
    config = synthetic_config_from_metadata(
        {
            "pair_source": SYNTHETIC_PAIR_SOURCE,
            "synthetic_source_translation_range_m": [0.0, 0.02],
            "synthetic_source_rotation_range_deg": [0.0, 0.5],
            "synthetic_target_translation_range_m": [0.03, 0.10],
            "synthetic_target_rotation_range_deg": [1.0, 3.0],
            "synthetic_min_supervision_count": 128,
            "synthetic_min_overlap": 0.25,
        }
    )

    assert config.source_translation_range_m == (0.0, 0.02)
    assert config.source_rotation_range_deg == (0.0, 0.5)
    assert config.target_translation_range_m == (0.03, 0.10)
    assert config.target_rotation_range_deg == (1.0, 3.0)
    assert config.min_supervision_count == 128
    assert config.min_overlap == 0.25


def test_synthetic_config_from_metadata_uses_design_defaults() -> None:
    config = synthetic_config_from_metadata({"pair_source": SYNTHETIC_PAIR_SOURCE})

    assert config.source_translation_range_m == (0.0, 0.03)
    assert config.source_rotation_range_deg == (0.0, 1.0)
    assert config.target_translation_range_m == (0.0, 0.25)
    assert config.target_rotation_range_deg == (0.0, 6.0)
    assert config.min_supervision_count == 128
    assert config.min_overlap == 0.20


def test_synthetic_config_rejects_wrong_pair_source() -> None:
    with pytest.raises(ValueError, match="pair_source"):
        synthetic_config_from_metadata({"pair_source": "real_query"})


def test_synthetic_config_rejects_min_overlap_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="min_overlap"):
        SyntheticPairSamplingConfig(min_overlap=1.1)


def test_synthetic_config_accepts_direct_list_ranges() -> None:
    config = SyntheticPairSamplingConfig(
        source_translation_range_m=[0.0, 0.02],
        source_rotation_range_deg=[0.0, 0.5],
        target_translation_range_m=[0.03, 0.10],
        target_rotation_range_deg=[1.0, 3.0],
    )

    assert config.source_translation_range_m == (0.0, 0.02)
    assert config.source_rotation_range_deg == (0.0, 0.5)
    assert config.target_translation_range_m == (0.03, 0.10)
    assert config.target_rotation_range_deg == (1.0, 3.0)


def test_sample_synthetic_pair_poses_is_deterministic() -> None:
    config = SyntheticPairSamplingConfig(
        source_translation_range_m=(0.0, 0.02),
        source_rotation_range_deg=(0.0, 0.5),
        target_translation_range_m=(0.03, 0.10),
        target_rotation_range_deg=(1.0, 3.0),
    )
    first = sample_synthetic_pair_poses(_pose(), config=config, seed=11, key="seq0/frame.png")
    second = sample_synthetic_pair_poses(_pose(), config=config, seed=11, key="seq0/frame.png")

    assert np.allclose(first.source_pose_w2c, second.source_pose_w2c)
    assert np.allclose(first.target_pose_w2c, second.target_pose_w2c)
    assert 0.0 <= first.source_anchor_translation_m <= 0.02
    assert 0.03 <= first.target_source_translation_m <= 0.10
    assert 1.0 <= first.target_source_rotation_deg <= 3.0
    assert first.pose_bin == "small"


def test_sample_synthetic_pair_poses_respects_fixed_measured_rotation_range() -> None:
    config = SyntheticPairSamplingConfig(
        source_translation_range_m=(0.0, 0.0),
        source_rotation_range_deg=(1.0, 1.0),
        target_translation_range_m=(0.0, 0.0),
        target_rotation_range_deg=(1.0, 1.0),
    )
    anchor_pose = _pose()
    sample = sample_synthetic_pair_poses(anchor_pose, config=config, seed=7, key="seq0/frame.png")

    _, source_rotation_deg = render_pose_error_fields(sample.source_pose_w2c, anchor_pose)
    _, target_rotation_deg = render_pose_error_fields(sample.target_pose_w2c, sample.source_pose_w2c)

    assert source_rotation_deg == pytest.approx(1.0, abs=1e-6)
    assert target_rotation_deg == pytest.approx(1.0, abs=1e-6)


def test_relative_pose_bin_uses_translation_and_rotation() -> None:
    assert relative_pose_bin(0.03, 1.0) == "micro"
    assert relative_pose_bin(0.10, 3.0) == "small"
    assert relative_pose_bin(0.25, 6.0) == "medium"
    assert relative_pose_bin(0.50, 10.0) == "wide"
    assert relative_pose_bin(0.51, 10.0) == "out_of_range"
    assert relative_pose_bin(0.50, 10.1) == "out_of_range"


def test_synthetic_manifest_cli_defaults_to_random_pair_type() -> None:
    args = parse_synthetic_manifest_args(
        [
            "--query_manifest",
            "train_manifest.json",
            "--output_manifest",
            "synthetic.json",
            "--summary_json",
            "summary.json",
            "--split_name",
            "train",
        ]
    )

    assert args.pair_type == "S2DGS_RANDOM"
    assert args.target_translation_range_m == _format_range_default(DEFAULT_TARGET_TRANSLATION_RANGE_M)
    assert args.target_rotation_range_deg == _format_range_default(DEFAULT_TARGET_ROTATION_RANGE_DEG)


def test_synthetic_manifest_pose_bin_preset_sets_expected_ranges() -> None:
    base_args = [
        "--query_manifest",
        "train_manifest.json",
        "--output_manifest",
        "synthetic.json",
        "--summary_json",
        "summary.json",
        "--split_name",
        "train",
    ]

    large = parse_synthetic_manifest_args([*base_args, "--pose_bin", "large"])
    reference_like = parse_synthetic_manifest_args([*base_args, "--pose_bin", "reference_like"])

    assert large.target_translation_range_m == "0.25,0.5"
    assert large.target_rotation_range_deg == "6.0,10.0"
    assert reference_like.target_translation_range_m == "0.5,4.5"
    assert reference_like.target_rotation_range_deg == "10.0,35.0"


def test_synthetic_manifest_cli_rejects_invalid_ranges() -> None:
    base_args = [
        "--query_manifest",
        "train_manifest.json",
        "--output_manifest",
        "synthetic.json",
        "--summary_json",
        "summary.json",
        "--split_name",
        "train",
    ]

    with pytest.raises(ValueError, match="non-negative"):
        parse_synthetic_manifest_args([*base_args, "--target_translation_range_m=-0.1,0.25"])
    with pytest.raises(ValueError, match="ordered"):
        parse_synthetic_manifest_args([*base_args, "--target_rotation_range_deg", "6.0,1.0"])


def test_build_synthetic_manifest_writes_pair_source_metadata(tmp_path: Path) -> None:
    query_manifest = tmp_path / "train_manifest.json"
    output_manifest = tmp_path / "synthetic_train.json"
    summary_json = tmp_path / "summary.json"
    _write_token_manifest(query_manifest)

    build_synthetic_manifest_main(
        [
            "--query_manifest",
            str(query_manifest),
            "--output_manifest",
            str(output_manifest),
            "--summary_json",
            str(summary_json),
            "--split_name",
            "train",
            "--seed",
            "123",
            "--target_translation_range_m",
            "0.03,0.10",
            "--target_rotation_range_deg",
            "1.0,3.0",
        ]
    )

    payload = json.loads(output_manifest.read_text())
    assert payload["pair_count"] == 2
    assert payload["pair_type_counts"] == {"S2DGS_RANDOM": 2}
    assert payload["metadata"]["pair_source"] == "2dgs_synthetic"
    assert payload["metadata"]["synthetic_target_translation_range_m"] == [0.03, 0.10]
    assert payload["records"][0]["pair_type_id"] == 100
    assert json.loads(summary_json.read_text())["stage"] == "matcha_2dgs_synthetic_manifest_builder"


def test_build_synthetic_manifest_prefix_limit_selects_first_record(tmp_path: Path) -> None:
    payload = _build_synthetic_manifest(tmp_path, extra_args=["--max_queries", "1"])

    assert payload["pair_count"] == 1
    assert payload["records"][0]["query_id"] == "seq0/frame00001.png"


def test_build_synthetic_manifest_start_index_applies_before_limit(tmp_path: Path) -> None:
    payload = _build_synthetic_manifest(tmp_path, extra_args=["--start_index", "1", "--max_queries", "1"])

    assert payload["pair_count"] == 1
    assert payload["records"][0]["query_id"] == "seq0/frame00002.png"


def test_build_synthetic_manifest_uniform_selection_is_deterministic(tmp_path: Path) -> None:
    payload = _build_synthetic_manifest(
        tmp_path,
        record_count=3,
        extra_args=["--max_queries", "2", "--view_selection", "uniform"],
    )

    assert payload["pair_count"] == 2
    assert [record["query_id"] for record in payload["records"]] == [
        "seq0/frame00001.png",
        "seq0/frame00003.png",
    ]


def test_build_synthetic_manifest_cycles_queries_to_target_pair_count(tmp_path: Path) -> None:
    payload = _build_synthetic_manifest(
        tmp_path,
        record_count=3,
        extra_args=["--target_pair_count", "7"],
    )

    assert payload["pair_count"] == 7
    assert payload["query_count"] == 3
    assert payload["metadata"]["target_pair_count"] == 7
    assert [record["query_id"] for record in payload["records"]] == [
        "seq0/frame00001.png",
        "seq0/frame00002.png",
        "seq0/frame00003.png",
        "seq0/frame00001.png",
        "seq0/frame00002.png",
        "seq0/frame00003.png",
        "seq0/frame00001.png",
    ]
    assert [record["record_index"] for record in payload["records"]] == list(range(7))


def test_build_synthetic_manifest_empty_selection_raises(tmp_path: Path) -> None:
    query_manifest = tmp_path / "train_manifest.json"
    output_manifest = tmp_path / "synthetic_train.json"
    summary_json = tmp_path / "summary.json"
    _write_token_manifest(query_manifest)

    with pytest.raises(ValueError, match="no records"):
        build_synthetic_manifest_main(
            [
                "--query_manifest",
                str(query_manifest),
                "--output_manifest",
                str(output_manifest),
                "--summary_json",
                str(summary_json),
                "--split_name",
                "train",
                "--start_index",
                "20",
            ]
        )


from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _aggregate_rows_by_bin
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _aggregate_overall_row_metrics
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _fine_offset_stats_row
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _replace_match_confidence_scores
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _matcha_eval_matches_for_pnp
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _order_pnp_matches_for_eval
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _pnp_reprojection_stats_row
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _target_depth_matches_to_pnp
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import parse_args as parse_synthetic_eval_args
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


def test_synthetic_eval_parser_requires_model_and_manifest() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
        ]
    )

    assert args.streaming_manifest == "synthetic_val.json"
    assert args.max_pairs == 0
    assert args.matcha_eval_preset == "radio_matcha_local_window"
    assert args.pose_confidence_label_source == "supervision"
    assert args.pnp_reprojection_error_px == 10.0
    assert args.pnp_refine_method == "LM"


def test_synthetic_eval_parser_accepts_patch_corr_preset() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
            "--matcha_eval_preset",
            "radio_matcha_patch_corr",
        ]
    )

    assert args.matcha_eval_preset == "radio_matcha_patch_corr"


def test_synthetic_eval_parser_accepts_pnp_reprojection_threshold() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
            "--pnp_reprojection_error_px",
            "4",
            "--pnp_refine_method",
            "none",
        ]
    )

    assert args.pnp_reprojection_error_px == 4.0
    assert args.pnp_refine_method == "NONE"


def test_synthetic_eval_parser_accepts_fine_confidence_blend() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
            "--fine_confidence_blend",
            "0.75",
        ]
    )

    assert args.fine_confidence_blend == 0.75


def test_synthetic_eval_parser_accepts_pnp_soft_order() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
            "--pnp_soft_order_mode",
            "confidence",
            "--pnp_soft_order_top_n",
            "1",
        ]
    )

    assert args.pnp_soft_order_mode == "confidence"
    assert args.pnp_soft_order_top_n == 1


def test_synthetic_eval_parser_accepts_pose_aware_pnp_selection() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_test.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
            "--pnp_pair_confidence",
            "--pnp_spatial_nms_radius_px",
            "12",
        ]
    )

    assert args.pnp_pair_confidence is True
    assert args.pnp_spatial_nms_radius_px == 12.0


def test_order_pnp_matches_for_eval_can_confidence_order_and_truncate() -> None:
    low = QueryTo3DMatch(
        token_index=0,
        xy=np.asarray([0.0, 0.0]),
        track_id=0,
        xyz=np.asarray([0.0, 0.0, 1.0]),
        similarity=0.1,
        ratio=1.0,
        landmark_variance=0.0,
        pnp_soft_score=0.1,
    )
    high = QueryTo3DMatch(
        token_index=1,
        xy=np.asarray([1.0, 0.0]),
        track_id=1,
        xyz=np.asarray([1.0, 0.0, 1.0]),
        similarity=0.1,
        ratio=1.0,
        landmark_variance=0.0,
        pnp_soft_score=0.9,
    )

    assert [item.token_index for item in _order_pnp_matches_for_eval([low, high], mode="none", top_n=1)] == [0]
    assert [item.token_index for item in _order_pnp_matches_for_eval([low, high], mode="confidence", top_n=1)] == [1]


def test_order_pnp_matches_for_eval_can_apply_spatially_diverse_topk() -> None:
    clustered_best = QueryTo3DMatch(
        token_index=0,
        xy=np.asarray([0.0, 0.0]),
        track_id=0,
        xyz=np.asarray([0.0, 0.0, 1.0]),
        similarity=0.1,
        ratio=1.0,
        landmark_variance=0.0,
        pnp_soft_score=0.99,
        render_xy=np.asarray([0.0, 0.0]),
    )
    clustered_second = QueryTo3DMatch(
        token_index=1,
        xy=np.asarray([3.0, 0.0]),
        track_id=1,
        xyz=np.asarray([1.0, 0.0, 1.0]),
        similarity=0.1,
        ratio=1.0,
        landmark_variance=0.0,
        pnp_soft_score=0.98,
        render_xy=np.asarray([3.0, 0.0]),
    )
    spatially_separate = QueryTo3DMatch(
        token_index=2,
        xy=np.asarray([30.0, 0.0]),
        track_id=2,
        xyz=np.asarray([2.0, 0.0, 1.0]),
        similarity=0.1,
        ratio=1.0,
        landmark_variance=0.0,
        pnp_soft_score=0.50,
        render_xy=np.asarray([30.0, 0.0]),
    )

    selected = _order_pnp_matches_for_eval(
        [clustered_best, clustered_second, spatially_separate],
        mode="confidence",
        top_n=2,
        spatial_nms_radius_px=8.0,
    )

    assert [item.token_index for item in selected] == [0, 2]


def test_replace_match_confidence_scores_uses_pose_head_scores() -> None:
    low = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([0.0, 0.0]),
        render_xy=np.asarray([0.0, 0.0]),
        similarity=0.2,
        ratio=1.0,
        dual_softmax_confidence=0.9,
    )
    high = KeypointFeatureMatch(
        query_index=1,
        render_index=1,
        query_xy=np.asarray([10.0, 0.0]),
        render_xy=np.asarray([10.0, 0.0]),
        similarity=0.2,
        ratio=1.0,
        dual_softmax_confidence=0.1,
    )

    rescored = _replace_match_confidence_scores([low, high], np.asarray([0.05, 0.95], dtype=np.float32))

    assert [item.query_index for item in rescored] == [1, 0]
    assert [item.dual_softmax_confidence for item in rescored] == [pytest.approx(0.95), pytest.approx(0.05)]


def test_replace_match_confidence_scores_can_blend_with_existing_scores() -> None:
    existing = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([0.0, 0.0]),
        render_xy=np.asarray([0.0, 0.0]),
        similarity=0.2,
        ratio=1.0,
        dual_softmax_confidence=0.8,
    )

    rescored = _replace_match_confidence_scores([existing], np.asarray([0.2], dtype=np.float32), blend=0.25)

    assert rescored[0].dual_softmax_confidence == pytest.approx(0.65)


def test_aggregate_rows_by_bin_reports_medians_and_rates() -> None:
    rows = [
        {
            "synthetic_pose_bin": "micro",
            "pnp_success": True,
            "translation_error_m": 0.02,
            "rotation_error_deg": 0.1,
            "gt_precision_5px": 0.8,
            "gt_precision_10px": 0.9,
            "gt_precision_16px": 1.0,
            "pnp_inlier_gt_precision_5px": 0.7,
            "pnp_inlier_gt_precision_10px": 0.8,
            "pnp_inlier_gt_precision_16px": 1.0,
            "pnp_inlier_count": 20,
            "fine_offset_before_median_px": 4.0,
            "fine_offset_after_median_px": 1.0,
            "fine_offset_improvement_mean_px": 3.0,
            "fine_offset_improved_ratio": 1.0,
        },
        {
            "synthetic_pose_bin": "micro",
            "pnp_success": False,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "gt_precision_5px": 0.2,
            "gt_precision_10px": 0.4,
            "gt_precision_16px": 0.5,
            "pnp_inlier_gt_precision_5px": None,
            "pnp_inlier_gt_precision_10px": None,
            "pnp_inlier_gt_precision_16px": None,
            "pnp_inlier_count": 0,
            "fine_offset_before_median_px": 6.0,
            "fine_offset_after_median_px": 2.0,
            "fine_offset_improvement_mean_px": 4.0,
            "fine_offset_improved_ratio": 0.5,
        },
    ]

    summary = _aggregate_rows_by_bin(rows)
    assert summary["micro"]["pair_count"] == 2
    assert summary["micro"]["pnp_solve_rate"] == 0.5
    assert summary["micro"]["median_translation_error_m"] == 0.02
    assert summary["micro"]["mean_gt_precision_5px"] == 0.5
    assert summary["micro"]["mean_gt_precision_10px"] == 0.65
    assert summary["micro"]["mean_gt_precision_16px"] == 0.75
    assert summary["micro"]["mean_pnp_inlier_gt_precision_5px"] == 0.7
    assert summary["micro"]["mean_pnp_inlier_gt_precision_10px"] == 0.8
    assert summary["micro"]["mean_fine_offset_before_median_px"] == 5.0
    assert summary["micro"]["mean_fine_offset_after_median_px"] == 1.5
    assert summary["micro"]["mean_fine_offset_improvement_px"] == 3.5
    assert summary["micro"]["mean_fine_offset_improved_ratio"] == 0.75
    overall = _aggregate_overall_row_metrics(rows)
    assert overall["mean_gt_precision_16px"] == 0.75
    assert overall["mean_gt_precision_5px"] == 0.5
    assert overall["mean_pnp_inlier_gt_precision_10px"] == 0.8
    assert overall["mean_fine_offset_improvement_px"] == 3.5


def test_synthetic_eval_match_generation_applies_offset_logits() -> None:
    query_map = np.ones((1, 1, 1), dtype=np.float32)
    render_map = np.ones((1, 1, 1), dtype=np.float32)
    query_offsets = np.zeros((65, 1, 1), dtype=np.float32)
    query_offsets[0, 0, 0] = 10.0

    matches, refinement_source, fallback_reason = _matcha_eval_matches_for_pnp(
        query_map,
        render_map,
        query_image_width=64,
        query_image_height=64,
        render_image_width=64,
        render_image_height=64,
        query_offset_logits=query_offsets,
    )

    assert refinement_source == "dense_offset_fallback"
    assert fallback_reason is None
    assert len(matches) == 1
    assert np.allclose(matches[0].query_xy, [4.0, 4.0])
    assert np.allclose(matches[0].render_xy, [32.0, 32.0])


def test_synthetic_eval_pnp_reprojection_stats_row_reports_gt_and_inlier_quality() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=128, height=128, params=(16.0, 16.0, 0.0, 0.0))
    pose = np.eye(4, dtype=np.float64)
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([0.0, 0.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            similarity=1.0,
            ratio=0.0,
            landmark_variance=0.0,
            source="test",
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([64.0, 0.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            similarity=1.0,
            ratio=0.0,
            landmark_variance=0.0,
            source="test",
        ),
    ]

    stats = _pnp_reprojection_stats_row(matches, pose, camera, pnp_inlier_mask=np.asarray([True, False]))

    assert stats["match_count"] == 2
    assert stats["gt_precision_5px"] == 0.5
    assert stats["gt_precision_10px"] == 0.5
    assert stats["gt_precision_16px"] == 0.5
    assert stats["gt_precision_32px"] == 0.5
    assert stats["gt_reproj_median_px"] == 32.0
    assert stats["pnp_inlier_count"] == 1
    assert stats["pnp_inlier_gt_precision_16px"] == 1.0
    assert stats["pnp_inlier_gt_precision_32px"] == 1.0
    assert stats["pnp_inlier_gt_reproj_median_px"] == 0.0


def test_synthetic_eval_fine_offset_stats_row_reports_before_after_error() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=64, height=64, params=(16.0, 16.0, 0.0, 0.0))
    pose = np.eye(4, dtype=np.float64)
    matches = [
        QueryTo3DMatch(
            token_index=0,
            xy=np.asarray([20.0, 16.0], dtype=np.float64),
            track_id=1,
            xyz=np.asarray([1.25, 1.0, 1.0], dtype=np.float64),
            similarity=1.0,
            ratio=0.0,
            landmark_variance=0.0,
            source="test",
        ),
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([48.0, 16.0], dtype=np.float64),
            track_id=2,
            xyz=np.asarray([3.0, 1.0, 1.0], dtype=np.float64),
            similarity=1.0,
            ratio=0.0,
            landmark_variance=0.0,
            source="test",
        ),
    ]

    stats = _fine_offset_stats_row(matches, pose, camera, query_grid_hw=(2, 2))

    assert stats["fine_offset_eval_count"] == 2
    assert stats["fine_offset_before_median_px"] == 2.0
    assert stats["fine_offset_after_median_px"] == 0.0
    assert stats["fine_offset_improvement_mean_px"] == 2.0
    assert stats["fine_offset_improved_ratio"] == 0.5
    assert stats["fine_offset_gt16_before"] == 1.0
    assert stats["fine_offset_gt16_after"] == 1.0


def test_target_depth_matches_to_pnp_backprojects_render_points() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(4.0, 4.0, 2.0, 2.0))
    pose = np.eye(4, dtype=np.float64)
    depth = np.ones((4, 4), dtype=np.float32)
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([2.0, 2.0], dtype=np.float64),
        render_xy=np.asarray([2.0, 2.0], dtype=np.float64),
        similarity=1.0,
        ratio=0.0,
    )

    pnp_matches = _target_depth_matches_to_pnp([match], render_depth=depth, render_pose_w2c=pose, render_camera=camera)

    assert len(pnp_matches) == 1
    assert np.allclose(pnp_matches[0].xy, [2.0, 2.0])
    assert np.allclose(pnp_matches[0].xyz, [0.0, 0.0, 1.0])


def test_target_depth_matches_to_pnp_preserves_gt_pose_with_distorted_camera() -> None:
    cv2 = pytest.importorskip("cv2")
    camera = ColmapCamera(camera_id=1, model_id=2, width=640, height=480, params=(520.0, 320.0, 240.0, 0.18))
    render_pose = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(4.0)
    source_pose = pose_w2c_from_center_rotation(
        np.asarray([0.2, -0.05, 0.1], dtype=np.float64),
        np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float64,
        ),
    )
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    render_normalized_xy = np.asarray(
        [
            [-0.35, -0.22],
            [-0.18, 0.28],
            [0.12, -0.25],
            [0.32, 0.18],
            [-0.40, 0.05],
            [0.05, 0.36],
            [0.38, -0.05],
            [-0.08, -0.34],
            [0.24, 0.30],
            [-0.30, 0.25],
        ],
        dtype=np.float64,
    )
    world_xyz = np.concatenate([render_normalized_xy * 4.0, np.full((render_normalized_xy.shape[0], 1), 4.0)], axis=1)
    render_xy, _ = cv2.projectPoints(
        world_xyz,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    source_rvec, _ = cv2.Rodrigues(source_pose[:3, :3])
    query_xy, _ = cv2.projectPoints(world_xyz, source_rvec, source_pose[:3, 3], camera_matrix, distortion)
    matches = [
        KeypointFeatureMatch(
            query_index=index,
            render_index=index,
            query_xy=query.reshape(2),
            render_xy=render.reshape(2),
            similarity=1.0,
            ratio=0.0,
        )
        for index, (query, render) in enumerate(zip(query_xy.reshape(-1, 2), render_xy.reshape(-1, 2)))
    ]
    depth = np.full((int(camera.height), int(camera.width)), 4.0, dtype=np.float32)

    pnp_matches = _target_depth_matches_to_pnp(
        matches,
        render_depth=depth,
        render_pose_w2c=render_pose,
        render_camera=camera,
    )
    pnp = estimate_pose_pnp_ransac(pnp_matches, camera, reprojection_error_px=2.0, iterations=1000)
    error = pnp_pose_error(pnp.pose_w2c, source_pose)

    assert pnp.success
    assert pnp.inlier_count == len(matches)
    assert error.translation_m < 1e-4
    assert error.rotation_deg < 1e-3


def test_target_depth_matches_to_pnp_skips_out_of_bounds_render_points() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(4.0, 4.0, 2.0, 2.0))
    pose = np.eye(4, dtype=np.float64)
    depth = np.ones((4, 4), dtype=np.float32)
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([2.0, 2.0], dtype=np.float64),
        render_xy=np.asarray([-1.0, 2.0], dtype=np.float64),
        similarity=1.0,
        ratio=0.0,
    )

    pnp_matches = _target_depth_matches_to_pnp([match], render_depth=depth, render_pose_w2c=pose, render_camera=camera)

    assert pnp_matches == []
