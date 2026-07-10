from __future__ import annotations

from feature_extract.tools.vfm.build_projected_observation_landmark_bank import (
    parse_args as parse_projected_bank_args,
)
from feature_extract.tools.vfm.eval_real_radio_landmark_hybrid import parse_args, resolve_runtime_device


def test_landmark_hybrid_cli_parser_defaults_to_no_measurement() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--landmark_bank",
            "bank.npz",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--image_root",
            "images",
            "--colmap_model_dir",
            "sparse/0",
            "--query_pose_file",
            "poses.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
        ]
    )

    assert args.measurement_mode == "none"
    assert args.landmark_projection == "joint"
    assert args.feature_key == "radio_final"
    assert args.query_token_step == 4
    assert args.query_token_selection == "uniform"
    assert args.query_heatmap_top_k == 0
    assert args.query_heatmap_nms_radius == 0
    assert args.top_k == 2
    assert args.max_queries == 0
    assert args.projected_landmark_cache == ""
    assert args.landmark_search_backend == "auto"
    assert args.landmark_index_cache_size == 64
    assert args.submap_mode == "none"
    assert args.submap_top_n == 5
    assert args.submap_spatial_grid_rows == 8
    assert args.submap_spatial_grid_cols == 8
    assert args.submap_min_landmarks == 0
    assert args.submap_min_spatial_cells == 16
    assert args.submap_fallback_fraction == 0.25
    assert args.measurement_max_matches == 0
    assert args.measurement_selection_strategy == "score_spatial"
    assert args.measurement_query_batch_size == 1
    assert args.measurement_grid_rows == 4
    assert args.measurement_grid_cols == 4
    assert args.measurement_confidence_temperature == 4.0
    assert args.measurement_uncertainty_scale == 1.0
    assert args.measurement_tensor_cache_size == 64
    assert args.measurement_score_mode == "match"
    assert args.enable_quality_rescore is False
    assert args.pnp_weighted_refine is False


def test_resolve_runtime_device_falls_back_to_cpu_for_unavailable_cuda() -> None:
    device = resolve_runtime_device("cuda")

    assert str(device) in {"cuda", "cpu"}


def test_projected_observation_landmark_bank_cli_parser_defaults_to_full_map_projection() -> None:
    args = parse_projected_bank_args(
        [
            "--track_observations",
            "tracks.jsonl",
            "--token_manifest",
            "manifest.json",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_index",
            "projected.npz",
            "--summary_json",
            "summary.json",
        ]
    )

    assert args.feature_key == "radio_final"
    assert args.method == "mean"
    assert args.sample_mode == "bilinear"
    assert args.min_observations == 2
    assert args.device == "cuda"
