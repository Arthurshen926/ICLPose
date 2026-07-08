from __future__ import annotations

import pytest

from feature_extract.tools.vfm.run_actual_coarse_measurement_protocol import parse_args


def test_actual_coarse_measurement_protocol_parses_training_balance_and_prior_scale_args() -> None:
    args = parse_args(
        [
            "--train_match_table_csv",
            "train.csv",
            "--train_render_cache_manifest_csv",
            "train_manifest.csv",
            "--query_pose_file",
            "poses.txt",
            "--camera_model_dir",
            "sparse/0",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--query_image_width",
            "1920",
            "--query_image_height",
            "1080",
            "--render_image_width",
            "1280",
            "--render_image_height",
            "720",
            "--coarse_likelihood_loss_weight",
            "1.5",
            "--epe_weight",
            "0.25",
            "--residual_balanced_sampling",
            "--residual_sampling_bins_px",
            "0",
            "2",
            "5",
            "20",
            "28",
            "--condition_on_prior_scale",
            "--prior_scale_key",
            "center_residual_px",
            "--prior_scale_expert_centers_px",
            "2",
            "8",
            "20",
            "28",
            "--prior_scale_expert_projection",
            "--prior_scale_expert_gate",
            "hard",
            "--gate_center_radius_px",
            "2",
            "--gate_full_radius_px",
            "28",
            "--gate_utility_temperature_px",
            "1.25",
        ]
    )

    assert args.coarse_likelihood_loss_weight == pytest.approx(1.5)
    assert args.epe_weight == pytest.approx(0.25)
    assert args.residual_balanced_sampling is True
    assert args.residual_sampling_bins_px == [0.0, 2.0, 5.0, 20.0, 28.0]
    assert args.condition_on_prior_scale is True
    assert args.prior_scale_key == "center_residual_px"
    assert args.prior_scale_expert_centers_px == [2.0, 8.0, 20.0, 28.0]
    assert args.prior_scale_expert_projection is True
    assert args.prior_scale_expert_gate == "hard"
    assert args.gate_center_radius_px == pytest.approx(2.0)
    assert args.gate_full_radius_px == pytest.approx(28.0)
    assert args.gate_utility_temperature_px == pytest.approx(1.25)
