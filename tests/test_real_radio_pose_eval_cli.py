from __future__ import annotations

from feature_extract.tools.vfm.eval_real_radio_pose_localization import parse_args, resolve_runtime_device


def test_eval_real_radio_pose_localization_cli_args() -> None:
    args = parse_args(
        [
            "--pairs_csv",
            "pairs.csv",
            "--image_root",
            "images",
            "--feature_root",
            "features",
            "--colmap_model_dir",
            "sparse/0",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--query_pose_file",
            "dataset_test.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
            "--feature_key",
            "radio_final",
            "--feature_path_template",
            "{image_token}.npz",
            "--k_per_query",
            "2",
            "--max_support_distance_px",
            "6",
            "--pnp_reprojection_error_px",
            "8",
            "--pnp_min_inliers",
            "6",
        ]
    )

    assert args.colmap_model_dir == "sparse/0"
    assert args.track_observations_jsonl == "tracks.jsonl"
    assert args.query_pose_file == "dataset_test.txt"
    assert args.feature_key == "radio_final"
    assert args.k_per_query == 2
    assert args.max_support_distance_px == 6.0
    assert args.pnp_min_inliers == 6


def test_eval_real_radio_pose_runtime_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert str(resolve_runtime_device("cuda")) == "cpu"
    assert str(resolve_runtime_device("cpu")) == "cpu"
