from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.evaluate_impl import (
    resolve_eval_num_workers,
    resolve_eval_pose_update_scale,
)
from pose_refine.train_impl import (
    build_eval_sample_record,
    build_eval_sweep_record,
    resolve_best_metric_value,
)
from feature_field.utils.project_config import load_mainline_config
from pose_refine.tools.diag_corr_split import parse_split_names
from pose_refine.tools.eval_pose_update_scale import parse_scales


def test_eval_num_workers_defaults_to_training_config():
    config = {"training": {"num_workers": 0}}

    assert resolve_eval_num_workers(None, config) == 0


def test_eval_num_workers_cli_overrides_config():
    config = {"training": {"num_workers": 0}}

    assert resolve_eval_num_workers(2, config) == 2


def test_eval_pose_update_scale_cli_overrides_model_config():
    config = {"model": {"pose_update_scale": 0.5}}

    assert resolve_eval_pose_update_scale(0.25, config) == 0.25


def test_eval_pose_update_scale_parser_preserves_order():
    assert parse_scales(["0", "0.25", "1.0"]) == [0.0, 0.25, 1.0]


def test_diag_corr_split_parser_expands_aliases():
    assert parse_split_names(["both"]) == ["train", "val"]
    assert parse_split_names(["test", "train"]) == ["val", "train"]


def test_mainline_config_supports_relative_base_config(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "exp_name: base_exp",
                "model:",
                "  full_wls: true",
                "  rot_mode: wls",
                "training:",
                "  epochs: 8",
                "  loss:",
                "    flow_weight: 40.0",
                "    rot_weight: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "base_config: base.yaml",
                "exp_name: child_exp",
                "model:",
                "  rot_mode: hybrid",
                "training:",
                "  loss:",
                "    rot_weight: 120.0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_mainline_config(str(child))

    assert cfg["exp_name"] == "child_exp"
    assert cfg["model"]["full_wls"] is True
    assert cfg["model"]["rot_mode"] == "hybrid"
    assert cfg["training"]["epochs"] == 8
    assert cfg["training"]["loss"]["flow_weight"] == 40.0
    assert cfg["training"]["loss"]["rot_weight"] == 120.0


def test_pose_refine_best_metric_defaults_to_lower_translation():
    value, higher_is_better, label = resolve_best_metric_value(
        {"val_trans_median": 73.0, "val_rot_median": 0.64},
        "trans_median",
    )

    assert value == 73.0
    assert higher_is_better is False
    assert label == "val_trans_median"


def test_pose_refine_best_metric_can_maximize_joint_accuracy():
    value, higher_is_better, label = resolve_best_metric_value(
        {"val_joint_1deg_50mm": 17.6, "val_trans_median": 73.0},
        "joint_1deg_50mm",
    )

    assert value == 17.6
    assert higher_is_better is True
    assert label == "val_joint_1deg_50mm"


def test_eval_sweep_record_preserves_core_metrics():
    record = build_eval_sweep_record(
        outer_iters=3,
        gru_iters=2,
        metrics={
            "val_rot_median": 0.36,
            "val_trans_median": 73.6,
            "val_pct_1deg": 86.8,
            "val_joint_1deg_50mm": 20.9,
        },
        seed_count=1,
    )

    assert record["outer_iters"] == 3
    assert record["gru_iters"] == 2
    assert record["seed_count"] == 1
    assert record["rot_median_deg"] == 0.36
    assert record["trans_median_mm"] == 73.6
    assert record["pct_1deg"] == 86.8
    assert record["joint_1deg_50mm"] == 20.9


def test_eval_sample_record_preserves_identity_and_stage_errors():
    record = build_eval_sample_record(
        image_id=42,
        image_name="seq1/frame00042.png",
        outer_iters=5,
        gru_iters=2,
        seed=0,
        init_rot_deg=0.64,
        init_trans_mm=74.0,
        one_rot_deg=0.47,
        one_trans_mm=73.2,
        final_rot_deg=0.34,
        final_trans_mm=73.2,
        flow_epe_px=2.95,
    )

    assert record["image_id"] == 42
    assert record["image_name"] == "seq1/frame00042.png"
    assert record["outer_iters"] == 5
    assert record["gru_iters"] == 2
    assert record["seed"] == 0
    assert record["final_rot_deg"] == 0.34
    assert record["final_trans_mm"] == 73.2
    assert record["flow_epe_px"] == 2.95


if __name__ == "__main__":
    test_eval_num_workers_defaults_to_training_config()
    test_eval_num_workers_cli_overrides_config()
    test_eval_pose_update_scale_cli_overrides_model_config()
    test_eval_pose_update_scale_parser_preserves_order()
    test_diag_corr_split_parser_expands_aliases()
    test_eval_sweep_record_preserves_core_metrics()
    test_eval_sample_record_preserves_identity_and_stage_errors()
    print("pose refine eval config tests passed")
