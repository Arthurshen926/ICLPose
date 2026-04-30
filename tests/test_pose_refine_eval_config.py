from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.evaluate_impl import (
    resolve_eval_num_workers,
    resolve_eval_pose_update_scale,
)
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


if __name__ == "__main__":
    test_eval_num_workers_defaults_to_training_config()
    test_eval_num_workers_cli_overrides_config()
    test_eval_pose_update_scale_cli_overrides_model_config()
    test_eval_pose_update_scale_parser_preserves_order()
    test_diag_corr_split_parser_expands_aliases()
    print("pose refine eval config tests passed")
