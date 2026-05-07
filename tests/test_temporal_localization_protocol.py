from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from feature_extract.temporal_protocol import (
    TemporalInitSelector,
    build_temporal_pose_grid,
    camera_center_from_w2c,
    summarize_temporal_errors,
    temporal_sort_records,
)
from feature_extract.tools.eval_poseinit_render_score import (
    _parse_float_list,
    _records_for_protocol,
    _summarize_error_pairs,
)


def _pose_at(x: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = -float(x)
    return pose


def test_temporal_sort_records_orders_by_sequence_and_numeric_frame():
    records = [
        {"sample_name": "seq8/frame00010.png"},
        {"sample_name": "seq4/frame00002.png"},
        {"sample_name": "seq8/frame00002.png"},
        {"sample_name": "seq4/frame00001.png"},
    ]

    sorted_records = temporal_sort_records(records)

    assert [record["sample_name"] for record in sorted_records] == [
        "seq4/frame00001.png",
        "seq4/frame00002.png",
        "seq8/frame00002.png",
        "seq8/frame00010.png",
    ]


def test_prev_gt_oracle_uses_previous_gt_and_resets_at_sequence_boundary():
    selector = TemporalInitSelector(protocol="temporal_prev_gt_oracle", init_mode="prev_pose")
    real_init = _pose_at(100.0)

    init0, meta0 = selector.select("seq4/frame00001.png", real_init)
    selector.update("seq4/frame00001.png", gt_pose=_pose_at(1.0), final_pose=_pose_at(10.0))
    init1, meta1 = selector.select("seq4/frame00002.png", real_init)
    selector.update("seq4/frame00002.png", gt_pose=_pose_at(2.0), final_pose=_pose_at(20.0))
    init_new_seq, meta_new_seq = selector.select("seq8/frame00001.png", real_init)

    assert torch.allclose(init0, real_init)
    assert meta0["is_sequence_start"] is True
    assert torch.allclose(camera_center_from_w2c(init1), torch.tensor([1.0, 0.0, 0.0]))
    assert meta1["source"] == "prev_gt"
    assert torch.allclose(init_new_seq, real_init)
    assert meta_new_seq["is_sequence_start"] is True


def test_prev_pred_tracking_uses_previous_final_prediction():
    selector = TemporalInitSelector(protocol="temporal_prev_pred_tracking", init_mode="prev_pose")
    real_init = _pose_at(100.0)

    init0, _meta0 = selector.select("seq8/frame00001.png", real_init)
    selector.update("seq8/frame00001.png", gt_pose=_pose_at(1.0), final_pose=_pose_at(11.0))
    init1, meta1 = selector.select("seq8/frame00002.png", real_init)

    assert torch.allclose(init0, real_init)
    assert torch.allclose(camera_center_from_w2c(init1), torch.tensor([11.0, 0.0, 0.0]))
    assert meta1["source"] == "prev_pred"


def test_constant_velocity_uses_two_previous_predictions_within_sequence():
    selector = TemporalInitSelector(protocol="temporal_prev_pred_tracking", init_mode="constant_velocity")
    real_init = _pose_at(100.0)

    selector.select("seq8/frame00001.png", real_init)
    selector.update("seq8/frame00001.png", gt_pose=_pose_at(1.0), final_pose=_pose_at(11.0))
    selector.select("seq8/frame00002.png", real_init)
    selector.update("seq8/frame00002.png", gt_pose=_pose_at(2.0), final_pose=_pose_at(13.0))
    init2, meta2 = selector.select("seq8/frame00003.png", real_init)

    assert torch.allclose(camera_center_from_w2c(init2), torch.tensor([15.0, 0.0, 0.0]))
    assert meta2["source"] == "constant_velocity_pred"


def test_temporal_pose_grid_samples_world_xz_offsets_and_yaw():
    grid = build_temporal_pose_grid(
        _pose_at(0.0),
        trans_offsets_m=[-1.0, 0.0, 1.0],
        yaw_offsets_deg=[-5.0, 0.0, 5.0],
    )
    centers = camera_center_from_w2c(grid)

    assert grid.shape == (27, 4, 4)
    assert any(torch.allclose(center, torch.tensor([1.0, 0.0, -1.0])) for center in centers)
    assert any(torch.allclose(center, torch.tensor([0.0, 0.0, 0.0])) for center in centers)
    assert not torch.allclose(grid[0, :3, :3], grid[1, :3, :3])


def test_temporal_summary_reports_fixed_recalls_and_non_start_subset():
    samples = [
        {"init_trans_mm": 1000.0, "init_rot_deg": 10.0, "final_trans_mm": 1000.0, "final_rot_deg": 10.0, "is_sequence_start": True},
        {"init_trans_mm": 80.0, "init_rot_deg": 0.8, "final_trans_mm": 40.0, "final_rot_deg": 0.5, "is_sequence_start": False},
        {"init_trans_mm": 120.0, "init_rot_deg": 1.5, "final_trans_mm": 90.0, "final_rot_deg": 1.5, "is_sequence_start": False},
    ]

    summary = summarize_temporal_errors(samples)

    assert summary["final"]["recall_1deg_50mm"] == pytest.approx(100.0 / 3.0)
    assert summary["final"]["recall_2deg_100mm"] == pytest.approx(200.0 / 3.0)
    assert summary["final_non_start"]["recall_1deg_50mm"] == 50.0
    assert summary["init_to_final_gain"]["mean_trans_mm"] > 0.0


def test_eval_records_sort_only_for_temporal_protocols():
    records = [
        {"sample_name": "seq8/frame00010.png"},
        {"sample_name": "seq4/frame00001.png"},
        {"sample_name": "seq8/frame00002.png"},
    ]

    single = _records_for_protocol(records, "single_frame_real_init")
    temporal = _records_for_protocol(records, "temporal_prev_pred_tracking")

    assert [record["sample_name"] for record in single] == [
        "seq8/frame00010.png",
        "seq4/frame00001.png",
        "seq8/frame00002.png",
    ]
    assert [record["sample_name"] for record in temporal] == [
        "seq4/frame00001.png",
        "seq8/frame00002.png",
        "seq8/frame00010.png",
    ]


def test_eval_error_summary_uses_mainline_recall_thresholds():
    summary = _summarize_error_pairs([(40.0, 0.5), (90.0, 1.5), (200.0, 4.0)])

    assert summary["recall_1deg_50mm"] == pytest.approx(100.0 / 3.0)
    assert summary["recall_1deg_100mm"] == pytest.approx(100.0 / 3.0)
    assert summary["recall_2deg_100mm"] == pytest.approx(200.0 / 3.0)
    assert summary["recall_5deg_250mm"] == pytest.approx(100.0)


def test_eval_parse_float_list_supports_disabled_and_comma_values():
    assert _parse_float_list(None) == []
    assert _parse_float_list("") == []
    assert _parse_float_list("-1,0,1") == [-1.0, 0.0, 1.0]
