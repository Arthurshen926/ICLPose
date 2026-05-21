import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest
import numpy as np

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries
from feature_extract.localizability.failure_replay import (
    batch_failure_pairs,
    cached_failure_pair_margin_loss,
    load_mined_score_hard_pairs,
    mine_score_hard_candidate_pairs,
    summarize_candidate_failure_modes,
    load_failure_replay_rows,
)
from feature_extract.localizability.handoff_cache import export_selected_init_cache
from feature_extract.tools.report_localizability import format_markdown_table, summarize_jsonl
from feature_extract.tools.eval_feature_track_mapability import project_points_to_image, sample_feature_at_pixels
from feature_extract.tools.augment_pose_candidate_cache import (
    append_near_identity_candidate_arrays,
    build_near_identity_pose_candidates,
)
from feature_extract.tools.export_query_student_descriptors import (
    apply_student_feature_hw_defaults,
    build_rgb_records,
    prepare_checkpoint_for_rgb_export,
    resolve_descriptor_key,
    resolve_feature_dir,
)


def _pose_at(value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(value)
    return pose


def test_resolve_feature_dir_falls_back_to_nested_result_root(tmp_path):
    old_root = tmp_path / "result" / "feature_extract" / "features" / "scene"
    nested_root = tmp_path / "result" / "result" / "feature_extract" / "features" / "scene"
    (nested_root / "fine_geo").mkdir(parents=True)
    (nested_root / "coarse_sem").mkdir()

    resolved = resolve_feature_dir(old_root)

    assert resolved == nested_root


def test_apply_student_feature_hw_defaults_prefers_student_export_resolution():
    cfg = {
        "dataset": {
            "feature_hw": [68, 120],
            "coarse_feature_hw": [17, 30],
            "student_feature_hw": [272, 480],
            "student_coarse_feature_hw": [34, 60],
        }
    }

    apply_student_feature_hw_defaults(cfg)

    assert cfg["dataset"]["feature_hw"] == [272, 480]
    assert cfg["dataset"]["coarse_feature_hw"] == [34, 60]


def test_build_rgb_records_uses_cambridge_split_order_without_teacher_cache(tmp_path):
    source_dir = tmp_path / "processed"
    (source_dir / "seq1").mkdir(parents=True)
    (source_dir / "seq1" / "frame0001.png").write_bytes(b"")
    (source_dir / "seq1" / "frame0002.png").write_bytes(b"")
    split_path = tmp_path / "dataset_test.txt"
    split_path.write_text("seq1/frame0002.png\n", encoding="utf-8")
    cfg = {
        "source_dir": str(source_dir),
        "image_patterns": ["seq*/*.png"],
        "train_split": str(tmp_path / "missing_train.txt"),
        "val_split": str(split_path),
    }

    records = build_rgb_records(cfg, split="val", limit=None)

    assert [record["sample_name"] for record in records] == ["seq1/frame0002.png"]


def test_prepare_checkpoint_for_rgb_export_drops_legacy_fine_loc_keys():
    checkpoint = {
        "model_state_dict": {
            "fine_head.weight": torch.tensor([1.0]),
            "fine_loc_head.0.weight": torch.tensor([2.0]),
            "fine_loc_scale": torch.tensor([3.0]),
            "fine_loc_highres_fuse.0.weight": torch.tensor([4.0]),
        }
    }

    cleaned, dropped = prepare_checkpoint_for_rgb_export(checkpoint)

    assert sorted(dropped) == [
        "fine_loc_head.0.weight",
        "fine_loc_highres_fuse.0.weight",
        "fine_loc_scale",
    ]
    assert list(cleaned["model_state_dict"]) == ["fine_head.weight"]
    assert "fine_loc_head.0.weight" in checkpoint["model_state_dict"]


def test_resolve_descriptor_key_keeps_model_fine_explicit():
    assert resolve_descriptor_key("fine", fine_key="fine_loc") == "fine"
    assert resolve_descriptor_key("export_fine", fine_key="fine_loc") == "fine_loc"


def test_append_near_identity_candidate_arrays_extends_candidate_axis():
    pose_inits = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    pose_inits[1, 0, 3] = 1.0
    existing = pose_inits[:, None].copy()
    cache = {
        "pose_inits": pose_inits,
        "pose_init_candidates": existing,
        "candidate_valid_mask": np.ones((2, 1), dtype=bool),
        "retrieval_scores": np.array([0.3, 0.4], dtype=np.float32),
        "retrieval_scores_candidates": np.array([[0.1], [0.2]], dtype=np.float32),
    }

    augmented, metadata = append_near_identity_candidate_arrays(
        cache,
        trans_cm=[5.0],
        rot_deg=[1.0],
        num_jitter=2,
        seed=7,
        include_identity=True,
    )

    assert augmented["pose_init_candidates"].shape == (2, 4, 4, 4)
    assert augmented["candidate_valid_mask"].shape == (2, 4)
    assert augmented["candidate_valid_mask"].all()
    assert np.allclose(augmented["pose_init_candidates"][:, 1], pose_inits)
    assert augmented["retrieval_scores_candidates"].shape == (2, 4)
    assert np.allclose(augmented["retrieval_scores_candidates"][:, 1], cache["retrieval_scores"])
    assert metadata["num_existing_candidates"] == 1
    assert metadata["num_added_candidates"] == 3


def test_build_near_identity_pose_candidates_is_deterministic_for_seed():
    pose_inits = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))

    first = build_near_identity_pose_candidates(
        pose_inits,
        trans_cm=[2.0],
        rot_deg=[1.0],
        num_jitter=3,
        seed=11,
        include_identity=True,
    )
    second = build_near_identity_pose_candidates(
        pose_inits,
        trans_cm=[2.0],
        rot_deg=[1.0],
        num_jitter=3,
        seed=11,
        include_identity=True,
    )

    assert first.shape == (1, 4, 4, 4)
    assert np.allclose(first, second)
    assert np.allclose(first[:, 0], pose_inits)


def test_mine_score_hard_candidate_pairs_selects_score_high_wrong_candidate():
    rows = [
        {
            "sample_name": "q1.png",
            "candidate_idx": 0,
            "score": 0.1,
            "pose_cost_m": 0.10,
            "valid": True,
            "in_basin": True,
        },
        {
            "sample_name": "q1.png",
            "candidate_idx": 1,
            "score": 0.9,
            "pose_cost_m": 0.40,
            "valid": True,
            "in_basin": False,
            "delta_trans_m": 0.02,
        },
        {
            "sample_name": "q1.png",
            "candidate_idx": 2,
            "score": 0.8,
            "pose_cost_m": 0.60,
            "valid": True,
            "in_basin": False,
            "delta_trans_m": 0.30,
        },
        {
            "sample_name": "q2.png",
            "candidate_idx": 0,
            "score": 0.5,
            "pose_cost_m": 0.10,
            "valid": True,
            "in_basin": True,
        },
        {
            "sample_name": "q2.png",
            "candidate_idx": 1,
            "score": 0.6,
            "pose_cost_m": 0.15,
            "valid": True,
            "in_basin": True,
        },
    ]

    pairs, summary = mine_score_hard_candidate_pairs(rows, cost_gap_m=0.12, near_identity_trans_m=0.05)

    assert len(pairs) == 1
    assert pairs[0]["sample_name"] == "q1.png"
    assert pairs[0]["positive_idx"] == 0
    assert pairs[0]["negative_idx"] == 1
    assert pairs[0]["negative_is_near_identity"] is True
    assert pairs[0]["negative_in_basin"] is False
    assert summary["num_samples"] == 2
    assert summary["num_pairs"] == 1
    assert summary["near_identity_negative_pairs"] == 1
    assert summary["score_margin_positive_pairs"] == 1


def test_load_mined_score_hard_pairs_uses_positive_and_negative_indices(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_name": "seq/frame.png",
                "positive_idx": 2,
                "negative_idx": 5,
                "cost_gap_m": 0.20,
                "positive_cost_m": 0.10,
                "negative_cost_m": 0.30,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    replay = load_mined_score_hard_pairs(path)

    assert list(replay) == ["seq/frame.png"]
    assert replay["seq/frame.png"].positive_idx == 2
    assert replay["seq/frame.png"].negative_idx == 5
    assert replay["seq/frame.png"].oracle_gap_m == 0.20


def test_report_localizability_summarizes_best_eval_row(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    rows = [
        {"split": "train", "step": 1, "pred_cost_m": 0.40},
        {
            "split": "eval",
            "step": 10,
            "pred_cost_m": 0.31,
            "oracle_cost_m": 0.12,
            "top1_acc": 0.60,
            "spearman": 0.20,
            "basin_recall@5": 0.75,
        },
        {
            "split": "eval",
            "step": 20,
            "pred_cost_m": 0.25,
            "oracle_cost_m": 0.11,
            "top1_acc": 0.70,
            "spearman": 0.40,
            "basin_recall@5": 0.80,
        },
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = summarize_jsonl(log_path)
    markdown = format_markdown_table([summary])

    assert summary["step"] == 20
    assert summary["oracle_gap_m"] == 0.14
    assert "| run | step | pred | oracle | gap | top1 | spearman | basin@5 |" in markdown


def test_export_selected_init_cache_keeps_solver_selected_candidate_metadata(tmp_path):
    cache_path = tmp_path / "base_candidates.npz"
    entries = [
        {
            "query_img_id": 1,
            "query_image_name": "seq/frame00001.png",
            "query_image_stem": "seq_frame00001",
            "pose_init": _pose_at(0.0),
            "init_source": "base",
            "retrieval_frame_id": 10,
            "retrieval_image_name": "ref/init.png",
            "retrieval_score": 0.0,
            "pose_init_candidates": np.stack([_pose_at(10.0), _pose_at(20.0), _pose_at(30.0)]),
            "candidate_valid_mask": np.array([True, True, True], dtype=bool),
            "retrieval_frame_ids_candidates": np.array([101, 102, 103], dtype=np.int64),
            "retrieval_image_names_candidates": np.array(["ref/a.png", "ref/b.png", "ref/c.png"]),
            "retrieval_scores_candidates": np.array([0.4, 0.3, 0.2], dtype=np.float32),
            "retrieval_pnp_success_candidates": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "retrieval_pnp_num_inliers_candidates": np.array([10.0, 50.0, 20.0], dtype=np.float32),
            "retrieval_pnp_num_matches_candidates": np.array([30.0, 80.0, 40.0], dtype=np.float32),
            "retrieval_pnp_reproj_rmse_candidates": np.array([4.0, 2.0, 3.0], dtype=np.float32),
            "retrieval_pnp_reproj_median_candidates": np.array([3.0, 1.0, 2.0], dtype=np.float32),
            "retrieval_pnp_inlier_ratio_candidates": np.array([0.3, 0.6, 0.5], dtype=np.float32),
            "retrieval_pnp_inlier_conf_mean_candidates": np.array([0.2, 0.8, 0.4], dtype=np.float32),
        }
    ]
    save_retrieval_init_entries(entries, {"method_used": "base"}, str(cache_path))
    table_path = tmp_path / "candidate_table.jsonl"
    rows = [
        {
            "sample_name": "seq/frame00001.png",
            "candidate_idx": 0,
            "score": 0.9,
            "pose_cost_m": 0.40,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 10.0,
        },
        {
            "sample_name": "seq/frame00001.png",
            "candidate_idx": 1,
            "score": 0.8,
            "pose_cost_m": 0.10,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 50.0,
        },
        {
            "sample_name": "seq/frame00001.png",
            "candidate_idx": 2,
            "score": 0.1,
            "pose_cost_m": 0.30,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 20.0,
        },
    ]
    with table_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    _exported, stats = export_selected_init_cache(
        source_cache_path=cache_path,
        candidate_table_path=table_path,
        save_path=tmp_path / "selected.npz",
        topk=2,
        selection_mode="pnp_inliers",
        source_name="pofd_fs_top2_pnp_inliers",
    )
    loaded, loaded_stats = load_retrieval_init_entries(str(tmp_path / "selected.npz"))
    entry = loaded[0]

    assert stats == loaded_stats
    assert stats["selection_mode"] == "pnp_inliers"
    assert stats["topk"] == 2
    assert stats["num_entries"] == 1
    assert stats["selected_candidate_indices"] == [1]
    assert np.allclose(entry["pose_init"], _pose_at(20.0))
    assert entry["init_source"] == "pofd_fs_top2_pnp_inliers"
    assert entry["retrieval_frame_id"] == 102
    assert entry["retrieval_image_name"] == "ref/b.png"
    assert entry["pose_init_candidates"].shape == (1, 4, 4)
    assert np.allclose(entry["pose_init_candidates"][0], _pose_at(20.0))
    assert entry["candidate_valid_mask"].tolist() == [True]
    assert entry["retrieval_pnp_num_inliers_candidates"].tolist() == [50.0]
    assert entry["retrieval_pnp_reproj_median_candidates"].tolist() == [1.0]


def test_failure_replay_loads_wrong_top1_rows_by_sample_name(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {
            "sample_name": "seq/frame0001.png",
            "top1": True,
            "selected_idx": 0,
            "oracle_idx": 0,
            "oracle_gap_m": 0.0,
        },
        {
            "sample_name": "seq/frame0002.png",
            "top1": False,
            "selected_idx": 5,
            "oracle_idx": 1,
            "oracle_gap_m": 0.20,
        },
        {
            "sample_name": "seq/frame0003.png",
            "top1": False,
            "selected_idx": 6,
            "oracle_idx": 2,
            "oracle_gap_m": 0.01,
        },
    ]
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    replay = load_failure_replay_rows(rows_path, min_gap_m=0.05)

    assert list(replay) == ["seq/frame0002.png"]
    assert replay["seq/frame0002.png"].positive_idx == 1
    assert replay["seq/frame0002.png"].negative_idx == 5


def test_batch_failure_pairs_maps_sample_names_to_indices():
    replay = load_failure_replay_rows.from_rows(
        [
            {"sample_name": "seq/frame0002.png", "top1": False, "selected_idx": 5, "oracle_idx": 1},
            {"sample_name": "frame0004.png", "top1": False, "selected_idx": 3, "oracle_idx": 0},
        ]
    )

    active, positive, negative = batch_failure_pairs(
        ["seq/frame0001.png", "seq/frame0002.png", "images/frame0004.png"],
        replay,
        device=torch.device("cpu"),
    )

    assert active.tolist() == [False, True, True]
    assert positive.tolist() == [0, 1, 0]
    assert negative.tolist() == [0, 5, 3]


def test_cached_failure_pair_margin_loss_only_uses_active_rows():
    scores = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 4.0, 1.0],
            [3.0, 0.0, 2.0],
        ]
    )
    active = torch.tensor([False, True, True])
    positive = torch.tensor([0, 0, 2])
    negative = torch.tensor([0, 1, 0])

    loss, stats = cached_failure_pair_margin_loss(
        scores,
        positive_idx=positive,
        negative_idx=negative,
        active_mask=active,
        margin=0.5,
    )

    expected = torch.stack(
        [
            torch.nn.functional.softplus(scores[1, 1] - scores[1, 0] + 0.5),
            torch.nn.functional.softplus(scores[2, 0] - scores[2, 2] + 0.5),
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    assert stats["failure_pair_active_frac"].item() == pytest.approx(2 / 3)


def test_summarize_candidate_failure_modes_detects_near_identity_wrong_top1():
    rows = [
        {
            "sample_name": "a",
            "candidate_idx": 0,
            "score": 2.0,
            "pose_cost_m": 0.50,
            "trans_err_m": 0.50,
            "rot_err_deg": 2.0,
            "delta_trans_m": 0.0,
            "delta_rot_deg": 7.5,
            "in_basin": False,
            "valid": True,
        },
        {
            "sample_name": "a",
            "candidate_idx": 1,
            "score": 1.9,
            "pose_cost_m": 0.10,
            "trans_err_m": 0.10,
            "rot_err_deg": 1.0,
            "delta_trans_m": 0.35,
            "delta_rot_deg": 7.5,
            "in_basin": True,
            "valid": True,
        },
        {
            "sample_name": "b",
            "candidate_idx": 0,
            "score": 1.0,
            "pose_cost_m": 0.20,
            "trans_err_m": 0.20,
            "rot_err_deg": 2.0,
            "delta_trans_m": 0.2,
            "delta_rot_deg": 5.0,
            "in_basin": True,
            "valid": True,
        },
    ]

    summary = summarize_candidate_failure_modes(rows, near_identity_trans_m=0.05, min_oracle_gap_m=0.05)

    assert summary["num_samples"] == 2
    assert summary["wrong_top1"] == 1
    assert summary["wrong_near_identity"] == 1
    assert summary["wrong_outside_basin"] == 1
    assert summary["replay_candidates"] == ["a"]


def test_sample_feature_at_pixels_maps_image_coordinates_to_feature_grid():
    feature = torch.zeros(2, 4, 4)
    feature[:, 0, 0] = torch.tensor([1.0, 0.0])
    feature[:, 3, 3] = torch.tensor([0.0, 1.0])
    xy = torch.tensor([[0.0, 0.0], [99.0, 99.0]])

    sampled = sample_feature_at_pixels(feature, xy, image_width=100, image_height=100)

    assert torch.allclose(sampled[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(sampled[1], torch.tensor([0.0, 1.0]))


def test_project_points_to_image_keeps_points_in_front_and_inside():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    w2c = torch.eye(4)
    xy, valid = project_points_to_image(
        xyz,
        w2c,
        camera_model="PINHOLE",
        camera_params=torch.tensor([10.0, 10.0, 5.0, 5.0]),
        image_width=10,
        image_height=10,
    )

    assert valid.tolist() == [True, False, False]
    assert torch.allclose(xy[0], torch.tensor([5.0, 5.0]))
