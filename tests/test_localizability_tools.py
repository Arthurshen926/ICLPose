import json
import argparse
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
from feature_extract.localizability.pose_cache_report import (
    query_names_from_candidate_table,
    summarize_pose_candidate_cache_geometry,
)
from feature_extract.tools.report_localizability import format_markdown_table, summarize_jsonl
from feature_extract.tools.eval_feature_track_mapability import (
    lookup_point_xyz,
    project_points_to_image,
    sample_feature_at_pixels,
)
from feature_extract.tools.train_localizability_selector_stream import _build_selector_stream_scorer
from feature_extract.tools.augment_pose_candidate_cache import (
    append_near_identity_candidate_arrays,
    build_near_identity_pose_candidates,
    parse_args as parse_augment_pose_candidate_args,
)
from feature_extract.tools.export_query_student_descriptors import (
    apply_student_feature_hw_defaults,
    apply_dataset_overrides,
    build_rgb_records,
    load_checkpoint_for_rgb_export,
    prepare_checkpoint_for_rgb_export,
    resolve_descriptor_key,
    resolve_feature_dir,
    save_dense_feature_map,
)
from feature_extract.tools.select_score_hard_candidate_cache import select_score_hard_candidate_arrays
from feature_extract.tools.select_score_hard_candidate_cache import parse_args as parse_score_hard_candidate_args
from feature_extract.tools.shuffle_pose_candidate_cache import permute_pose_candidate_cache_arrays
from feature_extract.tools.train_localizability_score_calibrator import (
    _row_with_selection_eligibility,
    _selection_eligible,
    parse_args as parse_score_calibrator_args,
)


def _pose_at(value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(value)
    return pose


def test_score_calibrator_cli_accepts_selection_spearman_min(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_localizability_score_calibrator.py",
            "--train-table",
            "train.jsonl",
            "--val-table",
            "val.jsonl",
            "--out-dir",
            "out",
            "--selection-spearman-min",
            "0.55",
        ],
    )

    args = parse_score_calibrator_args()

    assert args.selection_spearman_min == 0.55


def test_score_calibrator_selection_gate_rejects_low_spearman_checkpoint():
    best = {"pred_cost_m": 0.23}
    low_spearman = {"pred_cost_m": 0.20, "spearman": 0.50}
    stable = {"pred_cost_m": 0.22, "spearman": 0.56}

    assert _selection_eligible(low_spearman, best, selection_spearman_min=None) is True
    assert _selection_eligible(low_spearman, best, selection_spearman_min=0.55) is False
    assert _selection_eligible(stable, best, selection_spearman_min=0.55) is True


def test_score_calibrator_eval_row_marks_selection_eligibility_before_logging():
    row = {"val": {"pred_cost_m": 0.21, "spearman": 0.50}}

    marked = _row_with_selection_eligibility(row, {"pred_cost_m": 0.23}, selection_spearman_min=0.55)

    assert marked is row
    assert marked["selection_eligible"] is False


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


def test_apply_dataset_overrides_updates_scene_paths():
    cfg = {"dataset": {"source_dir": "old", "image_patterns": ["*.png"]}}
    args = type(
        "Args",
        (),
        {
            "source_dir": "/data/scene",
            "colmap_dir": "/data/scene/sparse/0",
            "train_split": "/data/scene/train.txt",
            "val_split": "/data/scene/test.txt",
            "image_patterns": "seq*/*.png,images/*.jpg",
        },
    )()

    apply_dataset_overrides(cfg, args)

    assert cfg["dataset"]["source_dir"] == "/data/scene"
    assert cfg["dataset"]["colmap_dir"] == "/data/scene/sparse/0"
    assert cfg["dataset"]["train_split"] == "/data/scene/train.txt"
    assert cfg["dataset"]["val_split"] == "/data/scene/test.txt"
    assert cfg["dataset"]["image_patterns"] == ["seq*/*.png", "images/*.jpg"]


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


def test_load_checkpoint_for_rgb_export_falls_back_for_trusted_legacy_checkpoint(monkeypatch):
    import pickle
    import feature_extract.tools.export_query_student_descriptors as exporter

    expected = {"model_state_dict": {"fine_head.weight": torch.tensor([1.0])}}

    def fake_safe_load(path):
        raise pickle.UnpicklingError("Weights only load failed")

    def fake_torch_load(path, map_location=None, weights_only=None):
        assert str(path) == "legacy.pth"
        assert map_location == "cpu"
        assert weights_only is False
        return expected

    monkeypatch.setattr(exporter, "safe_torch_load", fake_safe_load)
    monkeypatch.setattr(exporter.torch, "load", fake_torch_load)

    assert load_checkpoint_for_rgb_export("legacy.pth") is expected


def test_resolve_descriptor_key_keeps_model_fine_explicit():
    assert resolve_descriptor_key("fine", fine_key="fine_loc") == "fine"
    assert resolve_descriptor_key("export_fine", fine_key="fine_loc") == "fine_loc"


def test_save_dense_feature_map_uses_colmap_image_id_and_requested_resolution(tmp_path):
    feature = torch.randn(4, 8, 10)

    path = save_dense_feature_map(
        feature,
        output_root=tmp_path,
        subdir="fine_geo",
        record={"image_id": 7, "sample_name": "seq/frame.png"},
        feature_key="fine_geo",
        resize_hw=(2, 3),
        dtype="float16",
    )

    assert path.name == "rgb_7_fine_geo_4x2x3.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved.shape == (4, 2, 3)
    assert saved.dtype == torch.float16


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


def test_summarize_pose_candidate_cache_geometry_reports_oracle_basin(tmp_path):
    cache_path = tmp_path / "toy_candidates.npz"
    candidates = np.tile(np.eye(4, dtype=np.float32), (2, 3, 1, 1))
    candidates[0, 0, 0, 3] = 0.40
    candidates[0, 1, 0, 3] = 0.10
    candidates[0, 2, 0, 3] = 0.60
    candidates[1, 0, 0, 3] = 0.30
    candidates[1, 1, 0, 3] = 0.35
    candidates[1, 2, 0, 3] = 0.05
    np.savez(
        cache_path,
        query_image_names=np.array(["q0.png", "q1.png"]),
        pose_init_candidates=candidates,
        candidate_valid_mask=np.ones((2, 3), dtype=bool),
    )

    report = summarize_pose_candidate_cache_geometry(
        cache_path=cache_path,
        label="toy",
        gt_poses_by_name={"q0.png": np.eye(4, dtype=np.float32), "q1.png": np.eye(4, dtype=np.float32)},
        basin_trans_m=0.25,
        basin_rot_deg=10.0,
        topk=(1, 2, 3),
    )

    assert report["num_samples"] == 2
    assert report["num_candidates"] == 3
    assert report["basin_recall_by_order"]["@1"] == pytest.approx(0.0)
    assert report["basin_recall_by_order"]["@2"] == pytest.approx(0.5)
    assert report["basin_recall_by_order"]["@3"] == pytest.approx(1.0)
    assert report["oracle"]["trans_m_quantiles"]["q50"] == pytest.approx(0.075)


def test_permute_pose_candidate_cache_arrays_keeps_candidate_metadata_aligned():
    poses = np.zeros((2, 4, 4, 4), dtype=np.float32)
    for row in range(2):
        for cand in range(4):
            poses[row, cand] = np.eye(4, dtype=np.float32)
            poses[row, cand, 0, 3] = row * 10 + cand
    cache = {
        "query_image_names": np.array(["q0.png", "q1.png"]),
        "pose_init_candidates": poses,
        "candidate_valid_mask": np.array([[True, False, True, True], [True, True, False, True]]),
        "retrieval_scores_candidates": np.array([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
        "retrieval_image_names_candidates": np.array([["a", "b", "c", "d"], ["e", "f", "g", "h"]]),
        "pose_inits": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }
    permutations = np.array([[2, 0, 3, 1], [3, 1, 0, 2]], dtype=np.int64)

    shuffled, metadata = permute_pose_candidate_cache_arrays(cache, permutations=permutations, seed=123)

    assert np.allclose(shuffled["pose_init_candidates"][0, :, 0, 3], [2, 0, 3, 1])
    assert np.allclose(shuffled["pose_init_candidates"][1, :, 0, 3], [13, 11, 10, 12])
    assert shuffled["candidate_valid_mask"].tolist() == [[True, True, True, False], [True, True, True, False]]
    assert np.allclose(shuffled["retrieval_scores_candidates"], [[0.2, 0.0, 0.3, 0.1], [1.3, 1.1, 1.0, 1.2]])
    assert shuffled["retrieval_image_names_candidates"].tolist() == [["c", "a", "d", "b"], ["h", "f", "e", "g"]]
    assert shuffled["candidate_permutation"].tolist() == permutations.tolist()
    assert metadata["num_rows"] == 2
    assert metadata["num_candidates"] == 4
    assert metadata["seed"] == 123


def test_permute_pose_candidate_cache_arrays_balances_generated_candidate_positions():
    poses = np.zeros((8, 4, 4, 4), dtype=np.float32)
    old_permutation = np.tile(np.array([10, 11, 12, 13], dtype=np.int64), (8, 1))
    cache = {
        "pose_init_candidates": poses,
        "candidate_valid_mask": np.ones((8, 4), dtype=bool),
        "candidate_permutation": old_permutation,
    }

    shuffled, metadata = permute_pose_candidate_cache_arrays(cache, seed=5)
    permutation = shuffled["candidate_permutation"]

    assert metadata["shuffle_mode"] == "balanced_cyclic_random_base"
    counts = np.zeros((4, 4), dtype=np.int64)
    for row in permutation:
        for rank, old_idx in enumerate(row - 10):
            counts[rank, old_idx] += 1
    assert counts.min() == counts.max()
    assert np.all(np.sort(permutation, axis=1) == old_permutation)


def test_argparse_boolean_optional_compat_supports_no_include_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "augment_pose_candidate_cache.py",
            "--input",
            "in.npz",
            "--output",
            "out.npz",
            "--no-include-identity",
        ],
    )
    augment_args = parse_augment_pose_candidate_args()
    assert augment_args.include_identity is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_score_hard_candidate_cache.py",
            "--input",
            "in.npz",
            "--candidate-table",
            "rows.jsonl",
            "--output",
            "out.npz",
            "--no-include-oracle",
        ],
    )
    select_args = parse_score_hard_candidate_args()
    assert select_args.include_oracle is False


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


def test_select_score_hard_candidate_arrays_keeps_oracle_and_score_high_wrong():
    poses = np.zeros((2, 4, 4, 4), dtype=np.float32)
    for row in range(2):
        for cand in range(4):
            poses[row, cand] = np.eye(4, dtype=np.float32)
            poses[row, cand, 0, 3] = row * 10 + cand
    cache = {
        "pose_init_candidates": poses,
        "candidate_valid_mask": np.ones((2, 4), dtype=bool),
        "retrieval_scores_candidates": np.arange(8, dtype=np.float32).reshape(2, 4),
        "pose_inits": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }
    rows = [
        {"global_row": 0, "candidate_idx": 0, "pose_cost_m": 0.10, "score": 0.2, "valid": True},
        {"global_row": 0, "candidate_idx": 1, "pose_cost_m": 0.15, "score": 0.9, "valid": True},
        {"global_row": 0, "candidate_idx": 2, "pose_cost_m": 0.35, "score": 0.8, "valid": True},
        {"global_row": 1, "candidate_idx": 0, "pose_cost_m": 0.50, "score": 0.7, "valid": True},
        {"global_row": 1, "candidate_idx": 1, "pose_cost_m": 0.05, "score": 0.1, "valid": True},
        {"global_row": 1, "candidate_idx": 3, "pose_cost_m": 0.30, "score": 0.9, "valid": True},
    ]

    selected, metadata = select_score_hard_candidate_arrays(
        cache,
        rows,
        hard_count=1,
        cost_gap_m=0.12,
        include_oracle=True,
    )

    assert selected["pose_init_candidates"].shape == (2, 2, 4, 4)
    assert selected["candidate_valid_mask"].all()
    assert selected["pose_init_candidates"][0, 0, 0, 3] == 0.0
    assert selected["pose_init_candidates"][0, 1, 0, 3] == 2.0
    assert selected["pose_init_candidates"][1, 0, 0, 3] == 11.0
    assert selected["pose_init_candidates"][1, 1, 0, 3] == 13.0
    assert metadata["selected_indices"] == [[0, 2], [1, 3]]


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


def test_lookup_point_xyz_aligns_observation_tracks_to_colmap_points():
    point_ids = torch.tensor([10, 42, 99])
    point_xyz = torch.tensor([[1.0, 0.0, 0.0], [4.0, 2.0, 1.0], [9.0, 9.0, 9.0]])
    obs_point_ids = torch.tensor([42, 10, 42])

    xyz = lookup_point_xyz(point_ids, point_xyz, obs_point_ids)

    assert torch.allclose(xyz, torch.tensor([[4.0, 2.0, 1.0], [1.0, 0.0, 0.0], [4.0, 2.0, 1.0]]))


def test_query_names_from_candidate_table_deduplicates_in_order(tmp_path):
    table = tmp_path / "hard_case.jsonl"
    table.write_text(
        "\n".join(
            [
                json.dumps({"sample_name": "a", "candidate_idx": 0}),
                json.dumps({"sample_name": "a", "candidate_idx": 1}),
                json.dumps({"sample_name": "b", "candidate_idx": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert query_names_from_candidate_table(table) == ["a", "b"]


def test_selector_stream_pair_matcher_mode_requires_loaded_pair_matcher():
    args = argparse.Namespace(
        score_mode="pair_matcher_local",
        score_radius=16,
        score_temperature=0.04,
        pair_matcher_stride=4,
        pair_matcher_chunk_points=256,
        pair_matcher_candidate_chunk_size=1,
        pair_matcher_offset_chunk_size=32,
        pair_matcher_candidate_score_mode="center_logprob_margin",
        pair_matcher_score_channel=0,
    )

    with pytest.raises(ValueError, match="requires --pose-adapter-checkpoint"):
        _build_selector_stream_scorer(args, pair_matcher=None)

    matcher = torch.nn.Identity()
    scorer = _build_selector_stream_scorer(args, pair_matcher=matcher)

    assert scorer.mode == "pair_matcher_local"
    assert scorer.pair_matcher is matcher
    assert scorer.pair_matcher_stride == 4
