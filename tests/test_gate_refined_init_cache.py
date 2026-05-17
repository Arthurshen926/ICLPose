from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
try:
    import faiss  # noqa: F401
except ImportError:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries
from feature_retrieval.tools.gate_refined_init_cache import (
    gate_refined_init_cache,
    gate_refined_init_cache_by_pose_step,
    gate_refined_init_cache_by_score_delta,
    main as gate_refined_init_cache_main,
)


def _pose_at(value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(value)
    return pose


def _base_entry(idx: int) -> dict:
    candidates = np.stack(
        [
            _pose_at(idx * 10.0 + 1.0),
            _pose_at(idx * 10.0 + 2.0),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "query_img_id": idx,
        "query_image_name": f"seq/frame{idx:05d}.png",
        "query_image_stem": f"seq_frame{idx:05d}",
        "pose_init": candidates[0].copy(),
        "init_source": "base_init",
        "retrieval_frame_id": 100 + idx,
        "retrieval_image_name": f"ref/frame{idx:05d}.png",
        "retrieval_score": 0.5 + idx,
        "pose_init_candidates": candidates,
        "candidate_valid_mask": np.array([True, True], dtype=bool),
        "retrieval_frame_ids_candidates": np.array([100 + idx, 200 + idx], dtype=np.int64),
        "retrieval_image_names_candidates": np.array(
            [f"ref/a{idx:05d}.png", f"ref/b{idx:05d}.png"]
        ),
        "retrieval_scores_candidates": np.array([0.9, 0.8], dtype=np.float32),
        "retrieval_pnp_num_inliers_candidates": np.array([11.0, 22.0], dtype=np.float32),
    }


def _write_base_cache(tmp_path: Path) -> Path:
    path = tmp_path / "base_init.npz"
    save_retrieval_init_entries([_base_entry(1), _base_entry(2)], {"method_used": "base"}, str(path))
    return path


def _write_refined_cache(tmp_path: Path) -> Path:
    path = tmp_path / "refined_init.npz"
    np.savez(
        path,
        query_img_ids=np.array([0, 1], dtype=np.int64),
        query_image_names=np.array(["seq/frame00001.png", "seq/frame00002.png"]),
        query_image_stems=np.array(["seq_frame00001", "seq_frame00002"]),
        pose_inits=np.stack([_pose_at(101.0), _pose_at(202.0)]).astype(np.float32),
        init_sources=np.array(["render_loftr_refine", "render_loftr_refine"]),
        retrieval_frame_ids=np.array([-1, -1], dtype=np.int64),
        retrieval_image_names=np.array(["seq/frame00001.png", "seq/frame00002.png"]),
        retrieval_scores=np.array([0.0, 0.0], dtype=np.float32),
        refine_success=np.array([True, True], dtype=bool),
        refine_num_inliers=np.array([30, 5], dtype=np.int32),
        refine_num_raw_matches=np.array([50, 40], dtype=np.int32),
    )
    return path


def test_gate_refined_init_cache_accepts_quality_refined_pose_and_preserves_base_metadata(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    refined_cache = _write_refined_cache(tmp_path)
    save_path = tmp_path / "gated_init.npz"

    exported_entries, stats = gate_refined_init_cache(
        str(base_cache),
        str(refined_cache),
        str(save_path),
        min_inliers=20,
        min_raw_matches=10,
        min_inlier_ratio=0.5,
        source_name="loftr_quality_gate",
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert stats == loaded_stats
    assert stats["num_entries"] == 2
    assert stats["num_refined_accepted"] == 1
    assert stats["num_refined_rejected"] == 1
    assert len(exported_entries) == 2

    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(101.0))
    assert loaded_entries[0]["init_source"] == "loftr_quality_gate"
    assert loaded_entries[0]["retrieval_frame_id"] == 101
    assert loaded_entries[0]["retrieval_image_name"] == "ref/frame00001.png"
    assert np.allclose(loaded_entries[0]["pose_init_candidates"], _base_entry(1)["pose_init_candidates"])
    assert np.allclose(
        loaded_entries[0]["retrieval_pnp_num_inliers_candidates"],
        np.array([11.0, 22.0], dtype=np.float32),
    )

    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(21.0))
    assert loaded_entries[1]["init_source"] == "base_init"
    assert loaded_entries[1]["retrieval_frame_id"] == 102


def test_gate_refined_init_cache_rejects_unmatched_refined_cache(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    refined_cache = tmp_path / "bad_refined.npz"
    np.savez(
        refined_cache,
        query_image_names=np.array(["seq/frame99999.png"]),
        query_image_stems=np.array(["seq_frame99999"]),
        pose_inits=np.stack([_pose_at(999.0)]).astype(np.float32),
        init_sources=np.array(["render_loftr_refine"]),
        refine_success=np.array([True], dtype=bool),
        refine_num_inliers=np.array([30], dtype=np.int32),
        refine_num_raw_matches=np.array([50], dtype=np.int32),
    )

    try:
        gate_refined_init_cache(str(base_cache), str(refined_cache), str(tmp_path / "out.npz"))
    except ValueError as exc:
        assert "missing refined entries" in str(exc)
    else:
        raise AssertionError("expected unmatched refined cache to raise")


def _write_pose_step_refined_cache(
    tmp_path: Path,
    name: str,
    pose_values: list[float],
    *,
    inliers: list[int] | None = None,
    raw_matches: list[int] | None = None,
) -> Path:
    if inliers is None:
        inliers = [40 for _ in pose_values]
    if raw_matches is None:
        raw_matches = [80 for _ in pose_values]
    path = tmp_path / f"{name}.npz"
    np.savez(
        path,
        query_img_ids=np.array([0, 1], dtype=np.int64),
        query_image_names=np.array(["seq/frame00001.png", "seq/frame00002.png"]),
        query_image_stems=np.array(["seq_frame00001", "seq_frame00002"]),
        pose_inits=np.stack([_pose_at(v) for v in pose_values]).astype(np.float32),
        init_sources=np.array([name, name]),
        refine_success=np.array([True, True], dtype=bool),
        refine_num_inliers=np.array(inliers, dtype=np.int32),
        refine_num_raw_matches=np.array(raw_matches, dtype=np.int32),
    )
    return path


def test_gate_refined_init_cache_by_pose_step_selects_candidate_only_for_large_step(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    reference_cache = _write_pose_step_refined_cache(tmp_path, "one_pass", [101.0, 202.0])
    candidate_cache = _write_pose_step_refined_cache(tmp_path, "iter3", [101.1, 203.0])
    save_path = tmp_path / "step_gated_init.npz"

    exported_entries, stats = gate_refined_init_cache_by_pose_step(
        str(base_cache),
        str(reference_cache),
        str(candidate_cache),
        str(save_path),
        min_step_m=0.5,
        reference_source_name="one_pass_default",
        candidate_source_name="iter3_large_step",
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert stats == loaded_stats
    assert stats["num_reference_selected"] == 1
    assert stats["num_candidate_selected"] == 1
    assert stats["num_base_fallback"] == 0
    assert len(exported_entries) == 2

    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(101.0))
    assert loaded_entries[0]["init_source"] == "one_pass_default"
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(203.0))
    assert loaded_entries[1]["init_source"] == "iter3_large_step"
    assert np.allclose(loaded_entries[1]["pose_init_candidates"], _base_entry(2)["pose_init_candidates"])


def test_gate_refined_init_cache_by_pose_step_can_use_separate_step_cache(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    reference_cache = _write_pose_step_refined_cache(
        tmp_path,
        "one_pass",
        [101.0, 202.0],
        inliers=[21, 22],
        raw_matches=[51, 52],
    )
    step_cache = _write_pose_step_refined_cache(tmp_path, "iter2", [101.1, 203.0])
    candidate_cache = _write_pose_step_refined_cache(
        tmp_path,
        "iter3",
        [301.0, 402.0],
        inliers=[31, 32],
        raw_matches=[61, 62],
    )
    save_path = tmp_path / "separate_step_gated_init.npz"

    _exported_entries, stats = gate_refined_init_cache_by_pose_step(
        str(base_cache),
        str(reference_cache),
        str(candidate_cache),
        str(save_path),
        step_candidate_refined_cache_path=str(step_cache),
        min_step_m=0.5,
        reference_source_name="one_pass_default",
        candidate_source_name="iter3_large_iter2_step",
    )
    loaded_entries, _loaded_stats = load_retrieval_init_entries(str(save_path))

    assert stats["step_candidate_refined_cache"] == str(step_cache)
    assert stats["num_reference_selected"] == 1
    assert stats["num_candidate_selected"] == 1
    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(101.0))
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(402.0))
    assert loaded_entries[1]["init_source"] == "iter3_large_iter2_step"
    saved = np.load(save_path, allow_pickle=True)
    assert saved["refine_success"].tolist() == [True, True]
    assert saved["refine_num_inliers"].tolist() == [21, 32]
    assert saved["refine_num_raw_matches"].tolist() == [51, 62]


def test_gate_refined_init_cache_cli_exports_pose_step_gate(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    reference_cache = _write_pose_step_refined_cache(tmp_path, "one_pass", [101.0, 202.0])
    step_cache = _write_pose_step_refined_cache(tmp_path, "iter2", [101.1, 203.0])
    candidate_cache = _write_pose_step_refined_cache(tmp_path, "iter3", [301.0, 402.0])
    save_path = tmp_path / "cli_step_gated_init.npz"

    rc = gate_refined_init_cache_main(
        [
            "--base_cache",
            str(base_cache),
            "--refined_cache",
            str(reference_cache),
            "--candidate_refined_cache",
            str(candidate_cache),
            "--step_candidate_refined_cache",
            str(step_cache),
            "--save_path",
            str(save_path),
            "--min_step_m",
            "0.5",
            "--reference_source_name",
            "one_pass_default",
            "--candidate_source_name",
            "iter3_large_iter2_step",
        ]
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert rc == 0
    assert loaded_stats["selection_source"] == "pose_step_gated_refined_pose"
    assert loaded_stats["num_candidate_selected"] == 1
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(402.0))
    assert loaded_entries[1]["init_source"] == "iter3_large_iter2_step"


def _write_scored_refined_cache(
    tmp_path: Path,
    name: str,
    pose_values: list[float],
    scores: list[float],
    *,
    inliers: list[int] | None = None,
    raw_matches: list[int] | None = None,
) -> Path:
    if inliers is None:
        inliers = [40 for _ in pose_values]
    if raw_matches is None:
        raw_matches = [80 for _ in pose_values]
    path = tmp_path / f"{name}.npz"
    np.savez(
        path,
        query_img_ids=np.array([0, 1], dtype=np.int64),
        query_image_names=np.array(["seq/frame00001.png", "seq/frame00002.png"]),
        query_image_stems=np.array(["seq_frame00001", "seq_frame00002"]),
        pose_inits=np.stack([_pose_at(v) for v in pose_values]).astype(np.float32),
        init_sources=np.array([name, name]),
        refine_success=np.array([True, True], dtype=bool),
        refine_num_inliers=np.array(inliers, dtype=np.int32),
        refine_num_raw_matches=np.array(raw_matches, dtype=np.int32),
        feature_residual_mean=np.array(scores, dtype=np.float32),
        feature_valid_frac=np.array([0.75, 0.80], dtype=np.float32),
    )
    return path


def test_gate_refined_init_cache_by_score_delta_selects_lower_feature_residual(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    reference_cache = _write_scored_refined_cache(
        tmp_path,
        "one_pass_scored",
        [101.0, 202.0],
        [0.30, 0.30],
    )
    candidate_cache = _write_scored_refined_cache(
        tmp_path,
        "iter3_scored",
        [301.0, 402.0],
        [0.28, 0.10],
    )
    save_path = tmp_path / "score_delta_gated_init.npz"

    _exported_entries, stats = gate_refined_init_cache_by_score_delta(
        str(base_cache),
        str(reference_cache),
        str(candidate_cache),
        str(save_path),
        score_key="feature_residual_mean",
        min_score_improvement=0.05,
        reference_source_name="one_pass_feature_default",
        candidate_source_name="iter3_feature_better",
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))
    saved = np.load(save_path, allow_pickle=True)

    assert stats == loaded_stats
    assert stats["selection_source"] == "score_delta_gated_refined_pose"
    assert stats["score_key"] == "feature_residual_mean"
    assert stats["num_reference_selected"] == 1
    assert stats["num_candidate_selected"] == 1
    assert np.isclose(stats["mean_candidate_score_improvement"], 0.20)
    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(101.0))
    assert loaded_entries[0]["init_source"] == "one_pass_feature_default"
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(402.0))
    assert loaded_entries[1]["init_source"] == "iter3_feature_better"
    assert np.allclose(saved["refine_score_selected"], np.array([0.30, 0.10], dtype=np.float32))
    assert np.allclose(saved["refine_score_reference"], np.array([0.30, 0.30], dtype=np.float32))
    assert np.allclose(saved["refine_score_candidate"], np.array([0.28, 0.10], dtype=np.float32))


def test_gate_refined_init_cache_by_score_delta_can_require_pose_step(tmp_path):
    base_cache = _write_base_cache(tmp_path)
    reference_cache = _write_scored_refined_cache(
        tmp_path,
        "one_pass_scored",
        [101.0, 202.0],
        [0.30, 0.30],
    )
    step_cache = _write_pose_step_refined_cache(tmp_path, "iter2_step", [101.1, 203.0])
    candidate_cache = _write_scored_refined_cache(
        tmp_path,
        "iter3_scored",
        [301.0, 402.0],
        [0.10, 0.10],
    )
    save_path = tmp_path / "score_step_gated_init.npz"

    _exported_entries, stats = gate_refined_init_cache_by_score_delta(
        str(base_cache),
        str(reference_cache),
        str(candidate_cache),
        str(save_path),
        step_candidate_refined_cache_path=str(step_cache),
        score_key="feature_residual_mean",
        min_score_improvement=0.05,
        min_step_m=0.5,
        reference_source_name="one_pass_feature_default",
        candidate_source_name="iter3_feature_step_better",
    )
    loaded_entries, _loaded_stats = load_retrieval_init_entries(str(save_path))

    assert stats["step_candidate_refined_cache"] == str(step_cache)
    assert stats["num_reference_selected"] == 1
    assert stats["num_candidate_selected"] == 1
    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(101.0))
    assert loaded_entries[0]["init_source"] == "one_pass_feature_default"
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(402.0))
    assert loaded_entries[1]["init_source"] == "iter3_feature_step_better"
