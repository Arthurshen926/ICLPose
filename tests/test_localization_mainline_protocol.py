from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries
from feature_retrieval.localization_mainline import (
    build_ablation_report,
    build_hybrid_init_entries,
    sha256_file,
    summarize_init_entries,
)


def _pose_at(center_x_m: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = -float(center_x_m)
    return pose


def _entry(
    image_name: str,
    pose: np.ndarray,
    *,
    source: str,
    score: float = 1.0,
    frame_id: int = 10,
) -> dict:
    image_stem = image_name.replace("/", "_").rsplit(".", 1)[0]
    return {
        "query_img_id": frame_id,
        "query_image_name": image_name,
        "query_image_stem": image_stem,
        "pose_init": pose.astype(np.float32),
        "init_source": source,
        "retrieval_frame_id": frame_id + 100,
        "retrieval_image_name": f"ref/{frame_id:06d}.png",
        "retrieval_score": float(score),
        "pose_init_candidates": pose[None].astype(np.float32),
        "candidate_valid_mask": np.array([True], dtype=bool),
        "retrieval_frame_ids_candidates": np.array([frame_id + 100], dtype=np.int64),
        "retrieval_image_names_candidates": np.array([f"ref/{frame_id:06d}.png"]),
        "retrieval_scores_candidates": np.array([score], dtype=np.float32),
    }


def test_hybrid_external_first_cache_prefers_successful_external_candidate(tmp_path):
    external_entries = [
        _entry(
            "seq/q1.png",
            _pose_at(0.05),
            source="netvlad_render_loftr_pnp",
            score=42.0,
            frame_id=1,
        ),
        _entry(
            "seq/q2.png",
            _pose_at(8.0),
            source="netvlad_render_loftr_pnp_retrieval_fallback",
            score=0.5,
            frame_id=2,
        ),
    ]
    learned_entries = [
        _entry("seq/q1.png", _pose_at(2.0), source="learned_pose_bank", score=0.7, frame_id=11),
        _entry("seq/q2.png", _pose_at(0.10), source="learned_pose_bank", score=0.9, frame_id=12),
    ]

    entries, stats = build_hybrid_init_entries(
        external_entries=external_entries,
        learned_entries=learned_entries,
        topk=3,
        external_min_score=5.0,
        source_name="fixed_hybrid_test",
    )
    save_path = tmp_path / "fixed_hybrid_init.npz"
    save_retrieval_init_entries(entries, stats, str(save_path))
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))

    assert loaded_stats["method_used"] == "fixed_hybrid_test"
    assert loaded_stats["counts_by_selected_source"] == {
        "netvlad_render_loftr_pnp": 1,
        "learned_pose_bank": 1,
    }
    assert loaded_entries[0]["init_source"] == "netvlad_render_loftr_pnp"
    assert np.allclose(loaded_entries[0]["pose_init"], _pose_at(0.05))
    assert np.allclose(loaded_entries[0]["pose_init_candidates"][1], _pose_at(2.0))
    assert loaded_entries[0]["candidate_valid_mask"].tolist() == [True, True, False]

    assert loaded_entries[1]["init_source"] == "learned_pose_bank"
    assert np.allclose(loaded_entries[1]["pose_init"], _pose_at(0.10))
    assert loaded_entries[1]["candidate_valid_mask"].tolist() == [True, True, False]


def test_init_cache_summary_reports_hash_and_pose_metrics(tmp_path):
    entries = [
        _entry("seq/q1.png", _pose_at(0.04), source="fixed_hybrid_test", frame_id=1),
        _entry("seq/q2.png", _pose_at(0.20), source="fixed_hybrid_test", frame_id=2),
    ]
    save_path = tmp_path / "fixed_init.npz"
    save_retrieval_init_entries(entries, {"method_used": "fixed_hybrid_test"}, str(save_path))
    loaded_entries, _stats = load_retrieval_init_entries(str(save_path))

    summary = summarize_init_entries(
        loaded_entries,
        gt_poses_by_name={
            "seq/q1.png": _pose_at(0.0),
            "seq/q2.png": _pose_at(0.0),
        },
        cache_path=str(save_path),
    )

    assert summary["init_cache_sha256"] == sha256_file(str(save_path))
    assert summary["num_samples"] == 2
    assert summary["init_metrics"]["trans_median"] == pytest.approx(120.0)
    assert summary["init_metrics"]["joint_1deg_50mm"] == 50.0
    assert summary["init_metrics"]["joint_1deg_100mm"] == 50.0


def test_ablation_report_requires_fixed_init_and_reports_gain():
    shared_hash = "abc123"
    run_summaries = [
        {
            "name": "fixed-init-existing",
            "component": "existing feature map/refiner",
            "init_pose_cache": "/tmp/fixed_init.npz",
            "init_cache_sha256": shared_hash,
            "init_metrics": {
                "rot_median": 2.0,
                "trans_median": 500.0,
                "joint_1deg_50mm": 10.0,
                "joint_1deg_100mm": 20.0,
            },
            "final_metrics": {
                "rot_median": 1.5,
                "trans_median": 450.0,
                "joint_1deg_50mm": 12.0,
                "joint_1deg_100mm": 25.0,
            },
        },
        {
            "name": "plus-loc-feature-reconstruction",
            "component": "localization-driven map feature reconstruction",
            "init_pose_cache": "/tmp/fixed_init.npz",
            "init_cache_sha256": shared_hash,
            "init_metrics": {
                "rot_median": 2.0,
                "trans_median": 500.0,
                "joint_1deg_50mm": 10.0,
                "joint_1deg_100mm": 20.0,
            },
            "final_metrics": {
                "rot_median": 0.8,
                "trans_median": 300.0,
                "joint_1deg_50mm": 40.0,
                "joint_1deg_100mm": 55.0,
            },
            "feature_metrics": {
                "flow_epe": 1.2,
                "peak_acc": 62.0,
                "wls_gain_mm": 35.0,
                "feature_reconstruction_cosine": 0.84,
            },
        },
    ]

    report = build_ablation_report(run_summaries, fixed_init_sha256=shared_hash)

    assert report["fixed_init_cache"]["sha256"] == shared_hash
    assert report["runs"][0]["trans_median_gain_mm"] == 50.0
    assert report["runs"][1]["trans_median_gain_mm"] == 200.0
    assert report["runs"][1]["final_recall_1deg_50mm"] == 40.0
    assert report["runs"][1]["feature_reconstruction_cosine"] == 0.84

    mismatched = dict(run_summaries[1])
    mismatched["init_cache_sha256"] = "different"
    with pytest.raises(ValueError, match="same fixed init cache"):
        build_ablation_report([run_summaries[0], mismatched], fixed_init_sha256=shared_hash)
