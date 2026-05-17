from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.tools.eval_render_loftr_refine import (
    cache_stem_for_image_name,
    maybe_write_teacher_correspondence,
    pose_cache_payload_from_records,
    resolve_split_file,
    samples_from_init_cache,
)
from pose_refine.tools import eval_render_loftr_refine as render_loftr_refine
from data.radio_loc_retrieval_dataset import load_retrieval_init_entries


def test_cache_stem_for_image_name_matches_pose_init_export_schema():
    assert cache_stem_for_image_name("seq8/frame00110.png") == "seq8_frame00110"
    assert cache_stem_for_image_name("frame00001.jpg") == "frame00001"


def test_resolve_split_file_uses_requested_dataset_key():
    ds_cfg = {
        "train_split": "/data/train.txt",
        "test_split": "/data/test.txt",
    }

    assert resolve_split_file(ds_cfg, "train_split") == "/data/train.txt"
    assert resolve_split_file(ds_cfg, "test_split") == "/data/test.txt"


def test_resolve_split_file_rejects_missing_dataset_key():
    try:
        resolve_split_file({"test_split": "/data/test.txt"}, "train_split")
    except KeyError as exc:
        assert "train_split" in str(exc)
    else:
        raise AssertionError("expected missing split key to raise")


class _FakeLoFTRResult:
    success = True
    extra = {
        "query_keypoints": np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
        "ref_keypoints": np.array([[11.0, 21.0], [31.0, 41.0], [51.0, 61.0]], dtype=np.float32),
        "pts3d_world": np.ones((3, 3), dtype=np.float32),
        "confidence": np.array([0.2, 0.9, 0.6], dtype=np.float32),
        "pnp_inlier_mask": np.array([True, False, True]),
        "loftr_hw": np.array([120, 200], dtype=np.int32),
    }


class _FakeIterLoFTRResult:
    def __init__(
        self,
        *,
        success: bool,
        pose_w2c: np.ndarray | None,
        num_inliers: int,
        num_raw_matches: int,
        failure_reason: str = "",
    ) -> None:
        self.success = success
        self.pose_w2c = pose_w2c
        self.num_inliers = num_inliers
        self.num_raw_matches = num_raw_matches
        self.failure_reason = failure_reason


def test_maybe_write_teacher_correspondence_uses_cache_stem_and_inlier_payload(tmp_path):
    out_path = maybe_write_teacher_correspondence(
        tmp_path,
        "seq8/frame00110.png",
        _FakeLoFTRResult(),
        max_points=8,
        inlier_only=True,
    )

    assert out_path == tmp_path / "seq8_frame00110.npz"
    payload = np.load(out_path, allow_pickle=True)
    assert payload["query_xy"].tolist() == [[50.0, 60.0], [10.0, 20.0]]
    assert payload["map_xy"].tolist() == [[51.0, 61.0], [11.0, 21.0]]
    assert payload["confidence"].tolist() == [0.6000000238418579, 0.20000000298023224]
    assert payload["pnp_inlier_mask"].tolist() == [True, True]
    assert str(payload["source"]) == "render_init_loftr"


def test_maybe_write_teacher_correspondence_preserves_all_match_inlier_mask(tmp_path):
    out_path = maybe_write_teacher_correspondence(
        tmp_path,
        "seq8/frame00110.png",
        _FakeLoFTRResult(),
        max_points=8,
        inlier_only=False,
    )

    payload = np.load(out_path, allow_pickle=True)
    assert payload["confidence"].tolist() == [
        0.8999999761581421,
        0.6000000238418579,
        0.20000000298023224,
    ]
    assert payload["pnp_inlier_mask"].tolist() == [False, True, True]


def test_record_from_iterative_refinement_results_keeps_last_success_after_failure():
    init_pose = np.eye(4, dtype=np.float32)
    first_pose = np.eye(4, dtype=np.float32)
    first_pose[:3, 3] = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    failed_pose = np.eye(4, dtype=np.float32)
    failed_pose[:3, 3] = np.array([0.2, 0.0, 0.0], dtype=np.float32)

    record = render_loftr_refine.record_from_iterative_refinement_results(
        query_image_name="seq1/frame00001.png",
        query_image_stem="seq1_frame00001",
        pose_init=init_pose,
        results=[
            _FakeIterLoFTRResult(
                success=True,
                pose_w2c=first_pose,
                num_inliers=21,
                num_raw_matches=44,
            ),
            _FakeIterLoFTRResult(
                success=False,
                pose_w2c=failed_pose,
                num_inliers=4,
                num_raw_matches=12,
                failure_reason="not_enough_inliers",
            ),
        ],
        requested_iterations=3,
    )

    assert record["success"] is True
    assert np.allclose(record["pose_refined"], first_pose)
    assert record["num_inliers"] == 21
    assert record["num_raw_matches"] == 44
    assert record["refine_iterations_requested"] == 3
    assert record["refine_attempted_iterations"] == 2
    assert record["refine_successful_iterations"] == 1
    assert record["refine_last_success_iteration"] == 1
    assert record["iteration_success"] == [True, False]
    assert record["iteration_num_inliers"] == [21, 4]
    assert record["iteration_num_raw_matches"] == [44, 12]
    assert record["failure_reason"] == "not_enough_inliers"


def test_pose_cache_payload_from_records_exports_iterative_refinement_scalars():
    init_pose = np.eye(4, dtype=np.float32)
    refined_pose = np.eye(4, dtype=np.float32)

    payload = pose_cache_payload_from_records(
        [
            {
                "query_image_name": "seq1/frame00001.png",
                "query_image_stem": "seq1_frame00001",
                "pose_init": init_pose,
                "pose_refined": refined_pose,
                "success": True,
                "num_inliers": 21,
                "num_raw_matches": 44,
                "refine_iterations_requested": 3,
                "refine_attempted_iterations": 2,
                "refine_successful_iterations": 1,
                "refine_last_success_iteration": 1,
            }
        ],
        source="render_loftr_refine",
    )

    assert payload["refine_iterations_requested"].tolist() == [3]
    assert payload["refine_attempted_iterations"].tolist() == [2]
    assert payload["refine_successful_iterations"].tolist() == [1]
    assert payload["refine_last_success_iteration"].tolist() == [1]


def test_samples_from_init_cache_keeps_cache_order_and_skips_missing_entries():
    pose_a = np.eye(4, dtype=np.float32)
    pose_b = np.eye(4, dtype=np.float32) * 2.0
    init_a = np.eye(4, dtype=np.float32)
    init_a[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    init_missing = np.eye(4, dtype=np.float32) * 9.0

    base_samples = [
        {"name": "seq1/frame00001.png", "query_path": Path("q1.png"), "pose_gt": pose_a},
        {"name": "seq2/frame00002.png", "query_path": Path("q2.png"), "pose_gt": pose_b},
    ]
    cache = {
        "query_image_stems": np.array(["seq1_frame00001", "seq9_frame99999"]),
        "pose_inits": np.stack([init_a, init_missing]).astype(np.float32),
        "init_sources": np.array(["renderloftr", "missing"]),
    }

    samples = samples_from_init_cache(base_samples, cache, max_samples=4)

    assert len(samples) == 1
    assert samples[0]["name"] == "seq1/frame00001.png"
    assert samples[0]["query_path"] == Path("q1.png")
    assert np.allclose(samples[0]["pose_gt"], pose_a)
    assert np.allclose(samples[0]["pose_init"], init_a)
    assert samples[0]["init_source"] == "renderloftr"


def test_pose_cache_payload_from_records_uses_refined_pose_when_successful():
    init_pose = np.eye(4, dtype=np.float32)
    refined_pose = np.eye(4, dtype=np.float32)
    refined_pose[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    failed_init = np.eye(4, dtype=np.float32) * 2.0

    payload = pose_cache_payload_from_records(
        [
            {
                "query_image_name": "seq1/frame00001.png",
                "query_image_stem": "seq1_frame00001",
                "pose_init": init_pose,
                "pose_refined": refined_pose,
                "success": True,
                "num_inliers": 21,
                "num_raw_matches": 30,
            },
            {
                "query_image_name": "seq2/frame00002.png",
                "query_image_stem": "seq2_frame00002",
                "pose_init": failed_init,
                "pose_refined": None,
                "success": False,
                "num_inliers": 0,
                "num_raw_matches": 4,
            },
        ],
        source="render_loftr_refine",
    )

    assert payload["query_image_stems"].tolist() == ["seq1_frame00001", "seq2_frame00002"]
    assert payload["query_image_names"].tolist() == ["seq1/frame00001.png", "seq2/frame00002.png"]
    assert np.allclose(payload["pose_inits"][0], refined_pose)
    assert np.allclose(payload["pose_inits"][1], failed_init)
    assert payload["refine_success"].tolist() == [True, False]
    assert payload["refine_num_inliers"].tolist() == [21, 0]
    assert payload["init_sources"].tolist() == ["render_loftr_refine", "render_loftr_refine_failed"]


def test_load_retrieval_init_entries_accepts_minimal_refined_pose_cache(tmp_path):
    pose = np.eye(4, dtype=np.float32)
    save_path = tmp_path / "minimal_refined_cache.npz"
    np.savez(
        save_path,
        query_image_names=np.array(["seq1/frame00001.png"]),
        query_image_stems=np.array(["seq1_frame00001"]),
        pose_inits=pose[None],
        init_sources=np.array(["render_loftr_refine"]),
    )

    entries, stats = load_retrieval_init_entries(str(save_path))

    assert stats == {}
    assert len(entries) == 1
    assert entries[0]["query_img_id"] == 0
    assert entries[0]["query_image_name"] == "seq1/frame00001.png"
    assert np.allclose(entries[0]["pose_init"], pose)
    assert entries[0]["retrieval_frame_id"] == -1
    assert entries[0]["pose_init_candidates"].shape == (1, 4, 4)
    assert entries[0]["candidate_valid_mask"].tolist() == [True]


def test_pose_cache_payload_from_records_is_retrieval_init_compatible(tmp_path):
    pose = np.eye(4, dtype=np.float32)
    payload = pose_cache_payload_from_records(
        [
            {
                "query_image_name": "seq1/frame00001.png",
                "query_image_stem": "seq1_frame00001",
                "pose_init": pose,
                "pose_refined": pose,
                "success": True,
                "num_inliers": 21,
                "num_raw_matches": 30,
            }
        ],
        source="render_loftr_refine",
    )
    save_path = tmp_path / "render_loftr_refined_cache.npz"
    np.savez(save_path, **payload)

    entries, _stats = load_retrieval_init_entries(str(save_path))

    assert entries[0]["query_img_id"] == 0
    assert entries[0]["retrieval_frame_id"] == -1
    assert entries[0]["retrieval_image_name"] == "seq1/frame00001.png"
    assert entries[0]["pose_init_candidates"].shape == (1, 4, 4)
    assert entries[0]["retrieval_scores_candidates"].shape == (1,)
