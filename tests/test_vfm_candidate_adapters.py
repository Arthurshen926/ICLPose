import json

import numpy as np

from feature_extract.vfm.candidate_adapters import (
    candidate_from_normalized_record,
    load_candidate_records,
    load_pose_init_npz_records,
    load_reference_pose_bank_npz_records,
)


def test_candidate_adapter_loads_normalized_json_and_preserves_query_id(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q0",
                    "candidate_id": "c0",
                    "candidate_type": "reference_pose",
                    "translation_m": 0.1,
                    "rotation_deg": 2.0,
                    "prior_score": 0.7,
                    "retrieval_rank": 1,
                }
            ]
        )
    )

    candidates = load_candidate_records(path)

    assert candidates[0].query_id == "q0"
    assert candidates[0].pose_error.translation_m == 0.1
    assert candidates[0].metadata["retrieval_rank"] == 1


def test_candidate_adapter_loads_csv_with_optional_pose_cost(tmp_path):
    path = tmp_path / "candidates.csv"
    path.write_text(
        "query_id,candidate_id,candidate_type,prior_score,retrieval_rank\n"
        "q0,c0,place,0.5,3\n"
    )

    candidate = load_candidate_records(path)[0]

    assert candidate.query_id == "q0"
    assert candidate.pose_error is None
    assert candidate.prior_score == 0.5
    assert candidate.metadata["retrieval_rank"] == 3


def test_candidate_adapter_accepts_nested_pose_error():
    candidate = candidate_from_normalized_record(
        {
            "query_id": "q1",
            "candidate_id": "c1",
            "candidate_type": "rendered_pose",
            "pose_error": {"translation_m": 0.2, "rotation_deg": 4.0},
        }
    )

    assert candidate.query_id == "q1"
    assert candidate.pose_error.rotation_deg == 4.0


def test_pose_init_npz_adapter_emits_valid_retrieval_candidates(tmp_path):
    path = tmp_path / "pose_init.npz"
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
    np.savez_compressed(
        path,
        query_image_names=np.array(["seq8/frame00110.png"]),
        query_image_stems=np.array(["frame00110"]),
        pose_init_candidates=poses,
        candidate_valid_mask=np.array([[True, False]]),
        retrieval_image_names_candidates=np.array([["seq1/frame00001.png", "seq1/frame00002.png"]]),
        retrieval_scores_candidates=np.array([[0.9, 0.1]], dtype=np.float32),
        retrieval_pnp_num_inliers_candidates=np.array([[42, 3]]),
        retrieval_pnp_reproj_median_candidates=np.array([[0.7, 2.0]], dtype=np.float32),
        retrieval_pnp_inlier_ratio_candidates=np.array([[0.8, 0.1]], dtype=np.float32),
        init_sources=np.array(["netvlad_rendered_pose"]),
    )

    records = load_pose_init_npz_records(path)

    assert len(records) == 1
    assert records[0].query_id == "seq8/frame00110.png"
    assert records[0].candidate_type == "real_retrieval"
    assert records[0].reference_image == "seq1/frame00001.png"
    assert records[0].prior_score == np.float32(0.9)
    assert records[0].metadata["retrieval_rank"] == 1
    assert records[0].metadata["pnp_inliers"] == 42


def test_reference_pose_bank_npz_adapter_emits_pose_labels(tmp_path):
    path = tmp_path / "reference_pose.npz"
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
    np.savez_compressed(
        path,
        sample_names=np.array(["seq8/frame00110.png"]),
        candidates=poses,
        trans_err_m=np.array([[0.1, 0.5]], dtype=np.float32),
        rot_err_deg=np.array([[2.0, 8.0]], dtype=np.float32),
        pose_cost_m=np.array([[0.2, 0.8]], dtype=np.float32),
        valid_mask=np.array([[True, True]]),
        reference_names=np.array([["seq1/frame00001.png", "seq1/frame00002.png"]]),
        scene=np.array("OldHospital"),
        candidate_source=np.array("hloc_netvlad_top10"),
        rot_cost_weight=np.array(0.01, dtype=np.float32),
    )

    records = load_reference_pose_bank_npz_records(path)

    assert len(records) == 2
    assert records[0].candidate_type == "reference_pose"
    assert records[0].pose_error.translation_m == np.float32(0.1)
    assert records[0].metadata["pose_cost_m"] == np.float32(0.2)
    assert records[0].metadata["scene"] == "OldHospital"


def test_score_table_jsonl_adapter_emits_labeled_candidates(tmp_path):
    path = tmp_path / "candidate_table.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_name": "seq8/frame00110.png",
                "candidate_idx": 3,
                "score": 0.9,
                "pose_cost_m": 0.2,
                "trans_err_m": 0.1,
                "rot_err_deg": 2.0,
                "in_basin": True,
                "is_oracle": False,
                "score_rank": 1,
                "delta_trans_m": 0.03,
                "retrieval_pnp_num_inliers_candidates": 12,
            }
        )
        + "\n"
    )

    records = load_candidate_records(path)

    assert records[0].query_id == "seq8/frame00110.png"
    assert records[0].candidate_type == "score_table_candidate"
    assert records[0].prior_score == 0.9
    assert records[0].pose_error.translation_m == 0.1
    assert records[0].metadata["in_basin"] is True
    assert records[0].metadata["identity_delta_m"] == 0.03
    assert records[0].metadata["pnp_inliers"] == 12
