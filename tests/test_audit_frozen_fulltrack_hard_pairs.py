from __future__ import annotations

import json
import hashlib

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    HardPairSelection,
    coherent_shift_mask,
    hard_pair_raw_screen,
    pairwise_feature_metrics,
    select_rank2_to_top1_wrong_pairs,
    wilson_lower_bound,
    _array_sha256_short,
    _validate_projected_bank_lineage,
)
from feature_extract.vfm.artifacts import file_sha256_short


def test_rank2_pairs_keep_fixed_top_wrong_and_ignore_visual_scores() -> None:
    selection = select_rank2_to_top1_wrong_pairs(
        query_ids=np.asarray(["q0", "q0", "q1", "q2"]),
        candidate_track_ids=np.asarray(
            [[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]], dtype=np.int64
        ),
        candidate_probabilities=np.asarray(
            [[0.2, 0.6, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.4, 0.3, 0.2]],
            dtype=np.float32,
        ),
        correct_labels=np.asarray(
            [[True, False, False], [False, True, False], [True, False, False], [False, False, True]],
            dtype=bool,
        ),
        xyz_by_track={
            10: np.asarray([0.0, 0.0, 0.0]),
            20: np.asarray([1.0, 0.0, 0.0]),
            11: np.asarray([0.0, 1.0, 0.0]),
            21: np.asarray([1.0, 1.0, 0.0]),
            12: np.asarray([0.0, 2.0, 0.0]),
            22: np.asarray([1.0, 2.0, 0.0]),
            13: np.asarray([0.0, 3.0, 0.0]),
            23: np.asarray([1.0, 3.0, 0.0]),
            33: np.asarray([2.0, 3.0, 0.0]),
        },
    )
    assert selection.row_indices.tolist() == [0, 1, 3]
    assert selection.positive_columns.tolist() == [0, 1, 2]
    assert selection.negative_columns.tolist() == [1, 0, 0]
    assert selection.positive_ranks.tolist() == [2, 2, 3]
    np.testing.assert_allclose(
        selection.wrong_minus_correct_xyz,
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
    )


def test_coherent_shift_and_pair_metrics_keep_scopes_separate() -> None:
    selection = HardPairSelection(
        row_indices=np.asarray([0, 1, 2, 3]),
        positive_columns=np.asarray([1, 1, 1, 1]),
        negative_columns=np.asarray([0, 0, 0, 0]),
        positive_ranks=np.asarray([2, 2, 2, 2]),
        query_ids=np.asarray(["q0", "q0", "q0", "q1"]),
        wrong_minus_correct_xyz=np.asarray(
            [[0.5, 0.0, 0.0], [0.52, 0.01, 0.0], [0.48, -0.01, 0.0], [0.1, 0.8, 0.0]]
        ),
    )
    coherent, diagnostics = coherent_shift_mask(
        selection,
        minimum_pairs=3,
        minimum_shift_m=0.1,
        maximum_dispersion_m=0.05,
        maximum_relative_dispersion=0.35,
    )
    assert coherent.tolist() == [True, True, True, False]
    assert [item["query_id"] for item in diagnostics if item["coherent"]] == ["q0"]
    scores = np.asarray([[0.2, 0.7], [0.5, 0.5], [0.8, 0.6], [0.1, 0.9]], dtype=np.float32)
    metrics = pairwise_feature_metrics(
        scores=scores,
        feature_valid=np.ones_like(scores, dtype=bool),
        selection=selection,
        subset_mask=coherent,
    )
    assert metrics["pair_count"] == 3
    assert metrics["win_count"] == 1
    assert metrics["tie_count"] == 1
    assert metrics["loss_count"] == 1
    assert metrics["win_rate"] == 1.0 / 3.0
    assert metrics["median_correct_minus_wrong"] == 0.0


def test_hard_pair_raw_screen_requires_confident_coverage() -> None:
    assert wilson_lower_bound(70, 100) is not None
    passed = hard_pair_raw_screen(
        {
            "usable_pair_count": 120,
            "usable_pair_rate": 0.8,
            "win_rate_wilson95_lower": 0.55,
            "median_correct_minus_wrong": 0.01,
        },
        minimum_usable_pairs=100,
        minimum_usable_pair_rate=0.5,
    )
    assert passed["passed"] is True
    rejected = hard_pair_raw_screen(
        {
            "usable_pair_count": 80,
            "usable_pair_rate": 0.9,
            "win_rate_wilson95_lower": 0.7,
            "median_correct_minus_wrong": 0.01,
        },
        minimum_usable_pairs=100,
        minimum_usable_pair_rate=0.5,
    )
    assert rejected["passed"] is False


def test_global_context_lineage_must_match_its_raw_per_view_bridge(tmp_path) -> None:
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"projected-bank")
    s0 = tmp_path / "s0.npz"
    np.savez_compressed(
        s0,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "inputs": {
                        "projected_landmark_bank": {
                            "sha256": file_sha256_short(bank)
                        }
                    }
                }
            )
        ),
    )
    raw = tmp_path / "raw.npz"
    np.savez_compressed(
        raw,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "frozen_fulltrack_candidate_per_view_appearance_v1",
                    "per_view_edge_feature_semantics": "raw_aligned_ncc_per_real_sfm_observation_v1",
                    "source_frozen_appearance_artifact": str(s0),
                    "source_frozen_appearance_artifact_sha256": file_sha256_short(s0),
                }
            )
        ),
    )
    child = {
        "source_frozen_appearance_artifact": str(s0),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(s0),
        "source_fulltrack_per_view_artifact": str(raw),
        "source_fulltrack_per_view_artifact_sha256": file_sha256_short(raw),
    }
    _validate_projected_bank_lineage((child,), projected_landmark_bank=bank)
    child["source_frozen_appearance_artifact_sha256"] = "wrong-root"
    with np.testing.assert_raises_regex(ValueError, "S0 root differs"):
        _validate_projected_bank_lineage((child,), projected_landmark_bank=bank)


def test_raw_csr_child_lineage_binds_declared_array_hashes(tmp_path) -> None:
    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"projected-bank")
    s0 = tmp_path / "s0.npz"
    np.savez_compressed(
        s0,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "inputs": {
                        "projected_landmark_bank": {
                            "sha256": file_sha256_short(bank)
                        }
                    }
                }
            )
        ),
    )
    arrays = {
        "edge_candidate_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "edge_geometry_rows": np.asarray([4, 5, 6], dtype=np.int64),
        "candidate_track_ids": np.asarray([[10, 11]], dtype=np.int64),
        "candidate_probabilities": np.asarray([[0.4, 0.5]], dtype=np.float32),
        "null_probabilities": np.asarray([0.1], dtype=np.float32),
        "verification_source_row_indices": np.asarray([9], dtype=np.int64),
    }
    raw = tmp_path / "raw.npz"
    np.savez_compressed(
        raw,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "frozen_fulltrack_candidate_per_view_appearance_v1",
                    "per_view_edge_feature_semantics": "raw_aligned_ncc_per_real_sfm_observation_v1",
                    "source_frozen_appearance_artifact": str(s0),
                    "source_frozen_appearance_artifact_sha256": file_sha256_short(s0),
                }
            )
        ),
    )
    child = {
        "source_frozen_appearance_artifact": str(s0),
        "source_frozen_appearance_artifact_sha256": file_sha256_short(s0),
        "source_fulltrack_per_view_artifact": str(raw),
        "source_fulltrack_per_view_artifact_sha256": file_sha256_short(raw),
        "source_fulltrack_edge_contract": "raw_fulltrack_per_view_csr_v1",
        "source_edge_feature_semantics": "raw_aligned_ncc_per_real_sfm_observation_v1",
        **{
            f"source_{name}_sha256": _array_sha256_short(value)
            for name, value in arrays.items()
        },
    }
    _validate_projected_bank_lineage((child,), projected_landmark_bank=bank)
    child["source_csr_array_hash_scheme"] = "dtype_shape_bytes_sha256_v1"
    for name, value in arrays.items():
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        child[f"source_{name}_sha256"] = digest.hexdigest()[:16]
    _validate_projected_bank_lineage((child,), projected_landmark_bank=bank)
    child["source_edge_geometry_rows_sha256"] = "stale"
    with np.testing.assert_raises_regex(ValueError, "edge_geometry_rows"):
        _validate_projected_bank_lineage((child,), projected_landmark_bank=bank)
