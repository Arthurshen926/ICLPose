import json

import numpy as np
import pytest

from feature_extract.tools.vfm.eval_pose_hypothesis_verification import (
    _load_candidate_posterior_ensemble,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _write_overlay(
    path,
    *,
    candidate_path,
    proposals_path,
    bank_path,
    split_path,
    candidate_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.parent / "best.pt"
    checkpoint.write_bytes(f"checkpoint:{candidate_value}".encode("ascii"))
    image_ids = np.asarray(["a", "b", "c", "d"])
    split_indices = {
        "train": np.asarray([0, 1]),
        "validation": np.asarray([2]),
        "test": np.asarray([3]),
    }
    full = np.full((4, 2, 4), candidate_value, dtype=np.float32)
    full[:, :, -1] = 1.0 - 3.0 * candidate_value
    metadata = {
        "format": "whole_image_latent_pose_relation_candidate_graph_v3_posterior",
        "checkpoint_sha256": file_sha256_short(checkpoint),
        "descriptor_space_id": "space",
        "query_count": 2,
        "candidate_count": 3,
        "input_hashes": {
            "candidate_artifact_sha256": file_sha256_short(candidate_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json_sha256": file_sha256_short(split_path),
            "score_artifact_sha256": "source",
        },
    }
    np.savez_compressed(
        path,
        image_ids=image_ids,
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
        **{
            f"{name}_indices": indices
            for name, indices in split_indices.items()
        },
        **{
            f"{name}_posterior": full[indices]
            for name, indices in split_indices.items()
        },
    )


def test_candidate_posterior_overlay_averages_aligned_probability_mass(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.npz"
    proposals_path = tmp_path / "proposals.npz"
    bank_path = tmp_path / "bank.npz"
    split_path = tmp_path / "split.json"
    for path in (candidate_path, proposals_path, bank_path):
        path.write_bytes(path.name.encode("ascii"))
    split = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    split_path.write_text(json.dumps(split))
    first = tmp_path / "first" / "posterior.npz"
    second = tmp_path / "second" / "posterior.npz"
    _write_overlay(
        first,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        split_path=split_path,
        candidate_value=0.1,
    )
    _write_overlay(
        second,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        split_path=split_path,
        candidate_value=0.2,
    )

    candidate, null, manifests = _load_candidate_posterior_ensemble(
        (first, second),
        expected_image_ids=np.asarray(["a", "b", "c", "d"]),
        expected_split=split,
        query_count=2,
        candidate_count=3,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        split_path=split_path,
        descriptor_space_id="space",
    )

    assert candidate.shape == (8, 3)
    assert null.shape == (8, 3)
    np.testing.assert_allclose(candidate, 0.15)
    np.testing.assert_allclose(null[:, 0], 0.55)
    np.testing.assert_allclose(candidate.sum(1) + null[:, 0], 1.0)
    assert len(manifests) == 2


def test_candidate_posterior_overlay_rejects_stale_inputs(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.npz"
    proposals_path = tmp_path / "proposals.npz"
    bank_path = tmp_path / "bank.npz"
    split_path = tmp_path / "split.json"
    for path in (candidate_path, proposals_path, bank_path):
        path.write_bytes(path.name.encode("ascii"))
    split = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    split_path.write_text(json.dumps(split))
    overlay = tmp_path / "overlay" / "posterior.npz"
    _write_overlay(
        overlay,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        split_path=split_path,
        candidate_value=0.1,
    )
    candidate_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="stale or misaligned"):
        _load_candidate_posterior_ensemble(
            (overlay,),
            expected_image_ids=np.asarray(["a", "b", "c", "d"]),
            expected_split=split,
            query_count=2,
            candidate_count=3,
            candidate_path=candidate_path,
            proposals_path=proposals_path,
            bank_path=bank_path,
            split_path=split_path,
            descriptor_space_id="space",
        )


def _write_maplet_overlay(
    path,
    *,
    proposals_path,
    bank_path,
    maplet_hash: str,
) -> None:
    with np.load(proposals_path, allow_pickle=False) as payload:
        tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
    candidate = np.tile(
        np.asarray([0.2, 0.3, 0.1], dtype=np.float32), (len(tracks), 1)
    )
    null = np.full((len(tracks),), 0.4, dtype=np.float32)
    metadata = {
        "format": "candidate_maplet_prior_overlay_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "probability_semantics": (
            "candidate_identity_probability_plus_explicit_null_equals_one"
        ),
        "proposals_sha256": file_sha256_short(proposals_path),
        "checkpoint_sha256": ["first", "second"],
        "inference_scores_sha256": "scores",
        "inference_data_manifest": {
            "proposals_sha256": file_sha256_short(proposals_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "maplet_support_index_sha256": maplet_hash,
        },
        "inference_query_set": {
            "query_point_count": int(len(tracks)),
            "candidate_top_k": int(tracks.shape[1]),
        },
    }
    np.savez_compressed(
        path,
        candidate_track_ids=tracks,
        candidate_probabilities=candidate,
        null_probabilities=null,
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
    )


def test_candidate_maplet_overlay_compacts_pose_rows_and_preserves_omitted_mass(
    tmp_path,
) -> None:
    proposals_path = tmp_path / "proposals.npz"
    candidate_path = tmp_path / "candidate.npz"
    bank_path = tmp_path / "bank.npz"
    split_path = tmp_path / "split.json"
    maplet_hash = "maplet-hash"
    tracks = np.arange(18, dtype=np.int64).reshape(6, 3) + 10
    np.savez_compressed(
        proposals_path,
        candidate_track_ids=tracks,
        query_ids=np.asarray(["a", "a", "a", "b", "b", "b"]),
    )
    bank_path.write_bytes(b"bank")
    candidate_metadata = {
        "maplet_support_index_sha256": maplet_hash,
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "query_points_per_image": 2,
    }
    np.savez_compressed(
        candidate_path,
        selected_rows=np.asarray([0, 2, 3, 5], dtype=np.int64),
        selected_columns=np.asarray(
            [[0, 2], [2, 0], [0, 2], [2, 0]], dtype=np.int64
        ),
        valid_edges=np.ones((4, 2), dtype=bool),
        metadata_json=np.asarray(json.dumps(candidate_metadata), dtype=np.str_),
    )
    split = {"train": ["a"], "validation": ["b"], "test": []}
    split_path.write_text(json.dumps(split))
    overlay = tmp_path / "maplet_overlay.npz"
    _write_maplet_overlay(
        overlay,
        proposals_path=proposals_path,
        bank_path=bank_path,
        maplet_hash=maplet_hash,
    )

    candidate, null, manifests = _load_candidate_posterior_ensemble(
        (overlay,),
        expected_image_ids=np.asarray(["a", "b"]),
        expected_split=split,
        query_count=2,
        candidate_count=2,
        candidate_path=candidate_path,
        proposals_path=proposals_path,
        bank_path=bank_path,
        split_path=split_path,
        descriptor_space_id="unused-for-maplet",
    )

    np.testing.assert_allclose(
        candidate,
        np.asarray(
            [[0.2, 0.1], [0.1, 0.2], [0.2, 0.1], [0.1, 0.2]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(null[:, 0], 0.7)
    np.testing.assert_allclose(candidate.sum(axis=1) + null[:, 0], 1.0)
    assert manifests[0]["omitted_candidate_mass_transferred_to_null"] is True


def test_candidate_maplet_overlay_rejects_stale_bank_lineage(tmp_path) -> None:
    proposals_path = tmp_path / "proposals.npz"
    candidate_path = tmp_path / "candidate.npz"
    bank_path = tmp_path / "bank.npz"
    split_path = tmp_path / "split.json"
    np.savez_compressed(
        proposals_path,
        candidate_track_ids=np.asarray([[10, 11]], dtype=np.int64),
        query_ids=np.asarray(["a"]),
    )
    bank_path.write_bytes(b"bank")
    np.savez_compressed(
        candidate_path,
        selected_rows=np.asarray([0], dtype=np.int64),
        selected_columns=np.asarray([[0, 1]], dtype=np.int64),
        valid_edges=np.ones((1, 2), dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "maplet_support_index_sha256": "maplet",
                    "proposals_sha256": file_sha256_short(proposals_path),
                    "projected_landmark_bank_sha256": file_sha256_short(bank_path),
                }
            ),
            dtype=np.str_,
        ),
    )
    split_path.write_text(json.dumps({"train": ["a"], "validation": [], "test": []}))
    overlay = tmp_path / "maplet_overlay.npz"
    tracks = np.asarray([[10, 11]], dtype=np.int64)
    np.savez_compressed(
        overlay,
        candidate_track_ids=tracks,
        candidate_probabilities=np.asarray([[0.4, 0.4]], dtype=np.float32),
        null_probabilities=np.asarray([0.2], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_prior_overlay_v1",
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "probability_semantics": "candidate_identity_probability_plus_explicit_null_equals_one",
                    "proposals_sha256": file_sha256_short(proposals_path),
                    "inference_data_manifest": {
                        "proposals_sha256": file_sha256_short(proposals_path),
                        "projected_landmark_bank_sha256": "stale",
                        "maplet_support_index_sha256": "maplet",
                    },
                    "inference_query_set": {
                        "query_point_count": 1,
                        "candidate_top_k": 2,
                    },
                }
            ),
            dtype=np.str_,
        ),
    )

    with pytest.raises(ValueError, match="stale or misaligned"):
        _load_candidate_posterior_ensemble(
            (overlay,),
            expected_image_ids=np.asarray(["a"]),
            expected_split={"train": ["a"], "validation": [], "test": []},
            query_count=1,
            candidate_count=2,
            candidate_path=candidate_path,
            proposals_path=proposals_path,
            bank_path=bank_path,
            split_path=split_path,
            descriptor_space_id=None,
        )
