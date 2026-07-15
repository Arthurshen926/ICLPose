import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _load_pose_conditioned_hard_mode_artifact,
    _load_pose_conditioned_hard_negative_artifact,
    _query_grouped_training_batches,
    _validate_complete_hard_modes_in_batches,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    pose_conditioned_hard_mode_margin_loss,
    pose_conditioned_hard_negative_margin_loss,
)


def test_pose_conditioned_margin_is_group_balanced_and_mode_weighted() -> None:
    logits = torch.tensor(
        [[2.0, 1.8, 0.0, -3.0], [3.0, 2.0, 1.0, 0.0]],
        requires_grad=True,
    )
    positives = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    hard = torch.tensor(
        [[False, True, True, False], [False, False, False, False]]
    )
    mode_counts = torch.tensor(
        [[0.0, 1.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )

    loss, metrics = pose_conditioned_hard_negative_margin_loss(
        logits,
        positives,
        hard,
        hard_negative_mode_counts=mode_counts,
        margin=0.5,
    )
    # Candidate one violates by 0.3 with weight 1; candidate two is satisfied
    # with weight sqrt(4)=2. The group-normalized loss is therefore 0.1.
    torch.testing.assert_close(loss.detach(), torch.tensor(0.1))
    assert metrics["pose_hard_negative_group_count"] == 1.0
    assert metrics["pose_hard_negative_candidate_count"] == 2.0
    assert metrics["pose_hard_negative_active_fraction"] == 0.5
    assert metrics["pose_hard_negative_margin_satisfied_fraction"] == 0.5
    assert metrics["pose_hard_group_top1_accuracy"] == 1.0

    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[1], torch.zeros(4))
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] > 0.0


def test_pose_conditioned_margin_ignores_empty_batches_and_rejects_bad_targets() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    positives = torch.tensor(
        [[True, False, False], [False, True, False]], dtype=torch.bool
    )
    empty = torch.zeros_like(positives)
    loss, metrics = pose_conditioned_hard_negative_margin_loss(
        logits, positives, empty
    )
    loss.backward()
    assert loss.item() == 0.0
    assert metrics["pose_hard_negative_group_count"] == 0.0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    with pytest.raises(ValueError, match="both positive and a hard negative"):
        pose_conditioned_hard_negative_margin_loss(
            logits.detach(), positives, positives
        )
    with pytest.raises(ValueError, match="requires a positive candidate"):
        pose_conditioned_hard_negative_margin_loss(
            logits.detach(),
            torch.zeros_like(positives),
            torch.tensor(
                [[False, True, False], [False, False, False]], dtype=torch.bool
            ),
        )


def test_pose_conditioned_mode_margin_scores_coherent_group_sets() -> None:
    logits = torch.tensor(
        [
            [1.0, 0.9, 0.0],
            [1.0, 1.2, 0.0],
            [1.0, 0.0, 0.8],
            [1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    positives = torch.zeros_like(logits, dtype=torch.bool)
    positives[:, 0] = True
    mode_ids = torch.tensor(
        [[10, -1], [10, -1], [10, 20], [-1, 20]], dtype=torch.long
    )
    mode_candidates = torch.zeros((4, 2, 3), dtype=torch.bool)
    mode_candidates[:3, 0, 1] = True
    mode_candidates[2:, 1, 2] = True

    loss, metrics = pose_conditioned_hard_mode_margin_loss(
        logits,
        positives,
        mode_ids,
        mode_candidates,
        margin=0.2,
        top_group_fraction=0.5,
        minimum_mode_groups=2,
    )

    # Mode 10 keeps its two strongest wrong-over-positive groups: +0.2 and
    # -0.1, hence a -0.05 positive gap and 0.25 violation. Mode 20 has a
    # +0.6 positive gap and no violation. Modes are weighted equally.
    torch.testing.assert_close(loss.detach(), torch.tensor(0.125))
    assert metrics["pose_hard_mode_count"] == 2.0
    assert metrics["pose_hard_mode_group_incidence_count"] == 5.0
    assert metrics["pose_hard_mode_active_fraction"] == 0.5
    assert metrics["pose_hard_mode_margin_satisfied_fraction"] == 0.5
    assert metrics["pose_hard_mode_mean_positive_gap"] == pytest.approx(0.275)
    assert metrics["pose_hard_mode_group_top1_accuracy"] == 0.75

    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 0] < 0.0
    assert logits.grad[1, 1] > 0.0
    torch.testing.assert_close(logits.grad[2:], torch.zeros_like(logits.grad[2:]))


def test_pose_conditioned_mode_margin_rejects_partial_mode_batches() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    positives = torch.tensor(
        [[True, False, False], [True, False, False]], dtype=torch.bool
    )
    mode_ids = torch.tensor([[7], [-1]], dtype=torch.long)
    mode_candidates = torch.zeros((2, 1, 3), dtype=torch.bool)
    mode_candidates[0, 0, 1] = True
    with pytest.raises(ValueError, match="minimum_mode_groups"):
        pose_conditioned_hard_mode_margin_loss(
            logits,
            positives,
            mode_ids,
            mode_candidates,
            minimum_mode_groups=2,
        )


def _artifact_fixture(tmp_path, *, stale: bool = False, leak: bool = False):
    selected_rows = np.asarray([10, 20, 30], dtype=np.int64)
    selected_columns = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    query_ids = np.asarray(["train/a.png", "val/b.png", "test/c.png"])
    valid = np.ones((3, 2), dtype=bool)
    positives = np.asarray(
        [[True, False], [True, False], [True, False]], dtype=bool
    )
    hard = np.zeros((3, 2), dtype=bool)
    hard[0, 1] = True
    candidate_counts = np.zeros((3, 2), dtype=np.uint16)
    candidate_counts[0, 1] = 2
    group_hard = np.asarray([True, False, False])
    group_counts = np.asarray([2, 0, 0], dtype=np.uint16)
    if leak:
        hard[1, 1] = True
        candidate_counts[1, 1] = 1
        group_hard[1] = True
        group_counts[1] = 1
    manifest = {
        "feature_artifact_sha256": "candidate-sha",
        "proposals_sha256": "proposal-sha",
        "projected_landmark_bank_sha256": "bank-sha",
        "query_split_manifest_sha256": "split-sha",
        "colmap_cameras_sha256": "cameras-sha",
        "colmap_images_sha256": "images-sha",
    }
    inputs = {
        "candidate_artifact_sha256": (
            "stale-candidate-sha" if stale else "candidate-sha"
        ),
        "proposals_sha256": "proposal-sha",
        "projected_landmark_bank_sha256": "bank-sha",
        "split_json_sha256": "split-sha",
        "colmap_cameras_sha256": "cameras-sha",
        "colmap_images_sha256": "images-sha",
        "score_artifact_sha256": "score-sha",
        "score_key": "candidate_probability",
    }
    metadata = {
        "format": "pose_conditioned_system_hard_negatives_v1",
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "ground_truth_joined_after_generation": True,
        "split_names": ["train"],
        "config": {},
        "inputs": inputs,
    }
    path = tmp_path / "hard_negatives.npz"
    np.savez_compressed(
        path,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        query_ids=query_ids,
        valid_edges=valid,
        positive_mask_TARGET_ONLY=positives,
        hard_negative_mask_TARGET_ONLY=hard,
        candidate_bad_mode_counts_TARGET_ONLY=candidate_counts,
        group_hard_mask_TARGET_ONLY=group_hard,
        group_bad_mode_counts_TARGET_ONLY=group_counts,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    split = {
        "train": ["train/a.png"],
        "validation": ["val/b.png"],
        "test": ["test/c.png"],
    }
    expected = {
        "selected_rows": selected_rows,
        "selected_columns": selected_columns,
        "query_ids": query_ids,
        "valid": valid,
        "positives": positives,
    }
    return path, manifest, split, expected


def _load_fixture(path, manifest, split, expected):
    return _load_pose_conditioned_hard_negative_artifact(
        path,
        expected_manifest=manifest,
        expected_selected_rows=expected["selected_rows"],
        expected_selected_columns=expected["selected_columns"],
        expected_query_ids=expected["query_ids"],
        expected_valid_edges=expected["valid"],
        expected_positive_mask=expected["positives"],
        split=split,
    )


def test_pose_conditioned_training_artifact_loads_with_strict_lineage(tmp_path) -> None:
    path, manifest, split, expected = _artifact_fixture(tmp_path)
    hard, mode_counts, audit = _load_fixture(path, manifest, split, expected)

    assert hard.sum() == 1
    assert mode_counts[0, 1] == 2
    assert audit["hard_group_count"] == 1
    assert audit["hard_candidate_count"] == 1
    assert audit["validation_or_test_training_leakage"] is False


def test_pose_conditioned_training_artifact_rejects_stale_lineage(tmp_path) -> None:
    path, manifest, split, expected = _artifact_fixture(tmp_path, stale=True)
    with pytest.raises(ValueError, match="stale or misaligned"):
        _load_fixture(path, manifest, split, expected)


def test_pose_conditioned_training_artifact_rejects_validation_target_leakage(
    tmp_path,
) -> None:
    path, manifest, split, expected = _artifact_fixture(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leaks validation/test targets"):
        _load_fixture(path, manifest, split, expected)


def _structured_artifact_fixture(tmp_path, *, cross_query: bool = False):
    path, manifest, split, expected = _artifact_fixture(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    metadata["format"] = "pose_conditioned_system_hard_modes_v2"
    metadata["config"] = {"min_consistent_groups": 1}
    mode_ids = np.asarray([[0], [-1], [-1]], dtype=np.int32)
    mode_candidates = np.zeros((3, 1, 2), dtype=bool)
    mode_candidates[0, 0, 1] = True
    payload["candidate_bad_mode_counts_TARGET_ONLY"] = np.sum(
        mode_candidates, axis=1, dtype=np.uint16
    )
    payload["group_bad_mode_counts_TARGET_ONLY"] = np.sum(
        mode_ids >= 0, axis=1, dtype=np.uint16
    )
    payload["group_hard_mask_TARGET_ONLY"] = np.any(mode_ids >= 0, axis=1)
    mode_query_ids = np.asarray(
        ["val/b.png" if cross_query else "train/a.png"]
    )
    structured_path = tmp_path / "hard_modes.npz"
    np.savez_compressed(
        structured_path,
        **payload,
        hard_mode_ids_TARGET_ONLY=mode_ids,
        hard_mode_candidate_mask_TARGET_ONLY=mode_candidates,
        hard_mode_query_ids_TARGET_ONLY=mode_query_ids,
        hard_mode_support_group_counts_TARGET_ONLY=np.asarray(
            [1], dtype=np.uint16
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return structured_path, manifest, split, expected


def test_pose_conditioned_structured_mode_artifact_loads_exact_membership(
    tmp_path,
) -> None:
    path, manifest, split, expected = _structured_artifact_fixture(tmp_path)
    mode_ids, mode_candidates, audit = _load_pose_conditioned_hard_mode_artifact(
        path,
        expected_manifest=manifest,
        expected_selected_rows=expected["selected_rows"],
        expected_selected_columns=expected["selected_columns"],
        expected_query_ids=expected["query_ids"],
        expected_valid_edges=expected["valid"],
        expected_positive_mask=expected["positives"],
        split=split,
    )

    assert mode_ids[0, 0] == 0
    assert mode_candidates[0, 0, 1]
    assert audit["hard_mode_count"] == 1
    assert audit["hard_mode_group_incidence_count"] == 1
    assert audit["validation_or_test_training_leakage"] is False


def test_pose_conditioned_structured_mode_artifact_rejects_cross_query_mode(
    tmp_path,
) -> None:
    path, manifest, split, expected = _structured_artifact_fixture(
        tmp_path, cross_query=True
    )
    with pytest.raises(ValueError, match="crosses query images"):
        _load_pose_conditioned_hard_mode_artifact(
            path,
            expected_manifest=manifest,
            expected_selected_rows=expected["selected_rows"],
            expected_selected_columns=expected["selected_columns"],
            expected_query_ids=expected["query_ids"],
            expected_valid_edges=expected["valid"],
            expected_positive_mask=expected["positives"],
            split=split,
        )


def test_query_grouped_training_batches_never_split_query_blocks() -> None:
    query_ids = np.asarray(["a", "a", "b", "b", "b", "c"])
    groups = np.arange(len(query_ids), dtype=np.int64)
    batches = _query_grouped_training_batches(
        groups,
        query_ids,
        groups_per_batch=3,
        rng=np.random.default_rng(9),
    )

    np.testing.assert_array_equal(
        np.sort(np.concatenate(batches)), groups
    )
    owner_batch = {}
    for batch_index, batch in enumerate(batches):
        assert len(batch) <= 3
        for owner in np.unique(query_ids[batch]):
            owner_batch.setdefault(str(owner), set()).add(batch_index)
    assert all(len(indices) == 1 for indices in owner_batch.values())


def test_complete_hard_mode_batch_validation_rejects_split_or_missing_modes() -> None:
    mode_ids = np.asarray(
        [[0, -1], [0, 1], [0, 1], [-1, 1], [-1, -1]], dtype=np.int64
    )
    audit = _validate_complete_hard_modes_in_batches(
        [np.asarray([0, 1, 2, 3, 4], dtype=np.int64)],
        mode_ids,
    )
    assert audit == {
        "hard_mode_count": 2,
        "hard_mode_group_incidence_count": 6,
    }

    with pytest.raises(ValueError, match="cross training batches"):
        _validate_complete_hard_modes_in_batches(
            [
                np.asarray([0, 1, 4], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
            ],
            mode_ids,
        )
    with pytest.raises(ValueError, match="omitted support groups"):
        _validate_complete_hard_modes_in_batches(
            [np.asarray([0, 1, 2, 4], dtype=np.int64)],
            mode_ids,
        )
