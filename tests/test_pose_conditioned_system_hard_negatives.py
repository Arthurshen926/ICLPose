from __future__ import annotations

import numpy as np
import pytest

import feature_extract.tools.vfm.mine_pose_conditioned_system_hard_negatives as hard_mining
from feature_extract.tools.vfm.mine_pose_conditioned_system_hard_negatives import (
    PoseConditionedHardNegativeConfig,
    mine_query_system_error_modes,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def _repeated_shift_fixture():
    camera = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=1000,
        height=800,
        params=(1000.0, 1000.0, 500.0, 400.0),
    )
    correct_xyz = np.asarray(
        [
            [-3.0, -2.0, 10.0],
            [-1.0, -2.0, 10.0],
            [1.0, -2.0, 10.0],
            [-3.0, 2.0, 10.0],
            [-1.0, 2.0, 10.0],
            [1.0, 2.0, 10.0],
        ],
        dtype=np.float64,
    )
    query_xy = np.stack(
        [
            1000.0 * correct_xyz[:, 0] / correct_xyz[:, 2] + 500.0,
            1000.0 * correct_xyz[:, 1] / correct_xyz[:, 2] + 400.0,
        ],
        axis=1,
    )
    wrong_xyz = correct_xyz.copy()
    wrong_xyz[:, 0] -= 1.0
    candidate_xyz = np.stack([correct_xyz, wrong_xyz], axis=1)
    valid = np.ones((len(query_xy), 2), dtype=bool)
    positive = np.zeros_like(valid)
    positive[:, 0] = True
    scores = np.tile(np.asarray([0.85, 0.90]), (len(query_xy), 1))
    gt_pose = np.eye(4, dtype=np.float64)
    bad_pose = np.eye(4, dtype=np.float64)
    bad_pose[0, 3] = 1.0
    return camera, query_xy, candidate_xyz, valid, positive, scores, gt_pose, bad_pose


def test_pose_conditioned_mining_recovers_coherent_repeated_shift() -> None:
    camera, xy, xyz, valid, positive, scores, gt_pose, bad_pose = (
        _repeated_shift_fixture()
    )
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=np.stack([gt_pose, bad_pose]),
        hypothesis_scores=np.asarray([-0.4, -0.2]),
        hypothesis_translation_errors_m=np.asarray([0.0, 1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0, 0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_translation_m=0.25,
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            min_consistent_groups=6,
            min_consistent_grid_cells=3,
        ),
    )

    expected = np.zeros_like(valid)
    expected[:, 1] = True
    np.testing.assert_array_equal(result["hard_negative_mask"], expected)
    np.testing.assert_array_equal(result["group_hard_mask"], np.ones(6, dtype=bool))
    assert len(result["selected_modes"]) == 1
    assert result["selected_modes"][0]["consistent_group_count"] == 6
    assert result["selected_mode_hard_masks"].shape == (1, 6, 2)
    np.testing.assert_array_equal(
        result["selected_mode_hard_masks"][0], expected
    )
    assert np.max(np.sum(result["hard_negative_mask"], axis=1)) == 1


def test_pose_conditioned_mining_requires_score_plausible_wrong_identity() -> None:
    camera, xy, xyz, valid, positive, scores, _, bad_pose = (
        _repeated_shift_fixture()
    )
    scores[:, 1] = 0.2
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=bad_pose[None],
        hypothesis_scores=np.asarray([-0.2]),
        hypothesis_translation_errors_m=np.asarray([1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            hard_score_log_margin=0.7,
            min_consistent_groups=6,
            min_consistent_grid_cells=3,
        ),
    )

    assert not np.any(result["hard_negative_mask"])
    assert result["selected_modes"] == []
    assert result["selected_mode_hard_masks"].shape == (0, 6, 2)


def test_pose_conditioned_mining_never_labels_no_positive_group() -> None:
    camera, xy, xyz, valid, positive, scores, _, bad_pose = (
        _repeated_shift_fixture()
    )
    positive[0] = False
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=bad_pose[None],
        hypothesis_scores=np.asarray([-0.2]),
        hypothesis_translation_errors_m=np.asarray([1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            min_consistent_groups=5,
            min_consistent_grid_cells=3,
        ),
    )

    assert not np.any(result["hard_negative_mask"][0])
    assert not result["group_hard_mask"][0]
    assert np.all(result["hard_negative_mask"][1:, 1])


def test_hard_mode_mining_rejects_mixed_pose_evidence_versions(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.touch()
    second.touch()
    metadata = {
        "format": "grouped_pose_hypotheses_inference_only_v1",
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_generation": False,
        "inputs": {"candidate_artifact_sha256": "same"},
        "grouped_config": {"latent_em_enabled": True},
    }

    def fake_load(path, _fields):
        version = "spatial_kernel_mixture_v10" if path == first else "spatial_kernel_mixture_v11"
        return (
            {
                "query_ids": np.asarray([path.stem]),
                "evaluation_labels": np.asarray(["frozen"]),
                "hypothesis_indices": np.asarray([0], dtype=np.int64),
            },
            {**metadata, "candidate_pose_evidence_version": version},
        )

    monkeypatch.setattr(hard_mining, "load_inference_artifact_fields", fake_load)
    with pytest.raises(ValueError, match="incompatible"):
        hard_mining._load_hypotheses((first, second))


def test_hard_mode_mining_requires_pose_evidence_version(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "hypotheses.npz"
    artifact.touch()

    def fake_load(_path, _fields):
        return (
            {
                "query_ids": np.asarray(["query.png"]),
                "evaluation_labels": np.asarray(["frozen"]),
                "hypothesis_indices": np.asarray([0], dtype=np.int64),
            },
            {
                "inputs": {"candidate_artifact_sha256": "same"},
                "grouped_config": {"latent_em_enabled": True},
            },
        )

    monkeypatch.setattr(hard_mining, "load_inference_artifact_fields", fake_load)
    with pytest.raises(ValueError, match="evidence version"):
        hard_mining._load_hypotheses((artifact,))


def test_hard_mode_mining_merges_variable_relation_edge_axes(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.touch()
    second.touch()
    metadata = {
        "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
        "inputs": {"candidate_artifact_sha256": "same"},
        "grouped_config": {"latent_em_enabled": True},
    }

    def fake_load(path, _fields):
        edge_count = 54 if path == first else 55
        row_offset = 0 if path == first else 1
        return (
            {
                "query_ids": np.asarray([path.stem]),
                "evaluation_labels": np.asarray(["frozen"]),
                "hypothesis_indices": np.asarray([row_offset], dtype=np.int64),
                "verification_relation_feature_edge_histograms": np.full(
                    (1, edge_count, 208), float(row_offset), dtype=np.float32
                ),
                "verification_relation_feature_edge_null_touching_masses": np.full(
                    (1, edge_count), float(row_offset), dtype=np.float32
                ),
            },
            metadata,
        )

    monkeypatch.setattr(hard_mining, "load_inference_artifact_fields", fake_load)
    merged, _compatibility, _manifests = hard_mining._load_hypotheses(
        (first, second)
    )

    histograms = merged["verification_relation_feature_edge_histograms"]
    null_masses = merged["verification_relation_feature_edge_null_touching_masses"]
    assert histograms.shape == (2, 55, 208)
    assert null_masses.shape == (2, 55)
    assert np.isnan(histograms[0, 54]).all()
    assert np.isnan(null_masses[0, 54])
    np.testing.assert_array_equal(histograms[1], np.ones((55, 208), dtype=np.float32))


def test_hard_mode_mining_loads_only_fields_required_for_mode_targets(
    monkeypatch, tmp_path
) -> None:
    artifact = tmp_path / "hypotheses.npz"
    artifact.touch()
    requested_fields: list[tuple[str, ...]] = []

    def fake_load(path, fields):
        assert path == artifact
        requested_fields.append(tuple(fields))
        return (
            {
                "query_ids": np.asarray(["query.png"]),
                "split_names": np.asarray(["train"]),
                "evaluation_labels": np.asarray(["frozen"]),
                "hypothesis_indices": np.asarray([0], dtype=np.int64),
                "poses_w2c": np.eye(4, dtype=np.float64)[None],
                "verification_log_likelihood_means": np.asarray([0.0]),
            },
            {
                "candidate_pose_evidence_version": "spatial_kernel_mixture_v10",
                "inputs": {"candidate_artifact_sha256": "same"},
                "grouped_config": {"latent_em_enabled": True},
            },
        )

    monkeypatch.setattr(
        hard_mining, "load_inference_artifact_fields", fake_load, raising=False
    )
    hard_mining._load_hypotheses((artifact,))

    assert requested_fields == [
        (
            "query_ids",
            "split_names",
            "evaluation_labels",
            "hypothesis_indices",
            "poses_w2c",
            "verification_log_likelihood_means",
        )
    ]


def test_hard_mode_mining_derives_positive_mask_from_posthoc_residuals() -> None:
    proposal_residuals = np.asarray(
        [[1.5, 3.0, np.nan], [0.5, 5.0, 2.0]], dtype=np.float64
    )
    selected_rows = np.asarray([0, 1], dtype=np.int64)
    selected_columns = np.asarray([[0, 1, -1], [1, 0, 2]], dtype=np.int64)
    valid_edges = np.asarray([[True, True, False], [True, True, True]])

    positive = hard_mining.positive_mask_from_posthoc_gt_residuals(
        proposal_residuals=proposal_residuals,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        valid_edges=valid_edges,
        positive_threshold_px=2.0,
    )

    np.testing.assert_array_equal(
        positive,
        np.asarray([[True, False, False], [False, True, True]]),
    )


def test_hard_mode_mining_restricts_all_candidate_rows_to_requested_queries() -> None:
    query_ids = np.asarray(
        ["train/a.png", "validation/b.png", "train/c.png", "test/d.png"]
    )
    selected_ids, fields = hard_mining.select_candidate_rows_for_allowed_queries(
        query_ids=query_ids,
        allowed_query_ids={"train/a.png", "train/c.png"},
        fields={
            "selected_rows": np.asarray([2, 3, 5, 7], dtype=np.int64),
            "valid_edges": np.asarray(
                [[True, False], [False, True], [True, True], [False, False]]
            ),
        },
    )

    np.testing.assert_array_equal(selected_ids, ["train/a.png", "train/c.png"])
    np.testing.assert_array_equal(fields["selected_rows"], [2, 5])
    np.testing.assert_array_equal(
        fields["valid_edges"], np.asarray([[True, False], [True, True]])
    )
