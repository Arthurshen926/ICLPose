import numpy as np

from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSet, PatchPositiveSets, TokenPatchBox
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.reference_patch_maplets import (
    ReferencePatchMapletConfig,
    ReferencePatchMapletMatch,
    apply_maplet_verifier_to_matches,
    build_reference_patch_maplet_bank,
    collect_maplet_verifier_training_examples,
    collect_support_selector_training_examples,
    compute_patch_context_feature_map,
    evaluate_reference_patch_maplet_matches,
    expand_reference_patch_maplet_matches_to_query_to_3d,
    fit_maplet_verifier_model,
    fit_support_selector_model,
    match_query_patches_to_reference_patch_maplets,
    project_feature_map_tokens,
    reference_patch_maplet_positive_stats,
)
from feature_extract.vfm.patch_footprint_matching import FootprintObservation


def _toy_index() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11, 12], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0], [2.0, 0.0, 4.0]], dtype=np.float64),
        features=np.ones((3, 2), dtype=np.float32),
        mean_variances=np.zeros((3,), dtype=np.float32),
        observation_counts=np.asarray([4, 4, 4], dtype=np.int64),
        observation_image_ids=(("ref.png",), ("ref.png",), ("ref.png",)),
        reprojection_errors=np.zeros((3,), dtype=np.float32),
    )


def _support_selection_index() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.0, 0.0], dtype=np.float32),
        observation_counts=np.asarray([9, 3], dtype=np.int64),
        observation_image_ids=(("ref.png",), ("ref.png",)),
        reprojection_errors=np.zeros((2,), dtype=np.float32),
    )


def test_reference_patch_maplet_bank_uses_reference_token_feature_and_landmark_set():
    feature_map = np.zeros((2, 2, 2), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    feature_map[:, 1, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 12, np.asarray([90.0, 90.0]), 100, 100),
        ]
    }

    bank = build_reference_patch_maplet_bank(
        _toy_index(),
        observations,
        {"ref.png": feature_map},
        ["ref.png"],
        min_landmarks_per_maplet=1,
    )

    assert len(bank) == 2
    assert bank.units[0].track_ids == (10, 11)
    assert np.allclose(bank.units[0].feature, np.asarray([1.0, 0.0], dtype=np.float32))
    assert bank.units[1].track_ids == (12,)
    assert np.allclose(bank.units[1].feature, np.asarray([0.0, 1.0], dtype=np.float32))


def test_query_patch_to_reference_patch_maplet_matching_uses_set_overlap_metrics():
    feature_map = np.zeros((2, 2, 2), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    feature_map[:, 1, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 12, np.asarray([90.0, 90.0]), 100, 100),
        ]
    }
    bank = build_reference_patch_maplet_bank(_toy_index(), observations, {"ref.png": feature_map}, ["ref.png"])
    query = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)

    matches = match_query_patches_to_reference_patch_maplets(
        query,
        bank,
        ReferencePatchMapletConfig(top_k=1, min_similarity=-1.0),
        image_width=100,
        image_height=100,
    )
    positives = PatchPositiveSets(
        by_token={
            0: PatchPositiveSet(
                token_index=0,
                patch_box=TokenPatchBox(0, np.asarray([0.0, 0.0]), 0.0, 0.0, 20.0, 20.0),
                track_ids={10, 11},
            )
        },
        stride_x_px=10.0,
        stride_y_px=10.0,
        visible_track_ids={10, 11},
        projected_xy_by_track={},
    )

    metrics = evaluate_reference_patch_maplet_matches(matches, positives, top_k=1)

    assert len(matches) == 1
    assert matches[0].reference_image_id == "ref.png"
    assert matches[0].support_track_ids == (10, 11)
    assert metrics["maplet_at_1"] == 1.0
    assert metrics["positive_landmark_recall_at_1"] == 1.0


def test_reference_patch_maplet_positive_stats_reports_oracle_coverage():
    feature_map = np.zeros((2, 2, 2), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
        ]
    }
    bank = build_reference_patch_maplet_bank(_toy_index(), observations, {"ref.png": feature_map}, ["ref.png"])
    positives = PatchPositiveSets(
        by_token={
            0: PatchPositiveSet(
                token_index=0,
                patch_box=TokenPatchBox(0, np.asarray([0.0, 0.0]), 0.0, 0.0, 20.0, 20.0),
                track_ids={10, 11},
            ),
            1: PatchPositiveSet(
                token_index=1,
                patch_box=TokenPatchBox(1, np.asarray([90.0, 90.0]), 80.0, 80.0, 99.0, 99.0),
                track_ids={12},
            ),
        },
        stride_x_px=10.0,
        stride_y_px=10.0,
        visible_track_ids={10, 11, 12},
        projected_xy_by_track={},
    )

    stats = reference_patch_maplet_positive_stats(positives, bank)

    assert stats["positive_token_count"] == 2
    assert stats["covered_positive_token_fraction"] == 0.5
    assert stats["mean_positive_maplets_per_token"] == 0.5


def test_project_feature_map_tokens_preserves_grid_and_normalizes_output():
    import torch

    class SliceProjector(torch.nn.Module):
        def forward(self, features):
            return features[:, :2]

    feature_map = np.zeros((4, 1, 2), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([3.0, 4.0, 9.0, 0.0], dtype=np.float32)
    feature_map[:, 0, 1] = np.asarray([0.0, 5.0, 1.0, 0.0], dtype=np.float32)

    projected = project_feature_map_tokens(feature_map, SliceProjector(), output_dim=2)

    assert projected.shape == (2, 1, 2)
    assert np.allclose(projected[:, 0, 0], np.asarray([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert np.allclose(projected[:, 0, 1], np.asarray([0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_project_feature_map_tokens_passes_active_group_mask_to_selector():
    import torch

    class MaskAwareProjector(torch.nn.Module):
        def forward(self, features, active_group_mask=None):
            if active_group_mask is not None:
                features = features * active_group_mask
            return features[:, :2]

    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 1.0], dtype=np.float32)

    projected = project_feature_map_tokens(
        feature_map,
        MaskAwareProjector(),
        output_dim=2,
        active_group_mask=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert np.allclose(projected[:, 0, 0], np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_compute_patch_context_feature_map_averages_local_tokens_and_normalizes():
    feature_map = np.ones((2, 3, 3), dtype=np.float32)
    feature_map[1, :, :] = 0.0
    feature_map[1, 1, 1] = 2.0

    contextual = compute_patch_context_feature_map(feature_map, context="3x3")

    expected = np.asarray([1.0, 2.0 / 9.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert contextual.shape == feature_map.shape
    assert np.allclose(contextual[:, 1, 1], expected, atol=1e-6)


def test_compute_patch_context_feature_map_1x1_normalizes_original_tokens():
    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([3.0, 4.0], dtype=np.float32)

    contextual = compute_patch_context_feature_map(feature_map, context="1x1")

    assert np.allclose(contextual[:, 0, 0], np.asarray([0.6, 0.8], dtype=np.float32), atol=1e-6)


def test_expand_reference_patch_maplet_matches_prefers_high_quality_support_landmarks():
    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
        ]
    }
    bank = build_reference_patch_maplet_bank(_toy_index(), observations, {"ref.png": feature_map}, ["ref.png"])
    matches = [
        ReferencePatchMapletMatch(
            token_index=0,
            xy=np.asarray([5.0, 5.0], dtype=np.float64),
            reference_image_id="ref.png",
            reference_token_index=0,
            unit_id=0,
            support_track_ids=(10, 11),
            similarity=0.7,
            rank=0,
        )
    ]

    expanded = expand_reference_patch_maplet_matches_to_query_to_3d(
        matches,
        bank,
        _toy_index(),
        support_per_maplet=1,
    )

    assert len(expanded) == 1
    assert expanded[0].track_id == 10
    assert expanded[0].source == "reference_patch_maplet"
    assert expanded[0].token_match_rank == 0


def test_expand_reference_patch_maplet_matches_can_select_feature_consistent_support_point():
    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
        ]
    }
    index = _support_selection_index()
    bank = build_reference_patch_maplet_bank(index, observations, {"ref.png": feature_map}, ["ref.png"])
    matches = [
        ReferencePatchMapletMatch(0, np.asarray([5.0, 5.0]), "ref.png", 0, 0, (10, 11), 0.7, 0)
    ]

    expanded = expand_reference_patch_maplet_matches_to_query_to_3d(
        matches,
        bank,
        index,
        support_per_maplet=1,
        support_recovery_mode="feature_consistent_point",
    )

    assert len(expanded) == 1
    assert expanded[0].track_id == 11


def test_expand_reference_patch_maplet_matches_can_recover_soft_xyz_measurement_with_sigma():
    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
        ]
    }
    index = _support_selection_index()
    bank = build_reference_patch_maplet_bank(index, observations, {"ref.png": feature_map}, ["ref.png"])
    matches = [
        ReferencePatchMapletMatch(0, np.asarray([5.0, 5.0]), "ref.png", 0, 0, (10, 11), 0.7, 0)
    ]

    expanded = expand_reference_patch_maplet_matches_to_query_to_3d(
        matches,
        bank,
        index,
        support_recovery_mode="soft_xyz",
        measurement_sigma_px=4.0,
    )

    assert len(expanded) == 1
    assert expanded[0].track_id < 0
    assert np.allclose(expanded[0].xyz, np.asarray([1.0, 0.0, 4.0], dtype=np.float64))
    assert expanded[0].measurement_sigma_px == 4.0


def test_learned_support_selector_can_select_labeled_positive_support_point():
    feature_map = np.zeros((2, 1, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 11, np.asarray([12.0, 10.0]), 100, 100),
        ]
    }
    index = _support_selection_index()
    bank = build_reference_patch_maplet_bank(index, observations, {"ref.png": feature_map}, ["ref.png"])
    matches = [
        ReferencePatchMapletMatch(0, np.asarray([5.0, 5.0]), "ref.png", 0, 0, (10, 11), 0.7, 0)
    ]
    positives = PatchPositiveSets(
        by_token={
            0: PatchPositiveSet(
                token_index=0,
                patch_box=TokenPatchBox(0, np.asarray([0.0, 0.0]), 0.0, 0.0, 20.0, 20.0),
                track_ids={11},
            )
        },
        stride_x_px=10.0,
        stride_y_px=10.0,
        visible_track_ids={10, 11},
        projected_xy_by_track={},
    )

    features, labels = collect_support_selector_training_examples(matches, bank, index, positives)
    model = fit_support_selector_model(features, labels, steps=200, lr=0.5)
    expanded = expand_reference_patch_maplet_matches_to_query_to_3d(
        matches,
        bank,
        index,
        support_per_maplet=1,
        support_recovery_mode="learned_point",
        support_selector_model=model,
    )

    assert labels.tolist() == [0.0, 1.0]
    assert expanded[0].track_id == 11


def test_maplet_verifier_can_rerank_labeled_positive_maplet():
    feature_map = np.zeros((2, 2, 1), dtype=np.float32)
    feature_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    feature_map[:, 1, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
    observations = {
        "ref.png": [
            FootprintObservation("ref.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref.png", 12, np.asarray([10.0, 90.0]), 100, 100),
        ]
    }
    bank = build_reference_patch_maplet_bank(_toy_index(), observations, {"ref.png": feature_map}, ["ref.png"])
    matches = [
        ReferencePatchMapletMatch(0, np.asarray([5.0, 5.0]), "ref.png", 0, 0, (10,), 0.9, 0),
        ReferencePatchMapletMatch(0, np.asarray([5.0, 5.0]), "ref.png", 1, 1, (12,), 0.4, 1),
    ]
    positives = PatchPositiveSets(
        by_token={
            0: PatchPositiveSet(
                token_index=0,
                patch_box=TokenPatchBox(0, np.asarray([0.0, 0.0]), 0.0, 0.0, 20.0, 20.0),
                track_ids={12},
            )
        },
        stride_x_px=10.0,
        stride_y_px=10.0,
        visible_track_ids={10, 12},
        projected_xy_by_track={},
    )

    features, labels = collect_maplet_verifier_training_examples(matches, bank, positives)
    model = fit_maplet_verifier_model(features, labels, steps=200, lr=0.5)
    reranked = apply_maplet_verifier_to_matches(matches, bank, model, score_weight=1.0)

    assert labels.tolist() == [0.0, 1.0]
    assert reranked[0].unit_id == 1
    assert reranked[0].rank == 0
