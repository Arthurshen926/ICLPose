import numpy as np

from feature_extract.vfm.patch_footprint_matching import (
    FootprintObservation,
    PatchToFootprintMatchingConfig,
    build_reference_token_footprint_bank,
    evaluate_footprint_matches,
    footprint_bank_stats,
    footprint_matches_to_pnp_matches,
    match_query_patches_to_footprints,
    query_footprint_positive_stats,
)
from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSet, PatchPositiveSets, TokenPatchBox
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _toy_landmarks() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11, 12], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0], [3.0, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        observation_counts=np.asarray([5, 4, 3], dtype=np.int64),
        observation_image_ids=(("ref_a.png",), ("ref_a.png",), ("ref_a.png",)),
        reprojection_errors=np.asarray([0.2, 0.3, 0.4], dtype=np.float32),
    )


def test_reference_token_footprint_bank_groups_landmarks_by_reference_patch() -> None:
    index = _toy_landmarks()
    observations = {
        "ref_a.png": [
            FootprintObservation("ref_a.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref_a.png", 11, np.asarray([12.0, 11.0]), 100, 100),
            FootprintObservation("ref_a.png", 12, np.asarray([90.0, 90.0]), 100, 100),
        ]
    }
    bank = build_reference_token_footprint_bank(index, observations, ["ref_a.png"], 2, 2)

    assert len(bank) == 2
    assert bank.units[0].track_ids == (10, 11)
    assert bank.units[1].track_ids == (12,)
    stats = footprint_bank_stats(bank)
    assert stats["footprint_count"] == 2
    assert stats["covered_track_count"] == 3
    assert stats["max_landmarks_per_footprint"] == 2

    expanded = build_reference_token_footprint_bank(
        index,
        observations,
        ["ref_a.png"],
        2,
        2,
        footprint_cell_radius=1,
    )
    assert len(expanded) == 4
    assert max(unit.num_landmarks for unit in expanded.units) == 3


def test_patch_to_footprint_matching_scores_local_landmark_sets() -> None:
    index = _toy_landmarks()
    observations = {
        "ref_a.png": [
            FootprintObservation("ref_a.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref_a.png", 11, np.asarray([12.0, 11.0]), 100, 100),
            FootprintObservation("ref_a.png", 12, np.asarray([90.0, 90.0]), 100, 100),
        ]
    }
    bank = build_reference_token_footprint_bank(index, observations, ["ref_a.png"], 2, 2)
    query = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)

    matches = match_query_patches_to_footprints(
        query,
        index,
        bank,
        PatchToFootprintMatchingConfig(
            footprint_top_k=1,
            footprint_score_mode="max",
            min_similarity=-1.0,
            max_matches=None,
        ),
        image_width=100,
        image_height=100,
    )

    assert len(matches) == 1
    assert set(matches[0].footprint_track_ids) == {10, 11}
    assert matches[0].selected_track_id == 10
    pnp_matches = footprint_matches_to_pnp_matches(matches)
    assert pnp_matches[0].track_id == 10
    assert pnp_matches[0].source == "patch_footprint_max"


def test_footprint_metrics_use_patch_positive_set_overlap() -> None:
    index = _toy_landmarks()
    observations = {
        "ref_a.png": [
            FootprintObservation("ref_a.png", 10, np.asarray([10.0, 10.0]), 100, 100),
            FootprintObservation("ref_a.png", 11, np.asarray([12.0, 11.0]), 100, 100),
            FootprintObservation("ref_a.png", 12, np.asarray([90.0, 90.0]), 100, 100),
        ]
    }
    bank = build_reference_token_footprint_bank(index, observations, ["ref_a.png"], 2, 2)
    positives = PatchPositiveSets(
        by_token={
            0: PatchPositiveSet(
                token_index=0,
                patch_box=TokenPatchBox(0, np.asarray([0.0, 0.0]), 0.0, 0.0, 10.0, 10.0),
                track_ids={10, 11},
            )
        },
        stride_x_px=10.0,
        stride_y_px=10.0,
        visible_track_ids={10, 11},
        projected_xy_by_track={},
    )
    query = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    matches = match_query_patches_to_footprints(
        query,
        index,
        bank,
        PatchToFootprintMatchingConfig(
            footprint_top_k=1,
            footprint_score_mode="mean",
            min_similarity=-1.0,
            max_matches=None,
        ),
        image_width=100,
        image_height=100,
    )

    metrics = evaluate_footprint_matches(matches, positives, top_k=1)
    assert metrics["footprint_at_1"] == 1.0
    assert metrics["strong_footprint_at_1"] == 1.0
    assert metrics["positive_landmark_recall_at_1"] == 1.0
    positive_stats = query_footprint_positive_stats(positives, bank)
    assert positive_stats["mean_positive_footprints_per_token"] == 1.0
