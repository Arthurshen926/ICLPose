from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.matcha_coarse_to_fine import (
    apply_cell_reliability_prior_to_matches,
    apply_fine_logit_confidence_to_matches,
    apply_keypoint_cell_prior_to_matches,
    apply_pair_fine_logits_to_matches,
    cross_attention_enhance_feature_maps,
    expand_matches_with_render_local_offsets,
    apply_offset_logits_to_matches,
    deduplicate_repeated_correspondences,
    feature_map_to_coarse_grid,
    matcha_coarse_dual_softmax_matches,
    matcha_coarse_topk_matches,
    matcha_coarse_to_fine_keypoint_matches,
    refine_render_matches_by_local_attention,
    refine_matches_by_bilateral_local_correlation,
    retain_topk_matches_per_query,
    rescore_keypoint_matches_by_feature_similarity,
)
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch, keypoint_feature_matches_to_pnp_matches


def test_feature_map_to_coarse_grid_uses_patch_centers() -> None:
    fmap = np.zeros((3, 2, 4), dtype=np.float32)

    grid = feature_map_to_coarse_grid(fmap, image_width=400, image_height=200)

    assert grid.xy.shape == (8, 2)
    assert np.allclose(grid.xy[0], [50.0, 50.0])
    assert np.allclose(grid.xy[-1], [350.0, 150.0])
    assert grid.descriptors.shape == (8, 3)


def test_matcha_coarse_dual_softmax_matches_finds_mutual_patch_matches() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    render = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    render[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 1] = [0.0, 1.0]
    render[:, 0, 2] = [0.7, 0.7]

    matches = matcha_coarse_dual_softmax_matches(
        query,
        render,
        query_image_width=20,
        query_image_height=10,
        render_image_width=30,
        render_image_height=10,
        logit_scale=12.0,
        min_confidence=0.0,
        mutual=True,
    )

    assert sorted((match.query_index, match.render_index) for match in matches) == [(0, 0), (1, 1)]
    assert all(match.dual_softmax_confidence is not None for match in matches)


def test_matcha_coarse_dual_softmax_respects_candidate_cell_indices() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    render = np.zeros((2, 1, 2), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    render[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 1] = [0.0, 1.0]

    matches = matcha_coarse_dual_softmax_matches(
        query,
        render,
        query_image_width=20,
        query_image_height=10,
        render_image_width=20,
        render_image_height=10,
        logit_scale=12.0,
        query_candidate_indices=np.asarray([1], dtype=np.int64),
        render_candidate_indices=np.asarray([1], dtype=np.int64),
        mutual=True,
    )

    assert [(match.query_index, match.render_index) for match in matches] == [(1, 1)]


def test_matcha_coarse_topk_matches_keeps_non_mutual_candidates_with_rank_metadata() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    render = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 0] = [0.8, 0.2]
    render[:, 0, 1] = [1.0, 0.0]
    render[:, 0, 2] = [0.7, 0.3]

    matches = matcha_coarse_topk_matches(
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=30,
        render_image_height=10,
        k_per_query=2,
        mutual_mode="annotate",
        logit_scale=10.0,
    )

    assert len(matches) == 2
    assert [match.coarse_rank for match in matches] == [0, 1]
    assert matches[0].render_index == 1
    assert matches[0].base_render_index == 1
    assert matches[0].candidate_render_index == 1
    assert matches[0].candidate_id == 0
    assert matches[0].coarse_score > matches[1].coarse_score
    assert matches[0].coarse_score_gap == 0.0
    assert matches[1].coarse_score_gap > 0.0


def test_matcha_coarse_to_fine_can_use_topk_coarse_candidates_without_mutual_filter() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    render = np.zeros((2, 1, 3), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    render[:, 0, 0] = [0.8, 0.2]
    render[:, 0, 1] = [1.0, 0.0]
    render[:, 0, 2] = [0.7, 0.3]

    matches = matcha_coarse_to_fine_keypoint_matches(
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=30,
        render_image_height=10,
        logit_scale=10.0,
        fine_search_radius_px=0.0,
        coarse_top_k_per_query=2,
        coarse_mutual_mode="annotate",
    )

    assert len(matches) == 2
    assert [match.coarse_rank for match in matches] == [0, 1]
    assert [match.render_index for match in matches] == [1, 0]
    assert all(match.mutual_rank is not None for match in matches)


def test_matcha_coarse_to_fine_moves_render_measurement_to_local_peak() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = [0.0, 1.0]
    render = np.zeros((2, 10, 10), dtype=np.float32)
    render[:, 5, 5] = [0.6, 0.1]
    render[:, 5, 7] = [0.0, 1.0]

    matches = matcha_coarse_to_fine_keypoint_matches(
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=10,
        render_image_height=10,
        logit_scale=10.0,
        min_confidence=0.0,
        min_similarity=-1.0,
        fine_search_radius_px=3.0,
        fine_search_step_px=1.0,
        mutual=True,
    )

    assert len(matches) == 1
    assert np.allclose(matches[0].render_xy, [7.5, 5.5], atol=1e-6)
    assert matches[0].similarity > 0.99


def test_matcha_coarse_to_fine_softargmax_outputs_continuous_peak() -> None:
    query = np.zeros((2, 1, 1), dtype=np.float32)
    query[:, 0, 0] = [0.0, 1.0]
    render = np.zeros((2, 10, 10), dtype=np.float32)
    render[:, 5, 7] = [0.0, 1.0]

    matches = matcha_coarse_to_fine_keypoint_matches(
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=10,
        render_image_height=10,
        logit_scale=10.0,
        min_confidence=0.0,
        min_similarity=-1.0,
        fine_search_radius_px=3.0,
        fine_search_step_px=1.0,
        fine_mode="softargmax",
        fine_softmax_temperature=50.0,
        mutual=True,
    )

    assert len(matches) == 1
    assert np.allclose(matches[0].render_xy, [7.0, 5.0], atol=1e-3)


def test_pair_fine_render_offset_updates_depth_sample_used_for_pnp_3d() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([10.0, 10.0], dtype=np.float64),
        render_xy=np.asarray([4.0, 4.0], dtype=np.float64),
        similarity=0.9,
        ratio=1.0,
    )
    logits = np.full((1, 64), -20.0, dtype=np.float32)
    logits[0, 63] = 20.0
    refined = apply_pair_fine_logits_to_matches(
        [match],
        logits,
        render_image_width=16,
        render_image_height=16,
        render_grid_width=2,
        render_grid_height=2,
    )
    depth = np.tile(np.arange(16, dtype=np.float32)[None, :], (16, 1))
    camera = ColmapCamera(camera_id=1, model_id=1, width=16, height=16, params=(10.0, 10.0, 8.0, 8.0))

    pnp_matches = keypoint_feature_matches_to_pnp_matches(
        refined,
        depth,
        camera,
        np.eye(4, dtype=np.float64),
        image_width=16,
        image_height=16,
        render_grid_width=2,
        render_grid_height=2,
    )

    assert np.allclose(refined[0].render_xy, [7.5, 7.5])
    assert len(pnp_matches) == 1
    assert np.isclose(pnp_matches[0].xyz[2], 7.5)


def test_bilateral_local_correlation_moves_query_and_render_measurements() -> None:
    query = np.zeros((2, 10, 10), dtype=np.float32)
    render = np.zeros((2, 10, 10), dtype=np.float32)
    query[:, 3, 3] = [1.0, 0.0]
    render[:, 5, 7] = [1.0, 0.0]
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([5.0, 5.0]),
        render_xy=np.asarray([5.0, 5.0]),
        similarity=0.0,
        ratio=1.0,
    )

    refined = refine_matches_by_bilateral_local_correlation(
        [match],
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=10,
        render_image_height=10,
        search_radius_px=3.0,
        step_px=1.0,
        mode="argmax",
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].query_xy, [3.0, 3.0], atol=1e-6)
    assert np.allclose(refined[0].render_xy, [7.0, 5.0], atol=1e-6)
    assert refined[0].similarity > 0.99


def test_fine_attention_uses_query_context_without_moving_query_measurement() -> None:
    query = np.zeros((2, 10, 10), dtype=np.float32)
    render = np.zeros((2, 10, 10), dtype=np.float32)
    query[:, 5, 5] = [1.0, 0.0]
    query[:, 5, 6] = [0.0, 1.0]
    render[:, 5, 5] = [0.6, 0.1]
    render[:, 5, 7] = [0.0, 1.0]
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([5.0, 5.0]),
        render_xy=np.asarray([5.0, 5.0]),
        similarity=0.0,
        ratio=1.0,
    )

    refined = refine_render_matches_by_local_attention(
        [match],
        query,
        render,
        query_image_width=10,
        query_image_height=10,
        render_image_width=10,
        render_image_height=10,
        search_radius_px=3.0,
        step_px=1.0,
        mode="argmax",
        query_spatial_sigma_px=3.0,
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].query_xy, [5.0, 5.0], atol=1e-6)
    assert np.allclose(refined[0].render_xy, [7.0, 5.0], atol=1e-6)
    assert refined[0].similarity > 0.99


def test_cross_attention_enhance_feature_maps_preserves_shape_and_normalizes() -> None:
    query = np.zeros((3, 1, 2), dtype=np.float32)
    render = np.zeros((3, 1, 2), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0, 0.0]
    render[:, 0, 0] = [1.0, 0.0, 0.0]
    render[:, 0, 1] = [0.0, 1.0, 0.0]

    fused_query, fused_render = cross_attention_enhance_feature_maps(
        query,
        render,
        alpha=0.25,
        logit_scale=8.0,
    )

    assert fused_query.shape == query.shape
    assert fused_render.shape == render.shape
    assert np.allclose(np.linalg.norm(fused_query.reshape(3, -1), axis=0), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(fused_render.reshape(3, -1), axis=0), 1.0, atol=1e-5)


def test_matcha_cross_argmax_mode_runs_coarse_to_fine_matching() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    render = np.zeros((2, 4, 6), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    render[:, 1, 1] = [1.0, 0.0]
    render[:, 2, 4] = [0.0, 1.0]

    matches = matcha_coarse_to_fine_keypoint_matches(
        query,
        render,
        query_image_width=20,
        query_image_height=10,
        render_image_width=60,
        render_image_height=40,
        logit_scale=10.0,
        min_confidence=0.0,
        min_similarity=-1.0,
        fine_search_radius_px=3.0,
        fine_search_step_px=1.0,
        fine_mode="cross_argmax",
        mutual=True,
    )

    assert len(matches) >= 1


def test_expand_matches_with_render_local_offsets_keeps_query_fixed_and_moves_render() -> None:
    match = KeypointFeatureMatch(
        query_index=5,
        render_index=5,
        query_xy=np.asarray([20.0, 20.0]),
        render_xy=np.asarray([24.0, 24.0]),
        similarity=0.7,
        ratio=0.0,
        dual_softmax_confidence=0.8,
    )

    expanded = expand_matches_with_render_local_offsets(
        [match],
        render_image_width=64,
        render_image_height=64,
        render_grid_width=4,
        render_grid_height=4,
        cell_radius=1,
        max_candidates_per_match=9,
    )

    assert len(expanded) == 9
    assert all(np.allclose(item.query_xy, match.query_xy) for item in expanded)
    assert any(not np.allclose(item.render_xy, match.render_xy) for item in expanded)
    assert {item.render_index for item in expanded} == {0, 1, 2, 4, 5, 6, 8, 9, 10}
    assert all(item.base_render_index == 5 for item in expanded)
    assert all(item.candidate_render_index == item.render_index for item in expanded)
    center = [item for item in expanded if item.render_index == 5][0]
    right = [item for item in expanded if item.render_index == 6][0]
    assert center.cell_delta_x == 0
    assert center.cell_delta_y == 0
    assert right.cell_delta_x == 1
    assert right.cell_delta_y == 0
    assert len({item.candidate_id for item in expanded}) == len(expanded)


def test_rescore_keypoint_matches_by_feature_similarity_ranks_moved_render_candidate() -> None:
    query = np.zeros((2, 4, 4), dtype=np.float32)
    render = np.zeros((2, 4, 4), dtype=np.float32)
    query[:, 1, 1] = [1.0, 0.0]
    render[:, 1, 1] = [0.0, 1.0]
    render[:, 1, 2] = [1.0, 0.0]
    low = KeypointFeatureMatch(
        query_index=5,
        render_index=5,
        query_xy=np.asarray([24.0, 24.0]),
        render_xy=np.asarray([24.0, 24.0]),
        similarity=0.0,
        ratio=0.0,
        dual_softmax_confidence=0.8,
    )
    high = KeypointFeatureMatch(
        query_index=5,
        render_index=6,
        query_xy=np.asarray([24.0, 24.0]),
        render_xy=np.asarray([40.0, 24.0]),
        similarity=0.0,
        ratio=0.0,
        dual_softmax_confidence=0.8,
    )

    rescored = rescore_keypoint_matches_by_feature_similarity(
        [low, high],
        query,
        render,
        query_image_width=64,
        query_image_height=64,
        render_image_width=64,
        render_image_height=64,
    )

    assert [item.render_index for item in rescored] == [6, 5]
    assert rescored[0].similarity > 0.99
    assert rescored[0].dual_softmax_confidence > rescored[1].dual_softmax_confidence


def test_retain_topk_matches_per_query_limits_expanded_candidates() -> None:
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([0.0, 0.0]),
            render_xy=np.asarray([0.0, 0.0]),
            similarity=0.1,
            ratio=1.0,
            dual_softmax_confidence=0.1,
        ),
        KeypointFeatureMatch(
            query_index=0,
            render_index=1,
            query_xy=np.asarray([0.0, 0.0]),
            render_xy=np.asarray([1.0, 0.0]),
            similarity=0.9,
            ratio=1.0,
            dual_softmax_confidence=0.8,
        ),
        KeypointFeatureMatch(
            query_index=0,
            render_index=2,
            query_xy=np.asarray([0.0, 0.0]),
            render_xy=np.asarray([2.0, 0.0]),
            similarity=0.8,
            ratio=1.0,
            dual_softmax_confidence=0.7,
        ),
        KeypointFeatureMatch(
            query_index=1,
            render_index=3,
            query_xy=np.asarray([1.0, 0.0]),
            render_xy=np.asarray([3.0, 0.0]),
            similarity=0.5,
            ratio=1.0,
            dual_softmax_confidence=0.6,
        ),
    ]

    kept = retain_topk_matches_per_query(matches, max_per_query=2)

    assert [(match.query_index, match.render_index) for match in kept] == [(0, 1), (0, 2), (1, 3)]


def test_deduplicate_repeated_correspondences_keeps_best_per_cell_pair() -> None:
    low = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([8.0, 8.0]),
        similarity=0.1,
        ratio=1.0,
        dual_softmax_confidence=0.1,
    )
    high = KeypointFeatureMatch(
        query_index=1,
        render_index=1,
        query_xy=np.asarray([5.0, 5.0]),
        render_xy=np.asarray([9.0, 9.0]),
        similarity=0.9,
        ratio=0.1,
        dual_softmax_confidence=0.9,
    )

    matches = deduplicate_repeated_correspondences([low, high], query_cell_size_px=10.0, render_cell_size_px=10.0)

    assert matches == [high]


def test_apply_offset_logits_to_matches_moves_cell_centers_and_filters_dustbin() -> None:
    keep = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    drop = KeypointFeatureMatch(
        query_index=1,
        render_index=0,
        query_xy=np.asarray([12.0, 4.0]),
        render_xy=np.asarray([4.0, 4.0]),
        similarity=0.8,
        ratio=0.2,
    )
    query_logits = np.full((65, 1, 2), -10.0, dtype=np.float32)
    render_logits = np.full((65, 1, 2), -10.0, dtype=np.float32)
    query_logits[9, 0, 0] = 10.0
    render_logits[18, 0, 1] = 10.0
    query_logits[64, 0, 1] = 10.0
    render_logits[64, 0, 0] = 10.0

    refined = apply_offset_logits_to_matches(
        [keep, drop],
        query_logits,
        render_logits,
        query_image_width=16,
        query_image_height=8,
        render_image_width=16,
        render_image_height=8,
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].query_xy, [1.5, 1.5])
    assert np.allclose(refined[0].render_xy, [10.5, 2.5])


def test_apply_offset_logits_to_matches_supports_side_ablations() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    query_logits = np.full((65, 1, 2), -10.0, dtype=np.float32)
    render_logits = np.full((65, 1, 2), -10.0, dtype=np.float32)
    query_logits[9, 0, 0] = 10.0
    render_logits[18, 0, 1] = 10.0

    query_only = apply_offset_logits_to_matches(
        [match],
        query_logits,
        None,
        query_image_width=16,
        query_image_height=8,
        render_image_width=16,
        render_image_height=8,
    )
    render_only = apply_offset_logits_to_matches(
        [match],
        None,
        render_logits,
        query_image_width=16,
        query_image_height=8,
        render_image_width=16,
        render_image_height=8,
    )
    none = apply_offset_logits_to_matches(
        [match],
        None,
        None,
        query_image_width=16,
        query_image_height=8,
        render_image_width=16,
        render_image_height=8,
    )

    assert np.allclose(query_only[0].query_xy, [1.5, 1.5])
    assert np.allclose(query_only[0].render_xy, [12.0, 4.0])
    assert np.allclose(render_only[0].query_xy, [4.0, 4.0])
    assert np.allclose(render_only[0].render_xy, [10.5, 2.5])
    assert none == [match]


def test_apply_pair_fine_logits_to_matches_refines_render_side_only() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    logits = np.full((1, 65), -10.0, dtype=np.float32)
    logits[0, 18] = 10.0

    refined = apply_pair_fine_logits_to_matches(
        [match],
        logits,
        render_image_width=16,
        render_image_height=8,
        render_grid_width=2,
        render_grid_height=1,
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].query_xy, [4.0, 4.0])
    assert np.allclose(refined[0].render_xy, [10.5, 2.5])


def test_apply_pair_fine_logits_to_matches_accepts_original_matcha_64_bins() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    logits = np.full((1, 64), -10.0, dtype=np.float32)
    logits[0, 18] = 10.0

    refined = apply_pair_fine_logits_to_matches(
        [match],
        logits,
        render_image_width=16,
        render_image_height=8,
        render_grid_width=2,
        render_grid_height=1,
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].render_xy, [10.5, 2.5])


def test_apply_pair_fine_logits_to_matches_can_softargmax_offset_bins() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    logits = np.full((1, 64), -30.0, dtype=np.float32)
    logits[0, 18] = 5.0
    logits[0, 19] = 5.0

    refined = apply_pair_fine_logits_to_matches(
        [match],
        logits,
        render_image_width=16,
        render_image_height=8,
        render_grid_width=2,
        render_grid_height=1,
        coordinate_mode="softargmax",
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].render_xy, [11.0, 2.5])


def test_apply_pair_fine_logits_to_matches_can_refine_query_side_only() -> None:
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=1,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
    )
    logits = np.full((1, 65), -10.0, dtype=np.float32)
    logits[0, 18] = 10.0

    refined = apply_pair_fine_logits_to_matches(
        [match],
        logits,
        query_image_width=16,
        query_image_height=8,
        query_grid_width=2,
        query_grid_height=1,
        target_side="query",
    )

    assert len(refined) == 1
    assert np.allclose(refined[0].query_xy, [2.5, 2.5])
    assert np.allclose(refined[0].render_xy, [12.0, 4.0])


def test_apply_keypoint_cell_prior_boosts_supported_matches_without_dropping() -> None:
    supported = KeypointFeatureMatch(
        query_index=1,
        render_index=2,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([8.0, 8.0]),
        similarity=0.5,
        ratio=0.5,
        dual_softmax_confidence=0.4,
    )
    unsupported = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([2.0, 2.0]),
        render_xy=np.asarray([2.0, 2.0]),
        similarity=0.9,
        ratio=0.1,
        dual_softmax_confidence=0.4,
    )

    updated = apply_keypoint_cell_prior_to_matches(
        [unsupported, supported],
        query_candidate_indices=np.asarray([1]),
        render_candidate_indices=np.asarray([2]),
        boost=0.5,
        penalty=0.25,
    )

    assert len(updated) == 2
    assert updated[0].query_index == 1
    assert np.isclose(updated[0].dual_softmax_confidence, 0.6)
    assert np.isclose(updated[1].dual_softmax_confidence, 0.3)


def test_apply_cell_reliability_prior_uses_continuous_scores_without_dropping() -> None:
    high = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([4.0, 4.0]),
        similarity=0.1,
        ratio=0.1,
        dual_softmax_confidence=0.4,
    )
    low = KeypointFeatureMatch(
        query_index=1,
        render_index=1,
        query_xy=np.asarray([12.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
        dual_softmax_confidence=0.4,
    )
    query_reliability = np.asarray([1.0, 0.0], dtype=np.float32)
    render_reliability = np.asarray([1.0, 0.0], dtype=np.float32)

    updated = apply_cell_reliability_prior_to_matches(
        [low, high],
        query_reliability=query_reliability,
        render_reliability=render_reliability,
        boost=0.5,
        penalty=0.5,
    )

    assert len(updated) == 2
    assert updated[0].query_index == 0
    assert updated[0].dual_softmax_confidence > updated[1].dual_softmax_confidence


def test_apply_fine_logit_confidence_to_matches_uses_local_window_peak() -> None:
    ambiguous = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([4.0, 4.0]),
        render_xy=np.asarray([4.0, 4.0]),
        similarity=0.9,
        ratio=0.1,
        dual_softmax_confidence=0.4,
    )
    sharp = KeypointFeatureMatch(
        query_index=1,
        render_index=1,
        query_xy=np.asarray([12.0, 4.0]),
        render_xy=np.asarray([12.0, 4.0]),
        similarity=0.1,
        ratio=0.1,
        dual_softmax_confidence=0.4,
    )
    logits = np.zeros((2, 64), dtype=np.float32)
    logits[1, 9] = 10.0

    updated = apply_fine_logit_confidence_to_matches([ambiguous, sharp], logits, blend=0.5)

    assert len(updated) == 2
    assert updated[0].query_index == 1
    assert updated[0].dual_softmax_confidence > updated[1].dual_softmax_confidence
