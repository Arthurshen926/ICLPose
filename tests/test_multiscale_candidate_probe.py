from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapImageObservation
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES,
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_FAMILIES,
    GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES,
    MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES,
    STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    ContextDescriptorSummary,
    LocalContextNodes,
    PerViewLinearCandidateProbe,
    PerViewMLPCandidateProbe,
    aggregate_per_view_logits,
    aggregate_per_view_learned_mixture,
    absolute_global_transport_scale_feature_names,
    batched_absolute_global_transport_features,
    batched_absolute_global_transport_position_features,
    batched_context_cost_volume_hough_features,
    batched_dense_local_translation_mode_features,
    batched_masked_translation_correlation_features,
    batched_spatial_pyramid_shift_correlation_features,
    batched_spatial_pyramid_shift_overlap_features,
    batched_wide_context_full_correlation_features,
    cost_volume_multiscale_per_view_feature_vector,
    context_shift_correlation,
    context_shift_feature_names,
    context_shift_overlap_feature_names,
    context_shift_statistics,
    context_summary_similarity,
    context_regional_similarity,
    crop_spatial_grid_context,
    feature_indices_for_family,
    fixed_candidate_support_global_context_cosine,
    landmark_region_prototype_similarity,
    multiscale_per_view_feature_vector,
    fit_per_view_feature_normalizer,
    normalized_per_view_model_input,
    predict_per_view_linear_probe,
    resample_context_descriptor_grid,
    set_membership_negative_log_likelihood,
    summarize_multiscale_context,
    structured_multiscale_per_view_feature_vector,
    train_per_view_linear_probe,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
)
from feature_extract.tools.vfm.fit_multiscale_candidate_probe import (
    _cost_volume_materialized_feature_names,
    _replace_overlay_rows,
    _stable_family_seed,
    _train_geometric_target_membership,
    _train_registered_track_identity_target_membership,
)


def _context(*, swapped: bool = False) -> LocalContextNodes:
    # The center, left, and right descriptors are intentionally distinct.  A
    # coordinate swap should preserve pooled similarity but lower grid-aligned
    # similarity, which is the invariant the S1 probe needs.
    xy = np.asarray([[0.0, 0.0], [-4.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    descriptors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    if swapped:
        descriptors = descriptors[[0, 2, 1]]
    return LocalContextNodes(xy=xy, alike=descriptors, intermediate=descriptors)


def test_context_grid_preserves_relative_layout() -> None:
    from feature_extract.vfm.localization.multiscale_candidate_probe import (
        summarize_context_descriptors,
    )

    query = summarize_context_descriptors(
        _context(), descriptor_name="alike", grid_size=3, radius_px=8.0
    )
    matched = summarize_context_descriptors(
        _context(), descriptor_name="alike", grid_size=3, radius_px=8.0
    )
    swapped = summarize_context_descriptors(
        _context(swapped=True), descriptor_name="alike", grid_size=3, radius_px=8.0
    )
    pooled_match, grid_match, overlap_match = context_summary_similarity(query, matched)
    pooled_swap, grid_swap, overlap_swap = context_summary_similarity(query, swapped)

    assert np.isclose(pooled_match, 1.0)
    assert np.isclose(grid_match, 1.0)
    assert np.isclose(overlap_match, overlap_swap)
    assert np.isclose(pooled_match, pooled_swap)
    assert grid_swap < grid_match


def test_dense_local_mode_features_recover_bounded_translation() -> None:
    # The support patch is shifted one cell to the right.  The translation
    # mode statistic must retain that peak without exposing a full pair matrix.
    query = torch.eye(25, dtype=torch.float32).reshape(1, 5, 5, 25)
    support = torch.zeros_like(query)
    support[:, :, 1:] = query[:, :, :-1]
    valid = torch.ones((1, 5, 5), dtype=torch.bool)

    features = batched_dense_local_translation_mode_features(
        query,
        valid,
        support,
        valid,
        maximum_shift=1,
        temperature=0.1,
    )

    assert features.shape == (1, 10)
    assert torch.isfinite(features).all()
    assert float(features[0, 2]) > float(features[0, 0])
    assert float(features[0, 7]) == 1.0
    assert float(features[0, 8]) == 0.0


def test_anchor_aligned_translation_features_keep_all_shift_scores_and_coverage() -> None:
    query = torch.zeros((1, 3, 3, 2), dtype=torch.float32)
    support = torch.zeros((1, 3, 3, 2), dtype=torch.float32)
    query[..., 0] = 1.0
    support[..., 0] = 1.0
    valid = torch.ones((1, 3, 3), dtype=torch.bool)

    output = batched_masked_translation_correlation_features(
        query, valid, support, valid, maximum_shift=1
    )

    assert output.shape == (1, 18)
    np.testing.assert_allclose(output[0, :9].numpy(), np.ones((9,)), atol=1e-6)
    np.testing.assert_allclose(
        output[0, 9:].numpy(),
        np.asarray([4, 6, 4, 6, 9, 6, 4, 6, 4], dtype=np.float32) / 9.0,
        atol=1e-6,
    )
    assert len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES) > 18
    assert feature_indices_for_family(
        "anchor_global_layout_context_only",
        feature_names=ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
    ).size == len(ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES)


def test_spatial_pyramid_shift_features_keep_region_and_shift_separate() -> None:
    # The support agrees with the query in the upper-left region but loses the
    # lower-right content.  A global translation statistic cannot represent
    # this distinction; the spatial-pyramid tensor must retain it.
    side = 6
    query = torch.eye(side * side, dtype=torch.float32).reshape(1, side, side, -1)
    support = query.clone()
    support[:, 4:, 4:] = 0.0
    valid = torch.ones((1, side, side), dtype=torch.bool)

    features = batched_spatial_pyramid_shift_correlation_features(
        query,
        valid,
        support,
        valid,
        maximum_shift=1,
        spatial_bin_count=3,
    )

    shifts = 3
    center = 1 * shifts + 1
    upper_left_center = 0 * shifts * shifts + center
    lower_right_center = 8 * shifts * shifts + center
    assert features.shape == (1, 3 * 3 * shifts * shifts)
    assert torch.isfinite(features).all()
    assert float(features[0, upper_left_center]) > 0.99
    assert float(features[0, lower_right_center]) < 0.01


def test_spatial_pyramid_shift_features_recover_relative_shift_per_region() -> None:
    query = torch.eye(25, dtype=torch.float32).reshape(1, 5, 5, 25)
    support = torch.zeros_like(query)
    support[:, :, 1:] = query[:, :, :-1]
    valid = torch.ones((1, 5, 5), dtype=torch.bool)

    features = batched_spatial_pyramid_shift_correlation_features(
        query,
        valid,
        support,
        valid,
        maximum_shift=1,
        spatial_bin_count=1,
    )

    shifts = 3
    center = 1 * shifts + 1
    right = 1 * shifts + 2
    assert float(features[0, right]) > float(features[0, center])


def test_spatial_pyramid_overlap_control_contains_no_descriptor_values() -> None:
    query_valid = torch.ones((1, 5, 5), dtype=torch.bool)
    support_valid = torch.ones_like(query_valid)
    support_valid[:, :, -1] = False

    controls = batched_spatial_pyramid_shift_overlap_features(
        query_valid,
        support_valid,
        maximum_shift=1,
        spatial_bin_count=1,
    )

    shifts = 3
    center = 1 * shifts + 1
    right = 1 * shifts + 2
    assert controls.shape == (1, shifts * shifts)
    assert torch.isfinite(controls).all()
    assert float(controls[0, right]) < float(controls[0, center])


def test_absolute_global_transport_preserves_full_image_region_phase() -> None:
    # A horizontal half swap moves the descriptors from the query's left
    # region to the support's right region.  This must be visible in the
    # full-image transport matrix; an anchor-relative crop cannot express it.
    query = torch.eye(16, dtype=torch.float32).reshape(1, 4, 4, 16)
    support = query.clone()
    swapped = query.clone()
    swapped[:, :, :2] = query[:, :, 2:]
    swapped[:, :, 2:] = query[:, :, :2]
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    matched = batched_absolute_global_transport_features(
        query, valid, support, valid, region_grid_size=2, temperature=0.01
    )[0]
    shifted = batched_absolute_global_transport_features(
        query, valid, swapped, valid, region_grid_size=2, temperature=0.01
    )[0]
    names = absolute_global_transport_scale_feature_names("unit", region_grid_size=2)
    q_left_to_left = names.index("unit_qregion_r0_c0_sregion_r0_c0_attention_mass")
    q_left_to_right = names.index("unit_qregion_r0_c0_sregion_r0_c1_attention_mass")
    assert matched.shape == (len(names),)
    assert torch.isfinite(matched).all()
    assert matched[q_left_to_left] > matched[q_left_to_right]
    assert shifted[q_left_to_right] > shifted[q_left_to_left]


def test_absolute_global_transport_position_control_keeps_only_mask_geometry() -> None:
    query_valid = torch.ones((1, 4, 4), dtype=torch.bool)
    support_valid = torch.ones((1, 4, 4), dtype=torch.bool)
    query_valid[:, :3, :3] = False
    position = batched_absolute_global_transport_position_features(
        query_valid, support_valid, region_grid_size=2
    )[0]
    zero_grid = torch.zeros((1, 4, 4, 1), dtype=torch.float32)
    expected = batched_absolute_global_transport_features(
        zero_grid, query_valid, zero_grid, support_valid, region_grid_size=2, temperature=1.0
    )[0]
    names = absolute_global_transport_scale_feature_names("position", region_grid_size=2)
    coverage = names.index("position_qregion_r0_c0_query_coverage")
    assert position.shape == (len(names),)
    assert torch.isfinite(position).all()
    assert position[coverage] < 1.0
    torch.testing.assert_close(position, expected)
    assert feature_indices_for_family(
        "absolute_global_transport_context_only",
        feature_names=ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    ).size == len(ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES)
    assert feature_indices_for_family(
        "absolute_global_transport_position_only",
        feature_names=ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES,
    ).size == len(ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES)


def test_spatial_grid_crop_and_shift_correlation_preserve_layout() -> None:
    # Every source cell has a distinct unit descriptor.  The support layout is
    # shifted one cell right, so the best correlation must occur at dx=+1.
    query_grid = np.eye(25, dtype=np.float32).reshape(5, 5, 25)
    support_grid = np.zeros_like(query_grid)
    support_grid[:, 1:] = query_grid[:, :-1]
    query = crop_spatial_grid_context(
        query_grid,
        image_size=np.asarray([500, 500]),
        xy=np.asarray([250.0, 250.0]),
        window_size=3,
    )
    support = crop_spatial_grid_context(
        support_grid,
        image_size=np.asarray([500, 500]),
        xy=np.asarray([250.0, 250.0]),
        window_size=3,
    )
    scores = context_shift_correlation(query, support, maximum_shift=1)
    statistics_scores, overlaps = context_shift_statistics(query, support, maximum_shift=1)
    names = context_shift_feature_names("unit", maximum_shift=1)
    overlap_names = context_shift_overlap_feature_names("unit", maximum_shift=1)

    assert query.grid.shape == (9, 25)
    assert len(scores) == len(names) == 9
    assert np.isclose(scores[names.index("unit_shift_dy0_dx1_cosine")], 1.0)
    np.testing.assert_allclose(scores, statistics_scores, equal_nan=True)
    assert np.isclose(
        overlaps[overlap_names.index("unit_shift_dy0_dx1_overlap_fraction")], 2.0 / 3.0
    )
    assert scores[names.index("unit_shift_dy0_dx0_cosine")] < 1.0


def _full_grid_summary(size: int) -> ContextDescriptorSummary:
    grid = np.zeros((int(size) ** 2, 2), dtype=np.float32)
    grid[:, 0] = 1.0
    return ContextDescriptorSummary(
        pooled=np.asarray([1.0, 0.0], dtype=np.float32),
        grid=grid,
        valid_cells=np.ones((int(size) ** 2,), dtype=bool),
    )


def test_cost_volume_hough_retains_large_anchor_relative_translation() -> None:
    # A support layout shifted right by one cell has a coherent dx=+1 Hough
    # peak.  The old S1b feature representation only recorded local aligned
    # averages; this fixture guards the full pairwise cost-volume path.
    query = np.eye(9, dtype=np.float32)
    support = np.zeros_like(query)
    support.reshape(3, 3, 9)[:, 1:] = query.reshape(3, 3, 9)[:, :-1]
    query_valid = np.ones((1, 9), dtype=bool)
    support_valid = np.ones((1, 3, 3), dtype=bool)
    support_valid[:, :, 0] = False
    value = batched_context_cost_volume_hough_features(
        torch.from_numpy(query[None]),
        torch.from_numpy(query_valid),
        torch.from_numpy(support[None]),
        torch.from_numpy(support_valid.reshape(1, 9)),
        temperature=0.05,
    ).cpu().numpy()[0]
    # First field is mean cosine over shifts ordered dy=-2..2, dx=-2..2.
    shift_dx1 = (0 + 2) * 5 + (1 + 2)
    shift_dx0 = (0 + 2) * 5 + (0 + 2)
    assert np.isclose(value[shift_dx1], 1.0)
    assert value[shift_dx1] > value[shift_dx0]
    attention_offset = 25
    assert value[attention_offset + shift_dx1] > value[attention_offset + shift_dx0]


def test_cost_volume_missing_context_is_finite_and_pose_free() -> None:
    grid = torch.zeros((2, 9, 4), dtype=torch.float32)
    missing = torch.zeros((2, 9), dtype=torch.bool)
    value = batched_context_cost_volume_hough_features(grid, missing, grid, missing)
    assert value.shape == (2, 102)
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(value) == 0


def test_wide_full_correlation_retains_absolute_cell_layout() -> None:
    # S1e must retain cell identity rather than collapsing every pair with the
    # same translation into a Hough mean. A reversed support grid therefore
    # moves a unit correlation from the diagonal to a different explicit cell.
    query = torch.eye(25, dtype=torch.float32)[None]
    support = torch.eye(25, dtype=torch.float32)[torch.arange(24, -1, -1)][None]
    valid = torch.ones((1, 25), dtype=torch.bool)
    matched = batched_wide_context_full_correlation_features(query, valid, query, valid)
    swapped = batched_wide_context_full_correlation_features(query, valid, support, valid)
    assert matched.shape == swapped.shape == (1, 676)
    assert torch.isclose(matched[0, 0], torch.tensor(1.0))
    assert torch.isclose(swapped[0, 24], torch.tensor(1.0))
    assert torch.isclose(swapped[0, 0], torch.tensor(0.0))
    # The two 25-cell masks and pair-valid fraction follow the 625 correlations.
    assert torch.all(matched[0, 625:675] == 1.0)
    assert torch.isclose(matched[0, 675], torch.tensor(1.0))


def test_wide_full_correlation_missing_context_is_finite_and_pose_free() -> None:
    grid = torch.zeros((2, 25, 4), dtype=torch.float32)
    missing = torch.zeros((2, 25), dtype=torch.bool)
    value = batched_wide_context_full_correlation_features(grid, missing, grid, missing)
    assert value.shape == (2, 676)
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(value) == 0


def test_cost_volume_multiscale_vector_uses_declared_schema() -> None:
    final5 = _full_grid_summary(5)
    final7 = _full_grid_summary(7)
    local11 = _full_grid_summary(11)
    vector = cost_volume_multiscale_per_view_feature_vector(
        radio_final_anchor_cosine=1.0,
        radio_intermediate_anchor_cosine=1.0,
        alike_anchor_cosine=1.0,
        query_final_window5=final5,
        support_final_window5=final5,
        query_final_window7=final7,
        support_final_window7=final7,
        query_intermediate11=local11,
        support_intermediate11=local11,
        query_alike11=local11,
        support_alike11=local11,
    )
    resampled, valid = resample_context_descriptor_grid(final7)
    assert vector.shape == (len(COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    assert resampled.shape == (9, 2)
    assert valid.all()


def test_structured_multiscale_vector_retains_regional_layout() -> None:
    final3 = _full_grid_summary(3)
    final5 = _full_grid_summary(5)
    intermediate7 = _full_grid_summary(7)
    intermediate11 = _full_grid_summary(11)
    vector = structured_multiscale_per_view_feature_vector(
        radio_final_anchor_cosine=1.0,
        radio_intermediate_anchor_cosine=1.0,
        alike_anchor_cosine=1.0,
        query_final_window3=final3,
        support_final_window3=final3,
        query_final_window5=final5,
        support_final_window5=final5,
        query_intermediate7=intermediate7,
        support_intermediate7=intermediate7,
        query_intermediate11=intermediate11,
        support_intermediate11=intermediate11,
        query_alike7=intermediate7,
        support_alike7=intermediate7,
        query_alike11=intermediate11,
        support_alike11=intermediate11,
    )
    regional, coverage = context_regional_similarity(intermediate7, intermediate7)

    assert vector.shape == (len(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(regional, 1.0)
    assert coverage == 1.0


def test_landmark_region_prototype_pools_spatial_bins_before_comparison() -> None:
    # The full crop has the same descriptor multiset after swapping two
    # relative regions.  A landmark-centred prototype must preserve that
    # absolute layout through its pooled 3x3 bins instead of collapsing it.
    query_grid = np.eye(9, dtype=np.float32)
    support_grid = query_grid.copy()
    support_grid[[0, 2]] = support_grid[[2, 0]]
    query = ContextDescriptorSummary(
        pooled=np.mean(query_grid, axis=0),
        grid=query_grid,
        valid_cells=np.ones((9,), dtype=bool),
    )
    matched = ContextDescriptorSummary(
        pooled=np.mean(query_grid, axis=0),
        grid=query_grid,
        valid_cells=np.ones((9,), dtype=bool),
    )
    swapped = ContextDescriptorSummary(
        pooled=np.mean(support_grid, axis=0),
        grid=support_grid,
        valid_cells=np.ones((9,), dtype=bool),
    )
    match_scores, match_coverage = landmark_region_prototype_similarity(query, matched)
    swapped_scores, swapped_coverage = landmark_region_prototype_similarity(query, swapped)

    assert match_scores.shape == (10,)
    np.testing.assert_allclose(match_scores, 1.0)
    assert match_coverage == swapped_coverage == 1.0
    assert np.isclose(swapped_scores[0], 1.0)
    assert swapped_scores[1] < 1.0


def test_multiscale_feature_vector_is_finite_for_matching_context() -> None:
    direct = multiscale_per_view_feature_vector(
        radio_final_anchor_cosine=1.0,
        query_final_grid4=np.asarray([1.0, 0.0], dtype=np.float32),
        support_final_grid4=np.asarray([1.0, 0.0], dtype=np.float32),
        query_context3=_context(),
        support_context3=_context(),
        query_context5=_context(),
        support_context5=_context(),
        radius3_px=8.0,
        radius5_px=16.0,
    )
    summaries = summarize_multiscale_context(
        _context(), _context(), radius3_px=8.0, radius5_px=16.0
    )
    cached = multiscale_per_view_feature_vector(
        radio_final_anchor_cosine=1.0,
        query_final_grid4=np.asarray([1.0, 0.0], dtype=np.float32),
        support_final_grid4=np.asarray([1.0, 0.0], dtype=np.float32),
        query_context3=_context(),
        support_context3=_context(),
        query_context5=_context(),
        support_context5=_context(),
        radius3_px=8.0,
        radius5_px=16.0,
        query_summaries=summaries,
        support_summaries=summaries,
    )
    assert direct.shape == (14,)
    assert np.isfinite(direct).all()
    assert np.allclose(direct[:12], 1.0)
    np.testing.assert_allclose(cached, direct)


def test_support_view_aggregation_is_order_invariant() -> None:
    logits = torch.tensor([[[0.1, 1.2, -0.4], [2.0, 0.3, -1.0]]])
    valid = torch.tensor([[[True, True, False], [True, True, True]]])
    expected = aggregate_per_view_logits(logits, valid)
    permuted = aggregate_per_view_logits(logits[:, :, [2, 0, 1]], valid[:, :, [2, 0, 1]])
    assert torch.allclose(expected, permuted)


def test_learned_support_view_mixture_is_order_invariant() -> None:
    evidence = torch.tensor([[[0.1, 1.2, -0.4], [2.0, 0.3, -1.0]]])
    view = torch.tensor([[[0.7, -0.2, 0.5], [-0.4, 0.9, 0.2]]])
    valid = torch.tensor([[[True, True, False], [True, True, True]]])
    expected = aggregate_per_view_learned_mixture(evidence, view, valid)
    permutation = [2, 0, 1]
    permuted = aggregate_per_view_learned_mixture(
        evidence[:, :, permutation], view[:, :, permutation], valid[:, :, permutation]
    )
    assert torch.allclose(expected, permuted)


def test_fixed_s1_families_exclude_whole_image_retrieval_terms() -> None:
    assert "multiscale_per_view_no_global_retrieval" in MULTISCALE_CANDIDATE_PROBE_FAMILIES
    for fields in MULTISCALE_CANDIDATE_PROBE_FAMILIES.values():
        assert all("global" not in field and "summary" not in field for field in fields)
    indices = feature_indices_for_family("multiscale_per_view_no_global_retrieval")
    assert indices.ndim == 1
    assert len(indices) == 14


def test_structured_context_only_control_excludes_existing_anchor_scores() -> None:
    anchor_indices = feature_indices_for_family(
        "structured_existing_anchor_control",
        feature_names=STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    )
    context_indices = feature_indices_for_family(
        "structured_candidate_specific_context_only",
        feature_names=STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    )
    assert len(anchor_indices) == len(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES)
    assert len(context_indices) == len(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES)
    assert not set(anchor_indices.tolist()) & set(context_indices.tolist())


def test_cost_volume_final_context_control_excludes_anchor_scores() -> None:
    fields = COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FAMILIES[
        "cost_volume_radio_final_context_only"
    ]
    assert fields
    assert not set(fields) & set(COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES)
    assert all(name.startswith("radio_final_grid8_") for name in fields)


def test_wide_full_correlation_context_control_excludes_anchor_scores() -> None:
    fields = WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FAMILIES[
        "wide_fullcorr_candidate_specific_context_only"
    ]
    assert fields
    assert not set(fields) & set(
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    )
    assert len(WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES) == 2031


def test_global_context_factor_is_fixed_candidate_support_view_specific() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = np.asarray(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1.0], [1.0, 1.0]]],
        ],
        dtype=np.float32,
    )
    valid = np.asarray(
        [[[True, True], [True, False]], [[True, True], [True, False]]], dtype=bool
    )
    scores = fixed_candidate_support_global_context_cosine(query, support, valid)

    assert scores.shape == valid.shape
    np.testing.assert_allclose(scores[0, 0], [1.0, 0.0])
    np.testing.assert_allclose(scores[1, 0], [1.0, 0.0])
    assert np.isclose(scores[0, 1, 0], -1.0)
    assert scores[0, 1, 1] == 0.0
    # The function only follows supplied candidate/view slots.  Swapping two
    # fixed support views swaps their evidence instead of doing any pool-wide
    # normalization or image search.
    swapped = fixed_candidate_support_global_context_cosine(
        query, support[:, :, [1, 0]], valid[:, :, [1, 0]]
    )
    np.testing.assert_allclose(swapped, scores[:, :, [1, 0]])


def test_global_context_control_excludes_anchor_scores() -> None:
    context_fields = GLOBAL_CONTEXT_CANDIDATE_PROBE_FAMILIES[
        "global_context_candidate_specific_context_only"
    ]
    primary_fields = GLOBAL_CONTEXT_CANDIDATE_PROBE_FAMILIES[
        "global_context_radio_final_soft_factor_with_anchor"
    ]
    assert context_fields == ("radio_final_global_support_image_cosine",)
    assert not set(context_fields) & set(GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES)
    assert set(GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES).issubset(primary_fields)
    assert primary_fields[-1] == "radio_final_global_support_image_cosine"
    assert len(GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES) == 4


def test_wide_full_correlation_materialization_rejects_unexported_scales() -> None:
    exported = WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES[0]
    available = _cost_volume_materialized_feature_names(
        {
            "format": "wide_full_correlation_multiscale_candidate_probe_features_v1",
            "cost_volume": {"materialized_scale_names": [exported]},
        },
        feature_names=WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    )
    assert all(
        name in available
        for name in WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    )
    assert any(name.startswith(f"{exported}_") for name in available)
    assert not any(
        name.startswith(f"{WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES[1]}_")
        for name in available
    )


def test_probe_probability_mass_includes_explicit_null() -> None:
    rng = np.random.default_rng(7)
    # Two candidates, two views, and all declared feature columns.  Candidate
    # zero receives a separable score for half of the train groups, candidate
    # one for the remainder; one group is explicitly null.
    features = rng.normal(size=(12, 2, 2, 14)).astype(np.float32)
    targets = np.asarray([0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    for row, target in enumerate(targets):
        if target < 2:
            features[row, target, :, 0] += 4.0
    view_valid = np.ones((12, 2, 2), dtype=bool)
    model, normalizer, metadata = train_per_view_linear_probe(
        features=features,
        view_valid=view_valid,
        targets=targets,
        train_groups=np.ones((12,), dtype=bool),
        family="radio_final_center",
        device=torch.device("cpu"),
        epochs=20,
        batch_size=4,
        learning_rate=0.05,
        seed=3,
    )
    probability, null_probability, per_view = predict_per_view_linear_probe(
        model,
        features=features,
        view_valid=view_valid,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=5,
    )
    assert metadata["fit_group_count"] == 12
    assert per_view.shape == (12, 2, 2)
    assert np.allclose(probability.sum(axis=1) + null_probability, 1.0, atol=1e-6)


def test_overlay_replacement_preserves_other_proposal_rows() -> None:
    base = {
        "candidate_track_ids": np.asarray(
            [[1, 2], [3, 4], [5, 6]], dtype=np.int64
        ),
        "candidate_probabilities": np.asarray(
            [[0.2, 0.3], [0.1, 0.7], [0.4, 0.2]], dtype=np.float32
        ),
        "null_probabilities": np.asarray([0.5, 0.2, 0.4], dtype=np.float32),
    }
    candidate, null, audit = _replace_overlay_rows(
        base=base,
        source_rows=np.asarray([1], dtype=np.int64),
        source_tracks=np.asarray([[3, 4]], dtype=np.int64),
        probabilities=np.asarray([[0.6, 0.1]], dtype=np.float32),
        null_probabilities=np.asarray([0.3], dtype=np.float32),
    )
    np.testing.assert_allclose(candidate[0], base["candidate_probabilities"][0])
    np.testing.assert_allclose(candidate[2], base["candidate_probabilities"][2])
    np.testing.assert_allclose(candidate[1], [0.6, 0.1])
    np.testing.assert_allclose(null, [0.5, 0.3, 0.4])
    assert audit["maximum_full_overlay_mass_error"] < 1e-6


def test_probe_accepts_set_valued_candidate_membership() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(6, 2, 2, 14)).astype(np.float32)
    view_valid = np.ones((6, 2, 2), dtype=bool)
    target = np.zeros((6, 3), dtype=bool)
    target[0, :2] = True
    target[1, 0] = True
    target[2, 1] = True
    target[3:, 2] = True
    model, normalizer, metadata = train_per_view_linear_probe(
        features=features,
        view_valid=view_valid,
        target_membership=target,
        train_groups=np.ones((6,), dtype=bool),
        family="radio_final_center",
        device=torch.device("cpu"),
        epochs=4,
        batch_size=3,
        learning_rate=0.02,
        seed=1,
    )
    probability, null_probability, _ = predict_per_view_linear_probe(
        model,
        features=features,
        view_valid=view_valid,
        normalizer=normalizer,
        device=torch.device("cpu"),
        batch_size=6,
    )
    assert metadata["target_mode"] == "set_membership_candidate_or_null"
    assert metadata["training_objective"] == "set_log_mass_nll_over_target_membership_v1"
    assert np.allclose(probability.sum(axis=1) + null_probability, 1.0, atol=1e-6)


def test_streamed_per_view_normalizer_matches_full_tensor_reference() -> None:
    rng = np.random.default_rng(27)
    features = rng.normal(size=(7, 2, 3, 5)).astype(np.float16)
    features[1, 0, 1, 3] = np.nan
    view_valid = np.asarray(
        [
            [[True, True, False], [True, False, True]],
            [[True, False, True], [True, True, True]],
            [[False, True, True], [True, True, False]],
            [[True, True, True], [False, True, True]],
            [[True, False, True], [True, True, True]],
            [[True, True, False], [True, True, True]],
            [[False, True, True], [True, False, True]],
        ],
        dtype=bool,
    )
    train_groups = np.asarray([True, False, True, True, False, True, True])
    indices = np.asarray([3, 1, 4], dtype=np.int64)

    normalizer = fit_per_view_feature_normalizer(
        features,
        view_valid,
        feature_indices=indices,
        train_groups=train_groups,
        batch_size=2,
    )
    selected = np.asarray(features, dtype=np.float32)[train_groups][..., indices]
    selected_valid = view_valid[train_groups]
    expected_mean = np.zeros((len(indices),), dtype=np.float32)
    expected_scale = np.ones((len(indices),), dtype=np.float32)
    for column in range(len(indices)):
        values = selected[..., column]
        finite = values[selected_valid & np.isfinite(values)].astype(np.float64)
        if finite.size:
            expected_mean[column] = np.mean(finite)
            expected_scale[column] = max(np.std(finite), 1e-3)
    np.testing.assert_allclose(normalizer.mean, expected_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(normalizer.scale, expected_scale, rtol=1e-5, atol=1e-6)

    actual_input = normalized_per_view_model_input(features[:3], normalizer)
    selected = np.asarray(features[:3], dtype=np.float32)[..., indices]
    finite = np.isfinite(selected)
    standardized = (
        np.where(finite, selected, expected_mean) - expected_mean
    ) / expected_scale
    expected_input = np.concatenate(
        [standardized.astype(np.float32), finite.astype(np.float32)], axis=-1
    )
    np.testing.assert_allclose(actual_input, expected_input, rtol=1e-6, atol=1e-6)


def test_set_membership_loss_credits_any_geometrically_valid_candidate() -> None:
    log_probability = torch.log_softmax(torch.zeros((1, 3)), dim=1)
    singleton = set_membership_negative_log_likelihood(
        log_probability, torch.tensor([[True, False, False]])
    )
    two_candidate_set = set_membership_negative_log_likelihood(
        log_probability, torch.tensor([[True, True, False]])
    )
    assert torch.isclose(singleton, torch.tensor(float(np.log(3.0))))
    assert torch.isclose(two_candidate_set, torch.tensor(float(-np.log(2.0 / 3.0))))
    assert two_candidate_set < singleton


def test_zero_initialized_prior_residual_reproduces_fixed_base_distribution() -> None:
    model = PerViewLinearCandidateProbe(3, use_base_prior_residual=True)
    features = torch.zeros((1, 2, 2, 3))
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    base_candidate = torch.log(torch.tensor([[0.2, 0.3]]))
    base_null = torch.log(torch.tensor([0.5]))
    logits, _ = model(features, valid, base_candidate, base_null)
    np.testing.assert_allclose(
        torch.softmax(logits, dim=1).detach().numpy(), [[0.2, 0.3, 0.5]], atol=1e-6
    )


def test_zero_initialized_mlp_prior_residual_reproduces_fixed_base_distribution() -> None:
    model = PerViewMLPCandidateProbe(3, hidden_dim=8, use_base_prior_residual=True)
    features = torch.randn((1, 2, 2, 3))
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    base_candidate = torch.log(torch.tensor([[0.2, 0.3]]))
    base_null = torch.log(torch.tensor([0.5]))
    logits, _ = model(features, valid, base_candidate, base_null)
    np.testing.assert_allclose(
        torch.softmax(logits, dim=1).detach().numpy(), [[0.2, 0.3, 0.5]], atol=1e-6
    )


def test_train_geometric_targets_only_read_train_source_rows(tmp_path) -> None:
    proposals = tmp_path / "proposals.npz"
    np.savez(
        proposals,
        candidate_gt_residuals_px=np.asarray(
            [[0.5, 1.0], [-7.0, np.nan]], dtype=np.float32
        ),
    )
    rows, target, audit = _train_geometric_target_membership(
        proposals_path=proposals,
        source_rows=np.asarray([0, 1], dtype=np.int64),
        candidate_tracks=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        split_names=np.asarray(["train", "validation"]),
        positive_threshold_px=2.0,
    )
    assert rows.tolist() == [0]
    np.testing.assert_array_equal(target, [[True, True, False]])
    assert audit["multi_positive_train_row_count"] == 1


def test_registered_track_targets_only_fit_registered_train_anchors() -> None:
    image = ColmapImageObservation(
        image_id=1,
        image_name="train.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros((3,), dtype=np.float64),
        xys=np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64),
        point3d_ids=np.asarray([101, 202], dtype=np.int64),
    )
    rows, target, audit = _train_registered_track_identity_target_membership(
        query_ids=np.asarray(["train.png", "train.png", "train.png", "train.png"]),
        query_xy=np.asarray(
            [[10.0, 10.0], [20.0, 20.0], [40.0, 40.0], [10.0, 10.0]],
            dtype=np.float32,
        ),
        candidate_tracks=np.asarray(
            [[101, 5], [8, 7], [202, 1], [101, 2]], dtype=np.int64
        ),
        split_names=np.asarray(["train", "train", "train", "validation"]),
        images_by_name={"train.png": image},
        identity_radius_px=0.25,
    )

    # The validation row is never inspected by the target builder.  Train row
    # zero has the exact track; train row one is registered but missing top-L.
    assert rows.tolist() == [0, 1]
    assert target.tolist() == [[True, False, False], [False, False, True]]
    assert audit["registered_supervised_train_row_count"] == 2
    assert audit["unsupervised_train_row_count"] == 1
    assert audit["explicit_null_registered_train_row_count"] == 1


def test_probe_family_seed_is_stable_across_family_ordering() -> None:
    full_seed = _stable_family_seed(7, "structured_multiscale_per_view_no_global_retrieval")
    assert full_seed == _stable_family_seed(
        7, "structured_multiscale_per_view_no_global_retrieval"
    )
    assert full_seed != _stable_family_seed(
        7, "structured_candidate_specific_context_only"
    )
