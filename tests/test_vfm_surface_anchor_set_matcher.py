from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _match_radio_final_descriptor_points,
    _sample_mapped_vfm_at_pixels,
    _strict_pose_gate_group_count,
)
from feature_extract.vfm.localization.anchor_feature_contract import (
    RADIO_FINAL_ANCHOR_FEATURE,
    anchor_feature_kind,
    compose_anchor_query_descriptors,
)
from feature_extract.vfm.localization.surface_anchor_set_matcher import (
    SurfaceAnchorSetMatcher,
    SurfaceAnchorSetMatcherConfig,
    build_surface_anchor_episode,
    load_surface_anchor_set_matcher,
    match_query_to_surface_anchors,
    save_surface_anchor_set_matcher,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
    LocalFeatureFrame,
    SurfaceMapletMatchResult,
)
from feature_extract.vfm.surface_maplet_bank import (
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
)


def _fixture():
    descriptor_dim = 8
    anchor_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
    descriptors = np.eye(descriptor_dim, dtype=np.float32)[:4]
    maplets = VfmSurfaceMapletBank(
        maplet_ids=np.asarray([5], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float64),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        tangent_frames=np.eye(3, dtype=np.float32)[None],
        extents=np.ones((1, 3), dtype=np.float32),
        descriptors=np.ones((1, descriptor_dim), dtype=np.float32),
        quality_scores=np.ones((1,), dtype=np.float32),
        descriptor_variances=np.zeros((1,), dtype=np.float32),
        anchor_offsets=np.asarray([0, 4], dtype=np.int64),
        anchor_ids=anchor_ids,
        support_offsets=np.asarray([0, 4], dtype=np.int64),
        support_element_ids=anchor_ids,
        view_offsets=np.asarray([0, 2], dtype=np.int64),
        view_image_ids=("view_a", "view_b"),
        view_token_xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        view_grid_sizes=np.asarray([[4, 4], [4, 4]], dtype=np.int32),
        view_descriptors=np.ones((2, descriptor_dim), dtype=np.float32),
        view_quality_scores=np.ones((2,), dtype=np.float32),
        metadata={
            "vfm_layer": "radio_final",
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    xyz = np.asarray(
        [
            [-0.5, -0.5, 5.0],
            [0.5, -0.5, 5.0],
            [-0.5, 0.5, 5.0],
            [0.5, 0.5, 5.0],
        ],
        dtype=np.float64,
    )
    anchors = StableSurfaceAnchorMap(
        anchor_ids=anchor_ids,
        owner_maplet_ids=np.full((4,), 5, dtype=np.int64),
        surface_element_ids=anchor_ids,
        parent_primitive_indices=np.arange(4, dtype=np.int64),
        xyz=xyz,
        normals=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)
        ),
        tangent_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None], (4, 1, 1)
        ),
        support_radii=np.full((4,), 0.05, dtype=np.float32),
        quality_scores=np.ones((4,), dtype=np.float32),
        geometry_confidence=np.ones((4,), dtype=np.float32),
        opacity=np.ones((4,), dtype=np.float32),
        observation_offsets=np.arange(0, 9, 2, dtype=np.int64),
        observation_image_ids=tuple(
            value for _ in range(4) for value in ("view_a", "view_b")
        ),
        observation_xy=np.zeros((8, 2), dtype=np.float32),
        observation_depth=np.full((8,), 5.0, dtype=np.float32),
        observation_weights=np.ones((8,), dtype=np.float32),
        metadata={
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    bank_descriptors = np.repeat(descriptors, 2, axis=0)
    bank = AnchorLocalDescriptorBank(
        anchor_ids=anchor_ids,
        descriptor_offsets=np.arange(0, 9, 2, dtype=np.int64),
        descriptors=bank_descriptors,
        support_image_ids=tuple(
            value for _ in range(4) for value in ("view_a", "view_b")
        ),
        descriptor_quality=np.ones((8,), dtype=np.float32),
        metadata={
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    query = LocalFeatureFrame(
        image_id="query",
        keypoints_xy=np.asarray(
            [[20.0, 20.0], [80.0, 20.0], [20.0, 80.0], [80.0, 80.0]],
            dtype=np.float32,
        ),
        descriptors=descriptors,
        scores=np.ones((4,), dtype=np.float32),
    )
    return maplets, anchors, bank, query


def test_surface_anchor_episode_is_feature_only_and_image_disjoint() -> None:
    maplets, anchors, bank, query = _fixture()
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=8,
        model_dim=16,
        num_heads=4,
        query_layers=1,
        anchor_layers=1,
        dropout=0.0,
        maximum_support_descriptors=2,
        maximum_query_nodes=8,
        maximum_anchors=8,
    )
    episode, index = build_surface_anchor_episode(
        query=query,
        query_rows=np.arange(4, dtype=np.int64),
        query_image_size=(100, 100),
        maplet_id=5,
        maplet_probabilities=np.ones((4,), dtype=np.float32),
        region_distances=np.zeros((4,), dtype=np.float32),
        maplets=maplets,
        anchors=anchors,
        descriptor_bank=bank,
        config=config,
        target_anchor_ids=anchors.anchor_ids,
        excluded_support_image_ids=("view_a",),
    )
    episode.validate()
    np.testing.assert_array_equal(index.anchor_ids, anchors.anchor_ids)
    np.testing.assert_array_equal(
        episode.target_track_indices.numpy(), np.arange(4)
    )
    assert episode.support_mask[:, 0].all()
    assert not episode.support_mask[:, 1].any()
    model = SurfaceAnchorSetMatcher(config).eval()
    output = model(episode)
    assert output["query_log_probabilities"].shape == (4, 5)
    assert torch.isfinite(output["query_log_probabilities"]).all()


def test_maplet_first_surface_set_matcher_returns_anchor_groups() -> None:
    maplets, anchors, bank, query = _fixture()
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=8,
        model_dim=16,
        num_heads=4,
        query_layers=1,
        anchor_layers=1,
        dropout=0.0,
        maximum_support_descriptors=2,
        maximum_query_nodes=8,
        maximum_anchors=8,
        maximum_maplets_per_region=1,
    )
    model = SurfaceAnchorSetMatcher(config).eval()
    maplet_match = SurfaceMapletMatchResult(
        candidate_maplet_ids=np.asarray([[5]], dtype=np.int64),
        candidate_logits=np.asarray([[5.0]], dtype=np.float32),
        candidate_probabilities=np.asarray([[0.95]], dtype=np.float32),
        null_probabilities=np.asarray([0.05], dtype=np.float32),
        selected_maplet_ids=np.asarray([5], dtype=np.int64),
        layout_residuals=np.asarray([[0.0]], dtype=np.float32),
        support_view_id=None,
        support_view_score=0.0,
    )
    pool, diagnostics = match_query_to_surface_anchors(
        model=model,
        query=query,
        query_image_size=(100, 100),
        query_region_xy=np.asarray([[2.0, 2.0]], dtype=np.float32),
        query_region_grid_size=(4, 4),
        maplet_match=maplet_match,
        maplets=maplets,
        anchors=anchors,
        descriptor_bank=bank,
        device="cpu",
        top_l=3,
    )
    assert len(pool) == 4
    assert np.all(np.any(pool.valid_mask, axis=1))
    assert set(pool.anchor_ids[pool.valid_mask].tolist()) <= set(
        anchors.anchor_ids.tolist()
    )
    assert diagnostics["episode_count"] == 1
    assert diagnostics["raw_episode_count_before_scene_budget"] == 1
    assert diagnostics["uses_mapping_rgb_at_inference"] is False
    assert diagnostics["uses_sfm_tracks"] is False


def test_surface_anchor_top_l_moves_omitted_probability_to_null() -> None:
    maplets, anchors, bank, query = _fixture()
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=8,
        model_dim=16,
        num_heads=4,
        query_layers=1,
        anchor_layers=1,
        dropout=0.0,
        maximum_support_descriptors=2,
        maximum_query_nodes=8,
        maximum_anchors=8,
        maximum_maplets_per_region=1,
    )
    model = SurfaceAnchorSetMatcher(config).eval()
    maplet_match = SurfaceMapletMatchResult(
        candidate_maplet_ids=np.asarray([[5]], dtype=np.int64),
        candidate_logits=np.asarray([[5.0]], dtype=np.float32),
        candidate_probabilities=np.asarray([[0.95]], dtype=np.float32),
        null_probabilities=np.asarray([0.05], dtype=np.float32),
        selected_maplet_ids=np.asarray([5], dtype=np.int64),
        layout_residuals=np.asarray([[0.0]], dtype=np.float32),
        support_view_id=None,
        support_view_score=0.0,
    )
    common = dict(
        model=model,
        query=query,
        query_image_size=(100, 100),
        query_region_xy=np.asarray([[2.0, 2.0]], dtype=np.float32),
        query_region_grid_size=(4, 4),
        maplet_match=maplet_match,
        maplets=maplets,
        anchors=anchors,
        descriptor_bank=bank,
        device="cpu",
    )
    full, _diagnostics = match_query_to_surface_anchors(top_l=4, **common)
    truncated, diagnostics = match_query_to_surface_anchors(top_l=2, **common)
    for row in range(len(query.keypoints_xy)):
        for column in range(2):
            anchor_id = int(truncated.anchor_ids[row, column])
            full_column = int(
                np.flatnonzero(full.anchor_ids[row] == anchor_id)[0]
            )
            np.testing.assert_allclose(
                truncated.candidate_probabilities[row, column],
                full.candidate_probabilities[row, full_column],
                atol=1e-6,
            )
        np.testing.assert_allclose(
            truncated.candidate_probabilities[row].sum()
            + truncated.null_probabilities[row],
            1.0,
            atol=1e-6,
        )
    assert diagnostics["omitted_top_l_anchor_mass_transferred_to_null"] is True


def test_query_aligned_radio_maplets_override_sparse_region_assignment() -> None:
    maplets, anchors, bank, query = _fixture()
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=8,
        model_dim=16,
        num_heads=4,
        query_layers=1,
        anchor_layers=1,
        dropout=0.0,
        maximum_support_descriptors=2,
        maximum_query_nodes=8,
        maximum_anchors=8,
        maximum_maplets_per_region=1,
    )
    model = SurfaceAnchorSetMatcher(config).eval()
    sparse_match = SurfaceMapletMatchResult(
        candidate_maplet_ids=np.asarray([[999]], dtype=np.int64),
        candidate_logits=np.asarray([[5.0]], dtype=np.float32),
        candidate_probabilities=np.asarray([[0.95]], dtype=np.float32),
        null_probabilities=np.asarray([0.05], dtype=np.float32),
        selected_maplet_ids=np.asarray([999], dtype=np.int64),
        layout_residuals=np.asarray([[0.0]], dtype=np.float32),
        support_view_id=None,
        support_view_score=0.0,
    )
    aligned_match = SurfaceMapletMatchResult(
        candidate_maplet_ids=np.full((4, 1), 5, dtype=np.int64),
        candidate_logits=np.full((4, 1), 5.0, dtype=np.float32),
        candidate_probabilities=np.full((4, 1), 0.95, dtype=np.float32),
        null_probabilities=np.full((4,), 0.05, dtype=np.float32),
        selected_maplet_ids=np.full((4,), 5, dtype=np.int64),
        layout_residuals=np.full((4, 1), np.inf, dtype=np.float32),
        support_view_id=None,
        support_view_score=float("-inf"),
    )
    pool, diagnostics = match_query_to_surface_anchors(
        model=model,
        query=query,
        query_image_size=(100, 100),
        query_region_xy=np.asarray([[2.0, 2.0]], dtype=np.float32),
        query_region_grid_size=(4, 4),
        maplet_match=sparse_match,
        maplets=maplets,
        anchors=anchors,
        descriptor_bank=bank,
        device="cpu",
        top_l=3,
        query_aligned_maplet_match=aligned_match,
    )
    assert np.all(np.any(pool.valid_mask, axis=1))
    np.testing.assert_allclose(
        pool.candidate_probabilities.sum(axis=1) + pool.null_probabilities,
        np.ones((len(pool),), dtype=np.float32),
        atol=1e-5,
    )
    assert diagnostics["uses_query_aligned_radio_final_maplets"] is True
    assert diagnostics["omitted_top_l_anchor_mass_transferred_to_null"] is True
    assert diagnostics["posterior_probability_conserved"] is True


def test_surface_anchor_checkpoint_fails_closed_on_runtime_contract(
    tmp_path: Path,
) -> None:
    config = SurfaceAnchorSetMatcherConfig(
        descriptor_dim=8,
        model_dim=16,
        num_heads=4,
        query_layers=1,
        anchor_layers=1,
        dropout=0.0,
    )
    model = SurfaceAnchorSetMatcher(config)
    checkpoint = tmp_path / "matcher.pt"
    save_surface_anchor_set_matcher(
        checkpoint, model, metadata={"experiment": "unit"}
    )
    loaded, payload = load_surface_anchor_set_matcher(
        checkpoint, device="cpu"
    )
    assert isinstance(loaded, SurfaceAnchorSetMatcher)
    assert payload["metadata"]["uses_mapping_rgb_at_inference"] is False
    payload["metadata"]["uses_sfm_tracks"] = True
    torch.save(payload, checkpoint)
    try:
        load_surface_anchor_set_matcher(checkpoint, device="cpu")
    except ValueError as error:
        assert "map-only contract" in str(error)
    else:
        raise AssertionError("illegal checkpoint must fail closed")


def test_pose_gate_does_not_treat_nonzero_candidate_mass_as_confidence() -> None:
    diagnostics = {
        "conditionally_confident_group_count": 3,
        "conditionally_matchable_group_count": 817,
    }
    assert _strict_pose_gate_group_count(diagnostics) == 3


def test_mapped_radio_final_sampling_aligns_image_corners() -> None:
    feature = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sampled = _sample_mapped_vfm_at_pixels(
        feature,
        np.asarray([[0.0, 0.0], [99.0, 0.0]], dtype=np.float32),
        image_width=100,
        image_height=50,
    )
    np.testing.assert_allclose(
        sampled,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_radio_only_anchor_contract_discards_alike_descriptors() -> None:
    alike = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    radio = np.asarray([[0.0, 3.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    output = compose_anchor_query_descriptors(
        alike_descriptors=alike,
        radio_final_descriptors=radio,
        feature_kind=RADIO_FINAL_ANCHOR_FEATURE,
        expected_dim=3,
    )
    np.testing.assert_allclose(
        output,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert (
        anchor_feature_kind(
            {"local_feature": "radio_final_at_alike_detection"}
        )
        == RADIO_FINAL_ANCHOR_FEATURE
    )


def test_radio_only_local_measurement_uses_radio_correlation() -> None:
    feature = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    xy, descriptors, scores = _match_radio_final_descriptor_points(
        mapped_feature=feature,
        predicted_xy=np.asarray([[99.0, 0.0]], dtype=np.float32),
        support_descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        image_width=100,
        image_height=100,
        search_radius_px=99,
        search_step_px=99,
    )
    np.testing.assert_allclose(xy, np.asarray([[0.0, 0.0]], np.float32))
    np.testing.assert_allclose(
        descriptors, np.asarray([[1.0, 0.0]], np.float32), atol=1e-6
    )
    np.testing.assert_allclose(scores, np.ones((1,), np.float32), atol=1e-6)


def test_anchor_descriptor_bank_roundtrips_view_directions(
    tmp_path: Path,
) -> None:
    _maplets, _anchors, bank, _query = _fixture()
    directions = np.tile(
        np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32),
        (len(bank.descriptors), 1),
    )
    conditioned = AnchorLocalDescriptorBank(
        anchor_ids=bank.anchor_ids,
        descriptor_offsets=bank.descriptor_offsets,
        descriptors=bank.descriptors,
        support_image_ids=bank.support_image_ids,
        descriptor_quality=bank.descriptor_quality,
        support_view_directions=directions,
        metadata=bank.metadata,
    )
    path = tmp_path / "bank.npz"
    conditioned.save_npz(path)
    loaded = AnchorLocalDescriptorBank.load_npz(path)
    np.testing.assert_allclose(
        loaded.support_view_directions,
        np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(bank.descriptors), 1),
        ),
    )


def test_production_localizer_has_no_mapping_image_or_pairwise_match_cli() -> None:
    source = Path(
        "feature_extract/tools/vfm/localize_2dgs_surface_queries.py"
    ).read_text()
    for forbidden_option in (
        "--observation_bank",
        "--mapping_pose_file",
        "--surface_depth_bank",
        "--loftr",
    ):
        assert forbidden_option not in source
