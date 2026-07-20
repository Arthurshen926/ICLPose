from __future__ import annotations

import json

import numpy as np
import torch

from feature_extract.tools.vfm.build_pose_conditioned_support_alignment_layout import (
    select_rank_band_support_images,
)
from feature_extract.tools.vfm.score_pose_conditioned_support_alignment import (
    _constant_string_column,
    _fixed_same_view_observation_groups,
    _load_spatial_source,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    FixedAffineMapletTopology,
    FrozenSupportAlignmentLayout,
    ImageGridFeatureSource,
    aggregate_fixed_support_image_log_ratios,
    aggregate_observation_group_log_ratios,
    aggregate_view_log_ratios,
    bilinear_sample_image_grid,
    build_fixed_affine_maplet_topology,
    build_context_position_likelihood_maps,
    fit_fixed_affine_maplet_transforms_torch,
    normalized_position_log_ratios,
    project_simple_radial_torch,
    score_affine_warped_maplet_patch_likelihoods,
    sample_context_position_log_ratios,
    summarize_track_log_ratios,
)


def test_bilinear_image_grid_sampling_matches_align_corners_geometry() -> None:
    grid = np.asarray([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=np.float32)
    sampled = bilinear_sample_image_grid(
        grid,
        np.asarray([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float32),
        image_size=(3, 3),
    )
    assert np.allclose(sampled[:, 0], [0.0, 3.0, 1.5], rtol=0.0, atol=1e-6)


def test_simple_radial_projection_and_invalid_depth_contract() -> None:
    xyz = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    pose = torch.eye(4).unsqueeze(0)
    projected, valid = project_simple_radial_torch(
        xyz,
        pose,
        focal_length=2.0,
        principal_x=10.0,
        principal_y=5.0,
        radial_k=0.0,
        image_width=21,
        image_height=11,
    )
    assert torch.allclose(projected[0, :2], torch.tensor([[10.0, 5.0], [11.0, 5.0]]))
    assert valid.tolist() == [[True, True, False]]


def test_full_image_normalized_position_likelihood_is_spatial_and_neutral_when_invalid() -> None:
    grid = torch.eye(4, dtype=torch.float32).reshape(2, 2, 4)
    support = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    projected = torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]], [[0.0, 0.0]]], dtype=torch.float32)
    valid = torch.tensor([[True], [True], [False]])
    values = normalized_position_log_ratios(
        query_grid=grid,
        support_descriptors=support,
        projected_xy=projected,
        projection_valid=valid,
        image_width=3,
        image_height=3,
        temperature=0.1,
    )
    assert values[0, 0] > 1.0
    assert values[1, 0] < -5.0
    assert values[2, 0].item() == 0.0


def test_context_ncc_likelihood_is_full_image_normalized_and_neutral_when_unknown() -> None:
    query = torch.eye(25, dtype=torch.float32).reshape(5, 5, 25)
    support = query[1:4, 1:4].permute(2, 0, 1).unsqueeze(0)
    maps = build_context_position_likelihood_maps(
        query_grid=query,
        support_patches=support,
        support_patch_valid=torch.ones((1, 3, 3), dtype=torch.bool),
        temperature=0.1,
        minimum_support_fraction=1.0,
        minimum_query_overlap_fraction=1.0,
    )
    ratios, evidence_valid = sample_context_position_log_ratios(
        maps=maps,
        projected_xy=torch.tensor(
            [[[2.0, 2.0]], [[1.0, 1.0]], [[2.0, 2.0]]], dtype=torch.float32
        ),
        projection_valid=torch.tensor([[True], [True], [False]]),
        image_width=5,
        image_height=5,
    )
    assert ratios[0, 0] > ratios[1, 0] + 1.0
    assert evidence_valid.tolist() == [[True], [True], [False]]
    assert ratios[2, 0].item() == 0.0


def test_context_template_sampling_uses_source_grid_coordinates() -> None:
    descriptors = torch.eye(25, dtype=torch.float32).reshape(1, 25, 25).numpy()
    source = ImageGridFeatureSource(
        name="synthetic",
        image_ids=np.asarray(["support"]),
        image_sizes=np.asarray([[5, 5]], dtype=np.int64),
        grid_size=5,
        descriptors=descriptors,
        metadata={},
    )
    patches, valid = source.context_patches_torch(
        ["support"],
        np.asarray([[2.0, 2.0]], dtype=np.float32),
        window_size=3,
        device=torch.device("cpu"),
    )
    expected = torch.eye(25, dtype=torch.float32).reshape(5, 5, 25)[1:4, 1:4]
    assert torch.all(valid)
    assert torch.allclose(patches[0].permute(1, 2, 0), expected, atol=1e-6)


def test_view_marginalization_preserves_neutral_invalid_observations() -> None:
    ratios = torch.tensor([[np.log(4.0), 0.0, np.log(2.0)]], dtype=torch.float32)
    valid = torch.tensor([[True, False, True]])
    tracks, active = aggregate_view_log_ratios(
        ratios,
        valid,
        observation_track_groups=torch.tensor([0, 0, 1]),
        track_group_count=2,
        group_observation_counts=torch.tensor([2.0, 1.0]),
    )
    assert torch.allclose(
        tracks,
        torch.tensor([[np.log(2.5), np.log(2.0)]], dtype=torch.float32),
        atol=1e-6,
    )
    assert active.tolist() == [[True, True]]
    summary = summarize_track_log_ratios(tracks)
    assert torch.allclose(
        summary["mean"],
        torch.tensor([(np.log(2.5) + np.log(2.0)) / 2], dtype=torch.float32),
    )


def test_same_view_group_aggregation_keeps_invalid_observations_neutral() -> None:
    ratios = torch.tensor([[2.0, 4.0, 10.0, -2.0]], dtype=torch.float32)
    valid = torch.tensor([[True, False, True, False]])
    groups, active = aggregate_observation_group_log_ratios(
        ratios,
        valid,
        observation_groups=torch.tensor([0, 0, 1, 1]),
        group_count=2,
        group_observation_counts=torch.tensor([2.0, 2.0]),
    )
    assert torch.allclose(groups, torch.tensor([[1.0, 5.0]], dtype=torch.float32))
    assert active.tolist() == [[True, True]]


def test_same_view_group_partition_is_fixed_by_image_and_spatial_block() -> None:
    source = ImageGridFeatureSource(
        name="synthetic",
        image_ids=np.asarray(["a", "b"]),
        image_sizes=np.asarray([[101, 101], [101, 101]], dtype=np.int64),
        grid_size=2,
        descriptors=np.ones((2, 4, 1), dtype=np.float32),
        metadata={},
    )
    groups, counts = _fixed_same_view_observation_groups(
        support_image_ids=np.asarray(["b", "a", "b", "a"]),
        support_xy=np.asarray(
            [[75.0, 75.0], [5.0, 5.0], [25.0, 75.0], [75.0, 5.0]],
            dtype=np.float32,
        ),
        geometry_source=source,
        block_grid=2,
    )
    assert groups.tolist() == [3, 0, 2, 1]
    assert counts.tolist() == [1, 1, 1, 1]


def test_fixed_affine_maplet_topology_uses_same_image_neighbors_only() -> None:
    source = ImageGridFeatureSource(
        name="synthetic",
        image_ids=np.asarray(["a", "b"]),
        image_sizes=np.asarray([[101, 101], [101, 101]], dtype=np.int64),
        grid_size=2,
        descriptors=np.ones((2, 4, 1), dtype=np.float32),
        metadata={},
    )
    topology = build_fixed_affine_maplet_topology(
        support_image_ids=np.asarray(["a", "a", "a", "a", "b"]),
        support_xy=np.asarray(
            [[50.0, 50.0], [35.0, 50.0], [50.0, 35.0], [65.0, 65.0], [50.0, 50.0]],
            dtype=np.float32,
        ),
        support_reprojection_errors=np.asarray([0.1, 0.2, 0.3, 0.4, 0.1], dtype=np.float32),
        geometry_source=source,
        anchor_block_grid=1,
        anchors_per_block=1,
        neighbor_radius_px=32.0,
        max_neighbors=4,
        min_neighbors=3,
    )
    assert topology.maplet_count == 1
    assert topology.anchor_observation_indices.tolist() == [0]
    assert topology.neighbor_counts.tolist() == [3]
    assert set(topology.neighbor_observation_indices[0, :3].tolist()) == {1, 2, 3}


def test_candidate_affine_maplet_fit_recovers_fixed_local_transform() -> None:
    source_xy = torch.tensor(
        [[50.0, 50.0], [35.0, 50.0], [50.0, 35.0], [65.0, 65.0]],
        dtype=torch.float32,
    )
    matrix = torch.tensor([[2.0, 0.5], [-0.25, 1.5]], dtype=torch.float32)
    target_anchor = torch.tensor([100.0, 80.0], dtype=torch.float32)
    projected = target_anchor[None, None, :] + torch.einsum(
        "ij,nj->ni", matrix, source_xy - source_xy[0]
    )[None]
    topology = FixedAffineMapletTopology(
        anchor_observation_indices=np.asarray([0]),
        neighbor_observation_indices=np.asarray([[1, 2, 3]], dtype=np.int64),
        neighbor_counts=np.asarray([3], dtype=np.int64),
    )
    affine, anchors, valid, active, rmse = fit_fixed_affine_maplet_transforms_torch(
        support_xy=source_xy,
        projected_xy=projected,
        projection_valid=torch.ones((1, 4), dtype=torch.bool),
        topology=topology,
        neighbor_sigma_px=64.0,
        minimum_neighbors=3,
        maximum_condition_number=100.0,
        maximum_rmse_px=1e-3,
    )
    assert valid.tolist() == [[True]]
    assert active.tolist() == [[3]]
    assert torch.allclose(anchors[0, 0], target_anchor, atol=1e-5)
    assert torch.allclose(affine[0, 0], matrix, atol=1e-4)
    assert rmse[0, 0].item() < 1e-4


def test_affine_maplet_patch_likelihood_is_spatial_and_neutral_when_invalid() -> None:
    query = torch.eye(49, dtype=torch.float32).reshape(7, 7, 49)
    support = query[2:5, 2:5].permute(2, 0, 1).unsqueeze(0)
    summaries, active, usable = score_affine_warped_maplet_patch_likelihoods(
        query_grid=query,
        support_patches=support,
        support_patch_valid=torch.ones((1, 3, 3), dtype=torch.bool),
        support_image_sizes=torch.tensor([[7.0, 7.0]], dtype=torch.float32),
        support_grid_size=7,
        affine_matrices=torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2).repeat(3, 1, 1, 1),
        projected_anchor_xy=torch.tensor(
            [[[3.0, 3.0]], [[1.0, 1.0]], [[3.0, 3.0]]], dtype=torch.float32
        ),
        maplet_geometry_valid=torch.tensor([[True], [True], [False]]),
        image_width=7,
        image_height=7,
        temperature=0.1,
        minimum_support_fraction=1.0,
    )
    assert summaries["mean"][0] > summaries["mean"][1] + 1.0
    assert summaries["mean"][2].item() == 0.0
    assert active.tolist() == [1, 1, 0]
    assert usable.tolist() == [1, 1, 1]


def test_affine_maplet_support_image_mixture_preserves_unknown_view_mass() -> None:
    query = torch.eye(49, dtype=torch.float32).reshape(7, 7, 49)
    patch = query[2:5, 2:5].permute(2, 0, 1)
    summaries, _active, _usable = score_affine_warped_maplet_patch_likelihoods(
        query_grid=query,
        support_patches=torch.stack([patch, patch]),
        support_patch_valid=torch.ones((2, 3, 3), dtype=torch.bool),
        support_image_sizes=torch.tensor([[7.0, 7.0], [7.0, 7.0]], dtype=torch.float32),
        support_grid_size=7,
        affine_matrices=torch.eye(2, dtype=torch.float32)
        .reshape(1, 1, 2, 2)
        .repeat(2, 2, 1, 1),
        projected_anchor_xy=torch.tensor(
            [[[3.0, 3.0], [1.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]],
            dtype=torch.float32,
        ),
        maplet_geometry_valid=torch.tensor([[True, False], [True, False]]),
        image_width=7,
        image_height=7,
        temperature=0.1,
        minimum_support_fraction=1.0,
        maplet_support_image_groups=torch.tensor([0, 1]),
        support_image_count=2,
        fixed_support_image_priors=torch.tensor([0.25, 0.75]),
    )
    # The second image is unavailable for both poses. Its neutral likelihood
    # remains in the mixture and therefore cannot be silently renormalized.
    assert summaries["support_image_uniform_mixture"][0] > 0.0
    assert summaries["support_image_prior_mixture"][0] > 0.0
    assert torch.allclose(
        summaries["support_image_uniform_mixture"][0:1],
        summaries["support_image_uniform_mixture"][1:2],
        atol=1e-6,
    )


def test_fixed_support_image_aggregation_keeps_missing_image_neutral() -> None:
    summaries = aggregate_fixed_support_image_log_ratios(
        torch.tensor([[2.0], [0.0]], dtype=torch.float32),
        maplet_support_image_groups=torch.tensor([0]),
        support_image_count=2,
        fixed_support_image_priors=torch.tensor([0.25, 0.75]),
    )
    assert torch.allclose(
        summaries["uniform_mixture"],
        torch.logsumexp(torch.tensor([[2.0, 0.0], [0.0, 0.0]]), dim=1)
        - float(np.log(2.0)),
        atol=1e-6,
    )
    assert torch.allclose(
        summaries["prior_mixture"],
        torch.logsumexp(
            torch.log(torch.tensor([[0.25, 0.75]]))
            + torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
            dim=1,
        ),
        atol=1e-6,
    )


def test_rank_band_support_selection_keeps_fixed_rank_diversity() -> None:
    probabilities = np.asarray([[0.40, 0.30, 0.20, 0.10]], dtype=np.float32)
    valid = np.ones((1, 4, 1), dtype=bool)
    image_ids = np.asarray([[["r1"], ["r1"], ["r2"], ["r3"]]], dtype=np.str_)
    selected, scores = select_rank_band_support_images(
        candidate_probabilities=probabilities,
        candidate_view_valid=valid,
        candidate_support_image_ids=image_ids,
        rank_band_ends=(2, 4),
        images_per_rank_band=(1, 2),
    )
    assert selected == ("r1", "r2", "r3")
    assert set(scores) == {"r1", "r2", "r3"}
    assert np.isclose(scores["r1"], 0.7)
    assert np.isclose(scores["r2"], 0.2)
    assert np.isclose(scores["r3"], 0.1)


def test_score_artifact_string_columns_do_not_truncate_ids_or_labels() -> None:
    values = _constant_string_column("seq1/frame00051.png", 2)
    assert values.tolist() == ["seq1/frame00051.png", "seq1/frame00051.png"]
    labels = _constant_string_column("grouped_candidate_pool__policy", 1)
    assert labels.tolist() == ["grouped_candidate_pool__policy"]


def test_spatial_source_loader_materializes_arrays_before_npz_is_closed(tmp_path) -> None:
    path = tmp_path / "spatial.npz"
    np.savez_compressed(
        path,
        image_ids=np.asarray(["support"]),
        image_sizes=np.asarray([[2, 2]], dtype=np.int64),
        grid2_descriptors=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "synthetic_spatial_v1",
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "render": False,
                }
            )
        ),
    )
    source = _load_spatial_source(
        path,
        name="synthetic",
        expected_format="synthetic_spatial_v1",
        grid_size=2,
    )
    grid, image_size = source.image_grid("support")
    assert image_size.tolist() == [2, 2]
    assert np.allclose(grid.reshape(4, 4), np.eye(4), atol=1e-6)


def test_frozen_layout_rejects_track_groups_that_cross_queries() -> None:
    kwargs = dict(
        query_ids=np.asarray(["q0", "q1"]),
        split_names=np.asarray(["validation", "test"]),
        query_observation_offsets=np.asarray([0, 1, 2]),
        query_track_offsets=np.asarray([0, 1, 2]),
        track_observation_offsets=np.asarray([0, 2, 2]),
        observation_track_ids=np.asarray([3, 3]),
        observation_xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        support_image_ids=np.asarray(["s0", "s1"]),
        support_xy=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        support_image_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        support_reprojection_errors=np.asarray([0.0, 0.0], dtype=np.float32),
        metadata={},
    )
    try:
        FrozenSupportAlignmentLayout(**kwargs)
    except ValueError as error:
        assert "offset" in str(error)
    else:  # pragma: no cover
        raise AssertionError("layout accepted a track group that crosses query boundaries")
