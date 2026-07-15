import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_whole_image_candidate_graph import (
    POSE_RELATION_FEATURE_NAMES,
    RELATION_FEATURE_NAMES,
    build_explicit_relation_graph,
    relation_feature_names,
)
from feature_extract.vfm.local_maplet_matching import (
    LocalMapletBank,
    LocalMapletSupportIndex,
)
from feature_extract.vfm.localization.whole_image_candidate_graph import (
    ExplicitCandidateRelationBlock,
    WholeImageCandidateGraph,
    WholeImageLatentCandidateGraph,
    set_identity_nll,
    system_hard_negative_margin_loss,
)


def test_graph_preserves_prior_at_initialization_and_trains_set_null() -> None:
    torch.manual_seed(0)
    model = WholeImageCandidateGraph(6, model_dim=16, heads=4, layers=1, dropout=0.0)
    features = torch.randn(1, 3, 4, 6)
    xy = torch.rand(1, 3, 2)
    neighbors = torch.arange(12).reshape(1, 3, 4, 1).expand(-1, -1, -1, 2)
    prior = torch.rand(1, 3, 4)
    prior = prior / (prior.sum(2, keepdim=True) + 1.0)
    null = 1.0 - prior.sum(2)
    logits, null_logits = model(features, xy, neighbors, prior, null)
    posterior = torch.softmax(torch.cat([logits, null_logits.unsqueeze(2)], 2), 2)
    torch.testing.assert_close(posterior[:, :, :-1], prior, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(posterior[:, :, -1], null, atol=1e-6, rtol=1e-6)
    labels = torch.zeros_like(prior, dtype=torch.bool)
    labels[0, 0, 1] = True
    labels[0, 2, 3] = True
    loss = set_identity_nll(logits, null_logits, labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_graph_is_equivariant_to_query_and_candidate_order() -> None:
    torch.manual_seed(4)
    model = WholeImageCandidateGraph(5, model_dim=16, heads=4, layers=1, dropout=0.0)
    torch.nn.init.normal_(model.candidate_residual_head.weight)
    torch.nn.init.normal_(model.null_residual_head[-1].weight)
    model.eval()
    features = torch.randn(1, 4, 3, 5)
    xy = torch.rand(1, 4, 2)
    prior = torch.softmax(torch.randn(1, 4, 4), dim=2)
    candidate_prior, null_prior = prior[:, :, :3], prior[:, :, 3]
    self_neighbors = torch.arange(12).reshape(1, 4, 3, 1)
    original = model(features, xy, self_neighbors, candidate_prior, null_prior)

    query_permutation = torch.tensor([2, 0, 3, 1])
    candidate_permutation = torch.tensor([1, 2, 0])
    permuted_features = features[:, query_permutation][:, :, candidate_permutation]
    permuted_prior = candidate_prior[:, query_permutation][:, :, candidate_permutation]
    permuted_xy = xy[:, query_permutation]
    # Self-neighbor links are rebuilt after reindexing, as they are in the data builder.
    permuted_neighbors = torch.arange(12).reshape(1, 4, 3, 1)
    permuted = model(
        permuted_features,
        permuted_xy,
        permuted_neighbors,
        permuted_prior,
        null_prior[:, query_permutation],
    )
    expected_candidates = original[0][:, query_permutation][:, :, candidate_permutation]
    expected_null = original[1][:, query_permutation]
    torch.testing.assert_close(permuted[0], expected_candidates, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted[1], expected_null, atol=1e-5, rtol=1e-5)


def test_latent_graph_preserves_prior_and_support_view_permutation() -> None:
    torch.manual_seed(8)
    model = WholeImageLatentCandidateGraph(
        6, 4, model_dim=16, heads=4, layers=1, dropout=0.0
    )
    latents = torch.randn(1, 3, 4, 2, 6)
    scalars = torch.randn(1, 3, 4, 4)
    view_probability = torch.softmax(torch.randn(1, 3, 4, 2), dim=3)
    view_mask = torch.ones_like(view_probability, dtype=torch.bool)
    xy = torch.rand(1, 3, 2)
    neighbors = torch.arange(12).reshape(1, 3, 4, 1)
    prior = torch.softmax(torch.randn(1, 3, 5), dim=2)
    candidate_prior, null_prior = prior[:, :, :4], prior[:, :, 4]

    initialized = model(
        latents,
        scalars,
        view_probability,
        view_mask,
        xy,
        neighbors,
        candidate_prior,
        null_prior,
    )
    posterior = torch.softmax(
        torch.cat([initialized[0], initialized[1].unsqueeze(2)], dim=2), dim=2
    )
    torch.testing.assert_close(posterior[:, :, :4], candidate_prior, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(posterior[:, :, 4], null_prior, atol=1e-6, rtol=1e-6)

    torch.nn.init.normal_(model.view_residual_head.weight)
    torch.nn.init.normal_(model.null_residual_head[-1].weight)
    model.eval()
    original = model(
        latents,
        scalars,
        view_probability,
        view_mask,
        xy,
        neighbors,
        candidate_prior,
        null_prior,
    )
    permutation = torch.tensor([1, 0])
    permuted = model(
        latents[:, :, :, permutation],
        scalars,
        view_probability[:, :, :, permutation],
        view_mask[:, :, :, permutation],
        xy,
        neighbors,
        candidate_prior,
        null_prior,
    )
    torch.testing.assert_close(permuted[0], original[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted[1], original[1], atol=1e-5, rtol=1e-5)


def test_explicit_relation_block_uses_edge_factors_without_changing_shape() -> None:
    torch.manual_seed(13)
    block = ExplicitCandidateRelationBlock(8, 5).eval()
    nodes = torch.randn(1, 2, 3, 8)
    neighbors = torch.arange(6).reshape(1, 2, 3, 1).expand(-1, -1, -1, 2)
    first = torch.zeros(1, 2, 3, 2, 5)
    second = first.clone()
    second[:, :, :, :, 0] = 1.0

    with torch.no_grad():
        output_first = block(nodes, neighbors, first)
        output_second = block(nodes, neighbors, second)

    assert output_first.shape == nodes.shape
    assert not torch.allclose(output_first, output_second)


def test_explicit_relation_block_is_equivariant_to_group_and_candidate_order() -> None:
    torch.manual_seed(14)
    block = ExplicitCandidateRelationBlock(8, 5).eval()
    nodes = torch.randn(1, 3, 2, 8)
    neighbors = torch.tensor(
        [[[[2, 4], [3, 5]], [[0, 4], [1, 5]], [[0, 2], [1, 3]]]]
    )
    relations = torch.randn(1, 3, 2, 2, 5)
    original = block(nodes, neighbors, relations)

    query_permutation = torch.tensor([2, 0, 1])
    candidate_permutation = torch.tensor([1, 0])
    old_grid = torch.arange(6).reshape(3, 2)
    new_to_old = old_grid[query_permutation][:, candidate_permutation].reshape(-1)
    old_to_new = torch.empty_like(new_to_old)
    old_to_new[new_to_old] = torch.arange(6)
    old_neighbors = neighbors.reshape(6, 2)[new_to_old]
    permuted_neighbors = old_to_new[old_neighbors].reshape(1, 3, 2, 2)
    permuted_relations = relations.reshape(6, 2, 5)[new_to_old].reshape(
        1, 3, 2, 2, 5
    )
    permuted_nodes = nodes[:, query_permutation][:, :, candidate_permutation]

    permuted = block(permuted_nodes, permuted_neighbors, permuted_relations)
    expected = original[:, query_permutation][:, :, candidate_permutation]
    torch.testing.assert_close(permuted, expected, atol=1e-5, rtol=1e-5)


def test_system_hard_negative_loss_rescues_errors_and_preserves_correct_top1() -> None:
    logits = torch.tensor([[[0.2, 0.1, -0.5], [0.8, 0.0, -0.2]]])
    labels = torch.tensor([[[False, True, False], [True, False, False]]])
    prior = torch.tensor([[[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]])

    loss = system_hard_negative_margin_loss(
        logits, labels, prior, margin=0.2, hard_negative_k=2
    )
    improved = system_hard_negative_margin_loss(
        logits + torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]]),
        labels,
        prior,
        margin=0.2,
        hard_negative_k=2,
    )

    assert torch.isfinite(loss)
    assert improved < loss

    broken_preserve = system_hard_negative_margin_loss(
        logits + torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        labels,
        prior,
        margin=0.2,
        hard_negative_k=2,
    )
    protected_preserve = system_hard_negative_margin_loss(
        logits + torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        labels,
        prior,
        margin=0.2,
        hard_negative_k=2,
    )

    assert protected_preserve < broken_preserve


def test_system_hard_negative_loss_rejects_negative_preserve_weight() -> None:
    with pytest.raises(ValueError, match="preserve weight"):
        system_hard_negative_margin_loss(
            torch.zeros((1, 1, 2)),
            torch.tensor([[[True, False]]]),
            torch.tensor([[[0.5, 0.5]]]),
            preserve_weight=-1.0,
        )


def test_explicit_relation_builder_excludes_same_query_and_marks_maplets() -> None:
    tracks = np.asarray([10, 11, 20, 21], dtype=np.int64)
    neighbor_tracks = np.asarray([[20], [21], [10], [11]], dtype=np.int64)
    bank = LocalMapletBank(
        neighbor_indices=np.asarray([[2], [3], [0], [1]], dtype=np.int64),
        context_features=np.ones((4, 2), dtype=np.float32),
        neighbor_counts=np.ones((4,), dtype=np.int64),
        context_radius=np.ones((4,), dtype=np.float32),
        context_feature_variance=np.ones((4,), dtype=np.float32),
        covisibility_strength=np.asarray([2.0, 1.0, 2.0, 1.0], dtype=np.float32),
        xyz_cov_eigvals=np.ones((4, 3), dtype=np.float32),
        neighbor_idf_mean=np.ones((4,), dtype=np.float32),
        maplet_type="toy",
        maplet_k=1,
    )
    maplets = LocalMapletSupportIndex(
        maplets=bank,
        anchor_track_ids=tracks,
        neighbor_track_ids=neighbor_tracks,
        support_image_ids=(),
        support_image_indices=np.full((4, 1), -1, dtype=np.int64),
        support_coverage_counts=np.zeros((4, 1), dtype=np.int64),
        candidate_k=1,
    )
    xy = np.asarray([[[0.1, 0.2], [0.8, 0.7]]], dtype=np.float32)
    xyz = np.asarray(
        [[[[0.0, 0.0, 5.0], [5.0, 0.0, 5.0]],
          [[0.1, 0.0, 5.0], [5.1, 0.0, 5.0]]]],
        dtype=np.float32,
    )
    rows = np.arange(4, dtype=np.int64).reshape(1, 2, 2)
    track_grid = tracks.reshape(1, 2, 2)

    neighbors, relations = build_explicit_relation_graph(
        xy, xyz, rows, track_grid, maplets, neighbor_k=1
    )

    flat_neighbors = neighbors.reshape(-1)
    assert np.all(flat_neighbors[:2] // 2 == 1)
    assert np.all(flat_neighbors[2:] // 2 == 0)
    source_contains_index = RELATION_FEATURE_NAMES.index(
        "source_maplet_contains_target"
    )
    mutual_index = RELATION_FEATURE_NAMES.index("mutual_maplet_relation")
    assert np.all(relations[..., source_contains_index] == 1.0)
    assert np.all(relations[..., mutual_index] == 1.0)


def test_pose_conditioned_relation_builder_encodes_projected_delta_error() -> None:
    tracks = np.asarray([10, 11, 20, 21], dtype=np.int64)
    neighbor_tracks = np.asarray([[20], [21], [10], [11]], dtype=np.int64)
    bank = LocalMapletBank(
        neighbor_indices=np.asarray([[2], [3], [0], [1]], dtype=np.int64),
        context_features=np.ones((4, 2), dtype=np.float32),
        neighbor_counts=np.ones((4,), dtype=np.int64),
        context_radius=np.ones((4,), dtype=np.float32),
        context_feature_variance=np.ones((4,), dtype=np.float32),
        covisibility_strength=np.ones((4,), dtype=np.float32),
        xyz_cov_eigvals=np.ones((4, 3), dtype=np.float32),
        neighbor_idf_mean=np.ones((4,), dtype=np.float32),
        maplet_type="toy",
        maplet_k=1,
    )
    maplets = LocalMapletSupportIndex(
        maplets=bank,
        anchor_track_ids=tracks,
        neighbor_track_ids=neighbor_tracks,
        support_image_ids=(),
        support_image_indices=np.full((4, 1), -1, dtype=np.int64),
        support_coverage_counts=np.zeros((4, 1), dtype=np.int64),
        candidate_k=1,
    )
    xy = np.asarray([[[0.1, 0.2], [0.8, 0.7]]], dtype=np.float32)
    xyz = np.asarray(
        [[[[0.0, 0.0, 5.0], [5.0, 0.0, 5.0]],
          [[0.1, 0.0, 5.0], [5.1, 0.0, 5.0]]]],
        dtype=np.float32,
    )
    rows = np.arange(4, dtype=np.int64).reshape(1, 2, 2)
    projected = np.asarray(
        [[[[0.1, 0.2], [0.2, 0.2]],
          [[0.8, 0.7], [0.9, 0.7]]]],
        dtype=np.float32,
    )

    _neighbors, relations = build_explicit_relation_graph(
        xy,
        xyz,
        rows,
        tracks.reshape(1, 2, 2),
        maplets,
        neighbor_k=1,
        projected_xy_normalized=projected,
        candidate_in_front=np.ones((1, 2, 2), dtype=bool),
        initial_pose_available=np.ones((1,), dtype=bool),
    )

    names = relation_feature_names(pose_conditioned=True)
    assert names == RELATION_FEATURE_NAMES + POSE_RELATION_FEATURE_NAMES
    error_index = names.index("observed_projected_delta_error")
    source_distance_index = names.index("source_reprojection_distance_log1p")
    assert relations.shape[-1] == len(names)
    np.testing.assert_allclose(relations[0, 0, 0, 0, error_index], 0.0, atol=1e-6)
    assert relations[0, 0, 1, 0, source_distance_index] > 0.0
