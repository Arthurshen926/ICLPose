import torch

from feature_extract.vfm.localization.whole_image_candidate_graph import (
    WholeImageCandidateGraph,
    WholeImageLatentCandidateGraph,
    set_identity_nll,
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
