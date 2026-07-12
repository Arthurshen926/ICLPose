import pytest
import torch

from feature_extract.vfm.localization.local_assignment_matcher import (
    LocalAssignmentEpisode,
    LocalAssignmentMatcher,
    LocalAssignmentMatcherConfig,
    local_assignment_loss,
    log_optimal_transport,
)


def _episode():
    edge_query = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    edge_track = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)
    return LocalAssignmentEpisode(
        query_features=torch.randn(3, 5),
        track_features=torch.randn(4, 6),
        edge_query_indices=edge_query,
        edge_track_indices=edge_track,
        edge_features=torch.randn(6, 2),
        support_features=torch.randn(6, 3, 3),
        support_mask=torch.tensor([[1, 1, 0]] * 6, dtype=torch.bool),
        target_track_indices=torch.tensor([0, 2, 4], dtype=torch.long),
        query_rows=torch.tensor([10, 11, 12], dtype=torch.long),
        candidate_columns=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    )


def test_partial_assignment_matcher_outputs_dustbin_and_support_posterior():
    torch.manual_seed(3)
    episode = _episode()
    model = LocalAssignmentMatcher(
        LocalAssignmentMatcherConfig(
            query_input_dim=5,
            track_input_dim=6,
            support_input_dim=3,
            edge_input_dim=2,
            model_dim=16,
            num_heads=4,
            dropout=0.0,
            sinkhorn_iterations=10,
        )
    )

    output = model(episode)
    loss, metrics = local_assignment_loss(output, episode)
    loss.backward()

    assert output["query_log_probabilities"].shape == (3, 5)
    assert output["support_weights"].shape == (6, 3)
    torch.testing.assert_close(output["support_weights"][:, 2], torch.zeros(6))
    torch.testing.assert_close(output["support_weights"].sum(dim=1), torch.ones(6))
    assert torch.isfinite(loss)
    assert metrics["dustbin_target_rate"] == pytest.approx(1.0 / 3.0)
    expected_prior = torch.exp(model.edge_prior_log_scale) * (
        episode.edge_features[:, 0] - model.edge_prior_center
    )
    torch.testing.assert_close(output["edge_logits"].detach(), expected_prior.detach())


def test_log_optimal_transport_has_finite_rectangular_dustbin_matrix():
    output = log_optimal_transport(
        torch.tensor([[2.0, -1.0, 0.0], [2.0, 1.0, -2.0]]),
        torch.zeros(2),
        torch.zeros(3),
        torch.tensor(0.0),
        iterations=20,
    )

    assert output.shape == (3, 4)
    assert torch.all(torch.isfinite(output))
