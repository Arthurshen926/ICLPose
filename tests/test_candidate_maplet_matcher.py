import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _validation_strategy_names,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletBatch,
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
    candidate_maplet_assignment_loss,
    candidate_maplet_group_loss,
    log_optimal_transport_batched,
    rescue_policy_targets,
    set_valued_candidate_loss,
)
from feature_extract.vfm.localization.local_assignment_matcher import log_optimal_transport


def test_rescue_validation_strategy_is_added_before_training() -> None:
    assert _validation_strategy_names(
        "geometry_p05px_prior_row_confidence",
        rescue_policy_enabled=True,
    ) == ("geometry_p05px_prior_row_confidence", "rescue_policy_resolved")
    assert _validation_strategy_names(
        "rescue_policy_resolved,rescue_policy_resolved",
        rescue_policy_enabled=True,
    ) == ("rescue_policy_resolved",)


def _batch() -> CandidateMapletBatch:
    return CandidateMapletBatch(
        query_features=torch.randn(2, 3, 8),
        query_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        support_features=torch.randn(2, 3, 9),
        support_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        static_features=torch.randn(2, 4),
        target_track_indices=torch.tensor([[0, 1, 3], [2, 0, -1]], dtype=torch.long),
        candidate_labels=torch.tensor([1.0, 0.0]),
        edge_indices=torch.tensor([10, 11], dtype=torch.long),
        anchor_residuals_px=torch.tensor([1.5, float("inf")]),
        candidate_visible=torch.tensor([True, False]),
    )


def test_candidate_maplet_matcher_outputs_partial_assignment_and_dustbin() -> None:
    torch.manual_seed(4)
    batch = _batch()
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
        )
    )
    output = model(batch)
    loss, metrics = candidate_maplet_assignment_loss(output, batch)
    loss.backward()

    assert len(output["query_log_probabilities"]) == 2
    assert output["query_log_probabilities"][0].shape == (3, 4)
    assert output["query_log_probabilities"][1].shape == (2, 3)
    assert output["candidate_logits"].shape == (2,)
    assert output["pair_mask"].sum().item() == 13
    assert torch.isfinite(loss)
    assert metrics["matched_query_rate"] == 0.6
    assert model.prior_log_scale.grad is not None


def test_candidate_maplet_matcher_inference_does_not_require_supervision() -> None:
    supervised = _batch()
    batch = CandidateMapletBatch(
        query_features=supervised.query_features,
        query_mask=supervised.query_mask,
        support_features=supervised.support_features,
        support_mask=supervised.support_mask,
        static_features=supervised.static_features,
        target_track_indices=None,
        candidate_labels=None,
        edge_indices=supervised.edge_indices,
    )
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
        )
    )

    output = model(batch)

    assert output["candidate_logits"].shape == (2,)
    assert len(output["query_log_probabilities"]) == 2


def test_candidate_maplet_batch_requires_anchor_nodes() -> None:
    batch = _batch()
    invalid = CandidateMapletBatch(
        query_features=batch.query_features,
        query_mask=torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=torch.bool),
        support_features=batch.support_features,
        support_mask=batch.support_mask,
        static_features=batch.static_features,
        target_track_indices=batch.target_track_indices,
        candidate_labels=batch.candidate_labels,
        edge_indices=batch.edge_indices,
    )
    try:
        invalid.validate()
    except ValueError as error:
        assert "anchor" in str(error)
    else:
        raise AssertionError("invalid anchor mask was accepted")


def test_batched_transport_matches_scalar_without_padding() -> None:
    scores = torch.tensor([[[2.0, -1.0, 0.5], [0.2, 1.2, -0.4]]])
    query_bin = torch.tensor([[0.1, -0.2]])
    support_bin = torch.tensor([[0.0, 0.3, -0.1]])
    corner = torch.tensor(0.2)
    batched = log_optimal_transport_batched(
        scores,
        query_bin,
        support_bin,
        corner,
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones((1, 3), dtype=torch.bool),
        iterations=20,
    )[0]
    scalar = log_optimal_transport(
        scores[0],
        query_bin[0],
        support_bin[0],
        corner,
        iterations=20,
    )
    torch.testing.assert_close(batched, scalar, atol=1e-5, rtol=1e-5)


def test_set_valued_candidate_loss_learns_multi_positive_and_no_match_rows() -> None:
    logits = torch.tensor([[3.0, 2.0, -1.0], [0.0, -1.0, -2.0]], requires_grad=True)
    dustbin = torch.tensor([-2.0, 3.0], requires_grad=True)
    positives = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

    loss, metrics = set_valued_candidate_loss(logits, dustbin, positives)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["candidate_set_accuracy"] == 1.0
    assert metrics["candidate_set_top1_accuracy_mappable"] == 1.0
    assert metrics["candidate_set_no_match_accuracy"] == 1.0
    assert logits.grad is not None
    assert dustbin.grad is not None


def test_group_loss_aggregates_views_and_requires_complete_candidate_rows() -> None:
    torch.manual_seed(9)
    batch = _batch()
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
            candidate_set_layers=1,
        )
    )
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["candidate_set_mappable_rate"] == 1.0
    assert 0.0 <= metrics["support_view_max_probability"] <= 1.0
    assert model.set_candidate_head.weight.grad is not None
    assert model.support_view_head[-1].weight.grad is not None


def test_candidate_set_residual_initially_preserves_prior_ranking() -> None:
    torch.manual_seed(13)
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
            candidate_prior_index=1,
            candidate_prior_scale=20.0,
        )
    )
    embeddings = torch.randn(2, 3, 16)
    prior = torch.tensor([[0.6, 0.8, 0.7], [0.4, 0.3, 0.9]])

    resolved = model.resolve_candidate_sets(
        embeddings, candidate_prior_scores=prior
    )

    assert torch.equal(
        torch.argmax(resolved["candidate_logits"], dim=1), torch.argmax(prior, dim=1)
    )
    expected = 20.0 * (prior - torch.mean(prior, dim=1, keepdim=True))
    torch.testing.assert_close(resolved["candidate_logits"], expected, atol=1e-5, rtol=1e-5)


def test_static_normalization_is_configured_but_not_checkpoint_state() -> None:
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
            static_feature_mean=(1.0, 2.0, 3.0, 4.0),
            static_feature_scale=(2.0, 4.0, 5.0, 8.0),
        )
    )
    torch.testing.assert_close(model.static_feature_mean, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(model.static_feature_scale, torch.tensor([2.0, 4.0, 5.0, 8.0]))
    assert "static_feature_mean" not in model.state_dict()
    assert "static_feature_scale" not in model.state_dict()


def test_geometry_validity_heads_are_monotonic_and_train_on_real_residuals() -> None:
    torch.manual_seed(21)
    batch = _batch()
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
            geometry_validity_enabled=True,
        )
    )
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
        geometry_validity_weight=1.0,
        candidate_visibility_weight=0.25,
    )
    loss.backward()

    with torch.no_grad():
        embeddings = torch.stack(
            [output["candidate_embeddings"] for output in outputs], dim=1
        )
        aggregated, _weights = model.aggregate_candidate_views(embeddings)
        resolved = model.resolve_candidate_sets(aggregated.reshape(1, 2, 16))
        probabilities = torch.sigmoid(resolved["geometry_validity_logits"])
    assert torch.all(probabilities[:, :, 0] <= probabilities[:, :, 1])
    assert torch.all(probabilities[:, :, 1] <= probabilities[:, :, 2])
    assert metrics["geometry_monotonic_violation_rate"] == 0.0
    assert metrics["geometry_p02px_positive_rate"] == 0.5
    assert model.geometry_logit_gap_head is not None
    assert model.geometry_logit_gap_head.weight.grad is not None
    assert model.candidate_visibility_head is not None
    assert model.candidate_visibility_head.weight.grad is not None


def test_rescue_targets_only_replace_an_invalid_baseline_with_a_valid_candidate() -> None:
    residuals = torch.tensor(
        [[1.0, 8.0, 4.0], [1.0, 2.0, 9.0]], dtype=torch.float32
    )
    targets = rescue_policy_targets(
        residuals,
        torch.tensor([1, 0]),
        candidate_threshold_px=5.0,
        baseline_invalid_threshold_px=5.0,
    )
    assert torch.equal(
        targets,
        torch.tensor([[True, False, True], [False, False, False]]),
    )


def test_rescue_policy_head_learns_keep_or_switch_from_real_residuals() -> None:
    torch.manual_seed(31)
    original = _batch()
    batch = CandidateMapletBatch(
        query_features=original.query_features,
        query_mask=original.query_mask,
        support_features=original.support_features,
        support_mask=original.support_mask,
        static_features=torch.tensor(
            [[0.0, 0.2, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        target_track_indices=original.target_track_indices,
        candidate_labels=original.candidate_labels,
        edge_indices=original.edge_indices,
        anchor_residuals_px=original.anchor_residuals_px,
        candidate_visible=original.candidate_visible,
    )
    model = CandidateMapletMatcher(
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            layers=1,
            dropout=0.0,
            sinkhorn_iterations=5,
            candidate_prior_index=1,
            rescue_policy_enabled=True,
        )
    )
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
        rescue_policy_weight=1.0,
    )
    loss.backward()

    with torch.no_grad():
        embeddings = torch.stack(
            [output["candidate_embeddings"] for output in outputs], dim=1
        )
        aggregated, _weights = model.aggregate_candidate_views(embeddings)
        resolved = model.resolve_candidate_sets(
            aggregated.reshape(1, 2, 16),
            candidate_prior_scores=torch.tensor([[0.2, 0.8]]),
        )
    assert torch.equal(resolved["baseline_candidate_indices"], torch.tensor([1]))
    assert resolved["rescue_candidate_logits"].shape == (1, 2)
    assert resolved["rescue_keep_logits"].shape == (1,)
    assert metrics["rescue_group_positive_rate"] == 1.0
    assert model.rescue_candidate_head is not None
    assert model.rescue_candidate_head[-1].weight.grad is not None
    assert model.rescue_keep_head is not None
    assert model.rescue_keep_head[-1].weight.grad is not None
