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


def test_non_anchor_node_permutation_preserves_candidate_output() -> None:
    torch.manual_seed(41)
    supervised = _batch()
    original = CandidateMapletBatch(
        query_features=supervised.query_features[:1],
        query_mask=supervised.query_mask[:1],
        support_features=supervised.support_features[:1],
        support_mask=supervised.support_mask[:1],
        static_features=supervised.static_features[:1],
        target_track_indices=None,
        candidate_labels=None,
        edge_indices=supervised.edge_indices[:1],
    )
    query_order = torch.tensor([0, 2, 1])
    support_order = torch.tensor([0, 2, 1])
    permuted = CandidateMapletBatch(
        query_features=original.query_features[:, query_order],
        query_mask=original.query_mask[:, query_order],
        support_features=original.support_features[:, support_order],
        support_mask=original.support_mask[:, support_order],
        static_features=original.static_features,
        target_track_indices=None,
        candidate_labels=None,
        edge_indices=original.edge_indices,
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
    ).eval()

    with torch.no_grad():
        expected = model(original)
        actual = model(permuted)

    torch.testing.assert_close(actual["candidate_logits"], expected["candidate_logits"])
    torch.testing.assert_close(
        actual["candidate_embeddings"], expected["candidate_embeddings"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        actual["pair_logits"][:, 0, 0], expected["pair_logits"][:, 0, 0]
    )


def test_support_view_permutation_preserves_marginal_embedding() -> None:
    torch.manual_seed(43)
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
    ).eval()
    values = torch.randn(3, 4, 16)
    order = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        expected_embedding, expected_weights = model.aggregate_candidate_views(values)
        actual_embedding, actual_weights = model.aggregate_candidate_views(values[:, order])

    torch.testing.assert_close(actual_embedding, expected_embedding, atol=1e-6, rtol=1e-6)
    inverse = torch.argsort(order)
    torch.testing.assert_close(
        actual_weights[:, inverse], expected_weights, atol=1e-6, rtol=1e-6
    )


def test_candidate_rank_permutation_is_equivariant_and_preserves_dustbin() -> None:
    torch.manual_seed(47)
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
    ).eval()
    values = torch.randn(2, 5, 16)
    prior = torch.rand(2, 5)
    order = torch.tensor([3, 1, 4, 0, 2])

    with torch.no_grad():
        expected = model.resolve_candidate_sets(values, candidate_prior_scores=prior)
        actual = model.resolve_candidate_sets(
            values[:, order], candidate_prior_scores=prior[:, order]
        )

    inverse = torch.argsort(order)
    torch.testing.assert_close(
        actual["candidate_logits"][:, inverse],
        expected["candidate_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        actual["dustbin_logits"], expected["dustbin_logits"], atol=1e-6, rtol=1e-6
    )


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


def test_decoupled_candidate_heads_do_not_reuse_identity_logits() -> None:
    torch.manual_seed(23)
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
            decoupled_candidate_heads=True,
        )
    )
    embeddings = torch.randn(2, 4, 16)
    prior = torch.rand(2, 4)
    before = model.resolve_candidate_sets(
        embeddings, candidate_prior_scores=prior
    )
    assert model.geometry_base_head is not None
    with torch.no_grad():
        model.geometry_base_head.weight.fill_(3.0)
        model.geometry_base_head.bias.fill_(-2.0)
    after = model.resolve_candidate_sets(
        embeddings, candidate_prior_scores=prior
    )
    torch.testing.assert_close(after["candidate_logits"], before["candidate_logits"])
    probabilities = torch.sigmoid(after["geometry_validity_logits"])
    assert torch.all(probabilities[:, :, 0] <= probabilities[:, :, 1])
    assert torch.all(probabilities[:, :, 1] <= probabilities[:, :, 2])

    geometry_loss = after["geometry_validity_logits"].sum()
    geometry_loss.backward()
    assert model.geometry_base_head.weight.grad is not None
    assert model.set_candidate_head.weight.grad is None


def test_candidate_view_marginalization_preserves_prior_and_view_order() -> None:
    torch.manual_seed(29)
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
            decoupled_candidate_heads=True,
            candidate_view_marginalization_enabled=True,
        )
    ).eval()
    views = torch.randn(2, 5, 3, 16)
    view_probability = torch.softmax(torch.randn(2, 5, 3), dim=2)
    prior = torch.rand(2, 5)
    original = model.resolve_candidate_view_sets(
        views, view_probability, candidate_prior_scores=prior
    )
    expected = model.resolve_candidate_sets(
        torch.sum(views * view_probability.unsqueeze(3), dim=2),
        candidate_prior_scores=prior,
    )
    torch.testing.assert_close(original["candidate_logits"], expected["candidate_logits"])

    assert model.candidate_view_head is not None
    with torch.no_grad():
        torch.nn.init.normal_(model.candidate_view_head.weight)
    permutation = torch.tensor([2, 0, 1])
    expected_permuted = model.resolve_candidate_view_sets(
        views, view_probability, candidate_prior_scores=prior
    )
    actual_permuted = model.resolve_candidate_view_sets(
        views[:, :, permutation],
        view_probability[:, :, permutation],
        candidate_prior_scores=prior,
    )
    torch.testing.assert_close(
        actual_permuted["candidate_logits"],
        expected_permuted["candidate_logits"],
        atol=1e-6,
        rtol=1e-6,
    )


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
