import pytest
import torch

from feature_extract.tools.vfm.train_candidate_maplet_matcher import (
    _relative_pose_risk,
    _relative_pose_risk_rank_key,
    _validation_strategy_names,
)
from feature_extract.vfm.localization.candidate_maplet_matcher import (
    CandidateMapletBatch,
    CandidateMapletMatcher,
    CandidateMapletMatcherConfig,
    candidate_maplet_assignment_loss,
    candidate_maplet_group_loss,
    conditional_set_identity_loss,
    factorized_candidate_posterior,
    factorized_top_l_availability_loss,
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


def test_training_pose_risk_ranks_failed_pnp_below_finite_trials() -> None:
    baseline = {
        "median_translation_m_success": 0.3,
        "p90_translation_m_success": 0.7,
        "median_rotation_deg_success": 0.4,
    }
    failed = _relative_pose_risk(
        {
            "median_translation_m_success": None,
            "p90_translation_m_success": None,
            "median_rotation_deg_success": None,
        },
        baseline,
    )
    finite = _relative_pose_risk(
        {
            "median_translation_m_success": 0.4,
            "p90_translation_m_success": 0.8,
            "median_rotation_deg_success": 0.5,
        },
        baseline,
    )

    assert failed["valid"] is False
    assert failed["failure_reason"] == "missing_pose_error_metric"
    assert _relative_pose_risk_rank_key(failed) < _relative_pose_risk_rank_key(
        finite
    )


def test_full_view_identity_mixture_does_not_require_geometry_heads() -> None:
    config = CandidateMapletMatcherConfig(
        query_input_dim=8,
        support_input_dim=9,
        static_input_dim=4,
        descriptor_dim=4,
        model_dim=16,
        num_heads=4,
        decoupled_candidate_heads=True,
        candidate_view_marginalization_enabled=True,
        identity_conditioned_view_posterior_enabled=True,
        full_candidate_view_mixture_enabled=True,
        geometry_validity_enabled=False,
    )
    model = CandidateMapletMatcher(config)

    assert model.geometry_logit_gap_head is None
    assert model.candidate_view_head is not None
    assert len(model.candidate_view_set_blocks) == config.candidate_set_layers

    batch = _batch()
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
    )
    assert torch.isfinite(loss)
    assert "geometry_validity_loss" not in metrics
    assert "rescue_policy_loss" not in metrics


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


def test_batched_assignment_probabilities_match_ragged_contract() -> None:
    torch.manual_seed(40)
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
    ragged = model(batch)
    batched = model(batch, return_ragged_query_probabilities=False)

    assert batched["query_log_probabilities"] is None
    assert torch.allclose(
        ragged["query_log_probabilities_batched"],
        batched["query_log_probabilities_batched"],
        atol=1e-7,
        rtol=1e-7,
    )
    for row, values in enumerate(ragged["query_log_probabilities"]):
        query_count = int(batch.query_mask[row].sum().item())
        support_count = int(batch.support_mask[row].sum().item())
        expected = torch.cat(
            [
                batched["query_log_probabilities_batched"][
                    row, :query_count, :support_count
                ],
                batched["query_log_probabilities_batched"][row, :query_count, -1:],
            ],
            dim=1,
        )
        assert torch.allclose(values, expected, atol=1e-7, rtol=1e-7)

    ragged_loss, ragged_metrics = candidate_maplet_assignment_loss(ragged, batch)
    batched_loss, batched_metrics = candidate_maplet_assignment_loss(batched, batch)
    assert torch.allclose(ragged_loss, batched_loss, atol=1e-7, rtol=1e-7)
    assert ragged_metrics == batched_metrics


def test_prior_free_set_identity_does_not_receive_static_features() -> None:
    torch.manual_seed(41)
    original = _batch()
    changed_static = CandidateMapletBatch(
        query_features=original.query_features,
        query_mask=original.query_mask,
        support_features=original.support_features,
        support_mask=original.support_mask,
        static_features=original.static_features + 10.0,
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
            prior_free_set_identity_enabled=True,
        )
    ).eval()

    with torch.no_grad():
        original_output = model(original)
        changed_output = model(changed_static)

    assert not torch.allclose(
        original_output["candidate_embeddings"],
        changed_output["candidate_embeddings"],
    )
    torch.testing.assert_close(
        original_output["set_identity_candidate_embeddings"],
        changed_output["set_identity_candidate_embeddings"],
        atol=0.0,
        rtol=0.0,
    )


def test_deployable_identity_context_excludes_prior_fields_but_uses_context() -> None:
    torch.manual_seed(42)
    original = _batch()

    def with_static(static_features: torch.Tensor) -> CandidateMapletBatch:
        return CandidateMapletBatch(
            query_features=original.query_features,
            query_mask=original.query_mask,
            support_features=original.support_features,
            support_mask=original.support_mask,
            static_features=static_features,
            target_track_indices=original.target_track_indices,
            candidate_labels=original.candidate_labels,
            edge_indices=original.edge_indices,
            anchor_residuals_px=original.anchor_residuals_px,
            candidate_visible=original.candidate_visible,
        )

    changed_prior = original.static_features.clone()
    changed_prior[:, :2] += 10.0
    changed_context = original.static_features.clone()
    changed_context[:, 2:] += 10.0
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
            prior_free_set_identity_enabled=True,
            deployable_identity_context_enabled=True,
            deployable_identity_static_start_index=2,
        )
    ).eval()

    with torch.no_grad():
        expected = model(original)["set_identity_candidate_embeddings"]
        prior_changed = model(with_static(changed_prior))[
            "set_identity_candidate_embeddings"
        ]
        context_changed = model(with_static(changed_context))[
            "set_identity_candidate_embeddings"
        ]

    torch.testing.assert_close(prior_changed, expected, atol=0.0, rtol=0.0)
    assert not torch.allclose(context_changed, expected)
    assert model.prior_free_set_identity_encoder is not None
    first_layer = model.prior_free_set_identity_encoder[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 16 * 4 + 2 + 4


def test_deployable_identity_context_requires_prior_free_identity() -> None:
    with pytest.raises(ValueError, match="requires prior-free set identity"):
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            deployable_identity_context_enabled=True,
            deployable_identity_static_start_index=2,
        )

    with pytest.raises(ValueError, match="outside static input"):
        CandidateMapletMatcherConfig(
            query_input_dim=8,
            support_input_dim=9,
            static_input_dim=4,
            descriptor_dim=4,
            model_dim=16,
            num_heads=4,
            prior_free_set_identity_enabled=True,
            deployable_identity_context_enabled=True,
            deployable_identity_static_start_index=5,
        )


def test_prior_free_set_identity_receives_set_loss_gradients() -> None:
    torch.manual_seed(43)
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
            prior_free_set_identity_enabled=True,
        )
    )
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
        assignment_weight=0.0,
        pair_weight=0.0,
        candidate_aux_weight=0.0,
        candidate_set_weight=1.0,
        prior_free_identity_weight=1.0,
    )
    loss.backward()

    assert model.prior_free_set_identity_encoder is not None
    first_layer = model.prior_free_set_identity_encoder[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.weight.grad is not None
    assert torch.any(first_layer.weight.grad != 0)
    assert "prior_free_identity_loss" in metrics


def test_candidate_evidence_logits_exclude_explicit_coarse_prior() -> None:
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
    embeddings = torch.randn(2, 3, 16)
    first = model.resolve_candidate_sets(
        embeddings, candidate_prior_scores=torch.tensor([[0.8, 0.1, 0.0]] * 2)
    )
    second = model.resolve_candidate_sets(
        embeddings, candidate_prior_scores=torch.tensor([[0.0, 0.1, 0.8]] * 2)
    )

    torch.testing.assert_close(
        first["candidate_evidence_logits"],
        second["candidate_evidence_logits"],
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.allclose(first["candidate_logits"], second["candidate_logits"])


def test_conditional_identity_loss_ignores_null_and_no_match_groups() -> None:
    logits = torch.tensor(
        [[0.0, 2.0, -1.0], [20.0, -20.0, -20.0]], requires_grad=True
    )
    positives = torch.tensor(
        [[False, True, False], [False, False, False]], dtype=torch.bool
    )
    loss, metrics = conditional_set_identity_loss(logits, positives)
    loss.backward()

    expected = torch.logsumexp(torch.tensor([0.0, 2.0, -1.0]), dim=0) - 2.0
    torch.testing.assert_close(loss.detach(), expected)
    assert metrics["conditional_identity_top1_accuracy"] == 1.0
    assert metrics["conditional_identity_group_count"] == 1.0
    torch.testing.assert_close(logits.grad[1], torch.zeros(3))

    no_match_logits = torch.randn(2, 3, requires_grad=True)
    no_match_loss, no_match_metrics = conditional_set_identity_loss(
        no_match_logits, torch.zeros(2, 3, dtype=torch.bool)
    )
    no_match_loss.backward()
    assert no_match_loss.item() == 0.0
    assert no_match_metrics["conditional_identity_group_count"] == 0.0
    torch.testing.assert_close(no_match_logits.grad, torch.zeros_like(no_match_logits))


def test_conditional_identity_weights_do_not_reweight_no_match_groups() -> None:
    logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [20.0, -20.0]], requires_grad=True
    )
    positives = torch.tensor(
        [[True, False], [True, False], [False, False]], dtype=torch.bool
    )
    weights = torch.tensor([3.0, 1.0, 1000.0])

    loss, metrics = conditional_set_identity_loss(
        logits, positives, group_weights=weights
    )
    per_group = torch.stack(
        [
            torch.logsumexp(logits.detach()[0], dim=0) - logits.detach()[0, 0],
            torch.logsumexp(logits.detach()[1], dim=0) - logits.detach()[1, 0],
        ]
    )
    expected = (3.0 * per_group[0] + per_group[1]) / 4.0
    torch.testing.assert_close(loss.detach(), expected)
    assert metrics["conditional_identity_group_weight_sum"] == 4.0
    torch.testing.assert_close(
        torch.tensor(metrics["conditional_identity_unweighted_loss"]),
        torch.mean(per_group),
    )
    loss.backward()
    torch.testing.assert_close(logits.grad[2], torch.zeros(2))


def test_conditional_identity_weights_reject_invalid_values() -> None:
    logits = torch.zeros((2, 2))
    positives = torch.tensor([[True, False], [False, True]])
    with pytest.raises(ValueError, match="one value per candidate group"):
        conditional_set_identity_loss(
            logits, positives, group_weights=torch.ones(3)
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        conditional_set_identity_loss(
            logits, positives, group_weights=torch.tensor([1.0, -1.0])
        )


def test_factorized_top_l_availability_is_independent_of_candidate_softmax() -> None:
    logits = torch.tensor([2.0, -2.0], requires_grad=True)
    positives = torch.tensor(
        [[False, True, False], [False, False, False]], dtype=torch.bool
    )

    loss, metrics = factorized_top_l_availability_loss(logits, positives)
    loss.backward()

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.detach(), torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(loss.detach(), expected)
    assert logits.grad is not None
    assert metrics["factorized_top_l_availability_accuracy"] == 1.0
    assert metrics["factorized_top_l_availability_positive_rate"] == 0.5

    short_candidates, short_null, _ = factorized_candidate_posterior(
        torch.tensor([[0.0, 1.0]]), torch.tensor([0.4])
    )
    long_candidates, long_null, _ = factorized_candidate_posterior(
        torch.tensor([[0.0, 1.0, -1.0, 2.0]]), torch.tensor([0.4])
    )
    torch.testing.assert_close(short_null, long_null)
    torch.testing.assert_close(short_candidates.sum(dim=1), long_candidates.sum(dim=1))
    torch.testing.assert_close(
        short_candidates.sum(dim=1) + short_null, torch.ones_like(short_null)
    )


def test_factorized_set_head_trains_separately_from_conditional_identity() -> None:
    torch.manual_seed(53)
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
            prior_free_set_identity_enabled=True,
            factorized_set_posterior_enabled=True,
        )
    )
    outputs = [model(batch), model(batch)]
    loss, metrics = candidate_maplet_group_loss(
        model,
        outputs,
        [batch, batch],
        candidate_group_size=2,
        assignment_weight=0.0,
        pair_weight=0.0,
        candidate_aux_weight=0.0,
        candidate_set_weight=0.0,
        prior_free_conditional_identity_weight=1.0,
        factorized_top_l_availability_weight=1.0,
    )
    loss.backward()

    assert model.set_top_l_availability_head is not None
    first_layer = model.set_top_l_availability_head[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.weight.grad is not None
    assert torch.any(first_layer.weight.grad != 0)
    assert "factorized_top_l_availability_loss" in metrics


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


def test_full_candidate_view_mixture_is_candidate_and_view_equivariant() -> None:
    torch.manual_seed(83)
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
            full_candidate_view_mixture_enabled=True,
            identity_conditioned_view_posterior_enabled=True,
        )
    ).eval()
    views = torch.randn(2, 5, 3, 16)
    view_probability = torch.softmax(torch.randn(2, 5, 3), dim=2)
    prior = torch.rand(2, 5)
    candidate_order = torch.tensor([3, 1, 4, 0, 2])
    view_order = torch.tensor([2, 0, 1])
    assert model.candidate_view_head is not None
    with torch.no_grad():
        torch.nn.init.normal_(model.candidate_view_head.weight)

    with torch.no_grad():
        original = model.resolve_candidate_view_sets(
            views, view_probability, candidate_prior_scores=prior
        )
        permuted = model.resolve_candidate_view_sets(
            views[:, candidate_order][:, :, view_order],
            view_probability[:, candidate_order][:, :, view_order],
            candidate_prior_scores=prior[:, candidate_order],
        )

    candidate_inverse = torch.argsort(candidate_order)
    torch.testing.assert_close(
        permuted["candidate_logits"][:, candidate_inverse],
        original["candidate_logits"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        permuted["dustbin_logits"],
        original["dustbin_logits"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        permuted["support_view_probabilities"][:, candidate_inverse][:, :, view_order.argsort()],
        original["support_view_probabilities"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["support_view_probabilities"].sum(dim=2),
        torch.ones_like(original["support_view_probabilities"][:, :, 0]),
    )
    assert not torch.allclose(
        original["support_view_probabilities"], view_probability
    )


def test_explicit_anchor_role_keeps_nonanchor_permutation_invariant() -> None:
    torch.manual_seed(89)
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
            explicit_anchor_role_embedding=True,
        )
    ).eval()

    with torch.no_grad():
        expected = model(original)
        actual = model(permuted)

    torch.testing.assert_close(
        actual["candidate_logits"], expected["candidate_logits"], atol=1e-6, rtol=1e-6
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
