import pytest
import torch

from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind
from feature_extract.vfm.reporting import build_gate_table, ensure_protocols_not_mixed
from feature_extract.vfm.selector import LocalizableFeatureSelector


def test_selector_outputs_compact_feature_utility_and_uncertainty():
    selector = LocalizableFeatureSelector(input_dim=8, output_dim=4, group_size=2)
    tokens = torch.randn(2, 8, 5, 6, requires_grad=True)

    output = selector(tokens)
    loss = output.selected.pow(2).mean() + output.utility.mean() + output.uncertainty.mean()
    loss.backward()

    assert output.selected.shape == (2, 4, 5, 6)
    assert output.utility.shape == (2, 1, 5, 6)
    assert output.uncertainty.shape == (2, 1, 5, 6)
    assert output.channel_gates.shape == (4,)
    assert tokens.grad is not None


def test_reporting_refuses_to_merge_controlled_and_real_protocols():
    controlled = EvaluationProtocol(
        name="oldhospital_controlled_lattice_val",
        kind=ProtocolKind.CONTROLLED_LATTICE,
        split="val",
        candidate_generator="gt_centered_local_lattice",
        allowed_training_inputs=("query_tokens", "candidate_tokens"),
        candidate_uses_gt=True,
        solver_conditioned=False,
    )
    real = EvaluationProtocol(
        name="oldhospital_real_top20_test",
        kind=ProtocolKind.REAL_RETRIEVAL,
        split="test",
        candidate_generator="hloc_retrieval_top20",
        allowed_training_inputs=("query_tokens", "candidate_tokens", "candidate_prior"),
        candidate_uses_gt=False,
        solver_conditioned=False,
    )

    with pytest.raises(ValueError, match="protocol kinds"):
        ensure_protocols_not_mixed([controlled, real])


def test_gate_table_uses_field_standard_metric_names():
    table = build_gate_table(
        gate_name="Feature Utility",
        rows=[
            {
                "method": "selected_feature",
                "pred_cost_m": 0.19,
                "top1_acc": 0.78,
                "spearman": 0.58,
                "basin_recall_at_5": 0.91,
            }
        ],
    )

    assert "Feature Utility" in table
    assert "pred_cost_m" in table
    assert "basin_recall_at_5" in table
