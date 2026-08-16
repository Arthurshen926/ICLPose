import numpy as np
import pytest
from scipy import sparse

from feature_extract.vfm.localization_goal_maplet.child_loss_diagnostics import (
    evaluate_child_loss_decomposition,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    PhysicalIncidence,
)
from test_goal_maplet_pure_retrieval import _physical, _result


def test_child_loss_decomposition_separates_budget_and_feature_support():
    physical = _physical()
    incidence = PhysicalIncidence.from_physical_map(physical)
    correct_child = int(incidence.primitive_to_child.getrow(0).indices[0])
    wrong_child = (correct_child + 1) % int(physical.child_parent_rows.size)
    base = _result(physical)
    token_child_rows = np.full((4, 1), correct_child, dtype=np.int64)
    retrieval = PureRadioPhysicalRetrieval(
        **{
            **base.__dict__,
            "token_child_rows": token_child_rows,
            "token_child_probabilities": np.full((4, 1), 0.8, dtype=np.float32),
            "scene_child_rows": np.asarray([wrong_child], dtype=np.int64),
            "scene_child_scores": np.asarray([0.7], dtype=np.float32),
        }
    )
    token_primitive = sparse.csr_matrix(
        (
            np.ones(4, dtype=np.float64),
            (np.arange(4, dtype=np.int64), np.zeros(4, dtype=np.int64)),
        ),
        shape=(4, physical.primitive_ids.size),
    )
    supported = np.ones((physical.primitive_ids.size,), dtype=bool)
    supported[0] = False
    report = evaluate_child_loss_decomposition(
        retrieval,
        token_primitive,
        incidence.primitive_to_parent,
        incidence.primitive_to_child,
        physical,
        canonical_supported_primitive=supported,
        current_surface_metrics={
            "exact_visible_mass_recall": 0.0,
            "tolerant_visible_mass_recall_0.5m": 1.0,
        },
        area_budgets=(1.0,),
    )
    attribution = report["attribution"]
    assert attribution["C0_visible_mass_without_canonical_feature"] == 1.0
    assert (
        attribution["C4_downstream_ranking_budget_and_set_construction_residual"]
        == 1.0
    )
    assert attribution["C5_boundary_or_wrong_scale_credit_within_0.5m"] == 1.0
    assert report["token_candidate_union_exact_recall_ceiling"] == 1.0
    oracle = report["area_curves"]["area_1.00"][
        "current_parent_oracle_all_children"
    ]
    assert oracle["exact_visible_mass_recall"] == pytest.approx(1.0)
    matched = report["area_curves"]["area_1.00"][
        "matched_candidate_gt_budget_oracle"
    ]
    assert matched["exact_visible_mass_recall"] == pytest.approx(1.0)
    assert report["claim_scope"]["ground_truth_not_returned_to_retrieval"] is True
