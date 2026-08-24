from __future__ import annotations

import torch

from feature_extract.vfm.localization_goal_maplet.query_typed_geometry import (
    QueryTypedGeometryPredictor,
    append_query_typed_geometry_consistency,
    conjunct_phase_and_typed_geometry_score,
    fixed_denominator_query_typed_geometry_score,
    query_typed_geometry_supervision_loss,
)


def test_query_geometry_is_pose_free_typed_and_differentiable():
    model = QueryTypedGeometryPredictor()
    query = torch.randn(2, 128, 36, 64, requires_grad=True)
    prediction = model(query)
    assert prediction["normal_axis_moment"].shape == (2, 6, 36, 64)
    torch.testing.assert_close(
        prediction["normal_axis_moment"][:, :3].sum(1), torch.ones(2, 36, 64),
        atol=2e-6, rtol=2e-6,
    )
    target = torch.zeros(2, 9, 36, 64)
    target[:, 0] = 1.0
    loss = query_typed_geometry_supervision_loss(
        prediction, target, torch.ones(2, 36, 64),
    )
    loss.backward()
    assert torch.isfinite(loss) and query.grad is not None


def test_candidate_missingness_cannot_create_typed_consistency_evidence():
    model = QueryTypedGeometryPredictor()
    prediction = model(torch.randn(1, 128, 36, 64))
    candidate = torch.zeros(1, 2, 18, 36, 64)
    candidate[:, 0, 1] = 1.0
    candidate[:, 0, 9] = 1.0
    candidate[:, 1] = candidate[:, 0]
    candidate[:, 1, 1] = 0.0
    feature = append_query_typed_geometry_consistency(candidate, prediction)
    assert feature.shape == (1, 2, 23, 36, 64)
    assert torch.all(feature[:, 1, 19:] == 0.0)
    assert torch.all(feature[:, 0, 19:] >= 0.0)


def test_fixed_denominator_geometry_and_phase_conjunction_have_unknown_floor():
    prediction = {
        "normal_axis_moment": torch.zeros(1, 6, 36, 64),
        "relative_log_depth": torch.zeros(1, 36, 64),
        "log_depth_std": torch.zeros(1, 36, 64),
        "boundary": torch.zeros(1, 36, 64),
        "confidence": torch.ones(1, 36, 64),
    }
    prediction["normal_axis_moment"][:, 0] = 1.0
    moment = prediction["normal_axis_moment"].permute(0, 2, 3, 1)
    full = fixed_denominator_query_typed_geometry_score(
        prediction, moment, torch.zeros(1, 36, 64),
        torch.zeros(1, 36, 64), torch.zeros(1, 36, 64),
        torch.ones(1, 36, 64),
    )
    missing = fixed_denominator_query_typed_geometry_score(
        prediction, moment, torch.zeros(1, 36, 64),
        torch.zeros(1, 36, 64), torch.zeros(1, 36, 64),
        torch.zeros(1, 36, 64),
    )
    torch.testing.assert_close(full, torch.ones(1))
    torch.testing.assert_close(missing, -torch.ones(1))
    joint = conjunct_phase_and_typed_geometry_score(
        torch.tensor([0.5, 0.5]), torch.tensor([1.0, -1.0]),
    )
    torch.testing.assert_close(joint, torch.tensor([0.5, -1.0]))


def test_normal_axis_consistency_is_sign_invariant():
    prediction = {
        "normal_axis_moment": torch.zeros(1, 6, 36, 64),
        "relative_log_depth": torch.zeros(1, 36, 64),
        "log_depth_std": torch.zeros(1, 36, 64),
        "boundary": torch.zeros(1, 36, 64),
        "confidence": torch.ones(1, 36, 64),
    }
    prediction["normal_axis_moment"][:, 0] = 1.0
    candidate = torch.zeros(1, 1, 18, 36, 64)
    candidate[:, :, 1] = 1.0
    candidate[:, :, 9] = 1.0
    first = append_query_typed_geometry_consistency(candidate, prediction)
    # Both +x and -x have the same second moment; no signed normal is accepted.
    second = append_query_typed_geometry_consistency(candidate.clone(), prediction)
    torch.testing.assert_close(first[:, :, 19], second[:, :, 19])
