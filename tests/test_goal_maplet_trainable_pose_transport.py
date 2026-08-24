from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _feature_only_query,
)

from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    MinimalPoseTransportConfig,
    MinimalPoseTransportReadout,
    anonymous_view_conditioned_map_code,
    load_minimal_pose_transport_readout,
    pose_transport_model_content_sha256,
    save_minimal_pose_transport_readout,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
    pose_transport_hierarchy_content_sha256,
)
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
    QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS,
    CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS,
    FrozenSparseTransportEdges,
    build_frozen_sparse_transport_edges,
    categorical_token_capacity_pose_transport,
    differentiable_fixed_kernel_capacity_pose_transport,
    differentiable_sparse_pose_transport,
    query_token_capacity_pose_transport,
)


def test_minimal_query_head_outputs_typed_32d_field_and_gradients():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(hidden_dim=16))
    radio = torch.randn(2, 1280, 5, 7, requires_grad=True)
    ray = torch.randn(2, 2, 5, 7)
    output = model(radio, ray)
    assert output.pose_code.shape == (2, 32, 5, 7)
    assert output.normal_camera.shape == (2, 3, 5, 7)
    assert output.relative_depth.shape == (2, 5, 7)
    assert output.boundary.shape == (2, 5, 7)
    assert output.confidence.shape == (2, 4, 5, 7)
    assert output.normal_frame == "camera"
    assert output.depth_valid.all()
    assert output.boundary_valid.all()
    assert torch.all((output.boundary >= 0.0) & (output.boundary <= 1.0))
    assert torch.all((output.confidence >= 0.0) & (output.confidence <= 1.0))
    output.pose_code.sum().backward()
    assert torch.isfinite(radio.grad).all()


def test_map_projection_is_trainable_normalized_and_zero_safe():
    model = MinimalPoseTransportReadout(
        MinimalPoseTransportConfig(hidden_dim=8, map_feature_dim=5, pose_code_dim=3)
    )
    value, valid = model.project_map_code(torch.randn(7, 5, requires_grad=True))
    assert value.shape == (7, 3)
    assert valid.all()
    torch.testing.assert_close(torch.linalg.vector_norm(value, dim=1), torch.ones(7))
    value.sum().backward()
    assert model.map_projection.weight.grad is not None


def test_shared_surface_space_uses_identical_query_and_map_projection_at_initialization():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=5, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
        shared_query_map_projection=True,
    ))
    feature = torch.randn(1, 5, 2, 3)
    query = model(feature, torch.zeros(1, 2, 2, 3))
    mapped, valid = model.project_map_code(feature.permute(0, 2, 3, 1))
    assert valid.all()
    torch.testing.assert_close(
        query.pose_code.permute(0, 2, 3, 1), mapped, atol=1.0e-6, rtol=1.0e-6
    )


def test_feature_only_query_removes_random_geometry_and_confidence():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=5, map_feature_dim=5, pose_code_dim=5, hidden_dim=8,
        shared_query_map_projection=True,
    ))
    with torch.no_grad():
        model.map_projection.weight.copy_(torch.eye(5))
    original = model(torch.randn(1, 5, 2, 3), torch.zeros(1, 2, 2, 3))
    feature_only = _feature_only_query(original)
    torch.testing.assert_close(feature_only.pose_code, original.pose_code)
    torch.testing.assert_close(
        feature_only.confidence[:, 0],
        original.pose_code_valid.to(dtype=feature_only.confidence.dtype),
    )
    assert not torch.any(feature_only.confidence[:, 1:])
    assert not torch.any(feature_only.normal_valid)
    assert not torch.any(feature_only.depth_valid)
    assert not torch.any(feature_only.boundary_valid)


def test_zero_vector_heads_are_invalid_and_have_zero_confidence():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(hidden_dim=8))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    output = model(torch.zeros(1, 1280, 2, 2), torch.zeros(1, 2, 2, 2))
    assert not output.pose_code_valid.any()
    assert not output.normal_valid.any()
    assert torch.count_nonzero(output.confidence[:, :2]) == 0


def test_anonymous_low_rank_map_field_changes_with_view_without_mode_label():
    canonical = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    basis = torch.zeros(2, 2, 2)
    basis[0, 0, 1] = 1.0
    coordinate = torch.zeros(2, 2, 2)
    coordinate[1, 0, 0] = 1.0
    code, valid = anonymous_view_conditioned_map_code(canonical, basis, coordinate)
    assert valid.all()
    assert not torch.equal(code[0, 0], code[1, 0])
    torch.testing.assert_close(code[0, 1], code[1, 1])


def test_model_content_hash_is_deterministic_and_parameter_sensitive():
    torch.manual_seed(7)
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(hidden_dim=8))
    first = pose_transport_model_content_sha256(model)
    assert first == pose_transport_model_content_sha256(model)
    with torch.no_grad():
        model.source_bias.add_(0.1)
    assert pose_transport_model_content_sha256(model) != first


def test_model_artifact_roundtrip_and_tamper_detection(tmp_path: Path):
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(hidden_dim=8))
    path = tmp_path / "transport.pt"
    save_minimal_pose_transport_readout(
        model, path, metadata={"leave_one_view_out_required": True}
    )
    loaded, metadata = load_minimal_pose_transport_readout(path)
    assert pose_transport_model_content_sha256(loaded) == pose_transport_model_content_sha256(model)
    assert metadata["uses_pnp"] is False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict"]["source_bias"] += 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="content hash"):
        load_minimal_pose_transport_readout(path)


def test_query_head_rejects_nonfinite_or_wrong_ray_grid():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(hidden_dim=8))
    radio = torch.zeros(1, 1280, 2, 3)
    with pytest.raises(ValueError, match="ray_xy"):
        model(radio, torch.zeros(1, 2, 3, 2))
    radio[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(radio, torch.zeros(1, 2, 2, 3))


def _tiny_hierarchy():
    parent = np.asarray([0, 0], dtype=np.int64)
    support = np.asarray([-1, -1], dtype=np.int64)
    offsets = np.asarray([0, 1, 2], dtype=np.int64)
    adjacency = np.asarray([1, 0], dtype=np.int64)
    return PoseTransportHierarchy(
        child_parent_ids=parent, child_support_ids=support,
        adjacency_offsets=offsets, adjacency_child_rows=adjacency,
        content_sha256=pose_transport_hierarchy_content_sha256(
            parent, support, offsets, adjacency
        ),
    )


def _tiny_transport(model, *, target_weight=0.8, stage="medium"):
    radio = torch.randn(1, 4, 2, 2, requires_grad=True)
    ray = torch.zeros(1, 2, 2, 2)
    query = model(radio, ray)
    source_rows = np.asarray([[0], [1], [0], [1]], dtype=np.int64)
    source_mass = np.full((4, 1), 0.7, dtype=np.float32)
    target_rows = source_rows.copy()
    target_mass = np.full((4, 1), target_weight, dtype=np.float32)
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    edges = build_frozen_sparse_transport_edges(
        source_rows, source_mass, xy, target_rows, target_mass,
        _tiny_hierarchy(), stage=stage,
    )
    feature = torch.randn(4, 1, 5)
    normal = torch.zeros(4, 1, 3); normal[..., 2] = 1.0
    valid = torch.ones(4, 1, 4, dtype=torch.bool)
    confidence = torch.ones(4, 1, 4)
    result = differentiable_sparse_pose_transport(
        model, query, torch.as_tensor(source_mass), torch.ones(4),
        torch.as_tensor(target_mass), feature, normal,
        torch.zeros(4, 1, dtype=torch.bool), torch.zeros(4, 1),
        torch.zeros(4, 1), valid, confidence, edges,
    )
    return result, radio, edges


def _tiny_capacity_transport(model, *, target_weight=0.8, stage="medium"):
    radio = torch.randn(1, 4, 2, 2, requires_grad=True)
    query = model(radio, torch.zeros(1, 2, 2, 2))
    source_rows = np.asarray([[0], [1], [0], [1]], dtype=np.int64)
    source_mass = np.full((4, 1), 0.7, dtype=np.float32)
    target_rows = source_rows.copy()
    target_mass = np.full((4, 1), target_weight, dtype=np.float32)
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    edges = build_frozen_sparse_transport_edges(
        source_rows, source_mass, xy, target_rows, target_mass,
        _tiny_hierarchy(), stage=stage,
    )
    feature = torch.randn(4, 1, 5)
    normal = torch.zeros(4, 1, 3); normal[..., 2] = 1.0
    valid = torch.ones(4, 1, 4, dtype=torch.bool)
    confidence = torch.ones(4, 1, 4)
    result = differentiable_fixed_kernel_capacity_pose_transport(
        model, query, torch.as_tensor(source_mass), torch.ones(4),
        torch.as_tensor(target_mass), feature, normal,
        torch.zeros(4, 1, dtype=torch.bool), torch.zeros(4, 1),
        torch.zeros(4, 1), valid, confidence, edges,
    )
    return result, radio, edges


def test_differentiable_transport_conserves_source_and_backpropagates():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    result, radio, edges = _tiny_transport(model)
    torch.testing.assert_close(
        result.matched_source_probability + result.unmatched_source_probability,
        result.source_probability,
    )
    assert edges.source_index.size > 0
    (-result.combined_score).backward()
    assert torch.isfinite(radio.grad).all()
    assert model.map_projection.weight.grad is not None


def test_sparse_transport_backward_is_finite_with_unused_source_slots():
    """Unused retrieval slots must not poison the scatter-softmax gradient."""

    torch.manual_seed(17)
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=4, pose_code_dim=4, hidden_dim=8,
    ))
    query = model(torch.randn(1, 4, 1, 2), torch.zeros(1, 2, 1, 2))
    edges = FrozenSparseTransportEdges(
        source_index=np.asarray([0], dtype=np.int64),
        target_index=np.asarray([0], dtype=np.int64),
        hierarchy_score=np.asarray([1.0], dtype=np.float32),
        layout_score=np.asarray([1.0], dtype=np.float32),
        source_count=4,
        target_count=2,
        stage="medium",
    )
    result = differentiable_sparse_pose_transport(
        model, query,
        torch.tensor([[0.7, 0.0], [0.0, 0.0]]), torch.ones(2),
        torch.tensor([0.8, 0.0]), torch.randn(2, 4),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        torch.tensor([True, True]), torch.zeros(2), torch.zeros(2),
        torch.ones(2, 4, dtype=torch.bool),
        # Zero confidence is a real typed-missingness state and must have a
        # finite derivative rather than the singular derivative of sqrt(0).
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        edges,
    )
    result.combined_score.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_differentiable_transport_missing_target_cannot_improve_score():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    present, _, _ = _tiny_transport(model, target_weight=0.8)
    absent, _, edges = _tiny_transport(model, target_weight=0.0)
    assert edges.source_index.size == 0
    assert float(absent.combined_score) == pytest.approx(-1.0)
    assert float(present.combined_score) >= float(absent.combined_score)


def test_fixed_kernel_transport_obeys_both_source_and_target_capacities():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    result, radio, edges = _tiny_capacity_transport(model)
    assert edges.source_index.size > 0
    assert result.transport_semantics == FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
    torch.testing.assert_close(
        result.matched_source_probability + result.unmatched_source_probability,
        result.source_probability,
    )
    assert torch.all(
        result.matched_source_probability <= result.source_probability + 3.0e-6
    )
    assert result.matched_target_probability is not None
    assert result.target_probability is not None
    assert torch.all(
        result.matched_target_probability <= result.target_probability + 3.0e-6
    )
    (-result.combined_score).backward()
    assert torch.isfinite(radio.grad).all()


def test_fixed_kernel_transport_target_or_modality_disappearance_cannot_improve():
    torch.manual_seed(23)
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    present, _, edges = _tiny_capacity_transport(model, target_weight=0.8)
    lower_mass, _, _ = _tiny_capacity_transport(model, target_weight=0.4)
    # The random query/map tensors differ between helper calls, so the strict
    # target-mass check is replayed below with an identical observation.
    radio = torch.randn(1, 4, 2, 2)
    query = model(radio, torch.zeros(1, 2, 2, 2))
    source = torch.full((4, 1), 0.7)
    feature = torch.randn(4, 1, 5)
    normal = torch.zeros(4, 1, 3); normal[..., 2] = 1.0
    valid = torch.ones(4, 1, 4, dtype=torch.bool)
    confidence = torch.ones(4, 1, 4)

    def score(weight, mask):
        return differentiable_fixed_kernel_capacity_pose_transport(
            model, query, source, torch.ones(4), torch.full((4, 1), weight),
            feature, normal, torch.zeros(4, 1, dtype=torch.bool),
            torch.zeros(4, 1), torch.zeros(4, 1), mask, confidence, edges,
        ).combined_score

    full = score(0.8, valid)
    half = score(0.4, valid)
    missing_feature = valid.clone(); missing_feature[..., 0] = False
    without_feature = score(0.8, missing_feature)
    assert float(full) >= float(half)
    assert float(full) >= float(without_feature)
    # Keep the independently constructed values live so accidental NaNs are
    # caught even though their random features are not compared numerically.
    assert torch.isfinite(present.combined_score)
    assert torch.isfinite(lower_mass.combined_score)


def test_fixed_kernel_transport_rejects_non_subprobability_token_masses():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    result, _, edges = _tiny_capacity_transport(model)
    query = model(torch.randn(1, 4, 2, 2), torch.zeros(1, 2, 2, 2))
    feature = torch.randn(4, 1, 5)
    normal = torch.zeros(4, 1, 3); normal[..., 2] = 1.0
    valid = torch.ones(4, 1, 4, dtype=torch.bool)
    confidence = torch.ones(4, 1, 4)
    with pytest.raises(ValueError, match="sub-probabilities"):
        differentiable_fixed_kernel_capacity_pose_transport(
            model, query, torch.full((4, 1), 1.01), torch.ones(4),
            torch.full((4, 1), 0.8), feature, normal,
            torch.zeros(4, 1, dtype=torch.bool), torch.zeros(4, 1),
            torch.zeros(4, 1), valid, confidence, edges,
        )
    assert torch.isfinite(result.combined_score)


def test_torch_transport_rejects_wrong_depth_schema_and_normal_frame():
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    _, _, edges = _tiny_capacity_transport(model)
    query = model(torch.randn(1, 4, 2, 2), torch.zeros(1, 2, 2, 2))
    arguments = (
        torch.full((4, 1), 0.7), torch.ones(4),
        torch.full((4, 1), 0.8), torch.randn(4, 1, 5),
        torch.nn.functional.normalize(torch.randn(4, 1, 3), dim=2),
        torch.zeros(4, 1, dtype=torch.bool), torch.zeros(4, 1),
        torch.zeros(4, 1), torch.ones(4, 1, 4, dtype=torch.bool),
        torch.ones(4, 1, 4), edges,
    )
    with pytest.raises(ValueError, match="depth semantics"):
        differentiable_fixed_kernel_capacity_pose_transport(
            model, replace(query, depth_semantics="ordinal_depth_v1"), *arguments
        )
    with pytest.raises(ValueError, match="camera frame"):
        differentiable_fixed_kernel_capacity_pose_transport(
            model, replace(query, normal_frame="world"), *arguments
        )


def test_fixed_kernel_auto_invalidates_zero_map_normal():
    torch.manual_seed(53)
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=4, map_feature_dim=5, pose_code_dim=3, hidden_dim=8,
    ))
    _, _, edges = _tiny_capacity_transport(model)
    query = model(torch.randn(1, 4, 2, 2), torch.zeros(1, 2, 2, 2))
    source = torch.full((4, 1), 0.7)
    target_mass = torch.full((4, 1), 0.8)
    feature = torch.randn(4, 1, 5)
    zero_normal = torch.zeros(4, 1, 3)
    validity = torch.ones(4, 1, 4, dtype=torch.bool)
    confidence = torch.ones(4, 1, 4)

    def score(mask: torch.Tensor) -> torch.Tensor:
        return differentiable_fixed_kernel_capacity_pose_transport(
            model, query, source, torch.ones(4), target_mass, feature,
            zero_normal, torch.zeros(4, 1, dtype=torch.bool),
            torch.zeros(4, 1), torch.zeros(4, 1), mask, confidence, edges,
        ).combined_score

    normal_missing = validity.clone(); normal_missing[..., 1] = False
    torch.testing.assert_close(score(validity), score(normal_missing))


def test_frozen_edges_reject_duplicate_child_even_when_slot_mass_is_zero():
    rows = np.asarray([[0, 0], [0, -1], [0, -1], [0, -1]], dtype=np.int64)
    mass = np.asarray([[0.8, 0.0], [0.8, 0.0], [0.8, 0.0], [0.8, 0.0]], dtype=np.float32)
    targets = np.asarray([[0], [0], [0], [0]], dtype=np.int64)
    target_mass = np.full((4, 1), 0.8, dtype=np.float32)
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate child"):
        build_frozen_sparse_transport_edges(
            rows, mass, xy, targets, target_mass, _tiny_hierarchy(), stage="medium"
        )


def test_medium_transport_adds_adjacent_edges_that_fine_excludes():
    rows = np.asarray([[0], [0], [0], [0]], dtype=np.int64)
    targets = np.asarray([[1], [1], [1], [1]], dtype=np.int64)
    mass = np.ones((4, 1), dtype=np.float32)
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    medium = build_frozen_sparse_transport_edges(
        rows, mass, xy, targets, mass, _tiny_hierarchy(), stage="medium"
    )
    fine = build_frozen_sparse_transport_edges(
        rows, mass, xy, targets, mass, _tiny_hierarchy(), stage="fine"
    )
    assert medium.source_index.size > 0
    assert fine.source_index.size == 0


def test_query_token_capacity_transport_is_source_and_target_sub_stochastic():
    query = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    target = query.detach()[:, None, :].clone()
    result = query_token_capacity_pose_transport(
        query,
        torch.full((4,), 0.8),
        torch.ones(4),
        target,
        torch.full((4, 1), 0.7),
        torch.ones(4, 1, dtype=torch.bool),
        height=2,
        width=2,
        local_radius_tokens=0,
    )
    assert result.transport_semantics == QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS
    torch.testing.assert_close(
        result.matched_source_probability + result.unmatched_source_probability,
        torch.full((4,), 0.8),
    )
    assert torch.all(result.matched_target_probability <= result.target_probability + 3.0e-6)
    assert float(result.combined_score) == pytest.approx(0.12, abs=1.0e-6)
    (-result.combined_score).backward()
    assert torch.isfinite(query.grad).all()


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_query_token_capacity_target_disappearance_cannot_improve(radius: int):
    generator = torch.Generator().manual_seed(211 + radius)
    height, width, slots, channels = 5, 6, 3, 7
    query = torch.randn(height * width, channels, generator=generator)
    target = torch.randn(height * width, slots, channels, generator=generator)
    source = torch.rand(height * width, generator=generator)
    target_mass = torch.rand(height * width, slots, generator=generator)
    target_mass = 0.9 * target_mass / target_mass.sum(dim=1, keepdim=True)
    valid = torch.rand(height * width, slots, generator=generator) > 0.1
    reduced_mass = target_mass * torch.rand(
        height * width, slots, generator=generator
    )
    reduced_valid = valid & (
        torch.rand(height * width, slots, generator=generator) > 0.3
    )
    before = query_token_capacity_pose_transport(
        query, source, torch.ones_like(source), target, target_mass, valid,
        height=height, width=width, local_radius_tokens=radius,
    )
    after = query_token_capacity_pose_transport(
        query, source, torch.ones_like(source), target, reduced_mass, reduced_valid,
        height=height, width=width, local_radius_tokens=radius,
    )
    assert float(after.combined_score) <= float(before.combined_score) + 2.0e-7
    assert torch.all(
        after.matched_source_probability
        <= before.matched_source_probability + 2.0e-7
    )


def test_categorical_parent_layout_transport_conserves_both_capacities():
    source_ids = torch.tensor([[3, 7], [3, 7], [3, 7], [3, 7]])
    source_mass = torch.tensor([[0.6, 0.2]] * 4)
    target_ids = torch.tensor([[3, 9], [3, 9], [3, 9], [3, 9]])
    target_mass = torch.tensor([[0.7, 0.1]] * 4)
    result = categorical_token_capacity_pose_transport(
        source_ids,
        source_mass,
        torch.ones(4),
        target_ids,
        target_mass,
        torch.ones(4, 2, dtype=torch.bool),
        height=2,
        width=2,
        local_radius_tokens=0,
    )
    assert result.transport_semantics == CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS
    matched = result.matched_source_probability.reshape(4, 2)
    unmatched = result.unmatched_source_probability.reshape(4, 2)
    torch.testing.assert_close(matched + unmatched, source_mass)
    torch.testing.assert_close(matched[:, 0], torch.full((4,), 0.42))
    torch.testing.assert_close(matched[:, 1], torch.zeros(4))
    assert torch.all(result.matched_target_probability <= target_mass + 3.0e-6)


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_categorical_parent_layout_target_disappearance_cannot_improve(radius: int):
    generator = torch.Generator().manual_seed(307 + radius)
    height, width, source_slots, target_slots = 5, 6, 4, 3
    source_ids = torch.randint(0, 7, (height * width, source_slots), generator=generator)
    target_ids = torch.randint(0, 7, (height * width, target_slots), generator=generator)
    source_mass = torch.rand(height * width, source_slots, generator=generator)
    source_mass = 0.9 * source_mass / source_mass.sum(dim=1, keepdim=True)
    target_mass = torch.rand(height * width, target_slots, generator=generator)
    target_mass = 0.9 * target_mass / target_mass.sum(dim=1, keepdim=True)
    valid = torch.rand(height * width, target_slots, generator=generator) > 0.1
    reduced_mass = target_mass * torch.rand(
        height * width, target_slots, generator=generator
    )
    reduced_valid = valid & (
        torch.rand(height * width, target_slots, generator=generator) > 0.3
    )
    before = categorical_token_capacity_pose_transport(
        source_ids, source_mass, torch.ones(height * width),
        target_ids, target_mass, valid, height=height, width=width,
        local_radius_tokens=radius,
    )
    after = categorical_token_capacity_pose_transport(
        source_ids, source_mass, torch.ones(height * width),
        target_ids, reduced_mass, reduced_valid, height=height, width=width,
        local_radius_tokens=radius,
    )
    assert float(after.combined_score) <= float(before.combined_score) + 2.0e-7
    assert torch.all(
        after.matched_source_probability
        <= before.matched_source_probability + 2.0e-7
    )
