from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
    pose_transport_hierarchy_content_sha256,
)
from feature_extract.vfm.localization_goal_maplet.dense_fixed_identity_transport import (
    DensePoseTransportHierarchyGPU,
    dense_fixed_identity_transport,
)
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    build_frozen_sparse_transport_edges,
    differentiable_fixed_kernel_capacity_pose_transport,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    MinimalPoseTransportConfig,
    MinimalPoseTransportReadout,
    QueryPoseHeadOutput,
)


def _hierarchy() -> PoseTransportHierarchy:
    parent = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    support = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    neighbours = ((1,), (0, 2), (1, 3), (2, 4), (3, 5), (4,))
    offsets = np.zeros(7, dtype=np.int64)
    for row, values in enumerate(neighbours):
        offsets[row + 1] = offsets[row] + len(values)
    adjacency = np.concatenate([np.asarray(values, dtype=np.int64) for values in neighbours])
    return PoseTransportHierarchy(
        child_parent_ids=parent,
        child_support_ids=support,
        adjacency_offsets=offsets,
        adjacency_child_rows=adjacency,
        content_sha256=pose_transport_hierarchy_content_sha256(
            parent, support, offsets, adjacency
        ),
    )


def _case(stage: str):
    torch.manual_seed(7)
    rng = np.random.default_rng(11)
    height, width, source_slots, target_slots, dimension, batch = 3, 4, 3, 2, 4, 3
    token_count = height * width
    model = MinimalPoseTransportReadout(MinimalPoseTransportConfig(
        radio_channels=dimension,
        map_feature_dim=dimension,
        pose_code_dim=dimension,
        hidden_dim=4,
        shared_query_map_projection=True,
    ))
    with torch.no_grad():
        model.map_projection.weight.copy_(torch.eye(dimension))
        model.edge_weight_unconstrained.copy_(
            torch.tensor([0.3, -1.0, -2.0, -3.0, 0.1, -0.2])
        )
    query_code = torch.nn.functional.normalize(
        torch.randn(1, dimension, height, width), dim=1
    )
    query_valid = torch.ones((1, height, width), dtype=torch.bool)
    query = QueryPoseHeadOutput(
        pose_code=query_code,
        normal_camera=torch.zeros((1, 3, height, width)),
        relative_depth=torch.zeros((1, height, width)),
        boundary=torch.zeros((1, height, width)),
        confidence=torch.stack(
            [query_valid.float(), torch.zeros_like(query_valid, dtype=torch.float32),
             torch.zeros_like(query_valid, dtype=torch.float32),
             torch.zeros_like(query_valid, dtype=torch.float32)], dim=1,
        ),
        pose_code_valid=query_valid,
        normal_valid=torch.zeros_like(query_valid),
        depth_valid=torch.zeros_like(query_valid),
        boundary_valid=torch.zeros_like(query_valid),
        normal_frame="camera",
        depth_semantics={
            "coarse": "ordinal_depth_v1",
            "medium": "centered_log_depth_v1",
            "fine": "metric_log_depth_with_uncertainty_v1",
        }[stage],
    )
    source_rows = np.stack([
        rng.choice(6, size=source_slots, replace=False) for _ in range(token_count)
    ]).astype(np.int64)
    source_probability = rng.uniform(0.01, 1.0, (token_count, source_slots)).astype(np.float32)
    source_probability /= 1.5 * source_probability.sum(axis=1, keepdims=True)
    target_rows = np.stack([
        np.stack([rng.choice(6, size=target_slots, replace=False) for _ in range(token_count)])
        for _ in range(batch)
    ]).astype(np.int64)
    target_weight = rng.uniform(0.01, 1.0, (batch, token_count, target_slots)).astype(np.float32)
    target_weight /= 1.4 * target_weight.sum(axis=2, keepdims=True)
    target_feature = rng.normal(size=(batch, token_count, target_slots, dimension)).astype(np.float32)
    target_feature /= np.maximum(np.linalg.norm(target_feature, axis=3, keepdims=True), 1.0e-8)
    target_feature_valid = rng.uniform(size=(batch, token_count, target_slots)) > 0.15
    # Confidence is an independent typed observation, not child identity
    # mass.  This catches the production bug where dense replay used
    # sqrt(target_weight) instead of the stored feature-confidence channel.
    target_feature_confidence = rng.uniform(
        0.0, 1.0, (batch, token_count, target_slots)
    ).astype(np.float32)
    target_feature_confidence[~target_feature_valid] = 0.0
    token_xy = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).reshape(-1, 2)
    reliability = rng.uniform(0.1, 1.0, token_count).astype(np.float32)
    return (
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows, target_weight, target_feature, target_feature_confidence,
        target_feature_valid, _hierarchy(),
        height, width,
    )


@pytest.mark.parametrize("stage", ("coarse", "medium", "fine"))
def test_dense_fixed_identity_matches_sparse_authority(stage: str) -> None:
    (
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows, target_weight, target_feature, target_feature_confidence,
        target_feature_valid, hierarchy,
        height, width,
    ) = _case(stage)
    dense = dense_fixed_identity_transport(
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows, target_weight, target_feature, target_feature_confidence,
        target_feature_valid,
        DensePoseTransportHierarchyGPU(hierarchy, device="cpu"),
        stage=stage, height=height, width=width,
    )
    authority = []
    for candidate in range(target_rows.shape[0]):
        edges = build_frozen_sparse_transport_edges(
            source_rows, source_probability, token_xy,
            target_rows[candidate], target_weight[candidate], hierarchy, stage=stage,
        )
        modality_valid = np.zeros(target_rows[candidate].shape + (4,), dtype=bool)
        modality_valid[..., 0] = target_feature_valid[candidate]
        confidence = np.zeros(target_rows[candidate].shape + (4,), dtype=np.float32)
        confidence[..., 0] = target_feature_confidence[candidate]
        value = differentiable_fixed_kernel_capacity_pose_transport(
            model, query, torch.as_tensor(source_probability), torch.as_tensor(reliability),
            torch.as_tensor(target_weight[candidate]),
            torch.as_tensor(target_feature[candidate]),
            torch.zeros(target_rows[candidate].shape + (3,)),
            torch.zeros(target_rows[candidate].shape, dtype=torch.bool),
            torch.zeros(target_rows[candidate].shape),
            torch.zeros(target_rows[candidate].shape),
            torch.as_tensor(modality_valid), torch.as_tensor(confidence), edges,
        )
        authority.append(value.combined_score)
    authority_score = torch.stack(authority)
    assert torch.allclose(dense.scores, authority_score, atol=2.0e-7, rtol=0.0)
    weights = model.edge_weights()
    reconstructed = -1.0 + torch.sum(
        dense.component_statistics * weights[None], dim=1
    ) / weights.sum()
    assert torch.allclose(dense.scores, reconstructed, atol=1.0e-7, rtol=1.0e-7)
    assert dense.maximum_source_capacity_excess <= 3.0e-6
    assert dense.maximum_target_capacity_excess <= 3.0e-6


def test_dense_fixed_identity_batch_and_order_are_stable() -> None:
    (
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows, target_weight, target_feature, target_feature_confidence,
        target_feature_valid, hierarchy,
        height, width,
    ) = _case("coarse")
    resident = DensePoseTransportHierarchyGPU(hierarchy, device="cpu")
    full = dense_fixed_identity_transport(
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows, target_weight, target_feature, target_feature_confidence,
        target_feature_valid, resident,
        stage="coarse", height=height, width=width,
    )
    individual = torch.cat([
        dense_fixed_identity_transport(
            model, query, source_rows, source_probability, reliability, token_xy,
            target_rows[row : row + 1], target_weight[row : row + 1],
            target_feature[row : row + 1],
            target_feature_confidence[row : row + 1],
            target_feature_valid[row : row + 1],
            resident, stage="coarse", height=height, width=width,
        ).scores
        for row in range(target_rows.shape[0])
    ])
    reverse = dense_fixed_identity_transport(
        model, query, source_rows, source_probability, reliability, token_xy,
        target_rows[::-1].copy(), target_weight[::-1].copy(),
        target_feature[::-1].copy(), target_feature_confidence[::-1].copy(),
        target_feature_valid[::-1].copy(), resident,
        stage="coarse", height=height, width=width,
    )
    assert torch.allclose(full.scores, individual, atol=2.0e-7, rtol=0.0)
    assert torch.allclose(full.scores, reverse.scores.flip(0), atol=2.0e-7, rtol=0.0)


def test_dense_fixed_identity_rejects_noncanonical_token_layout() -> None:
    case = list(_case("fine"))
    case[5] = case[5][::-1].copy()
    with pytest.raises(ValueError, match="row-major"):
        dense_fixed_identity_transport(
            case[0], case[1], case[2], case[3], case[4], case[5], case[6],
            case[7], case[8], case[9], case[10],
            DensePoseTransportHierarchyGPU(case[11], device="cpu"),
            stage="fine", height=case[12], width=case[13],
        )
