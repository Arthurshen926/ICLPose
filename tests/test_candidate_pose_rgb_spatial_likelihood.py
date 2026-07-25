from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    _source_safe_joint_log_probabilities,
    candidate_pose_rgb_spatial_component_edge_usable,
    candidate_pose_rgb_spatial_score_component_prediction,
    CandidatePoseRGBSpatialEdgePrediction,
    context_candidate_logit_mixture,
    context_identity_cross_entropy_loss,
    context_identity_support_permutation_margin_loss,
    continuous_joint_log_probability_at_offsets,
    edge_log_likelihood_ratio_at_pose_projection,
    permute_runtime_candidate_slots,
    permute_runtime_support_appearance,
    permute_runtime_support_image_appearance_only,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
    selected_candidate_view_log_likelihood_ratio_at_offsets,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    spatial_density_nll,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    crop_rgb_window,
    template_search_cost_volume_logits,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([11, 13], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/b.png"]),
        split_names=np.asarray(["train", "train"]),
        xy=np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
        point_sources=np.asarray(["alike", "alike"]),
        candidate_track_ids=np.asarray([[101, 102], [201, 202]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray(
            [[0.9, 0.8], [0.7, 0.6]], dtype=np.float32
        ),
        candidate_prior_probabilities=np.asarray(
            [[0.45, 0.35], [0.50, 0.30]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.20, 0.20], dtype=np.float32),
        support_image_ids=np.asarray(
            [
                [["map/a.png", "map/b.png"], ["map/c.png", "map/d.png"]],
                [["map/e.png", "map/f.png"], ["map/g.png", "map/h.png"]],
            ]
        ),
        support_xy=np.asarray(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            ],
            dtype=np.float32,
        ),
        support_view_valid=np.ones((2, 2, 2), dtype=bool),
        support_view_weights=np.full((2, 2, 2), 0.5, dtype=np.float32),
        support_coverage_counts=np.ones((2, 2, 2), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
        },
    )


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return runtime_from_target_free_layout(
        _layout(),
        image_ids=np.asarray(
            [
                "query/a.png",
                "query/b.png",
                "map/a.png",
                "map/b.png",
                "map/c.png",
                "map/d.png",
                "map/e.png",
                "map/f.png",
                "map/g.png",
                "map/h.png",
            ]
        ),
    )


def _prediction() -> CandidatePoseRGBSpatialEdgePrediction:
    # Offset order is row-major y/x: (-1,-1), (0,-1), ... (1,1).
    local_probability = torch.tensor(
        [
            [
                [
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                ],
                [
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                ],
            ],
            [
                [
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                ],
                [
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                ],
            ],
        ],
        dtype=torch.float32,
    )
    dustbin = 1.0 - local_probability.sum(dim=-1, keepdim=True)
    joint = torch.cat([local_probability, dustbin], dim=-1)
    return CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=torch.zeros_like(local_probability),
        non_dustbin_logits=torch.zeros(local_probability.shape[:-1]),
        joint_log_probabilities=torch.log(joint),
        offsets_xy=torch.tensor(
            [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        ),
        context_log_likelihood_ratios=torch.zeros(local_probability.shape[:-1]),
        edge_usable=torch.ones(local_probability.shape[:-1], dtype=torch.bool),
    )


def test_continuous_joint_probability_bilinearly_interpolates_local_density() -> None:
    prediction = _prediction()
    log_probability, in_window = continuous_joint_log_probability_at_offsets(
        joint_log_probabilities=prediction.joint_log_probabilities,
        offsets_xy=prediction.offsets_xy,
        query_offsets_xy=torch.full((1, 2, 2, 2), 0.5, dtype=torch.float32),
    )

    # The four neighbours at (0,0), (1,0), (0,1), (1,1) are 0.05, 0.06,
    # 0.08, 0.09, so bilinear sampling at their midpoint is 0.07.
    torch.testing.assert_close(log_probability[0, 0, 0, 0].exp(), torch.tensor(0.07))
    assert bool(in_window[0, 0, 0, 0])


def test_cost_volume_offset_is_query_anchor_displacement() -> None:
    """A template right of the query anchor must be emitted at positive dx.

    Pose targets use ``projected_track_xy - query_anchor_xy``.  The RGB
    cost-volume coordinate has to use exactly that query-frame convention:
    the support template is fixed at its support observation while the query
    window is sampled at each proposed projected location.
    """

    query_features = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    support_features = torch.zeros_like(query_features)
    query_features[0, 0, 1, 2] = 1.0  # query anchor + (1, 0)
    support_features[0, 0, 1, 1] = 1.0  # fixed support-track center

    logits, offsets_xy = template_search_cost_volume_logits(
        query_features,
        support_features,
        search_radius_px=1.0,
        context_radius_px=0.0,
        step_px=1.0,
        temperature=1.0,
    )

    mode = offsets_xy[int(torch.argmax(logits[0]).item())]
    torch.testing.assert_close(mode, torch.tensor([1.0, 0.0]))


def test_out_of_window_projection_is_exact_fixed_non_support_not_dustbin_reward() -> None:
    prediction = _prediction()
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **prediction.__dict__,
            "joint_log_probabilities": torch.log(
                torch.cat(
                    [
                        torch.full((2, 2, 2, 9), 1e-4),
                        torch.full((2, 2, 2, 1), 1.0 - 9e-4),
                    ],
                    dim=-1,
                )
            ),
        }
    )
    edge_llr, usable = edge_log_likelihood_ratio_at_pose_projection(
        prediction=prediction,
        candidate_projection_offsets_xy=torch.full((1, 2, 2, 2), 99.0),
        candidate_projection_valid=torch.ones((1, 2, 2), dtype=torch.bool),
        missing_edge_log_likelihood_ratio=0.0,
    )

    torch.testing.assert_close(edge_llr, torch.zeros_like(edge_llr))
    assert not bool(usable.any())


def test_source_specific_masks_keep_rgb_and_context_evidence_independent() -> None:
    prediction = _prediction()
    raw = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)
    full_joint = torch.tensor(
        [[[[0.01, 0.01, 0.01, 0.01, 0.96]]]], dtype=torch.float32
    ).log()
    rgb = torch.tensor([[[True, False]]])
    context = torch.tensor([[[False, True]]])
    # Use a single point/candidate/two-view prediction so each source-only
    # fallback can be inspected directly.
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=raw.repeat(1, 1, 2, 1),
        non_dustbin_logits=torch.zeros((1, 1, 2)),
        joint_log_probabilities=torch.cat(
            [full_joint, full_joint], dim=2
        ),
        offsets_xy=torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        context_log_likelihood_ratios=torch.tensor([[[0.0, 1.0]]]),
        edge_usable=rgb | context,
        raw_spatial_logits=raw.repeat(1, 1, 2, 1),
        spatial_residual_logits=torch.zeros((1, 1, 2, 4)),
        rgb_edge_usable=rgb,
        context_edge_usable=context,
    )
    assert torch.equal(
        candidate_pose_rgb_spatial_component_edge_usable(
            prediction=prediction, component="rgb_cost_volume"
        ),
        rgb,
    )
    assert torch.equal(
        candidate_pose_rgb_spatial_component_edge_usable(
            prediction=prediction, component="context_only"
        ),
        context,
    )
    assert not bool(
        candidate_pose_rgb_spatial_component_edge_usable(
            prediction=prediction, component="learned_spatial_no_context"
        ).any()
    )
    rgb_component = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction, component="rgb_cost_volume"
    )
    context_component = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction, component="context_only"
    )
    assert torch.equal(rgb_component.edge_usable, rgb)
    assert torch.equal(context_component.edge_usable, context)


def test_source_safe_density_uses_raw_rgb_or_neutral_context_fallback() -> None:
    full = torch.tensor(
        [[[[0.70, 0.10, 0.10, 0.10, 0.00]]]], dtype=torch.float32
    ).clamp_min(1e-6).log()
    raw = torch.tensor([[[[0.0, 2.0, 0.0, 0.0]]]], dtype=torch.float32)
    rgb = torch.tensor([[[True, False, True]]])
    context = torch.tensor([[[True, True, False]]])
    full = full.expand(1, 1, 3, -1).clone()
    raw = raw.expand(1, 1, 3, -1).clone()
    joint = _source_safe_joint_log_probabilities(
        full_joint_log_probabilities=full,
        raw_spatial_logits=raw,
        rgb_edge_usable=rgb,
        context_edge_usable=context,
    )
    # Both sources: preserve learned fusion. RGB only: raw cost volume. Context
    # only: exactly neutral local density with the scalar context path separate.
    torch.testing.assert_close(joint[0, 0, 0], full[0, 0, 0])
    torch.testing.assert_close(
        joint[0, 0, 1].exp(),
        torch.tensor([0.125, 0.125, 0.125, 0.125, 0.5], dtype=torch.float32),
    )
    assert int(torch.argmax(joint[0, 0, 2, :-1]).item()) == 1


def test_runtime_preserves_fixed_candidate_and_null_mass() -> None:
    runtime = _runtime()
    torch.testing.assert_close(
        runtime.candidate_probabilities.sum(dim=1) + runtime.null_probabilities,
        torch.ones(2),
    )
    with pytest.raises(ValueError, match="priors"):
        CandidatePoseRGBSpatialRuntime(
            **{
                **runtime.__dict__,
                "null_probabilities": torch.tensor([0.1, 0.2]),
            }
        )


def test_support_permutation_keeps_geometry_priors_and_query_anchors_fixed() -> None:
    runtime = _runtime()
    permuted = permute_runtime_support_appearance(runtime, shift=1)

    torch.testing.assert_close(permuted.query_xy, runtime.query_xy)
    torch.testing.assert_close(
        permuted.candidate_probabilities, runtime.candidate_probabilities
    )
    torch.testing.assert_close(permuted.null_probabilities, runtime.null_probabilities)
    torch.testing.assert_close(
        permuted.candidate_view_weights, runtime.candidate_view_weights
    )
    torch.testing.assert_close(permuted.support_view_valid, runtime.support_view_valid)
    assert not torch.equal(permuted.support_image_indices, runtime.support_image_indices)
    assert not torch.equal(permuted.support_xy, runtime.support_xy)


def test_image_only_support_permutation_keeps_coordinates_and_mixture_fixed() -> None:
    runtime = _runtime()
    permuted = permute_runtime_support_image_appearance_only(runtime, shift=1)

    torch.testing.assert_close(permuted.query_image_indices, runtime.query_image_indices)
    torch.testing.assert_close(permuted.query_xy, runtime.query_xy)
    torch.testing.assert_close(permuted.support_xy, runtime.support_xy)
    torch.testing.assert_close(permuted.support_view_valid, runtime.support_view_valid)
    torch.testing.assert_close(
        permuted.candidate_view_weights, runtime.candidate_view_weights
    )
    torch.testing.assert_close(
        permuted.candidate_probabilities, runtime.candidate_probabilities
    )
    torch.testing.assert_close(permuted.null_probabilities, runtime.null_probabilities)
    assert not torch.equal(permuted.support_image_indices, runtime.support_image_indices)


def test_support_patch_permutation_matches_runtime_appearance_slots() -> None:
    runtime = _runtime()
    support_patches = torch.arange(
        2 * 2 * 2 * 3,
        dtype=torch.float32,
    ).reshape(2, 2, 2, 1, 1, 3)
    runtime = CandidatePoseRGBSpatialRuntime(
        **{
            **runtime.__dict__,
            "support_view_valid": torch.tensor(
                [[[True, True], [True, False]], [[True, True], [True, True]]]
            ),
        }
    )
    shift = 1
    permuted_runtime = permute_runtime_support_appearance(runtime, shift=shift)
    permuted_patches = permute_support_patch_appearance(
        runtime=permuted_runtime,
        support_patches=support_patches,
        shift=shift,
    )
    for point_index in range(runtime.point_count):
        slots = torch.nonzero(runtime.support_view_valid[point_index], as_tuple=False)
        expected = torch.roll(
            support_patches[point_index, slots[:, 0], slots[:, 1]], shifts=shift, dims=0
        )
        torch.testing.assert_close(
            permuted_patches[point_index, slots[:, 0], slots[:, 1]], expected
        )
        invalid = ~runtime.support_view_valid[point_index]
        assert torch.count_nonzero(permuted_patches[point_index][invalid]) == 0


def test_support_patch_permutation_equals_recropping_deranged_runtime() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[6.0, 6.0]]),
        support_image_indices=torch.tensor([[[1, 2], [3, 4]]]),
        support_xy=torch.tensor(
            [[[[3.0, 3.0], [4.0, 4.0]], [[7.0, 7.0], [8.0, 8.0]]]]
        ),
        support_view_valid=torch.ones((1, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((1, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.45, 0.45]]),
        null_probabilities=torch.tensor([0.10]),
    )
    images = torch.stack(
        [
            torch.full((3, 12, 12), float(index), dtype=torch.float32)
            + torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
            for index in range(5)
        ]
    )

    def crop_support(active: CandidatePoseRGBSpatialRuntime) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for candidate_index in range(active.candidate_count):
            views: list[torch.Tensor] = []
            for view_index in range(active.support_view_count):
                image_index = int(active.support_image_indices[0, candidate_index, view_index])
                patch, _ = crop_rgb_window(
                    images[image_index : image_index + 1],
                    active.support_xy[0, candidate_index, view_index].reshape(1, 2),
                    radius_px=1.0,
                    step_px=1.0,
                    image_width=12,
                    image_height=12,
                )
                views.append(patch[0])
            rows.append(torch.stack(views))
        return torch.stack(rows).unsqueeze(0)

    shift = 1
    original_patches = crop_support(runtime)
    permuted_runtime = permute_runtime_support_appearance(runtime, shift=shift)
    recropped = crop_support(permuted_runtime)
    reindexed = permute_support_patch_appearance(
        runtime=permuted_runtime,
        support_patches=original_patches,
        shift=shift,
    )
    torch.testing.assert_close(reindexed, recropped, atol=0.0, rtol=0.0)


def test_runtime_factory_rejects_train_only_targets() -> None:
    with pytest.raises(ValueError, match="target-free layout"):
        runtime_from_target_free_layout(
            object.__new__(CandidatePoseRGBSpatialTrainingTargets),
            image_ids=np.asarray(["query/a.png"]),
        )


@pytest.mark.parametrize(
    "context_encoder_arch",
    ["conv_v1", "cross_attention_v2", "absolute_cross_attention_v3"],
)
def test_small_model_outputs_normalized_density_without_pose_input(
    context_encoder_arch: str,
) -> None:
    torch.manual_seed(7)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        context_encoder_arch=context_encoder_arch,
    )
    runtime = _runtime()
    query_patches = torch.rand(2, 3, 9, 9)
    support_patches = torch.rand(2, 2, 2, 3, 9, 9)

    prediction = model(
        runtime=runtime,
        query_rgb_patches=query_patches,
        support_rgb_patches=support_patches,
    )

    torch.testing.assert_close(
        prediction.joint_log_probabilities.exp().sum(dim=-1),
        torch.ones((2, 2, 2)),
        atol=1e-5,
        rtol=1e-5,
    )
    assert prediction.raw_spatial_logits is not None
    assert prediction.spatial_residual_logits is not None
    torch.testing.assert_close(
        prediction.spatial_logits,
        prediction.raw_spatial_logits + prediction.spatial_residual_logits,
    )
    assert torch.isfinite(prediction.context_log_likelihood_ratios).all()
    with pytest.raises(TypeError):
        model(runtime, query_patches, support_patches, torch.zeros((1, 2, 2, 2)))


def test_context_only_forward_skips_texture_branch_and_keeps_context_output() -> None:
    torch.manual_seed(11)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
    ).eval()
    runtime = _runtime()
    full = model(
        runtime=runtime,
        query_rgb_patches=torch.rand(2, 3, 9, 9),
        support_rgb_patches=torch.rand(2, 2, 2, 3, 9, 9),
    )

    class _TextureMustNotRun(torch.nn.Module):
        def forward(self, _: torch.Tensor) -> torch.Tensor:  # pragma: no cover - must not execute
            raise AssertionError("context-only forward invoked the RGB texture branch")

    model.texture_encoder = _TextureMustNotRun()
    context_only = model(runtime=runtime, context_only=True)

    torch.testing.assert_close(
        context_only.context_log_likelihood_ratios,
        full.context_log_likelihood_ratios,
    )
    torch.testing.assert_close(
        context_only.spatial_logits,
        torch.zeros_like(context_only.spatial_logits),
    )


def test_context_only_activation_checkpoint_preserves_context_gradients() -> None:
    torch.manual_seed(13)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        activation_checkpointing=True,
        context_encoder_arch="absolute_cross_attention_v3",
    ).train()
    prediction = model(runtime=_runtime(), context_only=True)
    prediction.context_log_likelihood_ratios.sum().backward()
    final = model.context_identity_head[-1]
    assert isinstance(final, torch.nn.Linear)
    assert float(final.weight.detach().std()) == pytest.approx(
        CONTEXT_IDENTITY_HEAD_FINAL_WEIGHT_STD, rel=0.6
    )
    assert torch.count_nonzero(final.weight.detach()) > 0
    assert torch.count_nonzero(final.bias.detach()) == 0
    gradient = next(model.context_encoders.parameters()).grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_rgb_cost_volume_only_isolated_from_context_and_rejects_padded_windows() -> None:
    torch.manual_seed(17)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    runtime = _runtime()
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        activation_checkpointing=True,
    ).train()
    query_patches = torch.rand(2, 3, 9, 9)
    support_patches = torch.rand(2, 2, 2, 3, 9, 9)

    full = model(
        runtime=runtime,
        query_rgb_patches=query_patches,
        support_rgb_patches=support_patches,
    )
    model.zero_grad(set_to_none=True)
    rgb_only = model(
        runtime=runtime,
        query_rgb_patches=query_patches,
        support_rgb_patches=support_patches,
        rgb_cost_volume_only=True,
    )

    assert rgb_only.raw_spatial_logits is not None
    assert rgb_only.spatial_residual_logits is not None
    torch.testing.assert_close(rgb_only.spatial_logits, full.raw_spatial_logits)
    torch.testing.assert_close(
        rgb_only.context_log_likelihood_ratios,
        torch.zeros_like(rgb_only.context_log_likelihood_ratios),
    )
    torch.testing.assert_close(
        rgb_only.spatial_residual_logits,
        torch.zeros_like(rgb_only.spatial_residual_logits),
    )
    torch.testing.assert_close(
        rgb_only.joint_log_probabilities.exp().sum(dim=-1),
        torch.ones_like(rgb_only.non_dustbin_logits),
    )
    # First point/candidate support coordinates are inside the tensor but too
    # close to the image border for a 4px RGB patch radius.  They are fixed
    # non-support rather than border-padded visual evidence.
    assert not bool(rgb_only.edge_usable[0, 0].any())

    rgb_only.spatial_logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.texture_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.context_encoders.parameters())

    with pytest.raises(ValueError, match="exclusive"):
        model(runtime=runtime, context_only=True, rgb_cost_volume_only=True)


def test_rgb_only_model_can_omit_context_grid_buffers_and_load_checkpoint_state() -> None:
    torch.manual_seed(23)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 6), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 5), dim=-1),
    }
    common = {
        "image_sizes": torch.full((10, 2), 64.0),
        "search_radius_px": 2.0,
        "context_radius_px": 2.0,
        "step_px": 1.0,
        "texture_feature_dim": 4,
        "hidden_dim": 8,
        "edge_chunk_size": 4,
    }
    full = CandidatePoseRGBSpatialLikelihood(sources=source_grids, **common).eval()
    source_free = CandidatePoseRGBSpatialLikelihood(
        sources=None,
        context_source_dimensions={
            name: int(grid.shape[-1]) for name, grid in source_grids.items()
        },
        **common,
    ).eval()
    source_free.load_state_dict(full.state_dict(), strict=True)
    assert not source_free.context_sources_available
    assert not hasattr(source_free, "_radio_final_grid")

    query_patches = torch.rand(2, 3, 9, 9)
    support_patches = torch.rand(2, 2, 2, 3, 9, 9)
    expected = full(
        runtime=_runtime(),
        query_rgb_patches=query_patches,
        support_rgb_patches=support_patches,
        rgb_cost_volume_only=True,
    )
    actual = source_free(
        runtime=_runtime(),
        query_rgb_patches=query_patches,
        support_rgb_patches=support_patches,
        rgb_cost_volume_only=True,
    )
    torch.testing.assert_close(actual.spatial_logits, expected.spatial_logits)
    torch.testing.assert_close(
        actual.joint_log_probabilities, expected.joint_log_probabilities
    )
    with pytest.raises(RuntimeError, match="intentionally omitted"):
        source_free(
            runtime=_runtime(),
            query_rgb_patches=query_patches,
            support_rgb_patches=support_patches,
        )


def test_context_identity_losses_use_target_join_after_target_free_edge_outputs() -> None:
    runtime = _runtime()
    context = torch.tensor(
        [
            [[2.0, 1.5], [-1.0, -1.0]],
            [[-1.0, -1.0], [2.5, 2.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        **{**_prediction().__dict__, "context_log_likelihood_ratios": context}
    )
    observed = torch.tensor([[True, False], [False, True]])
    logits, usable = context_candidate_logit_mixture(runtime=runtime, prediction=prediction)
    assert bool(usable.all())
    assert torch.argmax(logits, dim=1).tolist() == [0, 1]
    identity_loss, identity_metrics = context_identity_cross_entropy_loss(
        runtime=runtime,
        prediction=prediction,
        target_observed=observed,
    )
    assert identity_metrics["context_identity_active_rows"] == pytest.approx(2.0)
    assert identity_metrics["context_identity_top1_accuracy"] == pytest.approx(1.0)
    identity_loss.backward(retain_graph=True)
    assert context.grad is not None

    permuted_context = torch.zeros_like(context, requires_grad=True)
    permuted_prediction = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **_prediction().__dict__,
            "context_log_likelihood_ratios": permuted_context,
        }
    )
    permutation_loss, permutation_metrics = context_identity_support_permutation_margin_loss(
        runtime=runtime,
        prediction=prediction,
        permuted_runtime=permute_runtime_support_appearance(runtime, shift=1),
        permuted_prediction=permuted_prediction,
        target_observed=observed,
        margin=0.25,
    )
    assert permutation_metrics["context_identity_permutation_active_rows"] == pytest.approx(2.0)
    assert permutation_metrics["context_identity_permutation_mean_gap"] > 0.0
    permutation_loss.backward()
    assert permuted_context.grad is not None
    with pytest.raises(ValueError, match="targets"):
        context_identity_cross_entropy_loss(
            runtime=runtime,
            prediction=prediction,
            target_observed=torch.tensor([[True, True], [False, True]]),
        )


def test_score_component_prediction_neutralizes_unselected_branches() -> None:
    prediction = _prediction()
    raw = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).reshape(1, 1, 1, 9)
    raw = raw.expand(2, 2, 2, -1).clone()
    residual = torch.full_like(raw, 0.25)
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **prediction.__dict__,
            "non_dustbin_logits": torch.full((2, 2, 2), 2.0),
            "raw_spatial_logits": raw,
            "spatial_residual_logits": residual,
        }
    )
    rgb = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction, component="rgb_cost_volume"
    )
    rgb_with_dustbin = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction, component="rgb_cost_volume_with_dustbin"
    )
    context = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction, component="context_only"
    )
    torch.testing.assert_close(
        rgb.context_log_likelihood_ratios,
        torch.zeros_like(rgb.context_log_likelihood_ratios),
    )
    torch.testing.assert_close(
        context.joint_log_probabilities[..., :-1].exp(),
        torch.full_like(context.joint_log_probabilities[..., :-1], 0.5 / 9.0),
    )
    torch.testing.assert_close(
        context.joint_log_probabilities[..., -1].exp(),
        torch.full_like(context.joint_log_probabilities[..., -1], 0.5),
    )
    assert bool(
        torch.all(
            rgb_with_dustbin.joint_log_probabilities[..., -1].exp()
            < rgb.joint_log_probabilities[..., -1].exp()
        )
    )
    without_components = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **prediction.__dict__,
            "raw_spatial_logits": None,
            "spatial_residual_logits": None,
        }
    )
    with pytest.raises(ValueError, match="decomposed"):
        candidate_pose_rgb_spatial_score_component_prediction(
            prediction=without_components, component="rgb_cost_volume"
        )


def test_context_window_configuration_is_explicit_and_does_not_change_default() -> None:
    assert resolve_candidate_pose_rgb_spatial_context_windows() == {
        "radio_final": 15,
        "radio_intermediate": 15,
        "alike": 13,
    }
    assert resolve_candidate_pose_rgb_spatial_context_windows(
        {"radio_final": 9, "radio_intermediate": 7, "alike": 11}
    ) == {"radio_final": 9, "radio_intermediate": 7, "alike": 11}
    with pytest.raises(ValueError, match="odd"):
        resolve_candidate_pose_rgb_spatial_context_windows(
            {"radio_final": 8, "radio_intermediate": 7, "alike": 11}
        )


def test_context_encoder_architecture_is_explicit() -> None:
    assert resolve_candidate_pose_rgb_spatial_context_encoder_arch() == "conv_v1"
    assert (
        resolve_candidate_pose_rgb_spatial_context_encoder_arch("cross_attention_v2")
        == "cross_attention_v2"
    )
    assert (
        resolve_candidate_pose_rgb_spatial_context_encoder_arch("absolute_cross_attention_v3")
        == "absolute_cross_attention_v3"
    )
    with pytest.raises(ValueError, match="architecture"):
        resolve_candidate_pose_rgb_spatial_context_encoder_arch("implicit_unknown")


def test_cross_attention_context_is_candidate_and_view_permutation_equivariant() -> None:
    torch.manual_seed(19)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(torch.randn(10, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    runtime = _runtime()
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        context_encoder_arch="cross_attention_v2",
    ).eval()
    query_patches = torch.rand(runtime.point_count, 3, 9, 9)
    support_patches = torch.rand(
        runtime.point_count,
        runtime.candidate_count,
        runtime.support_view_count,
        3,
        9,
        9,
    )
    candidate_order = torch.tensor([1, 0])
    view_order = torch.tensor([1, 0])
    permuted_runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=runtime.support_image_indices.index_select(1, candidate_order).index_select(
            2, view_order
        ),
        support_xy=runtime.support_xy.index_select(1, candidate_order).index_select(2, view_order),
        support_view_valid=runtime.support_view_valid.index_select(1, candidate_order).index_select(
            2, view_order
        ),
        candidate_view_weights=runtime.candidate_view_weights.index_select(
            1, candidate_order
        ).index_select(2, view_order),
        candidate_probabilities=runtime.candidate_probabilities.index_select(1, candidate_order),
        null_probabilities=runtime.null_probabilities,
    )
    with torch.no_grad():
        prediction = model(
            runtime=runtime,
            query_rgb_patches=query_patches,
            support_rgb_patches=support_patches,
        )
        permuted = model(
            runtime=permuted_runtime,
            query_rgb_patches=query_patches,
            support_rgb_patches=support_patches.index_select(1, candidate_order).index_select(
                2, view_order
            ),
        )
    torch.testing.assert_close(
        permuted.context_log_likelihood_ratios,
        prediction.context_log_likelihood_ratios.index_select(1, candidate_order).index_select(
            2, view_order
        ),
    )
    torch.testing.assert_close(
        permuted.joint_log_probabilities,
        prediction.joint_log_probabilities.index_select(1, candidate_order).index_select(
            2, view_order
        ),
    )


def test_absolute_context_position_control_and_candidate_slot_permutation_are_explicit() -> None:
    torch.manual_seed(29)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(torch.randn(10, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    runtime = _runtime()
    absolute = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        context_encoder_arch="absolute_cross_attention_v3",
    ).eval()
    with torch.no_grad():
        visual = absolute(runtime=runtime, context_only=True)
        position_only = absolute(
            runtime=runtime, context_only=True, context_appearance_mode="position_only"
        )
    assert visual.context_log_likelihood_ratios.shape == position_only.context_log_likelihood_ratios.shape
    assert torch.isfinite(position_only.context_log_likelihood_ratios).all()

    relative = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
        context_encoder_arch="cross_attention_v2",
    ).eval()
    with pytest.raises(ValueError, match="absolute_cross_attention"):
        relative(runtime=runtime, context_only=True, context_appearance_mode="position_only")

    order = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    permuted_runtime = permute_runtime_candidate_slots(runtime, permutations=order)
    torch.testing.assert_close(
        permuted_runtime.support_xy,
        runtime.support_xy.gather(1, order[:, :, None, None].expand(-1, -1, 2, 2)),
    )
    torch.testing.assert_close(
        permuted_runtime.candidate_probabilities,
        runtime.candidate_probabilities.gather(1, order),
    )
def test_query_texture_is_encoded_once_per_point_not_once_per_candidate_edge() -> None:
    torch.manual_seed(11)
    source_grids = {
        "radio_final": torch.nn.functional.normalize(torch.randn(10, 16, 16, 4), dim=-1),
        "radio_intermediate": torch.nn.functional.normalize(
            torch.randn(10, 16, 16, 4), dim=-1
        ),
        "alike": torch.nn.functional.normalize(torch.randn(10, 32, 32, 4), dim=-1),
    }
    runtime = _runtime()
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_grids,
        image_sizes=torch.full((10, 2), 64.0),
        search_radius_px=2.0,
        context_radius_px=2.0,
        step_px=1.0,
        texture_feature_dim=4,
        hidden_dim=8,
        edge_chunk_size=4,
    )
    query_patches = torch.rand(runtime.point_count, 3, 9, 9)
    support_patches = torch.rand(
        runtime.point_count,
        runtime.candidate_count,
        runtime.support_view_count,
        3,
        9,
        9,
    )
    encoded_batch_sizes: list[int] = []

    def record_texture_batch(_module, inputs) -> None:
        encoded_batch_sizes.append(int(inputs[0].shape[0]))

    handle = model.texture_encoder.register_forward_pre_hook(record_texture_batch)
    try:
        prediction = model(
            runtime=runtime,
            query_rgb_patches=query_patches,
            support_rgb_patches=support_patches,
        )
        prediction.spatial_logits.mean().backward()
    finally:
        handle.remove()

    edge_count = runtime.point_count * runtime.candidate_count * runtime.support_view_count
    assert encoded_batch_sizes[0] == runtime.point_count
    assert sum(encoded_batch_sizes) == runtime.point_count + edge_count
    assert any(parameter.grad is not None for parameter in model.texture_encoder.parameters())


def test_pose_score_has_one_finite_value_per_hypothesis() -> None:
    score = score_candidate_pose_rgb_spatial_batch(
        runtime=_runtime(),
        prediction=_prediction(),
        candidate_projection_offsets_xy=torch.zeros((3, 2, 2, 2)),
        candidate_projection_valid=torch.ones((3, 2, 2), dtype=torch.bool),
    )
    assert score.pose_log_likelihood_ratios.shape == (3,)
    assert torch.isfinite(score.pose_log_likelihood_ratios).all()


def test_selected_candidate_view_score_is_target_free_and_rejects_out_of_window_support() -> None:
    scores, usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=_runtime(),
        prediction=_prediction(),
        point_indices=torch.tensor([0, 1]),
        candidate_indices=torch.tensor([0, 1]),
        offsets_xy=torch.tensor([[0.0, 0.0], [99.0, 99.0]]),
        missing_edge_log_likelihood_ratio=0.0,
    )
    assert scores.shape == (2,)
    assert bool(usable[0])
    assert not bool(usable[1])
    assert float(scores[1]) == pytest.approx(0.0)


def test_spatial_density_ignores_unsupervised_identity_anchors_and_balances_classes() -> None:
    prediction = _prediction()
    targets = torch.zeros((2, 2, 2), dtype=torch.float32)
    dustbin = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
    supervised = torch.tensor([[True, True], [False, False]], dtype=torch.bool)
    loss, metrics = spatial_density_nll(
        prediction=prediction,
        target_offsets_xy=targets,
        target_dustbin=dustbin,
        target_supervised=supervised,
        dustbin_weight=0.5,
        balance_observed_and_dustbin=True,
    )
    shifted_unsupervised = targets.clone()
    shifted_unsupervised[1] = 99.0
    shifted_loss, shifted_metrics = spatial_density_nll(
        prediction=prediction,
        target_offsets_xy=shifted_unsupervised,
        target_dustbin=dustbin,
        target_supervised=supervised,
        dustbin_weight=0.5,
        balance_observed_and_dustbin=True,
    )

    torch.testing.assert_close(loss, shifted_loss)
    assert metrics["spatial_density_active_edges"] == 4.0
    assert metrics["spatial_density_observed_edges"] == 2.0
    assert metrics["spatial_density_dustbin_edges"] == 2.0
    assert shifted_metrics["spatial_density_active_edges"] == 4.0
