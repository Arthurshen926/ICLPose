from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
    CandidateBidirectionalAbsoluteContextLikelihood,
    CONTEXT_ATTENTION_SCALES,
    CandidateContextAttentionProbe,
    ContextAttentionRuntimeArrays,
    _candidate_set_visual_statistics,
    context_attention_global_region_sizes,
    context_valid_mask,
    crop_anchor_aligned_grid_absolute_coordinates,
    load_context_attention_source_headers,
    load_context_attention_sources,
    masked_view_log_mean,
)


def _unit_grid(image_count: int, grid_size: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17 + grid_size)
    values = torch.randn(
        image_count, grid_size, grid_size, dimension, generator=generator, dtype=torch.float32
    )
    return torch.nn.functional.normalize(values, p=2, dim=-1)


def _write_raw_context_cache(
    path, *, image_ids: np.ndarray, image_sizes: np.ndarray, grid_name: str, grid: np.ndarray, metadata: dict[str, object]
) -> None:
    grid_size = int(str(grid_name)[4 : -len("_descriptors")])
    np.savez_compressed(
        path,
        image_ids=np.asarray(image_ids),
        image_sizes=np.asarray(image_sizes, dtype=np.int64),
        **{grid_name: np.asarray(grid, dtype=np.float32)},
        metadata_json=np.asarray(
            __import__("json").dumps(
                {**metadata, "spatial_grid_sizes": [grid_size]}, sort_keys=True
            )
        ),
    )


def test_context_source_loader_accepts_current_raw_radio_final_schema_without_sizes(tmp_path) -> None:
    image_ids = np.asarray(["seq/a.png", "seq/b.png"])
    image_sizes = np.asarray([[32, 24], [32, 24]], dtype=np.int64)
    final_grid = _unit_grid(2, 16, 4).numpy().reshape(2, 256, 4)
    intermediate_grid = _unit_grid(2, 16, 4).numpy().reshape(2, 256, 4)
    alike_grid = _unit_grid(2, 32, 4).numpy().reshape(2, 1024, 4)
    common = {
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "source_image_manifest_sha256": "source-manifest",
    }
    final_path = tmp_path / "raw_final.npz"
    # The current raw final cache intentionally has no image_sizes field.  The
    # matched intermediate cache carries the authoritative processed-RGB sizes.
    np.savez_compressed(
        final_path,
        image_ids=image_ids,
        grid16_descriptors=final_grid,
        metadata_json=np.asarray(
            __import__("json").dumps(
                {
                    **common,
                    "format": "radio_image_multiscale_context_v1",
                    "radio_checkpoint_sha256": "radio-sha",
                    "image_source_contract": {
                        "source_image_dimensions": {"64x48": 2}
                    },
                },
                sort_keys=True,
            )
        ),
    )
    intermediate_path = tmp_path / "intermediate.npz"
    _write_raw_context_cache(
        intermediate_path,
        image_ids=image_ids,
        image_sizes=image_sizes,
        grid_name="grid16_descriptors",
        grid=intermediate_grid,
        metadata={
            **common,
            "format": "radio_intermediate_image_spatial_context_v1",
            "radio_checkpoint_sha256": "radio-sha",
            "intermediate_index": -6,
            "pca_fit_scope": "mapping_train_images_only",
        },
    )
    alike_path = tmp_path / "alike.npz"
    _write_raw_context_cache(
        alike_path,
        image_ids=image_ids,
        image_sizes=image_sizes,
        grid_name="grid32_descriptors",
        grid=alike_grid,
        metadata={
            **common,
            "format": "alike_image_spatial_context_v1",
            "alike_checkpoint_sha256": "alike-sha",
        },
    )

    sources = load_context_attention_sources(
        radio_final_context_cache=final_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        expected_radio_checkpoint="radio-sha",
        require_equal_descriptor_dimensions=True,
    )

    assert [source.name for source in sources] == [
        "radio_final",
        "radio_intermediate",
        "alike",
    ]
    np.testing.assert_array_equal(sources[0].image_sizes, image_sizes)
    assert sources[0].metadata["coordinate_bridge"]["raw_pixels_per_aligned_pixel"] == 2.0


def test_lightweight_context_headers_preserve_rgb_only_lineage_without_grids(tmp_path) -> None:
    image_ids = np.asarray(["seq/a.png", "seq/b.png"])
    image_sizes = np.asarray([[32, 24], [32, 24]], dtype=np.int64)
    common = {
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "source_image_manifest_sha256": "source-manifest",
    }
    final_path = tmp_path / "raw_final.npz"
    np.savez_compressed(
        final_path,
        image_ids=image_ids,
        # Descriptor values intentionally need not be read by the header-only
        # path; their shape is sufficient to rebuild checkpoint modules.
        grid16_descriptors=np.zeros((2, 256, 7), dtype=np.float16),
        metadata_json=np.asarray(
            __import__("json").dumps(
                {
                    **common,
                    "format": "radio_image_multiscale_context_v1",
                    "radio_checkpoint_sha256": "radio-sha",
                    "image_source_contract": {
                        "source_image_dimensions": {"64x48": 2}
                    },
                },
                sort_keys=True,
            )
        ),
    )
    intermediate_path = tmp_path / "intermediate.npz"
    _write_raw_context_cache(
        intermediate_path,
        image_ids=image_ids,
        image_sizes=image_sizes,
        grid_name="grid16_descriptors",
        grid=np.zeros((2, 256, 11), dtype=np.float16),
        metadata={
            **common,
            "format": "radio_intermediate_image_spatial_context_v1",
            "radio_checkpoint_sha256": "radio-sha",
            "intermediate_index": -6,
        },
    )
    alike_path = tmp_path / "alike.npz"
    _write_raw_context_cache(
        alike_path,
        image_ids=image_ids,
        image_sizes=image_sizes,
        grid_name="grid32_descriptors",
        grid=np.zeros((2, 1024, 5), dtype=np.float16),
        metadata={
            **common,
            "format": "alike_image_spatial_context_v1",
            "alike_checkpoint_sha256": "alike-sha",
        },
    )

    headers = load_context_attention_source_headers(
        radio_final_context_cache=final_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
        expected_radio_checkpoint="radio-sha",
    )

    np.testing.assert_array_equal(headers.image_ids, image_ids)
    np.testing.assert_array_equal(headers.image_sizes, image_sizes)
    assert headers.descriptor_dimensions == {
        "radio_final": 7,
        "radio_intermediate": 11,
        "alike": 5,
    }
    assert (
        headers.metadata_by_name["radio_final"]["coordinate_bridge"]
        ["raw_pixels_per_aligned_pixel"]
        == 2.0
    )


def _model(
    *,
    family: str = "context_attention_multiscale_context_only",
    position_encoding: str | None = None,
) -> CandidateContextAttentionProbe:
    sources = {
        scale.name: _unit_grid(3, scale.grid_size, 4) for scale in CONTEXT_ATTENTION_SCALES
    }
    runtime = ContextAttentionRuntimeArrays(
        query_image_indices=np.asarray([0], dtype=np.int64),
        support_image_indices=np.asarray([[[1, 2], [1, 2]]], dtype=np.int64),
        support_xy=np.asarray(
            [[[[50.0, 50.0], [50.0, 50.0]], [[50.0, 50.0], [50.0, 50.0]]]],
            dtype=np.float32,
        ),
        view_valid=np.asarray([[[True, True], [True, True]]]),
    )
    model = CandidateContextAttentionProbe(
        family=family,
        sources=sources,
        image_sizes=torch.as_tensor([[100.0, 100.0]] * 3),
        runtime=runtime,
        query_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        base_candidate_probabilities=np.asarray([[0.4, 0.5]], dtype=np.float32),
        base_null_probabilities=np.asarray([0.1], dtype=np.float32),
        hidden_dim=8,
        heads=2,
        dropout=0.0,
        **({"position_encoding": position_encoding} if position_encoding is not None else {}),
    )
    # Make outputs depend on the context encoder rather than the intentionally
    # zero-initialized residual head.
    with torch.no_grad():
        model.view_head[-1].weight.fill_(0.05)
        model.view_head[-1].bias.zero_()
    return model.eval()


def _v2_model(
    *,
    family: str = "bidirectional_absolute_visual_v2",
    candidate_permutation: tuple[int, int] = (0, 1),
    view_permutation: tuple[int, int] = (0, 1),
) -> CandidateBidirectionalAbsoluteContextLikelihood:
    sources = {
        scale.name: _unit_grid(3, scale.grid_size, 8)
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
    }
    base_candidate = np.asarray([[0.4, 0.5]], dtype=np.float32)[:, candidate_permutation]
    support_indices = np.asarray([[[1, 2], [1, 2]]], dtype=np.int64)
    support_xy = np.asarray(
        [[[[45.0, 55.0], [60.0, 40.0]], [[35.0, 65.0], [55.0, 45.0]]]],
        dtype=np.float32,
    )
    support_indices = support_indices[:, candidate_permutation, :][:, :, view_permutation]
    support_xy = support_xy[:, candidate_permutation, :][:, :, view_permutation]
    runtime = ContextAttentionRuntimeArrays(
        query_image_indices=np.asarray([0], dtype=np.int64),
        support_image_indices=support_indices,
        support_xy=support_xy,
        view_valid=np.ones((1, 2, 2), dtype=bool),
    )
    return CandidateBidirectionalAbsoluteContextLikelihood(
        family=family,
        sources=sources,
        image_sizes=torch.as_tensor([[100.0, 100.0]] * 3),
        runtime=runtime,
        query_xy=np.asarray([[50.0, 50.0]], dtype=np.float32),
        base_candidate_probabilities=base_candidate,
        base_null_probabilities=np.asarray([0.1], dtype=np.float32),
        hidden_dim=8,
        heads=2,
        dropout=0.0,
    ).eval()


def test_context_mask_removes_the_entire_anchor_neighbourhood() -> None:
    valid = torch.ones((1, 25), dtype=torch.bool)
    masked = context_valid_mask(valid, window_size=5, center_mask_radius=1)

    assert int(masked.sum()) == 16
    assert not bool(masked[0, 2 * 5 + 2])
    assert not bool(masked[0, 1 * 5 + 1])
    assert bool(masked[0, 0])


def test_context_only_family_is_invariant_to_center_descriptor_changes() -> None:
    model = _model()
    rows = torch.as_tensor([0], dtype=torch.long)
    before, null_before, _views, _logits = model(rows)

    with torch.no_grad():
        for scale in CONTEXT_ATTENTION_SCALES:
            grid = model._grid(scale.name)
            center = scale.grid_size // 2
            replacement = torch.nn.functional.normalize(
                torch.tensor([7.0, 3.0, -5.0, 11.0], dtype=torch.float32), p=2, dim=0
            ).to(dtype=grid.dtype)
            grid[:, center, center] = replacement
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_per_view_marginal_is_permutation_invariant_and_handles_padding() -> None:
    logits = torch.as_tensor([[[0.2, 1.1, 3.0], [-4.0, 2.0, 7.0]]])
    valid = torch.as_tensor([[[True, True, False], [False, True, False]]])
    original = masked_view_log_mean(logits, valid)
    permutation = torch.as_tensor([1, 0, 2])
    permuted = masked_view_log_mean(logits[:, :, permutation], valid[:, :, permutation])

    torch.testing.assert_close(original, permuted)
    expected = torch.logsumexp(torch.as_tensor([0.2, 1.1]), dim=0) - np.log(2.0)
    torch.testing.assert_close(original[0, 0], expected)
    torch.testing.assert_close(original[0, 1], torch.as_tensor(2.0))


def test_per_view_marginal_has_finite_backward_for_an_empty_padded_candidate() -> None:
    logits = torch.tensor([[[0.4, -0.2], [3.0, -7.0]]], requires_grad=True)
    valid = torch.as_tensor([[[True, True], [False, False]]])

    residual = masked_view_log_mean(logits, valid)
    residual.sum().backward()

    torch.testing.assert_close(residual[0, 1], torch.as_tensor(0.0))
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros((2,)))


def test_context_probe_probability_mass_is_conserved() -> None:
    model = _model()
    candidate, null, view_logits, _logits = model(torch.as_tensor([0], dtype=torch.long))

    torch.testing.assert_close(candidate.sum(dim=1) + null, torch.ones((1,)))
    assert view_logits.shape == (1, 2, 2)


def test_absolute_crop_coordinates_preserve_full_image_phase() -> None:
    grid = torch.ones((1, 5, 5, 2), dtype=torch.float32)
    coordinates = crop_anchor_aligned_grid_absolute_coordinates(
        image_grids=grid,
        image_sizes=torch.as_tensor([[100.0, 100.0]]),
        image_indices=torch.as_tensor([0, 0]),
        xy=torch.as_tensor([[10.0, 10.0], [90.0, 90.0]]),
        window_size=3,
    )

    centre = 4
    torch.testing.assert_close(coordinates[0, centre], torch.as_tensor([0.0, 0.0]))
    torch.testing.assert_close(coordinates[1, centre], torch.as_tensor([1.0, 1.0]))


def test_absolute_position_only_control_is_descriptor_invariant() -> None:
    model = _model(
        family="absolute_phase_attention_position_only",
        position_encoding=ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    )
    rows = torch.as_tensor([0], dtype=torch.long)
    before, null_before, _views, _logits = model(rows)

    with torch.no_grad():
        for scale in CONTEXT_ATTENTION_SCALES:
            model._grid(scale.name).normal_()
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_context_family_cannot_silently_mix_position_spaces() -> None:
    with pytest.raises(ValueError, match="family and position encoding differ"):
        _model(position_encoding=ABSOLUTE_DUAL_FRAME_POSITION_ENCODING)


def test_bidirectional_absolute_context_probability_and_view_semantics() -> None:
    model = _v2_model()
    details = model.forward_with_details(torch.as_tensor([0], dtype=torch.long))

    torch.testing.assert_close(
        details["candidate_probabilities"].sum(dim=1) + details["null_probabilities"],
        torch.ones((1,)),
    )
    assert details["view_logits"].shape == (1, 2, 2)
    assert details["view_log_probabilities"].shape == (1, 2, 2)
    assert details["null_log_likelihood_ratio"].shape == (1,)
    assert details["per_scale_view_logits"].shape == (
        1,
        2,
        2,
        len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES),
    )
    torch.testing.assert_close(
        torch.exp(details["view_log_probabilities"]).sum(dim=2), torch.ones((1, 2))
    )
    # Empty ALIKE global regions are generated lazily instead of registered as
    # [N, 0, C] DDP buffers, which NCCL's initial broadcast cannot coalesce.
    assert "_alike_global" not in dict(model.named_buffers())
    assert model._global("alike").shape == (3, 0, 8)


def test_bidirectional_absolute_context_is_candidate_and_view_permutation_equivariant() -> None:
    baseline = _v2_model()
    candidate_permuted = _v2_model(candidate_permutation=(1, 0))
    view_permuted = _v2_model(view_permutation=(1, 0))
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    rows = torch.as_tensor([0], dtype=torch.long)

    base_candidate, base_null, base_views, _ = baseline(rows)
    candidate_values, candidate_null, candidate_views, _ = candidate_permuted(rows)
    view_values, view_null, view_views, _ = view_permuted(rows)

    torch.testing.assert_close(candidate_values[:, [1, 0]], base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_views[:, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_values, base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_views[:, :, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_null, base_null, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_null, base_null, atol=1e-6, rtol=1e-6)


def test_bidirectional_position_control_cannot_use_descriptor_values() -> None:
    model = _v2_model(family="bidirectional_absolute_position_control_v2")
    rows = torch.as_tensor([0], dtype=torch.long)
    before, null_before, _views, _logits = model(rows)
    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_candidate_visual_null_statistics_are_candidate_permutation_invariant() -> None:
    values = torch.as_tensor([[0.3, -1.0, 2.0]])
    valid = torch.as_tensor([[True, False, True]])
    permutation = torch.as_tensor([2, 0, 1])

    original = _candidate_set_visual_statistics(values, valid)
    permuted = _candidate_set_visual_statistics(values[:, permutation], valid[:, permutation])

    torch.testing.assert_close(original, permuted)


def test_bidirectional_visual_null_residual_responds_to_image_evidence() -> None:
    model = _v2_model()
    rows = torch.as_tensor([0], dtype=torch.long)
    with torch.no_grad():
        model.fusion_head[-1].weight.fill_(0.05)
        model.fusion_head[-1].bias.zero_()
        model.null_head[-1].weight.fill_(0.05)
        model.null_head[-1].bias.zero_()
    _candidate_before, null_before, _views, _logits = model(rows)

    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    _candidate_after, null_after, _views, _logits = model(rows)

    assert not torch.allclose(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_raw_v3_uses_shared_projection_and_preserves_raw_cost_volume_evidence() -> None:
    model = _v2_model(family="bidirectional_absolute_raw_visual_v3")
    rows = torch.as_tensor([0], dtype=torch.long)

    assert context_attention_global_region_sizes("bidirectional_absolute_raw_v3") == {
        "radio_final": 6,
        "radio_intermediate": 6,
        "alike": 0,
    }
    for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
        encoder = model.encoders[scale.name]
        assert encoder.shared_descriptor_projection is True
        assert encoder.include_raw_descriptor_statistics is True
        assert encoder.descriptor_projection is not None
        assert encoder.query_projection is None
        assert encoder.support_projection is None
    assert model._global("radio_final").shape == (3, 36, 8)
    assert model.raw_scale_heads is not None

    with torch.no_grad():
        for head in model.raw_scale_heads.values():
            head.weight.zero_()
            head.bias.zero_()
        # Isolate a raw frozen-descriptor statistic from the learned attention
        # path.  The candidate residual must react before any coarse prior is
        # added to it.
        model.raw_scale_heads["radio_final"].weight[0, 0] = 1.0
    before = model.forward_with_details(rows)["candidate_log_likelihood_ratios"]
    with torch.no_grad():
        replacement = torch.randn_like(model._grid("radio_final")[1])
        model._grid("radio_final")[1].copy_(
            torch.nn.functional.normalize(replacement.float(), p=2, dim=-1).to(
                dtype=model._grid("radio_final").dtype
            )
        )
    after = model.forward_with_details(rows)["candidate_log_likelihood_ratios"]

    assert not torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_raw_v3_position_control_masks_raw_descriptor_statistics() -> None:
    model = _v2_model(family="bidirectional_absolute_raw_position_control_v3")
    rows = torch.as_tensor([0], dtype=torch.long)
    assert model.raw_scale_heads is not None
    with torch.no_grad():
        for head in model.raw_scale_heads.values():
            head.weight.fill_(0.05)
            head.bias.zero_()
    before, null_before, _views, _logits = model(rows)
    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_raw_v3_is_candidate_and_view_permutation_equivariant() -> None:
    baseline = _v2_model(family="bidirectional_absolute_raw_visual_v3")
    candidate_permuted = _v2_model(
        family="bidirectional_absolute_raw_visual_v3", candidate_permutation=(1, 0)
    )
    view_permuted = _v2_model(
        family="bidirectional_absolute_raw_visual_v3", view_permutation=(1, 0)
    )
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    rows = torch.as_tensor([0], dtype=torch.long)

    base_candidate, base_null, base_views, _ = baseline(rows)
    candidate_values, candidate_null, candidate_views, _ = candidate_permuted(rows)
    view_values, view_null, view_views, _ = view_permuted(rows)

    torch.testing.assert_close(candidate_values[:, [1, 0]], base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_views[:, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_values, base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_views[:, :, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_null, base_null, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_null, base_null, atol=1e-6, rtol=1e-6)


def test_raw_layout_v4_uses_no_attention_and_retains_global_phase_costs() -> None:
    model = _v2_model(family="bidirectional_absolute_raw_layout_visual_v4")
    rows = torch.as_tensor([0], dtype=torch.long)

    assert len(model.encoders) == 0
    assert len(model.scale_heads) == 0
    assert model.fusion_head is None
    assert model.raw_scale_heads is None
    assert model.raw_layout_heads is not None
    # Six compact statistics, three 6x6 phase vectors, and the matched
    # query/support anchor-position control.
    assert model.raw_layout_heads["radio_final"].in_features == 6 + 3 * 36 + 6
    assert model.raw_layout_heads["alike"].in_features == 12

    with torch.no_grad():
        for head in model.raw_layout_heads.values():
            head.weight.zero_()
            head.bias.zero_()
        model.raw_layout_heads["radio_final"].weight[0, 6] = 1.0
    before = model.forward_with_details(rows)["candidate_log_likelihood_ratios"]
    with torch.no_grad():
        replacement = torch.randn_like(model._grid("radio_final")[1])
        model._grid("radio_final")[1].copy_(
            torch.nn.functional.normalize(replacement.float(), p=2, dim=-1).to(
                dtype=model._grid("radio_final").dtype
            )
        )
    after = model.forward_with_details(rows)["candidate_log_likelihood_ratios"]

    assert not torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_raw_layout_v4_position_control_is_descriptor_invariant() -> None:
    model = _v2_model(family="bidirectional_absolute_raw_layout_position_control_v4")
    rows = torch.as_tensor([0], dtype=torch.long)
    assert model.raw_layout_heads is not None
    with torch.no_grad():
        for head in model.raw_layout_heads.values():
            head.weight.fill_(0.05)
            head.bias.zero_()
    before, null_before, _views, _logits = model(rows)
    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    after, null_after, _views, _logits = model(rows)

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(null_before, null_after, atol=1e-6, rtol=1e-6)


def test_raw_layout_v4_is_candidate_and_view_permutation_equivariant() -> None:
    baseline = _v2_model(family="bidirectional_absolute_raw_layout_visual_v4")
    candidate_permuted = _v2_model(
        family="bidirectional_absolute_raw_layout_visual_v4", candidate_permutation=(1, 0)
    )
    view_permuted = _v2_model(
        family="bidirectional_absolute_raw_layout_visual_v4", view_permutation=(1, 0)
    )
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    rows = torch.as_tensor([0], dtype=torch.long)

    base_candidate, base_null, base_views, _ = baseline(rows)
    candidate_values, candidate_null, candidate_views, _ = candidate_permuted(rows)
    view_values, view_null, view_views, _ = view_permuted(rows)

    torch.testing.assert_close(candidate_values[:, [1, 0]], base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_views[:, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_values, base_candidate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_views[:, :, [1, 0]], base_views, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_null, base_null, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(view_null, base_null, atol=1e-6, rtol=1e-6)


def test_raw_layout_v5_keeps_geometry_and_identity_probability_spaces_separate() -> None:
    model = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    rows = torch.as_tensor([0], dtype=torch.long)

    assert model.raw_layout_heads is not None
    assert model.identity_raw_layout_heads is not None
    assert set(model.identity_raw_layout_heads) == {"radio_final", "radio_intermediate"}
    with torch.no_grad():
        for head in model.raw_layout_heads.values():
            head.weight.zero_()
            head.bias.zero_()
        for head in model.identity_raw_layout_heads.values():
            head.weight.zero_()
            head.bias.zero_()
        model.raw_layout_heads["radio_final"].weight[0, 6] = 1.0
        model.identity_raw_layout_heads["radio_intermediate"].weight[0, 6] = -1.0

    details = model.forward_with_details(rows)
    torch.testing.assert_close(
        details["candidate_probabilities"].sum(dim=1) + details["null_probabilities"],
        torch.ones((1,)),
    )
    torch.testing.assert_close(
        details["identity_candidate_probabilities"].sum(dim=1)
        + details["identity_null_probabilities"],
        torch.ones((1,)),
    )
    assert details["identity_per_scale_view_logits"].shape[-1] == len(
        BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES
    )
    # ALIKE is deliberately spatial-only for V5 strict identity evidence.
    torch.testing.assert_close(
        details["identity_per_scale_view_logits"][..., 2],
        torch.zeros_like(details["identity_per_scale_view_logits"][..., 2]),
    )
    assert not torch.allclose(
        details["candidate_log_likelihood_ratios"],
        details["identity_candidate_log_likelihood_ratios"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_raw_layout_v5_position_control_masks_visual_evidence_for_both_heads() -> None:
    model = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_position_control_v5"
    )
    rows = torch.as_tensor([0], dtype=torch.long)
    assert model.raw_layout_heads is not None
    assert model.identity_raw_layout_heads is not None
    with torch.no_grad():
        for head in (*model.raw_layout_heads.values(), *model.identity_raw_layout_heads.values()):
            head.weight.fill_(0.05)
            head.bias.zero_()
    before = model.forward_with_details(rows)
    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    after = model.forward_with_details(rows)

    for key in (
        "candidate_probabilities",
        "null_probabilities",
        "identity_candidate_probabilities",
        "identity_null_probabilities",
    ):
        torch.testing.assert_close(before[key], after[key], atol=1e-6, rtol=1e-6)


def test_raw_layout_dynamic_pairwise_scores_replay_the_static_query_crop() -> None:
    model = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    rows = torch.as_tensor([0], dtype=torch.long)
    with torch.no_grad():
        assert model.raw_layout_heads is not None
        for head in model.raw_layout_heads.values():
            head.weight.fill_(0.05)
            head.bias.zero_()
    static = model.forward_with_details(rows)
    coordinates = model._query_xy.index_select(0, rows)[:, None].expand(-1, 2, -1)
    dynamic = model.forward_pairwise_raw_layout_at_query_xy(rows, coordinates)

    torch.testing.assert_close(dynamic["view_raw_scores"], static["view_logits"])
    torch.testing.assert_close(
        dynamic["per_scale_view_raw_scores"], static["per_scale_view_logits"]
    )
    assert bool(dynamic["query_projection_in_image"].all())
    torch.testing.assert_close(
        dynamic["support_view_available"].to(dtype=torch.float32),
        model._view_valid.index_select(0, rows).to(dtype=torch.float32),
    )
    assert not any("identity" in key for key in dynamic)


def test_raw_layout_dynamic_strict_identity_scores_replay_the_static_query_crop() -> None:
    model = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    rows = torch.as_tensor([0], dtype=torch.long)
    with torch.no_grad():
        assert model.identity_raw_layout_heads is not None
        for head in model.identity_raw_layout_heads.values():
            head.weight.fill_(0.05)
            head.bias.zero_()
    static = model.forward_with_details(rows)
    coordinates = model._query_xy.index_select(0, rows)[:, None].expand(-1, 2, -1)
    dynamic = model.forward_pairwise_identity_raw_layout_at_query_xy(rows, coordinates)

    torch.testing.assert_close(
        dynamic["identity_view_raw_scores"], static["identity_view_logits"]
    )
    torch.testing.assert_close(
        dynamic["identity_per_scale_view_raw_scores"],
        static["identity_per_scale_view_logits"],
    )
    torch.testing.assert_close(
        dynamic["identity_per_scale_view_raw_scores"][..., 2],
        torch.zeros_like(dynamic["identity_per_scale_view_raw_scores"][..., 2]),
    )
    assert bool(dynamic["query_projection_in_image"].all())
    assert not any(
        key in dynamic
        for key in ("identity_candidate_probabilities", "identity_null_probabilities", "view_raw_scores")
    )


def test_raw_layout_dynamic_pairwise_marks_outside_projections_and_rejects_nonfinite() -> None:
    model = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    rows = torch.as_tensor([0], dtype=torch.long)
    coordinates = model._query_xy.index_select(0, rows)[:, None].expand(-1, 2, -1).clone()
    coordinates[0, 1] = torch.as_tensor([-1.0, 100.0])
    dynamic = model.forward_pairwise_raw_layout_at_query_xy(rows, coordinates)

    assert bool(dynamic["query_projection_in_image"][0, 0].all())
    assert not bool(dynamic["query_projection_in_image"][0, 1].any())
    coordinates[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="dynamic query coordinates"):
        model.forward_pairwise_raw_layout_at_query_xy(rows, coordinates)


def test_raw_layout_dynamic_strict_identity_position_control_is_descriptor_invariant() -> None:
    model = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_position_control_v5"
    )
    rows = torch.as_tensor([0], dtype=torch.long)
    coordinates = model._query_xy.index_select(0, rows)[:, None].expand(-1, 2, -1)
    with torch.no_grad():
        assert model.identity_raw_layout_heads is not None
        for head in model.identity_raw_layout_heads.values():
            head.weight.fill_(0.05)
            head.bias.zero_()
    before = model.forward_pairwise_identity_raw_layout_at_query_xy(rows, coordinates)
    with torch.no_grad():
        for scale in BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES:
            model._grid(scale.name).normal_()
            model._global(scale.name).normal_()
    after = model.forward_pairwise_identity_raw_layout_at_query_xy(rows, coordinates)

    torch.testing.assert_close(
        before["identity_view_raw_scores"], after["identity_view_raw_scores"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        before["identity_per_scale_view_raw_scores"],
        after["identity_per_scale_view_raw_scores"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_raw_layout_dynamic_strict_identity_is_candidate_and_view_permutation_equivariant() -> None:
    baseline = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    candidate_permuted = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_visual_v5",
        candidate_permutation=(1, 0),
    )
    view_permuted = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_visual_v5",
        view_permutation=(1, 0),
    )
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        assert baseline.identity_raw_layout_heads is not None
        for index, head in enumerate(baseline.identity_raw_layout_heads.values()):
            head.weight.fill_(0.03 * float(index + 1))
            head.bias.fill_(0.01 * float(index + 1))
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    rows = torch.as_tensor([0], dtype=torch.long)
    coordinates = baseline._query_xy.index_select(0, rows)[:, None].expand(-1, 2, -1)

    base = baseline.forward_pairwise_identity_raw_layout_at_query_xy(rows, coordinates)
    candidate = candidate_permuted.forward_pairwise_identity_raw_layout_at_query_xy(
        rows, coordinates
    )
    view = view_permuted.forward_pairwise_identity_raw_layout_at_query_xy(rows, coordinates)
    for key in ("identity_view_raw_scores", "identity_per_scale_view_raw_scores"):
        torch.testing.assert_close(candidate[key][:, [1, 0]], base[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(view[key][:, :, [1, 0]], base[key], atol=1e-6, rtol=1e-6)


def test_raw_layout_v5_is_candidate_and_view_permutation_equivariant_for_both_heads() -> None:
    baseline = _v2_model(family="bidirectional_absolute_dual_head_raw_layout_visual_v5")
    candidate_permuted = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_visual_v5",
        candidate_permutation=(1, 0),
    )
    view_permuted = _v2_model(
        family="bidirectional_absolute_dual_head_raw_layout_visual_v5",
        view_permutation=(1, 0),
    )
    candidate_permuted.load_state_dict(baseline.state_dict())
    view_permuted.load_state_dict(baseline.state_dict())
    rows = torch.as_tensor([0], dtype=torch.long)

    base = baseline.forward_with_details(rows)
    candidate = candidate_permuted.forward_with_details(rows)
    view = view_permuted.forward_with_details(rows)
    for key in (
        "candidate_probabilities",
        "candidate_log_likelihood_ratios",
        "identity_candidate_probabilities",
        "identity_candidate_log_likelihood_ratios",
    ):
        torch.testing.assert_close(candidate[key][:, [1, 0]], base[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(view[key], base[key], atol=1e-6, rtol=1e-6)
    for key in (
        "view_logits",
        "identity_view_logits",
        "per_scale_view_logits",
        "identity_per_scale_view_logits",
    ):
        torch.testing.assert_close(
            candidate[key][:, [1, 0]], base[key], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(view[key][:, :, [1, 0]], base[key], atol=1e-6, rtol=1e-6)
    for key in ("null_probabilities", "identity_null_probabilities"):
        torch.testing.assert_close(candidate[key], base[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(view[key], base[key], atol=1e-6, rtol=1e-6)
