from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
    _condition_canonical_codes_for_pose_torch,
    _composite_packed_hits_torch,
    _reduce_direct_canonical_token_hits_torch,
    _reduce_direct_typed_geometry_token_hits_torch,
)
from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    _RAW_TO_IDEAL_TOKEN_WARP_CACHE,
    _raw_to_ideal_token_warp,
    _remap_ideal_hits_to_raw_tokens,
    _reduce_soft_child_token_hits,
)
from feature_extract.vfm.vfm_2dgs_mapping import _composite_sorted_packed_hits
from test_goal_maplet_physical_map import _inputs


def _reference_composite(pixel, depth, row, alpha):
    order = np.lexsort((row, depth, pixel))
    packed = np.stack([
        pixel[order].astype(np.float64), depth[order].astype(np.float64),
        row[order].astype(np.float64), alpha[order].astype(np.float64),
    ], axis=1)
    return _composite_sorted_packed_hits(packed)


def test_torch_compositor_matches_numpy_authority_with_ties_and_early_stop():
    import torch

    rng = np.random.default_rng(20260817)
    pixel = rng.integers(0, 11, size=500, dtype=np.int64)
    depth = rng.choice(np.asarray([1.0, 1.5, 2.0, 3.0], dtype=np.float32), size=500)
    row = rng.integers(0, 37, size=500, dtype=np.int64)
    alpha = rng.uniform(0.01, 0.999, size=500).astype(np.float32)
    expected = _reference_composite(pixel, depth, row, alpha)
    actual = _composite_packed_hits_torch(
        torch.from_numpy(pixel.copy()), torch.from_numpy(depth.copy()),
        torch.from_numpy(row.copy()), torch.from_numpy(alpha.copy()),
    )
    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])
    np.testing.assert_allclose(actual[2].numpy(), expected[2], atol=2e-7, rtol=2e-7)


def test_torch_compositor_rejects_fractional_identity_and_nonfinite_depth():
    import torch

    with np.testing.assert_raises_regex(ValueError, "identity arrays"):
        _composite_packed_hits_torch(
            torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.int64), torch.ones(2)
        )


def test_direct_canonical_token_reducer_preserves_all_payload_mass_and_feature_mean():
    import torch

    feature, mass, valid = _reduce_direct_canonical_token_hits_torch(
        torch.tensor([0, 0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 1, 1, 2], dtype=torch.int64),
        torch.tensor([0.5, 0.25, 0.4, 0.8], dtype=torch.float32),
        torch.tensor([0, 1, -1], dtype=torch.int64),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        token_count=4,
        accumulation_chunk_rows=1,
    )
    expected = np.asarray([2.0, 1.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(feature[0].numpy(), expected, atol=1.0e-7)
    np.testing.assert_allclose(feature[1].numpy(), [0.0, 1.0], atol=1.0e-7)
    np.testing.assert_allclose(mass.numpy(), [0.75, 0.4, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_array_equal(valid.numpy(), [True, True, False, False])


def test_direct_canonical_sparse_override_changes_only_declared_field_row():
    import torch

    feature, mass, valid = _reduce_direct_canonical_token_hits_torch(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        token_count=2,
        override_field_rows=torch.tensor([1], dtype=torch.int64),
        override_codes=torch.tensor([[-1.0, 0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(feature, torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    torch.testing.assert_close(mass, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(valid, torch.tensor([True, True]))


def test_torch_view_conditioning_applies_active_low_rank_residual():
    import torch

    canonical = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    coefficients = torch.zeros((2, 5, 1), dtype=torch.float16)
    coefficients[0, 0, 0] = 0.5
    code, active = _condition_canonical_codes_for_pose_torch(
        canonical,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]] * 2, dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=torch.float32),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        coefficients,
        torch.tensor([4, 4], dtype=torch.int32),
        torch.tensor([[0.0, 0.0, -1.0]] * 2, dtype=torch.float32),
        torch.ones(2, dtype=torch.float32),
        -torch.ones(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        -torch.ones(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.float32),
        torch.eye(4, dtype=torch.float32),
        focal_pixels=1.0,
        minimum_views=3,
        direction_cosine_margin=0.05,
        log_scale_margin=0.25,
    )
    torch.testing.assert_close(
        code[0], torch.tensor([1.0, 0.5]) / np.sqrt(1.25),
    )
    torch.testing.assert_close(code[1], canonical[1])
    torch.testing.assert_close(active, torch.tensor([True, True]))


def test_direct_canonical_token_reducer_rejects_fractional_rows_and_nonfinite_codes():
    import torch

    common = dict(
        token_ids=torch.tensor([0], dtype=torch.int64),
        contribution=torch.tensor([1.0], dtype=torch.float32),
        field_row_by_primitive=torch.tensor([0], dtype=torch.int64),
        canonical_codes=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        token_count=1,
    )
    with np.testing.assert_raises_regex(ValueError, "identity arrays"):
        _reduce_direct_canonical_token_hits_torch(
            primitive_rows=torch.tensor([0.0]), **common
        )
    common["canonical_codes"] = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
    with np.testing.assert_raises_regex(ValueError, "finite"):
        _reduce_direct_canonical_token_hits_torch(
            primitive_rows=torch.tensor([0], dtype=torch.int64), **common
        )
    with np.testing.assert_raises_regex(ValueError, "finite"):
        _composite_packed_hits_torch(
            torch.ones(2, dtype=torch.int64), torch.tensor([1.0, float("nan")]),
            torch.ones(2, dtype=torch.int64), torch.ones(2),
        )


def test_direct_typed_geometry_reducer_is_normal_sign_invariant_and_depth_centered():
    import torch

    common = dict(
        token_ids=torch.tensor([0, 0, 1], dtype=torch.int64),
        primitive_rows=torch.tensor([0, 1, 2], dtype=torch.int64),
        contribution=torch.tensor([0.4, 0.2, 0.5], dtype=torch.float32),
        field_row_by_primitive=torch.tensor([0, 1, 2], dtype=torch.int64),
        primitive_centers_world=torch.tensor(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0]],
            dtype=torch.float32,
        ),
        poses_w2c=torch.eye(4, dtype=torch.float32)[None].contiguous(),
        token_count_per_pose=2,
    )
    positive = _reduce_direct_typed_geometry_token_hits_torch(
        primitive_normals_world=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        **common,
    )
    negative = _reduce_direct_typed_geometry_token_hits_torch(
        primitive_normals_world=-torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        **common,
    )
    torch.testing.assert_close(positive[0], negative[0])
    torch.testing.assert_close(positive[4], torch.tensor([0.6, 0.5]))
    # The mass-weighted mean relative log depth over the view is zero.
    torch.testing.assert_close(torch.sum(positive[1] * positive[4]), torch.tensor(0.0))
    assert torch.all(positive[2] >= 0.0)
    assert torch.all((positive[3] >= 0.0) & (positive[3] <= 1.0))


def _camera() -> ColmapCamera:
    return ColmapCamera(0, 2, 8, 8, (6.0, 4.0, 4.0, 0.04))


def test_raw_to_ideal_warp_is_camera_bound_and_cached():
    _RAW_TO_IDEAL_TOKEN_WARP_CACHE.clear()
    camera = _camera()
    first = _raw_to_ideal_token_warp(
        camera, token_width=2, token_height=2, supersample_factor=2
    )
    second = _raw_to_ideal_token_warp(
        camera, token_width=2, token_height=2, supersample_factor=2
    )
    assert first is second
    assert first.source_ideal_pixel_ids.flags.writeable is False
    assert first.destination_token_pixel_ids.flags.writeable is False
    changed = ColmapCamera(0, 2, 8, 8, (6.0, 4.0, 4.0, 0.05))
    third = _raw_to_ideal_token_warp(
        changed, token_width=2, token_height=2, supersample_factor=2
    )
    assert third is not first
    assert len(_RAW_TO_IDEAL_TOKEN_WARP_CACHE) == 2


def test_raw_warp_accepts_all_declared_pinhole_camera_models():
    simple = ColmapCamera(0, 0, 8, 8, (6.0, 4.0, 4.0))
    pinhole = ColmapCamera(0, 1, 8, 8, (6.0, 5.5, 4.0, 4.0))
    for camera in (simple, pinhole, _camera()):
        warp = _raw_to_ideal_token_warp(
            camera, token_width=2, token_height=2, supersample_factor=2
        )
        assert warp.source_ideal_pixel_ids.size > 0
        assert warp.source_ideal_pixel_ids.shape == warp.destination_token_pixel_ids.shape


def test_batched_raw_token_remap_matches_independent_scalar_remap():
    camera = _camera()
    # Two primitive hits at every ideal pixel.  The arrays are already in the
    # global-pixel order emitted by the resident compositor.
    local_pixels = np.repeat(np.arange(16, dtype=np.int64), 2)
    local_rows = np.tile(np.asarray([3, 7], dtype=np.int64), 16)
    local_weights = np.tile(np.asarray([0.6, 0.2], dtype=np.float32), 16)
    global_pixels = np.concatenate([local_pixels, local_pixels + 16])
    rows = np.concatenate([local_rows, local_rows])
    weights = np.concatenate([local_weights, local_weights])
    token, batch_rows, batch_weights = FrozenSoftSurfaceSceneGPU._batch_token_remap(
        global_pixels, rows, weights, camera, batch_size=2,
        token_width=2, token_height=2, supersample_factor=2,
    )
    for batch in range(2):
        scalar_token, scalar_rows, scalar_weights = _remap_ideal_hits_to_raw_tokens(
            local_pixels, local_rows, local_weights, camera,
            token_width=2, token_height=2, supersample_factor=2,
        )
        mask = (token >= batch * 4) & (token < (batch + 1) * 4)
        np.testing.assert_array_equal(token[mask] - batch * 4, scalar_token)
        np.testing.assert_array_equal(batch_rows[mask], scalar_rows)
        np.testing.assert_array_equal(batch_weights[mask], scalar_weights)


def test_device_batched_raw_token_remap_is_exactly_equivalent_on_cpu_tensors():
    import torch

    camera = _camera()
    local_pixels = np.repeat(np.arange(16, dtype=np.int64), 2)
    local_rows = np.tile(np.asarray([3, 7], dtype=np.int64), 16)
    local_weights = np.tile(np.asarray([0.6, 0.2], dtype=np.float32), 16)
    global_pixels = np.concatenate([local_pixels, local_pixels + 16])
    rows = np.concatenate([local_rows, local_rows])
    weights = np.concatenate([local_weights, local_weights])
    expected = FrozenSoftSurfaceSceneGPU._batch_token_remap(
        global_pixels, rows, weights, camera, batch_size=2,
        token_width=2, token_height=2, supersample_factor=2,
    )
    actual = FrozenSoftSurfaceSceneGPU._batch_token_remap_torch(
        torch.from_numpy(global_pixels.copy()), torch.from_numpy(rows.copy()),
        torch.from_numpy(weights.copy()), camera, batch_size=2,
        token_width=2, token_height=2, supersample_factor=2,
    )
    np.testing.assert_array_equal(actual[0].numpy(), expected[0])
    np.testing.assert_array_equal(actual[1].numpy(), expected[1])
    np.testing.assert_array_equal(actual[2].numpy(), expected[2])


def test_batched_raw_token_remap_rejects_unsorted_hits():
    with np.testing.assert_raises_regex(ValueError, "globally sorted"):
        FrozenSoftSurfaceSceneGPU._batch_token_remap(
            np.asarray([2, 1], dtype=np.int64),
            np.asarray([3, 4], dtype=np.int64),
            np.asarray([0.2, 0.1], dtype=np.float32),
            _camera(), batch_size=1, token_width=2, token_height=2,
            supersample_factor=2,
        )


def test_sparse_pose_conditioned_code_override_changes_only_requested_field_row():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses,
        minimum_child_count=2, maximum_child_count=4,
    )
    primitive_count = int(physical.primitive_ids.size)
    canonical = CanonicalSurfaceField(
        primitive_rows=np.arange(primitive_count, dtype=np.int64),
        codes=np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32), (primitive_count, 1)
        ),
        confidence=np.ones((primitive_count,), dtype=np.float32),
        uncertainty=np.zeros((primitive_count,), dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
    )
    rendered = _reduce_soft_child_token_hits(
        physical, canonical,
        token_pixel_ids=np.asarray([0], dtype=np.int64),
        primitive_rows=np.asarray([0], dtype=np.int64),
        contribution=np.asarray([1.0], dtype=np.float32),
        width=1, height=1, normalized_codes=canonical.codes,
        normalized_code_override_rows=np.asarray([0], dtype=np.int64),
        normalized_code_override_values=np.asarray([[0.0, 1.0]], dtype=np.float32),
        selected_child_rows=None, top_l=1, minimum_feature_alpha=1.0e-4,
        alpha_conservation_tolerance=2.0e-5,
    )
    np.testing.assert_allclose(
        rendered.child_features.reshape(-1, 2)[0], [0.0, 1.0], atol=1.0e-7
    )
