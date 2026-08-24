from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.resident_exact_reducer import (
    reduce_soft_child_token_hits_torch,
)
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    _primitive_parent_memberships,
    _reduce_soft_child_token_hits,
    dominant_child_owner,
)
from test_goal_maplet_physical_map import _inputs


def _fixture():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses,
        minimum_child_count=2, maximum_child_count=4,
    )
    rng = np.random.default_rng(20260823)
    codes = rng.normal(size=(physical.primitive_ids.size, 7)).astype(np.float32)
    codes /= np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1.0e-8)
    # Leave one primitive outside the canonical field to exercise missing mass.
    field_rows = np.arange(physical.primitive_ids.size - 1, dtype=np.int64)
    field = CanonicalSurfaceField(
        primitive_rows=field_rows,
        codes=codes[field_rows],
        confidence=np.ones(field_rows.size, dtype=np.float32),
        uncertainty=np.zeros(field_rows.size, dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
    )
    return physical, field, codes[field_rows]


def _hits(physical, *, token_count: int = 12, seed: int = 17):
    rng = np.random.default_rng(seed)
    count = 800
    token = rng.integers(0, token_count, size=count, dtype=np.int64)
    primitive = rng.integers(
        0, physical.primitive_ids.size, size=count, dtype=np.int64,
    )
    weight = rng.uniform(1.0e-6, 0.006, size=count).astype(np.float32)
    for row in range(token_count):
        selected = token == row
        total = float(np.sum(weight[selected], dtype=np.float64))
        if total > 0.75:
            weight[selected] *= np.float32(0.75 / total)
    return token, primitive, weight


def _device_reduce(
    physical, field, codes, token, primitive, weight, *, token_count: int,
    selected_child_rows=None,
):
    field_row = np.full(physical.primitive_ids.size, -1, dtype=np.int64)
    field_row[field.primitive_rows] = np.arange(field.primitive_rows.size)
    parent_offsets, parent_rows, parent_weights = _primitive_parent_memberships(physical)
    selected = (
        None if selected_child_rows is None
        else torch.as_tensor(selected_child_rows, dtype=torch.int64)
    )
    return reduce_soft_child_token_hits_torch(
        token_ids=torch.as_tensor(token, dtype=torch.int64).contiguous(),
        primitive_rows=torch.as_tensor(primitive, dtype=torch.int64).contiguous(),
        contribution=torch.as_tensor(weight, dtype=torch.float32).contiguous(),
        stable_primitive_ids=torch.as_tensor(
            np.array(physical.primitive_ids, copy=True), dtype=torch.int64,
        ).contiguous(),
        child_owner_by_primitive=torch.as_tensor(
            np.array(dominant_child_owner(physical), copy=True), dtype=torch.int64,
        ).contiguous(),
        field_row_by_primitive=torch.as_tensor(field_row, dtype=torch.int64).contiguous(),
        canonical_codes=torch.as_tensor(codes, dtype=torch.float32).contiguous(),
        parent_membership_offsets=torch.as_tensor(
            np.array(parent_offsets, copy=True), dtype=torch.int64,
        ).contiguous(),
        parent_membership_rows=torch.as_tensor(
            np.array(parent_rows, copy=True), dtype=torch.int64,
        ).contiguous(),
        parent_membership_weights=torch.as_tensor(
            np.array(parent_weights, copy=True), dtype=torch.float32,
        ).contiguous(),
        stable_parent_ids=torch.as_tensor(
            np.array(physical.maplet_ids, copy=True), dtype=torch.int64,
        ).contiguous(),
        token_count=token_count,
        child_count=physical.child_parent_rows.size,
        top_l=3,
        minimum_feature_alpha=1.0e-4,
        alpha_conservation_tolerance=2.0e-5,
        selected_child_rows=selected,
        feature_channel_block=3,
    )


def test_device_exact_reducer_matches_numpy_authority_with_missing_field_and_payload_gate():
    physical, field, codes = _fixture()
    token, primitive, weight = _hits(physical)
    selected_children = np.arange(
        0, physical.child_parent_rows.size, 2, dtype=np.int64,
    )
    authority = _reduce_soft_child_token_hits(
        physical, field,
        token_pixel_ids=token, primitive_rows=primitive, contribution=weight,
        width=4, height=3, normalized_codes=codes,
        selected_child_rows=selected_children, top_l=3,
        minimum_feature_alpha=1.0e-4,
        alpha_conservation_tolerance=2.0e-5,
    )
    actual = _device_reduce(
        physical, field, codes, token, primitive, weight, token_count=12,
        selected_child_rows=selected_children,
    )
    for name in ("child_rows", "child_feature_valid", "parent_rows"):
        expected = np.asarray(getattr(authority, name)).reshape(getattr(actual, name).shape)
        np.testing.assert_array_equal(getattr(actual, name).numpy(), expected)
    for name in (
        "child_weights", "child_features", "parent_weights", "parent_tail_weight",
        "child_tail_weight", "unassigned_geometry_weight", "background_weight",
        "canonical_field_missing_weight", "payload_excluded_weight", "null_weight",
        "total_alpha",
    ):
        expected = np.asarray(getattr(authority, name)).reshape(getattr(actual, name).shape)
        np.testing.assert_allclose(
            getattr(actual, name).numpy(), expected, atol=4.0e-7, rtol=4.0e-7,
        )


def test_device_exact_reducer_is_stable_under_pose_batch_permutation():
    physical, field, codes = _fixture()
    first = _hits(physical, token_count=9, seed=3)
    second = _hits(physical, token_count=9, seed=91)

    def combined(left, right):
        return (
            np.concatenate((left[0], right[0] + 9)),
            np.concatenate((left[1], right[1])),
            np.concatenate((left[2], right[2])),
        )

    ab = _device_reduce(
        physical, field, codes, *combined(first, second), token_count=18,
    )
    ba = _device_reduce(
        physical, field, codes, *combined(second, first), token_count=18,
    )
    exact_names = ("child_rows", "child_feature_valid", "parent_rows")
    numeric_names = (
        "child_weights", "child_features", "parent_weights", "parent_tail_weight",
        "child_tail_weight", "unassigned_geometry_weight", "background_weight",
        "canonical_field_missing_weight", "payload_excluded_weight", "null_weight",
        "total_alpha", "alpha_overflow",
    )
    for name in exact_names:
        forward = getattr(ab, name).numpy()
        reverse = getattr(ba, name).numpy()
        np.testing.assert_array_equal(forward[:9], reverse[9:])
        np.testing.assert_array_equal(forward[9:], reverse[:9])
    for name in numeric_names:
        forward = getattr(ab, name).numpy()
        reverse = getattr(ba, name).numpy()
        np.testing.assert_allclose(forward[:9], reverse[9:], atol=2.0e-7, rtol=2.0e-7)
        np.testing.assert_allclose(forward[9:], reverse[:9], atol=2.0e-7, rtol=2.0e-7)


def test_view_conditioned_exact_render_keeps_the_legacy_reducer_fallback():
    physical, field, codes = _fixture()
    scene = object.__new__(FrozenSoftSurfaceSceneGPU)
    scene.physical = physical
    scene.field = field
    scene.normalized_codes = codes
    scene.field_row_by_primitive_numpy = np.full(
        physical.primitive_ids.size, -1, dtype=np.int64,
    )
    scene.field_row_by_primitive_numpy[field.primitive_rows] = np.arange(
        field.primitive_rows.size,
    )
    scene._resident_geometry_bytes = 0
    scene._batch_ideal_hits = lambda *args, **kwargs: (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
        0,
        {
            "projection_tile_raster_seconds": 0.0,
            "device_to_host_seconds": 0.0,
            "depth_sort_composite_seconds": 0.0,
        },
    )
    scene._batch_token_remap = lambda *args, **kwargs: (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
    )
    scene._render_exact_batch_device_reduced = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("view-conditioned rendering entered the view-independent fast path")
    )
    result = scene.render_exact_batch(
        np.eye(4, dtype=np.float64)[None], object(), width=2, height=2, top_l=2,
        coordinate_supersample_factor=1,
        # An empty hit stream never evaluates the object, but its presence must
        # select the conservative legacy path.
        view_conditioned_field=object(),
    )
    assert len(result.rendered) == 1
    assert result.audit.gpu_child_reducer_implemented is False
