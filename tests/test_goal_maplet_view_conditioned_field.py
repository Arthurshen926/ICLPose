from pathlib import Path
from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    SCHEMA,
    ViewConditionedPrimitiveField,
    condition_canonical_codes_for_pose,
)


def _field() -> ViewConditionedPrimitiveField:
    return ViewConditionedPrimitiveField(
        primitive_rows=np.asarray([2, 5], dtype=np.int64),
        residual_basis=np.asarray([[0.0, 1.0]], dtype=np.float32),
        coefficients=np.asarray([
            [[0.5], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0]],
        ], dtype=np.float16),
        observation_count=np.asarray([4, 2], dtype=np.int32),
        mean_local_direction=np.asarray([[0.0, 0.0, 1.0]] * 2, dtype=np.float32),
        direction_concentration=np.asarray([0.9, 0.9], dtype=np.float32),
        minimum_direction_cosine=np.asarray([0.8, 0.8], dtype=np.float32),
        mean_log_projected_scale=np.zeros((2,), dtype=np.float32),
        minimum_log_projected_scale=np.full((2,), -0.5, dtype=np.float32),
        maximum_log_projected_scale=np.full((2,), 0.5, dtype=np.float32),
        physical_map_sha256="physical",
        canonical_field_sha256="canonical",
        metadata={
            "artifact_type": SCHEMA,
            "minimum_views": 3,
            "direction_cosine_margin": 0.0,
            "log_scale_margin": 0.0,
        },
    )


def test_view_conditioned_field_applies_only_inside_observed_chart():
    field = _field()
    canonical = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    code, active = field.condition_codes_numpy(
        canonical,
        np.asarray([0, 0, 1]),
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(active, np.asarray([True, False, False]))
    assert code[0, 1] > 0.0
    np.testing.assert_allclose(code[1], canonical[0], atol=1e-7)
    np.testing.assert_allclose(code[2], canonical[1], atol=1e-7)


def test_view_conditioned_field_roundtrip_preserves_content_hash(tmp_path: Path):
    field = _field()
    path = tmp_path / "field.npz"
    field.save_npz(path)
    loaded = ViewConditionedPrimitiveField.load_npz(path)
    assert loaded.content_sha256 == field.content_sha256
    loaded.validate_alignment(
        physical_map_sha256="physical",
        canonical_field_sha256="canonical",
        canonical_primitive_rows=np.asarray([2, 5]),
        canonical_feature_dim=2,
    )


def test_pose_conditioning_uses_candidate_geometry_and_only_requested_rows():
    physical = SimpleNamespace(
        content_sha256="physical",
        primitive_centers=np.asarray([
            [0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [1.0, 0.0, 4.0],
        ], dtype=np.float64),
        primitive_tangent1=np.tile([[1.0, 0.0, 0.0]], (6, 1)),
        primitive_tangent2=np.tile([[0.0, 1.0, 0.0]], (6, 1)),
        primitive_normals=np.tile([[0.0, 0.0, 1.0]], (6, 1)),
        primitive_scale1=np.full((6,), 0.4, dtype=np.float64),
        primitive_scale2=np.full((6,), 0.4, dtype=np.float64),
    )
    canonical = CanonicalSurfaceField(
        primitive_rows=np.asarray([2, 5], dtype=np.int64),
        codes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        confidence=np.ones((2,), dtype=np.float32),
        uncertainty=np.zeros((2,), dtype=np.float32),
        physical_map_sha256="physical",
    )
    log_scale = float(np.log(100.0))
    field = ViewConditionedPrimitiveField(
        primitive_rows=np.asarray([2, 5], dtype=np.int64),
        residual_basis=np.asarray([[0.0, 1.0]], dtype=np.float32),
        coefficients=np.asarray([
            [[0.5], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0]],
        ], dtype=np.float16),
        observation_count=np.asarray([4, 4], dtype=np.int32),
        mean_local_direction=np.asarray([[0.0, 0.0, -1.0]] * 2, dtype=np.float32),
        direction_concentration=np.ones((2,), dtype=np.float32),
        minimum_direction_cosine=np.full((2,), 0.9, dtype=np.float32),
        mean_log_projected_scale=np.full((2,), log_scale, dtype=np.float32),
        minimum_log_projected_scale=np.full((2,), log_scale - 0.1, dtype=np.float32),
        maximum_log_projected_scale=np.full((2,), log_scale + 0.1, dtype=np.float32),
        physical_map_sha256="physical",
        canonical_field_sha256=canonical.content_sha256,
        metadata={"artifact_type": SCHEMA, "minimum_views": 3},
    )
    camera = ColmapCamera(0, 0, 1024, 576, (1000.0, 512.0, 288.0))
    index, code, active = condition_canonical_codes_for_pose(
        field, canonical, physical, np.eye(4), camera,
        field_indices=np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(index, np.asarray([0]))
    np.testing.assert_array_equal(active, np.asarray([True]))
    assert code.shape == (1, 2)
    assert code[0, 1] > 0.0

    # Moving the camera behind the surface leaves the frozen observed chart;
    # the canonical code is used without unconstrained extrapolation.
    behind = np.eye(4)
    behind[2, 3] = -8.0
    _, fallback, active = condition_canonical_codes_for_pose(
        field, canonical, physical, behind, camera,
        field_indices=np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(active, np.asarray([False]))
    np.testing.assert_allclose(fallback, canonical.codes[[0]], atol=1e-7)


def test_pose_conditioning_rejects_duplicate_field_rows():
    field = _field()
    canonical = SimpleNamespace(
        content_sha256="canonical", primitive_rows=np.asarray([2, 5]),
        feature_dim=2, codes=np.eye(2, dtype=np.float32),
    )
    physical = SimpleNamespace(
        content_sha256="physical",
        primitive_centers=np.zeros((6, 3)),
        primitive_tangent1=np.tile([[1.0, 0.0, 0.0]], (6, 1)),
        primitive_tangent2=np.tile([[0.0, 1.0, 0.0]], (6, 1)),
        primitive_normals=np.tile([[0.0, 0.0, 1.0]], (6, 1)),
        primitive_scale1=np.ones((6,)), primitive_scale2=np.ones((6,)),
    )
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    try:
        condition_canonical_codes_for_pose(
            field, canonical, physical, np.eye(4), camera,
            field_indices=np.asarray([0, 0]),
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate field rows were accepted")
