from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    SCHEMA,
    ViewConditionedPrimitiveField,
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

