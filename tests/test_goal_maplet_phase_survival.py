import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_phase_survival_samples import (
    _g18_probe_descriptors,
    _joint_diagnostic_field,
    _map_probe_descriptor,
    _normalize_map,
    _phase_negative_index,
)
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    FEATURE_NAMES,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField


def test_phase_negative_selection_uses_pose_error_not_appearance() -> None:
    details = [
        {"translation_m": 0.2, "rotation_deg": 1.0},
        {"translation_m": 1.8, "rotation_deg": 1.0},
        {"translation_m": 0.8, "rotation_deg": 2.0},
        {"translation_m": 0.7, "rotation_deg": 20.0},
    ]
    assert _phase_negative_index(details) == 2


def test_phase_probe_descriptor_preserves_channel_and_spatial_evidence() -> None:
    query = np.zeros((4, 6, 8), dtype=np.float32)
    query[0] = 1.0
    rendered = query.copy()
    descriptor, score, cosine = _map_probe_descriptor(
        query, rendered, np.ones((6, 8), dtype=bool),
    )
    assert descriptor.shape == (4 * 2 + 6 * 8 * 3,)
    assert np.all(np.isfinite(descriptor))
    assert score == 1.0
    assert np.allclose(cosine, 1.0)


def test_g18_coordinate_audit_has_distinct_fixed_dimensions() -> None:
    token = np.zeros((48, len(FEATURE_NAMES)), dtype=np.float32)
    token[:, FEATURE_NAMES.index("grid_x")] = np.linspace(-1.0, 1.0, 48)
    token[:, FEATURE_NAMES.index("grid_y")] = np.linspace(1.0, -1.0, 48)
    rays = np.ones((48, 3), dtype=np.float32)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    values = _g18_probe_descriptors(token, rays)
    assert values["g18_summary_no_absolute_xy"].shape == (63,)
    assert values["g18_summary_grid_xy"].shape == (105,)
    assert values["g18_summary_camera_ray"].shape == (126,)
    assert all(np.all(np.isfinite(value)) for value in values.values())


def test_joint_diagnostic_render_preserves_each_normalized_feature_space() -> None:
    rng = np.random.default_rng(19)
    fields = []
    for dimensions in (7, 5, 3):
        fields.append(CanonicalSurfaceField(
            primitive_rows=np.arange(4),
            codes=rng.normal(size=(4, dimensions)).astype(np.float32),
            confidence=np.ones(4, dtype=np.float32),
            uncertainty=np.zeros(4, dtype=np.float32),
            physical_map_sha256="map",
        ))
    joint, slices = _joint_diagnostic_field(*fields)
    for field, section in zip(fields, slices):
        recovered = _normalize_map(joint.codes[:, section].T[:, :, None])[:, :, 0].T
        assert np.allclose(recovered, field.codes, atol=1.0e-6)
