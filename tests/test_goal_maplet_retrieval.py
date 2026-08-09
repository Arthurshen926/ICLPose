import numpy as np

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalFieldFusionAccumulator
from feature_extract.vfm.localization_goal_maplet.retrieval import fit_validity_calibration
from test_goal_maplet_physical_map import _inputs


def test_validity_calibration_is_monotonic_and_roundtrips(tmp_path):
    score = np.linspace(0.0, 1.0, 200)
    target = 1.0 / (1.0 + np.exp(-(score - 0.65) / 0.07))
    calibration = fit_validity_calibration(score, target, metadata={"fold": "validation"})
    assert abs(calibration.center - 0.65) < 1e-3
    assert abs(calibration.scale - 0.07) < 1e-3
    assert np.all(np.diff(calibration.predict_valid(score)) > 0.0)
    path = tmp_path / "calibration.json"
    calibration.save_json(path)
    loaded = type(calibration).load_json(path)
    assert loaded.content_sha256 == calibration.content_sha256


def test_exact_contributor_canonical_fusion_is_view_balanced():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    accumulator = CanonicalFieldFusionAccumulator(physical.primitive_ids.size, 2)
    accumulator.add_view(np.asarray([0, 1]), np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([100.0, 1.0]))
    accumulator.add_view(np.asarray([0]), np.asarray([[0.0, 1.0]]), np.asarray([1.0]))
    field = accumulator.finalize(physical, metadata={"fusion": "test"})
    assert field.primitive_rows.tolist() == [0, 1]
    assert field.metadata["stored_downstream_embedding_count"] == 0
    # The footprint cap prevents the first close-up view from receiving 100x
    # the influence of the second view.
    assert field.codes[0, 1] > 0.2


def test_offline_teacher_quality_downweights_but_does_not_remove_view():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4,
    )
    accumulator = CanonicalFieldFusionAccumulator(physical.primitive_ids.size, 2)
    accumulator.add_view(
        np.asarray([0]), np.asarray([[1.0, 0.0]]), np.asarray([1.0]),
        observation_quality=np.asarray([1.0]),
    )
    accumulator.add_view(
        np.asarray([0]), np.asarray([[0.0, 1.0]]), np.asarray([1.0]),
        observation_quality=np.asarray([0.25]),
    )
    field = accumulator.finalize(physical, metadata={"fusion": "teacher_quality_test"})
    assert field.primitive_rows.tolist() == [0]
    assert field.codes[0, 0] > field.codes[0, 1] > 0.0
    assert accumulator.view_count[0] == 2
