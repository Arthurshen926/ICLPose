from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PARENT_SCENE_RANK_RAW,
    PARENT_SCENE_RANK_SURFACE_DENSITY,
    SCENE_AGGREGATION,
    PureRadioPhysicalRetrieval,
    aggregate_sparse_token_evidence,
    all_radio_token_coordinates,
    rank_children_with_physical_iou_nms,
    rank_parent_regions,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
    LEGACY_COORDINATE_CONTRACT,
    _candidate_recall,
    evaluate_pure_retrieval_query,
    inverse_simple_radial,
    load_contributors_in_radio_coordinates,
    remap_pinhole_contributors_to_raw_grid,
)
from feature_extract.tools.vfm.retrieve_goal_maplet_pure_radio import (
    _validate_field_coordinate_contract,
)
from test_goal_maplet_physical_map import _inputs


def _physical():
    geometry, maplets, region, poses = _inputs()
    return build_goal_maplet_physical_map(
        maplets,
        region,
        geometry,
        poses,
        minimum_child_count=2,
        maximum_child_count=4,
    )


def _metadata(height=2, width=2):
    return {
        "artifact_type": "goal_maplet_pure_radio_physical_retrieval_v1",
        "token_height": height,
        "token_width": width,
        "scene_aggregation": SCENE_AGGREGATION,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
    }


def _result(physical):
    xy = all_radio_token_coordinates(2, 2)
    parent_id = int(physical.maplet_ids[0])
    child_count = int(physical.child_parent_rows.size)
    child_rows = (np.arange(4, dtype=np.int64) % child_count).reshape(4, 1)
    scene_child_rows = np.arange(child_count, dtype=np.int64)
    return PureRadioPhysicalRetrieval(
        image_id="seq/test.png",
        token_xy=xy,
        token_parent_ids=np.full((4, 1), parent_id, dtype=np.int64),
        token_parent_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        token_out_of_map_probabilities=np.full(4, 0.1, dtype=np.float32),
        token_in_map_tail_probabilities=np.full(4, 0.1, dtype=np.float32),
        token_child_rows=child_rows,
        token_child_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        scene_parent_ids=np.asarray([parent_id], dtype=np.int64),
        scene_parent_scores=np.asarray([3.2], dtype=np.float32),
        scene_child_rows=scene_child_rows,
        scene_child_scores=np.linspace(
            0.9, 0.6, child_count, dtype=np.float32
        ),
        physical_map_sha256=physical.content_sha256,
        metadata=_metadata(),
    )


def test_pure_retrieval_requires_full_grid_and_forbids_pose_methods(tmp_path):
    physical = _physical()
    result = _result(physical)
    path = tmp_path / "retrieval.npz"
    result.save_npz(path)
    loaded = PureRadioPhysicalRetrieval.load_npz(path)
    assert loaded.content_sha256 == result.content_sha256
    bad = _metadata()
    bad["uses_pnp"] = True
    with pytest.raises(ValueError, match="uses_pnp=false"):
        PureRadioPhysicalRetrieval(
            **{**result.__dict__, "metadata": bad}
        )


def test_production_retrieval_rejects_legacy_coordinate_field():
    class Field:
        metadata = {}

    with pytest.raises(ValueError, match="coordinate-correct"):
        _validate_field_coordinate_contract(
            Field(), allow_legacy_coordinate_misaligned_control=False,
        )
    correct, contract = _validate_field_coordinate_contract(
        Field(), allow_legacy_coordinate_misaligned_control=True,
    )
    assert correct is False
    assert contract == ""


def test_block_aggregation_does_not_reward_duplicate_tokens_inside_one_block():
    xy = all_radio_token_coordinates(4, 4)
    rows = np.full((16, 1), -1, dtype=np.int64)
    probability = np.zeros((16, 1), dtype=np.float32)
    rows[:4] = 0
    probability[:4] = np.asarray([[0.9], [0.8], [0.7], [0.6]])
    score = aggregate_sparse_token_evidence(
        xy,
        rows,
        probability,
        entity_count=1,
        token_height=4,
        token_width=4,
        block_rows=2,
        block_cols=2,
        top_blocks=4,
    )
    # The first two tokens share one block, the next two share another.
    np.testing.assert_allclose(score, [1.6])


def test_child_iou_nms_suppresses_exact_duplicate_support():
    physical = _physical()
    scores = np.zeros(physical.child_parent_rows.size, dtype=np.float64)
    scores[:3] = [1.0, 0.9, 0.8]
    rows, _, suppressed = rank_children_with_physical_iou_nms(
        scores, physical, maximum_children=3, maximum_primitive_iou=0.0
    )
    # At threshold zero, every later nonempty child conflicts by contract.
    assert rows.tolist() == [0]
    assert suppressed == 2


def test_parent_surface_density_ranking_charges_large_regions():
    physical = _physical()
    evidence = np.zeros(physical.maplet_ids.size, dtype=np.float64)
    evidence[:] = 1.0
    raw, _ = rank_parent_regions(
        evidence,
        physical,
        maximum_parents=physical.maplet_ids.size,
        semantics=PARENT_SCENE_RANK_RAW,
    )
    density, density_score = rank_parent_regions(
        evidence,
        physical,
        maximum_parents=physical.maplet_ids.size,
        semantics=PARENT_SCENE_RANK_SURFACE_DENSITY,
    )
    area = []
    for parent in range(physical.maplet_ids.size):
        start, end = physical.membership_offsets[parent : parent + 2]
        members = np.unique(physical.membership_primitive_rows[start:end])
        area.append(
            float(
                np.sum(
                    np.pi
                    * physical.primitive_scale1[members]
                    * physical.primitive_scale2[members]
                )
            )
        )
    expected = sorted(
        range(physical.maplet_ids.size),
        key=lambda row: (area[row], physical.maplet_ids[row]),
    )
    assert raw.tolist() == np.argsort(physical.maplet_ids, kind="stable").tolist()
    assert density.tolist() == expected
    np.testing.assert_allclose(density_score, [1.0 / area[row] for row in expected])


def test_simple_radial_inverse_roundtrip_and_zero_distortion_identity():
    value = np.asarray([[0.7, 0.4], [-0.5, 0.2], [0.0, 0.0]])
    undistorted = inverse_simple_radial(value, 0.04)
    radius2 = np.sum(undistorted * undistorted, axis=1)
    reconstructed = undistorted * (1.0 + 0.04 * radius2)[:, None]
    np.testing.assert_allclose(reconstructed, value, atol=1e-12)
    np.testing.assert_array_equal(inverse_simple_radial(value, 0.0), value)


def test_coordinate_remap_is_identity_for_pinhole_and_moves_radial_edges():
    ids = np.arange(8 * 8, dtype=np.int64).reshape(8, 8, 1)
    labels = ContributorLabels(ids, np.ones_like(ids, dtype=np.float32), np.eye(4))
    pinhole, audit = remap_pinhole_contributors_to_raw_grid(
        labels,
        camera_model_id=0,
        camera_width=8,
        camera_height=8,
        camera_params=np.asarray([4.0, 4.0, 4.0]),
    )
    np.testing.assert_array_equal(pinhole.topk_primitive_ids, ids)
    assert audit["coordinate_contract"] == COORDINATE_CONTRACT
    radial, radial_audit = remap_pinhole_contributors_to_raw_grid(
        labels,
        camera_model_id=2,
        camera_width=8,
        camera_height=8,
        camera_params=np.asarray([4.0, 4.0, 4.0, 0.5]),
    )
    assert np.any(radial.topk_primitive_ids != ids)
    assert radial_audit["maximum_inverse_roundtrip_residual_contributor_px"] < 1e-10


def test_canonical_builder_loads_contributors_in_radio_coordinates(tmp_path: Path):
    ids = np.arange(6 * 8, dtype=np.int32).reshape(6, 8, 1)
    weights = np.ones_like(ids, dtype=np.float32)
    path = tmp_path / "contributors.npz"
    np.savez_compressed(
        path,
        topk_ids=ids,
        topk_weights=weights,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera_model_id=np.asarray(2, dtype=np.int32),
        camera_width=np.asarray(8, dtype=np.int32),
        camera_height=np.asarray(6, dtype=np.int32),
        camera_params=np.asarray([2.0, 4.0, 3.0, 0.8], dtype=np.float64),
    )
    corrected, audit = load_contributors_in_radio_coordinates(
        path, legacy_pinhole_as_raw_diagnostic=False,
    )
    legacy, legacy_audit = load_contributors_in_radio_coordinates(
        path, legacy_pinhole_as_raw_diagnostic=True,
    )
    assert audit["coordinate_contract"] == COORDINATE_CONTRACT
    assert legacy_audit["coordinate_contract"] == LEGACY_COORDINATE_CONTRACT
    np.testing.assert_array_equal(legacy.topk_primitive_ids, ids)
    np.testing.assert_array_equal(legacy.topk_weights, weights)
    assert np.any(corrected.topk_primitive_ids != legacy.topk_primitive_ids)
    assert corrected.topk_weights.shape == weights.shape


def test_surface_metrics_credit_visible_physical_support_and_not_camera_distance():
    physical = _physical()
    result = _result(physical)
    ids = np.full((4, 4, 1), -1, dtype=np.int64)
    weights = np.zeros((4, 4, 1), dtype=np.float32)
    # Primitive zero belongs to the retrieved first parent and first child.
    ids[..., 0] = int(physical.primitive_ids[0])
    weights[..., 0] = 1.0
    labels = ContributorLabels(ids, weights, np.eye(4))
    report = evaluate_pure_retrieval_query(
        result,
        labels,
        physical,
        camera_model_id=0,
        camera_width=4,
        camera_height=4,
        camera_params=np.asarray([4.0, 2.0, 2.0]),
        ks=(1,),
        tolerances_m=(0.0, 0.5),
    )
    assert report["token_parent"]["conditional_recall_at_1"] == 1.0
    assert report["surface_sets"]["parent_top1"]["exact_visible_mass_recall"] == 1.0
    assert report["surface_sets"]["parent_top1"]["tolerant_visible_mass_recall_0m"] == 1.0
    area_control = report["surface_sets"]["parent_area20pct"]
    total_area = float(
        np.sum(np.pi * physical.primitive_scale1 * physical.primitive_scale2)
    )
    assert area_control["charged_surface_area_m2"] <= 0.20 * total_area + 1e-12
    assert area_control["selected_parent_count"] <= result.scene_parent_ids.size
    assert "area_le_20pct" in area_control["selection_semantics"]
    hierarchical = report["surface_sets"]["hierarchical_child_parent16_x4"]
    assert hierarchical["selected_child_count"] <= 64
    assert (
        hierarchical["selection_semantics"]
        == "global_top16_parents_then_top4_children_per_parent_v1"
    )
    assert report["claim_scope"]["metric_is_localization_success"] is False


def test_vectorized_candidate_recall_matches_unique_sparse_reference():
    truth = sparse.csr_matrix(
        np.asarray(
            [[0.0, 0.2, 0.3, 0.5], [0.4, 0.0, 0.6, 0.0]],
            dtype=np.float64,
        )
    )
    candidates = np.asarray([[3, 1, 3, -1], [2, 0, 1, 0]], dtype=np.int64)
    result = _candidate_recall(
        truth, candidates, (1, 2, 3, 4), absolute_denominator=np.ones(2),
    )
    for requested in (1, 2, 3, 4):
        retrieved = 0.0
        for token in range(2):
            rows = np.unique(
                candidates[token, :requested][
                    (candidates[token, :requested] >= 0)
                    & (candidates[token, :requested] < truth.shape[1])
                ]
            )
            retrieved += float(np.sum(truth[token, rows])) if rows.size else 0.0
        assert result[requested][0] == pytest.approx(retrieved / float(truth.sum()))
        assert result[requested][1] == pytest.approx(retrieved / 2.0)
