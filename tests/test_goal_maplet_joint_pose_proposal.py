from types import SimpleNamespace

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.child_retrieval import ChildTilePosterior
from feature_extract.vfm.localization_goal_maplet.joint_pose_proposal import (
    _projected_surface_assignment,
    generate_joint_configuration_pose_modes,
)
from feature_extract.vfm.localization_goal_maplet.geometry_guided_pose_proposal import (
    _greedy_pose_basin_count,
    _disconnected_query_token_masks,
    _per_anchor_pose_basin_diagnostics,
    _pose_basin_diverse_top_rows,
    _two_channel_pose_basin_union_rows,
    _structurally_stratified_top_rows,
    estimate_pose_from_scaled_query_geometry,
    generate_geometry_guided_configuration_pose_modes,
    unproject_query_depth,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import SINGLE_SIDED
from feature_extract.vfm.localization_goal_maplet.query_edge_factor import (
    build_query_edge_graph,
)
from feature_extract.vfm.localization_goal_maplet.soft_maplet_pose_likelihood import (
    pose_conditioned_soft_maplet_evidence,
)
from feature_extract.vfm.localization_goal_maplet.typed_graph import TypedParentGraph
from feature_extract.vfm.localization_goal_maplet.visibility_chart import (
    refine_pose_in_visibility_chart,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


def test_disconnected_query_mask_is_union_of_fixed_seed_regions():
    mask = _disconnected_query_token_masks(
        np.asarray([[0, 2, -1], [1, -1, -1]]),
        np.asarray([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]]),
        np.asarray([[0.13, 0.13], [0.13, 0.13], [0.13, 0.13]]),
        token_height=4,
        token_width=4,
    )
    assert mask.shape == (2, 16)
    assert int(np.sum(mask[0])) == 8
    assert int(np.sum(mask[1])) == 4
    assert not np.array_equal(mask[0], mask[1])


def test_structural_exact_pool_protects_each_anchor_then_fills_globally():
    rows, diagnostic = _structurally_stratified_top_rows(
        np.asarray([10.0, 9.0, 8.0, 7.0, 1.0, 0.0]),
        np.asarray([100, 101, 102, 103, 104, 105]),
        {100: 0, 101: 0, 102: 0, 103: 0, 104: 1, 105: 1},
        maximum_count=4,
        keep_per_anchor=1,
    )
    # Anchor 1 survives even though all of its states lie below anchor 0 in
    # the global sparse order. The rest of the budget follows global rank.
    np.testing.assert_array_equal(rows, np.asarray([0, 4, 1, 2]))
    assert diagnostic["protected_state_count"] == 2
    assert diagnostic["global_fill_count"] == 2
    assert diagnostic["protected_anchor_count"] == 2


def test_structural_exact_pool_quota_zero_is_stable_global_topk():
    rows, diagnostic = _structurally_stratified_top_rows(
        np.asarray([0.2, 0.5, 0.5, -1.0]),
        np.asarray([10, 11, 12, 13]),
        {10: 0, 13: 1},
        maximum_count=3,
        keep_per_anchor=0,
    )
    np.testing.assert_array_equal(rows, np.asarray([1, 2, 0]))
    assert diagnostic["selection_semantics"] == "global_sparse_topk"


def test_structural_exact_pool_can_protect_anchor_prefix_without_dropping_global_tail():
    scores = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    serials = np.arange(100, 106)
    all_anchors = {100: 0, 101: 0, 102: 1, 103: 1, 104: 2, 105: 2}
    protected_prefix = {
        serial: anchor for serial, anchor in all_anchors.items() if anchor < 2
    }
    rows, diagnostic = _structurally_stratified_top_rows(
        scores, serials, protected_prefix, maximum_count=5, keep_per_anchor=2,
    )
    # Anchors 0 and 1 consume four protected slots.  The best state from the
    # unprotected tail anchor remains eligible through the global fill.
    np.testing.assert_array_equal(rows, np.asarray([0, 1, 2, 3, 4]))
    assert diagnostic["protected_anchor_count"] == 2
    assert diagnostic["global_fill_count"] == 1


def test_pose_basin_exact_pool_removes_cross_anchor_duplicate_states():
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (5, 1, 1))
    poses[1, 0, 3] = 0.05  # Same camera basin as row 0.
    poses[2, 0, 3] = 2.0
    poses[3, 1, 3] = 3.0
    poses[4, 2, 3] = 4.0
    rows, diagnostic = _pose_basin_diverse_top_rows(
        np.asarray([10.0, 9.0, 8.0, 7.0, 6.0]), poses, 3,
        translation_radius_m=0.5, rotation_radius_deg=5.0,
    )
    np.testing.assert_array_equal(rows, np.asarray([0, 2, 3]))
    assert diagnostic["selected_pose_basin_count"] == 3
    assert diagnostic["near_duplicate_rejection_count"] == 1
    assert diagnostic["duplicate_fill_count"] == 0


def test_greedy_pose_basin_count_deduplicates_near_identical_states():
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (3, 1, 1))
    poses[1, 0, 3] = 0.05
    poses[2, 0, 3] = 2.0
    assert _greedy_pose_basin_count(
        poses, translation_radius_m=0.5, rotation_radius_deg=5.0,
    ) == 2


def test_per_anchor_basin_diagnostic_reports_multiplicity_and_exact_survival():
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (5, 1, 1))
    poses[1, 0, 3] = 0.05
    poses[2, 0, 3] = 2.0
    poses[3, 1, 3] = 3.0
    poses[4, 1, 3] = 3.05
    diagnostic = _per_anchor_pose_basin_diagnostics(
        poses, np.asarray([0, 0, 0, 1, 1]), np.asarray([0, 2, 3]),
    )
    assert diagnostic["0"] == {
        "raw_state_count": 3,
        "selected_exact_state_count": 2,
        "unique_pose_basin_count_0_5m_5deg": 2,
        "unique_pose_basin_count_1m_10deg": 2,
    }
    assert diagnostic["1"]["raw_state_count"] == 2
    assert diagnostic["1"]["selected_exact_state_count"] == 1
    assert diagnostic["1"]["unique_pose_basin_count_0_5m_5deg"] == 1


def test_two_channel_pose_basin_union_reserves_geometry_capacity():
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (6, 1, 1))
    for row in range(6):
        poses[row, 0, 3] = 2.0 * row
    rows, diagnostic = _two_channel_pose_basin_union_rows(
        np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0]),
        np.asarray([0.0, 1.0, 2.0, 7.0, 8.0, 9.0]),
        poses,
        4,
        translation_radius_m=0.5,
        rotation_radius_deg=5.0,
    )
    # Two VFM states and two different geometry states reach the shared pool.
    np.testing.assert_array_equal(rows, np.asarray([0, 1, 5, 4]))
    assert diagnostic["primary_channel_count"] == 2
    assert diagnostic["secondary_channel_count"] == 2


def test_visibility_chart_soft_update_improves_bounded_pose_without_gt():
    world = np.asarray([
        [-1.0, -0.8, 4.5], [1.0, -0.7, 5.0],
        [-0.8, 0.9, 5.4], [1.1, 0.8, 4.8],
        [0.1, -1.1, 5.8], [-1.2, 0.2, 5.1],
    ])
    query = world.copy()
    wrong = world + np.asarray([8.0, 0.0, 0.0])
    centers = np.concatenate((world, wrong), axis=0)
    child = np.stack((np.arange(6), np.arange(6, 12)), axis=1)
    parent = np.stack((np.zeros(6, dtype=np.int64), np.ones(6, dtype=np.int64)), axis=1)
    initial = np.eye(4)
    initial[:3, 3] = np.asarray([0.35, -0.15, 0.10])
    update = refine_pose_in_visibility_chart(
        initial, 1.0, query, child, parent,
        np.tile(np.asarray([[0.55, 0.45]]), (6, 1)),
        np.ones((6, 2), dtype=bool), centers,
        np.asarray([1.0, 0.0]), np.ones((6,)),
        iterations=3, maximum_translation_update_m=1.0,
    )
    assert update.accepted_iterations > 0
    assert update.final_objective > update.initial_objective
    assert np.linalg.norm(update.pose_w2c[:3, 3]) < np.linalg.norm(initial[:3, 3])


def _fixture():
    camera = ColmapCamera(
        camera_id=1, model_id=1, width=200, height=160,
        params=(140.0, 140.0, 100.0, 80.0),
    )
    centers = np.asarray([
        [-1.0, -0.8, 4.5], [1.0, -0.7, 5.0],
        [-0.8, 0.9, 5.4], [1.1, 0.8, 4.8],
    ])
    physical = SimpleNamespace(
        content_sha256="a" * 64,
        maplet_ids=np.arange(10, 14, dtype=np.int64),
        child_centers=centers,
        child_frames=np.repeat(np.eye(3)[None], 4, axis=0),
        child_extents=np.repeat(np.asarray([[0.25, 0.25, 0.02]]), 4, axis=0),
        child_normals=np.repeat(np.asarray([[0.0, 0.0, -1.0]]), 4, axis=0),
        child_parent_rows=np.arange(4, dtype=np.int64),
        maplet_sidedness=np.full((4,), SINGLE_SIDED, dtype=np.uint8),
    )
    graph = TypedParentGraph(
        edge_source=np.zeros((0,), dtype=np.int32),
        edge_target=np.zeros((0,), dtype=np.int32),
        edge_type=np.zeros((0,), dtype=np.uint8),
        edge_features=np.zeros((0, 6), dtype=np.float32),
        parent_view_count=np.ones((4,), dtype=np.int32),
        parent_distinctiveness=np.ones((4,), dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256="b" * 64,
        metadata={"artifact_type": "goal_maplet_typed_parent_graph_v1"},
    )
    matrix, distortion = camera_matrix_and_distortion(camera)
    projected, _ = cv2.projectPoints(
        centers, np.zeros((3, 1)), np.zeros((3, 1)), matrix, distortion,
    )
    xy = projected.reshape(-1, 2)
    extent = np.full((4, 2), 8.0)
    parent_ids = physical.maplet_ids[:, None]
    probability = np.full((4, 1), 0.9, dtype=np.float32)
    child = ChildTilePosterior(
        candidate_child_rows=np.arange(4, dtype=np.int64)[:, None],
        candidate_probabilities=probability,
        null_probabilities=np.full((4,), 0.1, dtype=np.float32),
        conditional_parent_ids=parent_ids,
        conditional_parent_log_evidence=np.zeros((4, 1), dtype=np.float32),
        best_child_rows_by_parent=np.arange(4, dtype=np.int64)[:, None],
        best_child_probabilities_by_parent=np.ones((4, 1), dtype=np.float32),
    )
    return physical, graph, camera, xy, extent, parent_ids, probability, child


def test_projected_surface_assignment_reads_identity_after_pose():
    physical, _graph, camera, xy, extent, _ids, probability, child = _fixture()
    score, parent, selected_child, support = _projected_surface_assignment(
        np.eye(4),
        xy / [camera.width, camera.height],
        extent / [camera.width, camera.height],
        probability,
        np.full((4,), 0.1),
        child.best_child_rows_by_parent,
        child.best_child_probabilities_by_parent,
        physical,
        camera,
        maximum_parent_candidates=1,
    )
    assert np.isfinite(score)
    assert support == 4
    np.testing.assert_array_equal(parent, np.arange(4))
    np.testing.assert_array_equal(selected_child, np.arange(4))


def test_joint_proposal_recovers_synthetic_pose_and_configuration():
    physical, graph, camera, xy, extent, parent_ids, probability, child = _fixture()
    descriptor = np.eye(4, dtype=np.float32)
    modes = generate_joint_configuration_pose_modes(
        xy,
        extent,
        descriptor,
        parent_ids,
        probability,
        np.zeros((4,), dtype=np.float32),
        np.full((4,), 0.1, dtype=np.float32),
        child,
        physical,
        graph,
        camera,
        maximum_modes=8,
        proposal_trials=32,
        maximum_parent_candidates=1,
        preliminary_pose_count=32,
        anchor_conditioned_fraction=0.0,
        random_seed=5,
    )
    assert modes.poses_w2c.shape[0] > 0
    centers = np.asarray([
        -pose[:3, :3].T @ pose[:3, 3] for pose in modes.poses_w2c
    ])
    assert np.min(np.linalg.norm(centers, axis=1)) < 1e-3
    assert np.any(np.all(modes.configuration_child_rows == np.arange(4), axis=1))


def test_scaled_geometry_alignment_recovers_metric_pose():
    world = np.asarray([
        [-1.0, -0.7, 4.0], [1.2, -0.4, 4.8],
        [-0.6, 1.1, 5.3], [1.0, 0.8, 4.4],
    ])
    rotation, _ = cv2.Rodrigues(np.asarray([0.04, -0.08, 0.03]))
    translation = np.asarray([0.3, -0.2, 0.7])
    nuisance_scale = 1.7
    query = nuisance_scale * (world @ rotation.T + translation)
    estimated = estimate_pose_from_scaled_query_geometry(world, query)
    assert estimated is not None
    pose, scale = estimated
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=1e-7)
    np.testing.assert_allclose(pose[:3, 3], translation, atol=1e-7)
    np.testing.assert_allclose(scale, nuisance_scale, atol=1e-7)


def test_geometry_guided_proposal_recovers_synthetic_pose_without_pnp():
    physical, graph, camera, xy, extent, parent_ids, probability, child = _fixture()
    query_xyz = unproject_query_depth(xy, physical.child_centers[:, 2], camera)
    np.testing.assert_allclose(query_xyz, physical.child_centers, atol=1e-6)
    modes = generate_geometry_guided_configuration_pose_modes(
        xy,
        extent,
        np.eye(4, dtype=np.float32),
        physical.child_centers[:, 2],
        np.repeat(np.asarray([[0.0, 0.0, -1.0]]), 4, axis=0),
        np.ones((4,), dtype=np.float32),
        parent_ids,
        probability,
        np.zeros((4,), dtype=np.float32),
        np.full((4,), 0.1, dtype=np.float32),
        child,
        physical,
        graph,
        camera,
        maximum_modes=8,
        proposal_trials=64,
        maximum_parent_candidates=1,
        extension_support_count=4,
        extensions_per_pair=2,
        preliminary_pose_count=32,
        random_seed=7,
    )
    assert modes.poses_w2c.shape[0] > 0
    centers = np.asarray([
        -pose[:3, :3].T @ pose[:3, 3] for pose in modes.poses_w2c
    ])
    assert np.min(np.linalg.norm(centers, axis=1)) < 1e-5
    assert np.any(np.all(modes.configuration_child_rows == np.arange(4), axis=1))


def test_soft_pose_likelihood_marginalizes_identity_and_typed_null():
    physical, _graph, camera, xy, extent, _ids, _probability, _child = _fixture()
    wrong = np.asarray([3, 2, 1, 0], dtype=np.int64)
    children = np.column_stack((np.arange(4, dtype=np.int64), wrong))
    parent_probability = np.full((4, 2), 0.45, dtype=np.float64)
    child_probability = np.ones((4, 2), dtype=np.float64)
    descriptor = np.eye(4, dtype=np.float64)
    graph = build_query_edge_graph(
        xy / [camera.width, camera.height], descriptor,
        local_neighbors=2, long_neighbors=1,
    )
    correct = pose_conditioned_soft_maplet_evidence(
        np.eye(4),
        xy / [camera.width, camera.height],
        extent / [camera.width, camera.height],
        graph,
        parent_probability,
        np.zeros((4,), dtype=np.float64),
        np.full((4,), 0.1, dtype=np.float64),
        children,
        child_probability,
        physical,
        camera,
        maximum_parent_candidates=2,
    )
    wrong_pose = np.eye(4)
    wrong_pose[0, 3] = 2.0
    displaced = pose_conditioned_soft_maplet_evidence(
        wrong_pose,
        xy / [camera.width, camera.height],
        extent / [camera.width, camera.height],
        graph,
        parent_probability,
        np.zeros((4,), dtype=np.float64),
        np.full((4,), 0.1, dtype=np.float64),
        children,
        child_probability,
        physical,
        camera,
        maximum_parent_candidates=2,
    )
    np.testing.assert_array_equal(correct.chosen_child_rows, np.arange(4))
    assert correct.supporting_region_count == 4
    assert 0.0 < correct.null_mass_mean < 1.0
    assert correct.score > displaced.score


def test_soft_pose_likelihood_does_not_harden_best_child_before_pose():
    physical, _graph, camera, xy, extent, _ids, _probability, _child = _fixture()
    wrong = np.asarray([3, 2, 1, 0], dtype=np.int64)
    children = np.stack((wrong, np.arange(4, dtype=np.int64)), axis=1)[:, None]
    child_probability = np.tile(
        np.asarray([[[0.51, 0.49]]], dtype=np.float64), (4, 1, 1),
    )
    graph = build_query_edge_graph(
        xy / [camera.width, camera.height], np.eye(4),
        local_neighbors=2, long_neighbors=1,
    )
    evidence = pose_conditioned_soft_maplet_evidence(
        np.eye(4),
        xy / [camera.width, camera.height],
        extent / [camera.width, camera.height],
        graph,
        np.full((4, 1), 0.9, dtype=np.float64),
        np.zeros((4,), dtype=np.float64),
        np.full((4,), 0.1, dtype=np.float64),
        children,
        child_probability,
        physical,
        camera,
        maximum_parent_candidates=1,
        edge_candidate_count=2,
    )
    np.testing.assert_array_equal(evidence.chosen_child_rows, np.arange(4))


def test_geometry_guided_soft_identity_branch_recovers_pose_without_pnp():
    physical, graph, camera, xy, extent, parent_ids, probability, child = _fixture()
    modes = generate_geometry_guided_configuration_pose_modes(
        xy,
        extent,
        np.eye(4, dtype=np.float32),
        physical.child_centers[:, 2],
        np.repeat(np.asarray([[0.0, 0.0, -1.0]]), 4, axis=0),
        np.ones((4,), dtype=np.float32),
        parent_ids,
        probability,
        np.zeros((4,), dtype=np.float32),
        np.full((4,), 0.1, dtype=np.float32),
        child,
        physical,
        graph,
        camera,
        maximum_modes=8,
        proposal_trials=64,
        maximum_parent_candidates=1,
        extension_support_count=4,
        extensions_per_pair=2,
        preliminary_pose_count=32,
        soft_identity_marginalization=True,
        soft_maximum_edges=16,
        random_seed=7,
    )
    assert modes.poses_w2c.shape[0] > 0
    centers = np.asarray([
        -pose[:3, :3].T @ pose[:3, 3] for pose in modes.poses_w2c
    ])
    assert np.min(np.linalg.norm(centers, axis=1)) < 1e-5
