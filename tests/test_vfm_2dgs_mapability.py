import numpy as np

from feature_extract.vfm.vfm_2dgs_mapability import evaluate_vfm_2dgs_mapability
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap, Vfm2DgsObservationBank


def _anchor_map() -> Vfm2DgsAnchorMap:
    support_offsets = np.asarray([0, 2, 4], dtype=np.int64)
    return Vfm2DgsAnchorMap(
        anchor_ids=np.asarray([10, 20], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None], (2, 1, 1)),
        features=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        feature_variances=np.asarray([0.01, 0.02], dtype=np.float32),
        quality_scores=np.asarray([0.8, 0.7], dtype=np.float32),
        purity_scores=np.asarray([0.9, 0.85], dtype=np.float32),
        observation_counts=np.asarray([2, 2], dtype=np.int64),
        surface_support_counts=np.asarray([2, 2], dtype=np.int64),
        support_offsets=support_offsets,
        support_element_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        support_weights=np.asarray([0.7, 0.3, 0.6, 0.4], dtype=np.float32),
        observed_view_ids=(("a.png", "b.png"), ("a.png", "b.png")),
    )


def _observation_bank() -> Vfm2DgsObservationBank:
    support_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    return Vfm2DgsObservationBank(
        image_ids=("a.png", "b.png", "a.png", "b.png"),
        token_indices=np.asarray([5, 6, 7, 8], dtype=np.int64),
        token_xy=np.asarray([[1.0, 1.0], [1.2, 1.0], [3.0, 3.0], [3.1, 3.0]], dtype=np.float32),
        features=np.asarray(
            [
                [0.98, 0.05, 0.0],
                [0.99, 0.02, 0.0],
                [0.02, 0.99, 0.0],
                [0.04, 0.98, 0.0],
            ],
            dtype=np.float32,
        ),
        centers=np.asarray(
            [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [1.0, 0.0, 4.0]],
            dtype=np.float64,
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None], (4, 1, 1)),
        support_offsets=support_offsets,
        element_ids=np.asarray([1, 2, 1, 2, 3, 4, 3, 4], dtype=np.int64),
        element_weights=np.asarray([0.7, 0.3, 0.6, 0.4, 0.6, 0.4, 0.5, 0.5], dtype=np.float32),
        purity_scores=np.asarray([0.9, 0.8, 0.85, 0.75], dtype=np.float32),
        component_concentrations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        quality_scores=np.asarray([0.8, 0.7, 0.7, 0.6], dtype=np.float32),
        view_directions=np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float32),
    )


def test_vfm_2dgs_mapability_reports_leave_one_out_retrieval_and_coverage() -> None:
    result = evaluate_vfm_2dgs_mapability(_anchor_map(), _observation_bank(), token_grid_shape=(4, 4))

    assert result["assignment"]["assigned_observation_count"] == 4
    assert result["retrieval"]["query_count"] == 4
    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["recall_at_5"] == 1.0
    assert result["retrieval"]["mean_positive_cosine"] > 0.95
    assert result["coverage"]["view_count"] == 2
    assert result["coverage"]["mean_observations_per_view"] == 2.0
    assert result["coverage"]["mean_token_grid_occupancy"] == 2.0 / 16.0


def test_vfm_2dgs_mapability_reports_source_wise_retrieval_when_source_ids_are_available() -> None:
    bank = Vfm2DgsObservationBank(
        image_ids=("a.png", "b.png", "a.png", "b.png"),
        token_indices=np.asarray([5, 6, 7, 8], dtype=np.int64),
        token_xy=np.asarray([[1.0, 1.0], [1.2, 1.0], [3.0, 3.0], [3.1, 3.0]], dtype=np.float32),
        features=np.asarray(
            [
                [0.98, 0.05, 0.0],
                [0.99, 0.02, 0.0],
                [0.02, 0.99, 0.0],
                [0.04, 0.98, 0.0],
            ],
            dtype=np.float32,
        ),
        centers=np.asarray(
            [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [1.0, 0.0, 4.0]],
            dtype=np.float64,
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None], (4, 1, 1)),
        support_offsets=np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        element_ids=np.asarray([1, 2, 1, 2, 3, 4, 3, 4], dtype=np.int64),
        element_weights=np.asarray([0.7, 0.3, 0.6, 0.4, 0.6, 0.4, 0.5, 0.5], dtype=np.float32),
        purity_scores=np.asarray([0.9, 0.8, 0.85, 0.75], dtype=np.float32),
        component_concentrations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        quality_scores=np.asarray([0.8, 0.7, 0.7, 0.6], dtype=np.float32),
        view_directions=np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float32),
        source_ids=("broad", "broad", "surface", "surface"),
    )

    result = evaluate_vfm_2dgs_mapability(_anchor_map(), bank, token_grid_shape=(4, 4))

    assert result["source_breakdown"]["broad"]["observation_count"] == 2
    assert result["source_breakdown"]["broad"]["retrieval"]["query_count"] == 2
    assert result["source_breakdown"]["surface"]["retrieval"]["recall_at_1"] == 1.0


def test_vfm_2dgs_mapability_reports_surface_seed_consensus_when_surface_layer_is_available() -> None:
    elements = SurfaceElementMap(
        element_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        parent_gaussian_indices=np.asarray([10, 10, 20, 20], dtype=np.int64),
        centers=np.asarray(
            [[0.0, 0.0, 4.0], [0.1, 0.0, 4.0], [1.0, 0.0, 4.0], [1.1, 0.0, 4.0]],
            dtype=np.float64,
        ),
        tangent1=np.asarray([[1.0, 0.0, 0.0]] * 4, dtype=np.float32),
        tangent2=np.asarray([[0.0, 1.0, 0.0]] * 4, dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        scale1=np.asarray([0.1] * 4, dtype=np.float32),
        scale2=np.asarray([0.1] * 4, dtype=np.float32),
        opacity=np.asarray([1.0] * 4, dtype=np.float32),
        area=np.asarray([0.01] * 4, dtype=np.float32),
        adjacency=tuple(np.zeros((0,), dtype=np.int64) for _ in range(4)),
    )

    result = evaluate_vfm_2dgs_mapability(
        _anchor_map(),
        _observation_bank(),
        token_grid_shape=(4, 4),
        surface_elements=elements,
    )

    assert result["surface_seed_consensus"]["parent_seed_count"] == 2
    assert result["surface_seed_consensus"]["multi_view_parent_seed_count"] == 2
    assert result["surface_seed_consensus"]["multi_view_parent_anchor_recall"] == 1.0
