from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    ParentLayoutCamera,
    score_parent_support_layout_guide,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    all_radio_token_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.streaming_parent_layout_guide_gpu import (
    _stable_topk,
    stream_parent_support_layout_topk_gpu,
)


def _physical():
    return SimpleNamespace(
        maplet_ids=np.asarray([10, 20], dtype=np.int64),
        maplet_centers=np.asarray(
            [[0.0, 0.0, 4.0], [1.25, 0.0, 5.0]], dtype=np.float64,
        ),
        maplet_frames=np.asarray([np.eye(3), np.eye(3)], dtype=np.float64),
        maplet_extents=np.asarray(
            [[0.45, 0.45, 0.05], [0.40, 0.60, 0.05]], dtype=np.float64,
        ),
        maplet_normals=np.asarray(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float64,
        ),
        content_sha256="physical",
    )


def _retrieval():
    xy = all_radio_token_coordinates(4, 4)
    ids = np.full((16, 2), -1, dtype=np.int64)
    probability = np.zeros((16, 2), dtype=np.float32)
    central = (
        (xy[:, 0] >= 1) & (xy[:, 0] < 3)
        & (xy[:, 1] >= 1) & (xy[:, 1] < 3)
    )
    right = xy[:, 0] >= 2
    ids[central, 0] = 10
    probability[central, 0] = 0.75
    ids[right, 1] = 20
    probability[right, 1] = 0.25
    return SimpleNamespace(
        token_xy=xy,
        token_parent_ids=ids,
        token_parent_probabilities=probability,
        metadata={"token_height": 4, "token_width": 4},
        physical_map_sha256="physical",
    )


def _factors() -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 8.0],
        ],
        dtype=np.float64,
    )
    rotation = np.stack(
        [
            np.eye(3, dtype=np.float64),
            np.diag([-1.0, 1.0, -1.0]),
            np.diag([-1.0, -1.0, 1.0]),
        ],
    )
    return position, rotation


def test_stable_topk_resolves_an_all_tied_boundary_by_factor_rows():
    score = np.full((8,), 0.5, dtype=np.float64)
    position = np.asarray([2, 0, 1, 0, 2, 1, 0, 1], dtype=np.int64)
    orientation = np.asarray([1, 2, 0, 0, 0, 2, 1, 1], dtype=np.int64)

    selected = _stable_topk(score, position, orientation, 4)
    selected_pairs = np.stack([position[selected], orientation[selected]], axis=1)
    np.testing.assert_array_equal(
        selected_pairs,
        np.asarray([[0, 0], [0, 1], [0, 2], [1, 0]], dtype=np.int64),
    )

    # Input and streaming order are not part of the tie contract.
    reverse = np.arange(score.size - 1, -1, -1)
    reverse_selected = _stable_topk(
        score[reverse], position[reverse], orientation[reverse], 4,
    )
    reverse_pairs = np.stack(
        [position[reverse][reverse_selected], orientation[reverse][reverse_selected]],
        axis=1,
    )
    np.testing.assert_array_equal(reverse_pairs, selected_pairs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_streaming_float64_matches_cpu_and_is_chunk_and_orientation_order_stable():
    position, rotation = _factors()
    retrieval = _retrieval()
    physical = _physical()
    camera = ParentLayoutCamera(1, 4, 4, (4.0, 4.0, 2.0, 2.0))
    pair_count = int(position.shape[0] * rotation.shape[0])
    cpu = score_parent_support_layout_guide(
        position[None], rotation,
        np.ones((rotation.shape[0],), dtype=bool),
        np.arange(rotation.shape[0], dtype=np.int64),
        retrieval, physical, camera,
        maximum_query_parents=2,
        topk=pair_count,
        candidate_batch_size=2,
    )
    forward = stream_parent_support_layout_topk_gpu(
        position, rotation, retrieval, physical, camera,
        maximum_query_parents=2,
        topk=pair_count,
        position_chunk_size=1,
        device="cuda:0",
        torch_dtype="float64",
    )
    reverse = stream_parent_support_layout_topk_gpu(
        position, rotation, retrieval, physical, camera,
        maximum_query_parents=2,
        topk=pair_count,
        position_chunk_size=3,
        device="cuda:0",
        torch_dtype="float64",
        reverse_position_chunks=True,
        reverse_orientation_evaluation=True,
    )

    np.testing.assert_array_equal(
        forward.top_position_rows, cpu.top_position_factor_indices,
    )
    np.testing.assert_array_equal(
        forward.top_orientation_rows, cpu.top_orientation_factor_indices,
    )
    np.testing.assert_allclose(forward.top_scores, cpu.top_scores, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(
        forward.selected_query_parent_ids, cpu.selected_query_parent_ids,
    )
    np.testing.assert_allclose(
        forward.selected_query_parent_probability_mass,
        cpu.selected_query_parent_probability_mass,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        forward.complete_query_parent_probability_mass
        == cpu.complete_query_parent_probability_mass
    )
    assert (
        forward.selected_query_parent_probability_mass_total
        == cpu.selected_query_parent_probability_mass_total
    )
    diagnostic_pairs = (
        ("top_visible_parent_counts", "top_visible_parent_counts"),
        ("top_front_facing_parent_counts", "top_front_facing_parent_counts"),
        ("top_positive_depth_parent_counts", "top_positive_depth_parent_counts"),
        ("top_center_in_image_parent_counts", "top_center_in_image_parent_counts"),
    )
    for gpu_name, cpu_name in diagnostic_pairs:
        np.testing.assert_array_equal(
            getattr(forward, gpu_name), getattr(cpu, cpu_name),
        )
    np.testing.assert_allclose(
        forward.top_projected_token_footprint_mass,
        cpu.top_projected_token_footprint_mass,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        forward.top_sqrt_overlap_mass,
        cpu.top_sqrt_overlap_mass,
        rtol=0.0,
        atol=1e-12,
    )

    # Reversing both chunk traversal and orientation evaluation cannot change
    # factor identity, score, or diagnostics.
    stable_fields = (
        "top_scores",
        "top_position_rows",
        "top_orientation_rows",
        "top_visible_parent_counts",
        "top_front_facing_parent_counts",
        "top_positive_depth_parent_counts",
        "top_center_in_image_parent_counts",
        "top_projected_token_footprint_mass",
        "top_sqrt_overlap_mass",
    )
    for name in stable_fields:
        np.testing.assert_array_equal(getattr(reverse, name), getattr(forward, name))
    assert forward.total_factor_pair_count == reverse.total_factor_pair_count == pair_count


def test_stable_topk_and_streaming_api_fail_closed_on_invalid_inputs():
    with pytest.raises(ValueError, match="arrays differ"):
        _stable_topk(np.ones((2,)), np.zeros((1,)), np.zeros((2,)), 1)
    with pytest.raises(ValueError, match="arrays differ"):
        _stable_topk(np.empty((0,)), np.empty((0,)), np.empty((0,)), 1)
    with pytest.raises(ValueError, match="budget"):
        _stable_topk(np.ones((2,)), np.zeros((2,)), np.zeros((2,)), 0)

    position, rotation = _factors()
    args = (_retrieval(), _physical(), ParentLayoutCamera(0, 4, 4, (4.0, 2.0, 2.0)))
    with pytest.raises(ValueError, match="factors/configuration"):
        stream_parent_support_layout_topk_gpu(
            position[None], rotation, *args, device="cuda:0",
        )
    with pytest.raises(ValueError, match="factors/configuration"):
        stream_parent_support_layout_topk_gpu(
            position, rotation, *args, topk=0, device="cuda:0",
        )
    with pytest.raises(ValueError, match="factors/configuration"):
        stream_parent_support_layout_topk_gpu(
            position, rotation, *args, position_chunk_size=0, device="cuda:0",
        )
    with pytest.raises(ValueError, match="available CUDA"):
        stream_parent_support_layout_topk_gpu(
            position, rotation, *args, device="cpu",
        )
