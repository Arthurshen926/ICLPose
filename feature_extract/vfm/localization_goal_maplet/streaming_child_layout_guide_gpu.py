"""Streaming parent/child RADIO-layout guide over the frozen global domain.

Child support is built solely from the query's sparse RADIO child posterior
and geometry in the physical map.  Each selected child is represented by the
tight center-fixed symmetric box in its frozen physical child frame that encloses every member
primitive rectangle.  Primitive half-extents are projected exactly into that
frame as ``abs(F*t1)*scale1 + abs(F*t2)*scale2``.  This OBB is a conservative
footprint carrier; it is not a renderer, visibility mesh, free-space
certificate, or hard image-to-map correspondence.  As in the frozen parent
control, image bounds are obtained from projected center/corners; under
SIMPLE_RADIAL this is a vertex-projected PFIR approximation rather than an
exact analytic projected silhouette.

Three rankings are frozen in one position-chunk traversal: the existing
parent affinity control, child affinity, and their unweighted geometric mean.
The geometric mean is symmetric, monotone, bounded in [0,1], has no tunable
continuous weight, and requires evidence from both hierarchy levels.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import torch

from .parent_support_layout_guide import (
    ParentLayoutCamera,
    _project_camera_points,
    projected_image_bounds_to_radio_token_footprint,
)
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval, all_radio_token_coordinates
from .streaming_parent_layout_guide_gpu import (
    _PreparedGuide,
    _prepare,
    _score_position_orientation_block,
    _score_selected_factor_pairs,
    _stable_topk,
    _torch_dtype,
)


CHILD_SCORE_SEMANTICS = (
    "fixed_complete_child_mass_denominator_symmetric_sqrt_token_posterior_"
    "exact_member_rectangle_center_fixed_child_frame_obb_binary_footprint_"
    "affinity_v2"
)
CHILD_GEOMETRY_CONTRACT = (
    "selected_scene_child_exact_member_primitive_rectangle_fixed_child_frame_"
    "tight_center_fixed_symmetric_obb_center_plus_8_corners_unsigned_child_"
    "normal_vertex_projected_pfir_bounds_approximation_v2"
)
FUSION_SEMANTICS = "unweighted_geometric_mean_parent_child_layout_affinity_v1"
MULTI_STREAMING_SEMANTICS = (
    "one_global_factor_stream_parent_control_child_and_geometric_mean_"
    "stable_topk_no_full_score_materialization_v1"
)
MODES = ("parent", "child", "geometric_mean")


@dataclass(frozen=True)
class RankedSupportLayout:
    mode: str
    top_scores: np.ndarray
    top_position_rows: np.ndarray
    top_orientation_rows: np.ndarray
    top_parent_scores: np.ndarray
    top_child_scores: np.ndarray
    top_parent_visible_counts: np.ndarray
    top_child_visible_counts: np.ndarray
    top_child_front_facing_counts: np.ndarray
    top_child_positive_depth_counts: np.ndarray
    top_child_center_in_image_counts: np.ndarray
    top_child_projected_token_footprint_mass: np.ndarray
    top_child_sqrt_overlap_mass: np.ndarray

    def __post_init__(self) -> None:
        score = np.asarray(self.top_scores, dtype=np.float64).reshape(-1)
        position = np.asarray(self.top_position_rows, dtype=np.int64).reshape(-1)
        orientation = np.asarray(self.top_orientation_rows, dtype=np.int64).reshape(-1)
        if (
            self.mode not in MODES or score.size == 0
            or any(np.asarray(value).shape != score.shape for value in (
                position, orientation, self.top_parent_scores,
                self.top_child_scores, self.top_parent_visible_counts,
                self.top_child_visible_counts, self.top_child_front_facing_counts,
                self.top_child_positive_depth_counts,
                self.top_child_center_in_image_counts,
                self.top_child_projected_token_footprint_mass,
                self.top_child_sqrt_overlap_mass,
            ))
            or np.any(~np.isfinite(score))
            or np.any((score < 0.0) | (score > 1.0 + 1e-12))
            or np.unique(np.stack([position, orientation], axis=1), axis=0).shape[0]
            != score.size
            or not np.array_equal(
                np.lexsort((orientation, position, -score)), np.arange(score.size),
            )
        ):
            raise ValueError("ranked support layout contract differs")


@dataclass(frozen=True)
class StreamingHierarchyLayoutResult:
    rankings: dict[str, RankedSupportLayout]
    selected_parent_ids: np.ndarray
    selected_child_rows: np.ndarray
    complete_parent_probability_mass: float
    complete_child_probability_mass: float
    selected_parent_probability_mass_total: float
    selected_child_probability_mass_total: float
    total_factor_pair_count: int
    elapsed_seconds: float
    scoring_seconds: float
    merge_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    position_chunk_size: int
    torch_dtype: str

    def __post_init__(self) -> None:
        if set(self.rankings) != set(MODES):
            raise ValueError("hierarchy layout result modes differ")


@dataclass(frozen=True)
class RankedPositionLayout:
    """A position ranking with orientation analytically retained as a factor."""

    mode: str
    top_scores: np.ndarray
    top_position_rows: np.ndarray
    diagnostic_best_orientation_rows: np.ndarray

    def __post_init__(self) -> None:
        score = np.asarray(self.top_scores, dtype=np.float64).reshape(-1)
        position = np.asarray(self.top_position_rows, dtype=np.int64).reshape(-1)
        orientation = np.asarray(
            self.diagnostic_best_orientation_rows, dtype=np.int64,
        ).reshape(-1)
        if (
            self.mode not in MODES or score.size == 0
            or position.shape != score.shape or orientation.shape != score.shape
            or np.any(~np.isfinite(score))
            or np.any((score < 0.0) | (score > 1.0 + 1e-12))
            or np.unique(position).size != score.size
            or not np.array_equal(
                np.lexsort((position, -score)), np.arange(score.size),
            )
        ):
            raise ValueError("ranked position layout contract differs")


@dataclass(frozen=True)
class StreamingHierarchyPositionResult:
    rankings: dict[str, RankedPositionLayout]
    total_position_count: int
    implicit_orientation_count: int
    elapsed_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int

    def __post_init__(self) -> None:
        if set(self.rankings) != set(MODES):
            raise ValueError("hierarchy position result modes differ")


def exact_child_member_rectangle_obbs(
    physical: GoalMapletPhysicalMap,
    child_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the center-fixed symmetric OBB enclosing member rectangles.

    The returned corner *set* is invariant to an independent sign flip of any
    frame axis: the corresponding local sign is simply permuted.  It is also
    invariant to a consistent axis permutation.  Normal sign is irrelevant to
    the incidence gate because the scorer consumes its absolute dot product.
    """

    rows = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    child_count = int(physical.child_parent_rows.size)
    if (
        rows.size == 0 or np.unique(rows).size != rows.size
        or np.any(rows < 0) or np.any(rows >= child_count)
    ):
        raise ValueError("selected physical child rows differ")
    child_centers = np.asarray(physical.child_centers[rows], dtype=np.float64)
    child_frames = np.asarray(physical.child_frames[rows], dtype=np.float64)
    if (
        not np.allclose(
            child_frames @ child_frames.transpose(0, 2, 1),
            np.eye(3), atol=1.0e-7, rtol=0.0,
        )
        or not np.allclose(np.abs(np.linalg.det(child_frames)), 1.0, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("selected child frames are not finite orthonormal O(3)")
    exact_extents = np.empty((rows.size, 3), dtype=np.float64)
    for output_row, child_row in enumerate(rows.tolist()):
        start = int(physical.child_member_offsets[child_row])
        end = int(physical.child_member_offsets[child_row + 1])
        members = np.unique(np.asarray(
            physical.child_member_primitive_rows[start:end], dtype=np.int64,
        ))
        if members.size == 0:
            raise ValueError("selected child has no exact primitive member")
        if (
            np.any(np.asarray(physical.primitive_scale1[members]) <= 0.0)
            or np.any(np.asarray(physical.primitive_scale2[members]) <= 0.0)
        ):
            raise ValueError("selected child primitive scale is not positive")
        frame = child_frames[output_row]
        member_center = np.asarray(physical.primitive_centers[members], dtype=np.float64)
        local_center = (member_center - child_centers[output_row]) @ frame.T
        # Rows of frame are its world-space axes.  This is the exact support
        # function of a primitive rectangle along each frozen OBB axis.
        projected_half_extent = (
            np.abs(np.asarray(physical.primitive_tangent1[members], dtype=np.float64) @ frame.T)
            * np.asarray(physical.primitive_scale1[members], dtype=np.float64)[:, None]
            + np.abs(np.asarray(physical.primitive_tangent2[members], dtype=np.float64) @ frame.T)
            * np.asarray(physical.primitive_scale2[members], dtype=np.float64)[:, None]
        )
        exact_extents[output_row] = np.max(
            np.abs(local_center) + projected_half_extent, axis=0,
        )
    signs = np.asarray([
        [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    corners = child_centers[:, None] + np.einsum(
        "nki,nij->nkj", signs[None] * exact_extents[:, None], child_frames,
    )
    return (
        child_centers,
        corners,
        np.asarray(physical.child_normals[rows], dtype=np.float64),
        rows.copy(),
    )


def query_child_layout(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_scene_children: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Expand selected scene-child posterior and preserve complete denominator."""

    height = int(retrieval.metadata["token_height"])
    width = int(retrieval.metadata["token_width"])
    if retrieval.physical_map_sha256 != physical.content_sha256:
        raise ValueError("child layout retrieval and physical lineage differ")
    if not np.array_equal(
        retrieval.token_xy, all_radio_token_coordinates(height, width),
    ):
        raise ValueError("child layout guide requires complete row-major tokens")
    child_count = int(physical.child_parent_rows.size)
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    probability = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    scene = np.asarray(retrieval.scene_child_rows, dtype=np.int64).reshape(-1)
    scene_scores = np.asarray(retrieval.scene_child_scores, dtype=np.float64).reshape(-1)
    budget = int(maximum_scene_children)
    if (
        rows.ndim != 2 or rows.shape[0] != height * width
        or probability.shape != rows.shape
        or scene.shape != scene_scores.shape or budget <= 0 or scene.size < budget
        or np.any(~np.isfinite(probability)) or np.any(probability < 0.0)
        or np.any(~np.isfinite(scene_scores))
    ):
        raise ValueError("child layout posterior/scene arrays differ")
    selected = scene[:budget]
    if (
        np.unique(selected).size != budget
        or np.any(selected < 0) or np.any(selected >= child_count)
    ):
        raise ValueError("selected scene-child K is invalid")
    valid = (rows >= 0) & (rows < child_count) & (probability > 0.0)
    if np.any((probability > 0.0) & ~((rows >= 0) & (rows < child_count))):
        raise ValueError("positive child posterior refers outside physical map")
    for token_rows, token_probability in zip(rows, probability):
        positive = token_rows[token_probability > 0.0]
        positive = positive[(positive >= 0) & (positive < child_count)]
        if np.unique(positive).size != positive.size:
            raise ValueError("child posterior repeats a row within one token")
    lookup = np.full((child_count,), -1, dtype=np.int64)
    lookup[selected] = np.arange(budget, dtype=np.int64)
    safe_rows = np.clip(rows, 0, max(child_count - 1, 0))
    selected_entry = valid & (lookup[safe_rows] >= 0)
    token_index = np.broadcast_to(
        np.arange(height * width, dtype=np.int64)[:, None], rows.shape,
    )
    layout = np.zeros((budget, height * width), dtype=np.float64)
    np.add.at(
        layout,
        (lookup[rows[selected_entry]], token_index[selected_entry]),
        probability[selected_entry],
    )
    sqrt_layout = np.sqrt(np.maximum(layout, 0.0)).reshape(budget, height, width)
    integral = np.pad(
        np.cumsum(np.cumsum(sqrt_layout, axis=1), axis=2),
        ((0, 0), (1, 0), (1, 0)),
    )
    return (
        selected,
        np.sum(layout, axis=1),
        integral,
        float(np.sum(probability[valid])),
    )


def _prepare_child(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_scene_children: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedGuide:
    rows, mass, integral, complete = query_child_layout(
        retrieval, physical, maximum_scene_children=maximum_scene_children,
    )
    centers, corners, normals, identities = exact_child_member_rectangle_obbs(
        physical, rows,
    )
    return _PreparedGuide(
        parent_centers=torch.as_tensor(centers, dtype=dtype, device=device),
        parent_corners=torch.as_tensor(corners, dtype=dtype, device=device),
        parent_normals=torch.as_tensor(normals, dtype=dtype, device=device),
        query_integral=torch.as_tensor(integral, dtype=dtype, device=device),
        query_mass=float(complete),
        parent_ids=identities,
        parent_probability_mass=np.asarray(mass, dtype=np.float64),
        token_height=int(retrieval.metadata["token_height"]),
        token_width=int(retrieval.metadata["token_width"]),
    )


def score_support_layout_numpy(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    support_centers_world: np.ndarray,
    support_corners_world: np.ndarray,
    support_normals_world: np.ndarray,
    query_sqrt_layout_integral: np.ndarray,
    complete_query_probability_mass: float,
    camera: ParentLayoutCamera,
    *,
    token_height: int,
    token_width: int,
    candidate_batch_size: int = 256,
    minimum_depth_m: float = 0.05,
    minimum_front_incidence: float = 0.02,
) -> dict[str, np.ndarray]:
    """Independent NumPy authority for parent/child support parity gates."""

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    center = np.asarray(support_centers_world, dtype=np.float64)
    corners = np.asarray(support_corners_world, dtype=np.float64)
    normal = np.asarray(support_normals_world, dtype=np.float64)
    integral = np.asarray(query_sqrt_layout_integral, dtype=np.float64)
    if (
        position.ndim != 2 or position.shape[1:] != (3,)
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or center.ndim != 2 or center.shape[1:] != (3,)
        or corners.ndim != 3 or corners.shape[0] != center.shape[0]
        or corners.shape[2:] != (3,) or normal.shape != center.shape
        or integral.shape != (center.shape[0], int(token_height) + 1, int(token_width) + 1)
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or np.any(~np.isfinite(center)) or np.any(~np.isfinite(corners))
        or np.any(~np.isfinite(normal)) or np.any(~np.isfinite(integral))
        or float(complete_query_probability_mass) <= 0.0
        or int(candidate_batch_size) <= 0
    ):
        raise ValueError("NumPy support-layout arrays/configuration differ")
    total = int(position.shape[0] * rotation.shape[0])
    result = {
        "score": np.empty((total,), dtype=np.float64),
        "visible": np.empty((total,), dtype=np.int16),
        "front": np.empty((total,), dtype=np.int16),
        "depth": np.empty((total,), dtype=np.int16),
        "center_image": np.empty((total,), dtype=np.int16),
        "footprint": np.empty((total,), dtype=np.float64),
        "overlap": np.empty((total,), dtype=np.float64),
    }
    parent_index = np.arange(center.shape[0], dtype=np.int64)[None]
    orientation_count = int(rotation.shape[0])
    for start in range(0, total, int(candidate_batch_size)):
        end = min(start + int(candidate_batch_size), total)
        pair = np.arange(start, end, dtype=np.int64)
        candidate_position = position[pair // orientation_count]
        candidate_rotation = rotation[pair % orientation_count]
        center_camera = np.einsum(
            "bij,bpj->bpi", candidate_rotation,
            center[None] - candidate_position[:, None],
        )
        corner_camera = np.einsum(
            "bij,bpkj->bpki", candidate_rotation,
            corners[None] - candidate_position[:, None, None],
        )
        view = candidate_position[:, None] - center[None]
        view /= np.maximum(np.linalg.norm(view, axis=2, keepdims=True), 1.0e-12)
        front = np.abs(np.sum(view * normal[None], axis=2)) >= float(
            minimum_front_incidence
        )
        positive = (
            (center_camera[..., 2] > float(minimum_depth_m))
            & np.all(corner_camera[..., 2] > float(minimum_depth_m), axis=2)
        )
        center_xy = _project_camera_points(center_camera, camera)
        corner_xy = _project_camera_points(corner_camera, camera)
        all_xy = np.concatenate((center_xy[:, :, None], corner_xy), axis=2)
        finite = np.all(np.isfinite(all_xy), axis=(2, 3))
        low = np.min(all_xy, axis=2)
        high = np.max(all_xy, axis=2)
        x0, y0, x1, y1 = projected_image_bounds_to_radio_token_footprint(
            low, high, image_width=camera.width, image_height=camera.height,
            token_width=int(token_width), token_height=int(token_height), finite=finite,
        )
        nonempty = (x1 > x0) & (y1 > y0)
        visible = front & positive & finite & nonempty
        area = np.where(visible, (x1 - x0) * (y1 - y0), 0).astype(np.float64)
        local_overlap = (
            integral[parent_index, y1, x1] - integral[parent_index, y0, x1]
            - integral[parent_index, y1, x0] + integral[parent_index, y0, x0]
        )
        local_overlap = np.where(visible, np.maximum(local_overlap, 0.0), 0.0)
        footprint = np.sum(area, axis=1)
        overlap = np.sum(local_overlap, axis=1)
        denominator = np.sqrt(float(complete_query_probability_mass) * footprint)
        result["score"][start:end] = np.clip(np.divide(
            overlap, denominator, out=np.zeros_like(overlap),
            where=denominator > 1.0e-12,
        ), 0.0, 1.0)
        result["visible"][start:end] = np.sum(visible, axis=1).astype(np.int16)
        result["front"][start:end] = np.sum(front, axis=1).astype(np.int16)
        result["depth"][start:end] = np.sum(positive, axis=1).astype(np.int16)
        center_in_image = (
            (center_xy[..., 0] >= 0.0) & (center_xy[..., 0] < camera.width)
            & (center_xy[..., 1] >= 0.0) & (center_xy[..., 1] < camera.height)
        )
        result["center_image"][start:end] = np.sum(
            visible & center_in_image, axis=1,
        ).astype(np.int16)
        result["footprint"][start:end] = footprint
        result["overlap"][start:end] = overlap
    shape = (position.shape[0], rotation.shape[0])
    return {key: value.reshape(shape) for key, value in result.items()}
def stream_hierarchy_support_layout_topk_gpu(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    camera: ParentLayoutCamera,
    *,
    maximum_query_parents: int = 32,
    maximum_scene_children: int = 64,
    topk: int = 4096,
    position_chunk_size: int = 512,
    device: str | torch.device = "cuda:0",
    torch_dtype: Literal["float64"] = "float64",
    minimum_depth_m: float = 0.05,
    minimum_front_incidence: float = 0.02,
    reverse_position_chunks: bool = False,
    reverse_orientation_evaluation: bool = False,
) -> StreamingHierarchyLayoutResult:
    """Freeze parent, child, and geometric-mean Top-K in one global stream."""

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    if (
        position.ndim != 2 or position.shape[1:] != (3,)
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or position.shape[0] <= 0 or rotation.shape[0] <= 0
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or int(topk) <= 0 or int(position_chunk_size) <= 0
        or str(torch_dtype) != "float64"
    ):
        raise ValueError("hierarchy streaming factors/configuration differ")
    device_value = torch.device(device)
    if device_value.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("hierarchy streaming layout requires CUDA")
    dtype = _torch_dtype(torch_dtype)
    device_index = (
        int(device_value.index) if device_value.index is not None
        else int(torch.cuda.current_device())
    )
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    parent = _prepare(
        retrieval, physical, maximum_query_parents=int(maximum_query_parents),
        device=device_value, dtype=dtype,
    )
    child = _prepare_child(
        retrieval, physical, maximum_scene_children=int(maximum_scene_children),
        device=device_value, dtype=dtype,
    )
    orientation_order = np.arange(rotation.shape[0], dtype=np.int64)
    if bool(reverse_orientation_evaluation):
        orientation_order = orientation_order[::-1].copy()
    inverse_orientation = np.argsort(orientation_order)
    inverse_gpu = torch.as_tensor(inverse_orientation, device=device_value)
    rotation_gpu = torch.as_tensor(
        rotation[orientation_order], dtype=dtype, device=device_value,
    )
    bounds = [
        (start, min(start + int(position_chunk_size), position.shape[0]))
        for start in range(0, position.shape[0], int(position_chunk_size))
    ]
    if bool(reverse_position_chunks):
        bounds.reverse()
    best = {
        mode: {
            "score": np.empty((0,), dtype=np.float64),
            "position": np.empty((0,), dtype=np.int64),
            "orientation": np.empty((0,), dtype=np.int64),
        }
        for mode in MODES
    }
    scoring_seconds = 0.0
    merge_seconds = 0.0
    with torch.inference_mode():
        for start, end in bounds:
            tick = time.perf_counter()
            positions_gpu = torch.as_tensor(
                position[start:end], dtype=dtype, device=device_value,
            )
            parent_score = _score_position_orientation_block(
                positions_gpu, rotation_gpu, parent, camera,
                minimum_depth_m=float(minimum_depth_m),
                minimum_front_incidence=float(minimum_front_incidence),
                return_diagnostics=False,
            )["score"][:, inverse_gpu].reshape(-1).to(torch.float64).cpu().numpy()
            child_score = _score_position_orientation_block(
                positions_gpu, rotation_gpu, child, camera,
                minimum_depth_m=float(minimum_depth_m),
                minimum_front_incidence=float(minimum_front_incidence),
                return_diagnostics=False,
            )["score"][:, inverse_gpu].reshape(-1).to(torch.float64).cpu().numpy()
            scores = {
                "parent": parent_score,
                "child": child_score,
                "geometric_mean": np.sqrt(parent_score * child_score),
            }
            scoring_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            block_position = np.repeat(
                np.arange(start, end, dtype=np.int64), rotation.shape[0],
            )
            block_orientation = np.tile(
                np.arange(rotation.shape[0], dtype=np.int64), end - start,
            )
            for mode in MODES:
                selected = _stable_topk(
                    scores[mode], block_position, block_orientation, int(topk),
                )
                merged_score = np.concatenate([
                    best[mode]["score"], scores[mode][selected],
                ])
                merged_position = np.concatenate([
                    best[mode]["position"], block_position[selected],
                ])
                merged_orientation = np.concatenate([
                    best[mode]["orientation"], block_orientation[selected],
                ])
                keep = _stable_topk(
                    merged_score, merged_position, merged_orientation, int(topk),
                )
                best[mode] = {
                    "score": merged_score[keep],
                    "position": merged_position[keep],
                    "orientation": merged_orientation[keep],
                }
            merge_seconds += time.perf_counter() - tick

        union_pairs = sorted({
            (int(position_row), int(orientation_row))
            for mode in MODES
            for position_row, orientation_row in zip(
                best[mode]["position"], best[mode]["orientation"],
            )
        })
        pair_position = np.asarray([value[0] for value in union_pairs], dtype=np.int64)
        pair_orientation = np.asarray([value[1] for value in union_pairs], dtype=np.int64)
        component: dict[str, dict[str, np.ndarray]] = {"parent": {}, "child": {}}
        selected_batch = 2048
        for start in range(0, pair_position.size, selected_batch):
            end = min(start + selected_batch, pair_position.size)
            factor_position = torch.as_tensor(
                position[pair_position[start:end]], dtype=dtype, device=device_value,
            )
            factor_rotation = torch.as_tensor(
                rotation[pair_orientation[start:end]], dtype=dtype, device=device_value,
            )
            for name, prepared in (("parent", parent), ("child", child)):
                values = _score_selected_factor_pairs(
                    factor_position, factor_rotation, prepared, camera,
                    minimum_depth_m=float(minimum_depth_m),
                    minimum_front_incidence=float(minimum_front_incidence),
                )
                keys = (
                    ("score", "visible") if name == "parent" else
                    (
                        "score", "visible", "front", "depth", "center_image",
                        "footprint", "overlap",
                    )
                )
                for key in keys:
                    component[name].setdefault(key, []).append(
                        values[key].cpu().numpy()
                    )
        component = {
            name: {key: np.concatenate(parts) for key, parts in values.items()}
            for name, values in component.items()
        }
    pair_lookup = {value: index for index, value in enumerate(union_pairs)}
    rankings = {}
    for mode in MODES:
        lookup = np.asarray([
            pair_lookup[(int(p), int(o))]
            for p, o in zip(best[mode]["position"], best[mode]["orientation"])
        ], dtype=np.int64)
        rankings[mode] = RankedSupportLayout(
            mode=mode,
            top_scores=best[mode]["score"],
            top_position_rows=best[mode]["position"],
            top_orientation_rows=best[mode]["orientation"],
            top_parent_scores=np.asarray(component["parent"]["score"])[lookup],
            top_child_scores=np.asarray(component["child"]["score"])[lookup],
            top_parent_visible_counts=np.asarray(component["parent"]["visible"])[lookup],
            top_child_visible_counts=np.asarray(component["child"]["visible"])[lookup],
            top_child_front_facing_counts=np.asarray(component["child"]["front"])[lookup],
            top_child_positive_depth_counts=np.asarray(component["child"]["depth"])[lookup],
            top_child_center_in_image_counts=np.asarray(
                component["child"]["center_image"]
            )[lookup],
            top_child_projected_token_footprint_mass=np.asarray(
                component["child"]["footprint"]
            )[lookup],
            top_child_sqrt_overlap_mass=np.asarray(
                component["child"]["overlap"]
            )[lookup],
        )
    torch.cuda.synchronize(device_index)
    return StreamingHierarchyLayoutResult(
        rankings=rankings,
        selected_parent_ids=parent.parent_ids,
        selected_child_rows=child.parent_ids,
        complete_parent_probability_mass=parent.query_mass,
        complete_child_probability_mass=child.query_mass,
        selected_parent_probability_mass_total=float(
            np.sum(parent.parent_probability_mass)
        ),
        selected_child_probability_mass_total=float(np.sum(child.parent_probability_mass)),
        total_factor_pair_count=int(position.shape[0] * rotation.shape[0]),
        elapsed_seconds=float(time.perf_counter() - started),
        scoring_seconds=float(scoring_seconds),
        merge_seconds=float(merge_seconds),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device_index)),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device_index)),
        position_chunk_size=int(position_chunk_size),
        torch_dtype=str(torch_dtype),
    )


def stream_hierarchy_position_topk_gpu(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    camera: ParentLayoutCamera,
    *,
    maximum_query_parents: int = 32,
    maximum_scene_children: int = 64,
    topk_positions: int = 4096,
    position_chunk_size: int = 1024,
    device: str | torch.device = "cuda:0",
    minimum_depth_m: float = 0.05,
    minimum_front_incidence: float = 0.02,
) -> StreamingHierarchyPositionResult:
    """Rank positions while retaining the complete analytic SO(3) factor.

    Each position's priority is the maximum guide score over all 60 frozen
    orientations.  The argmax orientation is diagnostic only: downstream
    support is ``selected positions x all orientations`` and therefore keeps
    the analytic 45-degree orientation cover instead of spending the position
    budget repeatedly on different orientations of the same cell.
    """

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    if (
        position.ndim != 2 or position.shape[1:] != (3,)
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or position.shape[0] <= 0 or rotation.shape[0] <= 0
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or int(topk_positions) <= 0 or int(position_chunk_size) <= 0
    ):
        raise ValueError("hierarchy position factors/configuration differ")
    device_value = torch.device(device)
    if device_value.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("hierarchy position guide requires CUDA")
    device_index = int(device_value.index or 0)
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    dtype = torch.float64
    parent = _prepare(
        retrieval, physical, maximum_query_parents=int(maximum_query_parents),
        device=device_value, dtype=dtype,
    )
    child = _prepare_child(
        retrieval, physical, maximum_scene_children=int(maximum_scene_children),
        device=device_value, dtype=dtype,
    )
    rotation_gpu = torch.as_tensor(rotation, dtype=dtype, device=device_value)
    best = {mode: {
        "score": np.empty((0,), np.float64),
        "position": np.empty((0,), np.int64),
        "orientation": np.empty((0,), np.int64),
    } for mode in MODES}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, position.shape[0], int(position_chunk_size)):
            end = min(start + int(position_chunk_size), position.shape[0])
            position_gpu = torch.as_tensor(
                position[start:end], dtype=dtype, device=device_value,
            )
            parent_matrix = _score_position_orientation_block(
                position_gpu, rotation_gpu, parent, camera,
                minimum_depth_m=float(minimum_depth_m),
                minimum_front_incidence=float(minimum_front_incidence),
                return_diagnostics=False,
            )["score"].cpu().numpy()
            child_matrix = _score_position_orientation_block(
                position_gpu, rotation_gpu, child, camera,
                minimum_depth_m=float(minimum_depth_m),
                minimum_front_incidence=float(minimum_front_incidence),
                return_diagnostics=False,
            )["score"].cpu().numpy()
            matrices = {
                "parent": parent_matrix,
                "child": child_matrix,
                "geometric_mean": np.sqrt(parent_matrix * child_matrix),
            }
            block_position = np.arange(start, end, dtype=np.int64)
            for mode in MODES:
                # np.argmax returns the smallest orientation row on a tie.
                winning_orientation = np.argmax(matrices[mode], axis=1).astype(np.int64)
                score = matrices[mode][np.arange(end - start), winning_orientation]
                selected = _stable_topk(
                    score, block_position, np.zeros_like(block_position), int(topk_positions),
                )
                merged_score = np.concatenate((best[mode]["score"], score[selected]))
                merged_position = np.concatenate((
                    best[mode]["position"], block_position[selected],
                ))
                merged_orientation = np.concatenate((
                    best[mode]["orientation"], winning_orientation[selected],
                ))
                keep = _stable_topk(
                    merged_score, merged_position, np.zeros_like(merged_position),
                    int(topk_positions),
                )
                best[mode] = {
                    "score": merged_score[keep],
                    "position": merged_position[keep],
                    "orientation": merged_orientation[keep],
                }
    torch.cuda.synchronize(device_index)
    return StreamingHierarchyPositionResult(
        rankings={mode: RankedPositionLayout(
            mode=mode, top_scores=best[mode]["score"],
            top_position_rows=best[mode]["position"],
            diagnostic_best_orientation_rows=best[mode]["orientation"],
        ) for mode in MODES},
        total_position_count=int(position.shape[0]),
        implicit_orientation_count=int(rotation.shape[0]),
        elapsed_seconds=float(time.perf_counter() - started),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device_index)),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device_index)),
    )


__all__ = [
    "CHILD_GEOMETRY_CONTRACT", "CHILD_SCORE_SEMANTICS", "FUSION_SEMANTICS",
    "MODES", "MULTI_STREAMING_SEMANTICS", "RankedPositionLayout",
    "RankedSupportLayout", "StreamingHierarchyPositionResult",
    "StreamingHierarchyLayoutResult", "exact_child_member_rectangle_obbs",
    "query_child_layout", "score_support_layout_numpy",
    "stream_hierarchy_position_topk_gpu", "stream_hierarchy_support_layout_topk_gpu",
]
