"""GPU streaming parent-token layout guide over a factored global pose domain.

The mathematical score is intentionally the same coarse, pose-free score as
``parent_support_layout_guide.score_parent_support_layout_guide``.  This
module changes only execution: position/orientation factors are evaluated in
bounded chunks and only a deterministic Top-K accumulator is retained.  It
never materialises either the Cartesian poses or the full score vector.

The public entry point accepts no query pose, label, correspondence, renderer,
or RGB image.  Candidate identity is the pair ``(position row, orientation
row)`` and its total order is exactly ``(-score, position row, orientation
row)`` independent of chunk or orientation evaluation order.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import torch

from .parent_support_layout_guide import (
    NORMAL_CONTRACT,
    ParentLayoutCamera,
    SCORE_SEMANTICS,
    TOKEN_FOOTPRINT_PHASE,
    _parent_rectangles,
    _query_parent_layout,
)
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval


STREAMING_SEMANTICS = (
    "global_factor_domain_gpu_chunked_exact_parent_layout_score_"
    "stable_topk_no_full_score_materialization_v1"
)


@dataclass(frozen=True)
class StreamingParentLayoutGuideResult:
    """Deterministically ranked coarse-guide factor pairs and diagnostics."""

    top_scores: np.ndarray
    top_position_rows: np.ndarray
    top_orientation_rows: np.ndarray
    top_visible_parent_counts: np.ndarray
    top_front_facing_parent_counts: np.ndarray
    top_positive_depth_parent_counts: np.ndarray
    top_center_in_image_parent_counts: np.ndarray
    top_projected_token_footprint_mass: np.ndarray
    top_sqrt_overlap_mass: np.ndarray
    selected_query_parent_ids: np.ndarray
    selected_query_parent_probability_mass: np.ndarray
    complete_query_parent_probability_mass: float
    selected_query_parent_probability_mass_total: float
    total_factor_pair_count: int
    position_chunk_size: int
    elapsed_seconds: float
    scoring_seconds: float
    merge_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    torch_dtype: str

    def __post_init__(self) -> None:
        score = np.asarray(self.top_scores, dtype=np.float64).reshape(-1)
        position = np.asarray(self.top_position_rows, dtype=np.int64).reshape(-1)
        orientation = np.asarray(self.top_orientation_rows, dtype=np.int64).reshape(-1)
        diagnostic_names = (
            "top_visible_parent_counts",
            "top_front_facing_parent_counts",
            "top_positive_depth_parent_counts",
            "top_center_in_image_parent_counts",
            "top_projected_token_footprint_mass",
            "top_sqrt_overlap_mass",
        )
        if (
            score.size == 0
            or position.shape != score.shape
            or orientation.shape != score.shape
            or any(np.asarray(getattr(self, name)).shape != score.shape for name in diagnostic_names)
            or np.any(~np.isfinite(score))
            or np.any((score < 0.0) | (score > 1.0 + 1.0e-6))
            or np.unique(np.stack([position, orientation], axis=1), axis=0).shape[0]
            != score.size
        ):
            raise ValueError("streaming parent layout result arrays differ")
        expected = np.lexsort((orientation, position, -score))
        if not np.array_equal(expected, np.arange(score.size)):
            raise ValueError("streaming parent layout result is not stably ranked")


@dataclass(frozen=True)
class _PreparedGuide:
    parent_centers: torch.Tensor
    parent_corners: torch.Tensor
    parent_normals: torch.Tensor
    query_integral: torch.Tensor
    query_mass: float
    parent_ids: np.ndarray
    parent_probability_mass: np.ndarray
    token_height: int
    token_width: int


def _torch_dtype(name: str) -> torch.dtype:
    if str(name) == "float32":
        return torch.float32
    if str(name) == "float64":
        return torch.float64
    raise ValueError("streaming guide torch dtype must be float32 or float64")


def _prepare(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_query_parents: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreparedGuide:
    if retrieval.physical_map_sha256 != physical.content_sha256:
        raise ValueError("query retrieval and physical map lineage differ")
    selected, parent_mass, integral, query_mass = _query_parent_layout(
        retrieval, physical, maximum_query_parents=int(maximum_query_parents),
    )
    centers, corners, normals, parent_ids = _parent_rectangles(physical, selected)
    return _PreparedGuide(
        parent_centers=torch.as_tensor(centers, dtype=dtype, device=device),
        parent_corners=torch.as_tensor(corners, dtype=dtype, device=device),
        parent_normals=torch.as_tensor(normals, dtype=dtype, device=device),
        query_integral=torch.as_tensor(integral, dtype=dtype, device=device),
        query_mass=float(query_mass),
        parent_ids=np.asarray(parent_ids, dtype=np.int64),
        parent_probability_mass=np.asarray(parent_mass, dtype=np.float64),
        token_height=int(retrieval.metadata["token_height"]),
        token_width=int(retrieval.metadata["token_width"]),
    )


def _project(points: torch.Tensor, camera: ParentLayoutCamera) -> torch.Tensor:
    depth = points[..., 2]
    nan = torch.full_like(depth, float("nan"))
    safe = torch.where(torch.abs(depth) > 1.0e-12, depth, nan)
    x = points[..., 0] / safe
    y = points[..., 1] / safe
    fx, fy, cx, cy, radial = camera.fx_fy_cx_cy_k1
    scale = 1.0 + float(radial) * (x * x + y * y)
    return torch.stack((float(fx) * x * scale + float(cx),
                        float(fy) * y * scale + float(cy)), dim=-1)


def _score_position_orientation_block(
    positions_world: torch.Tensor,
    rotations_w2c: torch.Tensor,
    prepared: _PreparedGuide,
    camera: ParentLayoutCamera,
    *,
    minimum_depth_m: float,
    minimum_front_incidence: float,
    return_diagnostics: bool,
) -> dict[str, torch.Tensor]:
    """Score one dense P x O block; factor rows remain separate."""

    # Rotating each fixed physical point once per orientation avoids a full
    # candidate-parent matrix multiplication.  Camera coordinates are then
    # R*X - R*C, preserving the CPU expression R*(X-C).
    rotated_candidate = torch.einsum("oij,pj->opi", rotations_w2c, positions_world)
    rotated_center = torch.einsum(
        "oij,nj->oni", rotations_w2c, prepared.parent_centers,
    )
    center_camera = (
        rotated_center[None, :, :, :] - rotated_candidate.permute(1, 0, 2)[:, :, None, :]
    )
    center_xy = _project(center_camera, camera)
    finite = torch.all(torch.isfinite(center_xy), dim=-1)
    low = center_xy.clone()
    high = center_xy.clone()
    positive_depth = center_camera[..., 2] > float(minimum_depth_m)

    # Stream the support corners to bound peak memory.  Parent rectangles use
    # four corners; exact child-member AABBs use eight.
    rotated_corners = torch.einsum(
        "oij,nkj->onki", rotations_w2c, prepared.parent_corners,
    )
    candidate_term = rotated_candidate.permute(1, 0, 2)[:, :, None, :]
    for corner_index in range(int(prepared.parent_corners.shape[1])):
        corner_camera = rotated_corners[None, :, :, corner_index, :] - candidate_term
        corner_xy = _project(corner_camera, camera)
        finite &= torch.all(torch.isfinite(corner_xy), dim=-1)
        low = torch.minimum(low, corner_xy)
        high = torch.maximum(high, corner_xy)
        positive_depth &= corner_camera[..., 2] > float(minimum_depth_m)

    view = positions_world[:, None, :] - prepared.parent_centers[None, :, :]
    view = view / torch.clamp(torch.linalg.norm(view, dim=2, keepdim=True), min=1.0e-12)
    incidence = torch.sum(view * prepared.parent_normals[None, :, :], dim=2)
    front_position = torch.abs(incidence) >= float(minimum_front_incidence)
    front = front_position[:, None, :].expand_as(positive_depth)

    scale = torch.as_tensor(
        [
            prepared.token_width / float(camera.width),
            prepared.token_height / float(camera.height),
        ],
        dtype=low.dtype,
        device=low.device,
    )
    safe_low = torch.where(
        finite[..., None], torch.clamp(low * scale, -1.0e12, 1.0e12),
        torch.zeros_like(low),
    )
    safe_high = torch.where(
        finite[..., None], torch.clamp(high * scale, -1.0e12, 1.0e12),
        torch.zeros_like(high),
    )
    x0 = torch.clamp(torch.floor(safe_low[..., 0]).to(torch.int64), 0, prepared.token_width)
    y0 = torch.clamp(torch.floor(safe_low[..., 1]).to(torch.int64), 0, prepared.token_height)
    x1 = torch.clamp(torch.ceil(safe_high[..., 0]).to(torch.int64), 0, prepared.token_width)
    y1 = torch.clamp(torch.ceil(safe_high[..., 1]).to(torch.int64), 0, prepared.token_height)
    nonempty = (x1 > x0) & (y1 > y0)
    visible = front & positive_depth & finite & nonempty

    area = ((x1 - x0) * (y1 - y0)).to(low.dtype)
    area = torch.where(visible, area, torch.zeros_like(area))
    parent_index = torch.arange(
        prepared.parent_centers.shape[0], dtype=torch.int64, device=low.device,
    ).view(1, 1, -1)
    integral = prepared.query_integral
    overlap = (
        integral[parent_index, y1, x1]
        - integral[parent_index, y0, x1]
        - integral[parent_index, y1, x0]
        + integral[parent_index, y0, x0]
    )
    # Integral-image inclusion/exclusion is mathematically non-negative, but
    # the four floating-point operations can leave a tiny negative residual.
    # Project onto the closed non-negative mass domain before aggregation so
    # artifact validity does not depend on cancellation at machine epsilon.
    overlap = torch.where(visible, torch.clamp(overlap, min=0.0), torch.zeros_like(overlap))
    map_mass = torch.sum(area, dim=2)
    overlap_mass = torch.sum(overlap, dim=2)
    denominator = torch.sqrt(map_mass * float(prepared.query_mass))
    score = torch.where(
        denominator > 1.0e-12,
        overlap_mass / denominator,
        torch.zeros_like(overlap_mass),
    ).clamp_(0.0, 1.0)
    result = {"score": score}
    if return_diagnostics:
        center_in_image = (
            (center_xy[..., 0] >= 0.0)
            & (center_xy[..., 0] < camera.width)
            & (center_xy[..., 1] >= 0.0)
            & (center_xy[..., 1] < camera.height)
        )
        result.update({
            "visible": torch.sum(visible, dim=2).to(torch.int16),
            "front": torch.sum(front, dim=2).to(torch.int16),
            "depth": torch.sum(positive_depth, dim=2).to(torch.int16),
            "center_image": torch.sum(visible & center_in_image, dim=2).to(torch.int16),
            "footprint": map_mass,
            "overlap": overlap_mass,
        })
    return result


def _score_selected_factor_pairs(
    positions_world: torch.Tensor,
    rotations_w2c: torch.Tensor,
    prepared: _PreparedGuide,
    camera: ParentLayoutCamera,
    *,
    minimum_depth_m: float,
    minimum_front_incidence: float,
) -> dict[str, torch.Tensor]:
    """Score aligned position/rotation rows without forming a Cartesian set."""

    center_delta = prepared.parent_centers[None] - positions_world[:, None]
    center_camera = torch.einsum("bij,bpj->bpi", rotations_w2c, center_delta)
    corner_delta = prepared.parent_corners[None] - positions_world[:, None, None]
    corner_camera = torch.einsum("bij,bpkj->bpki", rotations_w2c, corner_delta)
    center_xy = _project(center_camera, camera)
    corner_xy = _project(corner_camera, camera)
    all_xy = torch.cat((center_xy[:, :, None, :], corner_xy), dim=2)
    finite = torch.all(torch.all(torch.isfinite(all_xy), dim=3), dim=2)
    low = torch.amin(all_xy, dim=2)
    high = torch.amax(all_xy, dim=2)
    positive_depth = (
        (center_camera[..., 2] > float(minimum_depth_m))
        & torch.all(corner_camera[..., 2] > float(minimum_depth_m), dim=2)
    )
    view = positions_world[:, None] - prepared.parent_centers[None]
    view = view / torch.clamp(torch.linalg.norm(view, dim=2, keepdim=True), min=1.0e-12)
    incidence = torch.sum(view * prepared.parent_normals[None], dim=2)
    front = torch.abs(incidence) >= float(minimum_front_incidence)
    scale = torch.as_tensor(
        [
            prepared.token_width / float(camera.width),
            prepared.token_height / float(camera.height),
        ], dtype=low.dtype, device=low.device,
    )
    safe_low = torch.where(
        finite[..., None], torch.clamp(low * scale, -1.0e12, 1.0e12),
        torch.zeros_like(low),
    )
    safe_high = torch.where(
        finite[..., None], torch.clamp(high * scale, -1.0e12, 1.0e12),
        torch.zeros_like(high),
    )
    x0 = torch.clamp(torch.floor(safe_low[..., 0]).to(torch.int64), 0, prepared.token_width)
    y0 = torch.clamp(torch.floor(safe_low[..., 1]).to(torch.int64), 0, prepared.token_height)
    x1 = torch.clamp(torch.ceil(safe_high[..., 0]).to(torch.int64), 0, prepared.token_width)
    y1 = torch.clamp(torch.ceil(safe_high[..., 1]).to(torch.int64), 0, prepared.token_height)
    nonempty = (x1 > x0) & (y1 > y0)
    visible = front & positive_depth & finite & nonempty
    area = torch.where(
        visible, ((x1 - x0) * (y1 - y0)).to(low.dtype),
        torch.zeros_like(low[..., 0]),
    )
    parent_index = torch.arange(
        prepared.parent_centers.shape[0], dtype=torch.int64, device=low.device,
    ).view(1, -1)
    integral = prepared.query_integral
    overlap = (
        integral[parent_index, y1, x1]
        - integral[parent_index, y0, x1]
        - integral[parent_index, y1, x0]
        + integral[parent_index, y0, x0]
    )
    overlap = torch.where(visible, torch.clamp(overlap, min=0.0), torch.zeros_like(overlap))
    map_mass = torch.sum(area, dim=1)
    overlap_mass = torch.sum(overlap, dim=1)
    denominator = torch.sqrt(map_mass * float(prepared.query_mass))
    score = torch.where(
        denominator > 1.0e-12,
        overlap_mass / denominator,
        torch.zeros_like(overlap_mass),
    ).clamp_(0.0, 1.0)
    center_in_image = (
        (center_xy[..., 0] >= 0.0)
        & (center_xy[..., 0] < camera.width)
        & (center_xy[..., 1] >= 0.0)
        & (center_xy[..., 1] < camera.height)
    )
    return {
        "score": score,
        "visible": torch.sum(visible, dim=1).to(torch.int16),
        "front": torch.sum(front, dim=1).to(torch.int16),
        "depth": torch.sum(positive_depth, dim=1).to(torch.int16),
        "center_image": torch.sum(visible & center_in_image, dim=1).to(torch.int16),
        "footprint": map_mass,
        "overlap": overlap_mass,
    }


def _stable_topk(
    score: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Return exact lexicographic Top-K without relying on unstable GPU ties."""

    value = np.asarray(score).reshape(-1)
    pos = np.asarray(position, dtype=np.int64).reshape(-1)
    ori = np.asarray(orientation, dtype=np.int64).reshape(-1)
    if value.shape != pos.shape or value.shape != ori.shape or value.size == 0:
        raise ValueError("stable Top-K arrays differ")
    keep = min(int(topk), value.size)
    if keep <= 0:
        raise ValueError("stable Top-K budget must be positive")
    if keep == value.size:
        candidates = np.arange(value.size, dtype=np.int64)
    else:
        # Argpartition discovers only the score threshold.  All ties at that
        # threshold are resolved below by the canonical factor-row keys.
        threshold = np.partition(value, value.size - keep)[value.size - keep]
        above = np.flatnonzero(value > threshold)
        equal = np.flatnonzero(value == threshold)
        equal_order = np.lexsort((ori[equal], pos[equal]))
        candidates = np.concatenate([above, equal[equal_order[: keep - above.size]]])
    order = np.lexsort((ori[candidates], pos[candidates], -value[candidates]))
    return candidates[order[:keep]]


def stream_parent_support_layout_topk_gpu(
    position_centers_world: np.ndarray,
    orientation_rotations_w2c: np.ndarray,
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    camera: ParentLayoutCamera,
    *,
    maximum_query_parents: int = 32,
    topk: int = 4096,
    position_chunk_size: int = 512,
    device: str | torch.device = "cuda:0",
    torch_dtype: Literal["float32", "float64"] = "float32",
    minimum_depth_m: float = 0.05,
    minimum_front_incidence: float = 0.02,
    reverse_position_chunks: bool = False,
    reverse_orientation_evaluation: bool = False,
) -> StreamingParentLayoutGuideResult:
    """Stream a global P x O domain through the coarse guide on CUDA."""

    position = np.asarray(position_centers_world, dtype=np.float64)
    rotation = np.asarray(orientation_rotations_w2c, dtype=np.float64)
    if (
        position.ndim != 2 or position.shape[1:] != (3,)
        or rotation.ndim != 3 or rotation.shape[1:] != (3, 3)
        or position.shape[0] <= 0 or rotation.shape[0] <= 0
        or np.any(~np.isfinite(position)) or np.any(~np.isfinite(rotation))
        or int(topk) <= 0 or int(position_chunk_size) <= 0
        or int(maximum_query_parents) <= 0
        or float(minimum_depth_m) <= 0.0
    ):
        raise ValueError("streaming parent layout factors/configuration differ")
    device_value = torch.device(device)
    if device_value.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("streaming parent layout guide requires an available CUDA device")
    dtype = _torch_dtype(torch_dtype)
    started = time.perf_counter()
    # Torch 1.13's memory-stat bindings require the device context to have
    # been initialised and accept an integer index more reliably than a
    # ``torch.device`` object.
    device_index = (
        int(device_value.index) if device_value.index is not None
        else int(torch.cuda.current_device())
    )
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    prepared = _prepare(
        retrieval, physical, maximum_query_parents=int(maximum_query_parents),
        device=device_value, dtype=dtype,
    )
    rotation_order = np.arange(rotation.shape[0], dtype=np.int64)
    if bool(reverse_orientation_evaluation):
        rotation_order = rotation_order[::-1].copy()
    rotation_gpu = torch.as_tensor(
        rotation[rotation_order], dtype=dtype, device=device_value,
    )
    bounds = [
        (start, min(start + int(position_chunk_size), position.shape[0]))
        for start in range(0, position.shape[0], int(position_chunk_size))
    ]
    if bool(reverse_position_chunks):
        bounds.reverse()
    best_score = np.empty((0,), dtype=np.float64)
    best_position = np.empty((0,), dtype=np.int64)
    best_orientation = np.empty((0,), dtype=np.int64)
    scoring_seconds = 0.0
    merge_seconds = 0.0
    with torch.inference_mode():
        for start, end in bounds:
            tick = time.perf_counter()
            position_gpu = torch.as_tensor(position[start:end], dtype=dtype, device=device_value)
            block = _score_position_orientation_block(
                position_gpu, rotation_gpu, prepared, camera,
                minimum_depth_m=float(minimum_depth_m),
                minimum_front_incidence=float(minimum_front_incidence),
                return_diagnostics=False,
            )
            block_score_matrix = block["score"]
            # Restore canonical orientation-row order before assigning factor
            # identities.  This makes execution order a pure implementation
            # detail and not part of the score/tie contract.
            inverse = np.argsort(rotation_order)
            block_score = (
                block_score_matrix[:, torch.as_tensor(inverse, device=device_value)]
                .reshape(-1).to(torch.float64).cpu().numpy()
            )
            scoring_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            block_position = np.repeat(
                np.arange(start, end, dtype=np.int64), rotation.shape[0],
            )
            block_orientation = np.tile(
                np.arange(rotation.shape[0], dtype=np.int64), end - start,
            )
            selected = _stable_topk(
                block_score, block_position, block_orientation, int(topk),
            )
            merged_score = np.concatenate([best_score, block_score[selected]])
            merged_position = np.concatenate([best_position, block_position[selected]])
            merged_orientation = np.concatenate([
                best_orientation, block_orientation[selected],
            ])
            merged = _stable_topk(
                merged_score, merged_position, merged_orientation, int(topk),
            )
            best_score = merged_score[merged]
            best_position = merged_position[merged]
            best_orientation = merged_orientation[merged]
            merge_seconds += time.perf_counter() - tick

        # Recompute only the selected aligned pairs for diagnostics.  This is
        # one bounded K-row operation and cannot allocate an unselected
        # Cartesian factor pair.
        values = _score_selected_factor_pairs(
            torch.as_tensor(position[best_position], dtype=dtype, device=device_value),
            torch.as_tensor(rotation[best_orientation], dtype=dtype, device=device_value),
            prepared, camera,
            minimum_depth_m=float(minimum_depth_m),
            minimum_front_incidence=float(minimum_front_incidence),
        )
        recomputed_score = values["score"].to(torch.float64).cpu().numpy()
        diagnostic = {
            key: values[key].cpu().numpy()
            for key in (
                "visible", "front", "depth", "center_image", "footprint", "overlap",
            )
        }
    if not np.array_equal(recomputed_score, best_score):
        # A change of batch shape is not allowed to change ranking scores.
        maximum_delta = float(np.max(np.abs(recomputed_score - best_score)))
        if maximum_delta > (2.0e-6 if dtype == torch.float32 else 1.0e-12):
            raise RuntimeError(
                "streaming guide score changes across block shapes: "
                f"max_delta={maximum_delta}"
            )
        best_score = recomputed_score
        final_order = np.lexsort((best_orientation, best_position, -best_score))
        best_score = best_score[final_order]
        best_position = best_position[final_order]
        best_orientation = best_orientation[final_order]
        diagnostic = {key: value[final_order] for key, value in diagnostic.items()}
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - started
    return StreamingParentLayoutGuideResult(
        top_scores=best_score,
        top_position_rows=best_position,
        top_orientation_rows=best_orientation,
        top_visible_parent_counts=diagnostic["visible"],
        top_front_facing_parent_counts=diagnostic["front"],
        top_positive_depth_parent_counts=diagnostic["depth"],
        top_center_in_image_parent_counts=diagnostic["center_image"],
        top_projected_token_footprint_mass=diagnostic["footprint"],
        top_sqrt_overlap_mass=diagnostic["overlap"],
        selected_query_parent_ids=prepared.parent_ids,
        selected_query_parent_probability_mass=prepared.parent_probability_mass,
        complete_query_parent_probability_mass=prepared.query_mass,
        selected_query_parent_probability_mass_total=float(
            np.sum(prepared.parent_probability_mass)
        ),
        total_factor_pair_count=int(position.shape[0] * rotation.shape[0]),
        position_chunk_size=int(position_chunk_size),
        elapsed_seconds=float(elapsed),
        scoring_seconds=float(scoring_seconds),
        merge_seconds=float(merge_seconds),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device_index)),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device_index)),
        torch_dtype=str(torch_dtype),
    )


__all__ = [
    "NORMAL_CONTRACT",
    "SCORE_SEMANTICS",
    "STREAMING_SEMANTICS",
    "TOKEN_FOOTPRINT_PHASE",
    "StreamingParentLayoutGuideResult",
    "stream_parent_support_layout_topk_gpu",
]
