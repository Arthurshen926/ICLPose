"""MATCHA-style coarse-to-fine matching for rendered feature maps.

The implementation is deliberately lightweight but follows MATCHA's core
geometry: coarse grid dual-softmax matching, repeated-cell deduplication, then
local fine refinement instead of treating a coarse patch center as a precise
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.rendered_keypoint_matching import (
    KeypointFeatureMatch,
    bilinear_sample_feature_map,
    refine_render_keypoint_matches_by_local_correlation,
)


@dataclass(frozen=True)
class CoarseGrid:
    xy: np.ndarray
    descriptors: np.ndarray
    height: int
    width: int

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if descriptors.ndim != 2:
            raise ValueError("descriptors must have shape (N, C)")
        if descriptors.shape[0] != xy.shape[0]:
            raise ValueError("xy and descriptors must contain the same number of cells")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError("height and width must be positive")
        if xy.shape[0] != int(self.height) * int(self.width):
            raise ValueError("xy count must equal height * width")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "descriptors", descriptors)


def feature_map_to_coarse_grid(feature_map: np.ndarray, *, image_width: int, image_height: int) -> CoarseGrid:
    """Convert a CHW feature map into one descriptor per patch/cell."""

    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) * float(image_width) / float(width)
    ys = (np.arange(height, dtype=np.float64) + 0.5) * float(image_height) / float(height)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    descriptors = fmap.reshape(channels, height * width).T.astype(np.float32, copy=False)
    return CoarseGrid(xy=xy, descriptors=descriptors, height=height, width=width)


def _dual_softmax_confidence(scores: np.ndarray, logit_scale: float) -> np.ndarray:
    logits = np.asarray(scores, dtype=np.float32) * float(logit_scale)
    row = logits - np.max(logits, axis=1, keepdims=True)
    col = logits - np.max(logits, axis=0, keepdims=True)
    row_prob = np.exp(row)
    row_prob /= np.maximum(np.sum(row_prob, axis=1, keepdims=True), 1e-12)
    col_prob = np.exp(col)
    col_prob /= np.maximum(np.sum(col_prob, axis=0, keepdims=True), 1e-12)
    return row_prob * col_prob


def _top2_margin(scores: np.ndarray, row: int, col: int) -> tuple[float, float]:
    if scores.shape[1] <= 1:
        return 0.0, 0.0
    values = scores[int(row)]
    best = float(values[int(col)])
    copied = values.copy()
    copied[int(col)] = -np.inf
    second = float(np.max(copied))
    best_distance = max(0.0, 1.0 - best)
    second_distance = max(1e-6, 1.0 - second)
    return float(best - second), float(best_distance / second_distance)


def cross_attention_enhance_feature_maps(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    alpha: float = 0.25,
    logit_scale: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one residual bidirectional query-render attention layer.

    This is intentionally parameter-free and candidate-local: it tests whether
    MATCHA-style cross context helps before introducing a trainable large
    network. The output keeps the original CHW shapes and L2-normalizes cells.
    """

    query = np.asarray(query_feature_map, dtype=np.float32)
    render = np.asarray(render_feature_map, dtype=np.float32)
    if query.ndim != 3 or render.ndim != 3:
        raise ValueError("feature maps must have shape (C, H, W)")
    if query.shape[0] != render.shape[0]:
        raise ValueError("query and render feature dimensions must match")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")
    channels, qh, qw = query.shape
    _, rh, rw = render.shape
    qrows = query.reshape(channels, qh * qw).T.astype(np.float32, copy=False)
    rrows = render.reshape(channels, rh * rw).T.astype(np.float32, copy=False)
    qnorm, qvalid = normalize_rows(qrows)
    rnorm, rvalid = normalize_rows(rrows)
    if not np.any(qvalid) or not np.any(rvalid):
        return query.copy(), render.copy()
    q_context = np.zeros_like(qnorm)
    r_context = np.zeros_like(rnorm)
    q_valid_rows = np.flatnonzero(qvalid)
    r_valid_rows = np.flatnonzero(rvalid)
    valid_scores = (qnorm[q_valid_rows] @ rnorm[r_valid_rows].T).astype(np.float32, copy=False) * float(logit_scale)
    q_logits = valid_scores - np.max(valid_scores, axis=1, keepdims=True)
    q_weights = np.exp(q_logits)
    q_weights /= np.maximum(np.sum(q_weights, axis=1, keepdims=True), 1e-12)
    r_logits = valid_scores.T - np.max(valid_scores.T, axis=1, keepdims=True)
    r_weights = np.exp(r_logits)
    r_weights /= np.maximum(np.sum(r_weights, axis=1, keepdims=True), 1e-12)
    q_context[q_valid_rows] = q_weights @ rnorm[r_valid_rows]
    r_context[r_valid_rows] = r_weights @ qnorm[q_valid_rows]
    q_enhanced = (1.0 - float(alpha)) * qnorm + float(alpha) * q_context
    r_enhanced = (1.0 - float(alpha)) * rnorm + float(alpha) * r_context
    q_enhanced, _ = normalize_rows(q_enhanced.astype(np.float32, copy=False))
    r_enhanced, _ = normalize_rows(r_enhanced.astype(np.float32, copy=False))
    q_enhanced[~qvalid] = 0.0
    r_enhanced[~rvalid] = 0.0
    return (
        q_enhanced.T.reshape(channels, qh, qw).astype(np.float32, copy=False),
        r_enhanced.T.reshape(channels, rh, rw).astype(np.float32, copy=False),
    )


def deduplicate_repeated_correspondences(
    matches: Sequence[KeypointFeatureMatch],
    *,
    query_cell_size_px: float,
    render_cell_size_px: float,
) -> list[KeypointFeatureMatch]:
    """Keep one match for each coarse query/render cell pair.

    This mirrors MATCHA's repeated-correspondence cleanup: repeated grid
    correspondences are harmful because they overweight one patch during
    training and RANSAC.
    """

    best_by_pair: dict[tuple[int, int, int, int], KeypointFeatureMatch] = {}
    q_size = max(float(query_cell_size_px), 1e-6)
    r_size = max(float(render_cell_size_px), 1e-6)
    for match in matches:
        key = (
            int(np.floor(float(match.query_xy[0]) / q_size)),
            int(np.floor(float(match.query_xy[1]) / q_size)),
            int(np.floor(float(match.render_xy[0]) / r_size)),
            int(np.floor(float(match.render_xy[1]) / r_size)),
        )
        current = best_by_pair.get(key)
        current_score = -np.inf if current is None else float(current.dual_softmax_confidence or current.similarity)
        score = float(match.dual_softmax_confidence or match.similarity)
        if current is None or score > current_score:
            best_by_pair[key] = match
    output = list(best_by_pair.values())
    output.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    return output


def _offset_label_to_xy(
    label: int,
    cell_index: int,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
) -> np.ndarray | None:
    label = int(label)
    if label < 0 or label >= int(offset_bins) * int(offset_bins):
        return None
    cell_index = int(cell_index)
    if cell_index < 0 or cell_index >= int(grid_width) * int(grid_height):
        return None
    col = cell_index % int(grid_width)
    row = cell_index // int(grid_width)
    bin_x = label % int(offset_bins)
    bin_y = label // int(offset_bins)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    x = (float(col) + (float(bin_x) + 0.5) / float(offset_bins)) * cell_w
    y = (float(row) + (float(bin_y) + 0.5) / float(offset_bins)) * cell_h
    return np.asarray([x, y], dtype=np.float64)


def _offset_logits_to_xy(
    row_logits: np.ndarray,
    cell_index: int,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
    coordinate_mode: str = "argmax",
) -> np.ndarray | None:
    logits = np.asarray(row_logits, dtype=np.float32).reshape(-1)
    bins = int(offset_bins)
    mode = str(coordinate_mode)
    if mode not in {"argmax", "softargmax"}:
        raise ValueError("coordinate_mode must be 'argmax' or 'softargmax'")
    label_count = 65 if int(logits.shape[0]) >= bins * bins + 1 else bins * bins
    label = int(np.argmax(logits[:label_count]))
    if mode == "argmax":
        return _offset_label_to_xy(
            label,
            int(cell_index),
            image_width=int(image_width),
            image_height=int(image_height),
            grid_width=int(grid_width),
            grid_height=int(grid_height),
            offset_bins=bins,
        )
    if label == bins * bins:
        return None
    cell = int(cell_index)
    if cell < 0 or cell >= int(grid_width) * int(grid_height):
        return None
    scores = logits[: bins * bins].astype(np.float64)
    scores = scores - float(np.max(scores))
    probs = np.exp(scores)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    labels = np.arange(bins * bins, dtype=np.float64)
    bin_x = np.remainder(labels, float(bins)) + 0.5
    bin_y = np.floor(labels / float(bins)) + 0.5
    col = cell % int(grid_width)
    row = cell // int(grid_width)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    x = (float(col) + float(np.sum(probs * bin_x)) / float(bins)) * cell_w
    y = (float(row) + float(np.sum(probs * bin_y)) / float(bins)) * cell_h
    return np.asarray([x, y], dtype=np.float64)


def _offset_logits_confidence_sigma(
    row_logits: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
) -> tuple[float, float]:
    logits = np.asarray(row_logits, dtype=np.float32).reshape(-1)
    bins = int(offset_bins)
    scores = logits[: bins * bins].astype(np.float64)
    scores = scores - float(np.max(scores))
    probs = np.exp(scores)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    labels = np.arange(bins * bins, dtype=np.float64)
    bin_x = np.remainder(labels, float(bins)) + 0.5
    bin_y = np.floor(labels / float(bins)) + 0.5
    expected_x = float(np.sum(probs * bin_x))
    expected_y = float(np.sum(probs * bin_y))
    variance = float(np.sum(probs * ((bin_x - expected_x) ** 2 + (bin_y - expected_y) ** 2)))
    sigma_bins = float(np.sqrt(max(variance, 0.0)))
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    bin_px = 0.5 * (cell_w + cell_h) / float(bins)
    return float(np.max(probs)), float(max(sigma_bins * bin_px, 1e-6))


def apply_offset_logits_to_matches(
    matches: Sequence[KeypointFeatureMatch],
    query_offset_logits: np.ndarray | None,
    render_offset_logits: np.ndarray | None,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    offset_bins: int = 8,
) -> list[KeypointFeatureMatch]:
    """Move coarse cell-center matches to predicted 8x8 offset-bin centers.

    Channel 64 is treated as the MATCHA dustbin/no-keypoint class and filters
    the match for whichever side is active. Passing one side as None keeps that
    side at its current measurement, which is useful for query/render side
    ablations. Only geometry is changed; descriptor similarity/confidence are
    preserved.
    """

    if query_offset_logits is None and render_offset_logits is None:
        return list(matches)
    qlogits = None if query_offset_logits is None else np.asarray(query_offset_logits, dtype=np.float32)
    rlogits = None if render_offset_logits is None else np.asarray(render_offset_logits, dtype=np.float32)
    if qlogits is not None and (qlogits.ndim != 3 or qlogits.shape[0] < 65):
        raise ValueError("query_offset_logits must have shape (65, H, W) or larger in channel 0")
    if rlogits is not None and (rlogits.ndim != 3 or rlogits.shape[0] < 65):
        raise ValueError("render_offset_logits must have shape (65, H, W) or larger in channel 0")
    qh, qw = (0, 0) if qlogits is None else (int(qlogits.shape[1]), int(qlogits.shape[2]))
    rh, rw = (0, 0) if rlogits is None else (int(rlogits.shape[1]), int(rlogits.shape[2]))
    refined: list[KeypointFeatureMatch] = []
    for match in matches:
        qidx = int(match.query_index)
        ridx = int(match.render_index)
        if qlogits is not None and not (0 <= qidx < qh * qw):
            continue
        if rlogits is not None and not (0 <= ridx < rh * rw):
            continue
        qxy = np.asarray(match.query_xy, dtype=np.float64).reshape(2)
        rxy = np.asarray(match.render_xy, dtype=np.float64).reshape(2)
        if qlogits is not None:
            qrow, qcol = divmod(qidx, qw)
            qlabel = int(np.argmax(qlogits[:65, qrow, qcol]))
            qxy_refined = _offset_label_to_xy(
                qlabel,
                qidx,
                image_width=int(query_image_width),
                image_height=int(query_image_height),
                grid_width=qw,
                grid_height=qh,
                offset_bins=int(offset_bins),
            )
            if qxy_refined is None:
                continue
            qxy = qxy_refined
        if rlogits is not None:
            rrow, rcol = divmod(ridx, rw)
            rlabel = int(np.argmax(rlogits[:65, rrow, rcol]))
            rxy_refined = _offset_label_to_xy(
                rlabel,
                ridx,
                image_width=int(render_image_width),
                image_height=int(render_image_height),
                grid_width=rw,
                grid_height=rh,
                offset_bins=int(offset_bins),
            )
            if rxy_refined is None:
                continue
            rxy = rxy_refined
        refined.append(replace(match, query_xy=qxy, render_xy=rxy))
    return refined


def apply_pair_fine_logits_to_matches(
    matches: Sequence[KeypointFeatureMatch],
    pair_fine_logits: np.ndarray,
    *,
    render_image_width: int | None = None,
    render_image_height: int | None = None,
    render_grid_width: int | None = None,
    render_grid_height: int | None = None,
    query_image_width: int | None = None,
    query_image_height: int | None = None,
    query_grid_width: int | None = None,
    query_grid_height: int | None = None,
    target_side: str = "render",
    offset_bins: int = 8,
    coordinate_mode: str = "argmax",
) -> list[KeypointFeatureMatch]:
    """Move one side of a match using pair-conditioned 8x8 offset logits.

    This is the lightweight learned fine matcher path: unlike the per-cell
    offset head, the prediction depends on both query and render descriptors.
    64-bin logits are interpreted as pure spatial coordinate classification,
    matching MATCHA's fine matcher. If a legacy 65+ bin tensor is supplied,
    label 64 is treated as a dustbin and filters the match. The default keeps
    historical behavior and refines the render measurement; query-side
    refinement is useful when supervision starts from render cell centers and
    the informative sub-cell target is the projected query coordinate.
    """

    values = list(matches)
    logits = np.asarray(pair_fine_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] != len(values) or logits.shape[1] < 64:
        raise ValueError("pair_fine_logits must have shape (len(matches), 64) or larger")
    side = str(target_side)
    if side not in {"render", "query"}:
        raise ValueError("target_side must be 'render' or 'query'")
    mode = str(coordinate_mode)
    if mode not in {"argmax", "softargmax"}:
        raise ValueError("coordinate_mode must be 'argmax' or 'softargmax'")
    if side == "render":
        if render_image_width is None or render_image_height is None or render_grid_width is None or render_grid_height is None:
            raise ValueError("render image and grid dimensions are required for render-side refinement")
        image_width = int(render_image_width)
        image_height = int(render_image_height)
        grid_width = int(render_grid_width)
        grid_height = int(render_grid_height)
    else:
        if query_image_width is None or query_image_height is None or query_grid_width is None or query_grid_height is None:
            raise ValueError("query image and grid dimensions are required for query-side refinement")
        image_width = int(query_image_width)
        image_height = int(query_image_height)
        grid_width = int(query_grid_width)
        grid_height = int(query_grid_height)
    refined: list[KeypointFeatureMatch] = []
    for match, row_logits in zip(values, logits):
        xy = _offset_logits_to_xy(
            row_logits,
            int(match.render_index if side == "render" else match.query_index),
            image_width=image_width,
            image_height=image_height,
            grid_width=grid_width,
            grid_height=grid_height,
            offset_bins=int(offset_bins),
            coordinate_mode=mode,
        )
        if xy is None:
            continue
        fine_confidence, fine_sigma = _offset_logits_confidence_sigma(
            row_logits,
            image_width=image_width,
            image_height=image_height,
            grid_width=grid_width,
            grid_height=grid_height,
            offset_bins=int(offset_bins),
        )
        if side == "render":
            refined.append(
                replace(
                    match,
                    render_xy=xy.astype(np.float64, copy=False),
                    fine_offset_confidence=fine_confidence,
                    fine_offset_sigma_px=fine_sigma,
                )
            )
        else:
            refined.append(
                replace(
                    match,
                    query_xy=xy.astype(np.float64, copy=False),
                    fine_offset_confidence=fine_confidence,
                    fine_offset_sigma_px=fine_sigma,
                )
            )
    return refined


def apply_fine_logit_confidence_to_matches(
    matches: Sequence[KeypointFeatureMatch],
    fine_logits: np.ndarray,
    *,
    blend: float = 0.5,
) -> list[KeypointFeatureMatch]:
    """Blend local fine-head peak probability into match confidence."""

    values = list(matches)
    logits = np.asarray(fine_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] != len(values) or logits.shape[1] < 64:
        raise ValueError("fine_logits must have shape (len(matches), 64) or larger")
    amount = float(np.clip(float(blend), 0.0, 1.0))
    if not values or amount <= 0.0:
        return values
    spatial_logits = logits[:, :64]
    shifted = spatial_logits - np.max(spatial_logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)
    peak = np.max(probs, axis=1)
    uniform = 1.0 / 64.0
    fine_confidences = np.clip((peak - uniform) / (1.0 - uniform), 0.0, 1.0)
    updated: list[KeypointFeatureMatch] = []
    for match, fine_confidence in zip(values, fine_confidences):
        base = (
            float(match.dual_softmax_confidence)
            if match.dual_softmax_confidence is not None
            else float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
        )
        confidence = (1.0 - amount) * base + amount * float(fine_confidence)
        updated.append(replace(match, dual_softmax_confidence=float(np.clip(confidence, 0.0, 1.0))))
    updated.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    return updated


def expand_matches_with_render_local_offsets(
    matches: Sequence[KeypointFeatureMatch],
    *,
    render_image_width: int,
    render_image_height: int,
    render_grid_width: int,
    render_grid_height: int,
    cell_radius: int = 1,
    offset_bins: int = 8,
    max_candidates_per_match: int = 9,
) -> list[KeypointFeatureMatch]:
    """Expand matches into nearby render-cell candidates without moving query xy.

    The candidate location keeps the original within-cell fractional render
    offset when possible. This makes the candidate set depth-changing: each
    expanded match gets the neighboring render cell index and a render xy inside
    that cell, so later render-depth backprojection samples a different 3D
    point.
    """

    values = list(matches)
    if not values:
        return []
    grid_w = int(render_grid_width)
    grid_h = int(render_grid_height)
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("render grid dimensions must be positive")
    radius = max(int(cell_radius), 0)
    limit = max(int(max_candidates_per_match), 1)
    bins = max(int(offset_bins), 1)
    cell_w = float(render_image_width) / float(grid_w)
    cell_h = float(render_image_height) / float(grid_h)
    expanded: list[KeypointFeatureMatch] = []
    for match in values:
        ridx = int(match.render_index)
        if ridx < 0 or ridx >= grid_w * grid_h:
            expanded.append(match)
            continue
        row, col = divmod(ridx, grid_w)
        base_render_index = int(match.base_render_index) if match.base_render_index is not None else int(ridx)
        render_xy = np.asarray(match.render_xy, dtype=np.float64).reshape(2)
        local_x = render_xy[0] / max(cell_w, 1e-12) - float(col)
        local_y = render_xy[1] / max(cell_h, 1e-12) - float(row)
        if not np.isfinite(local_x) or not np.isfinite(local_y):
            local_x = local_y = 0.5
        local_x = float(np.clip(local_x, 0.5 / float(bins), 1.0 - 0.5 / float(bins)))
        local_y = float(np.clip(local_y, 0.5 / float(bins), 1.0 - 0.5 / float(bins)))
        candidates: list[tuple[int, int, int, float]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nrow = row + dy
                ncol = col + dx
                if nrow < 0 or nrow >= grid_h or ncol < 0 or ncol >= grid_w:
                    continue
                distance2 = float(dx * dx + dy * dy)
                candidates.append((int(nrow), int(ncol), int(dy * grid_w + dx), distance2))
        candidates.sort(key=lambda item: (item[3], abs(item[2]), item[0], item[1]))
        for nrow, ncol, _flat_delta, _distance2 in candidates[:limit]:
            candidate_idx = int(nrow * grid_w + ncol)
            xy = np.asarray([(float(ncol) + local_x) * cell_w, (float(nrow) + local_y) * cell_h], dtype=np.float64)
            expanded.append(
                replace(
                    match,
                    render_index=candidate_idx,
                    render_xy=xy,
                    base_render_index=base_render_index,
                    candidate_render_index=candidate_idx,
                    candidate_id=len(expanded),
                    cell_delta_x=int(ncol - col),
                    cell_delta_y=int(nrow - row),
                )
            )
    expanded.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence or 0.0),
            float(item.similarity),
        ),
        reverse=True,
    )
    return expanded


def rescore_keypoint_matches_by_feature_similarity(
    matches: Sequence[KeypointFeatureMatch],
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
) -> list[KeypointFeatureMatch]:
    """Recompute match similarity/confidence after keypoint measurements move."""

    values = list(matches)
    if not values:
        return []
    qxy = np.stack([match.query_xy for match in values], axis=0).astype(np.float64)
    rxy = np.stack([match.render_xy for match in values], axis=0).astype(np.float64)
    qdesc, qvalid = bilinear_sample_feature_map(
        query_feature_map,
        qxy,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    rdesc, rvalid = bilinear_sample_feature_map(
        render_feature_map,
        rxy,
        image_width=int(render_image_width),
        image_height=int(render_image_height),
    )
    qnorm, qnorm_valid = normalize_rows(qdesc)
    rnorm, rnorm_valid = normalize_rows(rdesc)
    valid = qvalid & rvalid & qnorm_valid & rnorm_valid
    rescored: list[KeypointFeatureMatch] = []
    for idx, (match, is_valid) in enumerate(zip(values, valid)):
        if not bool(is_valid):
            continue
        similarity = float(np.dot(qnorm[idx], rnorm[idx]))
        similarity_conf = float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))
        base_conf = (
            float(match.dual_softmax_confidence)
            if match.dual_softmax_confidence is not None and np.isfinite(float(match.dual_softmax_confidence))
            else similarity_conf
        )
        confidence = float(np.clip(base_conf * similarity_conf, 0.0, 1.0))
        rescored.append(
            replace(
                match,
                similarity=similarity,
                ratio=float(match.ratio),
                similarity_margin=None,
                dual_softmax_confidence=confidence,
            )
        )
    rescored.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence or 0.0),
            float(item.similarity),
        ),
        reverse=True,
    )
    return rescored


def retain_topk_matches_per_query(
    matches: Sequence[KeypointFeatureMatch],
    *,
    max_per_query: int,
) -> list[KeypointFeatureMatch]:
    """Keep at most K highest-scoring render candidates for each query token."""

    limit = int(max_per_query)
    values = list(matches)
    if limit <= 0 or not values:
        return values
    buckets: dict[int, list[KeypointFeatureMatch]] = {}
    for match in values:
        buckets.setdefault(int(match.query_index), []).append(match)
    kept: list[KeypointFeatureMatch] = []
    for query_index in sorted(buckets):
        candidates = sorted(
            buckets[query_index],
            key=lambda item: (
                float(item.dual_softmax_confidence or 0.0),
                float(item.similarity),
            ),
            reverse=True,
        )
        kept.extend(candidates[:limit])
    kept.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence or 0.0),
            float(item.similarity),
        ),
        reverse=True,
    )
    return kept


def _coarse_reciprocal_rank(confidence: np.ndarray, local_q: int, local_r: int) -> int:
    value = float(confidence[int(local_q), int(local_r)])
    column = np.asarray(confidence[:, int(local_r)], dtype=np.float32)
    return int(np.count_nonzero(column > value))


def _local_window_mask_for_query(
    local_q_index: int,
    qrows: np.ndarray,
    rrows: np.ndarray,
    *,
    query_grid_width: int,
    query_grid_height: int,
    render_grid_width: int,
    render_grid_height: int,
    radius: int | None,
) -> np.ndarray:
    if radius is None:
        return np.ones((rrows.shape[0],), dtype=bool)
    r = int(radius)
    if r < 0:
        return np.ones((rrows.shape[0],), dtype=bool)
    qidx = int(qrows[int(local_q_index)])
    qrow, qcol = divmod(qidx, int(query_grid_width))
    center_col = (float(qcol) + 0.5) / float(query_grid_width) * float(render_grid_width) - 0.5
    center_row = (float(qrow) + 0.5) / float(query_grid_height) * float(render_grid_height) - 0.5
    rcols = (rrows % int(render_grid_width)).astype(np.float64)
    rrows_grid = (rrows // int(render_grid_width)).astype(np.float64)
    return (np.abs(rcols - center_col) <= float(r) + 1e-9) & (np.abs(rrows_grid - center_row) <= float(r) + 1e-9)


def matcha_coarse_topk_matches(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    k_per_query: int = 5,
    logit_scale: float = 10.0,
    min_confidence: float = 0.0,
    min_similarity: float = -1.0,
    max_matches: int | None = None,
    mutual_mode: str = "annotate",
    local_window_radius_cells: int | None = None,
    query_candidate_indices: np.ndarray | None = None,
    render_candidate_indices: np.ndarray | None = None,
) -> list[KeypointFeatureMatch]:
    """Return multiple coarse render-cell candidates for each query cell.

    This is a candidate-recall path for pose refinement. Mutual nearest-neighbor
    is not a hard requirement by default; reciprocal rank is recorded as a
    feature so downstream ranking/PnP can decide how to use it.
    """

    k = int(k_per_query)
    if k <= 0:
        raise ValueError("k_per_query must be positive")
    mode = str(mutual_mode)
    if mode not in {"none", "annotate", "filter"}:
        raise ValueError("mutual_mode must be one of: none, annotate, filter")
    query_grid = feature_map_to_coarse_grid(
        query_feature_map,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    render_grid = feature_map_to_coarse_grid(
        render_feature_map,
        image_width=int(render_image_width),
        image_height=int(render_image_height),
    )
    qdesc, qvalid = normalize_rows(query_grid.descriptors)
    rdesc, rvalid = normalize_rows(render_grid.descriptors)
    qrows = np.flatnonzero(qvalid)
    rrows = np.flatnonzero(rvalid)
    if query_candidate_indices is not None:
        qmask = np.zeros((query_grid.xy.shape[0],), dtype=bool)
        qidx = np.asarray(query_candidate_indices, dtype=np.int64).reshape(-1)
        qidx = qidx[(qidx >= 0) & (qidx < qmask.shape[0])]
        qmask[qidx] = True
        qrows = qrows[qmask[qrows]]
    if render_candidate_indices is not None:
        rmask = np.zeros((render_grid.xy.shape[0],), dtype=bool)
        ridx = np.asarray(render_candidate_indices, dtype=np.int64).reshape(-1)
        ridx = ridx[(ridx >= 0) & (ridx < rmask.shape[0])]
        rmask[ridx] = True
        rrows = rrows[rmask[rrows]]
    if qrows.size == 0 or rrows.size == 0:
        return []
    qdesc_local = qdesc[qrows]
    rdesc_local = rdesc[rrows]
    scores = qdesc_local @ rdesc_local.T
    confidence = _dual_softmax_confidence(scores, float(logit_scale))
    matches: list[KeypointFeatureMatch] = []
    for local_q in range(qdesc_local.shape[0]):
        keep_mask = _local_window_mask_for_query(
            local_q,
            qrows,
            rrows,
            query_grid_width=int(query_grid.width),
            query_grid_height=int(query_grid.height),
            render_grid_width=int(render_grid.width),
            render_grid_height=int(render_grid.height),
            radius=local_window_radius_cells,
        )
        candidate_locals = np.flatnonzero(keep_mask)
        if candidate_locals.size == 0:
            continue
        row_conf = confidence[local_q, candidate_locals]
        order = np.lexsort((-scores[local_q, candidate_locals], -row_conf))
        ordered_locals = candidate_locals[order]
        top1_score = float(scores[local_q, ordered_locals[0]])
        for rank, local_r in enumerate(ordered_locals[:k]):
            conf = float(confidence[local_q, int(local_r)])
            sim = float(scores[local_q, int(local_r)])
            if conf < float(min_confidence) or sim < float(min_similarity):
                continue
            reciprocal_rank = _coarse_reciprocal_rank(confidence, local_q, int(local_r))
            if mode == "filter" and reciprocal_rank != 0:
                continue
            margin, ratio = _top2_margin(scores, local_q, int(local_r))
            render_index = int(rrows[int(local_r)])
            matches.append(
                KeypointFeatureMatch(
                    query_index=int(qrows[local_q]),
                    render_index=render_index,
                    query_xy=query_grid.xy[int(qrows[local_q])],
                    render_xy=render_grid.xy[render_index],
                    similarity=sim,
                    ratio=ratio,
                    similarity_margin=margin,
                    dual_softmax_confidence=conf,
                    base_render_index=render_index,
                    candidate_render_index=render_index,
                    candidate_id=len(matches),
                    coarse_rank=int(rank),
                    coarse_score=sim,
                    coarse_score_gap=float(top1_score - sim),
                    mutual_rank=int(reciprocal_rank),
                    cell_delta_x=0,
                    cell_delta_y=0,
                )
            )
    matches.sort(
        key=lambda item: (
            float(item.dual_softmax_confidence or 0.0),
            -float(item.coarse_rank or 0),
            float(item.similarity),
        ),
        reverse=True,
    )
    if max_matches is not None:
        matches = matches[: int(max_matches)]
    return matches


def matcha_coarse_dual_softmax_matches(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    logit_scale: float = 10.0,
    min_confidence: float = 0.0,
    min_similarity: float = -1.0,
    max_matches: int | None = None,
    mutual: bool = True,
    deduplicate: bool = True,
    query_candidate_indices: np.ndarray | None = None,
    render_candidate_indices: np.ndarray | None = None,
) -> list[KeypointFeatureMatch]:
    """Coarse patch matching over all feature-map cells."""

    query_grid = feature_map_to_coarse_grid(
        query_feature_map,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    render_grid = feature_map_to_coarse_grid(
        render_feature_map,
        image_width=int(render_image_width),
        image_height=int(render_image_height),
    )
    qdesc, qvalid = normalize_rows(query_grid.descriptors)
    rdesc, rvalid = normalize_rows(render_grid.descriptors)
    qrows = np.flatnonzero(qvalid)
    rrows = np.flatnonzero(rvalid)
    if query_candidate_indices is not None:
        qmask = np.zeros((query_grid.xy.shape[0],), dtype=bool)
        qidx = np.asarray(query_candidate_indices, dtype=np.int64).reshape(-1)
        qidx = qidx[(qidx >= 0) & (qidx < qmask.shape[0])]
        qmask[qidx] = True
        qrows = qrows[qmask[qrows]]
    if render_candidate_indices is not None:
        rmask = np.zeros((render_grid.xy.shape[0],), dtype=bool)
        ridx = np.asarray(render_candidate_indices, dtype=np.int64).reshape(-1)
        ridx = ridx[(ridx >= 0) & (ridx < rmask.shape[0])]
        rmask[ridx] = True
        rrows = rrows[rmask[rrows]]
    if qrows.size == 0 or rrows.size == 0:
        return []
    qdesc = qdesc[qrows]
    rdesc = rdesc[rrows]
    scores = qdesc @ rdesc.T
    confidence = _dual_softmax_confidence(scores, float(logit_scale))
    query_best = np.argmax(confidence, axis=1)
    render_best = np.argmax(confidence, axis=0)
    candidates: list[KeypointFeatureMatch] = []
    for local_q in range(qdesc.shape[0]):
        local_r = int(query_best[local_q])
        if mutual and int(render_best[local_r]) != local_q:
            continue
        conf = float(confidence[local_q, local_r])
        sim = float(scores[local_q, local_r])
        if conf < float(min_confidence) or sim < float(min_similarity):
            continue
        margin, ratio = _top2_margin(scores, local_q, local_r)
        reciprocal_rank = _coarse_reciprocal_rank(confidence, local_q, local_r)
        candidates.append(
            KeypointFeatureMatch(
                query_index=int(qrows[local_q]),
                render_index=int(rrows[local_r]),
                query_xy=query_grid.xy[int(qrows[local_q])],
                render_xy=render_grid.xy[int(rrows[local_r])],
                similarity=sim,
                ratio=ratio,
                similarity_margin=margin,
                dual_softmax_confidence=conf,
                base_render_index=int(rrows[local_r]),
                candidate_render_index=int(rrows[local_r]),
                candidate_id=len(candidates),
                coarse_rank=0,
                coarse_score=sim,
                coarse_score_gap=0.0,
                mutual_rank=int(reciprocal_rank),
                cell_delta_x=0,
                cell_delta_y=0,
            )
        )
    candidates.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    if deduplicate:
        query_cell = max(float(query_image_width) / float(query_grid.width), float(query_image_height) / float(query_grid.height))
        render_cell = max(float(render_image_width) / float(render_grid.width), float(render_image_height) / float(render_grid.height))
        candidates = deduplicate_repeated_correspondences(
            candidates,
            query_cell_size_px=query_cell,
            render_cell_size_px=render_cell,
        )
    if max_matches is not None:
        candidates = candidates[: int(max_matches)]
    return candidates


def refine_render_matches_by_local_softargmax(
    matches: Sequence[KeypointFeatureMatch],
    query_descriptors_by_match: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    search_radius_px: float = 8.0,
    step_px: float = 1.0,
    temperature: float = 20.0,
) -> list[KeypointFeatureMatch]:
    """Refine render-side measurements with local correlation soft-argmax."""

    if not matches:
        return []
    radius = float(search_radius_px)
    step = float(step_px)
    if radius <= 0.0:
        return list(matches)
    if step <= 0.0:
        raise ValueError("step_px must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    qdesc = np.asarray(query_descriptors_by_match, dtype=np.float32)
    if qdesc.ndim != 2 or qdesc.shape[0] != len(matches):
        raise ValueError("query_descriptors_by_match must have shape (len(matches), C)")
    qdesc, qvalid = normalize_rows(qdesc)
    offsets_1d = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
    offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=1)
    refined: list[KeypointFeatureMatch] = []
    for idx, match in enumerate(matches):
        if not bool(qvalid[idx]):
            refined.append(match)
            continue
        candidates = np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2) + offsets
        samples, valid = bilinear_sample_feature_map(
            render_feature_map,
            candidates,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        if samples.shape[0] == 0 or not np.any(valid):
            refined.append(match)
            continue
        samples, svalid = normalize_rows(samples)
        valid = valid & svalid
        if not np.any(valid):
            refined.append(match)
            continue
        valid_candidates = candidates[valid]
        scores = samples[valid] @ qdesc[idx]
        logits = scores.astype(np.float64) * float(temperature)
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        refined_xy = np.sum(valid_candidates * probs[:, None], axis=0)
        refined_similarity = float(np.sum(scores * probs))
        refined.append(replace(match, render_xy=refined_xy.astype(np.float64), similarity=refined_similarity))
    return refined


def refine_matches_by_bilateral_local_correlation(
    matches: Sequence[KeypointFeatureMatch],
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    search_radius_px: float = 4.0,
    step_px: float = 1.0,
    mode: str = "argmax",
    temperature: float = 20.0,
) -> list[KeypointFeatureMatch]:
    """Refine both query and render measurements by local patch correlation."""

    if not matches:
        return []
    radius = float(search_radius_px)
    step = float(step_px)
    if radius <= 0.0:
        return list(matches)
    if step <= 0.0:
        raise ValueError("step_px must be positive")
    mode = str(mode).lower()
    if mode not in {"argmax", "softargmax"}:
        raise ValueError("mode must be 'argmax' or 'softargmax'")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    offsets_1d = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
    offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=1)
    refined: list[KeypointFeatureMatch] = []
    for match in matches:
        query_candidates = np.asarray(match.query_xy, dtype=np.float64).reshape(1, 2) + offsets
        render_candidates = np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2) + offsets
        query_samples, query_valid = bilinear_sample_feature_map(
            query_feature_map,
            query_candidates,
            image_width=int(query_image_width),
            image_height=int(query_image_height),
        )
        render_samples, render_valid = bilinear_sample_feature_map(
            render_feature_map,
            render_candidates,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
        )
        if not np.any(query_valid) or not np.any(render_valid):
            refined.append(match)
            continue
        query_samples, qnorm_valid = normalize_rows(query_samples)
        render_samples, rnorm_valid = normalize_rows(render_samples)
        query_valid = query_valid & qnorm_valid
        render_valid = render_valid & rnorm_valid
        if not np.any(query_valid) or not np.any(render_valid):
            refined.append(match)
            continue
        qxy = query_candidates[query_valid]
        rxy = render_candidates[render_valid]
        qdesc = query_samples[query_valid]
        rdesc = render_samples[render_valid]
        scores = qdesc @ rdesc.T
        if mode == "argmax":
            best_score = float(np.max(scores))
            ties = np.argwhere(np.isclose(scores, best_score, rtol=1e-6, atol=1e-8))
            if ties.shape[0] > 1:
                qdist = np.linalg.norm(qxy[ties[:, 0]] - np.asarray(match.query_xy, dtype=np.float64).reshape(1, 2), axis=1)
                rdist = np.linalg.norm(rxy[ties[:, 1]] - np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2), axis=1)
                chosen = int(np.argmin(qdist + rdist))
                qi, ri = int(ties[chosen, 0]), int(ties[chosen, 1])
            else:
                qi, ri = int(ties[0, 0]), int(ties[0, 1])
            refined.append(replace(match, query_xy=qxy[qi], render_xy=rxy[ri], similarity=float(scores[qi, ri])))
            continue
        logits = scores.astype(np.float64) * float(temperature)
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        query_probs = np.sum(probs, axis=1)
        render_probs = np.sum(probs, axis=0)
        refined_query_xy = np.sum(qxy * query_probs[:, None], axis=0)
        refined_render_xy = np.sum(rxy * render_probs[:, None], axis=0)
        refined_similarity = float(np.sum(scores * probs))
        refined.append(
            replace(
                match,
                query_xy=refined_query_xy.astype(np.float64, copy=False),
                render_xy=refined_render_xy.astype(np.float64, copy=False),
                similarity=refined_similarity,
            )
        )
    return refined


def refine_render_matches_by_local_attention(
    matches: Sequence[KeypointFeatureMatch],
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    search_radius_px: float = 4.0,
    step_px: float = 1.0,
    mode: str = "argmax",
    temperature: float = 20.0,
    query_spatial_sigma_px: float = 4.0,
) -> list[KeypointFeatureMatch]:
    """Refine render measurements using a query local-context attention score.

    Unlike bilateral refinement, this keeps the query measurement fixed. That
    avoids injecting query-side offset noise into PnP while still allowing nearby
    query patch context to disambiguate the render-side local search.
    """

    if not matches:
        return []
    radius = float(search_radius_px)
    step = float(step_px)
    if radius <= 0.0:
        return list(matches)
    if step <= 0.0:
        raise ValueError("step_px must be positive")
    mode = str(mode).lower()
    if mode not in {"argmax", "softargmax"}:
        raise ValueError("mode must be 'argmax' or 'softargmax'")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    sigma = max(float(query_spatial_sigma_px), 1e-6)
    offsets_1d = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
    offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=1)
    query_prior = np.exp(-0.5 * np.sum(offsets * offsets, axis=1) / (sigma * sigma)).astype(np.float32)
    query_prior = query_prior / max(float(np.max(query_prior)), 1e-12)
    refined: list[KeypointFeatureMatch] = []
    for match in matches:
        query_candidates = np.asarray(match.query_xy, dtype=np.float64).reshape(1, 2) + offsets
        render_candidates = np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2) + offsets
        query_samples, query_valid = bilinear_sample_feature_map(
            query_feature_map,
            query_candidates,
            image_width=int(query_image_width),
            image_height=int(query_image_height),
        )
        render_samples, render_valid = bilinear_sample_feature_map(
            render_feature_map,
            render_candidates,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
        )
        if not np.any(query_valid) or not np.any(render_valid):
            refined.append(match)
            continue
        query_samples, qnorm_valid = normalize_rows(query_samples)
        render_samples, rnorm_valid = normalize_rows(render_samples)
        query_valid = query_valid & qnorm_valid
        render_valid = render_valid & rnorm_valid
        if not np.any(query_valid) or not np.any(render_valid):
            refined.append(match)
            continue
        qdesc = query_samples[query_valid]
        rdesc = render_samples[render_valid]
        rxy = render_candidates[render_valid]
        scores = qdesc @ rdesc.T
        qprior = query_prior[query_valid]
        weights = qprior[:, None] * np.exp(
            (scores - np.max(scores, axis=0, keepdims=True)).astype(np.float64) * float(temperature)
        )
        weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
        render_scores = np.sum(scores * weights, axis=0)
        if mode == "argmax":
            best_score = float(np.max(render_scores))
            tie = np.flatnonzero(np.isclose(render_scores, best_score, rtol=1e-6, atol=1e-8))
            if tie.size > 1:
                distances = np.linalg.norm(rxy[tie] - np.asarray(match.render_xy, dtype=np.float64).reshape(1, 2), axis=1)
                best = int(tie[int(np.argmin(distances))])
            else:
                best = int(tie[0])
            refined.append(replace(match, render_xy=rxy[best].astype(np.float64, copy=False), similarity=best_score))
            continue
        logits = render_scores.astype(np.float64) * float(temperature)
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        refined_xy = np.sum(rxy * probs[:, None], axis=0)
        refined_similarity = float(np.sum(render_scores * probs))
        refined.append(replace(match, render_xy=refined_xy.astype(np.float64, copy=False), similarity=refined_similarity))
    return refined


def apply_keypoint_cell_prior_to_matches(
    matches: Sequence[KeypointFeatureMatch],
    *,
    query_candidate_indices: np.ndarray | None,
    render_candidate_indices: np.ndarray | None,
    boost: float = 0.25,
    penalty: float = 0.0,
) -> list[KeypointFeatureMatch]:
    """Apply a soft detector/keypoint-cell prior to match confidence.

    The prior is intentionally non-destructive: it changes ordering/confidence
    used by later coverage filtering and pose scoring, but keeps all matches
    available to RANSAC.
    """

    if query_candidate_indices is None or render_candidate_indices is None:
        return list(matches)
    qset = {int(item) for item in np.asarray(query_candidate_indices, dtype=np.int64).reshape(-1).tolist()}
    rset = {int(item) for item in np.asarray(render_candidate_indices, dtype=np.int64).reshape(-1).tolist()}
    if not qset or not rset:
        return list(matches)
    positive_scale = 1.0 + max(float(boost), 0.0)
    negative_scale = max(0.0, 1.0 - max(float(penalty), 0.0))
    updated: list[KeypointFeatureMatch] = []
    for match in matches:
        base = float(match.dual_softmax_confidence) if match.dual_softmax_confidence is not None else float(np.clip((match.similarity + 1.0) * 0.5, 0.0, 1.0))
        supported = int(match.query_index) in qset and int(match.render_index) in rset
        confidence = base * (positive_scale if supported else negative_scale)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        updated.append(replace(match, dual_softmax_confidence=confidence))
    updated.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    return updated


def apply_cell_reliability_prior_to_matches(
    matches: Sequence[KeypointFeatureMatch],
    *,
    query_reliability: np.ndarray | None,
    render_reliability: np.ndarray | None,
    boost: float = 0.25,
    penalty: float = 0.0,
) -> list[KeypointFeatureMatch]:
    """Apply continuous detector/reliability scores without dropping matches."""

    if query_reliability is None or render_reliability is None:
        return list(matches)
    qrel = np.asarray(query_reliability, dtype=np.float32).reshape(-1)
    rrel = np.asarray(render_reliability, dtype=np.float32).reshape(-1)
    if qrel.size == 0 or rrel.size == 0:
        return list(matches)
    qrel = np.clip(qrel, 0.0, 1.0)
    rrel = np.clip(rrel, 0.0, 1.0)
    positive = max(float(boost), 0.0)
    negative = max(float(penalty), 0.0)
    updated: list[KeypointFeatureMatch] = []
    for match in matches:
        qidx = int(match.query_index)
        ridx = int(match.render_index)
        if qidx < 0 or qidx >= qrel.shape[0] or ridx < 0 or ridx >= rrel.shape[0]:
            updated.append(match)
            continue
        base = float(match.dual_softmax_confidence) if match.dual_softmax_confidence is not None else float(np.clip((match.similarity + 1.0) * 0.5, 0.0, 1.0))
        reliability = float(np.sqrt(float(qrel[qidx]) * float(rrel[ridx])))
        confidence = base * (1.0 + positive * reliability)
        if negative > 0.0:
            confidence *= max(0.0, 1.0 - negative * (1.0 - reliability))
        updated.append(replace(match, dual_softmax_confidence=float(np.clip(confidence, 0.0, 1.0))))
    updated.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    return updated


def matcha_coarse_to_fine_keypoint_matches(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    logit_scale: float = 10.0,
    min_confidence: float = 0.0,
    min_similarity: float = -1.0,
    max_matches: int | None = None,
    fine_search_radius_px: float = 8.0,
    fine_search_step_px: float = 1.0,
    fine_mode: str = "argmax",
    fine_softmax_temperature: float = 20.0,
    mutual: bool = True,
    coarse_top_k_per_query: int = 1,
    coarse_mutual_mode: str | None = None,
    coarse_local_window_radius_cells: int | None = None,
    query_offset_logits: np.ndarray | None = None,
    render_offset_logits: np.ndarray | None = None,
    query_candidate_indices: np.ndarray | None = None,
    render_candidate_indices: np.ndarray | None = None,
) -> list[KeypointFeatureMatch]:
    """Run coarse dual-softmax followed by render-side local refinement."""

    mode = str(fine_mode).lower()
    coarse_query_feature_map = query_feature_map
    coarse_render_feature_map = render_feature_map
    if mode in {"cross_argmax", "cross_softargmax"}:
        coarse_query_feature_map, coarse_render_feature_map = cross_attention_enhance_feature_maps(
            query_feature_map,
            render_feature_map,
            alpha=0.25,
            logit_scale=float(logit_scale),
        )
    top_k = max(int(coarse_top_k_per_query), 1)
    mutual_mode = None if coarse_mutual_mode is None else str(coarse_mutual_mode)
    if top_k > 1 or mutual_mode is not None:
        if mutual_mode is None:
            mutual_mode = "filter" if bool(mutual) else "none"
        matches = matcha_coarse_topk_matches(
            coarse_query_feature_map,
            coarse_render_feature_map,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            k_per_query=top_k,
            logit_scale=float(logit_scale),
            min_confidence=float(min_confidence),
            min_similarity=float(min_similarity),
            max_matches=max_matches,
            mutual_mode=mutual_mode,
            local_window_radius_cells=coarse_local_window_radius_cells,
            query_candidate_indices=query_candidate_indices,
            render_candidate_indices=render_candidate_indices,
        )
    else:
        matches = matcha_coarse_dual_softmax_matches(
            coarse_query_feature_map,
            coarse_render_feature_map,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            logit_scale=float(logit_scale),
            min_confidence=float(min_confidence),
            min_similarity=float(min_similarity),
            max_matches=max_matches,
            mutual=bool(mutual),
            deduplicate=True,
            query_candidate_indices=query_candidate_indices,
            render_candidate_indices=render_candidate_indices,
        )
    if not matches or float(fine_search_radius_px) <= 0.0:
        if query_offset_logits is not None or render_offset_logits is not None:
            return apply_offset_logits_to_matches(
                matches,
                query_offset_logits,
                render_offset_logits,
                query_image_width=int(query_image_width),
                query_image_height=int(query_image_height),
                render_image_width=int(render_image_width),
                render_image_height=int(render_image_height),
            )
        return matches
    if query_offset_logits is not None or render_offset_logits is not None:
        matches = apply_offset_logits_to_matches(
            matches,
            query_offset_logits,
            render_offset_logits,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
        )
        if not matches:
            return []
    query_xy = np.stack([match.query_xy for match in matches], axis=0)
    query_desc, qvalid = bilinear_sample_feature_map(
        coarse_query_feature_map,
        query_xy,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    compact_original = [match for match, valid in zip(matches, qvalid) if bool(valid)]
    compact_desc = query_desc[qvalid]
    if mode in {"bilateral_argmax", "bilateral_softargmax"}:
        return refine_matches_by_bilateral_local_correlation(
            matches,
            coarse_query_feature_map,
            coarse_render_feature_map,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            search_radius_px=float(fine_search_radius_px),
            step_px=float(fine_search_step_px),
            mode="softargmax" if mode == "bilateral_softargmax" else "argmax",
            temperature=float(fine_softmax_temperature),
        )
    if mode in {"fine_attention_argmax", "fine_attention_softargmax"}:
        return refine_render_matches_by_local_attention(
            matches,
            coarse_query_feature_map,
            coarse_render_feature_map,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            search_radius_px=float(fine_search_radius_px),
            step_px=float(fine_search_step_px),
            mode="softargmax" if mode == "fine_attention_softargmax" else "argmax",
            temperature=float(fine_softmax_temperature),
        )
    if mode in {"cross_argmax", "cross_softargmax"}:
        mode = "softargmax" if mode == "cross_softargmax" else "argmax"
    if mode == "softargmax":
        return refine_render_matches_by_local_softargmax(
            compact_original,
            compact_desc,
            coarse_render_feature_map,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
            search_radius_px=float(fine_search_radius_px),
            step_px=float(fine_search_step_px),
            temperature=float(fine_softmax_temperature),
        )
    if mode == "argmax":
        compact_matches = [replace(match, query_index=idx) for idx, match in enumerate(compact_original)]
        refined = refine_render_keypoint_matches_by_local_correlation(
            compact_matches,
            compact_desc,
            coarse_render_feature_map,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
            search_radius_px=float(fine_search_radius_px),
            step_px=float(fine_search_step_px),
        )
        return [replace(match, query_index=int(original.query_index)) for match, original in zip(refined, compact_original)]
    raise ValueError(
        "fine_mode must be 'argmax', 'softargmax', 'bilateral_argmax', 'bilateral_softargmax', "
        "'fine_attention_argmax', 'fine_attention_softargmax', 'cross_argmax', or 'cross_softargmax'"
    )
