"""Coarse-only fixed-anchor audit utilities.

These helpers intentionally avoid learned fine heads, learned confidence, and
pose scoring. They build controls that test whether GT-render localization is
driven by visual correspondence or by an identity-grid shortcut.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from feature_extract.vfm.matcha_coarse_to_fine import feature_map_to_coarse_grid
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


COARSE_CONTROL_MODES = (
    "learned",
    "same_cell_identity",
    "constant",
    "random_normalized",
    "spatial_permute_query",
    "shift_query_cells",
)


def parse_cell_shift(text: str | Sequence[int | float]) -> tuple[int, int]:
    """Parse a ``dx,dy`` cell shift string."""

    if isinstance(text, str):
        parts = [item.strip() for item in text.split(",") if item.strip()]
    else:
        parts = [str(item) for item in text]
    if len(parts) != 2:
        raise ValueError("cell shift must contain exactly two values: dx,dy")
    return int(float(parts[0])), int(float(parts[1]))


def _normalize_feature_cells(feature_map: np.ndarray) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    rows = fmap.reshape(channels, height * width).T
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    rows = rows.copy()
    rows[valid] /= norms[valid]
    return rows.T.reshape(channels, height, width).astype(np.float32, copy=False)


def _constant_like(feature_map: np.ndarray) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    out = np.ones_like(fmap, dtype=np.float32)
    return _normalize_feature_cells(out)


def _random_normalized_like(feature_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    values = rng.standard_normal(size=fmap.shape).astype(np.float32)
    return _normalize_feature_cells(values)


def _permute_query_spatial(feature_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    flat = fmap.reshape(channels, height * width)
    order = rng.permutation(height * width)
    return flat[:, order].reshape(channels, height, width).astype(np.float32, copy=False)


def _shift_feature_cells(feature_map: np.ndarray, dx: int, dy: int) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    out = np.zeros_like(fmap, dtype=np.float32)
    _channels, height, width = fmap.shape
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - int(dx))
    dst_x0 = max(0, int(dx))
    dst_x1 = min(width, width + int(dx))
    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - int(dy))
    dst_y0 = max(0, int(dy))
    dst_y1 = min(height, height + int(dy))
    if src_x0 < src_x1 and src_y0 < src_y1:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = fmap[:, src_y0:src_y1, src_x0:src_x1]
    return out


def apply_coarse_feature_control(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    mode: str = "learned",
    seed: int = 0,
    shift_cells: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a coarse-control feature intervention before matching."""

    normalized_mode = str(mode)
    if normalized_mode not in COARSE_CONTROL_MODES:
        raise ValueError(f"unsupported coarse control mode: {mode}")
    query = np.asarray(query_feature_map, dtype=np.float32)
    render = np.asarray(render_feature_map, dtype=np.float32)
    if query.ndim != 3 or render.ndim != 3:
        raise ValueError("feature maps must have shape (C, H, W)")
    if normalized_mode in {"learned", "same_cell_identity"}:
        return query, render
    if normalized_mode == "constant":
        return _constant_like(query), _constant_like(render)
    rng = np.random.default_rng(int(seed))
    if normalized_mode == "random_normalized":
        return _random_normalized_like(query, rng), _random_normalized_like(render, rng)
    if normalized_mode == "spatial_permute_query":
        return _permute_query_spatial(query, rng), render
    if normalized_mode == "shift_query_cells":
        dx, dy = shift_cells
        return _shift_feature_cells(query, int(dx), int(dy)), render
    raise AssertionError(f"unhandled coarse control mode: {mode}")


def same_cell_identity_keypoint_matches(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    max_matches: int | None = None,
) -> list[KeypointFeatureMatch]:
    """Build feature-free identity-cell matches.

    Cell ``i`` in the query grid is matched to cell ``i`` in the render grid.
    This is the strict shortcut baseline for GT-pose render protocols.
    """

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
    count = min(int(query_grid.xy.shape[0]), int(render_grid.xy.shape[0]))
    if max_matches is not None:
        count = min(count, max(int(max_matches), 0))
    matches: list[KeypointFeatureMatch] = []
    for idx in range(count):
        matches.append(
            KeypointFeatureMatch(
                query_index=idx,
                render_index=idx,
                query_xy=query_grid.xy[idx],
                render_xy=render_grid.xy[idx],
                similarity=1.0,
                ratio=0.0,
                similarity_margin=0.0,
                dual_softmax_confidence=1.0,
                base_render_index=idx,
                candidate_render_index=idx,
                candidate_id=idx,
                coarse_rank=0,
                coarse_score=1.0,
                coarse_score_gap=0.0,
                mutual_rank=idx,
                cell_delta_x=0,
                cell_delta_y=0,
            )
        )
    return matches


def shifted_query_identity_target_matches(
    matches: Sequence[KeypointFeatureMatch],
    *,
    query_grid_width: int,
    render_grid_width: int,
    shift_cells: tuple[int, int],
) -> list[KeypointFeatureMatch]:
    """Convert identity-cell matches into expected matches for a shifted query map.

    This is useful for tests: if the query feature map is shifted by ``dx,dy``,
    a content-driven matcher should move predicted render indices by roughly
    ``-dx,-dy`` relative to query index.
    """

    dx, dy = shift_cells
    out: list[KeypointFeatureMatch] = []
    for match in matches:
        qy, qx = divmod(int(match.query_index), int(query_grid_width))
        rx = qx - int(dx)
        ry = qy - int(dy)
        if rx < 0 or ry < 0 or rx >= int(render_grid_width):
            continue
        render_index = int(ry * int(render_grid_width) + rx)
        out.append(
            replace(
                match,
                render_index=render_index,
                base_render_index=render_index,
                candidate_render_index=render_index,
            )
        )
    return out


def coarse_match_index_summary(
    matches: Sequence[KeypointFeatureMatch],
    *,
    query_grid_width: int,
    render_grid_width: int,
) -> dict[str, float | int | None]:
    """Summarize cell-index displacement for coarse matches."""

    values = list(matches)
    if not values:
        return {
            "coarse_match_count": 0,
            "same_cell_fraction": None,
            "median_cell_delta_x": None,
            "median_cell_delta_y": None,
            "mean_cell_delta_x": None,
            "mean_cell_delta_y": None,
        }
    same = []
    dxs = []
    dys = []
    for match in values:
        qy, qx = divmod(int(match.query_index), int(query_grid_width))
        ry, rx = divmod(int(match.render_index), int(render_grid_width))
        same.append(int(match.query_index) == int(match.render_index))
        dxs.append(float(rx - qx))
        dys.append(float(ry - qy))
    dx_arr = np.asarray(dxs, dtype=np.float64)
    dy_arr = np.asarray(dys, dtype=np.float64)
    return {
        "coarse_match_count": int(len(values)),
        "same_cell_fraction": float(np.mean(np.asarray(same, dtype=np.float64))),
        "median_cell_delta_x": float(np.median(dx_arr)),
        "median_cell_delta_y": float(np.median(dy_arr)),
        "mean_cell_delta_x": float(np.mean(dx_arr)),
        "mean_cell_delta_y": float(np.mean(dy_arr)),
    }
