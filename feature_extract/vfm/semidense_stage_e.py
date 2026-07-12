"""Stage E semi-dense diagnostics and sparse-primary matching utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSet, PatchPositiveSets, TokenPatchBox
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    QueryTo3DMatch,
    match_reprojection_errors,
    match_spatial_distribution_stats,
    normalize_rows,
)
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


def match_source_label(match_or_track_id: QueryTo3DMatch | int) -> str:
    track_id = int(match_or_track_id.track_id if isinstance(match_or_track_id, QueryTo3DMatch) else match_or_track_id)
    return "semidense" if track_id < 0 else "sparse"


def concatenate_landmark_indices(indices: Sequence[LandmarkMapIndex]) -> LandmarkMapIndex:
    present = [index for index in indices if len(index) > 0]
    if not present:
        feature_dim = int(indices[0].feature_dim) if indices else 0
        return LandmarkMapIndex(
            track_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            features=np.zeros((0, feature_dim), dtype=np.float32),
            mean_variances=np.zeros((0,), dtype=np.float32),
            observation_counts=np.zeros((0,), dtype=np.int64),
            observation_image_ids=(),
            reprojection_errors=np.zeros((0,), dtype=np.float32),
            feature_ambiguities=np.zeros((0,), dtype=np.float32),
        )
    return LandmarkMapIndex(
        track_ids=np.concatenate([index.track_ids for index in present], axis=0),
        xyz=np.concatenate([index.xyz for index in present], axis=0),
        features=np.concatenate([index.features for index in present], axis=0),
        mean_variances=np.concatenate([index.mean_variances for index in present], axis=0),
        observation_counts=np.concatenate([index.observation_counts for index in present], axis=0),
        observation_image_ids=tuple(image_ids for index in present for image_ids in index.observation_image_ids),
        reprojection_errors=np.concatenate([index.reprojection_errors for index in present], axis=0),
        feature_ambiguities=np.concatenate([index.feature_ambiguities for index in present], axis=0),
    )


def semidense_gaussian_only(anchor_map: SemiDenseAnchorMap) -> SemiDenseAnchorMap:
    return anchor_map.subset(np.asarray(anchor_map.source_types, dtype=str) != "sfm")


@dataclass(frozen=True)
class SemiDensePruningConfig:
    min_support: int = 1
    min_quality: float | None = None
    max_distance: float | None = None
    max_feature_variance: float | None = None
    max_per_source_track: int | None = None


def prune_semidense_anchors(anchor_map: SemiDenseAnchorMap, config: SemiDensePruningConfig) -> SemiDenseAnchorMap:
    mask = np.ones((len(anchor_map),), dtype=bool)
    if int(config.min_support) > 1:
        mask &= anchor_map.support_counts >= int(config.min_support)
    if config.min_quality is not None:
        mask &= anchor_map.quality_scores >= float(config.min_quality)
    if config.max_distance is not None:
        mask &= anchor_map.mean_distances <= float(config.max_distance)
    if config.max_feature_variance is not None:
        mask &= anchor_map.feature_variances <= float(config.max_feature_variance)
    if config.max_per_source_track is not None and int(config.max_per_source_track) > 0:
        keep = np.zeros((len(anchor_map),), dtype=bool)
        by_source: dict[int, list[int]] = {}
        for idx, source_track_id in enumerate(anchor_map.source_track_ids.tolist()):
            if bool(mask[idx]):
                by_source.setdefault(int(source_track_id), []).append(int(idx))
        for _source_track_id, rows in by_source.items():
            order = sorted(
                rows,
                key=lambda row: (
                    -float(anchor_map.quality_scores[row]),
                    float(anchor_map.mean_distances[row]),
                    int(anchor_map.anchor_ids[row]),
                ),
            )
            keep[order[: int(config.max_per_source_track)]] = True
        mask &= keep
    return anchor_map.subset(mask)


def duplicate_anchor_stats(anchor_map: SemiDenseAnchorMap, radius_m: float = 0.02) -> dict[str, float | int]:
    if len(anchor_map) == 0:
        return {
            "anchor_count": 0,
            "source_track_count": 0,
            "mean_anchors_per_source": 0.0,
            "max_anchors_per_source": 0,
            "near_duplicate_pair_count": 0,
            "near_duplicate_anchor_fraction": 0.0,
        }
    counts = np.asarray(
        [np.sum(anchor_map.source_track_ids == source_id) for source_id in np.unique(anchor_map.source_track_ids)],
        dtype=np.int64,
    )
    tree = cKDTree(anchor_map.xyz)
    pairs = tree.query_pairs(r=float(radius_m), output_type="ndarray")
    duplicate_rows = np.unique(pairs.reshape(-1)) if pairs.size else np.zeros((0,), dtype=np.int64)
    return {
        "anchor_count": int(len(anchor_map)),
        "source_track_count": int(np.unique(anchor_map.source_track_ids).shape[0]),
        "mean_anchors_per_source": float(np.mean(counts)) if counts.size else 0.0,
        "max_anchors_per_source": int(np.max(counts)) if counts.size else 0,
        "near_duplicate_pair_count": int(pairs.shape[0]) if pairs.size else 0,
        "near_duplicate_anchor_fraction": float(duplicate_rows.size / max(len(anchor_map), 1)),
        "radius_m": float(radius_m),
    }


def _patch_correct(matches: Sequence[QueryTo3DMatch], positives: PatchPositiveSets) -> np.ndarray:
    result = []
    empty = PatchPositiveSet(0, TokenPatchBox(0, np.zeros(2), 0, 0, 0, 0), set())
    for match in matches:
        positive = positives.by_token.get(int(match.token_index), empty)
        result.append(int(match.track_id) in positive.track_ids)
    return np.asarray(result, dtype=bool)


def source_breakdown_stats(
    matches: Sequence[QueryTo3DMatch],
    positives: PatchPositiveSets,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    stride_px: float,
    pnp_inlier_mask: np.ndarray | None = None,
) -> dict[str, object]:
    errors = match_reprojection_errors(matches, pose_w2c, camera) if matches else np.zeros((0,), dtype=np.float64)
    patch_correct = _patch_correct(matches, positives) if matches else np.zeros((0,), dtype=bool)
    labels = np.asarray([match_source_label(match) for match in matches], dtype=str)
    inlier_mask = (
        np.zeros((len(matches),), dtype=bool)
        if pnp_inlier_mask is None
        else np.asarray(pnp_inlier_mask, dtype=bool).reshape(-1)
    )
    if inlier_mask.shape[0] != len(matches):
        raise ValueError("pnp_inlier_mask must have one value per match")
    output: dict[str, object] = {}
    for source in ("sparse", "semidense"):
        mask = labels == source
        inliers = mask & inlier_mask
        output[source] = {
            "match_count": int(np.sum(mask)),
            "match_fraction": float(np.mean(mask)) if len(matches) else 0.0,
            "patch_at_1": None if not np.any(mask) else float(np.mean(patch_correct[mask])),
            "gt_precision_stride": None if not np.any(mask) else float(np.mean(errors[mask] <= float(stride_px))),
            "gt_reproj_median_px": None if not np.any(mask) else float(np.median(errors[mask])),
            "pnp_inlier_count": int(np.sum(inliers)),
            "pnp_inlier_fraction": float(np.sum(inliers) / max(int(np.sum(inlier_mask)), 1)) if np.any(inlier_mask) else 0.0,
            "pnp_inlier_patch_at_1": None if not np.any(inliers) else float(np.mean(patch_correct[inliers])),
            "pnp_inlier_gt_precision_stride": None
            if not np.any(inliers)
            else float(np.mean(errors[inliers] <= float(stride_px))),
            "pnp_inlier_gt_reproj_median_px": None if not np.any(inliers) else float(np.median(errors[inliers])),
            "all_spatial": match_spatial_distribution_stats(
                [match for match, keep in zip(matches, mask) if bool(keep)],
                int(camera.width),
                int(camera.height),
                pose_w2c=pose_w2c,
            ),
            "pnp_inlier_spatial": match_spatial_distribution_stats(
                [match for match, keep in zip(matches, inliers) if bool(keep)],
                int(camera.width),
                int(camera.height),
                pose_w2c=pose_w2c,
            ),
        }
    duplicate_sources = [int(match.track_id) for match in matches if int(match.track_id) < 0]
    output["semidense_duplicate"] = {
        "selected_semidense_match_count": int(len(duplicate_sources)),
        "unique_semidense_track_count": int(len(set(duplicate_sources))),
    }
    return output


@dataclass(frozen=True)
class SparsePrimaryFillConfig:
    mode: str = "no_sparse"
    max_semidense_fraction: float = 0.2
    low_margin_threshold: float = 0.02
    grid_rows: int = 4
    grid_cols: int = 4
    min_sparse_per_cell: int = 4
    max_semidense_per_token: int = 1
    max_semidense_per_source_track: int = 1
    max_semidense_per_cell: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"all", "no_sparse", "low_margin", "low_coverage"}:
            raise ValueError("mode must be one of: all, no_sparse, low_margin, low_coverage")
        if not 0.0 <= float(self.max_semidense_fraction) < 1.0:
            raise ValueError("max_semidense_fraction must be in [0, 1)")


def _best_sparse_by_token(matches: Sequence[QueryTo3DMatch]) -> dict[int, QueryTo3DMatch]:
    best: dict[int, QueryTo3DMatch] = {}
    for match in matches:
        token = int(match.token_index)
        current = best.get(token)
        if current is None or float(match.similarity) > float(current.similarity):
            best[token] = match
    return best


def _grid_cell(match: QueryTo3DMatch, image_width: int, image_height: int, rows: int, cols: int) -> tuple[int, int]:
    col = int(np.clip(np.floor(float(match.xy[0]) / max(float(image_width), 1.0) * int(cols)), 0, int(cols) - 1))
    row = int(np.clip(np.floor(float(match.xy[1]) / max(float(image_height), 1.0) * int(rows)), 0, int(rows) - 1))
    return row, col


def sparse_primary_fill_matches(
    sparse_matches: Sequence[QueryTo3DMatch],
    semidense_matches: Sequence[QueryTo3DMatch],
    config: SparsePrimaryFillConfig,
    image_width: int,
    image_height: int,
) -> list[QueryTo3DMatch]:
    sparse = list(sparse_matches)
    semi = sorted(list(semidense_matches), key=lambda match: float(match.similarity), reverse=True)
    best_sparse = _best_sparse_by_token(sparse)
    sparse_cell_counts: dict[tuple[int, int], int] = {}
    for match in sparse:
        cell = _grid_cell(match, image_width, image_height, int(config.grid_rows), int(config.grid_cols))
        sparse_cell_counts[cell] = sparse_cell_counts.get(cell, 0) + 1

    max_semi = int(np.floor(float(config.max_semidense_fraction) / max(1.0 - float(config.max_semidense_fraction), 1e-8) * max(len(sparse), 1)))
    if config.max_semidense_fraction == 0.0:
        max_semi = 0
    selected = []
    per_token: dict[int, int] = {}
    per_source: dict[int, int] = {}
    per_cell: dict[tuple[int, int], int] = {}
    for match in semi:
        if len(selected) >= max_semi:
            break
        token = int(match.token_index)
        source_track = abs(int(match.track_id))
        sparse_match = best_sparse.get(token)
        allow = False
        if config.mode == "all":
            allow = True
        elif config.mode == "no_sparse":
            allow = sparse_match is None
        elif config.mode == "low_margin":
            margin = None if sparse_match is None else sparse_match.similarity_margin
            allow = sparse_match is None or margin is None or float(margin) < float(config.low_margin_threshold)
        elif config.mode == "low_coverage":
            cell = _grid_cell(match, image_width, image_height, int(config.grid_rows), int(config.grid_cols))
            allow = sparse_cell_counts.get(cell, 0) < int(config.min_sparse_per_cell)
        if not allow:
            continue
        if per_token.get(token, 0) >= int(config.max_semidense_per_token):
            continue
        if per_source.get(source_track, 0) >= int(config.max_semidense_per_source_track):
            continue
        cell = _grid_cell(match, image_width, image_height, int(config.grid_rows), int(config.grid_cols))
        if config.max_semidense_per_cell is not None and per_cell.get(cell, 0) >= int(config.max_semidense_per_cell):
            continue
        selected.append(match)
        per_token[token] = per_token.get(token, 0) + 1
        per_source[source_track] = per_source.get(source_track, 0) + 1
        per_cell[cell] = per_cell.get(cell, 0) + 1
    return sparse + selected


def apply_semidense_context_to_sparse_matches(
    sparse_matches: Sequence[QueryTo3DMatch],
    query_feature_map: np.ndarray,
    semidense_index: LandmarkMapIndex,
    radius_m: float = 0.10,
    context_weight: float = 0.10,
    keep_fraction: float | None = None,
) -> list[QueryTo3DMatch]:
    if not sparse_matches or len(semidense_index) == 0:
        return list(sparse_matches)
    features, valid = normalize_rows(semidense_index.features)
    xyz = semidense_index.xyz[valid]
    features = features[valid]
    if xyz.shape[0] == 0:
        return list(sparse_matches)
    query = np.asarray(query_feature_map, dtype=np.float32)
    channels, height, width = query.shape
    query_rows = query.reshape(channels, height * width).T
    query_rows, _query_valid = normalize_rows(query_rows)
    tree = cKDTree(xyz)
    rescored = []
    for match in sparse_matches:
        neighbor_ids = tree.query_ball_point(np.asarray(match.xyz, dtype=np.float64).reshape(3), r=float(radius_m))
        if neighbor_ids:
            query_feature = query_rows[int(match.token_index)]
            context_similarity = float(np.max(features[np.asarray(neighbor_ids, dtype=np.int64)] @ query_feature))
        else:
            context_similarity = 0.0
        score = float(match.similarity) + float(context_weight) * context_similarity
        rescored.append(replace(match, pnp_soft_score=score, quality_weighted_similarity=score))
    rescored.sort(key=lambda item: float(item.pnp_soft_score if item.pnp_soft_score is not None else item.similarity), reverse=True)
    if keep_fraction is not None:
        keep_count = max(4, int(np.ceil(len(rescored) * float(keep_fraction))))
        rescored = rescored[:keep_count]
    return rescored
