"""Two-stage localization against VFM-aligned 2DGS surface maps.

RADIO final descriptors operate at region/maplet scale.  A separate local
descriptor bank resolves metric surface anchors.  Ambiguous anchor identities
remain grouped during pose generation, and pose selection uses groups that were
not used to generate the hypothesis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap, VfmSurfaceMapletBank
from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsObservationBank


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("descriptor array must have shape (N, C)")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), float(eps))


def _softmax_with_null(logits: np.ndarray, null_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(logits, dtype=np.float64)
    null = np.asarray(null_logits, dtype=np.float64).reshape(-1, 1)
    joint = np.concatenate([candidate, null], axis=1)
    joint -= np.max(joint, axis=1, keepdims=True)
    probabilities = np.exp(joint)
    probabilities /= np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-12)
    return probabilities[:, :-1].astype(np.float32), probabilities[:, -1].astype(np.float32)


@dataclass(frozen=True)
class SurfaceMapletMatchConfig:
    top_k: int = 8
    descriptor_temperature: float = 0.08
    quality_prior_weight: float = 0.05
    variance_penalty_weight: float = 0.10
    null_logit: float = 0.0
    minimum_layout_pairs: int = 3
    layout_inlier_threshold: float = 0.06
    layout_logit_weight: float = 2.0
    maximum_layout_models: int = 4096
    maximum_support_views: int = 64
    maximum_layout_candidate_views: int = 32
    maximum_layout_modes: int = 4
    enable_support_layout: bool = True

    def __post_init__(self) -> None:
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if float(self.descriptor_temperature) <= 0.0:
            raise ValueError("descriptor_temperature must be positive")
        if int(self.minimum_layout_pairs) < 2:
            raise ValueError("minimum_layout_pairs must be at least two")
        if float(self.layout_inlier_threshold) <= 0.0:
            raise ValueError("layout_inlier_threshold must be positive")
        if int(self.maximum_layout_models) <= 0:
            raise ValueError("maximum_layout_models must be positive")
        if (
            int(self.maximum_support_views) <= 0
            or int(self.maximum_layout_candidate_views) <= 0
            or int(self.maximum_layout_modes) <= 0
        ):
            raise ValueError("support-view/layout-mode counts must be positive")


@dataclass(frozen=True)
class SurfaceMapletMatchResult:
    candidate_maplet_ids: np.ndarray
    candidate_logits: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    selected_maplet_ids: np.ndarray
    layout_residuals: np.ndarray
    support_view_id: str | None
    support_view_score: float
    support_transform_matrix: np.ndarray | None = None
    support_transform_translation: np.ndarray | None = None
    support_mode_view_ids: tuple[str, ...] = ()
    support_mode_scores: np.ndarray | None = None
    support_mode_matrices: np.ndarray | None = None
    support_mode_translations: np.ndarray | None = None

    def __post_init__(self) -> None:
        candidate_ids = np.asarray(self.candidate_maplet_ids, dtype=np.int64)
        logits = np.asarray(self.candidate_logits, dtype=np.float32)
        probabilities = np.asarray(self.candidate_probabilities, dtype=np.float32)
        if candidate_ids.ndim != 2 or logits.shape != candidate_ids.shape or probabilities.shape != candidate_ids.shape:
            raise ValueError("candidate maplet tensors must share shape (Q, K)")
        query_count = int(candidate_ids.shape[0])
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        selected = np.asarray(self.selected_maplet_ids, dtype=np.int64).reshape(-1)
        residuals = np.asarray(self.layout_residuals, dtype=np.float32)
        if null.shape != (query_count,) or selected.shape != (query_count,):
            raise ValueError("query-level maplet outputs have the wrong shape")
        if residuals.shape != candidate_ids.shape:
            raise ValueError("layout_residuals must have shape (Q, K)")
        object.__setattr__(self, "candidate_maplet_ids", candidate_ids)
        object.__setattr__(self, "candidate_logits", logits)
        object.__setattr__(self, "candidate_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "selected_maplet_ids", selected)
        object.__setattr__(self, "layout_residuals", residuals)
        if self.support_transform_matrix is not None:
            matrix = np.asarray(self.support_transform_matrix, dtype=np.float64)
            translation = np.asarray(self.support_transform_translation, dtype=np.float64).reshape(-1)
            if matrix.shape != (2, 2) or translation.shape != (2,):
                raise ValueError("support layout transform must have shapes (2,2) and (2,)")
            object.__setattr__(self, "support_transform_matrix", matrix)
            object.__setattr__(self, "support_transform_translation", translation)
        elif self.support_transform_translation is not None:
            raise ValueError("support transform translation requires a matrix")
        mode_ids = tuple(str(value) for value in self.support_mode_view_ids)
        mode_count = len(mode_ids)
        mode_scores = np.asarray(
            (
                np.zeros((0,), dtype=np.float32)
                if self.support_mode_scores is None
                else self.support_mode_scores
            ),
            dtype=np.float32,
        ).reshape(-1)
        mode_matrices = np.asarray(
            (
                np.zeros((0, 2, 2), dtype=np.float64)
                if self.support_mode_matrices is None
                else self.support_mode_matrices
            ),
            dtype=np.float64,
        )
        mode_translations = np.asarray(
            (
                np.zeros((0, 2), dtype=np.float64)
                if self.support_mode_translations is None
                else self.support_mode_translations
            ),
            dtype=np.float64,
        )
        if (
            mode_scores.shape != (mode_count,)
            or mode_matrices.shape != (mode_count, 2, 2)
            or mode_translations.shape != (mode_count, 2)
        ):
            raise ValueError("support layout modes have incompatible shapes")
        object.__setattr__(self, "support_mode_view_ids", mode_ids)
        object.__setattr__(self, "support_mode_scores", mode_scores)
        object.__setattr__(self, "support_mode_matrices", mode_matrices)
        object.__setattr__(self, "support_mode_translations", mode_translations)


def _maplet_view_lookup(bank: VfmSurfaceMapletBank) -> tuple[list[dict[str, int]], dict[str, list[int]]]:
    per_maplet: list[dict[str, int]] = []
    rows_by_image: dict[str, list[int]] = {}
    for maplet_row in range(len(bank)):
        lookup: dict[str, int] = {}
        for view_row in range(int(bank.view_offsets[maplet_row]), int(bank.view_offsets[maplet_row + 1])):
            image_id = str(bank.view_image_ids[view_row])
            previous = lookup.get(image_id)
            if previous is None or float(bank.view_quality_scores[view_row]) > float(bank.view_quality_scores[previous]):
                lookup[image_id] = view_row
            rows_by_image.setdefault(image_id, []).append(maplet_row)
        per_maplet.append(lookup)
    return per_maplet, rows_by_image


def _normalized_xy(xy: np.ndarray, grid_size: Sequence[int]) -> np.ndarray:
    width, height = int(grid_size[0]), int(grid_size[1])
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.asarray(xy, dtype=np.float32) / scale


def _similarity_transform_from_pairs(
    query_a: np.ndarray,
    query_b: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    query_delta = np.asarray(query_b, dtype=np.float64) - np.asarray(query_a, dtype=np.float64)
    support_delta = np.asarray(support_b, dtype=np.float64) - np.asarray(support_a, dtype=np.float64)
    denominator = float(np.dot(query_delta, query_delta))
    if denominator <= 1e-8 or float(np.dot(support_delta, support_delta)) <= 1e-8:
        return None
    a = float(np.dot(query_delta, support_delta) / denominator)
    b = float((query_delta[0] * support_delta[1] - query_delta[1] * support_delta[0]) / denominator)
    matrix = np.asarray([[a, -b], [b, a]], dtype=np.float64)
    translation = np.asarray(support_a, dtype=np.float64) - matrix @ np.asarray(query_a, dtype=np.float64)
    return matrix, translation


def _fit_similarity_transform(
    query_points: np.ndarray,
    support_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares 2D similarity fit for an already robust correspondence set."""

    query = np.asarray(query_points, dtype=np.float64).reshape(-1, 2)
    support = np.asarray(support_points, dtype=np.float64).reshape(-1, 2)
    if len(query) < 2 or query.shape != support.shape:
        return None
    query_mean = np.mean(query, axis=0)
    support_mean = np.mean(support, axis=0)
    centered_query = query - query_mean
    centered_support = support - support_mean
    denominator = float(np.sum(centered_query * centered_query))
    if denominator <= 1e-8:
        return None
    a = float(
        np.sum(
            centered_query[:, 0] * centered_support[:, 0]
            + centered_query[:, 1] * centered_support[:, 1]
        )
        / denominator
    )
    b = float(
        np.sum(
            centered_query[:, 0] * centered_support[:, 1]
            - centered_query[:, 1] * centered_support[:, 0]
        )
        / denominator
    )
    matrix = np.asarray([[a, -b], [b, a]], dtype=np.float64)
    translation = support_mean - matrix @ query_mean
    return matrix, translation


def _best_support_layout(
    query_xy_normalized: np.ndarray,
    candidate_rows: np.ndarray,
    base_scores: np.ndarray,
    bank: VfmSurfaceMapletBank,
    per_maplet_views: list[dict[str, int]],
    config: SurfaceMapletMatchConfig,
) -> tuple[
    str | None,
    np.ndarray | None,
    np.ndarray | None,
    float,
    tuple[tuple[str, float, np.ndarray, np.ndarray], ...],
]:
    support_images = sorted(
        {
            image_id
            for row in np.unique(candidate_rows[candidate_rows >= 0]).tolist()
            for image_id in per_maplet_views[int(row)]
        }
    )
    best_image: str | None = None
    best_matrix = None
    best_translation = None
    best_score = -np.inf
    modes: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    evidence_by_image: dict[str, tuple[list[int], list[np.ndarray], list[float]]] = {}
    for image_id in support_images:
        # A maplet has one representative location in a support view.  Allowing
        # several query regions to claim that same location creates a degenerate
        # layout model, so retain only the strongest query-to-maplet claim.
        best_by_maplet: dict[int, tuple[float, int, int]] = {}
        for query_row in range(candidate_rows.shape[0]):
            for column in range(candidate_rows.shape[1]):
                maplet_row = int(candidate_rows[query_row, column])
                if maplet_row < 0 or image_id not in per_maplet_views[maplet_row]:
                    continue
                view_row = per_maplet_views[maplet_row][image_id]
                claim = (float(base_scores[query_row, column]), query_row, view_row)
                previous = best_by_maplet.get(maplet_row)
                if previous is None or claim[0] > previous[0]:
                    best_by_maplet[maplet_row] = claim
        query_rows: list[int] = []
        support_points: list[np.ndarray] = []
        similarities: list[float] = []
        for maplet_row in sorted(best_by_maplet):
            similarity, query_row, view_row = best_by_maplet[maplet_row]
            query_rows.append(query_row)
            support_points.append(
                _normalized_xy(bank.view_token_xy[view_row], bank.view_grid_sizes[view_row])
            )
            similarities.append(similarity)
        evidence_by_image[image_id] = (query_rows, support_points, similarities)
    support_images = sorted(
        support_images,
        key=lambda image_id: (
            -len(evidence_by_image[image_id][0]),
            -float(np.mean(evidence_by_image[image_id][2]))
            if evidence_by_image[image_id][2]
            else float("inf"),
            image_id,
        ),
    )[
        : min(
            int(config.maximum_support_views),
            int(config.maximum_layout_candidate_views),
        )
    ]
    for image_id in support_images:
        query_rows, support_points, similarities = evidence_by_image[image_id]
        if len(query_rows) < int(config.minimum_layout_pairs):
            continue
        local_best_score = -np.inf
        local_best_matrix = None
        local_best_translation = None
        model_count = 0
        query_points = query_xy_normalized[np.asarray(query_rows, dtype=np.int64)]
        support = np.stack(support_points, axis=0).astype(np.float64)
        similarity_values = np.asarray(similarities, dtype=np.float64)
        per_view_model_budget = max(
            128,
            int(config.maximum_layout_models)
            // max(len(support_images), 1),
        )
        for left, right in combinations(range(len(query_rows)), 2):
            model_count += 1
            if model_count > per_view_model_budget:
                break
            transform = _similarity_transform_from_pairs(
                query_points[left], query_points[right], support[left], support[right]
            )
            if transform is None:
                continue
            matrix, translation = transform
            predicted = query_points @ matrix.T + translation
            residual = np.linalg.norm(predicted - support, axis=1)
            inliers = residual <= float(config.layout_inlier_threshold)
            if int(np.sum(inliers)) < int(config.minimum_layout_pairs):
                continue
            refined = _fit_similarity_transform(query_points[inliers], support[inliers])
            if refined is not None:
                matrix, translation = refined
                predicted = query_points @ matrix.T + translation
                residual = np.linalg.norm(predicted - support, axis=1)
                inliers = residual <= float(config.layout_inlier_threshold)
                if int(np.sum(inliers)) < int(config.minimum_layout_pairs):
                    continue
            coverage = float(np.mean(inliers))
            score = (
                float(np.mean(similarity_values[inliers]))
                + 0.5 * coverage
                - float(np.mean(residual[inliers])) / float(config.layout_inlier_threshold)
            )
            if score > best_score:
                best_score = score
                best_image = image_id
                best_matrix = matrix
                best_translation = translation
            if score > local_best_score:
                local_best_score = score
                local_best_matrix = matrix
                local_best_translation = translation
        if (
            local_best_matrix is not None
            and local_best_translation is not None
        ):
            modes.append(
                (
                    str(image_id),
                    float(local_best_score),
                    np.asarray(local_best_matrix, dtype=np.float64),
                    np.asarray(local_best_translation, dtype=np.float64),
                )
            )
    modes.sort(key=lambda item: (-item[1], item[0]))
    modes = modes[: int(config.maximum_layout_modes)]
    return (
        best_image,
        best_matrix,
        best_translation,
        float(best_score),
        tuple(modes),
    )


def match_radio_final_regions_to_maplets(
    query_region_xy: np.ndarray,
    query_descriptors: np.ndarray,
    query_grid_size: tuple[int, int],
    bank: VfmSurfaceMapletBank,
    config: SurfaceMapletMatchConfig = SurfaceMapletMatchConfig(),
) -> SurfaceMapletMatchResult:
    """Retrieve maplets and enforce whole-image support-layout consistency."""

    query_xy = np.asarray(query_region_xy, dtype=np.float32)
    descriptors = _normalize_rows(query_descriptors)
    if query_xy.shape != (descriptors.shape[0], 2):
        raise ValueError("query_region_xy must have shape (Q, 2)")
    if descriptors.shape[1] != bank.feature_dim:
        raise ValueError("query and maplet descriptor dimensions differ")
    query_count = int(descriptors.shape[0])
    keep = min(int(config.top_k), len(bank))
    if keep <= 0:
        empty = np.zeros((query_count, 0), dtype=np.float32)
        return SurfaceMapletMatchResult(
            candidate_maplet_ids=np.zeros((query_count, 0), dtype=np.int64),
            candidate_logits=empty,
            candidate_probabilities=empty,
            null_probabilities=np.ones((query_count,), dtype=np.float32),
            selected_maplet_ids=np.full((query_count,), -1, dtype=np.int64),
            layout_residuals=empty,
            support_view_id=None,
            support_view_score=float("-inf"),
            support_transform_matrix=None,
            support_transform_translation=None,
        )
    cosine = descriptors @ bank.descriptors.T
    quality = np.log(np.maximum(bank.quality_scores, 1e-8))[None, :]
    variance = bank.descriptor_variances[None, :]
    logits_full = (
        cosine / float(config.descriptor_temperature)
        + float(config.quality_prior_weight) * quality
        - float(config.variance_penalty_weight) * variance
    )
    columns = np.argpartition(-logits_full, kth=keep - 1, axis=1)[:, :keep]
    top_logits = np.take_along_axis(logits_full, columns, axis=1)
    order = np.argsort(-top_logits, axis=1, kind="mergesort")
    columns = np.take_along_axis(columns, order, axis=1)
    base_logits = np.take_along_axis(logits_full, columns, axis=1)
    candidate_ids = bank.maplet_ids[columns]

    layout_residuals = np.full(candidate_ids.shape, np.inf, dtype=np.float32)
    final_logits = np.asarray(base_logits, dtype=np.float64).copy()
    support_view: str | None = None
    matrix: np.ndarray | None = None
    translation: np.ndarray | None = None
    support_score = float("-inf")
    support_modes: tuple[
        tuple[str, float, np.ndarray, np.ndarray], ...
    ] = ()
    if bool(config.enable_support_layout):
        per_maplet_views, _rows_by_image = _maplet_view_lookup(bank)
        query_normalized = _normalized_xy(query_xy, query_grid_size)
        (
            support_view,
            matrix,
            translation,
            support_score,
            support_modes,
        ) = _best_support_layout(
            query_normalized,
            columns,
            base_logits,
            bank,
            per_maplet_views,
            config,
        )
        if (
            support_view is not None
            and matrix is not None
            and translation is not None
        ):
            predicted = query_normalized @ matrix.T + translation
            for query_row in range(query_count):
                for column in range(keep):
                    maplet_row = int(columns[query_row, column])
                    view_row = per_maplet_views[maplet_row].get(support_view)
                    if view_row is None:
                        continue
                    support_xy = _normalized_xy(
                        bank.view_token_xy[view_row],
                        bank.view_grid_sizes[view_row],
                    )
                    residual = float(
                        np.linalg.norm(predicted[query_row] - support_xy)
                    )
                    layout_residuals[query_row, column] = residual
                    compatibility = max(
                        0.0,
                        1.0
                        - residual / float(config.layout_inlier_threshold),
                    )
                    final_logits[query_row, column] += (
                        float(config.layout_logit_weight) * compatibility
                    )
    null_logits = np.full((query_count,), float(config.null_logit), dtype=np.float64)
    probabilities, null_probabilities = _softmax_with_null(final_logits, null_logits)
    best_columns = np.argmax(probabilities, axis=1)
    best_probability = probabilities[np.arange(query_count), best_columns]
    selected = candidate_ids[np.arange(query_count), best_columns].copy()
    selected[best_probability <= null_probabilities] = -1
    return SurfaceMapletMatchResult(
        candidate_maplet_ids=candidate_ids,
        candidate_logits=final_logits.astype(np.float32),
        candidate_probabilities=probabilities,
        null_probabilities=null_probabilities,
        selected_maplet_ids=selected,
        layout_residuals=layout_residuals,
        support_view_id=support_view,
        support_view_score=support_score,
        support_transform_matrix=matrix,
        support_transform_translation=translation,
        support_mode_view_ids=tuple(item[0] for item in support_modes),
        support_mode_scores=np.asarray(
            [item[1] for item in support_modes], dtype=np.float32
        ),
        support_mode_matrices=(
            np.stack([item[2] for item in support_modes])
            if support_modes
            else np.zeros((0, 2, 2), dtype=np.float64)
        ),
        support_mode_translations=(
            np.stack([item[3] for item in support_modes])
            if support_modes
            else np.zeros((0, 2), dtype=np.float64)
        ),
    )


@dataclass(frozen=True)
class LocalFeatureFrame:
    image_id: str
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_xy, dtype=np.float32)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError("keypoints_xy must have shape (N, 2)")
        if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
            raise ValueError("descriptors must have shape (N, C)")
        scores = (
            np.ones((keypoints.shape[0],), dtype=np.float32)
            if self.scores is None
            else np.asarray(self.scores, dtype=np.float32).reshape(-1)
        )
        if scores.shape != (keypoints.shape[0],):
            raise ValueError("scores must have shape (N,)")
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "keypoints_xy", keypoints)
        object.__setattr__(self, "descriptors", _normalize_rows(descriptors))
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class AnchorLocalDescriptorBank:
    anchor_ids: np.ndarray
    descriptor_offsets: np.ndarray
    descriptors: np.ndarray
    support_image_ids: tuple[str, ...]
    descriptor_quality: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if descriptors.ndim != 2:
            raise ValueError("descriptors must have shape (M, C)")
        offsets = np.asarray(self.descriptor_offsets, dtype=np.int64).reshape(-1)
        if offsets.shape != (anchor_ids.shape[0] + 1,):
            raise ValueError("descriptor_offsets must have shape (N + 1,)")
        if int(offsets[0]) != 0 or int(offsets[-1]) != descriptors.shape[0] or np.any(np.diff(offsets) < 0):
            raise ValueError("descriptor_offsets is invalid")
        image_ids = tuple(str(value) for value in self.support_image_ids)
        if len(image_ids) != descriptors.shape[0]:
            raise ValueError("support_image_ids must align with descriptors")
        quality = np.asarray(self.descriptor_quality, dtype=np.float32).reshape(-1)
        if quality.shape != (descriptors.shape[0],):
            raise ValueError("descriptor_quality must align with descriptors")
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "descriptor_offsets", offsets)
        object.__setattr__(self, "descriptors", _normalize_rows(descriptors))
        object.__setattr__(self, "support_image_ids", image_ids)
        object.__setattr__(self, "descriptor_quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(self.anchor_ids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.descriptors.shape[1])

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchor_ids=self.anchor_ids,
            descriptor_offsets=self.descriptor_offsets,
            descriptors=self.descriptors,
            descriptor_quality=self.descriptor_quality,
            support_image_ids_json=np.asarray(json.dumps(list(self.support_image_ids))),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "AnchorLocalDescriptorBank":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                descriptor_offsets=np.asarray(data["descriptor_offsets"], dtype=np.int64),
                descriptors=np.asarray(data["descriptors"], dtype=np.float32),
                support_image_ids=tuple(json.loads(str(np.asarray(data["support_image_ids_json"]).item()))),
                descriptor_quality=np.asarray(data["descriptor_quality"], dtype=np.float32),
                metadata=(
                    json.loads(str(np.asarray(data["metadata_json"]).item()))
                    if "metadata_json" in data
                    else {}
                ),
            )


def build_anchor_local_descriptor_bank(
    anchors: StableSurfaceAnchorMap,
    support_frames: Mapping[str, LocalFeatureFrame],
    maximum_pixel_distance: float = 3.0,
    minimum_observations: int = 2,
    maximum_prototypes: int = 8,
) -> AnchorLocalDescriptorBank:
    """Attach real-image local features to projected 2DGS surface identities."""

    if float(maximum_pixel_distance) <= 0.0:
        raise ValueError("maximum_pixel_distance must be positive")
    if int(minimum_observations) <= 0 or int(maximum_prototypes) <= 0:
        raise ValueError("observation/prototype counts must be positive")
    descriptor_dim = 0
    trees: dict[str, cKDTree] = {}
    for image_id, frame in support_frames.items():
        descriptor_dim = int(frame.descriptors.shape[1])
        trees[str(image_id)] = cKDTree(frame.keypoints_xy)
    output_anchor_ids: list[int] = []
    offsets = [0]
    descriptors: list[np.ndarray] = []
    image_ids: list[str] = []
    quality: list[float] = []
    for anchor_row, anchor_id in enumerate(anchors.anchor_ids.tolist()):
        candidates = []
        start, end = int(anchors.observation_offsets[anchor_row]), int(anchors.observation_offsets[anchor_row + 1])
        for observation_row in range(start, end):
            image_id = anchors.observation_image_ids[observation_row]
            frame = support_frames.get(image_id)
            tree = trees.get(image_id)
            if frame is None or tree is None or len(frame.keypoints_xy) == 0:
                continue
            distance, keypoint_row = tree.query(anchors.observation_xy[observation_row], k=1)
            keypoint_row = int(keypoint_row)
            if not np.isfinite(distance) or float(distance) > float(maximum_pixel_distance):
                continue
            score = (
                float(anchors.observation_weights[observation_row])
                * float(frame.scores[keypoint_row])
                * np.exp(-0.5 * (float(distance) / float(maximum_pixel_distance)) ** 2)
            )
            candidates.append((score, image_id, frame.descriptors[keypoint_row]))
        if len(candidates) < int(minimum_observations):
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen = candidates[: int(maximum_prototypes)]
        output_anchor_ids.append(int(anchor_id))
        for score, image_id, descriptor in chosen:
            descriptors.append(np.asarray(descriptor, dtype=np.float32))
            image_ids.append(str(image_id))
            quality.append(float(score))
        offsets.append(len(descriptors))
    return AnchorLocalDescriptorBank(
        anchor_ids=np.asarray(output_anchor_ids, dtype=np.int64),
        descriptor_offsets=np.asarray(offsets, dtype=np.int64),
        descriptors=(
            np.stack(descriptors, axis=0).astype(np.float32)
            if descriptors
            else np.zeros((0, descriptor_dim), dtype=np.float32)
        ),
        support_image_ids=tuple(image_ids),
        descriptor_quality=np.asarray(quality, dtype=np.float32),
        metadata={
            "representation": "2dgs_surface_anchor_local_descriptors",
            "feature_role": "pixel_metric_refinement",
            "maximum_pixel_distance": float(maximum_pixel_distance),
            "minimum_observations": int(minimum_observations),
            "maximum_prototypes": int(maximum_prototypes),
            "uses_radio_intermediate": False,
            "uses_sfm_tracks": False,
        },
    )


@dataclass(frozen=True)
class SurfaceAnchorCandidatePool:
    query_xy: np.ndarray
    anchor_ids: np.ndarray
    xyz: np.ndarray
    descriptor_scores: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        query_xy = np.asarray(self.query_xy, dtype=np.float32)
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        scores = np.asarray(self.descriptor_scores, dtype=np.float32)
        probabilities = np.asarray(self.candidate_probabilities, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if query_xy.ndim != 2 or query_xy.shape[1] != 2:
            raise ValueError("query_xy must have shape (Q, 2)")
        if anchor_ids.ndim != 2:
            raise ValueError("anchor_ids must have shape (Q, L)")
        if anchor_ids.shape[0] != query_xy.shape[0] or xyz.shape != anchor_ids.shape + (3,):
            raise ValueError("candidate geometry has an incompatible shape")
        if scores.shape != anchor_ids.shape or probabilities.shape != anchor_ids.shape or valid.shape != anchor_ids.shape:
            raise ValueError("candidate tensors must share shape (Q, L)")
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        if null.shape != (query_xy.shape[0],):
            raise ValueError("null_probabilities must have shape (Q,)")
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "descriptor_scores", scores)
        object.__setattr__(self, "candidate_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "valid_mask", valid)

    def __len__(self) -> int:
        return int(self.query_xy.shape[0])


@dataclass(frozen=True)
class VfmSurfaceObservationIndex:
    """Query-independent RADIO-final index over mapped 2DGS observations."""

    descriptors: np.ndarray
    image_ids: tuple[str, ...]
    row_offsets: np.ndarray
    rows: np.ndarray
    global_descriptors: np.ndarray

    @classmethod
    def from_bank(
        cls,
        observation_bank: Vfm2DgsObservationBank,
    ) -> "VfmSurfaceObservationIndex":
        descriptors = _normalize_rows(observation_bank.features)
        rows_by_image: dict[str, list[int]] = {}
        for row, image_id in enumerate(observation_bank.image_ids):
            rows_by_image.setdefault(str(image_id), []).append(int(row))
        image_ids = tuple(sorted(rows_by_image))
        offsets = [0]
        rows: list[int] = []
        global_descriptors: list[np.ndarray] = []
        for image_id in image_ids:
            image_rows = rows_by_image[image_id]
            rows.extend(image_rows)
            offsets.append(len(rows))
            global_descriptors.append(
                _normalize_rows(
                    np.mean(
                        descriptors[np.asarray(image_rows, dtype=np.int64)],
                        axis=0,
                        keepdims=True,
                    )
                )[0]
            )
        return cls(
            descriptors=descriptors,
            image_ids=image_ids,
            row_offsets=np.asarray(offsets, dtype=np.int64),
            rows=np.asarray(rows, dtype=np.int64),
            global_descriptors=np.stack(global_descriptors).astype(np.float32),
        )

    def rows_for_image(self, image_id: str) -> np.ndarray:
        try:
            image_row = self.image_ids.index(str(image_id))
        except ValueError:
            return np.zeros((0,), dtype=np.int64)
        return self.rows[
            int(self.row_offsets[image_row]) : int(self.row_offsets[image_row + 1])
        ]


def build_vfm_surface_observation_candidate_pool(
    query_feature_map: np.ndarray,
    query_image_size: tuple[int, int],
    observation_bank: Vfm2DgsObservationBank,
    maximum_global_support_views: int = 64,
    maximum_support_views: int = 8,
    maximum_query_points: int = 768,
    top_l: int = 5,
    minimum_similarity: float = 0.20,
    descriptor_temperature: float = 0.08,
    null_logit: float = 0.0,
    observation_index: VfmSurfaceObservationIndex | None = None,
) -> tuple[SurfaceAnchorCandidatePool, tuple[str, ...], np.ndarray]:
    """Match dense RADIO-final tokens directly to 2DGS surface observations.

    Support images are shortlisted by global VFM context, reranked by
    support-to-query token coverage, and contribute only mutual-nearest token
    matches.  Each 3D candidate is a rendered 2DGS surface center rather than
    an SfM point or a Gaussian index.
    """

    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim == 4 and int(feature_map.shape[0]) == 1:
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C,H,W)")
    channels, grid_height, grid_width = feature_map.shape
    if channels != observation_bank.features.shape[1]:
        raise ValueError("query and surface-observation descriptor dimensions differ")
    if (
        int(maximum_global_support_views) <= 0
        or int(maximum_support_views) <= 0
        or int(maximum_query_points) <= 0
        or int(top_l) <= 0
    ):
        raise ValueError("VFM candidate-pool limits must be positive")
    if float(descriptor_temperature) <= 0.0:
        raise ValueError("descriptor_temperature must be positive")

    query_descriptors = _normalize_rows(
        feature_map.transpose(1, 2, 0).reshape(-1, channels)
    )
    index = (
        VfmSurfaceObservationIndex.from_bank(observation_bank)
        if observation_index is None
        else observation_index
    )
    if len(index.descriptors) != len(observation_bank):
        raise ValueError("surface observation index and bank differ")
    if not index.image_ids:
        empty = SurfaceAnchorCandidatePool(
            query_xy=np.zeros((0, 2), dtype=np.float32),
            anchor_ids=np.zeros((0, int(top_l)), dtype=np.int64),
            xyz=np.zeros((0, int(top_l), 3), dtype=np.float64),
            descriptor_scores=np.zeros((0, int(top_l)), dtype=np.float32),
            candidate_probabilities=np.zeros((0, int(top_l)), dtype=np.float32),
            null_probabilities=np.zeros((0,), dtype=np.float32),
            valid_mask=np.zeros((0, int(top_l)), dtype=bool),
        )
        return empty, (), np.zeros((0,), dtype=np.float32)

    query_global = _normalize_rows(np.mean(query_descriptors, axis=0, keepdims=True))[0]
    global_values = index.global_descriptors @ query_global
    global_scores = [
        (float(global_values[row]), image_id)
        for row, image_id in enumerate(index.image_ids)
    ]
    global_scores.sort(key=lambda item: (-item[0], item[1]))
    shortlisted = global_scores[: int(maximum_global_support_views)]

    reranked: list[tuple[float, str, np.ndarray]] = []
    for global_score, image_id in shortlisted:
        rows = index.rows_for_image(image_id)
        similarities = query_descriptors @ index.descriptors[rows].T
        support_coverage = np.max(similarities, axis=0)
        coverage_keep = min(64, len(support_coverage))
        coverage_score = float(
            np.mean(np.partition(support_coverage, -coverage_keep)[-coverage_keep:])
        )
        rerank_score = coverage_score + 0.15 * float(global_score)
        reranked.append((rerank_score, image_id, similarities))
    reranked.sort(key=lambda item: (-item[0], item[1]))
    selected_views = reranked[: int(maximum_support_views)]

    candidates_by_query: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for view_rank, (view_score, image_id, similarities) in enumerate(selected_views):
        rows = index.rows_for_image(image_id)
        best_support_for_query = np.argmax(similarities, axis=1)
        best_query_for_support = np.argmax(similarities, axis=0)
        for support_column, query_row_value in enumerate(best_query_for_support.tolist()):
            query_row = int(query_row_value)
            if int(best_support_for_query[query_row]) != int(support_column):
                continue
            similarity = float(similarities[query_row, support_column])
            if similarity < float(minimum_similarity):
                continue
            observation_row = int(rows[support_column])
            quality = max(float(observation_bank.quality_scores[observation_row]), 1e-6)
            score = (
                similarity
                + 0.05 * float(view_score)
                + 0.01 * np.log(quality)
                - 0.002 * float(view_rank)
            )
            # Observation rows are persistent within the frozen 2DGS map and
            # serve only as candidate IDs; the metric value is the surface center.
            candidates_by_query.setdefault(query_row, []).append(
                (score, observation_row, observation_bank.centers[observation_row])
            )

    ordered_query_rows = sorted(
        candidates_by_query,
        key=lambda row: (
            -max(item[0] for item in candidates_by_query[row]),
            row,
        ),
    )[: int(maximum_query_points)]
    query_count = len(ordered_query_rows)
    output_ids = np.full((query_count, int(top_l)), -1, dtype=np.int64)
    output_xyz = np.zeros((query_count, int(top_l), 3), dtype=np.float64)
    output_scores = np.full((query_count, int(top_l)), -np.inf, dtype=np.float32)
    output_logits = np.full((query_count, int(top_l)), -np.inf, dtype=np.float64)
    valid = np.zeros((query_count, int(top_l)), dtype=bool)
    for output_row, query_row in enumerate(ordered_query_rows):
        records = sorted(
            candidates_by_query[query_row],
            key=lambda item: (-item[0], item[1]),
        )
        used_centers: list[np.ndarray] = []
        output_column = 0
        for score, observation_row, center in records:
            if any(float(np.linalg.norm(center - used)) < 0.01 for used in used_centers):
                continue
            output_ids[output_row, output_column] = int(observation_row)
            output_xyz[output_row, output_column] = center
            output_scores[output_row, output_column] = float(score)
            output_logits[output_row, output_column] = float(score) / float(
                descriptor_temperature
            )
            valid[output_row, output_column] = True
            used_centers.append(np.asarray(center, dtype=np.float64))
            output_column += 1
            if output_column >= int(top_l):
                break
    candidate_probabilities, null_probabilities = _softmax_with_null(
        output_logits,
        np.full((query_count,), float(null_logit), dtype=np.float64),
    )
    candidate_probabilities[~valid] = 0.0
    image_width, image_height = int(query_image_size[0]), int(query_image_size[1])
    query_rows_array = np.asarray(ordered_query_rows, dtype=np.int64)
    query_grid_xy = np.stack(
        [query_rows_array % grid_width, query_rows_array // grid_width],
        axis=1,
    ).astype(np.float32)
    # The 2DGS renderer scales the calibrated intrinsic matrix to the RADIO
    # grid, so the exact inverse is pixel = token * image_size / grid_size.
    query_xy = query_grid_xy * np.asarray(
        [
            image_width / max(grid_width, 1),
            image_height / max(grid_height, 1),
        ],
        dtype=np.float32,
    )
    return (
        SurfaceAnchorCandidatePool(
            query_xy=query_xy,
            anchor_ids=output_ids,
            xyz=output_xyz,
            descriptor_scores=output_scores,
            candidate_probabilities=candidate_probabilities,
            null_probabilities=null_probabilities,
            valid_mask=valid,
        ),
        tuple(item[1] for item in selected_views),
        np.asarray([item[0] for item in selected_views], dtype=np.float32),
    )


def select_vfm_surface_feature_modes(
    query_feature_map: np.ndarray,
    observation_bank: Vfm2DgsObservationBank,
    *,
    maximum_global_modes: int = 64,
    maximum_modes: int = 8,
    allowed_mode_ids: Sequence[str] | None = None,
    observation_index: VfmSurfaceObservationIndex | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Select view-conditioned RADIO-final map modes without reading images.

    Image IDs are persistent labels for appearance/visibility modes stored in
    the feature-bearing 2DGS map.  This function consumes only cached surface
    descriptors; it neither accepts nor opens mapping RGB paths.
    """

    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim == 4 and int(feature_map.shape[0]) == 1:
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C,H,W)")
    if int(maximum_global_modes) <= 0 or int(maximum_modes) <= 0:
        raise ValueError("surface feature-mode limits must be positive")
    channels = int(feature_map.shape[0])
    query_descriptors = _normalize_rows(
        feature_map.transpose(1, 2, 0).reshape(-1, channels)
    )
    index = (
        VfmSurfaceObservationIndex.from_bank(observation_bank)
        if observation_index is None
        else observation_index
    )
    if channels != int(index.descriptors.shape[1]):
        raise ValueError("query and surface feature-mode dimensions differ")
    allowed = (
        None
        if allowed_mode_ids is None
        else {str(value) for value in allowed_mode_ids}
    )
    query_global = _normalize_rows(
        np.mean(query_descriptors, axis=0, keepdims=True)
    )[0]
    global_values = index.global_descriptors @ query_global
    global_scores = [
        (float(global_values[row]), image_id)
        for row, image_id in enumerate(index.image_ids)
        if allowed is None or image_id in allowed
    ]
    global_scores.sort(key=lambda item: (-item[0], item[1]))
    shortlisted = global_scores[: int(maximum_global_modes)]
    reranked: list[tuple[float, str]] = []
    for global_score, image_id in shortlisted:
        rows = index.rows_for_image(image_id)
        if len(rows) == 0:
            continue
        similarities = query_descriptors @ index.descriptors[rows].T
        support_coverage = np.max(similarities, axis=0)
        coverage_keep = min(64, len(support_coverage))
        coverage_score = float(
            np.mean(
                np.partition(support_coverage, -coverage_keep)[
                    -coverage_keep:
                ]
            )
        )
        reranked.append(
            (coverage_score + 0.15 * float(global_score), image_id)
        )
    reranked.sort(key=lambda item: (-item[0], item[1]))
    selected = reranked[: int(maximum_modes)]
    return (
        tuple(item[1] for item in selected),
        np.asarray([item[0] for item in selected], dtype=np.float32),
    )


def estimate_vfm_query_to_support_layout(
    query_feature_map: np.ndarray,
    observation_bank: Vfm2DgsObservationBank,
    support_view_id: str,
    minimum_similarity: float = 0.20,
    ransac_threshold_tokens: float = 2.0,
    observation_index: VfmSurfaceObservationIndex | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float]:
    """Estimate query-token to support-token geometry from RADIO-final matches."""

    import cv2

    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim == 4 and int(feature_map.shape[0]) == 1:
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C,H,W)")
    channels, height, width = feature_map.shape
    query_descriptors = _normalize_rows(
        feature_map.transpose(1, 2, 0).reshape(-1, channels)
    )
    index = (
        VfmSurfaceObservationIndex.from_bank(observation_bank)
        if observation_index is None
        else observation_index
    )
    rows = index.rows_for_image(str(support_view_id))
    if len(rows) < 3:
        return None, None, 0, float("inf")
    support_descriptors = index.descriptors[rows]
    similarities = query_descriptors @ support_descriptors.T
    best_support_for_query = np.argmax(similarities, axis=1)
    best_query_for_support = np.argmax(similarities, axis=0)
    matches: list[tuple[int, int, float]] = []
    for support_column, query_row_value in enumerate(best_query_for_support.tolist()):
        query_row = int(query_row_value)
        if int(best_support_for_query[query_row]) != int(support_column):
            continue
        similarity = float(similarities[query_row, support_column])
        if similarity >= float(minimum_similarity):
            matches.append((query_row, support_column, similarity))
    if len(matches) < 3:
        return None, None, len(matches), float("inf")
    query_rows = np.asarray([item[0] for item in matches], dtype=np.int64)
    query_xy = np.stack(
        [query_rows % int(width), query_rows // int(width)],
        axis=1,
    ).astype(np.float64)
    support_xy = observation_bank.token_xy[
        rows[np.asarray([item[1] for item in matches], dtype=np.int64)]
    ].astype(np.float64)
    # A full affine model captures the anisotropic foreshortening between
    # nearby mapping/query views; RANSAC still prevents repeated VFM regions
    # from defining the layout.
    affine, inlier_mask = cv2.estimateAffine2D(
        query_xy,
        support_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_tokens),
        maxIters=4000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None:
        return None, None, 0, float("inf")
    matrix = np.asarray(affine[:, :2], dtype=np.float64)
    translation = np.asarray(affine[:, 2], dtype=np.float64)
    predicted = query_xy @ matrix.T + translation
    residuals = np.linalg.norm(predicted - support_xy, axis=1)
    inliers = (
        np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
        if inlier_mask is not None
        else residuals <= float(ransac_threshold_tokens)
    )
    return (
        matrix,
        translation,
        int(np.sum(inliers)),
        float(np.median(residuals[inliers])) if np.any(inliers) else float("inf"),
    )


def build_surface_anchor_candidate_pool(
    query: LocalFeatureFrame,
    candidate_anchor_ids: Sequence[int],
    descriptor_bank: AnchorLocalDescriptorBank,
    anchors: StableSurfaceAnchorMap,
    top_l: int = 5,
    descriptor_temperature: float = 0.08,
    null_logit: float = 0.0,
) -> SurfaceAnchorCandidatePool:
    """Create grouped, mutually exclusive surface-anchor candidates."""

    if int(top_l) <= 0 or float(descriptor_temperature) <= 0.0:
        raise ValueError("top_l and descriptor_temperature must be positive")
    if query.descriptors.shape[1] != descriptor_bank.feature_dim:
        raise ValueError("query and support local descriptor dimensions differ")
    requested = {int(value) for value in candidate_anchor_ids}
    anchor_row_by_id = anchors.row_by_id()
    bank_rows = [
        row
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
        if int(anchor_id) in requested and int(anchor_id) in anchor_row_by_id
    ]
    query_count = int(query.keypoints_xy.shape[0])
    keep = min(int(top_l), len(bank_rows))
    if keep <= 0:
        return SurfaceAnchorCandidatePool(
            query_xy=query.keypoints_xy,
            anchor_ids=np.zeros((query_count, 0), dtype=np.int64),
            xyz=np.zeros((query_count, 0, 3), dtype=np.float64),
            descriptor_scores=np.zeros((query_count, 0), dtype=np.float32),
            candidate_probabilities=np.zeros((query_count, 0), dtype=np.float32),
            null_probabilities=np.ones((query_count,), dtype=np.float32),
            valid_mask=np.zeros((query_count, 0), dtype=bool),
        )
    per_anchor_scores = np.full((query_count, len(bank_rows)), -1.0, dtype=np.float32)
    for output_column, bank_row in enumerate(bank_rows):
        start, end = (
            int(descriptor_bank.descriptor_offsets[bank_row]),
            int(descriptor_bank.descriptor_offsets[bank_row + 1]),
        )
        if start == end:
            continue
        similarities = query.descriptors @ descriptor_bank.descriptors[start:end].T
        per_anchor_scores[:, output_column] = np.max(similarities, axis=1)
    columns = np.argpartition(-per_anchor_scores, kth=keep - 1, axis=1)[:, :keep]
    top_scores = np.take_along_axis(per_anchor_scores, columns, axis=1)
    order = np.argsort(-top_scores, axis=1, kind="mergesort")
    columns = np.take_along_axis(columns, order, axis=1)
    top_scores = np.take_along_axis(per_anchor_scores, columns, axis=1)
    bank_rows_array = np.asarray(bank_rows, dtype=np.int64)
    selected_bank_rows = bank_rows_array[columns]
    selected_anchor_ids = descriptor_bank.anchor_ids[selected_bank_rows]
    anchor_rows = np.vectorize(lambda value: anchor_row_by_id[int(value)], otypes=[np.int64])(selected_anchor_ids)
    xyz = anchors.xyz[anchor_rows]
    candidate_logits = top_scores / float(descriptor_temperature)
    probabilities, null_probabilities = _softmax_with_null(
        candidate_logits,
        np.full((query_count,), float(null_logit), dtype=np.float32),
    )
    valid = np.isfinite(top_scores)
    probabilities[~valid] = 0.0
    return SurfaceAnchorCandidatePool(
        query_xy=query.keypoints_xy,
        anchor_ids=selected_anchor_ids,
        xyz=xyz,
        descriptor_scores=top_scores,
        candidate_probabilities=probabilities,
        null_probabilities=null_probabilities,
        valid_mask=valid,
    )


def build_maplet_conditioned_surface_anchor_candidate_pool(
    query: LocalFeatureFrame,
    query_image_size: tuple[int, int],
    query_region_xy: np.ndarray,
    query_region_grid_size: tuple[int, int],
    maplet_match: SurfaceMapletMatchResult,
    maplets: VfmSurfaceMapletBank,
    descriptor_bank: AnchorLocalDescriptorBank,
    anchors: StableSurfaceAnchorMap,
    top_l: int = 5,
    maximum_maplets_per_region: int = 3,
    minimum_maplet_probability: float = 0.0,
    descriptor_temperature: float = 0.08,
    maplet_prior_weight: float = 0.25,
    support_spatial_sigma: float = 0.03,
    maximum_support_distance: float = 0.15,
    null_logit: float = 0.0,
) -> SurfaceAnchorCandidatePool:
    """Condition every local point on its nearest RADIO-final region posterior."""

    if int(top_l) <= 0 or int(maximum_maplets_per_region) <= 0:
        raise ValueError("top_l and maximum_maplets_per_region must be positive")
    if float(descriptor_temperature) <= 0.0 or float(maplet_prior_weight) < 0.0:
        raise ValueError("descriptor temperature/prior weight is invalid")
    if float(support_spatial_sigma) <= 0.0 or float(maximum_support_distance) <= 0.0:
        raise ValueError("support spatial scales must be positive")
    if query.descriptors.shape[1] != descriptor_bank.feature_dim:
        raise ValueError("query and support local descriptor dimensions differ")
    region_xy = np.asarray(query_region_xy, dtype=np.float32)
    region_count = int(maplet_match.candidate_maplet_ids.shape[0])
    if region_xy.shape != (region_count, 2):
        raise ValueError("query_region_xy and maplet match rows must align")
    image_width, image_height = int(query_image_size[0]), int(query_image_size[1])
    if image_width <= 0 or image_height <= 0:
        raise ValueError("query image size must be positive")
    query_count = len(query.keypoints_xy)
    if query_count == 0 or region_count == 0:
        return SurfaceAnchorCandidatePool(
            query_xy=query.keypoints_xy,
            anchor_ids=np.zeros((query_count, 0), dtype=np.int64),
            xyz=np.zeros((query_count, 0, 3), dtype=np.float64),
            descriptor_scores=np.zeros((query_count, 0), dtype=np.float32),
            candidate_probabilities=np.zeros((query_count, 0), dtype=np.float32),
            null_probabilities=np.ones((query_count,), dtype=np.float32),
            valid_mask=np.zeros((query_count, 0), dtype=bool),
        )

    region_normalized = _normalized_xy(region_xy, query_region_grid_size)
    local_normalized = query.keypoints_xy / np.asarray(
        [max(image_width - 1, 1), max(image_height - 1, 1)],
        dtype=np.float32,
    )
    nearest_region = cKDTree(region_normalized).query(local_normalized, k=1)[1]
    maplet_row_by_id = {
        int(maplet_id): int(row) for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    anchor_row_by_id = anchors.row_by_id()
    descriptor_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    maplet_anchor_ids: dict[int, np.ndarray] = {}
    for maplet_id, maplet_row in maplet_row_by_id.items():
        start = int(maplets.anchor_offsets[maplet_row])
        end = int(maplets.anchor_offsets[maplet_row + 1])
        maplet_anchor_ids[maplet_id] = maplets.anchor_ids[start:end]
    support_projection_by_anchor: dict[int, np.ndarray] = {}
    if (
        maplet_match.support_view_id is not None
        and maplet_match.support_transform_matrix is not None
        and maplet_match.support_transform_translation is not None
    ):
        support_view = str(maplet_match.support_view_id)
        for anchor_row, anchor_id in enumerate(anchors.anchor_ids.tolist()):
            start = int(anchors.observation_offsets[anchor_row])
            end = int(anchors.observation_offsets[anchor_row + 1])
            rows = [
                row
                for row in range(start, end)
                if anchors.observation_image_ids[row] == support_view
            ]
            if rows:
                support_projection_by_anchor[int(anchor_id)] = (
                    anchors.observation_xy[int(rows[0])]
                    / np.asarray(
                        [max(image_width - 1, 1), max(image_height - 1, 1)],
                        dtype=np.float32,
                    )
                )
        predicted_support_xy = (
            local_normalized @ maplet_match.support_transform_matrix.T
            + maplet_match.support_transform_translation
        )
    else:
        predicted_support_xy = None

    output_ids = np.full((query_count, int(top_l)), -1, dtype=np.int64)
    output_xyz = np.zeros((query_count, int(top_l), 3), dtype=np.float64)
    output_scores = np.full((query_count, int(top_l)), -np.inf, dtype=np.float32)
    output_logits = np.full((query_count, int(top_l)), -np.inf, dtype=np.float64)
    valid = np.zeros((query_count, int(top_l)), dtype=bool)
    null_logits = np.full((query_count,), float(null_logit), dtype=np.float64)
    for query_row, region_row_value in enumerate(np.asarray(nearest_region).reshape(-1).tolist()):
        region_row = int(region_row_value)
        maplet_prior: dict[int, float] = {}
        for column in range(
            min(int(maximum_maplets_per_region), maplet_match.candidate_maplet_ids.shape[1])
        ):
            probability = float(maplet_match.candidate_probabilities[region_row, column])
            maplet_id = int(maplet_match.candidate_maplet_ids[region_row, column])
            if probability < float(minimum_maplet_probability) or maplet_id not in maplet_anchor_ids:
                continue
            maplet_prior[maplet_id] = max(probability, 1e-8)
        candidate_rows: list[tuple[int, int, float, float | None]] = []
        for maplet_id, prior in maplet_prior.items():
            for anchor_id_value in maplet_anchor_ids[maplet_id].tolist():
                anchor_id = int(anchor_id_value)
                descriptor_row = descriptor_row_by_id.get(anchor_id)
                anchor_row = anchor_row_by_id.get(anchor_id)
                if descriptor_row is None or anchor_row is None:
                    continue
                spatial_distance = None
                if predicted_support_xy is not None:
                    support_xy = support_projection_by_anchor.get(anchor_id)
                    if support_xy is None:
                        continue
                    spatial_distance = float(
                        np.linalg.norm(support_xy - predicted_support_xy[query_row])
                    )
                    if spatial_distance > float(maximum_support_distance):
                        continue
                candidate_rows.append((descriptor_row, anchor_row, prior, spatial_distance))
        if not candidate_rows:
            continue
        logits: list[float] = []
        local_scores: list[float] = []
        for descriptor_row, _anchor_row, prior, spatial_distance in candidate_rows:
            start = int(descriptor_bank.descriptor_offsets[descriptor_row])
            end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
            if start == end:
                local_score = -1.0
            else:
                local_score = float(
                    np.max(
                        descriptor_bank.descriptors[start:end]
                        @ query.descriptors[query_row]
                    )
                )
            local_scores.append(local_score)
            logit = (
                local_score / float(descriptor_temperature)
                + float(maplet_prior_weight) * np.log(max(prior, 1e-8))
            )
            if spatial_distance is not None:
                logit -= 0.5 * (float(spatial_distance) / float(support_spatial_sigma)) ** 2
            logits.append(logit)
        order = np.argsort(-np.asarray(logits), kind="mergesort")[: int(top_l)]
        for output_column, candidate_index in enumerate(order.tolist()):
            descriptor_row, anchor_row, _prior, _distance = candidate_rows[int(candidate_index)]
            anchor_id = int(descriptor_bank.anchor_ids[descriptor_row])
            output_ids[query_row, output_column] = anchor_id
            output_xyz[query_row, output_column] = anchors.xyz[anchor_row]
            output_scores[query_row, output_column] = float(local_scores[int(candidate_index)])
            output_logits[query_row, output_column] = float(logits[int(candidate_index)])
            valid[query_row, output_column] = True
        null_logits[query_row] += float(maplet_prior_weight) * np.log(
            max(float(maplet_match.null_probabilities[region_row]), 1e-8)
        )
    probabilities, null_probabilities = _softmax_with_null(output_logits, null_logits)
    probabilities[~valid] = 0.0
    return SurfaceAnchorCandidatePool(
        query_xy=query.keypoints_xy,
        anchor_ids=output_ids,
        xyz=output_xyz,
        descriptor_scores=output_scores,
        candidate_probabilities=probabilities,
        null_probabilities=null_probabilities,
        valid_mask=valid,
    )


def build_support_layout_guided_local_query(
    detected_query: LocalFeatureFrame,
    query_image_size: tuple[int, int],
    maplet_match: SurfaceMapletMatchResult,
    maplets: VfmSurfaceMapletBank,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    maximum_maplets_per_region: int = 2,
    search_radius_px: float = 32.0,
    maximum_query_points: int = 512,
) -> tuple[LocalFeatureFrame, np.ndarray]:
    """Use VFM whole-image layout to propose ALIKE anchor matches locally."""

    if (
        maplet_match.support_view_id is None
        or maplet_match.support_transform_matrix is None
        or maplet_match.support_transform_translation is None
    ):
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    if float(search_radius_px) <= 0.0 or int(maximum_query_points) <= 0:
        raise ValueError("guided local-query limits must be positive")
    image_width, image_height = int(query_image_size[0]), int(query_image_size[1])
    matrix = np.asarray(maplet_match.support_transform_matrix, dtype=np.float64)
    if abs(float(np.linalg.det(matrix))) <= 1e-8:
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    inverse = np.linalg.inv(matrix)
    translation = np.asarray(maplet_match.support_transform_translation, dtype=np.float64)
    support_view = str(maplet_match.support_view_id)
    maplet_row_by_id = {
        int(maplet_id): int(row) for row, maplet_id in enumerate(maplets.maplet_ids.tolist())
    }
    anchor_row_by_id = anchors.row_by_id()
    descriptor_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    proposal_prior: dict[int, float] = {}
    for region_row in range(maplet_match.candidate_maplet_ids.shape[0]):
        for column in range(
            min(
                int(maximum_maplets_per_region),
                maplet_match.candidate_maplet_ids.shape[1],
            )
        ):
            maplet_id = int(maplet_match.candidate_maplet_ids[region_row, column])
            probability = float(maplet_match.candidate_probabilities[region_row, column])
            maplet_row = maplet_row_by_id.get(maplet_id)
            if maplet_row is None:
                continue
            start = int(maplets.anchor_offsets[maplet_row])
            end = int(maplets.anchor_offsets[maplet_row + 1])
            for anchor_id_value in maplets.anchor_ids[start:end].tolist():
                anchor_id = int(anchor_id_value)
                proposal_prior[anchor_id] = max(proposal_prior.get(anchor_id, 0.0), probability)
    if not proposal_prior or len(detected_query.keypoints_xy) == 0:
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    tree = cKDTree(detected_query.keypoints_xy)
    proposals: list[tuple[float, int, int]] = []
    scale = np.asarray(
        [max(image_width - 1, 1), max(image_height - 1, 1)],
        dtype=np.float64,
    )
    for anchor_id, prior in proposal_prior.items():
        anchor_row = anchor_row_by_id.get(anchor_id)
        descriptor_row = descriptor_row_by_id.get(anchor_id)
        if anchor_row is None or descriptor_row is None:
            continue
        observation_start = int(anchors.observation_offsets[anchor_row])
        observation_end = int(anchors.observation_offsets[anchor_row + 1])
        support_rows = [
            row
            for row in range(observation_start, observation_end)
            if anchors.observation_image_ids[row] == support_view
        ]
        if not support_rows:
            continue
        support_normalized = anchors.observation_xy[int(support_rows[0])] / scale
        predicted_normalized = (support_normalized - translation) @ inverse.T
        predicted_xy = predicted_normalized * scale
        if (
            predicted_xy[0] < 0.0
            or predicted_xy[0] > image_width - 1
            or predicted_xy[1] < 0.0
            or predicted_xy[1] > image_height - 1
        ):
            continue
        nearby = tree.query_ball_point(predicted_xy, r=float(search_radius_px))
        if not nearby:
            continue
        start = int(descriptor_bank.descriptor_offsets[descriptor_row])
        end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
        if start == end:
            continue
        rows = np.asarray(nearby, dtype=np.int64)
        similarity = np.max(
            detected_query.descriptors[rows] @ descriptor_bank.descriptors[start:end].T,
            axis=1,
        )
        distance = np.linalg.norm(detected_query.keypoints_xy[rows] - predicted_xy, axis=1)
        score = (
            similarity
            - 0.25 * (distance / float(search_radius_px)) ** 2
            + 0.05 * np.log(max(float(prior), 1e-8))
        )
        best = int(np.argmax(score))
        proposals.append((float(score[best]), int(rows[best]), int(anchor_id)))
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_rows: list[int] = []
    selected_anchor_ids: list[int] = []
    used_detection_rows: set[int] = set()
    for _score, detection_row, anchor_id in proposals:
        if detection_row in used_detection_rows:
            continue
        used_detection_rows.add(detection_row)
        selected_rows.append(detection_row)
        selected_anchor_ids.append(anchor_id)
        if len(selected_rows) >= int(maximum_query_points):
            break
    if len(selected_rows) < 4:
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    rows = np.asarray(selected_rows, dtype=np.int64)
    return (
        LocalFeatureFrame(
            image_id=detected_query.image_id,
            keypoints_xy=detected_query.keypoints_xy[rows],
            descriptors=detected_query.descriptors[rows],
            scores=detected_query.scores[rows],
        ),
        np.asarray(selected_anchor_ids, dtype=np.int64),
    )


def build_vfm_support_guided_local_query(
    detected_query: LocalFeatureFrame,
    query_image_size: tuple[int, int],
    query_grid_size: tuple[int, int],
    support_view_id: str,
    query_to_support_matrix: np.ndarray,
    query_to_support_translation: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    search_radius_px: float = 24.0,
    maximum_query_points: int = 768,
) -> tuple[LocalFeatureFrame, np.ndarray]:
    """Transfer 2DGS anchors through a direct VFM query/support layout."""

    if float(search_radius_px) <= 0.0 or int(maximum_query_points) <= 0:
        raise ValueError("guided local-query limits must be positive")
    matrix = np.asarray(query_to_support_matrix, dtype=np.float64).reshape(2, 2)
    translation = np.asarray(query_to_support_translation, dtype=np.float64).reshape(2)
    if abs(float(np.linalg.det(matrix))) <= 1e-8:
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    inverse = np.linalg.inv(matrix)
    image_width, image_height = int(query_image_size[0]), int(query_image_size[1])
    grid_width, grid_height = int(query_grid_size[0]), int(query_grid_size[1])
    anchor_row_by_id = anchors.row_by_id()
    descriptor_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    tree = cKDTree(detected_query.keypoints_xy)
    proposals: list[tuple[float, int, int]] = []
    pixel_to_grid = np.asarray(
        [
            grid_width / max(image_width, 1),
            grid_height / max(image_height, 1),
        ],
        dtype=np.float64,
    )
    grid_to_pixel = 1.0 / pixel_to_grid
    for anchor_id, descriptor_row in descriptor_row_by_id.items():
        anchor_row = anchor_row_by_id.get(anchor_id)
        if anchor_row is None:
            continue
        start = int(anchors.observation_offsets[anchor_row])
        end = int(anchors.observation_offsets[anchor_row + 1])
        support_rows = [
            row
            for row in range(start, end)
            if anchors.observation_image_ids[row] == str(support_view_id)
        ]
        if not support_rows:
            continue
        support_grid_xy = (
            anchors.observation_xy[int(support_rows[0])].astype(np.float64)
            * pixel_to_grid
        )
        predicted_query_grid = (support_grid_xy - translation) @ inverse.T
        predicted_query_xy = predicted_query_grid * grid_to_pixel
        if (
            predicted_query_xy[0] < 0.0
            or predicted_query_xy[0] > image_width - 1
            or predicted_query_xy[1] < 0.0
            or predicted_query_xy[1] > image_height - 1
        ):
            continue
        nearby = tree.query_ball_point(predicted_query_xy, r=float(search_radius_px))
        if not nearby:
            continue
        descriptor_start = int(descriptor_bank.descriptor_offsets[descriptor_row])
        descriptor_end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
        if descriptor_start == descriptor_end:
            continue
        rows = np.asarray(nearby, dtype=np.int64)
        similarity = np.max(
            detected_query.descriptors[rows]
            @ descriptor_bank.descriptors[descriptor_start:descriptor_end].T,
            axis=1,
        )
        distance = np.linalg.norm(
            detected_query.keypoints_xy[rows] - predicted_query_xy,
            axis=1,
        )
        score = similarity - 0.5 * (
            distance / max(float(search_radius_px) * 0.5, 1.0)
        ) ** 2
        best = int(np.argmax(score))
        proposals.append((float(score[best]), int(rows[best]), int(anchor_id)))
    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_rows: list[int] = []
    selected_anchor_ids: list[int] = []
    used_detection_rows: set[int] = set()
    for _score, detection_row, anchor_id in proposals:
        if detection_row in used_detection_rows:
            continue
        used_detection_rows.add(detection_row)
        selected_rows.append(detection_row)
        selected_anchor_ids.append(anchor_id)
        if len(selected_rows) >= int(maximum_query_points):
            break
    if len(selected_rows) < 4:
        return detected_query, np.full(
            (len(detected_query.keypoints_xy),), -1, dtype=np.int64
        )
    rows = np.asarray(selected_rows, dtype=np.int64)
    return (
        LocalFeatureFrame(
            image_id=detected_query.image_id,
            keypoints_xy=detected_query.keypoints_xy[rows],
            descriptors=detected_query.descriptors[rows],
            scores=detected_query.scores[rows],
        ),
        np.asarray(selected_anchor_ids, dtype=np.int64),
    )


def predict_vfm_support_anchor_query_points(
    query_image_size: tuple[int, int],
    query_grid_size: tuple[int, int],
    support_view_id: str,
    query_to_support_matrix: np.ndarray,
    query_to_support_translation: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    allowed_anchor_ids: Sequence[int] | np.ndarray | None = None,
    maximum_points: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transfer visible support anchors and their ALIKE prototypes to a query."""

    if int(maximum_points) <= 0:
        raise ValueError("maximum_points must be positive")
    matrix = np.asarray(query_to_support_matrix, dtype=np.float64).reshape(2, 2)
    translation = np.asarray(query_to_support_translation, dtype=np.float64).reshape(2)
    if abs(float(np.linalg.det(matrix))) <= 1e-8:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, descriptor_bank.feature_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    inverse = np.linalg.inv(matrix)
    image_width, image_height = int(query_image_size[0]), int(query_image_size[1])
    grid_width, grid_height = int(query_grid_size[0]), int(query_grid_size[1])
    pixel_to_grid = np.asarray(
        [
            grid_width / max(image_width, 1),
            grid_height / max(image_height, 1),
        ],
        dtype=np.float64,
    )
    grid_to_pixel = 1.0 / pixel_to_grid
    anchor_row_by_id = anchors.row_by_id()
    allowed = (
        None
        if allowed_anchor_ids is None
        else {
            int(value)
            for value in np.asarray(allowed_anchor_ids, dtype=np.int64)
            .reshape(-1)
            .tolist()
        }
    )
    records: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for descriptor_row, anchor_id_value in enumerate(descriptor_bank.anchor_ids.tolist()):
        anchor_id = int(anchor_id_value)
        if allowed is not None and anchor_id not in allowed:
            continue
        anchor_row = anchor_row_by_id.get(anchor_id)
        if anchor_row is None:
            continue
        observation_start = int(anchors.observation_offsets[anchor_row])
        observation_end = int(anchors.observation_offsets[anchor_row + 1])
        support_observation_rows = [
            row
            for row in range(observation_start, observation_end)
            if anchors.observation_image_ids[row] == str(support_view_id)
        ]
        if not support_observation_rows:
            continue
        descriptor_start = int(descriptor_bank.descriptor_offsets[descriptor_row])
        descriptor_end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
        prototype_rows = [
            row
            for row in range(descriptor_start, descriptor_end)
            if descriptor_bank.support_image_ids[row] == str(support_view_id)
        ]
        if not prototype_rows:
            continue
        prototype_row = max(
            prototype_rows,
            key=lambda row: float(descriptor_bank.descriptor_quality[row]),
        )
        support_grid_xy = (
            anchors.observation_xy[int(support_observation_rows[0])].astype(np.float64)
            * pixel_to_grid
        )
        query_grid_xy = (support_grid_xy - translation) @ inverse.T
        query_xy = query_grid_xy * grid_to_pixel
        if (
            query_xy[0] < 0.0
            or query_xy[0] > image_width - 1
            or query_xy[1] < 0.0
            or query_xy[1] > image_height - 1
        ):
            continue
        quality = (
            float(anchors.quality_scores[anchor_row])
            * float(descriptor_bank.descriptor_quality[prototype_row])
        )
        records.append(
            (
                quality,
                anchor_id,
                query_xy.astype(np.float32),
                descriptor_bank.descriptors[prototype_row],
            )
        )
    records.sort(key=lambda item: (-item[0], item[1]))
    records = records[: int(maximum_points)]
    return (
        np.stack([item[2] for item in records]).astype(np.float32)
        if records
        else np.zeros((0, 2), dtype=np.float32),
        np.stack([item[3] for item in records]).astype(np.float32)
        if records
        else np.zeros((0, descriptor_bank.feature_dim), dtype=np.float32),
        np.asarray([item[1] for item in records], dtype=np.int64),
    )


def build_seeded_surface_anchor_candidate_pool(
    query: LocalFeatureFrame,
    seed_anchor_ids: np.ndarray,
    descriptor_bank: AnchorLocalDescriptorBank,
    anchors: StableSurfaceAnchorMap,
    top_l: int = 5,
    descriptor_temperature: float = 0.08,
    seed_prior_logit: float = 32.0,
    null_logit: float = 0.0,
) -> SurfaceAnchorCandidatePool:
    """Score local anchors inside the VFM-guided seed anchor's surface maplet."""

    if int(top_l) <= 0 or float(descriptor_temperature) <= 0.0:
        raise ValueError("top_l and descriptor_temperature must be positive")
    seeds = np.asarray(seed_anchor_ids, dtype=np.int64)
    if seeds.shape != (len(query.keypoints_xy),):
        raise ValueError("one seed anchor ID is required per query local feature")
    if query.descriptors.shape[1] != descriptor_bank.feature_dim:
        raise ValueError("query and support local descriptor dimensions differ")

    query_count = len(query.keypoints_xy)
    output_ids = np.full((query_count, int(top_l)), -1, dtype=np.int64)
    output_xyz = np.zeros((query_count, int(top_l), 3), dtype=np.float64)
    output_scores = np.full((query_count, int(top_l)), -np.inf, dtype=np.float32)
    output_logits = np.full((query_count, int(top_l)), -np.inf, dtype=np.float64)
    valid = np.zeros((query_count, int(top_l)), dtype=bool)
    anchor_row_by_id = anchors.row_by_id()
    descriptor_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(descriptor_bank.anchor_ids.tolist())
    }
    descriptor_rows_by_maplet: dict[int, list[int]] = {}
    for descriptor_row, anchor_id_value in enumerate(descriptor_bank.anchor_ids.tolist()):
        anchor_row = anchor_row_by_id.get(int(anchor_id_value))
        if anchor_row is None:
            continue
        maplet_id = int(anchors.owner_maplet_ids[anchor_row])
        descriptor_rows_by_maplet.setdefault(maplet_id, []).append(descriptor_row)

    for query_row, seed_anchor_id_value in enumerate(seeds.tolist()):
        seed_anchor_id = int(seed_anchor_id_value)
        seed_anchor_row = anchor_row_by_id.get(seed_anchor_id)
        if seed_anchor_row is None:
            continue
        maplet_id = int(anchors.owner_maplet_ids[seed_anchor_row])
        candidate_descriptor_rows = descriptor_rows_by_maplet.get(maplet_id, [])
        candidate_records: list[tuple[float, float, int, int]] = []
        for descriptor_row in candidate_descriptor_rows:
            anchor_id = int(descriptor_bank.anchor_ids[descriptor_row])
            anchor_row = anchor_row_by_id.get(anchor_id)
            if anchor_row is None:
                continue
            start = int(descriptor_bank.descriptor_offsets[descriptor_row])
            end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
            if start == end:
                continue
            local_score = float(
                np.max(
                    descriptor_bank.descriptors[start:end]
                    @ query.descriptors[query_row]
                )
            )
            logit = local_score / float(descriptor_temperature)
            if anchor_id == seed_anchor_id:
                logit += float(seed_prior_logit)
            candidate_records.append((logit, local_score, anchor_id, anchor_row))
        candidate_records.sort(key=lambda item: (-item[0], item[2]))
        for output_column, (logit, local_score, anchor_id, anchor_row) in enumerate(
            candidate_records[: int(top_l)]
        ):
            output_ids[query_row, output_column] = anchor_id
            output_xyz[query_row, output_column] = anchors.xyz[anchor_row]
            output_scores[query_row, output_column] = local_score
            output_logits[query_row, output_column] = logit
            valid[query_row, output_column] = True

    probabilities, null_probabilities = _softmax_with_null(
        output_logits,
        np.full((query_count,), float(null_logit), dtype=np.float64),
    )
    probabilities[~valid] = 0.0
    return SurfaceAnchorCandidatePool(
        query_xy=query.keypoints_xy,
        anchor_ids=output_ids,
        xyz=output_xyz,
        descriptor_scores=output_scores,
        candidate_probabilities=probabilities,
        null_probabilities=null_probabilities,
        valid_mask=valid,
    )


def build_pose_guided_surface_anchor_candidate_pool(
    query: LocalFeatureFrame,
    pose_w2c: np.ndarray,
    camera,
    descriptor_bank: AnchorLocalDescriptorBank,
    anchors: StableSurfaceAnchorMap,
    allowed_anchor_ids: Sequence[int] | np.ndarray | None = None,
    top_l: int = 5,
    search_radius_px: float = 32.0,
    maximum_query_points: int = 768,
    descriptor_temperature: float = 0.08,
    spatial_sigma_px: float = 8.0,
    null_logit: float = 0.0,
) -> SurfaceAnchorCandidatePool:
    """Refine a VFM pose by projecting metric 2DGS anchors into the query."""

    import cv2

    if (
        int(top_l) <= 0
        or float(search_radius_px) <= 0.0
        or int(maximum_query_points) <= 0
        or float(descriptor_temperature) <= 0.0
        or float(spatial_sigma_px) <= 0.0
    ):
        raise ValueError("pose-guided candidate-pool limits must be positive")
    if query.descriptors.shape[1] != descriptor_bank.feature_dim:
        raise ValueError("query and support local descriptor dimensions differ")
    anchor_row_by_id = anchors.row_by_id()
    allowed = (
        None
        if allowed_anchor_ids is None
        else {
            int(value)
            for value in np.asarray(allowed_anchor_ids, dtype=np.int64)
            .reshape(-1)
            .tolist()
        }
    )
    descriptor_anchor_rows: list[int] = []
    descriptor_rows: list[int] = []
    for descriptor_row, anchor_id_value in enumerate(descriptor_bank.anchor_ids.tolist()):
        anchor_id = int(anchor_id_value)
        if allowed is not None and anchor_id not in allowed:
            continue
        anchor_row = anchor_row_by_id.get(anchor_id)
        if anchor_row is not None:
            descriptor_rows.append(descriptor_row)
            descriptor_anchor_rows.append(anchor_row)
    if not descriptor_rows or len(query.keypoints_xy) == 0:
        return SurfaceAnchorCandidatePool(
            query_xy=np.zeros((0, 2), dtype=np.float32),
            anchor_ids=np.zeros((0, int(top_l)), dtype=np.int64),
            xyz=np.zeros((0, int(top_l), 3), dtype=np.float64),
            descriptor_scores=np.zeros((0, int(top_l)), dtype=np.float32),
            candidate_probabilities=np.zeros((0, int(top_l)), dtype=np.float32),
            null_probabilities=np.zeros((0,), dtype=np.float32),
            valid_mask=np.zeros((0, int(top_l)), dtype=bool),
        )

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    xyz = anchors.xyz[np.asarray(descriptor_anchor_rows, dtype=np.int64)]
    projected, _jacobian = cv2.projectPoints(
        xyz,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    margin = float(search_radius_px)
    visible = (
        (camera_xyz[:, 2] > 1e-6)
        & (projected[:, 0] >= -margin)
        & (projected[:, 0] <= float(camera.width - 1) + margin)
        & (projected[:, 1] >= -margin)
        & (projected[:, 1] <= float(camera.height - 1) + margin)
    )
    visible_rows = np.flatnonzero(visible)
    if len(visible_rows) == 0:
        return SurfaceAnchorCandidatePool(
            query_xy=np.zeros((0, 2), dtype=np.float32),
            anchor_ids=np.zeros((0, int(top_l)), dtype=np.int64),
            xyz=np.zeros((0, int(top_l), 3), dtype=np.float64),
            descriptor_scores=np.zeros((0, int(top_l)), dtype=np.float32),
            candidate_probabilities=np.zeros((0, int(top_l)), dtype=np.float32),
            null_probabilities=np.zeros((0,), dtype=np.float32),
            valid_mask=np.zeros((0, int(top_l)), dtype=bool),
        )
    tree = cKDTree(projected[visible_rows])
    candidates_by_query: dict[int, list[tuple[float, float, int, int]]] = {}
    for query_row, query_xy in enumerate(query.keypoints_xy):
        nearby_local_rows = tree.query_ball_point(query_xy, r=float(search_radius_px))
        for nearby_local_row in nearby_local_rows:
            visible_row = int(visible_rows[int(nearby_local_row)])
            descriptor_row = int(descriptor_rows[visible_row])
            anchor_row = int(descriptor_anchor_rows[visible_row])
            start = int(descriptor_bank.descriptor_offsets[descriptor_row])
            end = int(descriptor_bank.descriptor_offsets[descriptor_row + 1])
            if start == end:
                continue
            local_score = float(
                np.max(
                    descriptor_bank.descriptors[start:end]
                    @ query.descriptors[query_row]
                )
            )
            distance = float(np.linalg.norm(projected[visible_row] - query_xy))
            logit = (
                local_score / float(descriptor_temperature)
                - 0.5 * (distance / float(spatial_sigma_px)) ** 2
            )
            candidates_by_query.setdefault(query_row, []).append(
                (logit, local_score, descriptor_row, anchor_row)
            )
    ordered_query_rows = sorted(
        candidates_by_query,
        key=lambda row: (
            -max(item[0] for item in candidates_by_query[row]),
            row,
        ),
    )[: int(maximum_query_points)]
    query_count = len(ordered_query_rows)
    output_ids = np.full((query_count, int(top_l)), -1, dtype=np.int64)
    output_xyz = np.zeros((query_count, int(top_l), 3), dtype=np.float64)
    output_scores = np.full((query_count, int(top_l)), -np.inf, dtype=np.float32)
    output_logits = np.full((query_count, int(top_l)), -np.inf, dtype=np.float64)
    valid = np.zeros((query_count, int(top_l)), dtype=bool)
    for output_row, query_row in enumerate(ordered_query_rows):
        records = sorted(
            candidates_by_query[query_row],
            key=lambda item: (-item[0], int(descriptor_bank.anchor_ids[item[2]])),
        )
        for output_column, (logit, local_score, descriptor_row, anchor_row) in enumerate(
            records[: int(top_l)]
        ):
            output_ids[output_row, output_column] = int(
                descriptor_bank.anchor_ids[descriptor_row]
            )
            output_xyz[output_row, output_column] = anchors.xyz[anchor_row]
            output_scores[output_row, output_column] = local_score
            output_logits[output_row, output_column] = logit
            valid[output_row, output_column] = True
    probabilities, null_probabilities = _softmax_with_null(
        output_logits,
        np.full((query_count,), float(null_logit), dtype=np.float64),
    )
    probabilities[~valid] = 0.0
    rows = np.asarray(ordered_query_rows, dtype=np.int64)
    return SurfaceAnchorCandidatePool(
        query_xy=query.keypoints_xy[rows],
        anchor_ids=output_ids,
        xyz=output_xyz,
        descriptor_scores=output_scores,
        candidate_probabilities=probabilities,
        null_probabilities=null_probabilities,
        valid_mask=valid,
    )


@dataclass(frozen=True)
class SurfacePoseConfig:
    random_seed: int = 7
    hypothesis_count: int = 256
    minimal_sample_size: int = 6
    minimum_fit_groups: int = 8
    minimum_verification_groups: int = 4
    reprojection_sigma_px: float = 3.0
    inlier_threshold_px: float = 6.0
    null_likelihood: float = 1e-3
    outlier_likelihood: float = 1e-3
    heldout_stride: int = 5
    duplicate_translation_m: float = 0.02
    duplicate_rotation_deg: float = 0.25
    minimum_selected_inliers: int = 8

    def __post_init__(self) -> None:
        if int(self.hypothesis_count) <= 0:
            raise ValueError("hypothesis_count must be positive")
        if int(self.minimal_sample_size) < 4:
            raise ValueError("minimal_sample_size must be at least four")
        if int(self.minimum_fit_groups) < 4 or int(self.minimum_verification_groups) < 1:
            raise ValueError("fit/verification group counts are invalid")
        if float(self.reprojection_sigma_px) <= 0.0 or float(self.inlier_threshold_px) <= 0.0:
            raise ValueError("reprojection scales must be positive")
        if int(self.heldout_stride) < 2:
            raise ValueError("heldout_stride must be at least two")
        if int(self.minimum_selected_inliers) < 4:
            raise ValueError("minimum_selected_inliers must be at least four")


@dataclass(frozen=True)
class SurfacePoseHypothesis:
    pose_w2c: np.ndarray
    generation_score: float
    verification_score: float
    inlier_count: int
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_w2c", np.asarray(self.pose_w2c, dtype=np.float64).reshape(4, 4))


@dataclass(frozen=True)
class SurfacePoseResult:
    success: bool
    pose_w2c: np.ndarray
    hypotheses: tuple[SurfacePoseHypothesis, ...]
    fit_mask: np.ndarray
    verification_mask: np.ndarray
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_w2c", np.asarray(self.pose_w2c, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "fit_mask", np.asarray(self.fit_mask, dtype=bool).reshape(-1))
        object.__setattr__(self, "verification_mask", np.asarray(self.verification_mask, dtype=bool).reshape(-1))


def rerank_surface_pose_with_mapping_view_prior(
    result: SurfacePoseResult,
    mapping_view_pose_w2c: np.ndarray,
    translation_weight: float = 10.0,
    rotation_weight: float = 0.10,
) -> SurfacePoseResult:
    """Combine held-out evidence with a VFM-retrieved mapping-view pose prior."""

    if float(translation_weight) < 0.0 or float(rotation_weight) < 0.0:
        raise ValueError("pose-prior weights must be non-negative")
    if not result.hypotheses:
        return result
    reference = np.asarray(mapping_view_pose_w2c, dtype=np.float64).reshape(4, 4)
    ranked = sorted(
        result.hypotheses,
        key=lambda hypothesis: (
            float(hypothesis.verification_score)
            - float(translation_weight) * _pose_distance(hypothesis.pose_w2c, reference)[0]
            - float(rotation_weight) * _pose_distance(hypothesis.pose_w2c, reference)[1],
            float(hypothesis.generation_score),
            int(hypothesis.inlier_count),
        ),
        reverse=True,
    )
    selected = ranked[0]
    return SurfacePoseResult(
        success=bool(
            result.success
            or int(selected.inlier_count) >= SurfacePoseConfig().minimum_selected_inliers
        ),
        pose_w2c=selected.pose_w2c,
        hypotheses=tuple(ranked),
        fit_mask=result.fit_mask,
        verification_mask=result.verification_mask,
        failure_reason=(
            None
            if int(selected.inlier_count) >= SurfacePoseConfig().minimum_selected_inliers
            else result.failure_reason
        ),
    )


def _project_pool(pool: SurfaceAnchorCandidatePool, pose_w2c: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    matrix, distortion = camera_matrix_and_distortion(camera)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    flat_xyz = pool.xyz.reshape(-1, 3)
    projected, _jacobian = cv2.projectPoints(flat_xyz, rvec, pose[:3, 3], matrix, distortion)
    projected = projected.reshape(pool.xyz.shape[0], pool.xyz.shape[1], 2)
    camera_xyz = flat_xyz @ pose[:3, :3].T + pose[:3, 3]
    positive_depth = camera_xyz[:, 2].reshape(pool.valid_mask.shape) > 1e-6
    return projected, positive_depth & pool.valid_mask


def score_surface_pose(
    pool: SurfaceAnchorCandidatePool,
    pose_w2c: np.ndarray,
    camera,
    group_mask: np.ndarray,
    config: SurfacePoseConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fixed-candidate marginal likelihood with an explicit null."""

    projected, valid = _project_pool(pool, pose_w2c, camera)
    residual = np.linalg.norm(projected - pool.query_xy[:, None, :], axis=2)
    likelihood = np.full(residual.shape, float(config.outlier_likelihood), dtype=np.float64)
    likelihood[valid] += np.exp(
        -0.5 * (residual[valid] / float(config.reprojection_sigma_px)) ** 2
    )
    mixture = np.sum(pool.candidate_probabilities * likelihood, axis=1)
    mixture += pool.null_probabilities * float(config.null_likelihood)
    mask = np.asarray(group_mask, dtype=bool).reshape(-1)
    if mask.shape != (len(pool),):
        raise ValueError("group_mask must have shape (Q,)")
    score = float(np.sum(np.log(np.maximum(mixture[mask], 1e-12))))
    posterior = pool.candidate_probabilities * likelihood
    posterior /= np.maximum(
        np.sum(posterior, axis=1, keepdims=True)
        + pool.null_probabilities[:, None] * float(config.null_likelihood),
        1e-12,
    )
    return score, residual, posterior


def _pose_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64).reshape(4, 4)
    right = np.asarray(right, dtype=np.float64).reshape(4, 4)
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    translation = float(np.linalg.norm(left_center - right_center))
    relative = left[:3, :3] @ right[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation, float(np.degrees(np.arccos(cosine)))


def _pose_from_correspondences(xyz: np.ndarray, xy: np.ndarray, camera, use_ransac: bool) -> np.ndarray | None:
    import cv2

    if len(xyz) < 4 or len(np.unique(np.asarray(xyz), axis=0)) < 4:
        return None
    matrix, distortion = camera_matrix_and_distortion(camera)
    xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64))
    xy = np.ascontiguousarray(np.asarray(xy, dtype=np.float64))
    if use_ransac:
        success, rvec, tvec, _inliers = cv2.solvePnPRansac(
            xyz,
            xy,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=1000,
            reprojectionError=6.0,
            confidence=0.999,
        )
    else:
        flag = cv2.SOLVEPNP_AP3P if len(xyz) == 4 else cv2.SOLVEPNP_EPNP
        success, rvec, tvec = cv2.solvePnP(xyz, xy, matrix, distortion, flags=flag)
    if not bool(success):
        return None
    rotation, _jacobian = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return pose


def _refine_pose_from_pool(
    pool: SurfaceAnchorCandidatePool,
    pose: np.ndarray,
    camera,
    fit_mask: np.ndarray,
    config: SurfacePoseConfig,
) -> tuple[np.ndarray, int]:
    score, residual, posterior = score_surface_pose(pool, pose, camera, fit_mask, config)
    del score
    best_columns = np.argmax(posterior, axis=1)
    best_posterior = posterior[np.arange(len(pool)), best_columns]
    rows = np.flatnonzero(
        np.asarray(fit_mask, dtype=bool)
        & (best_posterior > pool.null_probabilities)
        & (residual[np.arange(len(pool)), best_columns] <= float(config.inlier_threshold_px))
    )
    if rows.size < 4:
        return pose, int(rows.size)
    order = rows[np.argsort(-best_posterior[rows], kind="mergesort")]
    unique_rows: list[int] = []
    used_anchors: set[int] = set()
    for row in order.tolist():
        anchor_id = int(pool.anchor_ids[row, best_columns[row]])
        if anchor_id in used_anchors:
            continue
        used_anchors.add(anchor_id)
        unique_rows.append(int(row))
    rows = np.asarray(unique_rows, dtype=np.int64)
    if rows.size < 4:
        return pose, int(rows.size)
    refined = _pose_from_correspondences(
        pool.xyz[rows, best_columns[rows]],
        pool.query_xy[rows],
        camera,
        use_ransac=True,
    )
    return (pose if refined is None else refined), int(rows.size)


def generate_grouped_surface_pose_hypotheses(
    pool: SurfaceAnchorCandidatePool,
    camera,
    config: SurfacePoseConfig = SurfacePoseConfig(),
    fit_mask: np.ndarray | None = None,
    verification_mask: np.ndarray | None = None,
) -> SurfacePoseResult:
    """Generate grouped PnP modes and select with independent query groups."""

    query_count = len(pool)
    if fit_mask is None and verification_mask is None:
        verification = np.zeros((query_count,), dtype=bool)
        verification[np.arange(query_count) % int(config.heldout_stride) == 0] = True
        fit = ~verification
    elif fit_mask is None or verification_mask is None:
        raise ValueError("fit_mask and verification_mask must be provided together")
    else:
        fit = np.asarray(fit_mask, dtype=bool).reshape(-1)
        verification = np.asarray(verification_mask, dtype=bool).reshape(-1)
    if fit.shape != (query_count,) or verification.shape != (query_count,):
        raise ValueError("pose masks must have shape (Q,)")
    if np.any(fit & verification):
        raise ValueError("fit and verification groups must be disjoint")
    usable = np.any(pool.valid_mask, axis=1)
    fit &= usable
    verification &= usable
    if int(np.sum(fit)) < int(config.minimum_fit_groups):
        return SurfacePoseResult(
            success=False,
            pose_w2c=np.eye(4),
            hypotheses=(),
            fit_mask=fit,
            verification_mask=verification,
            failure_reason="insufficient_fit_groups",
        )
    if int(np.sum(verification)) < int(config.minimum_verification_groups):
        return SurfacePoseResult(
            success=False,
            pose_w2c=np.eye(4),
            hypotheses=(),
            fit_mask=fit,
            verification_mask=verification,
            failure_reason="insufficient_independent_verification_groups",
        )

    generated: list[tuple[np.ndarray, str]] = []
    fit_rows = np.flatnonzero(fit)
    top_columns = np.argmax(pool.candidate_probabilities, axis=1)
    top_order = fit_rows[
        np.argsort(
            -pool.candidate_probabilities[fit_rows, top_columns[fit_rows]],
            kind="mergesort",
        )
    ]
    unique_rows: list[int] = []
    used_anchor_ids: set[int] = set()
    for row in top_order.tolist():
        anchor_id = int(pool.anchor_ids[row, top_columns[row]])
        if anchor_id in used_anchor_ids:
            continue
        used_anchor_ids.add(anchor_id)
        unique_rows.append(int(row))
    if len(unique_rows) >= int(config.minimum_fit_groups):
        rows = np.asarray(unique_rows, dtype=np.int64)
        pose = _pose_from_correspondences(
            pool.xyz[rows, top_columns[rows]],
            pool.query_xy[rows],
            camera,
            use_ransac=True,
        )
        if pose is not None:
            generated.append((pose, "top1_ransac"))

    rng = np.random.default_rng(int(config.random_seed))
    group_mass = np.sum(pool.candidate_probabilities, axis=1)
    fit_weights = group_mass[fit_rows].astype(np.float64)
    fit_weights /= max(float(np.sum(fit_weights)), 1e-12)
    sample_size = min(int(config.minimal_sample_size), len(fit_rows))
    for _iteration in range(int(config.hypothesis_count)):
        sampled_rows = rng.choice(fit_rows, size=sample_size, replace=False, p=fit_weights)
        sampled_columns = []
        used: set[int] = set()
        valid_sample = True
        for row in sampled_rows.tolist():
            valid_columns = np.flatnonzero(pool.valid_mask[row])
            if valid_columns.size == 0:
                valid_sample = False
                break
            probabilities = pool.candidate_probabilities[row, valid_columns].astype(np.float64)
            probabilities = np.maximum(probabilities, 0.0)
            positive = probabilities > 0.0
            positive_columns = valid_columns[positive]
            zero_columns = valid_columns[~positive]
            if positive_columns.size:
                positive_probabilities = probabilities[positive]
                positive_probabilities /= float(np.sum(positive_probabilities))
                sampled_positive = rng.choice(
                    positive_columns,
                    size=len(positive_columns),
                    replace=False,
                    p=positive_probabilities,
                )
            else:
                sampled_positive = np.zeros((0,), dtype=np.int64)
            column_order = np.concatenate(
                [sampled_positive, rng.permutation(zero_columns)]
            )
            chosen = next(
                (int(column) for column in column_order if int(pool.anchor_ids[row, column]) not in used),
                None,
            )
            if chosen is None:
                valid_sample = False
                break
            used.add(int(pool.anchor_ids[row, chosen]))
            sampled_columns.append(chosen)
        if not valid_sample:
            continue
        columns = np.asarray(sampled_columns, dtype=np.int64)
        pose = _pose_from_correspondences(
            pool.xyz[sampled_rows, columns],
            pool.query_xy[sampled_rows],
            camera,
            use_ransac=False,
        )
        if pose is not None:
            generated.append((pose, "grouped_sample"))

    hypotheses: list[SurfacePoseHypothesis] = []
    for pose, source in generated:
        refined, inlier_count = _refine_pose_from_pool(pool, pose, camera, fit, config)
        generation_score, _residual, _posterior = score_surface_pose(pool, refined, camera, fit, config)
        verification_score, _residual, _posterior = score_surface_pose(
            pool, refined, camera, verification, config
        )
        duplicate = False
        for existing in hypotheses:
            translation, rotation = _pose_distance(existing.pose_w2c, refined)
            if (
                translation <= float(config.duplicate_translation_m)
                and rotation <= float(config.duplicate_rotation_deg)
            ):
                duplicate = True
                break
        if duplicate:
            continue
        hypotheses.append(
            SurfacePoseHypothesis(
                pose_w2c=refined,
                generation_score=generation_score,
                verification_score=verification_score,
                inlier_count=inlier_count,
                source=source,
            )
        )
    hypotheses.sort(
        key=lambda item: (item.verification_score, item.generation_score, item.inlier_count),
        reverse=True,
    )
    if not hypotheses:
        return SurfacePoseResult(
            success=False,
            pose_w2c=np.eye(4),
            hypotheses=(),
            fit_mask=fit,
            verification_mask=verification,
            failure_reason="no_valid_surface_pose_hypothesis",
        )
    feasible = [
        hypothesis
        for hypothesis in hypotheses
        if int(hypothesis.inlier_count)
        >= int(config.minimum_selected_inliers)
    ]
    if not feasible:
        return SurfacePoseResult(
            success=False,
            pose_w2c=np.eye(4),
            hypotheses=tuple(hypotheses),
            fit_mask=fit,
            verification_mask=verification,
            failure_reason="insufficient_selected_pose_inliers",
        )
    # Reprojection support is a hard geometric feasibility condition.  Within
    # that feasible set, retain the independent held-out ordering used above.
    # This prevents a slightly better held-out score with too few fit inliers
    # from suppressing every geometrically valid mode.
    selected = feasible[0]
    hypotheses = [
        selected,
        *[
            hypothesis
            for hypothesis in hypotheses
            if hypothesis is not selected
        ],
    ]
    return SurfacePoseResult(
        success=True,
        pose_w2c=selected.pose_w2c,
        hypotheses=tuple(hypotheses),
        fit_mask=fit,
        verification_mask=verification,
        failure_reason=None,
    )
