"""Reference patch maplets for VFM patch-to-patch diagnostics.

A reference patch maplet is not a triangulated VFM point.  It is a reference
VFM token descriptor plus the set of existing SfM landmarks whose 2D
observations fall into that token footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.patch_footprint_matching import FootprintObservation
from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSets
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch, normalize_rows, token_grid_xy


@dataclass(frozen=True)
class ReferencePatchMaplet:
    unit_id: int
    reference_image_id: str
    token_index: int
    feature: np.ndarray
    track_ids: tuple[int, ...]
    landmark_indices: np.ndarray
    num_landmarks: int
    mean_observation_count: float
    mean_reprojection_error: float


@dataclass(frozen=True)
class ReferencePatchMapletBank:
    units: tuple[ReferencePatchMaplet, ...]
    feature_dim: int

    def __len__(self) -> int:
        return len(self.units)

    @property
    def features(self) -> np.ndarray:
        if not self.units:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([unit.feature for unit in self.units], axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ReferencePatchMapletConfig:
    top_k: int = 5
    min_similarity: float = 0.0
    query_token_step: int = 1
    block_size: int = 512
    max_matches: int | None = None

    def __post_init__(self) -> None:
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if int(self.query_token_step) <= 0:
            raise ValueError("query_token_step must be positive")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive")
        if self.max_matches is not None and int(self.max_matches) <= 0:
            raise ValueError("max_matches must be positive")


@dataclass(frozen=True)
class ReferencePatchMapletMatch:
    token_index: int
    xy: np.ndarray
    reference_image_id: str
    reference_token_index: int
    unit_id: int
    support_track_ids: tuple[int, ...]
    similarity: float
    rank: int
    source: str = "reference_patch_maplet"


@dataclass(frozen=True)
class SupportSelectorModel:
    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...] = (
        "maplet_similarity",
        "support_feature_similarity",
        "support_quality",
        "rank_score",
        "support_count_score",
    )

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != len(self.feature_names):
            raise ValueError("support selector weights must match feature_names")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", float(self.bias))


@dataclass(frozen=True)
class MapletVerifierModel:
    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...] = (
        "maplet_similarity",
        "rank_score",
        "support_count_score",
        "mean_observation_score",
        "reprojection_score",
    )

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != len(self.feature_names):
            raise ValueError("maplet verifier weights must match feature_names")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", float(self.bias))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def _token_index_from_xy(
    xy: np.ndarray,
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
) -> int:
    point = np.asarray(xy, dtype=np.float64).reshape(2)
    x_norm = float(point[0]) / max(float(image_width - 1), 1.0)
    y_norm = float(point[1]) / max(float(image_height - 1), 1.0)
    x_idx = int(round(np.clip(x_norm, 0.0, 1.0) * max(token_width - 1, 0)))
    y_idx = int(round(np.clip(y_norm, 0.0, 1.0) * max(token_height - 1, 0)))
    return int(y_idx * token_width + x_idx)


def _flatten_query_features(feature_map: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = values.shape
    features = []
    token_indices = []
    for y_idx in range(0, height, int(step)):
        for x_idx in range(0, width, int(step)):
            features.append(values[:, y_idx, x_idx])
            token_indices.append(y_idx * width + x_idx)
    if not features:
        return np.zeros((0, channels), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(token_indices, dtype=np.int64)


def compute_patch_context_feature_map(feature_map: np.ndarray, context: str = "1x1") -> np.ndarray:
    """Average local token neighborhoods and L2-normalize each patch descriptor."""

    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    if context not in {"1x1", "3x3", "5x5"}:
        raise ValueError("context must be one of: 1x1, 3x3, 5x5")
    channels, height, width = values.shape
    if context == "1x1":
        flat, valid = normalize_rows(values.reshape(channels, -1).T)
        flat[~valid] = 0.0
        return flat.T.reshape(values.shape).astype(np.float32, copy=False)

    radius = 1 if context == "3x3" else 2
    output = np.zeros_like(values, dtype=np.float32)
    for y_idx in range(height):
        y0 = max(0, y_idx - radius)
        y1 = min(height, y_idx + radius + 1)
        for x_idx in range(width):
            x0 = max(0, x_idx - radius)
            x1 = min(width, x_idx + radius + 1)
            output[:, y_idx, x_idx] = np.mean(values[:, y0:y1, x0:x1].reshape(channels, -1), axis=1)
    flat, valid = normalize_rows(output.reshape(channels, -1).T)
    flat[~valid] = 0.0
    return flat.T.reshape(output.shape).astype(np.float32, copy=False)


def project_feature_map_tokens(
    feature_map: np.ndarray,
    projector,
    output_dim: int,
    device: str = "cpu",
    batch_size: int = 4096,
    active_group_mask=None,
) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    if int(output_dim) <= 0:
        raise ValueError("output_dim must be positive")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for token projection") from exc

    channels, height, width = values.shape
    flat = values.reshape(channels, height * width).T.astype(np.float32, copy=False)
    outputs = []
    module = projector.to(device) if hasattr(projector, "to") else projector
    if hasattr(module, "eval"):
        module.eval()
    mask = None
    if active_group_mask is not None:
        mask = torch.as_tensor(np.asarray(active_group_mask, dtype=np.float32), device=device)
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(batch_size)):
            batch = torch.from_numpy(flat[start : start + int(batch_size)]).to(device=device)
            try:
                if mask is None:
                    projected = module(batch)
                else:
                    projected = module(batch, active_group_mask=mask)
            except TypeError:
                projected = module(batch)
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)
            outputs.append(projected.detach().cpu().numpy().astype(np.float32))
    if not outputs:
        return np.zeros((int(output_dim), height, width), dtype=np.float32)
    projected_flat = np.concatenate(outputs, axis=0)
    if projected_flat.shape != (height * width, int(output_dim)):
        raise ValueError("projector output shape does not match output_dim")
    return projected_flat.T.reshape(int(output_dim), height, width).astype(np.float32, copy=False)


def build_reference_patch_maplet_bank(
    landmark_index: LandmarkMapIndex,
    observations_by_image: Mapping[str, Sequence[FootprintObservation]],
    reference_feature_maps: Mapping[str, np.ndarray],
    reference_image_ids: Sequence[str],
    min_landmarks_per_maplet: int = 1,
    max_landmarks_per_maplet: int | None = 64,
    cell_radius: int = 0,
) -> ReferencePatchMapletBank:
    if int(min_landmarks_per_maplet) <= 0:
        raise ValueError("min_landmarks_per_maplet must be positive")
    if max_landmarks_per_maplet is not None and int(max_landmarks_per_maplet) <= 0:
        raise ValueError("max_landmarks_per_maplet must be positive")
    if int(cell_radius) < 0:
        raise ValueError("cell_radius must be non-negative")

    feature_dim = 0
    for feature_map in reference_feature_maps.values():
        values = np.asarray(feature_map, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("reference feature maps must have shape (C, H, W)")
        feature_dim = int(values.shape[0])
        break
    if feature_dim == 0 or len(landmark_index) == 0:
        return ReferencePatchMapletBank(units=(), feature_dim=feature_dim)

    track_to_landmark = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    cells: dict[tuple[str, int], set[int]] = {}
    for image_id in dict.fromkeys(str(item) for item in reference_image_ids):
        feature_map = reference_feature_maps.get(image_id)
        if feature_map is None:
            continue
        _channels, token_height, token_width = np.asarray(feature_map).shape
        for obs in observations_by_image.get(image_id, ()):
            landmark_idx = track_to_landmark.get(int(obs.track_id))
            if landmark_idx is None:
                continue
            token_index = _token_index_from_xy(
                obs.xy,
                int(obs.image_width),
                int(obs.image_height),
                int(token_width),
                int(token_height),
            )
            token_x = int(token_index) % int(token_width)
            token_y = int(token_index) // int(token_width)
            for yy in range(max(0, token_y - int(cell_radius)), min(int(token_height), token_y + int(cell_radius) + 1)):
                for xx in range(max(0, token_x - int(cell_radius)), min(int(token_width), token_x + int(cell_radius) + 1)):
                    local_token = int(yy * int(token_width) + xx)
                    cells.setdefault((image_id, local_token), set()).add(int(landmark_idx))

    units: list[ReferencePatchMaplet] = []
    for image_id, token_index in sorted(cells):
        landmark_indices = np.asarray(sorted(cells[(image_id, token_index)]), dtype=np.int64)
        if landmark_indices.size < int(min_landmarks_per_maplet):
            continue
        if max_landmarks_per_maplet is not None and landmark_indices.size > int(max_landmarks_per_maplet):
            order = np.lexsort(
                (
                    landmark_index.mean_variances[landmark_indices],
                    -landmark_index.observation_counts[landmark_indices],
                )
            )
            landmark_indices = landmark_indices[order[: int(max_landmarks_per_maplet)]]
        feature_map = np.asarray(reference_feature_maps[image_id], dtype=np.float32)
        _channels, token_height, token_width = feature_map.shape
        y_idx = int(token_index) // int(token_width)
        x_idx = int(token_index) % int(token_width)
        feature = _normalize_vector(feature_map[:, y_idx, x_idx])
        track_ids = tuple(int(landmark_index.track_ids[idx]) for idx in landmark_indices.tolist())
        units.append(
            ReferencePatchMaplet(
                unit_id=len(units),
                reference_image_id=str(image_id),
                token_index=int(token_index),
                feature=feature,
                track_ids=track_ids,
                landmark_indices=landmark_indices.astype(np.int64, copy=False),
                num_landmarks=int(landmark_indices.size),
                mean_observation_count=float(np.mean(landmark_index.observation_counts[landmark_indices])),
                mean_reprojection_error=float(np.mean(landmark_index.reprojection_errors[landmark_indices])),
            )
        )
    return ReferencePatchMapletBank(units=tuple(units), feature_dim=feature_dim)


def match_query_patches_to_reference_patch_maplets(
    query_feature_map: np.ndarray,
    bank: ReferencePatchMapletBank,
    config: ReferencePatchMapletConfig,
    image_width: int,
    image_height: int,
) -> list[ReferencePatchMapletMatch]:
    query_features, query_token_indices = _flatten_query_features(query_feature_map, int(config.query_token_step))
    if len(bank) == 0 or query_features.shape[0] == 0:
        return []
    query_features, query_valid = normalize_rows(query_features)
    maplet_features, maplet_valid = normalize_rows(bank.features)
    valid_maplet_indices = np.flatnonzero(maplet_valid)
    if valid_maplet_indices.size == 0:
        return []
    maplet_features = maplet_features[valid_maplet_indices]
    centers = token_grid_xy(
        int(query_feature_map.shape[2]),
        int(query_feature_map.shape[1]),
        int(image_width),
        int(image_height),
        step=1,
    )
    matches: list[ReferencePatchMapletMatch] = []
    top_k = min(int(config.top_k), int(valid_maplet_indices.size))
    for start in range(0, query_features.shape[0], int(config.block_size)):
        end = min(start + int(config.block_size), query_features.shape[0])
        scores = query_features[start:end] @ maplet_features.T
        scores[~query_valid[start:end], :] = -np.inf
        if top_k == 1:
            local_top = np.argmax(scores, axis=1)[:, None]
        else:
            local_top = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
            row_order = np.argsort(-np.take_along_axis(scores, local_top, axis=1), axis=1)
            local_top = np.take_along_axis(local_top, row_order, axis=1)
        for row, local_indices in enumerate(local_top):
            query_row = start + row
            token_index = int(query_token_indices[query_row])
            for rank, local_idx in enumerate(local_indices.tolist()):
                similarity = float(scores[row, int(local_idx)])
                if not np.isfinite(similarity) or similarity < float(config.min_similarity):
                    continue
                unit = bank.units[int(valid_maplet_indices[int(local_idx)])]
                matches.append(
                    ReferencePatchMapletMatch(
                        token_index=token_index,
                        xy=np.asarray(centers[token_index], dtype=np.float64),
                        reference_image_id=unit.reference_image_id,
                        reference_token_index=int(unit.token_index),
                        unit_id=int(unit.unit_id),
                        support_track_ids=unit.track_ids,
                        similarity=similarity,
                        rank=int(rank),
                    )
                )
    matches.sort(key=lambda item: (-float(item.similarity), int(item.token_index), int(item.rank)))
    if config.max_matches is not None:
        matches = matches[: int(config.max_matches)]
    return matches


def evaluate_reference_patch_maplet_matches(
    matches: Sequence[ReferencePatchMapletMatch],
    positives: PatchPositiveSets,
    top_k: int = 1,
) -> dict[str, float | int]:
    grouped: dict[int, list[ReferencePatchMapletMatch]] = {}
    for match in matches:
        grouped.setdefault(int(match.token_index), []).append(match)
    nonempty_tokens = [token_id for token_id, positive in positives.by_token.items() if positive.count > 0]
    if not nonempty_tokens:
        return {
            "evaluated_token_count": 0,
            "maplet_at_1": 0.0,
            f"maplet_at_{int(top_k)}": 0.0,
            "positive_landmark_recall_at_1": 0.0,
            "mean_overlap_at_1": 0.0,
        }
    hit1 = 0
    hitk = 0
    recall1 = []
    overlap1 = []
    for token_id in nonempty_tokens:
        positive_tracks = positives.by_token[int(token_id)].track_ids
        ranked = sorted(grouped.get(int(token_id), ()), key=lambda item: int(item.rank))
        if ranked:
            overlap = len(set(ranked[0].support_track_ids).intersection(positive_tracks))
            overlap1.append(float(overlap))
            recall1.append(float(overlap / max(len(positive_tracks), 1)))
            if overlap > 0:
                hit1 += 1
        else:
            overlap1.append(0.0)
            recall1.append(0.0)
        for match in ranked[: int(top_k)]:
            if set(match.support_track_ids).intersection(positive_tracks):
                hitk += 1
                break
    return {
        "evaluated_token_count": int(len(nonempty_tokens)),
        "matched_token_count": int(sum(1 for token_id in nonempty_tokens if token_id in grouped)),
        "maplet_at_1": float(hit1 / max(len(nonempty_tokens), 1)),
        f"maplet_at_{int(top_k)}": float(hitk / max(len(nonempty_tokens), 1)),
        "positive_landmark_recall_at_1": float(np.mean(recall1)) if recall1 else 0.0,
        "mean_overlap_at_1": float(np.mean(overlap1)) if overlap1 else 0.0,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))).astype(np.float32, copy=False)


def _fit_logistic_model(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int,
    lr: float,
    l2: float,
    class_balance: bool,
) -> tuple[np.ndarray, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if y.shape[0] != x.shape[0]:
        raise ValueError("labels must have shape (N,)")
    if x.shape[0] == 0:
        return np.zeros((x.shape[1],), dtype=np.float32), 0.0
    prior = np.clip(float(np.mean(y)), 1e-4, 1.0 - 1e-4)
    if not np.any(y > 0.5) or not np.any(y < 0.5):
        return np.zeros((x.shape[1],), dtype=np.float32), float(np.log(prior / (1.0 - prior)))
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    bias = float(np.log(prior / (1.0 - prior)))
    if bool(class_balance):
        positives = float(np.sum(y > 0.5))
        negatives = float(np.sum(y <= 0.5))
        sample_weights = np.where(
            y > 0.5,
            x.shape[0] / max(2.0 * positives, 1.0),
            x.shape[0] / max(2.0 * negatives, 1.0),
        ).astype(np.float32)
    else:
        sample_weights = np.ones((x.shape[0],), dtype=np.float32)
    weight_norm = max(float(np.sum(sample_weights)), 1.0)
    for _step in range(int(steps)):
        probs = _sigmoid(x @ weights + bias)
        error = (probs - y) * sample_weights
        grad_w = (x.T @ error) / weight_norm + float(l2) * weights
        grad_b = float(np.sum(error) / weight_norm)
        weights = (weights - float(lr) * grad_w).astype(np.float32, copy=False)
        bias = float(bias - float(lr) * grad_b)
    return weights, bias


def _maplet_verifier_feature_matrix(
    matches: Sequence[ReferencePatchMapletMatch],
    bank: ReferencePatchMapletBank,
) -> np.ndarray:
    unit_by_id = {int(unit.unit_id): unit for unit in bank.units}
    rows = []
    for match in matches:
        unit = unit_by_id.get(int(match.unit_id))
        if unit is None:
            continue
        maplet_similarity = (float(match.similarity) + 1.0) * 0.5
        rank_score = 1.0 / (1.0 + float(match.rank))
        support_count_score = float(np.clip(np.log1p(float(unit.num_landmarks)) / 8.0, 0.0, 1.0))
        mean_observation_score = float(np.clip(np.log1p(float(unit.mean_observation_count)) / 8.0, 0.0, 1.0))
        reprojection_score = float(1.0 / (1.0 + max(float(unit.mean_reprojection_error), 0.0) / 4.0))
        rows.append(
            [
                maplet_similarity,
                rank_score,
                support_count_score,
                mean_observation_score,
                reprojection_score,
            ]
        )
    if not rows:
        return np.zeros((0, len(MapletVerifierModel(np.zeros((5,), dtype=np.float32), 0.0).feature_names)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def collect_maplet_verifier_training_examples(
    matches: Sequence[ReferencePatchMapletMatch],
    bank: ReferencePatchMapletBank,
    positives: PatchPositiveSets,
) -> tuple[np.ndarray, np.ndarray]:
    features = _maplet_verifier_feature_matrix(matches, bank)
    labels = []
    for match in matches:
        positive = positives.by_token.get(int(match.token_index))
        positive_tracks = set() if positive is None else positive.track_ids
        labels.append(1.0 if set(match.support_track_ids).intersection(positive_tracks) else 0.0)
    if features.shape[0] != len(labels):
        raise ValueError("maplet verifier feature/label count mismatch")
    return features, np.asarray(labels, dtype=np.float32)


def fit_maplet_verifier_model(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 300,
    lr: float = 0.2,
    l2: float = 1e-3,
    class_balance: bool = True,
) -> MapletVerifierModel:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("maplet verifier features must have shape (N, 5)")
    weights, bias = _fit_logistic_model(x, labels, int(steps), float(lr), float(l2), bool(class_balance))
    return MapletVerifierModel(weights=weights, bias=bias)


def score_maplet_verifier_model(model: MapletVerifierModel, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model.weights.shape[0]:
        raise ValueError("maplet verifier feature shape does not match model")
    return (x @ model.weights + float(model.bias)).astype(np.float32, copy=False)


def apply_maplet_verifier_to_matches(
    matches: Sequence[ReferencePatchMapletMatch],
    bank: ReferencePatchMapletBank,
    model: MapletVerifierModel,
    score_weight: float = 0.5,
    keep_fraction: float | None = None,
) -> list[ReferencePatchMapletMatch]:
    if keep_fraction is not None and not 0.0 < float(keep_fraction) <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    match_list = list(matches)
    if not match_list:
        return []
    logits = score_maplet_verifier_model(model, _maplet_verifier_feature_matrix(match_list, bank))
    rescored = []
    for match, logit in zip(match_list, logits.tolist()):
        rescored.append((float(match.similarity) + float(score_weight) * float(logit), match))
    grouped: dict[int, list[tuple[float, ReferencePatchMapletMatch]]] = {}
    for score, match in rescored:
        grouped.setdefault(int(match.token_index), []).append((float(score), match))
    output: list[ReferencePatchMapletMatch] = []
    for token_index in sorted(grouped):
        group = sorted(grouped[token_index], key=lambda item: (-float(item[0]), int(item[1].rank), int(item[1].unit_id)))
        if keep_fraction is not None:
            keep_count = max(1, int(np.ceil(len(group) * float(keep_fraction))))
            group = group[:keep_count]
        for rank, (score, match) in enumerate(group):
            output.append(
                ReferencePatchMapletMatch(
                    token_index=int(match.token_index),
                    xy=np.asarray(match.xy, dtype=np.float64).reshape(2),
                    reference_image_id=match.reference_image_id,
                    reference_token_index=int(match.reference_token_index),
                    unit_id=int(match.unit_id),
                    support_track_ids=match.support_track_ids,
                    similarity=float(score),
                    rank=int(rank),
                    source="reference_patch_maplet_verified",
                )
            )
    output.sort(key=lambda item: (-float(item.similarity), int(item.token_index), int(item.rank)))
    return output


def _support_quality_order(landmark_index: LandmarkMapIndex, landmark_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices
    order = np.lexsort(
        (
            np.asarray(landmark_index.reprojection_errors[indices], dtype=np.float32),
            np.asarray(landmark_index.mean_variances[indices], dtype=np.float32),
            -np.asarray(landmark_index.observation_counts[indices], dtype=np.float32),
        )
    )
    return indices[order]


def _support_quality_scores(landmark_index: LandmarkMapIndex, landmark_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    counts = np.log1p(np.maximum(np.asarray(landmark_index.observation_counts[indices], dtype=np.float32), 0.0))
    if counts.size and float(np.max(counts)) > 0.0:
        counts = counts / float(np.max(counts))
    variances = np.asarray(landmark_index.mean_variances[indices], dtype=np.float32)
    variance_scale = float(np.percentile(variances, 90.0)) if variances.size else 1.0
    if variance_scale <= 1e-8:
        variance_scale = 1.0
    variance_score = 1.0 / (1.0 + np.maximum(variances, 0.0) / variance_scale)
    reproj = np.asarray(landmark_index.reprojection_errors[indices], dtype=np.float32)
    reproj_scale = float(np.percentile(reproj, 90.0)) if reproj.size else 1.0
    if reproj_scale <= 1e-8:
        reproj_scale = 1.0
    reproj_score = 1.0 / (1.0 + np.maximum(reproj, 0.0) / reproj_scale)
    return np.clip(counts * variance_score * reproj_score, 1e-6, 1.0).astype(np.float32, copy=False)


def _support_feature_scores(
    landmark_index: LandmarkMapIndex,
    landmark_indices: np.ndarray,
    maplet_feature: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    point_features, point_valid = normalize_rows(np.asarray(landmark_index.features[indices], dtype=np.float32))
    maplet = _normalize_vector(maplet_feature)
    scores = point_features @ maplet.reshape(-1, 1)
    scores = scores.reshape(-1).astype(np.float32, copy=False)
    scores[~point_valid] = -1.0
    return scores


def _support_selector_feature_matrix(
    match: ReferencePatchMapletMatch,
    unit: ReferencePatchMaplet,
    landmark_index: LandmarkMapIndex,
    landmark_indices: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.zeros((0, len(SupportSelectorModel(np.zeros((5,), dtype=np.float32), 0.0).feature_names)), dtype=np.float32)
    feature_scores = _support_feature_scores(landmark_index, indices, unit.feature)
    quality = _support_quality_scores(landmark_index, indices)
    maplet_similarity = np.full((indices.size,), (float(match.similarity) + 1.0) * 0.5, dtype=np.float32)
    support_feature_similarity = np.clip((feature_scores + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    rank_score = np.full((indices.size,), 1.0 / (1.0 + float(match.rank)), dtype=np.float32)
    support_count_score = np.full(
        (indices.size,),
        np.clip(np.log1p(float(unit.num_landmarks)) / 8.0, 0.0, 1.0),
        dtype=np.float32,
    )
    return np.stack(
        [maplet_similarity, support_feature_similarity, quality, rank_score, support_count_score],
        axis=1,
    ).astype(np.float32, copy=False)


def collect_support_selector_training_examples(
    matches: Sequence[ReferencePatchMapletMatch],
    bank: ReferencePatchMapletBank,
    landmark_index: LandmarkMapIndex,
    positives: PatchPositiveSets,
) -> tuple[np.ndarray, np.ndarray]:
    unit_by_id = {int(unit.unit_id): unit for unit in bank.units}
    features = []
    labels = []
    for match in matches:
        unit = unit_by_id.get(int(match.unit_id))
        if unit is None or unit.landmark_indices.size == 0:
            continue
        positive = positives.by_token.get(int(match.token_index))
        positive_tracks = set() if positive is None else positive.track_ids
        feature_matrix = _support_selector_feature_matrix(match, unit, landmark_index, unit.landmark_indices)
        for row, landmark_idx in zip(feature_matrix, unit.landmark_indices.tolist()):
            features.append(row)
            labels.append(1.0 if int(landmark_index.track_ids[int(landmark_idx)]) in positive_tracks else 0.0)
    if not features:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(labels, dtype=np.float32)


def fit_support_selector_model(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 300,
    lr: float = 0.2,
    l2: float = 1e-3,
    class_balance: bool = True,
) -> SupportSelectorModel:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("support selector features must have shape (N, 5)")
    if y.shape[0] != x.shape[0]:
        raise ValueError("support selector labels must have shape (N,)")
    if x.shape[0] == 0:
        return SupportSelectorModel(np.zeros((5,), dtype=np.float32), 0.0)
    if not np.any(y > 0.5) or not np.any(y < 0.5):
        prior = np.clip(float(np.mean(y)), 1e-4, 1.0 - 1e-4)
        return SupportSelectorModel(np.zeros((5,), dtype=np.float32), float(np.log(prior / (1.0 - prior))))
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    bias = float(np.log(np.clip(float(np.mean(y)), 1e-4, 1.0 - 1e-4) / np.clip(1.0 - float(np.mean(y)), 1e-4, 1.0)))
    if bool(class_balance):
        positives = float(np.sum(y > 0.5))
        negatives = float(np.sum(y <= 0.5))
        sample_weights = np.where(
            y > 0.5,
            x.shape[0] / max(2.0 * positives, 1.0),
            x.shape[0] / max(2.0 * negatives, 1.0),
        ).astype(np.float32)
    else:
        sample_weights = np.ones((x.shape[0],), dtype=np.float32)
    weight_norm = max(float(np.sum(sample_weights)), 1.0)
    for _step in range(int(steps)):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        error = (probs.astype(np.float32) - y) * sample_weights
        grad_w = (x.T @ error) / weight_norm + float(l2) * weights
        grad_b = float(np.sum(error) / weight_norm)
        weights = (weights - float(lr) * grad_w).astype(np.float32, copy=False)
        bias = float(bias - float(lr) * grad_b)
    return SupportSelectorModel(weights=weights, bias=bias)


def _score_support_selector_model(model: SupportSelectorModel, feature_matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != model.weights.shape[0]:
        raise ValueError("feature matrix shape does not match support selector model")
    return (features @ model.weights + float(model.bias)).astype(np.float32, copy=False)


def _select_support_indices(
    unit: ReferencePatchMaplet,
    match: ReferencePatchMapletMatch,
    landmark_index: LandmarkMapIndex,
    support_per_maplet: int,
    support_recovery_mode: str,
    support_selector_model: SupportSelectorModel | None = None,
) -> np.ndarray:
    ordered = _support_quality_order(landmark_index, unit.landmark_indices)
    if support_recovery_mode == "quality_point":
        return ordered if int(support_per_maplet) <= 0 else ordered[: int(support_per_maplet)]
    if support_recovery_mode not in {"feature_consistent_point", "learned_point"}:
        raise ValueError(f"unsupported point support recovery mode: {support_recovery_mode}")
    indices = np.asarray(unit.landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices
    if support_recovery_mode == "learned_point":
        if support_selector_model is None:
            raise ValueError("support_selector_model is required for learned_point recovery")
        feature_matrix = _support_selector_feature_matrix(match, unit, landmark_index, indices)
        combined = _score_support_selector_model(support_selector_model, feature_matrix)
    else:
        feature_scores = _support_feature_scores(landmark_index, indices, unit.feature)
        quality = _support_quality_scores(landmark_index, indices)
        combined = feature_scores + 0.05 * quality
    order = np.argsort(-combined)
    selected = indices[order]
    return selected if int(support_per_maplet) <= 0 else selected[: int(support_per_maplet)]


def _soft_support_xyz(unit: ReferencePatchMaplet, landmark_index: LandmarkMapIndex) -> tuple[np.ndarray, float]:
    indices = np.asarray(unit.landmark_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.zeros((3,), dtype=np.float64), 0.0
    feature_scores = np.maximum(_support_feature_scores(landmark_index, indices, unit.feature), 0.0)
    quality = _support_quality_scores(landmark_index, indices)
    weights = feature_scores * quality
    if not np.any(weights > 0.0):
        weights = quality
    if not np.any(weights > 0.0):
        weights = np.ones((indices.size,), dtype=np.float32)
    weights = weights.astype(np.float64, copy=False)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    xyz = np.sum(np.asarray(landmark_index.xyz[indices], dtype=np.float64) * weights[:, None], axis=0)
    confidence = float(np.max(weights))
    return xyz.astype(np.float64, copy=False), confidence


def expand_reference_patch_maplet_matches_to_query_to_3d(
    matches: Sequence[ReferencePatchMapletMatch],
    bank: ReferencePatchMapletBank,
    landmark_index: LandmarkMapIndex,
    support_per_maplet: int = 1,
    max_matches: int | None = None,
    support_recovery_mode: str = "quality_point",
    measurement_sigma_px: float | None = None,
    uncertainty_aware_sort: bool = False,
    support_selector_model: SupportSelectorModel | None = None,
) -> list[QueryTo3DMatch]:
    """Expand reference patch-maplet matches into 2D-3D candidates for PnP."""

    if int(support_per_maplet) < 0:
        raise ValueError("support_per_maplet must be non-negative")
    if max_matches is not None and int(max_matches) <= 0:
        raise ValueError("max_matches must be positive")
    if support_recovery_mode not in {"quality_point", "feature_consistent_point", "soft_xyz", "learned_point"}:
        raise ValueError(
            "support_recovery_mode must be one of: quality_point, feature_consistent_point, soft_xyz, learned_point"
        )
    if measurement_sigma_px is not None and float(measurement_sigma_px) <= 0.0:
        raise ValueError("measurement_sigma_px must be positive")
    unit_by_id = {int(unit.unit_id): unit for unit in bank.units}
    expanded: list[QueryTo3DMatch] = []
    for match in matches:
        unit = unit_by_id.get(int(match.unit_id))
        if unit is None:
            continue
        if support_recovery_mode == "soft_xyz":
            xyz, confidence = _soft_support_xyz(unit, landmark_index)
            sigma = None if measurement_sigma_px is None else float(measurement_sigma_px) * (1.25 - 0.25 * confidence)
            expanded.append(
                QueryTo3DMatch(
                    token_index=int(match.token_index),
                    xy=np.asarray(match.xy, dtype=np.float64).reshape(2),
                    track_id=-(len(expanded) + 1),
                    xyz=xyz,
                    similarity=float(match.similarity),
                    ratio=1.0,
                    landmark_variance=float(np.mean(landmark_index.mean_variances[unit.landmark_indices]))
                    if unit.landmark_indices.size
                    else 0.0,
                    source="reference_patch_maplet_soft_xyz",
                    observation_count=int(unit.num_landmarks),
                    visibility_count=int(unit.num_landmarks),
                    landmark_reprojection_error=float(unit.mean_reprojection_error),
                    token_match_rank=int(match.rank),
                    pnp_uncertainty_scale=None if sigma is None else float(sigma / max(float(measurement_sigma_px), 1e-6)),
                    measurement_sigma_px=sigma,
                )
            )
            continue
        ordered = _select_support_indices(
            unit,
            match,
            landmark_index,
            int(support_per_maplet),
            support_recovery_mode,
            support_selector_model=support_selector_model,
        )
        for landmark_idx in ordered.tolist():
            idx = int(landmark_idx)
            sigma = None if measurement_sigma_px is None else float(measurement_sigma_px)
            expanded.append(
                QueryTo3DMatch(
                    token_index=int(match.token_index),
                    xy=np.asarray(match.xy, dtype=np.float64).reshape(2),
                    track_id=int(landmark_index.track_ids[idx]),
                    xyz=np.asarray(landmark_index.xyz[idx], dtype=np.float64).reshape(3),
                    similarity=float(match.similarity),
                    ratio=1.0,
                    landmark_variance=float(landmark_index.mean_variances[idx]),
                    source="reference_patch_maplet",
                    observation_count=int(landmark_index.observation_counts[idx]),
                    visibility_count=int(landmark_index.observation_counts[idx]),
                    landmark_reprojection_error=float(landmark_index.reprojection_errors[idx]),
                    token_match_rank=int(match.rank),
                    pnp_uncertainty_scale=None if sigma is None else 1.0,
                    measurement_sigma_px=sigma,
                )
            )
    if uncertainty_aware_sort:
        expanded.sort(
            key=lambda item: (
                -float(item.similarity) / max(float(item.measurement_sigma_px or 1.0), 1e-6),
                int(item.token_index),
                0 if item.token_match_rank is None else int(item.token_match_rank),
                int(item.track_id),
            )
        )
    else:
        expanded.sort(
            key=lambda item: (
                -float(item.similarity),
                int(item.token_index),
                0 if item.token_match_rank is None else int(item.token_match_rank),
                int(item.track_id),
            )
        )
    if max_matches is not None:
        expanded = expanded[: int(max_matches)]
    return expanded


def reference_patch_maplet_positive_stats(
    positives: PatchPositiveSets,
    bank: ReferencePatchMapletBank,
) -> dict[str, float | int]:
    support_sets = [set(unit.track_ids) for unit in bank.units]
    positive_counts = []
    best_overlaps = []
    for positive in positives.by_token.values():
        if positive.count <= 0:
            continue
        count = 0
        best = 0
        positive_tracks = set(positive.track_ids)
        for support in support_sets:
            overlap = len(support.intersection(positive_tracks))
            if overlap > 0:
                count += 1
                best = max(best, overlap)
        positive_counts.append(float(count))
        best_overlaps.append(float(best))
    if not positive_counts:
        return {
            "positive_token_count": 0,
            "covered_positive_token_fraction": 0.0,
            "mean_positive_maplets_per_token": 0.0,
            "median_positive_maplets_per_token": 0.0,
            "mean_best_overlap_per_token": 0.0,
        }
    counts = np.asarray(positive_counts, dtype=np.float32)
    best = np.asarray(best_overlaps, dtype=np.float32)
    return {
        "positive_token_count": int(counts.size),
        "covered_positive_token_fraction": float(np.mean(counts > 0.0)),
        "mean_positive_maplets_per_token": float(np.mean(counts)),
        "median_positive_maplets_per_token": float(np.median(counts)),
        "mean_best_overlap_per_token": float(np.mean(best)),
    }
