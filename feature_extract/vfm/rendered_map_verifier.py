"""Verify candidates with rendered selected-map descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank
from feature_extract.vfm.score_table import ScoreRow
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl
from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapTrackObservation


@dataclass(frozen=True)
class RenderedMapEvidenceScore:
    score: float
    mean_similarity: float
    inlier_fraction: float
    match_count: int


def build_track_visibility_index(path: Path) -> dict[str, set[int]]:
    """Return image_id -> visible COLMAP track ids from observation JSONL."""

    index: dict[str, set[int]] = {}
    for observation in load_colmap_track_observations_jsonl(Path(path)):
        index.setdefault(observation.image_id, set()).add(int(observation.track_id))
    return index


def build_track_observation_index(path: Path) -> dict[str, list[ColmapTrackObservation]]:
    """Return image_id -> COLMAP track observations from observation JSONL."""

    index: dict[str, list[ColmapTrackObservation]] = {}
    for observation in load_colmap_track_observations_jsonl(Path(path)):
        index.setdefault(observation.image_id, []).append(observation)
    return index


def build_track_xyz_index(path: Path) -> dict[int, np.ndarray]:
    """Return track_id -> 3D point from COLMAP observation JSONL."""

    index: dict[int, np.ndarray] = {}
    for observation in load_colmap_track_observations_jsonl(Path(path)):
        index.setdefault(int(observation.track_id), np.asarray(observation.xyz, dtype=np.float64))
    return index


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _l2_normalize_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1e-6)


def _track_feature_weight(mean_utility: float) -> float:
    return max(float(mean_utility), 0.0)


def rendered_selected_map_descriptor(
    reference_image: str,
    track_bank: SelectedTrackFeatureBank,
    visibility_index: Mapping[str, Iterable[int]],
) -> tuple[np.ndarray, int, float]:
    """Mean-pool selected track features visible in a candidate reference image."""

    visible_track_ids = set(int(track_id) for track_id in visibility_index.get(reference_image, ()))
    if not visible_track_ids:
        return np.zeros((track_bank.feature_dim,), dtype=np.float32), 0, 0.0

    weighted_features = []
    weights = []
    visible_in_bank = 0
    for track_id in sorted(visible_track_ids):
        if track_id not in track_bank.tracks:
            continue
        visible_in_bank += 1
        track = track_bank.tracks[track_id]
        weight = _track_feature_weight(track.mean_utility)
        if weight <= 0.0:
            continue
        weighted_features.append(np.asarray(track.mean_feature, dtype=np.float32).reshape(-1) * weight)
        weights.append(weight)
    features = weighted_features
    if not features:
        return np.zeros((track_bank.feature_dim,), dtype=np.float32), 0, 0.0
    descriptor = (np.stack(features, axis=0).sum(axis=0) / max(float(np.sum(weights)), 1e-6)).astype(
        np.float32,
        copy=False,
    )
    visibility_fraction = float(visible_in_bank / len(visible_track_ids))
    return _l2_normalize(descriptor), visible_in_bank, visibility_fraction


def _nearest_token_xy(
    xy: tuple[float, float],
    image_width: int,
    image_height: int,
    token_width: int,
    token_height: int,
) -> tuple[int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x_norm = float(xy[0]) / max(float(image_width - 1), 1.0)
    y_norm = float(xy[1]) / max(float(image_height - 1), 1.0)
    x_idx = int(round(np.clip(x_norm, 0.0, 1.0) * max(token_width - 1, 0)))
    y_idx = int(round(np.clip(y_norm, 0.0, 1.0) * max(token_height - 1, 0)))
    return x_idx, y_idx


def rendered_selected_map_token_grid(
    reference_image: str,
    track_bank: SelectedTrackFeatureBank,
    observation_index: Mapping[str, Iterable[ColmapTrackObservation]],
    token_height: int,
    token_width: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Rasterize visible selected track features into a sparse token grid.

    The grid uses COLMAP observation coordinates from the candidate reference
    image, so two references with the same global track mean can still score
    differently when their selected tracks land at different token cells.
    """

    observations = list(observation_index.get(reference_image, ()))
    grid_sum = np.zeros((track_bank.feature_dim, token_height, token_width), dtype=np.float32)
    grid_count = np.zeros((token_height, token_width), dtype=np.float32)
    used_observations = 0
    for observation in observations:
        track_id = int(observation.track_id)
        if track_id not in track_bank.tracks:
            continue
        if observation.image_width is None or observation.image_height is None:
            continue
        feature = np.asarray(track_bank.tracks[track_id].mean_feature, dtype=np.float32).reshape(-1)
        if feature.shape[0] != track_bank.feature_dim:
            raise ValueError("track feature dimension does not match track bank feature_dim")
        weight = _track_feature_weight(track_bank.tracks[track_id].mean_utility)
        used_observations += 1
        if weight <= 0.0:
            continue
        x_idx, y_idx = _nearest_token_xy(
            observation.xy,
            int(observation.image_width),
            int(observation.image_height),
            token_width,
            token_height,
        )
        grid_sum[:, y_idx, x_idx] += feature * weight
        grid_count[y_idx, x_idx] += weight

    mask = grid_count > 0.0
    if np.any(mask):
        grid_sum[:, mask] /= grid_count[mask][None, :]
        flat = grid_sum.reshape(track_bank.feature_dim, token_height * token_width)
        flat_mask = mask.reshape(token_height * token_width)
        flat[:, flat_mask] = np.stack([_l2_normalize(flat[:, idx]) for idx in np.where(flat_mask)[0]], axis=1)
        grid_sum = flat.reshape(track_bank.feature_dim, token_height, token_width)
    visibility_fraction = 0.0 if not observations else float(used_observations / len(observations))
    return grid_sum, mask, used_observations, visibility_fraction


def project_xyz_to_image(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[float, float] | None:
    """Project a world-space 3D point with a COLMAP-style world-to-camera pose."""

    pose = np.asarray(pose_w2c, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("pose_w2c must have shape (4, 4)")
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    camera_point = pose @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    if float(camera_point[2]) <= 1e-6:
        return None
    x_norm = float(camera_point[0] / camera_point[2])
    y_norm = float(camera_point[1] / camera_point[2])
    if camera.model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = camera.params[:3]
    elif camera.model_id == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = camera.params[:4]
        return float(fx * x_norm + cx), float(fy * y_norm + cy)
    elif camera.model_id == 2:  # SIMPLE_RADIAL: f, cx, cy, k
        f, cx, cy, k = camera.params[:4]
        radial = 1.0 + float(k) * (x_norm * x_norm + y_norm * y_norm)
        x_norm *= radial
        y_norm *= radial
    else:
        raise ValueError(f"unsupported COLMAP camera model id for projection: {camera.model_id}")
    return float(f * x_norm + cx), float(f * y_norm + cy)


def projected_selected_map_token_grid(
    track_bank: SelectedTrackFeatureBank,
    track_xyz_index: Mapping[int, np.ndarray],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_height: int,
    token_width: int,
    visible_track_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Project selected 3D track features into a candidate camera token grid."""

    grid_sum = np.zeros((track_bank.feature_dim, token_height, token_width), dtype=np.float32)
    grid_count = np.zeros((token_height, token_width), dtype=np.float32)
    used_tracks = 0
    if visible_track_ids is None:
        candidate_track_ids = tuple(track_bank.tracks)
        visibility_denominator = len(track_bank.tracks)
    else:
        candidate_track_ids = tuple(int(track_id) for track_id in visible_track_ids)
        visibility_denominator = len(candidate_track_ids)
    for track_id in candidate_track_ids:
        track = track_bank.tracks.get(int(track_id))
        if track is None:
            continue
        xyz = track_xyz_index.get(int(track_id))
        if xyz is None:
            continue
        xy = project_xyz_to_image(xyz, pose_w2c, camera)
        if xy is None:
            continue
        if xy[0] < 0.0 or xy[0] > camera.width - 1 or xy[1] < 0.0 or xy[1] > camera.height - 1:
            continue
        feature = np.asarray(track.mean_feature, dtype=np.float32).reshape(-1)
        if feature.shape[0] != track_bank.feature_dim:
            raise ValueError("track feature dimension does not match track bank feature_dim")
        weight = _track_feature_weight(track.mean_utility)
        used_tracks += 1
        if weight <= 0.0:
            continue
        x_idx, y_idx = _nearest_token_xy(
            (float(xy[0]), float(xy[1])),
            int(camera.width),
            int(camera.height),
            token_width,
            token_height,
        )
        grid_sum[:, y_idx, x_idx] += feature * weight
        grid_count[y_idx, x_idx] += weight

    mask = grid_count > 0.0
    if np.any(mask):
        grid_sum[:, mask] /= grid_count[mask][None, :]
        flat = grid_sum.reshape(track_bank.feature_dim, token_height * token_width)
        flat_mask = mask.reshape(token_height * token_width)
        flat[:, flat_mask] = np.stack([_l2_normalize(flat[:, idx]) for idx in np.where(flat_mask)[0]], axis=1)
        grid_sum = flat.reshape(track_bank.feature_dim, token_height, token_width)
    visibility_fraction = 0.0 if visibility_denominator == 0 else float(used_tracks / visibility_denominator)
    return grid_sum, mask, used_tracks, visibility_fraction


def _validated_query_index(manifest: TokenBankManifest) -> dict[str, TokenBankRecord]:
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


def _load_query_layer_from_index(
    records: Mapping[str, TokenBankRecord],
    query_id: str,
    layer_name: str,
) -> np.ndarray:
    if query_id not in records:
        raise ValueError(f"query token record not found: {query_id}")
    record = records[query_id]
    with np.load(record.token_path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
        tokens = np.asarray(data[layer_name], dtype=np.float32)
    if tokens.ndim != 3:
        raise ValueError("query token feature map must have shape (C, H, W)")
    return tokens


def _load_query_layer(manifest: TokenBankManifest, query_id: str, layer_name: str) -> np.ndarray:
    return _load_query_layer_from_index(_validated_query_index(manifest), query_id, layer_name)


def query_selected_descriptor(
    manifest: TokenBankManifest,
    query_id: str,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    device: str = "cpu",
    record_index: Mapping[str, TokenBankRecord] | None = None,
) -> np.ndarray:
    """Run the selector on a dense query token map and mean-pool selected tokens."""

    if record_index is None:
        tokens = _load_query_layer(manifest, query_id, layer_name)
    else:
        tokens = _load_query_layer_from_index(record_index, query_id, layer_name)
    if tokens.shape[0] != selector.input_dim:
        raise ValueError(f"query token channels {tokens.shape[0]} do not match selector input_dim {selector.input_dim}")

    torch_device = torch.device(device)
    selector = selector.to(torch_device)
    selector.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(tokens[None, ...], dtype=torch.float32, device=torch_device)
        selected = selector(tensor).selected
        descriptor = selected.mean(dim=(2, 3))
        descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-6)
    return descriptor[0].detach().cpu().numpy().astype(np.float32, copy=False)


def query_selected_feature_map(
    manifest: TokenBankManifest,
    query_id: str,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    device: str = "cpu",
    record_index: Mapping[str, TokenBankRecord] | None = None,
) -> np.ndarray:
    """Run the selector on a dense query token map and keep its token grid."""

    if record_index is None:
        tokens = _load_query_layer(manifest, query_id, layer_name)
    else:
        tokens = _load_query_layer_from_index(record_index, query_id, layer_name)
    if tokens.shape[0] != selector.input_dim:
        raise ValueError(f"query token channels {tokens.shape[0]} do not match selector input_dim {selector.input_dim}")

    torch_device = torch.device(device)
    selector = selector.to(torch_device)
    selector.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(tokens[None, ...], dtype=torch.float32, device=torch_device)
        selected = selector(tensor).selected
    return selected[0].detach().cpu().numpy().astype(np.float32, copy=False)


def sparse_rendered_map_evidence(
    query_feature_map: np.ndarray,
    rendered_grid: np.ndarray,
    rendered_mask: np.ndarray,
    local_radius: int = 0,
    inlier_threshold: float = 0.5,
    inlier_weight: float = 0.0,
) -> RenderedMapEvidenceScore:
    """Summarize local feature evidence for sparse rendered selected-map cells."""

    if local_radius < 0:
        raise ValueError("local_radius must be non-negative")
    if query_feature_map.shape != rendered_grid.shape:
        raise ValueError("query and rendered feature maps must have matching shape")
    if rendered_mask.shape != query_feature_map.shape[1:]:
        raise ValueError("rendered mask shape must match token grid")
    ys, xs = np.where(rendered_mask)
    if len(ys) == 0:
        return RenderedMapEvidenceScore(
            score=-1.0,
            mean_similarity=-1.0,
            inlier_fraction=0.0,
            match_count=0,
        )
    _channels, height, width = query_feature_map.shape
    scores: list[float] = []
    for y_idx, x_idx in zip(ys, xs):
        rendered_feature = _l2_normalize(rendered_grid[:, y_idx, x_idx])
        y0 = max(0, int(y_idx) - local_radius)
        y1 = min(height, int(y_idx) + local_radius + 1)
        x0 = max(0, int(x_idx) - local_radius)
        x1 = min(width, int(x_idx) + local_radius + 1)
        patch = query_feature_map[:, y0:y1, x0:x1].reshape(query_feature_map.shape[0], -1)
        patch = _l2_normalize_columns(patch)
        scores.append(float(np.max(rendered_feature.reshape(1, -1) @ patch)))
    similarities = np.asarray(scores, dtype=np.float32)
    mean_similarity = float(np.mean(similarities))
    inlier_fraction = float(np.mean(similarities >= float(inlier_threshold)))
    return RenderedMapEvidenceScore(
        score=float(mean_similarity + float(inlier_weight) * inlier_fraction),
        mean_similarity=mean_similarity,
        inlier_fraction=inlier_fraction,
        match_count=int(similarities.size),
    )


def sparse_rendered_map_score(
    query_feature_map: np.ndarray,
    rendered_grid: np.ndarray,
    rendered_mask: np.ndarray,
    local_radius: int = 0,
) -> float:
    """Score sparse rendered selected-map cells against a query selected map."""

    return sparse_rendered_map_evidence(
        query_feature_map,
        rendered_grid,
        rendered_mask,
        local_radius=local_radius,
    ).score


def _validate_rendered_track_bank(
    track_bank: SelectedTrackFeatureBank,
    selector: LocalizableFeatureSelector,
    forbidden_image_ids: Iterable[str],
    require_track_provenance: bool,
) -> None:
    if track_bank.feature_dim != selector.output_dim:
        raise ValueError(
            f"track bank feature_dim {track_bank.feature_dim} does not match selector output_dim {selector.output_dim}"
        )
    forbidden = set(str(image_id) for image_id in forbidden_image_ids)
    if not require_track_provenance and not forbidden:
        return
    for track_id, track in track_bank.tracks.items():
        provenance = tuple(str(image_id) for image_id in track.observation_image_ids)
        if require_track_provenance and not provenance:
            raise ValueError(f"track {track_id} is missing observation provenance")
        overlap = forbidden.intersection(provenance)
        if overlap:
            image_id = sorted(overlap)[0]
            raise ValueError(f"track {track_id} contains forbidden image {image_id!r}")


def score_candidate_bank_by_rendered_selected_map(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    track_bank: SelectedTrackFeatureBank,
    visibility_index: Mapping[str, Iterable[int]],
    selector: LocalizableFeatureSelector,
    layer_name: str,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: str = "cpu",
    require_track_provenance: bool = True,
    forbidden_image_ids: Iterable[str] | None = None,
) -> list[ScoreRow]:
    """Score candidates by query selected descriptor vs rendered selected-map descriptor."""

    query_index = _validated_query_index(query_manifest)
    forbidden = set(query_index) if forbidden_image_ids is None else set(str(item) for item in forbidden_image_ids)
    _validate_rendered_track_bank(
        track_bank,
        selector,
        forbidden_image_ids=forbidden,
        require_track_provenance=require_track_provenance,
    )
    query_cache: dict[str, np.ndarray] = {}
    rendered_cache: dict[str, tuple[np.ndarray, int, float]] = {}
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_cache:
            query_cache[candidate.query_id] = query_selected_descriptor(
                query_manifest,
                candidate.query_id,
                selector,
                layer_name,
                device=device,
                record_index=query_index,
            )

        if candidate.reference_image not in rendered_cache:
            rendered_cache[candidate.reference_image] = rendered_selected_map_descriptor(
                candidate.reference_image,
                track_bank,
                visibility_index,
            )
        rendered_descriptor, visible_track_count, visibility_fraction = rendered_cache[candidate.reference_image]
        if visible_track_count == 0:
            score = -1.0
        else:
            if rendered_descriptor.shape[0] != query_cache[candidate.query_id].shape[0]:
                raise ValueError(
                    f"rendered descriptor dimension {rendered_descriptor.shape[0]} does not match "
                    f"query descriptor dimension {query_cache[candidate.query_id].shape[0]}"
                )
            score = float(np.dot(query_cache[candidate.query_id], rendered_descriptor))
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=score,
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
                risk=1.0 - float(visibility_fraction),
                mean_similarity=None if visible_track_count == 0 else float(score),
                inlier_fraction=None,
                match_count=int(visible_track_count),
                visibility_fraction=float(visibility_fraction),
            )
        )
    return rows


def score_candidate_bank_by_sparse_rendered_selected_map(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    track_bank: SelectedTrackFeatureBank,
    observation_index: Mapping[str, Iterable[ColmapTrackObservation]],
    selector: LocalizableFeatureSelector,
    layer_name: str,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: str = "cpu",
    local_radius: int = 0,
    inlier_threshold: float = 0.5,
    inlier_weight: float = 0.0,
    risk_from_inliers: bool = False,
    require_track_provenance: bool = True,
    forbidden_image_ids: Iterable[str] | None = None,
) -> list[ScoreRow]:
    """Score candidates with sparse token-grid rendered selected-map features."""

    query_index = _validated_query_index(query_manifest)
    forbidden = set(query_index) if forbidden_image_ids is None else set(str(item) for item in forbidden_image_ids)
    _validate_rendered_track_bank(
        track_bank,
        selector,
        forbidden_image_ids=forbidden,
        require_track_provenance=require_track_provenance,
    )
    query_cache: dict[str, np.ndarray] = {}
    rendered_cache: dict[str, tuple[np.ndarray, np.ndarray, int, float]] = {}
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_cache:
            query_cache[candidate.query_id] = query_selected_feature_map(
                query_manifest,
                candidate.query_id,
                selector,
                layer_name,
                device=device,
                record_index=query_index,
            )
        query_map = query_cache[candidate.query_id]
        if candidate.reference_image not in rendered_cache:
            rendered_cache[candidate.reference_image] = rendered_selected_map_token_grid(
                candidate.reference_image,
                track_bank,
                observation_index,
                token_height=int(query_map.shape[1]),
                token_width=int(query_map.shape[2]),
            )
        rendered_grid, rendered_mask, visible_track_count, visibility_fraction = rendered_cache[
            candidate.reference_image
        ]
        evidence = sparse_rendered_map_evidence(
            query_map,
            rendered_grid,
            rendered_mask,
            local_radius=local_radius,
            inlier_threshold=inlier_threshold,
            inlier_weight=inlier_weight,
        )
        risk = 1.0 - float(visibility_fraction)
        if risk_from_inliers:
            risk = 1.0 - float(visibility_fraction) * float(evidence.inlier_fraction)
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=float(evidence.score),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
                risk=risk,
                mean_similarity=float(evidence.mean_similarity),
                inlier_fraction=float(evidence.inlier_fraction),
                match_count=int(evidence.match_count),
                visibility_fraction=float(visibility_fraction),
            )
        )
    return rows


def score_candidate_bank_by_projected_rendered_selected_map(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    track_bank: SelectedTrackFeatureBank,
    track_xyz_index: Mapping[int, np.ndarray],
    camera_by_image: Mapping[str, ColmapCamera],
    default_camera: ColmapCamera,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: str = "cpu",
    local_radius: int = 0,
    inlier_threshold: float = 0.5,
    inlier_weight: float = 0.0,
    risk_from_inliers: bool = False,
    visibility_index: Mapping[str, Iterable[int]] | None = None,
    require_track_provenance: bool = True,
    forbidden_image_ids: Iterable[str] | None = None,
) -> list[ScoreRow]:
    """Score candidates by projecting selected 3D map features with candidate poses."""

    query_index = _validated_query_index(query_manifest)
    forbidden = set(query_index) if forbidden_image_ids is None else set(str(item) for item in forbidden_image_ids)
    _validate_rendered_track_bank(
        track_bank,
        selector,
        forbidden_image_ids=forbidden,
        require_track_provenance=require_track_provenance,
    )
    query_cache: dict[str, np.ndarray] = {}
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.pose is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose")
        if candidate.query_id not in query_cache:
            query_cache[candidate.query_id] = query_selected_feature_map(
                query_manifest,
                candidate.query_id,
                selector,
                layer_name,
                device=device,
                record_index=query_index,
            )
        query_map = query_cache[candidate.query_id]
        camera = camera_by_image.get(candidate.query_id, default_camera)
        visible_track_ids = None
        if visibility_index is not None and candidate.reference_image is not None:
            visible_track_ids = visibility_index.get(candidate.reference_image, ())
        rendered_grid, rendered_mask, visible_track_count, visibility_fraction = projected_selected_map_token_grid(
            track_bank=track_bank,
            track_xyz_index=track_xyz_index,
            pose_w2c=np.asarray(candidate.pose, dtype=np.float64),
            camera=camera,
            token_height=int(query_map.shape[1]),
            token_width=int(query_map.shape[2]),
            visible_track_ids=visible_track_ids,
        )
        evidence = sparse_rendered_map_evidence(
            query_map,
            rendered_grid,
            rendered_mask,
            local_radius=local_radius,
            inlier_threshold=inlier_threshold,
            inlier_weight=inlier_weight,
        )
        risk = 1.0 - float(visibility_fraction)
        if risk_from_inliers:
            risk = 1.0 - float(visibility_fraction) * float(evidence.inlier_fraction)
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=float(evidence.score),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
                risk=risk,
                mean_similarity=float(evidence.mean_similarity),
                inlier_fraction=float(evidence.inlier_fraction),
                match_count=int(evidence.match_count),
                visibility_fraction=float(visibility_fraction),
            )
        )
    return rows
