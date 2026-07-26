"""Extract local descriptors only at requested real-image SfM observations."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation


SPATIAL_DETECTION_SELECTION_VERSION = "score_nms_soft_grid_quota_spatial_hash_v2"


@dataclass(frozen=True)
class ExtractedObservationFeatures:
    descriptors: np.ndarray
    detector_scores: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        scores = np.asarray(self.detector_scores, dtype=np.float32).reshape(-1)
        if descriptors.ndim != 2 or descriptors.shape[0] != scores.shape[0]:
            raise ValueError("descriptors must have shape (N, C) with one detector score per row")
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "detector_scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DetectedImageFeatures:
    """Sub-pixel detector points and descriptors from one real image."""

    xy: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    dispersions: np.ndarray
    image_sha256: str

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float32)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        dispersions = np.asarray(self.dispersions, dtype=np.float32).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("detector xy must have shape (N, 2)")
        if descriptors.ndim != 2 or descriptors.shape[0] != xy.shape[0]:
            raise ValueError("detector descriptors must have shape (N, C)")
        if scores.shape[0] != xy.shape[0] or dispersions.shape[0] != xy.shape[0]:
            raise ValueError("detector score arrays must have one value per point")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "dispersions", dispersions)


def _sample_dense_features(
    descriptor_map: torch.Tensor,
    score_map: torch.Tensor | None,
    observations: Sequence[ColmapTrackObservation],
) -> tuple[np.ndarray, np.ndarray]:
    if descriptor_map.ndim != 4 or int(descriptor_map.shape[0]) != 1:
        raise ValueError("descriptor_map must have shape (1, C, H, W)")
    if not observations:
        return (
            np.zeros((0, int(descriptor_map.shape[1])), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    dimensions = {
        (int(observation.image_width or 0), int(observation.image_height or 0))
        for observation in observations
    }
    if len(dimensions) != 1 or next(iter(dimensions))[0] <= 0 or next(iter(dimensions))[1] <= 0:
        raise ValueError("all observations of one image must share positive SfM image dimensions")
    image_width, image_height = next(iter(dimensions))
    xy = np.asarray([observation.xy for observation in observations], dtype=np.float32)
    return sample_dense_feature_points(
        descriptor_map,
        xy,
        image_width=int(image_width),
        image_height=int(image_height),
        score_map=score_map,
    )


def sample_dense_feature_points(
    descriptor_map: torch.Tensor | np.ndarray,
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    score_map: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample one dense map at arbitrary endpoint-coordinate pixels."""

    descriptors = torch.as_tensor(descriptor_map)
    if descriptors.ndim == 3:
        descriptors = descriptors.unsqueeze(0)
    if descriptors.ndim != 4 or int(descriptors.shape[0]) != 1:
        raise ValueError("descriptor_map must have shape (C, H, W) or (1, C, H, W)")
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if points.shape[0] == 0:
        return (
            np.zeros((0, int(descriptors.shape[1])), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    point_tensor = torch.as_tensor(
        points,
        dtype=descriptors.dtype,
        device=descriptors.device,
    )
    x = 2.0 * point_tensor[:, 0] / max(float(image_width - 1), 1.0) - 1.0
    y = 2.0 * point_tensor[:, 1] / max(float(image_height - 1), 1.0) - 1.0
    grid = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        descriptors,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, 0].T
    sampled = F.normalize(sampled.float(), p=2, dim=1)
    if score_map is None:
        scores = torch.full(
            (len(points),),
            float("nan"),
            dtype=torch.float32,
            device=descriptors.device,
        )
    else:
        score_tensor = torch.as_tensor(score_map, device=descriptors.device)
        if score_tensor.ndim == 3:
            score_tensor = score_tensor.unsqueeze(0)
        if score_tensor.ndim != 4 or score_tensor.shape[0] != 1 or score_tensor.shape[1] != 1:
            raise ValueError("score_map must have shape (1, 1, H, W)")
        scores = F.grid_sample(
            score_tensor.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0, :, 0]
    return (
        sampled.detach().cpu().numpy().astype(np.float32, copy=False),
        scores.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def spatially_diverse_detection_indices(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    nms_radius_px: float,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
    min_score: float | None = None,
) -> np.ndarray:
    """Select high-score detections with deterministic NMS and a soft grid quota."""

    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.shape[0] != values.shape[0]:
        raise ValueError("xy and scores must have the same length")
    if int(top_k) <= 0 or int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("top_k and grid dimensions must be positive")
    valid = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    valid &= (points[:, 0] >= 0.0) & (points[:, 0] <= float(image_width - 1))
    valid &= (points[:, 1] >= 0.0) & (points[:, 1] <= float(image_height - 1))
    if min_score is not None:
        valid &= values >= float(min_score)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.lexsort((candidates, -values[candidates]))
    ranked = candidates[order]
    limit = min(int(top_k), int(ranked.size))
    quota = max(1, int(np.ceil(float(limit) / float(int(grid_rows) * int(grid_cols)))))
    selected: list[int] = []
    selected_set: set[int] = set()
    cell_counts: dict[tuple[int, int], int] = {}
    radius2 = max(float(nms_radius_px), 0.0) ** 2
    radius = max(float(nms_radius_px), 0.0)
    nms_bins: dict[tuple[int, int], list[int]] = {}

    def cell(index: int) -> tuple[int, int]:
        col = int(np.clip(np.floor(points[index, 0] / max(float(image_width), 1.0) * int(grid_cols)), 0, int(grid_cols) - 1))
        row = int(np.clip(np.floor(points[index, 1] / max(float(image_height), 1.0) * int(grid_rows)), 0, int(grid_rows) - 1))
        return row, col

    def can_add(index: int, *, use_quota: bool) -> bool:
        if radius2 > 0.0 and selected:
            bin_x = int(np.floor(float(points[index, 0]) / radius))
            bin_y = int(np.floor(float(points[index, 1]) / radius))
            nearby = [
                selected_index
                for y_offset in (-1, 0, 1)
                for x_offset in (-1, 0, 1)
                for selected_index in nms_bins.get(
                    (bin_y + y_offset, bin_x + x_offset), ()
                )
            ]
            if nearby:
                delta = points[np.asarray(nearby, dtype=np.int64)] - points[int(index)]
                if np.any(np.sum(delta * delta, axis=1) <= radius2):
                    return False
        return not use_quota or cell_counts.get(cell(index), 0) < quota

    def add(index: int) -> None:
        selected.append(int(index))
        selected_set.add(int(index))
        key = cell(index)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        if radius > 0.0:
            bin_key = (
                int(np.floor(float(points[index, 1]) / radius)),
                int(np.floor(float(points[index, 0]) / radius)),
            )
            nms_bins.setdefault(bin_key, []).append(int(index))

    for use_quota in (True, False):
        for index in ranked.tolist():
            if len(selected) >= limit:
                break
            if int(index) in selected_set:
                continue
            if can_add(int(index), use_quota=use_quota):
                add(int(index))
    return np.asarray(selected, dtype=np.int64)


def _decode_rgb(path: Path) -> tuple[np.ndarray, str]:
    import cv2

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), digest


class AlikeDenseObservationExtractor:
    """Pretrained ALIKE dense descriptor/FPN sampled at requested points."""

    def __init__(
        self,
        *,
        device: str,
        matcha_repo: Path = Path("/root/matcha"),
        model_name: str = "alike-t",
    ) -> None:
        repo = Path(matcha_repo)
        if not repo.exists():
            raise FileNotFoundError(f"MATCHA repo not found: {repo}")
        repo_value = str(repo)
        if repo_value not in sys.path:
            sys.path.insert(0, repo_value)
        from third_party.alike.alike import ALike, configs  # type: ignore

        if str(model_name) not in configs:
            raise ValueError(f"unknown ALIKE model: {model_name}")
        config = dict(configs[str(model_name)])
        config.update(device=str(device), top_k=0, scores_th=0.0, n_limit=0)
        self.device = torch.device(str(device))
        self.model_name = str(model_name)
        self.model_path = Path(str(config["model_path"]))
        self.model = ALike(**config).to(self.device).eval()
        self._cached_dense_key: tuple[str, int, int] | None = None
        self._cached_descriptor_map: torch.Tensor | None = None
        self._cached_score_map: torch.Tensor | None = None
        self._cached_image_hash: str | None = None

    @torch.no_grad()
    def _dense_maps(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Extract once and reuse the dense ALIKE map within one query image."""

        import cv2

        key = (str(Path(image_path).resolve()), int(image_width), int(image_height))
        if (
            key == self._cached_dense_key
            and self._cached_descriptor_map is not None
            and self._cached_score_map is not None
            and self._cached_image_hash is not None
        ):
            return (
                self._cached_descriptor_map,
                self._cached_score_map,
                self._cached_image_hash,
            )
        rgb, image_hash = _decode_rgb(Path(image_path))
        if rgb.shape[:2] != (int(image_height), int(image_width)):
            rgb = cv2.resize(
                rgb,
                (int(image_width), int(image_height)),
                interpolation=cv2.INTER_AREA,
            )
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device,
            dtype=torch.float32,
        )
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        descriptor_map, score_map = self.model.extract_dense_map(image)
        self._cached_dense_key = key
        self._cached_descriptor_map = descriptor_map
        self._cached_score_map = score_map
        self._cached_image_hash = image_hash
        return descriptor_map, score_map, image_hash

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "feature_type": "alike_dense_descriptor",
            "model_name": self.model_name,
            "model_checkpoint": str(self.model_path),
            "model_checkpoint_sha256": file_sha256_short(self.model_path),
            "output_stride": 1,
            "sampling_mode": "bilinear",
            "sampling_convention": "sfm_pixel_endpoint_to_dense_endpoint_v1",
            "sampling_align_corners": True,
            "image_preprocessing": "resize_rgb_to_sfm_image_dimensions",
        }

    @torch.no_grad()
    def extract(
        self,
        image_path: Path,
        observations: Sequence[ColmapTrackObservation],
    ) -> tuple[np.ndarray, np.ndarray, str]:
        import cv2

        if not observations:
            raise ValueError("observations must not be empty")
        rgb, image_hash = _decode_rgb(Path(image_path))
        width = int(observations[0].image_width or 0)
        height = int(observations[0].image_height or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"SfM image dimensions missing for {observations[0].image_id}")
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        descriptor_map, score_map = self.model.extract_dense_map(image)
        descriptors, scores = _sample_dense_features(descriptor_map, score_map, observations)
        return descriptors, scores, image_hash

    @torch.no_grad()
    def detect(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
        top_k: int = 512,
        candidate_top_k: int = 4096,
        nms_radius_px: float = 4.0,
        grid_rows: int = 4,
        grid_cols: int = 4,
        min_score: float | None = None,
        sub_pixel: bool = True,
    ) -> DetectedImageFeatures:
        """Detect deployable ALIKE points while retaining sub-pixel locations."""

        if int(top_k) <= 0 or int(candidate_top_k) < int(top_k):
            raise ValueError("candidate_top_k must be at least top_k > 0")
        descriptor_map, score_map, image_hash = self._dense_maps(
            image_path,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        previous_top_k = int(self.model.dkd.top_k)
        try:
            self.model.dkd.top_k = min(int(candidate_top_k), int(image_height) * int(image_width))
            normalized_xy, descriptors, scores, dispersions = self.model.dkd(
                score_map,
                descriptor_map,
                sub_pixel=bool(sub_pixel),
            )
        finally:
            self.model.dkd.top_k = previous_top_k
        points = normalized_xy[0]
        points = (points + 1.0) * 0.5 * points.new_tensor(
            [max(int(image_width) - 1, 1), max(int(image_height) - 1, 1)]
        )
        point_values = points.detach().cpu().numpy().astype(np.float32, copy=False)
        descriptor_values = F.normalize(descriptors[0].float(), p=2, dim=1).detach().cpu().numpy()
        score_values = scores[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if dispersions[0] is None:
            dispersion_values = np.full((len(point_values),), np.nan, dtype=np.float32)
        else:
            dispersion_values = dispersions[0].detach().cpu().numpy().astype(np.float32, copy=False)
        selected = spatially_diverse_detection_indices(
            point_values,
            score_values,
            top_k=int(top_k),
            nms_radius_px=float(nms_radius_px),
            image_width=int(image_width),
            image_height=int(image_height),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            min_score=min_score,
        )
        return DetectedImageFeatures(
            xy=point_values[selected],
            descriptors=descriptor_values[selected],
            scores=score_values[selected],
            dispersions=dispersion_values[selected],
            image_sha256=image_hash,
        )

    @torch.no_grad()
    def sample_points(
        self,
        image_path: Path,
        xy: np.ndarray,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Sample dense ALIKE at arbitrary pixels without constructing SfM tracks."""

        points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        descriptor_map, score_map, image_hash = self._dense_maps(
            image_path,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        descriptors, scores = sample_dense_feature_points(
            descriptor_map,
            points,
            image_width=int(image_width),
            image_height=int(image_height),
            score_map=score_map,
        )
        return descriptors, scores, image_hash

    @torch.no_grad()
    def match_descriptor_points(
        self,
        image_path: Path,
        predicted_xy: np.ndarray,
        reference_descriptors: np.ndarray,
        *,
        image_width: int,
        image_height: int,
        search_radius_px: int = 12,
        search_step_px: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Find local ALIKE modes around VFM-transferred anchor projections."""

        points = np.asarray(predicted_xy, dtype=np.float32).reshape(-1, 2)
        references = np.asarray(reference_descriptors, dtype=np.float32)
        if references.ndim != 2 or references.shape[0] != len(points):
            raise ValueError("one reference descriptor is required per predicted point")
        if int(search_radius_px) < 0 or int(search_step_px) <= 0:
            raise ValueError("local search radius/step is invalid")
        descriptor_map, score_map, image_hash = self._dense_maps(
            image_path,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        offsets = np.arange(
            -int(search_radius_px),
            int(search_radius_px) + 1,
            int(search_step_px),
            dtype=np.float32,
        )
        offset_xy = np.stack(
            np.meshgrid(offsets, offsets, indexing="xy"),
            axis=-1,
        ).reshape(-1, 2)
        candidate_xy = points[:, None, :] + offset_xy[None, :, :]
        candidate_xy[..., 0] = np.clip(
            candidate_xy[..., 0], 0.0, float(image_width - 1)
        )
        candidate_xy[..., 1] = np.clip(
            candidate_xy[..., 1], 0.0, float(image_height - 1)
        )
        sampled_descriptors, sampled_scores = sample_dense_feature_points(
            descriptor_map,
            candidate_xy.reshape(-1, 2),
            image_width=int(image_width),
            image_height=int(image_height),
            score_map=score_map,
        )
        sampled_descriptors = sampled_descriptors.reshape(
            len(points), len(offset_xy), -1
        )
        sampled_scores = sampled_scores.reshape(len(points), len(offset_xy))
        references /= np.maximum(
            np.linalg.norm(references, axis=1, keepdims=True), 1e-8
        )
        similarities = np.einsum(
            "nkd,nd->nk",
            sampled_descriptors,
            references,
        )
        spatial_prior = np.sum(offset_xy * offset_xy, axis=1)[None, :] / max(
            float(search_radius_px * search_radius_px), 1.0
        )
        # Descriptor agreement remains decisive.  Detector confidence and the
        # transferred VFM position only break near-ties.
        combined = similarities + 0.001 * sampled_scores - 0.001 * spatial_prior
        best = np.argmax(combined, axis=1)
        rows = np.arange(len(points), dtype=np.int64)
        return (
            candidate_xy[rows, best].astype(np.float32, copy=False),
            sampled_descriptors[rows, best].astype(np.float32, copy=False),
            similarities[rows, best].astype(np.float32, copy=False),
            image_hash,
        )


class RadioIntermediateObservationExtractor:
    """C-RADIO intermediate map sampled without materializing a full-map cache."""

    def __init__(
        self,
        *,
        device: str,
        version: str = "c-radio_v4-h",
        radio_repo: Path = Path("feature_extract/checkpoints/RADIO"),
        intermediate_index: int = -6,
    ) -> None:
        from feature_extract.extractors import RADIOFeatureExtractor

        self.device = torch.device(str(device))
        self.version = str(version)
        self.radio_repo = Path(radio_repo)
        self.intermediate_index = int(intermediate_index)
        self.extractor = RADIOFeatureExtractor(
            version=self.version,
            device=str(self.device),
            radio_repo=str(self.radio_repo),
        )
        self.weight_path = Path.home() / ".cache/torch/hub/checkpoints" / f"{self.version}_half.pth.tar"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "feature_type": "radio_intermediate",
            "model_name": self.version,
            "model_checkpoint": str(self.weight_path),
            "model_checkpoint_sha256": (
                file_sha256_short(self.weight_path) if self.weight_path.exists() else "unresolved"
            ),
            "radio_repo": str(self.radio_repo),
            "intermediate_index": int(self.intermediate_index),
            "output_stride": int(self.extractor.patch_size),
            "sampling_mode": "bilinear",
            "sampling_convention": "sfm_pixel_endpoint_to_radio_token_endpoint_v1",
            "sampling_align_corners": True,
            "image_preprocessing": "radio_nearest_supported_resolution_from_original_rgb",
        }

    @torch.no_grad()
    def extract(
        self,
        image_path: Path,
        observations: Sequence[ColmapTrackObservation],
    ) -> tuple[np.ndarray, np.ndarray, str]:
        rgb, image_hash = _decode_rgb(Path(image_path))
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        descriptor_map = self.extractor.extract_intermediate_batch(
            image,
            intermediate_index=int(self.intermediate_index),
        )
        descriptors, scores = _sample_dense_features(descriptor_map, None, observations)
        return descriptors, scores, image_hash

    @torch.no_grad()
    def extract_batch(
        self,
        image_paths: Sequence[Path],
        observations_by_image: Sequence[Sequence[ColmapTrackObservation]],
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        if len(image_paths) != len(observations_by_image):
            raise ValueError("image_paths and observations_by_image must have the same length")
        decoded = [_decode_rgb(Path(path)) for path in image_paths]
        shapes = {tuple(rgb.shape) for rgb, _digest in decoded}
        if len(shapes) != 1:
            return [self.extract(path, observations) for path, observations in zip(image_paths, observations_by_image)]
        images = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
                for rgb, _digest in decoded
            ],
            dim=0,
        ).to(self.device)
        maps = self.extractor.extract_intermediate_batch(
            images,
            intermediate_index=int(self.intermediate_index),
        )
        output: list[tuple[np.ndarray, np.ndarray, str]] = []
        for index, observations in enumerate(observations_by_image):
            descriptors, scores = _sample_dense_features(maps[index : index + 1], None, observations)
            output.append((descriptors, scores, decoded[index][1]))
        return output


def extract_real_image_observation_features(
    observations: Sequence[ColmapTrackObservation],
    *,
    image_root: Path,
    devices: Sequence[str],
    feature_type: str = "alike",
    alike_model_name: str = "alike-t",
    matcha_repo: Path = Path("/root/matcha"),
    radio_version: str = "c-radio_v4-h",
    radio_repo: Path = Path("feature_extract/checkpoints/RADIO"),
    radio_intermediate_index: int = -6,
    image_batch_size: int = 1,
) -> ExtractedObservationFeatures:
    """Multi-GPU extraction with deterministic image partitioning and row order."""

    from concurrent.futures import ThreadPoolExecutor

    rows = list(observations)
    if not rows:
        raise ValueError("observations must not be empty")
    device_values = tuple(str(device) for device in devices)
    if not device_values:
        raise ValueError("at least one extraction device is required")
    if int(image_batch_size) <= 0:
        raise ValueError("image_batch_size must be positive")
    by_image: dict[str, list[tuple[int, ColmapTrackObservation]]] = {}
    for row, observation in enumerate(rows):
        by_image.setdefault(str(observation.image_id), []).append((int(row), observation))
    image_ids = sorted(by_image)
    partitions = [image_ids[index:: len(device_values)] for index in range(len(device_values))]

    def worker(worker_index: int):
        if str(feature_type) == "alike":
            extractor = AlikeDenseObservationExtractor(
                device=device_values[worker_index],
                matcha_repo=Path(matcha_repo),
                model_name=str(alike_model_name),
            )
        elif str(feature_type) == "radio_intermediate":
            extractor = RadioIntermediateObservationExtractor(
                device=device_values[worker_index],
                version=str(radio_version),
                radio_repo=Path(radio_repo),
                intermediate_index=int(radio_intermediate_index),
            )
        else:
            raise ValueError("feature_type must be 'alike' or 'radio_intermediate'")
        result_rows: list[np.ndarray] = []
        result_descriptors: list[np.ndarray] = []
        result_scores: list[np.ndarray] = []
        image_hashes: dict[str, str] = {}
        partition = partitions[worker_index]
        for start in range(0, len(partition), int(image_batch_size)):
            batch_ids = partition[start : start + int(image_batch_size)]
            batch_indexed = [by_image[image_id] for image_id in batch_ids]
            batch_observations = [[item[1] for item in indexed] for indexed in batch_indexed]
            batch_paths = [Path(image_root) / image_id for image_id in batch_ids]
            for image_path in batch_paths:
                if not image_path.exists():
                    raise FileNotFoundError(f"real source image not found: {image_path}")
            batch_method = getattr(extractor, "extract_batch", None)
            if callable(batch_method) and len(batch_ids) > 1:
                extracted_rows = batch_method(batch_paths, batch_observations)
            else:
                extracted_rows = [
                    extractor.extract(image_path, observations)
                    for image_path, observations in zip(batch_paths, batch_observations)
                ]
            for image_id, indexed, (descriptors, scores, image_hash) in zip(
                batch_ids,
                batch_indexed,
                extracted_rows,
            ):
                result_rows.append(np.asarray([item[0] for item in indexed], dtype=np.int64))
                result_descriptors.append(descriptors)
                result_scores.append(scores)
                image_hashes[image_id] = image_hash
        return {
            "rows": np.concatenate(result_rows) if result_rows else np.zeros((0,), dtype=np.int64),
            "descriptors": (
                np.concatenate(result_descriptors, axis=0)
                if result_descriptors
                else np.zeros((0, 0), dtype=np.float32)
            ),
            "scores": np.concatenate(result_scores) if result_scores else np.zeros((0,), dtype=np.float32),
            "image_hashes": image_hashes,
            "extractor_metadata": extractor.metadata,
        }

    if len(device_values) == 1:
        results = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(device_values)) as executor:
            results = list(executor.map(worker, range(len(device_values))))
    dimensions = {
        int(result["descriptors"].shape[1])
        for result in results
        if int(result["descriptors"].shape[0]) > 0
    }
    if len(dimensions) != 1:
        raise ValueError(f"workers returned inconsistent descriptor dimensions: {sorted(dimensions)}")
    feature_dim = next(iter(dimensions))
    descriptors = np.zeros((len(rows), feature_dim), dtype=np.float32)
    scores = np.full((len(rows),), np.nan, dtype=np.float32)
    all_image_hashes: dict[str, str] = {}
    for result in results:
        indices = np.asarray(result["rows"], dtype=np.int64)
        descriptors[indices] = np.asarray(result["descriptors"], dtype=np.float32)
        scores[indices] = np.asarray(result["scores"], dtype=np.float32)
        all_image_hashes.update(dict(result["image_hashes"]))
    source_manifest = "\n".join(f"{image_id}:{all_image_hashes[image_id]}" for image_id in sorted(all_image_hashes))
    extractor_metadata = dict(results[0]["extractor_metadata"])
    metadata = {
        **extractor_metadata,
        "feature_dimension": int(feature_dim),
        "observation_count": int(len(rows)),
        "image_count": int(len(image_ids)),
        "devices": list(device_values),
        "partition_image_counts": [int(len(partition)) for partition in partitions],
        "image_batch_size": int(image_batch_size),
        "source_image_manifest_sha256": hashlib.sha256(source_manifest.encode("utf8")).hexdigest()[:16],
    }
    return ExtractedObservationFeatures(descriptors, scores, metadata)
