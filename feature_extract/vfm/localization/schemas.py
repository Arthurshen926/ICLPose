"""Typed records for the real-image RADIO localization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _feature_map(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (C, H, W)")
    if int(arr.shape[0]) <= 0 or int(arr.shape[1]) <= 0 or int(arr.shape[2]) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return arr


def _image_size(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be (width, height)")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return width, height


def _xy(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"{name} must contain exactly two values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class FeatureMapPair:
    """Raw or mapped feature maps for a real query/reference image pair."""

    query: np.ndarray
    reference: np.ndarray
    query_image_size: tuple[int, int]
    reference_image_size: tuple[int, int]

    def __post_init__(self) -> None:
        query = _feature_map(self.query, name="query")
        reference = _feature_map(self.reference, name="reference")
        if int(query.shape[0]) != int(reference.shape[0]):
            raise ValueError("query and reference feature maps must have the same channel count")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "query_image_size", _image_size(self.query_image_size, name="query_image_size"))
        object.__setattr__(
            self,
            "reference_image_size",
            _image_size(self.reference_image_size, name="reference_image_size"),
        )

    @property
    def channels(self) -> int:
        return int(self.query.shape[0])

    @property
    def query_grid_hw(self) -> tuple[int, int]:
        return int(self.query.shape[1]), int(self.query.shape[2])

    @property
    def reference_grid_hw(self) -> tuple[int, int]:
        return int(self.reference.shape[1]), int(self.reference.shape[2])


@dataclass(frozen=True)
class MappedFeatureMap:
    """Selector output used by coarse matching and optional measurement."""

    coarse_descriptors: np.ndarray
    measurement_context: np.ndarray
    offset_logits: np.ndarray | None = None
    heatmap: np.ndarray | None = None

    def __post_init__(self) -> None:
        coarse = _feature_map(self.coarse_descriptors, name="coarse_descriptors")
        context = _feature_map(self.measurement_context, name="measurement_context")
        if tuple(coarse.shape[1:]) != tuple(context.shape[1:]):
            raise ValueError("coarse_descriptors and measurement_context must share grid shape")
        object.__setattr__(self, "coarse_descriptors", coarse)
        object.__setattr__(self, "measurement_context", context)
        if self.offset_logits is not None:
            offsets = _feature_map(self.offset_logits, name="offset_logits")
            if int(offsets.shape[0]) != 65:
                raise ValueError("offset_logits must have 65 channels")
            if tuple(offsets.shape[1:]) != tuple(coarse.shape[1:]):
                raise ValueError("offset_logits must share descriptor grid shape")
            object.__setattr__(self, "offset_logits", offsets)
        if self.heatmap is not None:
            heat = np.asarray(self.heatmap, dtype=np.float32)
            if heat.shape != tuple(coarse.shape[1:]):
                raise ValueError("heatmap must have shape (H, W)")
            object.__setattr__(self, "heatmap", heat)


@dataclass(frozen=True)
class CoarseProposal:
    """One coarse match between real query and real reference feature cells."""

    query_index: int
    reference_index: int
    query_xy: np.ndarray
    reference_xy: np.ndarray
    score: float
    confidence: float | None = None
    rank: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.query_index) < 0 or int(self.reference_index) < 0:
            raise ValueError("proposal indices must be non-negative")
        object.__setattr__(self, "query_index", int(self.query_index))
        object.__setattr__(self, "reference_index", int(self.reference_index))
        object.__setattr__(self, "query_xy", _xy(self.query_xy, name="query_xy"))
        object.__setattr__(self, "reference_xy", _xy(self.reference_xy, name="reference_xy"))
        object.__setattr__(self, "score", float(self.score))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if self.rank is not None:
            object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MeasurementResult:
    """Optional patch measurement for one coarse proposal."""

    proposal: CoarseProposal
    measured_query_xy: np.ndarray
    measured_reference_xy: np.ndarray
    confidence: float | None = None
    uncertainty_px: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "measured_query_xy", _xy(self.measured_query_xy, name="measured_query_xy"))
        object.__setattr__(self, "measured_reference_xy", _xy(self.measured_reference_xy, name="measured_reference_xy"))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if self.uncertainty_px is not None:
            object.__setattr__(self, "uncertainty_px", float(self.uncertainty_px))


@dataclass(frozen=True)
class LocalizationMatchResult:
    """Output of selector + coarse matcher + optional measurement."""

    mapped_query: MappedFeatureMap
    mapped_reference: MappedFeatureMap
    coarse_proposals: list[CoarseProposal]
    measurements: list[MeasurementResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coarse_proposals", list(self.coarse_proposals))
        object.__setattr__(self, "measurements", list(self.measurements))
