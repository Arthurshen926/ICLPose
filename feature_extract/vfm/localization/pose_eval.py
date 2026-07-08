"""Pose-level evaluation helpers for real-image RADIO localization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation


@dataclass(frozen=True)
class NearestSupportObservation:
    observation: ColmapTrackObservation
    distance_px: float


class SupportObservationIndex:
    """Nearest-neighbor lookup over COLMAP observations grouped by image id."""

    def __init__(self, observations: Sequence[ColmapTrackObservation]) -> None:
        by_image: dict[str, list[ColmapTrackObservation]] = {}
        for observation in observations:
            by_image.setdefault(str(observation.image_id), []).append(observation)
        self._by_image = {image_id: tuple(items) for image_id, items in by_image.items()}
        self._xy_by_image = {
            image_id: np.asarray([item.xy for item in items], dtype=np.float64)
            for image_id, items in self._by_image.items()
        }

    def nearest(
        self,
        image_id: str,
        xy: np.ndarray | Sequence[float],
        *,
        max_distance_px: float,
    ) -> NearestSupportObservation | None:
        items = self._by_image.get(str(image_id))
        if not items:
            return None
        query_xy = np.asarray(xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(query_xy)):
            return None
        distances = np.linalg.norm(self._xy_by_image[str(image_id)] - query_xy[None, :], axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance > float(max_distance_px):
            return None
        return NearestSupportObservation(observation=items[best], distance_px=distance)


def build_support_observation_index(observations: Sequence[ColmapTrackObservation]) -> SupportObservationIndex:
    return SupportObservationIndex(observations)
