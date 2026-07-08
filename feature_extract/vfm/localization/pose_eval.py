"""Pose-level evaluation helpers for real-image RADIO localization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


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


@dataclass(frozen=True)
class ClosedLoopProposalRecord:
    query_id: str
    reference_image_id: str
    proposal: CoarseProposal
    measurement: MeasurementResult | None
    proposal_index: int


def _score_for_pnp(record: ClosedLoopProposalRecord) -> float:
    if record.measurement is not None and record.measurement.confidence is not None:
        return float(record.measurement.confidence)
    if record.proposal.confidence is not None:
        return float(record.proposal.confidence)
    return float(record.proposal.score)


def _measured_query_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_query_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.query_xy, dtype=np.float64).reshape(2)


def _measured_reference_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_reference_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.reference_xy, dtype=np.float64).reshape(2)


def convert_proposals_to_query_3d_matches(
    records: Sequence[ClosedLoopProposalRecord],
    *,
    observation_index: SupportObservationIndex,
    max_support_distance_px: float,
) -> tuple[list[dict[str, Any]], dict[str, list[QueryTo3DMatch]]]:
    rows: list[dict[str, Any]] = []
    matches_by_query: dict[str, list[QueryTo3DMatch]] = {}
    for record in records:
        query_xy = _measured_query_xy(record)
        reference_xy = _measured_reference_xy(record)
        nearest = observation_index.nearest(
            record.reference_image_id,
            reference_xy,
            max_distance_px=float(max_support_distance_px),
        )
        score = float(_score_for_pnp(record))
        base_row: dict[str, Any] = {
            "query_id": str(record.query_id),
            "reference_image_id": str(record.reference_image_id),
            "proposal_index": int(record.proposal_index),
            "query_x": float(query_xy[0]),
            "query_y": float(query_xy[1]),
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "score": score,
        }
        if nearest is None:
            rows.append({**base_row, "association_status": "missing_support_observation"})
            continue
        observation = nearest.observation
        match = QueryTo3DMatch(
            token_index=int(record.proposal_index),
            xy=query_xy.astype(np.float64, copy=False),
            track_id=int(observation.track_id),
            xyz=np.asarray(observation.xyz, dtype=np.float64).reshape(3),
            similarity=score,
            ratio=1.0,
            landmark_variance=0.0,
            source="real_radio_closed_loop",
            observation_count=int(observation.track_length),
            landmark_reprojection_error=float(observation.reprojection_error),
            pnp_soft_score=score,
            patch_offset_confidence=None if record.measurement is None else record.measurement.confidence,
            measurement_sigma_px=None if record.measurement is None else record.measurement.uncertainty_px,
            coarse_rank=record.proposal.rank,
            coarse_score=float(record.proposal.score),
        )
        matches_by_query.setdefault(str(record.query_id), []).append(match)
        rows.append(
            {
                **base_row,
                "association_status": "matched",
                "track_id": int(observation.track_id),
                "support_observation_distance_px": float(nearest.distance_px),
                "support_track_length": int(observation.track_length),
                "support_reprojection_error": float(observation.reprojection_error),
            }
        )
    return rows, matches_by_query


def deduplicate_query_3d_matches(matches: Sequence[QueryTo3DMatch]) -> list[QueryTo3DMatch]:
    best: dict[int, QueryTo3DMatch] = {}
    for match in matches:
        key = int(match.track_id)
        existing = best.get(key)
        current_score = 0.0 if match.pnp_soft_score is None else float(match.pnp_soft_score)
        existing_score = -float("inf") if existing is None or existing.pnp_soft_score is None else float(existing.pnp_soft_score)
        if existing is None or current_score > existing_score:
            best[key] = match
    return [best[key] for key in sorted(best)]
