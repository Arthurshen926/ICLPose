"""Composable real-image selector + coarse matcher + measurement model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from feature_extract.vfm.localization.coarse_matcher import CoarseMatcher
from feature_extract.vfm.localization.feature_mapper import FeatureMapper
from feature_extract.vfm.localization.schemas import LocalizationMatchResult, MappedFeatureMap, MeasurementResult


class MeasurementBranch(Protocol):
    """Optional real-image patch measurement branch."""

    def measure(
        self,
        query_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        proposals,
        *,
        mapped_query: MappedFeatureMap | None = None,
        mapped_reference: MappedFeatureMap | None = None,
    ) -> list[MeasurementResult]:
        """Return patch measurements for coarse proposals."""


@dataclass
class SelectorCoarseMeasurementModel:
    """Real-image localization module with explicit selector and coarse matcher."""

    feature_mapper: FeatureMapper
    coarse_matcher: CoarseMatcher
    measurement_branch: MeasurementBranch | None = None

    def match_pair(
        self,
        query_feature_map: np.ndarray,
        reference_feature_map: np.ndarray,
        *,
        query_image_size: tuple[int, int],
        reference_image_size: tuple[int, int],
        query_rgb: np.ndarray | None = None,
        reference_rgb: np.ndarray | None = None,
    ) -> LocalizationMatchResult:
        mapped_query = self.feature_mapper.project(query_feature_map)
        mapped_reference = self.feature_mapper.project(reference_feature_map)
        proposals = self.coarse_matcher.match(
            mapped_query.coarse_descriptors,
            mapped_reference.coarse_descriptors,
            query_image_size=query_image_size,
            reference_image_size=reference_image_size,
        )
        measurements: list[MeasurementResult] = []
        if self.measurement_branch is not None:
            if query_rgb is None or reference_rgb is None:
                raise ValueError("query_rgb and reference_rgb are required when measurement_branch is set")
            measurements = self.measurement_branch.measure(
                query_rgb,
                reference_rgb,
                proposals,
                mapped_query=mapped_query,
                mapped_reference=mapped_reference,
            )
        return LocalizationMatchResult(
            mapped_query=mapped_query,
            mapped_reference=mapped_reference,
            coarse_proposals=proposals,
            measurements=measurements,
        )
