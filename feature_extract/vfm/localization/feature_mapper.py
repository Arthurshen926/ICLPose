"""Feature mapper interfaces for raw RADIO descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from feature_extract.vfm.localization.schemas import MappedFeatureMap
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineAdapter,
    project_feature_map_with_matcha_adapter,
)


class FeatureMapper(Protocol):
    """Map one raw RADIO feature map to descriptors used by localization."""

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        """Return coarse descriptors and optional measurement context."""


@dataclass
class AdapterFeatureMapper:
    """Compatibility mapper backed by the existing MATCHA coarse adapter."""

    model: MatchaCoarseFineAdapter
    device: str = "cpu"
    batch_size: int = 65536

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        descriptors, offset_logits = project_feature_map_with_matcha_adapter(
            self.model,
            feature_map,
            device=str(self.device),
            batch_size=int(self.batch_size),
        )
        return MappedFeatureMap(
            coarse_descriptors=descriptors,
            measurement_context=descriptors,
            offset_logits=offset_logits,
            heatmap=None,
        )
