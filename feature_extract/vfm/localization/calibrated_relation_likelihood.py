"""Frozen density-ratio calibration for candidate relation residuals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.localization.candidate_relation_features import (
    RELATION_CHANNELS,
    RELATION_FEATURE_VERSION,
    RelationResidualHistograms,
)


RELATION_CALIBRATION_FORMAT = "calibrated_candidate_relation_likelihood_v4_topology_strength_repeat_geometry"


@dataclass(frozen=True)
class CalibratedRelationLikelihood:
    bin_edges_px: np.ndarray
    log_density_ratio: np.ndarray
    source_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        bins = np.asarray(self.bin_edges_px, dtype=np.float64).reshape(-1).copy()
        ratio = np.asarray(self.log_density_ratio, dtype=np.float64).copy()
        if ratio.shape != (len(RELATION_CHANNELS), len(bins) - 1):
            raise ValueError("relation calibration dimensions differ from schema")
        if np.any(~np.isfinite(ratio)) or np.any(np.diff(bins) <= 0.0):
            raise ValueError("relation calibration must be finite")
        if not isinstance(self.source_manifest, Mapping) or not self.source_manifest:
            raise ValueError("relation calibration requires a source manifest")
        bins.setflags(write=False)
        ratio.setflags(write=False)
        object.__setattr__(self, "bin_edges_px", bins)
        object.__setattr__(self, "log_density_ratio", ratio)

    def as_dict(self) -> dict[str, object]:
        return {
            "format": RELATION_CALIBRATION_FORMAT,
            "feature_version": RELATION_FEATURE_VERSION,
            "relation_channels": list(RELATION_CHANNELS),
            "bin_edges_px": self.bin_edges_px.tolist(),
            "log_density_ratio": self.log_density_ratio.tolist(),
            "source_manifest": dict(self.source_manifest),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibratedRelationLikelihood":
        if payload.get("format") != RELATION_CALIBRATION_FORMAT:
            raise ValueError("unsupported relation calibration format")
        if payload.get("feature_version") != RELATION_FEATURE_VERSION:
            raise ValueError("relation feature version differs")
        if tuple(payload.get("relation_channels", ())) != RELATION_CHANNELS:
            raise ValueError("relation calibration channels differ")
        return cls(
            np.asarray(payload["bin_edges_px"], dtype=np.float64),
            np.asarray(payload["log_density_ratio"], dtype=np.float64),
            payload["source_manifest"],
        )

    @classmethod
    def load(cls, path: Path) -> "CalibratedRelationLikelihood":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


def fit_relation_density_ratio(
    positive_histogram: np.ndarray,
    negative_histogram: np.ndarray,
    *,
    bin_edges_px: np.ndarray,
    source_manifest: Mapping[str, object],
    pseudocount: float = 1.0,
    max_abs_log_ratio: float = 6.0,
) -> CalibratedRelationLikelihood:
    positive = np.asarray(positive_histogram, dtype=np.float64)
    negative = np.asarray(negative_histogram, dtype=np.float64)
    bins = np.asarray(bin_edges_px, dtype=np.float64).reshape(-1)
    expected = (len(RELATION_CHANNELS), len(bins) - 1)
    if positive.shape != expected or negative.shape != expected:
        raise ValueError("relation calibration histograms have incompatible shapes")
    if np.any(positive < 0.0) or np.any(negative < 0.0):
        raise ValueError("relation calibration counts must be non-negative")
    alpha = float(pseudocount)
    if alpha <= 0.0:
        raise ValueError("relation calibration pseudocount must be positive")
    positive_density = positive + alpha
    negative_density = negative + alpha
    positive_density /= np.sum(positive_density, axis=1, keepdims=True)
    negative_density /= np.sum(negative_density, axis=1, keepdims=True)
    ratio = np.log(positive_density) - np.log(negative_density)
    ratio = np.clip(ratio, -float(max_abs_log_ratio), float(max_abs_log_ratio))
    return CalibratedRelationLikelihood(bins, ratio, source_manifest)


def score_relation_histograms(
    features: RelationResidualHistograms,
    calibration: CalibratedRelationLikelihood,
) -> dict[str, float | int]:
    if not np.array_equal(features.bin_edges_px, calibration.bin_edges_px):
        raise ValueError("relation features and calibration bins differ")
    likelihood_ratio = np.exp(calibration.log_density_ratio)
    candidate_ratio = np.sum(
        features.histograms * likelihood_ratio[None, :, :], axis=(1, 2)
    )
    evidence_ratio = features.null_touching_mass + candidate_ratio
    log_ratio = np.log(np.maximum(evidence_ratio, 1e-12))
    effective = features.candidate_pair_mass > 1e-12
    return {
        "edge_count": int(len(log_ratio)),
        "effective_edge_count": int(np.sum(effective)),
        "relation_log_likelihood_ratio_sum": float(np.sum(log_ratio)),
        "relation_log_likelihood_ratio_mean": (
            0.0 if len(log_ratio) == 0 else float(np.mean(log_ratio))
        ),
    }
