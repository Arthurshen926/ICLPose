"""Low-capacity safety residual on top of a predefined latent pose score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .configuration_evidence import LATENT_FEATURE_NAMES
from .configuration_ranker import BASE_FEATURE_NAMES, candidate_runtime_features


MODE = "actual_parent_actual_child"

RAW_FEATURE_NAMES = tuple(f"base__{name}" for name in BASE_FEATURE_NAMES) + tuple(
    f"latent__{name}" for name in LATENT_FEATURE_NAMES
)
FEATURE_NAMES = RAW_FEATURE_NAMES + tuple(f"relative__{name}" for name in RAW_FEATURE_NAMES)


def latent_selector_features(report_row: Mapping[str, object]) -> np.ndarray:
    base = candidate_runtime_features(report_row, mode_name=MODE)
    diagnostics = dict(report_row["ranking_diagnostics"][MODE])
    evidence = diagnostics.get("configuration_evidence_v3")
    if not isinstance(evidence, Mapping):
        raise ValueError("latent selector requires configuration evidence v3")
    count = base.shape[0]
    columns = []
    for name in LATENT_FEATURE_NAMES:
        value = np.asarray(evidence.get(name, []), dtype=np.float64)
        if value.shape != (count,):
            raise ValueError(f"latent selector evidence differs: {name}")
        columns.append(value)
    latent = np.stack(columns, axis=1) if columns else np.zeros((count, 0), dtype=np.float64)
    raw = np.concatenate([base, latent], axis=1).astype(np.float64)
    relative = raw - np.median(raw, axis=0, keepdims=True)
    output = np.concatenate([raw, relative], axis=1)
    if output.shape != (count, len(FEATURE_NAMES)) or not np.all(np.isfinite(output)):
        raise ValueError("latent selector features are invalid")
    return output.astype(np.float32)


@dataclass(frozen=True)
class LatentSafetySelectorArtifact:
    basin_estimator: object
    catastrophic_estimator: object
    metadata: Mapping[str, object]

    def predict_heads(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
            raise ValueError("latent safety selector feature dimension differs")
        basin = np.asarray(self.basin_estimator.predict_proba(value)[:, 1], dtype=np.float64)
        catastrophic = np.asarray(
            self.catastrophic_estimator.predict_proba(value)[:, 1], dtype=np.float64
        )
        return basin, catastrophic

    def score_candidates(
        self,
        features: np.ndarray,
        base_score: np.ndarray,
        *,
        safety_veto: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = np.asarray(base_score, dtype=np.float64).reshape(-1)
        basin, catastrophic = self.predict_heads(features)
        if base.shape != basin.shape:
            raise ValueError("latent safety selector score arrays differ")
        if base.size <= 1:
            return base.copy(), basin, catastrophic
        q25, q75 = np.percentile(base, [25.0, 75.0])
        scale = max(float(q75 - q25), 1e-3)
        correction = float(self.metadata.get("residual_scale", 0.25)) * scale * (
            basin - catastrophic
        )
        score = base + correction
        if safety_veto:
            safe = catastrophic < float(self.metadata.get("catastrophic_veto_probability", 0.50))
            if np.any(safe):
                score = np.where(safe, score, -np.inf)
        return score, basin, catastrophic

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "basin_estimator": self.basin_estimator,
            "catastrophic_estimator": self.catastrophic_estimator,
            "metadata": dict(self.metadata),
        }, Path(path))

    @classmethod
    def load(cls, path: Path) -> "LatentSafetySelectorArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_latent_safety_selector_v1":
            raise ValueError("not a latent safety selector")
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("latent safety selector contract differs")
        return cls(payload["basin_estimator"], payload["catastrophic_estimator"], metadata)
