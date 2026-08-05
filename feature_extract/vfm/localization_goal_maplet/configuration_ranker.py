"""Runtime-only candidate features for independent configuration ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .lineage import file_sha256
from .configuration_evidence import FEATURE_NAMES as CONFIGURATION_EVIDENCE_FEATURE_NAMES


BASE_FEATURE_NAMES = (
    "proposal_score",
    "log_support_count",
    "render_parent_log_likelihood",
    "render_child_conditional_log_likelihood",
    "rendered_coverage",
    "cheap_identity_score",
    "proposal_margin_from_best",
    "identity_margin_from_best",
    "original_rank_fraction",
    "mode_count_log",
    "nearest_mode_translation_m",
    "nearest_mode_rotation_deg",
    "density_0p5m_3deg",
    "density_1m_5deg",
    "median_mode_translation_m",
    "median_mode_rotation_deg",
)

EXACT_FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "exact_evaluated",
    "exact_parent_log_likelihood",
    "exact_child_conditional_log_likelihood",
    "exact_rendered_coverage",
    "exact_identity_score",
    "exact_identity_margin_from_best",
    "exact_minus_cheap_identity",
)

BASE_CONFIGURATION_FEATURE_NAMES = BASE_FEATURE_NAMES + CONFIGURATION_EVIDENCE_FEATURE_NAMES
EXACT_CONFIGURATION_FEATURE_NAMES = EXACT_FEATURE_NAMES + CONFIGURATION_EVIDENCE_FEATURE_NAMES

# Backwards-compatible public name for the frozen v1 cheap ranker.
FEATURE_NAMES = BASE_FEATURE_NAMES


def _rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left)[:3, :3] @ np.asarray(right)[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def candidate_runtime_features(
    report_row: Mapping[str, object],
    *,
    mode_name: str = "actual_parent_actual_child",
    include_exact: bool = False,
    include_configuration: bool = False,
) -> np.ndarray:
    details = list(report_row["mode_details"][mode_name])
    diagnostics = dict(report_row["ranking_diagnostics"][mode_name])
    count = len(details)
    for name in (
        "identity_scores", "parent_log_likelihood", "child_log_likelihood",
        "rendered_coverage", "proposal_scores", "original_indices",
    ):
        if len(diagnostics.get(name, [])) != count:
            raise ValueError(f"candidate rank feature {name} differs from mode count")
    feature_names = (
        EXACT_CONFIGURATION_FEATURE_NAMES if include_exact and include_configuration
        else EXACT_FEATURE_NAMES if include_exact
        else BASE_CONFIGURATION_FEATURE_NAMES if include_configuration
        else BASE_FEATURE_NAMES
    )
    if count == 0:
        return np.zeros((0, len(feature_names)), dtype=np.float32)
    proposal = np.asarray(diagnostics["proposal_scores"], dtype=np.float64)
    identity = np.asarray(diagnostics["identity_scores"], dtype=np.float64)
    parent_ll = np.asarray(diagnostics["parent_log_likelihood"], dtype=np.float64)
    child_ll = np.asarray(diagnostics["child_log_likelihood"], dtype=np.float64)
    coverage = np.asarray(diagnostics["rendered_coverage"], dtype=np.float64)
    original = np.asarray(diagnostics["original_indices"], dtype=np.float64)
    support = np.asarray([value["supporting_region_count"] for value in details], dtype=np.float64)
    poses = np.asarray([value["pose_w2c"] for value in details], dtype=np.float64)
    centers = np.asarray([value["camera_center"] for value in details], dtype=np.float64)
    translation = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    rotation = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = _rotation_distance_degrees(poses[left], poses[right])
            rotation[left, right] = rotation[right, left] = value
    diagonal = np.eye(count, dtype=bool)
    nearest_translation = np.min(np.where(diagonal, np.inf, translation), axis=1) if count > 1 else np.zeros(1)
    nearest_rotation = np.min(np.where(diagonal, np.inf, rotation), axis=1) if count > 1 else np.zeros(1)
    median_translation = np.nanmedian(np.where(diagonal, np.nan, translation), axis=1) if count > 1 else np.zeros(1)
    median_rotation = np.nanmedian(np.where(diagonal, np.nan, rotation), axis=1) if count > 1 else np.zeros(1)
    density_small = np.sum((translation <= 0.5) & (rotation <= 3.0) & ~diagonal, axis=1)
    density_large = np.sum((translation <= 1.0) & (rotation <= 5.0) & ~diagonal, axis=1)
    output = np.stack([
        proposal,
        np.log1p(support),
        parent_ll,
        child_ll,
        coverage,
        identity,
        proposal - np.max(proposal),
        identity - np.max(identity),
        original / max(count - 1, 1),
        np.full(count, np.log1p(count)),
        nearest_translation,
        nearest_rotation,
        density_small / max(count - 1, 1),
        density_large / max(count - 1, 1),
        np.nan_to_num(median_translation),
        np.nan_to_num(median_rotation),
    ], axis=1)
    if include_exact:
        for name in (
            "cascade_exact_evaluated", "cascade_exact_parent_log_likelihood",
            "cascade_exact_child_log_likelihood", "cascade_exact_rendered_coverage",
            "cascade_exact_scores",
        ):
            if len(diagnostics.get(name, [])) != count:
                raise ValueError(f"exact candidate rank feature {name} differs from mode count")
        evaluated = np.asarray(diagnostics["cascade_exact_evaluated"], dtype=bool)
        exact_parent = np.asarray(diagnostics["cascade_exact_parent_log_likelihood"], dtype=np.float64)
        exact_child = np.asarray(diagnostics["cascade_exact_child_log_likelihood"], dtype=np.float64)
        exact_coverage = np.asarray(diagnostics["cascade_exact_rendered_coverage"], dtype=np.float64)
        exact_identity = np.asarray(diagnostics["cascade_exact_scores"], dtype=np.float64)
        if not np.any(evaluated):
            raise ValueError("exact configuration ranker requires evaluated candidates")
        best_exact = float(np.max(exact_identity[evaluated]))
        exact_margin = np.where(evaluated, exact_identity - best_exact, 0.0)
        exact_delta = np.where(evaluated, exact_identity - identity, 0.0)
        output = np.concatenate([
            output,
            np.stack([
                evaluated.astype(np.float64), exact_parent, exact_child,
                exact_coverage, exact_identity, exact_margin, exact_delta,
            ], axis=1),
        ], axis=1)
    if include_configuration:
        evidence = diagnostics.get("configuration_evidence_v2")
        if not isinstance(evidence, Mapping):
            raise ValueError("configuration evidence is missing")
        columns = []
        for name in CONFIGURATION_EVIDENCE_FEATURE_NAMES:
            value = np.asarray(evidence.get(name, []), dtype=np.float64)
            if value.shape != (count,):
                raise ValueError(f"configuration evidence {name} differs from mode count")
            columns.append(value)
        output = np.concatenate([output, np.stack(columns, axis=1)], axis=1)
    if not np.all(np.isfinite(output)):
        raise ValueError("configuration rank features contain non-finite values")
    return output.astype(np.float32)


@dataclass(frozen=True)
class ConfigurationRankerArtifact:
    estimator: object
    metadata: Mapping[str, object]

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        expected = len(tuple(self.metadata.get("feature_names", ())))
        if value.ndim != 2 or value.shape[1] != expected:
            raise ValueError("configuration rank feature dimension differs")
        return np.asarray(self.estimator.predict_proba(value)[:, 1], dtype=np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": dict(self.metadata)}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ConfigurationRankerArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        names = tuple(metadata.get("feature_names", ()))
        if names not in (
            BASE_FEATURE_NAMES, EXACT_FEATURE_NAMES,
            BASE_CONFIGURATION_FEATURE_NAMES, EXACT_CONFIGURATION_FEATURE_NAMES,
        ):
            raise ValueError("configuration ranker feature contract differs")
        return cls(payload["estimator"], metadata)


def source_pool_sha256(path: Path) -> str:
    return file_sha256(Path(path))


@dataclass(frozen=True)
class ConfigurationPairwiseRankerArtifact:
    """Rank a complete query candidate set using ordered utility comparisons."""

    estimator: object
    metadata: Mapping[str, object]

    def score_candidates(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        names = tuple(self.metadata.get("feature_names", ()))
        if names not in (
            BASE_FEATURE_NAMES, EXACT_FEATURE_NAMES,
            BASE_CONFIGURATION_FEATURE_NAMES, EXACT_CONFIGURATION_FEATURE_NAMES,
        ):
            raise ValueError("configuration pairwise feature contract differs")
        if value.ndim != 2 or value.shape[1] != len(names):
            raise ValueError("configuration pairwise feature dimension differs")
        count = value.shape[0]
        if count == 0:
            return np.zeros((0,), dtype=np.float64)
        if count == 1:
            return np.ones((1,), dtype=np.float64)
        left, right = np.triu_indices(count, 1)
        probability = np.asarray(
            self.estimator.predict_proba(value[left] - value[right])[:, 1], dtype=np.float64
        )
        score = np.zeros((count,), dtype=np.float64)
        comparisons = np.zeros((count,), dtype=np.float64)
        np.add.at(score, left, probability)
        np.add.at(score, right, 1.0 - probability)
        np.add.at(comparisons, left, 1.0)
        np.add.at(comparisons, right, 1.0)
        return score / np.maximum(comparisons, 1.0)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": dict(self.metadata)}, Path(path))

    @classmethod
    def load(cls, path: Path) -> "ConfigurationPairwiseRankerArtifact":
        payload = joblib.load(Path(path))
        metadata = dict(payload["metadata"])
        if metadata.get("artifact_type") != "goal_maplet_configuration_pairwise_ranker_v2":
            raise ValueError("not a Goal-Maplet pairwise configuration ranker")
        names = tuple(metadata.get("feature_names", ()))
        if names not in (
            BASE_FEATURE_NAMES, EXACT_FEATURE_NAMES,
            BASE_CONFIGURATION_FEATURE_NAMES, EXACT_CONFIGURATION_FEATURE_NAMES,
        ):
            raise ValueError("configuration pairwise feature contract differs")
        return cls(payload["estimator"], metadata)
