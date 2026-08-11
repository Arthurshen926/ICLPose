"""Calibrated canonical-field retrieval for Goal-Maplet."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import minimize


CALIBRATION_SCHEMA = "goal_maplet_validity_calibration_v1"


@dataclass(frozen=True)
class SparseMapletPosterior:
    """Sparse in-map identity mass with explicit probability semantics.

    ``out_of_map_probabilities`` is the calibrated probability that the query
    observation has no physical support in this map.  ``truncated_tail`` is
    valid in-map mass whose identity was not retained by the sparse Top-K.
    Keeping them separate is essential on repeated structures: a diffuse but
    valid posterior must not be treated as an outlier merely because its mass
    is spread over many physical instances.
    """

    candidate_ids: np.ndarray
    candidate_probabilities: np.ndarray
    out_of_map_probabilities: np.ndarray
    truncated_tail_probabilities: np.ndarray
    best_similarities: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.candidate_ids, dtype=np.int64)
        probability = np.asarray(self.candidate_probabilities, dtype=np.float64)
        out_of_map = np.asarray(self.out_of_map_probabilities, dtype=np.float64).reshape(-1)
        tail = np.asarray(self.truncated_tail_probabilities, dtype=np.float64).reshape(-1)
        best = np.asarray(self.best_similarities, dtype=np.float64).reshape(-1)
        if (
            ids.ndim != 2
            or probability.shape != ids.shape
            or out_of_map.shape != (ids.shape[0],)
            or tail.shape != out_of_map.shape
            or best.shape != out_of_map.shape
            or np.any(probability < -1e-8)
            or np.any((out_of_map < -1e-8) | (out_of_map > 1.0 + 1e-8))
            or np.any((tail < -1e-8) | (tail > 1.0 + 1e-8))
            or np.any(~np.isfinite(probability))
            or np.any(~np.isfinite(out_of_map))
            or np.any(~np.isfinite(tail))
            or np.any(~np.isfinite(best))
        ):
            raise ValueError("invalid sparse maplet posterior")
        total = np.sum(probability, axis=1) + out_of_map + tail
        if np.any(np.abs(total - 1.0) > 2e-5):
            raise ValueError("sparse maplet posterior does not conserve probability mass")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "candidate_probabilities", probability.astype(np.float32))
        object.__setattr__(self, "out_of_map_probabilities", out_of_map.astype(np.float32))
        object.__setattr__(self, "truncated_tail_probabilities", tail.astype(np.float32))
        object.__setattr__(self, "best_similarities", best.astype(np.float32))

    @property
    def unresolved_probabilities(self) -> np.ndarray:
        """Legacy null branch used only when a consumer cannot model the tail."""

        return self.out_of_map_probabilities + self.truncated_tail_probabilities

    @property
    def in_map_probabilities(self) -> np.ndarray:
        return 1.0 - self.out_of_map_probabilities


def retrieve_maplet_posterior_decomposed(
    descriptor: np.ndarray,
    map_descriptor: np.ndarray,
    maplet_ids: np.ndarray,
    valid_maplets: np.ndarray,
    *,
    maximum_candidates: int,
    temperature: float,
    null_similarity_center: float,
    null_similarity_scale: float,
) -> SparseMapletPosterior:
    """Retrieve sparse physical identities without conflating tail and null."""

    query = np.asarray(descriptor, dtype=np.float32)
    map_feature = np.asarray(map_descriptor, dtype=np.float32)
    identifiers = np.asarray(maplet_ids, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid_maplets, dtype=bool).reshape(-1)
    if (
        query.ndim != 2
        or map_feature.ndim != 2
        or query.shape[1] != map_feature.shape[1]
        or identifiers.shape != (map_feature.shape[0],)
        or valid.shape != identifiers.shape
        or not np.any(valid)
        or int(maximum_candidates) <= 0
    ):
        raise ValueError("invalid maplet retrieval arrays")
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    map_feature = map_feature / np.maximum(np.linalg.norm(map_feature, axis=1, keepdims=True), 1e-8)
    score = query @ map_feature.T
    score[:, ~valid] = -np.inf
    count = min(int(maximum_candidates), int(np.sum(valid)))
    columns = np.argpartition(-score, kth=count - 1, axis=1)[:, :count]
    order = np.argsort(-np.take_along_axis(score, columns, axis=1), axis=1, kind="stable")
    columns = np.take_along_axis(columns, order, axis=1)
    selected_score = np.take_along_axis(score, columns, axis=1)
    maximum = np.max(score, axis=1, keepdims=True)
    exponential = np.exp((score - maximum) / max(float(temperature), 1e-4))
    exponential[:, ~valid] = 0.0
    normalizer = np.sum(exponential, axis=1, keepdims=True)
    conditional = np.take_along_axis(
        exponential / np.maximum(normalizer, 1e-12), columns, axis=1,
    )
    best = selected_score[:, 0]
    predicted_valid = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (best - float(null_similarity_center))
                / max(float(null_similarity_scale), 1e-4),
                -50.0,
                50.0,
            )
        )
    )
    probability = predicted_valid[:, None] * conditional
    out_of_map = 1.0 - predicted_valid
    tail = np.clip(predicted_valid - np.sum(probability, axis=1), 0.0, 1.0)
    return SparseMapletPosterior(
        identifiers[columns], probability, out_of_map, tail, best,
    )


def retrieve_maplet_posterior(
    descriptor: np.ndarray,
    map_descriptor: np.ndarray,
    maplet_ids: np.ndarray,
    valid_maplets: np.ndarray,
    *,
    maximum_candidates: int,
    temperature: float,
    null_similarity_center: float,
    null_similarity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posterior = retrieve_maplet_posterior_decomposed(
        descriptor,
        map_descriptor,
        maplet_ids,
        valid_maplets,
        maximum_candidates=maximum_candidates,
        temperature=temperature,
        null_similarity_center=null_similarity_center,
        null_similarity_scale=null_similarity_scale,
    )
    # Backward-compatible consumers cannot represent the sparse in-map tail.
    # Preserve their historical unresolved branch while new inference uses the
    # decomposed posterior directly.
    return (
        posterior.candidate_ids,
        posterior.candidate_probabilities,
        posterior.unresolved_probabilities,
        posterior.best_similarities,
    )


@dataclass(frozen=True)
class ValidityCalibration:
    center: float
    scale: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not np.isfinite(self.center) or not np.isfinite(self.scale) or float(self.scale) <= 0.0:
            raise ValueError("invalid validity calibration")
        metadata = dict(self.metadata)
        if metadata.get("artifact_type", CALIBRATION_SCHEMA) != CALIBRATION_SCHEMA:
            raise ValueError("not a Goal-Maplet validity calibration")
        object.__setattr__(self, "metadata", metadata)

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {"center": float(self.center), "scale": float(self.scale), "metadata": dict(self.metadata)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf8")
        return hashlib.sha256(payload).hexdigest()

    def predict_valid(self, similarity: np.ndarray) -> np.ndarray:
        value = (np.asarray(similarity, dtype=np.float64) - float(self.center)) / float(self.scale)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))

    def save_json(self, path: Path) -> None:
        payload = {
            "artifact_type": CALIBRATION_SCHEMA,
            "center": float(self.center),
            "scale": float(self.scale),
            "metadata": dict(self.metadata),
            "content_sha256": self.content_sha256,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "ValidityCalibration":
        payload = json.loads(Path(path).read_text())
        # The calibration command emits both a compact deployment artifact and
        # a richer metrics report.  They contain the same calibrated state.
        metadata = payload.get("metadata", payload.get("calibration_metadata"))
        expected_hash = payload.get("content_sha256", payload.get("calibration_sha256", ""))
        if metadata is None:
            raise ValueError("validity calibration metadata is missing")
        result = cls(float(payload["center"]), float(payload["scale"]), metadata)
        if str(expected_hash) != result.content_sha256:
            raise ValueError("validity calibration content hash mismatch")
        return result


def fit_validity_calibration(
    similarity: np.ndarray,
    valid_target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ValidityCalibration:
    score = np.asarray(similarity, dtype=np.float64).reshape(-1)
    target = np.asarray(valid_target, dtype=np.float64).reshape(-1)
    weight = np.ones_like(score) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if score.shape != target.shape or weight.shape != score.shape or score.size == 0:
        raise ValueError("calibration arrays differ or are empty")
    if np.any(~np.isfinite(score)) or np.any((target < 0.0) | (target > 1.0)) or np.any(weight <= 0.0):
        raise ValueError("invalid calibration samples")
    weight = weight / np.sum(weight)

    def objective(parameters: np.ndarray) -> float:
        center, log_scale = float(parameters[0]), float(parameters[1])
        scale = float(np.exp(log_scale))
        probability = 1.0 / (1.0 + np.exp(-np.clip((score - center) / scale, -50.0, 50.0)))
        loss = -target * np.log(np.maximum(probability, 1e-8))
        loss -= (1.0 - target) * np.log(np.maximum(1.0 - probability, 1e-8))
        return float(np.sum(weight * loss))

    initial_center = float(np.median(score))
    result = minimize(
        objective,
        np.asarray([initial_center, np.log(0.08)], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-1.0, 1.0), (np.log(0.005), np.log(1.0))),
    )
    if not result.success:
        raise RuntimeError(f"validity calibration failed: {result.message}")
    return ValidityCalibration(
        center=float(result.x[0]),
        scale=float(np.exp(result.x[1])),
        metadata={"artifact_type": CALIBRATION_SCHEMA, **dict(metadata or {})},
    )
