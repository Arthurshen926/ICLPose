"""Immutable baseline/optional pose contracts for abstaining promotion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


IMMUTABLE_POSE_RESULT_SCHEMA_VERSION = 1


def _finite_tuple(values: Sequence[float], *, count: int, name: str) -> tuple[float, ...]:
    output = tuple(float(value) for value in values)
    if len(output) != int(count) or not all(math.isfinite(value) for value in output):
        raise ValueError(f"{name} must contain {count} finite values")
    return output


def _canonical_evidence(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    output = tuple(sorted((str(key), float(value)) for key, value in values.items()))
    if any(not key or not math.isfinite(value) for key, value in output):
        raise ValueError("pose evidence must have non-empty keys and finite values")
    return output


@dataclass(frozen=True)
class BaselinePoseResult:
    pose_w2c: tuple[float, ...]
    source_policy: str
    match_source_query_rows: tuple[int, ...]
    match_track_ids: tuple[int, ...]
    match_xy: tuple[tuple[float, float], ...]
    fit_evidence: tuple[tuple[str, float], ...]
    verification_evidence: tuple[tuple[str, float], ...]
    score_schema_version: str
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        pose_w2c: Sequence[float],
        source_policy: str,
        match_source_query_rows: Sequence[int],
        match_track_ids: Sequence[int],
        match_xy: Sequence[Sequence[float]],
        fit_evidence: Mapping[str, float],
        verification_evidence: Mapping[str, float],
        score_schema_version: str,
        artifact_hash: str,
    ) -> "BaselinePoseResult":
        rows = tuple(int(value) for value in match_source_query_rows)
        tracks = tuple(int(value) for value in match_track_ids)
        xy = tuple(
            _finite_tuple(value, count=2, name="match_xy") for value in match_xy
        )
        if len(rows) != len(tracks) or len(rows) != len(xy):
            raise ValueError("baseline match arrays must have equal lengths")
        if any(value < 0 for value in rows) or any(value < 0 for value in tracks):
            raise ValueError("baseline match identities must be non-negative")
        if len(set(rows)) != len(rows):
            raise ValueError("baseline source query rows must be unique")
        if not str(source_policy) or not str(score_schema_version):
            raise ValueError("baseline policy and score schema must be named")
        if len(str(artifact_hash)) < 8:
            raise ValueError("baseline artifact hash is missing")
        return cls(
            pose_w2c=_finite_tuple(pose_w2c, count=16, name="pose_w2c"),
            source_policy=str(source_policy),
            match_source_query_rows=rows,
            match_track_ids=tracks,
            match_xy=xy,
            fit_evidence=_canonical_evidence(fit_evidence),
            verification_evidence=_canonical_evidence(verification_evidence),
            score_schema_version=str(score_schema_version),
            artifact_hash=str(artifact_hash),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "immutable_baseline_pose_result_v1",
            "schema_version": IMMUTABLE_POSE_RESULT_SCHEMA_VERSION,
            "pose_w2c": list(self.pose_w2c),
            "source_policy": self.source_policy,
            "matches": [
                {
                    "source_query_row": row,
                    "track_id": track,
                    "xy": list(xy),
                }
                for row, track, xy in zip(
                    self.match_source_query_rows, self.match_track_ids, self.match_xy
                )
            ],
            "fit_evidence": dict(self.fit_evidence),
            "verification_evidence": dict(self.verification_evidence),
            "score_schema_version": self.score_schema_version,
            "artifact_hash": self.artifact_hash,
        }

    @property
    def serialized(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.serialized).hexdigest()


@dataclass(frozen=True)
class OptionalPoseResult:
    parent_baseline_hash: str
    hypothesis_id: str
    pose_w2c: tuple[float, ...]
    absolute_evidence: tuple[tuple[str, float], ...]
    fixed_difference_evidence: tuple[tuple[str, float], ...]
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        parent_baseline_hash: str,
        hypothesis_id: str,
        pose_w2c: Sequence[float],
        absolute_evidence: Mapping[str, float],
        fixed_difference_evidence: Mapping[str, float],
        artifact_hash: str,
    ) -> "OptionalPoseResult":
        if len(str(parent_baseline_hash)) != 64:
            raise ValueError("optional pose must reference a full baseline digest")
        if not str(hypothesis_id) or len(str(artifact_hash)) < 8:
            raise ValueError("optional hypothesis identity is incomplete")
        return cls(
            parent_baseline_hash=str(parent_baseline_hash),
            hypothesis_id=str(hypothesis_id),
            pose_w2c=_finite_tuple(pose_w2c, count=16, name="pose_w2c"),
            absolute_evidence=_canonical_evidence(absolute_evidence),
            fixed_difference_evidence=_canonical_evidence(
                fixed_difference_evidence
            ),
            artifact_hash=str(artifact_hash),
        )


def select_optional_or_fallback(
    baseline: BaselinePoseResult,
    optional_hypotheses: Sequence[OptionalPoseResult],
    promotion_probabilities: Mapping[str, float],
    *,
    promotion_threshold: float,
) -> BaselinePoseResult | OptionalPoseResult:
    """Promote at most one child; otherwise return the exact baseline object."""

    if not 0.0 <= float(promotion_threshold) <= 1.0:
        raise ValueError("promotion_threshold must be in [0, 1]")
    eligible: list[tuple[float, str, OptionalPoseResult]] = []
    for hypothesis in optional_hypotheses:
        if hypothesis.parent_baseline_hash != baseline.digest:
            raise ValueError("optional hypothesis references a different baseline")
        probability = promotion_probabilities.get(hypothesis.hypothesis_id)
        if probability is None:
            continue
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("promotion probability must be finite and in [0, 1]")
        if value >= float(promotion_threshold):
            eligible.append((value, hypothesis.hypothesis_id, hypothesis))
    if not eligible:
        return baseline
    return max(eligible, key=lambda item: (item[0], item[1]))[2]

