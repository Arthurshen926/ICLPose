"""Typed identity and provenance contracts for candidate-level RGB measurement.

Candidate measurement is deliberately keyed by physical identity rather than
array position.  This prevents an RGB result produced for one landmark,
prototype, support-view set, crop convention, or checkpoint from being reused
for another candidate that happens to occupy the same top-L slot.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


CANDIDATE_MEASUREMENT_SCHEMA_VERSION = 1
CANDIDATE_CROP_GEOMETRY_VERSION = "real_rgb_centered_patch_v1"


def _stable_digest(payload: Mapping[str, object], *, length: int = 24) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


@dataclass(frozen=True)
class CandidateIdentityKey:
    """Identity of one query-token/3D-candidate/support-set measurement task."""

    query_id: str
    source_query_row: int
    track_id: int
    prototype_id: int
    support_view_set_id: str
    crop_geometry_version: str = CANDIDATE_CROP_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        if not str(self.query_id):
            raise ValueError("query_id must be non-empty")
        if int(self.source_query_row) < 0:
            raise ValueError("source_query_row must be non-negative")
        if int(self.track_id) < 0 or int(self.prototype_id) < 0:
            raise ValueError("track_id and prototype_id must be non-negative")
        if not str(self.support_view_set_id):
            raise ValueError("support_view_set_id must be non-empty")
        if not str(self.crop_geometry_version):
            raise ValueError("crop_geometry_version must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return _stable_digest(self.as_dict())


@dataclass(frozen=True)
class CandidateMeasurementCacheKey:
    """Strict cache key for one support image and one measurement checkpoint."""

    identity: CandidateIdentityKey
    support_image_id: str
    measurement_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not str(self.support_image_id):
            raise ValueError("support_image_id must be non-empty")
        if len(str(self.measurement_checkpoint_sha256)) < 8:
            raise ValueError("measurement checkpoint hash must contain at least 8 characters")

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.as_dict(),
            "support_image_id": str(self.support_image_id),
            "measurement_checkpoint_sha256": str(
                self.measurement_checkpoint_sha256
            ),
        }

    @property
    def digest(self) -> str:
        return _stable_digest(self.as_dict())


@dataclass(frozen=True)
class CandidateScoreBundle:
    """Named score semantics; probabilities cannot masquerade as similarities."""

    retrieval_similarity: float
    assignment_probability: float
    geometry_probability_5px: float
    support_view_probability: float

    def __post_init__(self) -> None:
        similarity = float(self.retrieval_similarity)
        if not math.isfinite(similarity) or not -1.0001 <= similarity <= 1.0001:
            raise ValueError("retrieval_similarity must be a finite cosine-like score")
        for name in (
            "assignment_probability",
            "geometry_probability_5px",
            "support_view_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True)
class MeasurementDecisionPolicy:
    """Versioned inference thresholds fitted on a named calibration artifact."""

    policy_version: str
    calibration_artifact_sha256: str
    geometry_probability_min: float
    measurement_probability_min: float
    update_probability_min: float

    def __post_init__(self) -> None:
        if not str(self.policy_version):
            raise ValueError("policy_version must be non-empty")
        if len(str(self.calibration_artifact_sha256)) < 8:
            raise ValueError("calibration artifact hash must contain at least 8 characters")
        for name in (
            "geometry_probability_min",
            "measurement_probability_min",
            "update_probability_min",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def support_view_set_id(support_image_ids: Iterable[str]) -> str:
    """Return an order-invariant id while preserving distinct view sets."""

    values = tuple(sorted({str(value) for value in support_image_ids if str(value)}))
    if not values:
        raise ValueError("support view set cannot be empty")
    return _stable_digest({"support_image_ids": values})


def validate_unique_cache_keys(keys: Iterable[CandidateMeasurementCacheKey]) -> None:
    """Reject duplicate logical tasks and the practically impossible hash collision."""

    seen_digest: dict[str, dict[str, object]] = {}
    for key in keys:
        payload = key.as_dict()
        digest = key.digest
        previous = seen_digest.get(digest)
        if previous is not None:
            if previous == payload:
                raise ValueError(f"duplicate candidate measurement cache key: {digest}")
            raise ValueError(f"candidate measurement cache-key digest collision: {digest}")
        seen_digest[digest] = payload

