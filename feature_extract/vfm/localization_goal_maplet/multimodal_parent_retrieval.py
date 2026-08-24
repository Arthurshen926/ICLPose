"""Anonymous multimode RADIO readout for one physical parent probability.

A large physical region may contain several visually distinct surfaces.  A
single normalized mean can cancel those modes.  This module deterministically
compresses the canonical primitive codes of each parent into a small anonymous
mixture, then produces exactly one score per physical parent with an
area-normalized log-mean-exp.  Mode multiplicity therefore cannot create extra
softmax prior mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical_field import CanonicalSurfaceField
from .lineage import arrays_sha256
from .physical_map import GoalMapletPhysicalMap


PARENT_SCORE_SINGLE_MEAN = "single_parent_mean_v1"
PARENT_SCORE_ANONYMOUS_MODES = (
    "anonymous_primitive_modes_area_logmeanexp_v1"
)
CHILD_SCORE_SINGLE_MEAN = "single_child_mean_v1"
CHILD_SCORE_ANONYMOUS_MODES = (
    "anonymous_child_primitive_modes_logmeanexp_v1"
)


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class AnonymousParentModeReadout:
    descriptors: np.ndarray
    weights: np.ndarray
    parent_coverage: np.ndarray

    def __post_init__(self) -> None:
        descriptor = np.asarray(self.descriptors, dtype=np.float32)
        weight = np.asarray(self.weights, dtype=np.float32)
        coverage = np.asarray(self.parent_coverage, dtype=np.float32).reshape(-1)
        if (
            descriptor.ndim != 3
            or weight.shape != descriptor.shape[:2]
            or coverage.shape != (descriptor.shape[0],)
            or np.any(~np.isfinite(descriptor))
            or np.any(~np.isfinite(weight))
            or np.any(weight < 0.0)
            or np.any((coverage < 0.0) | (coverage > 1.0 + 1e-6))
        ):
            raise ValueError("invalid anonymous parent mode readout")
        valid = weight > 0.0
        norms = np.linalg.norm(descriptor, axis=2)
        if np.any(np.abs(norms[valid] - 1.0) > 2e-5):
            raise ValueError("anonymous parent modes must be unit normalized")
        total = np.sum(weight, axis=1)
        if np.any((coverage > 0.0) != (total > 0.0)) or np.any(
            np.abs(total[total > 0.0] - 1.0) > 2e-5
        ):
            raise ValueError("anonymous parent mode weights must sum to one")
        object.__setattr__(self, "descriptors", descriptor)
        object.__setattr__(self, "weights", weight)
        object.__setattr__(self, "parent_coverage", coverage)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256(
            {
                "descriptors": self.descriptors,
                "weights": self.weights,
                "parent_coverage": self.parent_coverage,
            }
        )


@dataclass(frozen=True)
class AnonymousChildModeReadout:
    """A fixed number of appearance modes for each physical child event.

    Modes are alternatives inside one mutually-exclusive child identity.  They
    must therefore be marginalized before the child softmax, never emitted as
    separate candidates that would reward mode multiplicity.
    """

    descriptors: np.ndarray
    weights: np.ndarray
    child_coverage: np.ndarray

    def __post_init__(self) -> None:
        descriptor = np.asarray(self.descriptors, dtype=np.float32)
        weight = np.asarray(self.weights, dtype=np.float32)
        coverage = np.asarray(self.child_coverage, dtype=np.float32).reshape(-1)
        if (
            descriptor.ndim != 3
            or weight.shape != descriptor.shape[:2]
            or coverage.shape != (descriptor.shape[0],)
            or np.any(~np.isfinite(descriptor))
            or np.any(~np.isfinite(weight))
            or np.any(weight < 0.0)
            or np.any((coverage < 0.0) | (coverage > 1.0 + 1e-6))
        ):
            raise ValueError("invalid anonymous child mode readout")
        valid = weight > 0.0
        norms = np.linalg.norm(descriptor, axis=2)
        if np.any(np.abs(norms[valid] - 1.0) > 2e-5):
            raise ValueError("anonymous child modes must be unit normalized")
        total = np.sum(weight, axis=1)
        if np.any((coverage > 0.0) != (total > 0.0)) or np.any(
            np.abs(total[total > 0.0] - 1.0) > 2e-5
        ):
            raise ValueError("anonymous child mode weights must sum to one")
        object.__setattr__(self, "descriptors", descriptor)
        object.__setattr__(self, "weights", weight)
        object.__setattr__(self, "child_coverage", coverage)

    @property
    def content_sha256(self) -> str:
        return arrays_sha256(
            {
                "descriptors": self.descriptors,
                "weights": self.weights,
                "child_coverage": self.child_coverage,
            }
        )


def build_anonymous_child_mode_readout(
    field: CanonicalSurfaceField,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_modes: int = 4,
    minimum_angular_residual: float = 0.05,
    refinement_iterations: int = 2,
) -> AnonymousChildModeReadout:
    """Compress primitive codes within each child without merging its modes."""

    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    if (
        int(maximum_modes) <= 0
        or not 0.0 <= float(minimum_angular_residual) <= 2.0
        or int(refinement_iterations) < 0
    ):
        raise ValueError("invalid anonymous child mode configuration")
    child_count = int(physical.child_parent_rows.size)
    mode_count = int(maximum_modes)
    descriptors = np.zeros(
        (child_count, mode_count, field.feature_dim), dtype=np.float32
    )
    weights = np.zeros((child_count, mode_count), dtype=np.float32)
    coverage = np.zeros((child_count,), dtype=np.float32)
    field_row = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row[field.primitive_rows] = np.arange(field.primitive_rows.size)
    for child in range(child_count):
        start = int(physical.child_member_offsets[child])
        end = int(physical.child_member_offsets[child + 1])
        primitive = np.asarray(
            physical.child_member_primitive_rows[start:end], dtype=np.int64
        )
        source = field_row[primitive]
        valid = source >= 0
        raw_mass = np.maximum(
            np.asarray(
                physical.child_member_weights[start:end], dtype=np.float64
            ),
            0.0,
        )
        coverage[child] = float(
            np.sum(raw_mass[valid]) / max(float(np.sum(raw_mass)), 1e-12)
        )
        if not np.any(valid):
            continue
        codes = _unit(field.codes[source[valid]])
        mass = raw_mass[valid] * np.maximum(
            np.asarray(field.confidence[source[valid]], dtype=np.float64), 1e-3
        )
        mass /= max(float(np.sum(mass)), 1e-12)
        centers = [_unit(np.sum(mass[:, None] * codes, axis=0)[None])[0]]
        while len(centers) < mode_count and len(centers) < codes.shape[0]:
            similarity = codes @ np.asarray(centers, dtype=np.float32).T
            residual = np.maximum(1.0 - np.max(similarity, axis=1), 0.0)
            if float(np.max(residual, initial=0.0)) < float(
                minimum_angular_residual
            ):
                break
            priority = residual * np.sqrt(
                mass / max(float(np.max(mass)), 1e-12)
            )
            centers.append(codes[int(np.argmax(priority))])
        center = _unit(np.asarray(centers, dtype=np.float32))
        cluster_mass = np.zeros((center.shape[0],), dtype=np.float64)
        for _ in range(max(1, int(refinement_iterations) + 1)):
            assignment = np.argmax(codes @ center.T, axis=1)
            updated: list[np.ndarray] = []
            kept_mass: list[float] = []
            for mode in range(center.shape[0]):
                chosen = assignment == mode
                if not np.any(chosen):
                    continue
                value = np.sum(mass[chosen, None] * codes[chosen], axis=0)
                updated.append(_unit(value[None])[0])
                kept_mass.append(float(np.sum(mass[chosen])))
            center = _unit(np.asarray(updated, dtype=np.float32))
            cluster_mass = np.asarray(kept_mass, dtype=np.float64)
        order = np.lexsort(
            (np.arange(cluster_mass.size, dtype=np.int64), -cluster_mass)
        )
        center = center[order]
        cluster_mass = cluster_mass[order]
        cluster_mass /= max(float(np.sum(cluster_mass)), 1e-12)
        count = min(mode_count, int(center.shape[0]))
        descriptors[child, :count] = center[:count]
        weights[child, :count] = cluster_mass[:count].astype(np.float32)
    return AnonymousChildModeReadout(descriptors, weights, coverage)


def build_anonymous_parent_mode_readout(
    field: CanonicalSurfaceField,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_modes: int = 4,
    minimum_angular_residual: float = 0.05,
    refinement_iterations: int = 2,
) -> AnonymousParentModeReadout:
    """Compress canonical primitive features into deterministic parent modes."""

    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")
    if (
        int(maximum_modes) <= 0
        or not 0.0 <= float(minimum_angular_residual) <= 2.0
        or int(refinement_iterations) < 0
    ):
        raise ValueError("invalid anonymous parent mode configuration")
    parent_count = int(physical.maplet_ids.size)
    mode_count = int(maximum_modes)
    descriptors = np.zeros(
        (parent_count, mode_count, field.feature_dim), dtype=np.float32
    )
    weights = np.zeros((parent_count, mode_count), dtype=np.float32)
    coverage = np.zeros((parent_count,), dtype=np.float32)
    field_row = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row[field.primitive_rows] = np.arange(field.primitive_rows.size)
    for parent in range(parent_count):
        start = int(physical.membership_offsets[parent])
        end = int(physical.membership_offsets[parent + 1])
        primitive = np.asarray(
            physical.membership_primitive_rows[start:end], dtype=np.int64
        )
        source = field_row[primitive]
        valid = source >= 0
        raw_mass = np.maximum(
            np.asarray(physical.membership_weights[start:end], dtype=np.float64),
            0.0,
        )
        coverage[parent] = float(
            np.sum(raw_mass[valid]) / max(float(np.sum(raw_mass)), 1e-12)
        )
        if not np.any(valid):
            continue
        codes = _unit(field.codes[source[valid]])
        mass = raw_mass[valid] * np.maximum(
            np.asarray(field.confidence[source[valid]], dtype=np.float64), 1e-3
        )
        mass /= max(float(np.sum(mass)), 1e-12)
        centers = [_unit(np.sum(mass[:, None] * codes, axis=0)[None])[0]]
        while len(centers) < mode_count and len(centers) < codes.shape[0]:
            similarity = codes @ np.asarray(centers, dtype=np.float32).T
            residual = np.maximum(1.0 - np.max(similarity, axis=1), 0.0)
            if float(np.max(residual, initial=0.0)) < float(
                minimum_angular_residual
            ):
                break
            priority = residual * np.sqrt(mass / max(float(np.max(mass)), 1e-12))
            centers.append(codes[int(np.argmax(priority))])
        center = _unit(np.asarray(centers, dtype=np.float32))
        assignment = np.zeros((codes.shape[0],), dtype=np.int64)
        cluster_mass = np.zeros((center.shape[0],), dtype=np.float64)
        for _ in range(max(1, int(refinement_iterations) + 1)):
            assignment = np.argmax(codes @ center.T, axis=1)
            updated: list[np.ndarray] = []
            kept_mass: list[float] = []
            for mode in range(center.shape[0]):
                chosen = assignment == mode
                if not np.any(chosen):
                    continue
                value = np.sum(mass[chosen, None] * codes[chosen], axis=0)
                updated.append(_unit(value[None])[0])
                kept_mass.append(float(np.sum(mass[chosen])))
            center = _unit(np.asarray(updated, dtype=np.float32))
            cluster_mass = np.asarray(kept_mass, dtype=np.float64)
        order = np.lexsort(
            (np.arange(cluster_mass.size, dtype=np.int64), -cluster_mass)
        )
        center = center[order]
        cluster_mass = cluster_mass[order]
        cluster_mass /= max(float(np.sum(cluster_mass)), 1e-12)
        count = min(mode_count, int(center.shape[0]))
        descriptors[parent, :count] = center[:count]
        weights[parent, :count] = cluster_mass[:count].astype(np.float32)
    return AnonymousParentModeReadout(descriptors, weights, coverage)


def score_anonymous_parent_modes(
    query_descriptors: np.ndarray,
    readout: AnonymousParentModeReadout,
    *,
    mode_temperature: float = 0.03,
) -> np.ndarray:
    """Return one multiplicity-normalized compatibility per query×parent."""

    query = _unit(np.asarray(query_descriptors, dtype=np.float32))
    if query.ndim != 2 or query.shape[1] != readout.descriptors.shape[2]:
        raise ValueError("query/anonymous parent mode dimensions differ")
    temperature = max(float(mode_temperature), 1e-4)
    parent_count, maximum_modes, dimension = readout.descriptors.shape
    similarity = query @ readout.descriptors.reshape(-1, dimension).T
    similarity = similarity.reshape(query.shape[0], parent_count, maximum_modes)
    valid = readout.weights > 0.0
    similarity[:, ~valid] = -np.inf
    maximum = np.max(similarity, axis=2)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponential = np.exp((similarity - safe_maximum[..., None]) / temperature)
    exponential[:, ~valid] = 0.0
    mixture = np.sum(exponential * readout.weights[None, :, :], axis=2)
    score = safe_maximum + temperature * np.log(np.maximum(mixture, 1e-30))
    score[:, readout.parent_coverage <= 0.0] = -np.inf
    return score.astype(np.float32)
