"""Contracts for mixed, hypothesis-disjoint verification points.

The artifact defined here is deliberately target-free.  It contains query
appearance, fixed full-bank landmark candidates, and an explicit candidate
null mass.  SfM pose/residual labels are joined only by a separate audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MIXED_VERIFICATION_POINTS_FORMAT = "mixed_multiscale_verification_points_v1"
MIXED_VERIFICATION_POINTS_SCORING_COMPATIBILITY_FORMAT = (
    "mixed_multiscale_verification_points_scoring_compatibility_v1"
)
POINT_SOURCE_ALIKE = "alike_high_detail"
POINT_SOURCE_RADIO_INTERMEDIATE = "radio_intermediate_uniform_context"
POINT_SOURCE_RADIO_FINAL = "radio_final_uniform_context"
POINT_SOURCES = frozenset(
    {
        POINT_SOURCE_ALIKE,
        POINT_SOURCE_RADIO_INTERMEDIATE,
        POINT_SOURCE_RADIO_FINAL,
    }
)


def _canonical_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Copy a JSON-compatible mapping with deterministic nested key ordering."""

    try:
        canonical = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError("mixed verification metadata is not JSON-compatible") from error
    if not isinstance(canonical, dict):  # Defensive despite the input conversion.
        raise ValueError("mixed verification metadata is not an object")
    return canonical


def _declared_exported_splits(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = metadata.get("exported_splits")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("mixed verification exported splits are invalid")
    splits = tuple(str(value) for value in raw)
    if (
        not splits
        or len(splits) != len(set(splits))
        or set(splits) - {"train", "validation", "test"}
    ):
        raise ValueError("mixed verification exported splits are invalid")
    return tuple(sorted(splits))


def mixed_verification_points_scoring_compatibility(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return the split-independent provenance required for held-out scoring.

    The raw cache hash remains the normal checkpoint contract.  This manifest
    only permits a separately materialized held-out cache when every source
    that changes query descriptors or fixed global candidates agrees exactly.
    Per-split point membership, worker timing, and local source-point IDs are
    intentionally excluded because they must differ between train/validation
    and test-only artifacts.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("mixed verification scoring metadata is invalid")
    exported_splits = _declared_exported_splits(metadata)
    required_exact = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_evidence_format": "candidate_evidence_v3",
        "global_landmark_ann_used": True,
        "global_landmark_ann_scope": "full_projected_landmark_bank_only",
        "feature_key": "radio_final",
        "candidate_set": "fixed_full_global_faiss_top_l_unique_tracks",
        "candidate_top_k": 20,
        "candidate_prior_semantics": "fixed_top20_coarse_softmax_with_explicit_diagnostic_null_v1",
        "candidate_prior_pose_calibrated": False,
        "candidate_reselection": False,
        "query_split_source": "candidate_evidence_v3_split_names",
        "token_manifest_split_role": "feature_storage_only_not_evaluation_split",
        "test_target_labels_materialized": False,
    }
    if any(metadata.get(key) != value for key, value in required_exact.items()):
        raise ValueError("mixed verification scoring metadata has an incompatible contract")
    if bool(metadata.get("test_source_points_materialized")) != ("test" in exported_splits):
        raise ValueError("mixed verification test-source provenance is inconsistent")
    string_fields = (
        "query_manifest_sha256",
        "detector_query_cache_sha256",
        "candidate_evidence_sha256",
        "candidate_fit_artifact_sha256",
        "matcha_joint_checkpoint_sha256",
        "projected_landmark_bank_sha256",
        "descriptor_space_id",
        "projection_space_id",
        "faiss_index_cache_sha256",
        "faiss_index_metadata_sha256",
        "colmap_cameras_sha256",
        "colmap_images_sha256",
    )
    if any(not str(metadata.get(name, "")) for name in string_fields):
        raise ValueError("mixed verification scoring provenance is incomplete")
    numeric_fields = (
        "faiss_nprobe",
        "faiss_search_k",
        "candidate_prior_temperature",
        "candidate_null_probability",
    )
    try:
        nprobe = int(metadata["faiss_nprobe"])
        search_k = int(metadata["faiss_search_k"])
        temperature = float(metadata["candidate_prior_temperature"])
        null_probability = float(metadata["candidate_null_probability"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("mixed verification scoring numeric provenance is invalid") from error
    if (
        nprobe <= 0
        or search_k < 20
        or not np.isfinite(temperature)
        or temperature <= 0.0
        or not np.isfinite(null_probability)
        or not 0.0 <= null_probability < 1.0
    ):
        raise ValueError("mixed verification scoring numeric provenance is invalid")
    point_sources = metadata.get("point_sources")
    if not isinstance(point_sources, Mapping) or set(point_sources) != POINT_SOURCES:
        raise ValueError("mixed verification scoring point-source provenance is invalid")
    manifest: dict[str, object] = {
        "format": MIXED_VERIFICATION_POINTS_SCORING_COMPATIBILITY_FORMAT,
        "point_artifact_format": MIXED_VERIFICATION_POINTS_FORMAT,
        "query_manifest_sha256": str(metadata["query_manifest_sha256"]),
        "detector_query_cache_sha256": str(metadata["detector_query_cache_sha256"]),
        "candidate_evidence_format": str(metadata["candidate_evidence_format"]),
        "candidate_evidence_sha256": str(metadata["candidate_evidence_sha256"]),
        "candidate_fit_artifact_sha256": str(metadata["candidate_fit_artifact_sha256"]),
        "matcha_joint_checkpoint_sha256": str(metadata["matcha_joint_checkpoint_sha256"]),
        "projected_landmark_bank_sha256": str(metadata["projected_landmark_bank_sha256"]),
        "descriptor_space_id": str(metadata["descriptor_space_id"]),
        "projection_space_id": str(metadata["projection_space_id"]),
        "faiss_index_cache_sha256": str(metadata["faiss_index_cache_sha256"]),
        "faiss_index_metadata_sha256": str(metadata["faiss_index_metadata_sha256"]),
        "global_landmark_ann_scope": str(metadata["global_landmark_ann_scope"]),
        "faiss_nprobe": nprobe,
        "faiss_search_k": search_k,
        "feature_key": str(metadata["feature_key"]),
        "candidate_set": str(metadata["candidate_set"]),
        "candidate_top_k": int(metadata["candidate_top_k"]),
        "candidate_prior_semantics": str(metadata["candidate_prior_semantics"]),
        "candidate_prior_temperature": temperature,
        "candidate_null_probability": null_probability,
        "query_split_source": str(metadata["query_split_source"]),
        "token_manifest_split_role": str(metadata["token_manifest_split_role"]),
        "point_sources": _canonical_json_mapping(point_sources),
        "colmap_cameras_sha256": str(metadata["colmap_cameras_sha256"]),
        "colmap_images_sha256": str(metadata["colmap_images_sha256"]),
    }
    canonical = _canonical_json_mapping(manifest)
    serialized = metadata.get("scoring_compatibility")
    if serialized is not None:
        if not isinstance(serialized, Mapping) or _canonical_json_mapping(serialized) != canonical:
            raise ValueError("mixed verification serialized scoring compatibility is stale")
    return canonical


def validate_heldout_scoring_point_cache(
    *,
    metadata: Mapping[str, object],
    point_split_names: Sequence[str] | np.ndarray,
    requested_splits: Sequence[str],
) -> dict[str, object]:
    """Validate that a cache can supply the requested target-free held-out rows."""

    compatibility = mixed_verification_points_scoring_compatibility(metadata)
    declared = set(_declared_exported_splits(metadata))
    observed = {
        str(value) for value in np.asarray(point_split_names).astype(str).reshape(-1)
    }
    requested = {str(value) for value in requested_splits}
    if "test" in requested and observed != {"test"}:
        raise ValueError("mixed verification test scoring requires a test-only cache")
    if (
        not observed
        or observed - {"train", "validation", "test"}
        or not requested
        or requested - {"validation", "test"}
        or observed != declared
        or not requested.issubset(observed)
    ):
        raise ValueError("mixed verification held-out scoring split provenance is invalid")
    return compatibility


def fixed_topl_coarse_posterior(
    coarse_scores: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    temperature: float,
    null_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a fixed top-L score distribution plus an explicit unknown mass.

    This is a target-free diagnostic prior, not a calibrated identity model.
    Importantly, its retained candidate mass never changes when a later
    ablation masks candidate columns: removed mass must go to its null state.
    """

    scores = np.asarray(coarse_scores, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool)
    if scores.ndim != 2 or valid.shape != scores.shape:
        raise ValueError("coarse scores and candidate mask must have shape (N, L)")
    if (
        not np.isfinite(float(temperature))
        or float(temperature) <= 0.0
        or not np.isfinite(float(null_probability))
        or not 0.0 <= float(null_probability) < 1.0
    ):
        raise ValueError("coarse posterior temperature/null probability is invalid")
    if np.any(~np.isfinite(scores[valid])):
        raise ValueError("valid candidate coarse scores must be finite")

    output = np.zeros(scores.shape, dtype=np.float32)
    null = np.ones((scores.shape[0],), dtype=np.float32)
    retained_mass = 1.0 - float(null_probability)
    for row in range(scores.shape[0]):
        columns = np.flatnonzero(valid[row])
        if not len(columns):
            continue
        logits = scores[row, columns] / float(temperature)
        logits -= float(np.max(logits))
        weights = np.exp(np.maximum(logits, -80.0))
        weights /= max(float(np.sum(weights)), 1e-12)
        output[row, columns] = (retained_mass * weights).astype(np.float32)
        null[row] = np.float32(null_probability)
    if np.max(np.abs(output.sum(axis=1, dtype=np.float64) + null - 1.0)) > 2e-6:
        raise RuntimeError("fixed top-L posterior did not conserve mass")
    return output, null


def select_detector_rows_spatial_quota(
    *,
    source_rows: np.ndarray,
    xy: np.ndarray,
    scores: np.ndarray,
    excluded_rows: np.ndarray,
    point_count: int,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_columns: int,
) -> np.ndarray:
    """Choose score-ranked detector rows while cycling through image cells."""

    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    merits = np.asarray(scores, dtype=np.float64).reshape(-1)
    excluded = np.asarray(excluded_rows, dtype=np.int64).reshape(-1)
    if (
        len(rows) == 0
        or coordinates.shape != (len(rows), 2)
        or merits.shape != (len(rows),)
        or len(np.unique(rows)) != len(rows)
        or len(np.unique(excluded)) != len(excluded)
        or int(point_count) <= 0
        or int(image_width) <= 0
        or int(image_height) <= 0
        or int(grid_rows) <= 0
        or int(grid_columns) <= 0
        or np.any(~np.isfinite(coordinates))
        or np.any(~np.isfinite(merits))
    ):
        raise ValueError("detector spatial-quota inputs are invalid")
    available = ~np.isin(rows, excluded)
    candidate_rows = rows[available]
    candidate_xy = coordinates[available]
    candidate_scores = merits[available]
    if int(point_count) > len(candidate_rows):
        raise ValueError("not enough detector rows remain after fit-row exclusion")
    columns = np.clip(
        np.floor(candidate_xy[:, 0] * int(grid_columns) / float(image_width)).astype(
            np.int64
        ),
        0,
        int(grid_columns) - 1,
    )
    grid_indices = np.clip(
        np.floor(candidate_xy[:, 1] * int(grid_rows) / float(image_height)).astype(
            np.int64
        ),
        0,
        int(grid_rows) - 1,
    )
    cells = grid_indices * int(grid_columns) + columns
    per_cell: list[np.ndarray] = []
    for cell in range(int(grid_rows) * int(grid_columns)):
        local = np.flatnonzero(cells == cell)
        # lexsort's final key is primary; row ID makes equal scores reproducible.
        order = local[np.lexsort((candidate_rows[local], -candidate_scores[local]))]
        per_cell.append(order)
    cursors = np.zeros((len(per_cell),), dtype=np.int64)
    selected: list[int] = []
    while len(selected) < int(point_count):
        progressed = False
        for cell, values in enumerate(per_cell):
            cursor = int(cursors[cell])
            if cursor >= len(values):
                continue
            selected.append(int(values[cursor]))
            cursors[cell] += 1
            progressed = True
            if len(selected) == int(point_count):
                break
        if not progressed:
            raise RuntimeError("spatial quota selection could not fill its point budget")
    output = candidate_rows[np.asarray(selected, dtype=np.int64)]
    if len(np.unique(output)) != len(output) or np.any(np.isin(output, excluded)):
        raise RuntimeError("detector spatial quota violated its held-out-row contract")
    return output


def select_disjoint_lattice_points(
    *,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_columns: int,
    primary_phase: tuple[float, float],
    alternate_phases: Sequence[tuple[float, float]],
    excluded_xy: np.ndarray,
    minimum_distance_px: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Place one broad-context point per grid cell away from fit points.

    A deterministic fallback is retained for pathological dense fit layouts;
    the caller receives its count and must not describe that case as strictly
    spatially disjoint.
    """

    width = int(image_width)
    height = int(image_height)
    rows = int(grid_rows)
    columns = int(grid_columns)
    phases = (tuple(primary_phase), *tuple(tuple(value) for value in alternate_phases))
    excluded = np.asarray(excluded_xy, dtype=np.float32).reshape(-1, 2)
    if (
        width <= 0
        or height <= 0
        or rows <= 0
        or columns <= 0
        or not phases
        or any(len(value) != 2 for value in phases)
        or any(not (0.0 < float(value[0]) < 1.0 and 0.0 < float(value[1]) < 1.0) for value in phases)
        or not np.isfinite(float(minimum_distance_px))
        or float(minimum_distance_px) < 0.0
        or np.any(~np.isfinite(excluded))
    ):
        raise ValueError("lattice point selection inputs are invalid")
    selected: list[np.ndarray] = []
    distances: list[float] = []
    fallback_count = 0
    for row in range(rows):
        for column in range(columns):
            candidates = []
            for phase_x, phase_y in phases:
                xy = np.asarray(
                    [
                        min((float(column) + float(phase_x)) * width / columns, width - 1.0),
                        min((float(row) + float(phase_y)) * height / rows, height - 1.0),
                    ],
                    dtype=np.float32,
                )
                if len(excluded):
                    distance = float(np.min(np.linalg.norm(excluded - xy[None], axis=1)))
                else:
                    distance = float("inf")
                candidates.append((xy, distance))
            accepted = next(
                (value for value in candidates if value[1] >= float(minimum_distance_px)),
                None,
            )
            if accepted is None:
                accepted = max(candidates, key=lambda value: value[1])
                fallback_count += 1
            selected.append(accepted[0])
            distances.append(float(accepted[1]))
    points = np.stack(selected, axis=0).astype(np.float32, copy=False)
    if points.shape != (rows * columns, 2):
        raise RuntimeError("lattice point selector emitted an invalid layout")
    finite_distances = np.asarray([value for value in distances if np.isfinite(value)])
    return points, {
        "point_count": int(len(points)),
        "fallback_count": int(fallback_count),
        "minimum_fit_distance_px": (
            None if not len(finite_distances) else float(np.min(finite_distances))
        ),
        "median_fit_distance_px": (
            None if not len(finite_distances) else float(np.median(finite_distances))
        ),
    }


@dataclass(frozen=True)
class MixedVerificationPoints:
    """Validated target-free query points and immutable global candidates."""

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    xy: np.ndarray
    point_sources: np.ndarray
    source_detector_rows: np.ndarray
    descriptors: np.ndarray
    candidate_bank_rows: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_prototype_ids: np.ndarray
    candidate_coarse_similarities: np.ndarray
    candidate_prior_probabilities: np.ndarray
    null_probabilities: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        point_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32)
        sources = np.asarray(self.point_sources).astype(str).reshape(-1)
        detector_rows = np.asarray(self.source_detector_rows, dtype=np.int64).reshape(-1)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        bank_rows = np.asarray(self.candidate_bank_rows, dtype=np.int64)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        prototypes = np.asarray(self.candidate_prototype_ids, dtype=np.int64)
        similarities = np.asarray(self.candidate_coarse_similarities, dtype=np.float32)
        probabilities = np.asarray(self.candidate_prior_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        count = len(point_ids)
        if (
            count == 0
            or len(np.unique(point_ids)) != count
            or not (
                query_ids.shape
                == splits.shape
                == sources.shape
                == detector_rows.shape
                == null.shape
                == (count,)
            )
            or xy.shape != (count, 2)
            or descriptors.ndim != 2
            or descriptors.shape[0] != count
            or bank_rows.ndim != 2
            or not (
                bank_rows.shape
                == tracks.shape
                == prototypes.shape
                == similarities.shape
                == probabilities.shape
            )
            or bank_rows.shape[0] != count
            or bank_rows.shape[1] == 0
            or set(sources.tolist()) - POINT_SOURCES
            or set(splits.tolist()) - {"train", "validation", "test"}
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(descriptors))
            or np.any(~np.isfinite(null))
            or np.any((null < 0.0) | (null > 1.0))
        ):
            raise ValueError("mixed verification point arrays are invalid")
        valid = tracks >= 0
        if (
            np.any(valid != (bank_rows >= 0))
            or np.any(valid != (prototypes >= 0))
            or np.any(~np.isfinite(similarities[valid]))
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(np.abs(probabilities[~valid]) > 1e-6)
        ):
            raise ValueError("mixed verification candidates are invalid")
        for row in range(count):
            values = tracks[row, valid[row]]
            if len(values) != len(np.unique(values)):
                raise ValueError("mixed verification candidates repeat a physical track")
        mass = probabilities.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
        if np.max(np.abs(mass - 1.0)) > 2e-5:
            raise ValueError("mixed verification candidate/null mass is not conserved")
        metadata = dict(self.metadata)
        if metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT:
            raise ValueError("mixed verification artifact format is unsupported")
        if (
            metadata.get("contains_ground_truth") is not False
            or metadata.get("contains_target_errors") is not False
            or metadata.get("pose_or_ground_truth_used") is not False
            or metadata.get("image_retrieval_or_submap_used") is not False
            or metadata.get("render") is not False
        ):
            raise ValueError("mixed verification artifact violates the target-free protocol")
        object.__setattr__(self, "source_point_ids", point_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "point_sources", sources)
        object.__setattr__(self, "source_detector_rows", detector_rows)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "candidate_bank_rows", bank_rows)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_prototype_ids", prototypes)
        object.__setattr__(self, "candidate_coarse_similarities", similarities)
        object.__setattr__(self, "candidate_prior_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "metadata", metadata)

    def rows_for_query(self, query_id: str) -> np.ndarray:
        return np.flatnonzero(self.query_ids == str(query_id))


def load_mixed_verification_points(path: Path) -> MixedVerificationPoints:
    required = {
        "source_point_ids",
        "query_ids",
        "split_names",
        "xy",
        "point_sources",
        "source_detector_rows",
        "descriptors",
        "candidate_bank_rows",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "candidate_coarse_similarities",
        "candidate_prior_probabilities",
        "null_probabilities",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"mixed verification artifact lacks {sorted(missing)}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        arrays = {key: np.asarray(payload[key]) for key in required - {"metadata_json"}}
    return MixedVerificationPoints(metadata=metadata, **arrays)
