"""Audit target-free support-view mass for frozen candidate evidence.

Pair-conditioned appearance or geometry evidence is expensive because one query
image may reference many fixed maplet support images.  This audit measures that
cost without reading poses, targets, residuals, or prediction scores.  It only
uses the immutable S0 candidate posterior and the predeclared maplet view
mixture, then asks how much *existing mixture mass* is covered by a fixed pool
of support images.

The result is a feasibility diagnostic, not a support-image selector: an image
outside a hypothetical pool remains unobserved/unknown.  It must never be
silently renormalized away or treated as evidence against a candidate.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_maplet_support_fields,
    _resolve_rows,
)
from feature_extract.vfm.artifacts import file_sha256_short


FROZEN_APPEARANCE_FORMAT = "frozen_multiscale_candidate_absolute_appearance_v1"
SUPPORT_VIEW_MASS_AUDIT_FORMAT = "fixed_support_view_mass_audit_v1"


@dataclass(frozen=True)
class FrozenSupportViewMassShard:
    """One query's immutable candidate-plus-view mixture."""

    path: Path
    query_id: str
    split_name: str
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    candidate_view_weights: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        probabilities = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        weights = np.asarray(self.candidate_view_weights, dtype=np.float32)
        count = len(null)
        if (
            not str(self.query_id)
            or not str(self.split_name)
            or tracks.ndim != 2
            or tracks.shape[0] != count
            or probabilities.shape != tracks.shape
            or weights.ndim != 3
            or weights.shape[:2] != tracks.shape
            or weights.shape[2] == 0
            or np.any(~np.isfinite(probabilities))
            or np.any(~np.isfinite(null))
            or np.any(~np.isfinite(weights))
            or np.any(probabilities < 0.0)
            or np.any(null <= 0.0)
            or np.any(weights < 0.0)
            or np.max(
                np.abs(probabilities.sum(axis=1, dtype=np.float64) + null.astype(np.float64) - 1.0)
            )
            > 2e-4
        ):
            raise ValueError("frozen support-view mass shard arrays are invalid")
        supported = weights.sum(axis=2, dtype=np.float64)
        if (
            np.any((probabilities > 0.0) & ~np.isclose(supported, 1.0, atol=2e-4))
            or np.any((probabilities <= 0.0) & (supported > 2e-4))
        ):
            raise ValueError("frozen support-view weights do not preserve candidate mass")
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "candidate_view_weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--appearance-artifacts",
        help="comma-separated frozen appearance artifact paths",
    )
    source.add_argument(
        "--appearance-artifact-glob",
        help="glob resolving frozen appearance artifact paths",
    )
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument(
        "--pool-sizes",
        default="1,4,8,16,32,64,128,256,512",
        help="comma-separated support-image pool sizes to report",
    )
    parser.add_argument(
        "--coverage-levels",
        default="0.50,0.75,0.90,0.95,0.99",
        help="comma-separated candidate-view mass coverage levels",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _parse_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not parsed or len(set(parsed)) != len(parsed) or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} must contain unique positive integers")
    return tuple(sorted(parsed))


def _parse_floats(value: str, *, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if (
        not parsed
        or len(set(parsed)) != len(parsed)
        or any(not 0.0 < item <= 1.0 for item in parsed)
    ):
        raise ValueError(f"{name} must contain unique values in (0, 1]")
    return tuple(sorted(parsed))


def _artifact_paths(*, paths_value: str | None, glob_value: str | None) -> tuple[Path, ...]:
    if paths_value is not None:
        paths = tuple(Path(item.strip()) for item in str(paths_value).split(",") if item.strip())
    elif glob_value is not None:
        paths = tuple(Path(item) for item in sorted(glob.glob(str(glob_value))))
    else:  # pragma: no cover - argparse enforces one source.
        raise ValueError("appearance artifacts are required")
    if not paths or len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("appearance artifacts must be unique existing files")
    return paths


def _load_metadata(payload: Mapping[str, np.ndarray], *, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{path}: frozen appearance artifact lacks metadata")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    strict = metadata.get("strict_frozen_appearance_contract") if isinstance(metadata, dict) else None
    expected = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FROZEN_APPEARANCE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or not isinstance(strict, Mapping)
        or any(strict.get(key) is not expected_value for key, expected_value in expected.items())
        or int(strict.get("fixed_candidate_top_k", -1)) != 20
    ):
        raise ValueError(f"{path}: artifact violates the fixed target-free S0 appearance contract")
    return metadata


def load_frozen_support_view_mass_shard(path: Path) -> FrozenSupportViewMassShard:
    """Load only inference-safe mixture arrays from one frozen artifact."""

    required = (
        "metadata_json",
        "verification_query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: frozen appearance artifact lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    metadata = _load_metadata(arrays, path=path)
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    if (
        len(query_ids) != 192
        or len(set(query_ids.tolist())) != 1
        or len(split_names) != len(query_ids)
        or len(set(split_names.tolist())) != 1
        or str(metadata.get("query_id", "")) != str(query_ids[0])
        or str(metadata.get("split_name", "")) != str(split_names[0])
        or int(metadata.get("row_count", -1)) != len(query_ids)
    ):
        raise ValueError(f"{path}: frozen appearance query identity is inconsistent")
    return FrozenSupportViewMassShard(
        path=Path(path),
        query_id=str(query_ids[0]),
        split_name=str(split_names[0]),
        candidate_track_ids=np.asarray(arrays["candidate_track_ids"], dtype=np.int64),
        candidate_probabilities=np.asarray(arrays["candidate_probabilities"], dtype=np.float32),
        null_probabilities=np.asarray(arrays["null_probabilities"], dtype=np.float32),
        candidate_view_weights=np.asarray(arrays["candidate_view_weights"], dtype=np.float32),
        metadata=metadata,
    )


def _validate_maplet_lineage(*, shard: FrozenSupportViewMassShard, maplet_path: Path) -> None:
    inputs = shard.metadata.get("inputs")
    expected = inputs.get("maplet_support_index") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(expected, Mapping)
        or str(expected.get("sha256", "")) != file_sha256_short(maplet_path)
    ):
        raise ValueError(f"{shard.path}: maplet support index lineage differs from the appearance artifact")


def _expected_view_indices_and_weights(
    *,
    shard: FrozenSupportViewMassShard,
    maplet_track_ids: np.ndarray,
    support_image_indices: np.ndarray,
    support_coverage_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _resolve_rows(shard.candidate_track_ids, canonical_track_ids=maplet_track_ids)
    if np.any((shard.candidate_probabilities > 0.0) & (rows < 0)):
        raise ValueError("positive frozen candidate is absent from the maplet support index")
    expected_indices = np.full(
        (*shard.candidate_track_ids.shape, support_image_indices.shape[1]), -1, dtype=np.int64
    )
    expected_coverage = np.zeros_like(expected_indices)
    valid_rows = rows >= 0
    expected_indices[valid_rows] = support_image_indices[rows[valid_rows]]
    expected_coverage[valid_rows] = support_coverage_counts[rows[valid_rows]]
    valid = (expected_indices >= 0) & (expected_coverage > 0)
    expected_weights = np.where(valid, expected_coverage, 0.0).astype(np.float32)
    normalizer = expected_weights.sum(axis=2, keepdims=True)
    expected_weights = np.divide(
        expected_weights,
        normalizer,
        out=np.zeros_like(expected_weights),
        where=normalizer > 0.0,
    )
    expected_weights = np.where(
        (shard.candidate_probabilities > 0.0)[..., None], expected_weights, 0.0
    ).astype(np.float32, copy=False)
    if np.max(np.abs(expected_weights - shard.candidate_view_weights)) > 2e-5:
        raise ValueError("appearance artifact view weights differ from its fixed maplet mixture")
    return expected_indices, expected_weights


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) == 0 or np.any(~np.isfinite(numeric)):
        raise ValueError("summary values are invalid")
    return {
        "min": float(np.min(numeric)),
        "p10": float(np.quantile(numeric, 0.10)),
        "median": float(np.median(numeric)),
        "mean": float(np.mean(numeric)),
        "p90": float(np.quantile(numeric, 0.90)),
        "max": float(np.max(numeric)),
    }


def _coverage_count(cumulative: np.ndarray, level: float) -> int:
    if cumulative.ndim != 1 or len(cumulative) == 0 or not 0.0 < float(level) <= 1.0:
        raise ValueError("coverage count inputs are invalid")
    return int(np.searchsorted(cumulative, float(level), side="left") + 1)


def audit_fixed_support_view_mass(
    *,
    appearance_artifacts: Sequence[Path],
    maplet_support_index: Path,
    output_dir: Path,
    pool_sizes: Sequence[int],
    coverage_levels: Sequence[float],
) -> dict[str, Any]:
    """Summarize target-free candidate-view mass per support image and query."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    pools = tuple(int(size) for size in pool_sizes)
    levels = tuple(float(level) for level in coverage_levels)
    if (
        not appearance_artifacts
        or len(set(Path(path) for path in appearance_artifacts)) != len(appearance_artifacts)
        or not pools
        or len(set(pools)) != len(pools)
        or any(size <= 0 for size in pools)
        or not levels
        or len(set(levels)) != len(levels)
        or any(not 0.0 < level <= 1.0 for level in levels)
    ):
        raise ValueError("support-view mass audit arguments are invalid")
    shards = tuple(load_frozen_support_view_mass_shard(Path(path)) for path in appearance_artifacts)
    if len({shard.query_id for shard in shards}) != len(shards):
        raise ValueError("frozen appearance artifacts repeat a query id")
    maplet_tracks, support_image_ids, support_indices, maplet_coverage_counts, _maplet_metadata = (
        _load_maplet_support_fields(Path(maplet_support_index))
    )
    if len(support_image_ids) == 0:
        raise ValueError("maplet support index has no support images")

    ordered = tuple(sorted(shards, key=lambda item: (item.split_name, item.query_id)))
    per_query_ids: list[str] = []
    per_query_splits: list[str] = []
    candidate_mass: list[float] = []
    null_mass: list[float] = []
    positive_image_count: list[int] = []
    ranked_image_ids: list[np.ndarray] = []
    ranked_normalized_mass: list[np.ndarray] = []
    ranked_cumulative_mass: list[np.ndarray] = []
    pool_coverages: list[list[float]] = []
    per_query_coverage_counts: list[list[int]] = []

    for shard in ordered:
        _validate_maplet_lineage(shard=shard, maplet_path=Path(maplet_support_index))
        view_indices, expected_weights = _expected_view_indices_and_weights(
            shard=shard,
            maplet_track_ids=maplet_tracks,
            support_image_indices=support_indices,
            support_coverage_counts=maplet_coverage_counts,
        )
        joint_mass = shard.candidate_probabilities[..., None] * expected_weights
        expected_candidate_mass = float(shard.candidate_probabilities.sum(dtype=np.float64))
        if not np.isclose(float(joint_mass.sum(dtype=np.float64)), expected_candidate_mass, atol=2e-4):
            raise RuntimeError("candidate-view mass is not conserved")
        active = joint_mass > 0.0
        if np.any(active & (view_indices < 0)):
            raise RuntimeError("positive candidate-view mass lacks a support image")
        image_mass = np.zeros((len(support_image_ids),), dtype=np.float64)
        np.add.at(image_mass, view_indices[active], joint_mass[active].astype(np.float64))
        if not np.isclose(float(image_mass.sum()), expected_candidate_mass, atol=2e-4):
            raise RuntimeError("support-image mass is not conserved")
        order = np.argsort(-image_mass, kind="stable")
        positive = image_mass[order] > 0.0
        order = order[positive]
        if len(order) == 0:
            raise RuntimeError("frozen candidate mass has no active support image")
        normalized = image_mass[order] / expected_candidate_mass
        cumulative = np.cumsum(normalized)
        cumulative[-1] = 1.0  # Avoid only floating-point accumulation noise in reports.
        per_query_ids.append(shard.query_id)
        per_query_splits.append(shard.split_name)
        candidate_mass.append(expected_candidate_mass)
        null_mass.append(float(shard.null_probabilities.sum(dtype=np.float64)))
        positive_image_count.append(int(len(order)))
        ranked_image_ids.append(np.asarray([support_image_ids[index] for index in order], dtype=np.str_))
        ranked_normalized_mass.append(normalized.astype(np.float32))
        ranked_cumulative_mass.append(cumulative.astype(np.float32))
        pool_coverages.append(
            [float(cumulative[min(int(size), len(cumulative)) - 1]) for size in pools]
        )
        per_query_coverage_counts.append([_coverage_count(cumulative, level) for level in levels])

    maximum_images = max(len(values) for values in ranked_image_ids)
    image_id_width = max(
        1,
        max(len(str(image_id)) for values in ranked_image_ids for image_id in values),
    )
    image_id_matrix = np.full(
        (len(ordered), maximum_images), "", dtype=f"<U{image_id_width}"
    )
    normalized_matrix = np.zeros((len(ordered), maximum_images), dtype=np.float32)
    cumulative_matrix = np.zeros((len(ordered), maximum_images), dtype=np.float32)
    for index, (ids, mass, cumulative) in enumerate(
        zip(ranked_image_ids, ranked_normalized_mass, ranked_cumulative_mass)
    ):
        image_id_matrix[index, : len(ids)] = ids
        normalized_matrix[index, : len(ids)] = mass
        cumulative_matrix[index, : len(ids)] = cumulative

    pool_array = np.asarray(pool_coverages, dtype=np.float64)
    count_array = np.asarray(per_query_coverage_counts, dtype=np.float64)
    splits = sorted(set(per_query_splits))
    split_summary: dict[str, Any] = {}
    for split in splits:
        rows = np.asarray([name == split for name in per_query_splits], dtype=bool)
        split_summary[split] = {
            "query_count": int(np.sum(rows)),
            "candidate_mass_per_query": _quantile_summary(np.asarray(candidate_mass)[rows]),
            "null_mass_per_query": _quantile_summary(np.asarray(null_mass)[rows]),
            "positive_support_image_count": _quantile_summary(
                np.asarray(positive_image_count, dtype=np.float64)[rows]
            ),
            "mass_coverage_by_pool_size": {
                str(size): _quantile_summary(pool_array[rows, column])
                for column, size in enumerate(pools)
            },
            "support_images_needed_by_coverage": {
                f"{level:.2f}": _quantile_summary(count_array[rows, column])
                for column, level in enumerate(levels)
            },
        }

    output.mkdir(parents=True)
    rows_path = output / "per_query_support_view_mass.npz"
    metadata = {
        "format": SUPPORT_VIEW_MASS_AUDIT_FORMAT,
        "version": 1,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "candidate_or_support_reselection": False,
        "fixed_global_top_l": True,
        "candidate_null_and_view_mass_conserved": True,
        "support_pool_interpretation": (
            "diagnostic coverage only; omitted support views remain explicit unknown evidence "
            "and are never renormalized or converted into a negative likelihood"
        ),
        "appearance_artifacts": [
            {"path": str(shard.path), "sha256": file_sha256_short(shard.path)}
            for shard in ordered
        ],
        "maplet_support_index": {
            "path": str(maplet_support_index),
            "sha256": file_sha256_short(Path(maplet_support_index)),
        },
        "implementation": {
            "path": str(Path(__file__)),
            "sha256": file_sha256_short(Path(__file__)),
        },
    }
    with rows_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_ids=np.asarray(per_query_ids, dtype=np.str_),
            split_names=np.asarray(per_query_splits, dtype=np.str_),
            candidate_posterior_mass=np.asarray(candidate_mass, dtype=np.float32),
            null_posterior_mass=np.asarray(null_mass, dtype=np.float32),
            positive_support_image_count=np.asarray(positive_image_count, dtype=np.int32),
            pool_sizes=np.asarray(pools, dtype=np.int32),
            pool_mass_coverage=pool_array.astype(np.float32),
            coverage_levels=np.asarray(levels, dtype=np.float32),
            support_images_needed=count_array.astype(np.int32),
            ranked_support_image_ids=image_id_matrix,
            ranked_normalized_candidate_view_mass=normalized_matrix,
            ranked_cumulative_candidate_view_mass=cumulative_matrix,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "fixed_support_view_mass_audit",
        "rows_path": str(rows_path),
        "rows_sha256": file_sha256_short(rows_path),
        "query_count": len(ordered),
        "pool_sizes": list(pools),
        "coverage_levels": list(levels),
        "split_summary": split_summary,
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": True,
            "support_view_mixture": "fixed_maplet_coverage_weighted_v1",
            "image_retrieval_or_submap_used": False,
            "render": False,
            "hypothetical_pool_missing_mass": "explicit_unknown_not_renormalized_v1",
        },
        "metadata": metadata,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _artifact_paths(
        paths_value=args.appearance_artifacts,
        glob_value=args.appearance_artifact_glob,
    )
    summary = audit_fixed_support_view_mass(
        appearance_artifacts=paths,
        maplet_support_index=Path(args.maplet_support_index),
        output_dir=Path(args.output_dir),
        pool_sizes=_parse_ints(args.pool_sizes, name="pool sizes"),
        coverage_levels=_parse_floats(args.coverage_levels, name="coverage levels"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
