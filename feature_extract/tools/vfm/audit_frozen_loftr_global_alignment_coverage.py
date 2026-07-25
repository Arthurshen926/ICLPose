"""Audit complete target-free LoFTR global-alignment coverage.

This runs before any SfM identity labels are loaded.  It proves that every
global-alignment overlay is an exact frozen top-20/source-row overlay and
reports whether anchor-level global-alignment evidence is available throughout
the fixed proposal ranks.  It deliberately does not inspect candidate scores
other than immutable posterior mass for coverage accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.export_frozen_loftr_global_alignment_shards import (
    ARTIFACT_FORMAT as EXPORT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_loftr_global_alignment import (
    FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT,
    LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    validate_frozen_loftr_colmap_coordinate_metadata,
)


ARTIFACT_FORMAT = "frozen_loftr_global_alignment_coverage_audit_v1"
EXPECTED_QUERY_COUNTS = {"train": 63, "validation": 21}
EXPECTED_ROWS_PER_QUERY = 192
RANK_BUCKETS = (
    ("rank1_5", 1, 5),
    ("rank6_10", 6, 10),
    ("rank11_20", 11, 20),
)
_BASE_FIELDS = (
    "verification_query_ids",
    "split_names",
    "verification_source_row_indices",
    "verification_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "candidate_view_weights",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _load_export_entries(export_manifest: Path) -> list[dict[str, str]]:
    manifest_path = Path(export_manifest)
    payload = json.loads(manifest_path.read_text())
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != EXPORT_FORMAT
        or payload.get("complete") is not True
        or not isinstance(payload.get("config"), Mapping)
        or not isinstance(payload.get("completed"), Mapping)
        or not isinstance(payload.get("failed"), Mapping)
        or payload["failed"]
    ):
        raise ValueError("LoFTR global-alignment export manifest is incomplete")
    config = payload["config"]
    sources = config.get("source_cache_manifest")
    completed = payload["completed"]
    if not isinstance(sources, list) or len(sources) != sum(EXPECTED_QUERY_COUNTS.values()):
        raise ValueError("LoFTR global-alignment export has incomplete source coverage")
    for path_key, hash_key in (
        ("maplet_support_index", "maplet_support_index_sha256"),
        ("support_geometry_index", "support_geometry_index_sha256"),
        ("loftr_checkpoint", "loftr_checkpoint_sha256"),
    ):
        path = Path(str(config.get(path_key, "")))
        if not path.is_file() or file_sha256_short(path) != str(config.get(hash_key, "")):
            raise ValueError(f"LoFTR global-alignment export dependency is stale: {path_key}")
    entries: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    seen_queries: set[str] = set()
    for source_cache in sources:
        if not isinstance(source_cache, Mapping):
            raise ValueError("LoFTR global-alignment source/cache entry is invalid")
        query_id = str(source_cache.get("query_id", ""))
        source = Path(str(source_cache.get("source", ""))).resolve()
        pair_cache = Path(str(source_cache.get("pair_cache", ""))).resolve()
        key = source.parent.name
        completed_entry = completed.get(key)
        if (
            not query_id
            or key in seen_keys
            or query_id in seen_queries
            or not isinstance(completed_entry, Mapping)
            or str(completed_entry.get("query_id", "")) != query_id
        ):
            raise ValueError("LoFTR global-alignment exporter source pairing is invalid")
        seen_keys.add(key)
        seen_queries.add(query_id)
        expected_pairs = (
            ("source", "source_sha256", source),
            ("pair_cache", "pair_cache_sha256", pair_cache),
        )
        for path_key, hash_key, path in expected_pairs:
            if (
                not path.is_file()
                or str(source_cache.get(path_key, "")) != str(path)
                or str(completed_entry.get(path_key, "")) != str(path)
                or file_sha256_short(path) != str(source_cache.get(hash_key, ""))
                or file_sha256_short(path) != str(completed_entry.get(hash_key, ""))
            ):
                raise ValueError(f"LoFTR global-alignment source hash is stale: {path_key}")
        output = Path(str(completed_entry.get("output", ""))).resolve()
        summary = Path(str(completed_entry.get("summary", ""))).resolve()
        for path, hash_key in ((output, "output_sha256"), (summary, "summary_sha256")):
            if not path.is_file() or file_sha256_short(path) != str(completed_entry.get(hash_key, "")):
                raise ValueError("LoFTR global-alignment output hash is stale")
        entries.append(
            {
                "query_id": query_id,
                "source": str(source),
                "pair_cache": str(pair_cache),
                "output": str(output),
            }
        )
    if len(entries) != len(completed) or len(entries) != len(sources):
        raise ValueError("LoFTR global-alignment export contains orphaned shards")
    return entries


def _load_source_base(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(_BASE_FIELDS).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: frozen source lacks {missing}")
        return {name: np.asarray(payload[name]).copy() for name in _BASE_FIELDS}


def _validate_alignment_shard(
    *, entry: Mapping[str, str]
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = Path(str(entry["source"]))
    output = Path(str(entry["output"]))
    source_arrays = _load_source_base(source)
    required = (
        *_BASE_FIELDS,
        "feature_names",
        "candidate_view_usable",
        "candidate_view_features",
        "candidate_view_pair_match_counts",
        "candidate_view_homography_model_valid",
        "candidate_usable_view_weight_mass",
        "metadata_json",
    )
    with np.load(output, allow_pickle=False) as payload:
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"{output}: alignment artifact lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{output}: alignment metadata is invalid")
    strict = metadata.get("strict_frozen_loftr_global_alignment_contract")
    expected_strict = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "all_mapping_images_pair_cached": True,
        "pair_cache_image_level_selection": False,
        "global_alignment_pose_free": True,
        "global_alignment_model_per_candidate": False,
        "homography_fits_all_cached_pair_matches": True,
        "source_to_colmap_coordinate_contract": True,
        "legacy_coordinate_ambiguous_evidence_rejected": True,
    }
    inputs = metadata.get("inputs")
    source_input = inputs.get("appearance_artifact") if isinstance(inputs, Mapping) else None
    cache_input = inputs.get("loftr_pair_cache") if isinstance(inputs, Mapping) else None
    if (
        metadata.get("format") != FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or int(metadata.get("fixed_candidate_top_k", -1)) != 20
        or tuple(np.asarray(arrays["feature_names"]).astype(str).reshape(-1).tolist())
        != LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES
        or not isinstance(strict, Mapping)
        or any(strict.get(key) is not value for key, value in expected_strict.items())
        or not isinstance(source_input, Mapping)
        or not isinstance(cache_input, Mapping)
        or Path(str(source_input.get("path", ""))).resolve() != source.resolve()
        or Path(str(cache_input.get("path", ""))).resolve()
        != Path(str(entry["pair_cache"])).resolve()
        or str(source_input.get("sha256", "")) != file_sha256_short(source)
        or str(cache_input.get("sha256", ""))
        != file_sha256_short(Path(str(entry["pair_cache"])))
    ):
        raise ValueError(f"{output}: global-alignment target-free contract is invalid")
    validate_frozen_loftr_colmap_coordinate_metadata(metadata)
    if any(not np.array_equal(arrays[name], source_arrays[name]) for name in _BASE_FIELDS):
        raise ValueError(f"{output}: global alignment changed its frozen source layout")
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
    usable = np.asarray(arrays["candidate_view_usable"], dtype=bool)
    features = np.asarray(arrays["candidate_view_features"], dtype=np.float32)
    pair_counts = np.asarray(arrays["candidate_view_pair_match_counts"], dtype=np.int32)
    model_valid = np.asarray(arrays["candidate_view_homography_model_valid"], dtype=bool)
    usable_mass = np.asarray(arrays["candidate_usable_view_weight_mass"], dtype=np.float32)
    if (
        len(query_ids) != EXPECTED_ROWS_PER_QUERY
        or len(set(query_ids.tolist())) != 1
        or query_ids[0] != str(entry["query_id"])
        or len(set(split_names.tolist())) != 1
        or split_names[0] not in EXPECTED_QUERY_COUNTS
        or tracks.shape != (EXPECTED_ROWS_PER_QUERY, 20)
        or probabilities.shape != tracks.shape
        or null.shape != (EXPECTED_ROWS_PER_QUERY,)
        or weights.ndim != 3
        or weights.shape[:2] != tracks.shape
        or usable.shape != weights.shape
        or model_valid.shape != weights.shape
        or pair_counts.shape != weights.shape
        or features.shape != (*weights.shape, len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES))
        or usable_mass.shape != tracks.shape
        or np.any(~np.isfinite(probabilities))
        or np.any(~np.isfinite(null))
        or np.any(~np.isfinite(weights))
        or np.any(probabilities < 0.0)
        or np.any(null <= 0.0)
        or np.any(weights < 0.0)
        or np.max(np.abs(probabilities.sum(axis=1, dtype=np.float64) + null - 1.0))
        > 2e-4
        or np.any(usable & ~model_valid)
        or np.any(model_valid & (weights <= 0.0))
        or np.any(pair_counts[model_valid] <= 0)
        or np.any(~np.isfinite(features[usable]))
        or not np.allclose(
            usable_mass,
            (weights * usable.astype(np.float32)).sum(axis=2),
            rtol=0.0,
            atol=2e-5,
        )
    ):
        raise ValueError(f"{output}: global-alignment arrays are invalid")
    return str(split_names[0]), probabilities, weights, usable, model_valid


def _empty_totals() -> dict[str, float | int]:
    return {
        "candidate_count": 0,
        "candidate_prior_mass": 0.0,
        "candidate_usable_count": 0,
        "candidate_prior_mass_covered": 0.0,
        "usable_view_weight_sum": 0.0,
        "fixed_view_count": 0,
        "model_valid_view_count": 0,
        "usable_view_count": 0,
    }


def _accumulate_rank_buckets(
    *,
    totals: dict[tuple[str, str], dict[str, float | int]],
    split: str,
    probabilities: np.ndarray,
    weights: np.ndarray,
    usable: np.ndarray,
    model_valid: np.ndarray,
) -> None:
    """Accumulate only fixed rank/mask coverage; no labels or target scores."""

    for name, start, stop in RANK_BUCKETS:
        selected = slice(start - 1, stop)
        probability = np.asarray(probabilities[:, selected], dtype=np.float64)
        view_weights = np.asarray(weights[:, selected], dtype=np.float64)
        view_usable = np.asarray(usable[:, selected], dtype=bool)
        view_model = np.asarray(model_valid[:, selected], dtype=bool)
        candidate_valid = probability > 0.0
        candidate_mass = probability[candidate_valid]
        candidate_usable_mass = (view_weights * view_usable).sum(axis=2)
        totals_for_bucket = totals[(split, name)]
        totals_for_bucket["candidate_count"] += int(np.sum(candidate_valid))
        totals_for_bucket["candidate_prior_mass"] += float(np.sum(candidate_mass))
        totals_for_bucket["candidate_usable_count"] += int(
            np.sum(candidate_valid & (candidate_usable_mass > 0.0))
        )
        totals_for_bucket["candidate_prior_mass_covered"] += float(
            np.sum(probability[candidate_valid & (candidate_usable_mass > 0.0)])
        )
        totals_for_bucket["usable_view_weight_sum"] += float(
            np.sum(candidate_usable_mass[candidate_valid])
        )
        fixed_views = candidate_valid[:, :, None] & (view_weights > 0.0)
        totals_for_bucket["fixed_view_count"] += int(np.sum(fixed_views))
        totals_for_bucket["model_valid_view_count"] += int(np.sum(view_model & fixed_views))
        totals_for_bucket["usable_view_count"] += int(np.sum(view_usable & fixed_views))


def _coverage_rows(
    totals: Mapping[tuple[str, str], Mapping[str, float | int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in EXPECTED_QUERY_COUNTS:
        for bucket, _start, _stop in RANK_BUCKETS:
            value = totals[(split, bucket)]
            candidate_count = int(value["candidate_count"])
            fixed_view_count = int(value["fixed_view_count"])
            prior_mass = float(value["candidate_prior_mass"])
            rows.append(
                {
                    "split": split,
                    "rank_bucket": bucket,
                    **value,
                    "candidate_usable_rate": 0.0
                    if candidate_count == 0
                    else float(value["candidate_usable_count"]) / candidate_count,
                    "candidate_prior_mass_coverage": 0.0
                    if prior_mass <= 0.0
                    else float(value["candidate_prior_mass_covered"]) / prior_mass,
                    "mean_usable_view_weight_mass": 0.0
                    if candidate_count == 0
                    else float(value["usable_view_weight_sum"]) / candidate_count,
                    "model_valid_view_rate": 0.0
                    if fixed_view_count == 0
                    else float(value["model_valid_view_count"]) / fixed_view_count,
                    "usable_view_rate": 0.0
                    if fixed_view_count == 0
                    else float(value["usable_view_count"]) / fixed_view_count,
                }
            )
    return rows


def audit_frozen_loftr_global_alignment_coverage(
    *, export_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite LoFTR global-alignment coverage audit")
    entries = _load_export_entries(Path(export_manifest))
    totals = {
        (split, bucket): _empty_totals()
        for split in EXPECTED_QUERY_COUNTS
        for bucket, _start, _stop in RANK_BUCKETS
    }
    split_queries: dict[str, set[str]] = {split: set() for split in EXPECTED_QUERY_COUNTS}
    for entry in entries:
        split, probabilities, weights, usable, model_valid = _validate_alignment_shard(
            entry=entry
        )
        if str(entry["query_id"]) in split_queries[split]:
            raise ValueError("LoFTR global-alignment export repeats a query")
        split_queries[split].add(str(entry["query_id"]))
        _accumulate_rank_buckets(
            totals=totals,
            split=split,
            probabilities=probabilities,
            weights=weights,
            usable=usable,
            model_valid=model_valid,
        )
    if {split: len(ids) for split, ids in split_queries.items()} != EXPECTED_QUERY_COUNTS:
        raise ValueError("LoFTR global-alignment query split coverage is incomplete")
    rows = _coverage_rows(totals)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "coverage.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": ARTIFACT_FORMAT,
        "stage": "audit_frozen_loftr_global_alignment_target_free_coverage",
        "target_free": True,
        "contains_target_labels": False,
        "pose_or_ground_truth_loaded": False,
        "model_fit_or_selection": False,
        "promotion_allowed": False,
        "export_manifest": str(Path(export_manifest).resolve()),
        "export_manifest_sha256": file_sha256_short(Path(export_manifest)),
        "query_counts": {split: len(ids) for split, ids in split_queries.items()},
        "row_count": int(sum(EXPECTED_QUERY_COUNTS.values()) * EXPECTED_ROWS_PER_QUERY),
        "rank_buckets": [name for name, _start, _stop in RANK_BUCKETS],
        "coverage": rows,
        "protocol": {
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "all_mapping_images_pair_cached": True,
            "global_alignment_model_per_candidate": False,
            "unknown_view_semantics": "zero_usable_mass_not_negative_evidence_v1",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_frozen_loftr_global_alignment_coverage(
        export_manifest=Path(args.export_manifest), output_dir=Path(args.output_dir)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
