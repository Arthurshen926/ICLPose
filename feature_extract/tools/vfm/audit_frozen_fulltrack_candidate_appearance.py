"""Audit frozen all-observation appearance summaries after target-free export.

The companion builder never sees query pose or SfM target identities.  This
script is the only boundary that joins registered query observations, and it
is intentionally a raw separability audit: it does not fit a calibration,
combine a feature with the mapper posterior, or score a pose hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.full_track_support_view_probe import (
    FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


ARTIFACT_FORMAT = FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-splits", default="train")
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("appearance artifact paths must be non-empty and unique")
    return paths


def _splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("audit splits must be non-empty and unique")
    if set(splits) - {"train", "validation", "test"}:
        raise ValueError("audit splits contain an unsupported name")
    return splits


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def load_frozen_fulltrack_appearance(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a target-free artifact and reject any relaxed lineage contract."""

    required = {
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_support_observation_counts",
        "source_maplet_support_view_counts",
        "candidate_summary_features",
        "candidate_summary_feature_valid",
        "feature_names",
        "profile_names",
        "candidate_profile_usable_counts",
        "candidate_profile_usable_fractions",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"full-track appearance artifact lacks {sorted(missing)}")
        arrays = {name: np.asarray(data[name]).copy() for name in required}
        metadata = _metadata(data, context="full-track appearance artifact")
    strict = metadata.get("strict_fulltrack_appearance_contract")
    if (
        metadata.get("format") != ARTIFACT_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or not isinstance(strict, Mapping)
        or strict.get("candidate_identity_fixed") is not True
        or strict.get("candidate_posterior_preserved") is not True
        or strict.get("candidate_reselection") is not False
        or strict.get("support_reselection") is not False
        or strict.get("all_real_sfm_track_observations_enumerated") is not True
        or strict.get("support_view_count_cap") is not None
        or strict.get("candidate_3d_projection_or_pose_used") is not False
        or strict.get("image_retrieval_or_submap_used") is not False
        or strict.get("render") is not False
        or strict.get("heldout_s0_verification_rows") is not True
        or strict.get("raw_summary_not_calibrated_likelihood") is not True
    ):
        raise ValueError("full-track appearance artifact violates its strict contract")
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    full_counts = np.asarray(arrays["candidate_support_observation_counts"], dtype=np.int64)
    maplet_counts = np.asarray(arrays["source_maplet_support_view_counts"], dtype=np.int64)
    values = np.asarray(arrays["candidate_summary_features"], dtype=np.float32)
    valid = np.asarray(arrays["candidate_summary_feature_valid"], dtype=bool)
    names = np.asarray(arrays["feature_names"]).astype(str).reshape(-1)
    profiles = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
    profile_counts = np.asarray(arrays["candidate_profile_usable_counts"], dtype=np.int64)
    profile_fractions = np.asarray(arrays["candidate_profile_usable_fractions"], dtype=np.float32)
    count = len(query_ids)
    if (
        count != 192
        or len(set(query_ids.tolist())) != 1
        or len(set(split_names.tolist())) != 1
        or split_names[0] not in {"train", "validation", "test"}
        or len(np.unique(source_rows)) != count
        or xy.shape != (count, 2)
        or tracks.shape != (count, 20)
        or probabilities.shape != tracks.shape
        or null.shape != (count,)
        or full_counts.shape != tracks.shape
        or maplet_counts.shape != tracks.shape
        or values.ndim != 3
        or values.shape[:2] != tracks.shape
        or valid.shape != values.shape
        or len(names) != values.shape[2]
        or len(names) == 0
        or len(set(names.tolist())) != len(names)
        or profile_counts.shape != (*tracks.shape, len(profiles))
        or profile_fractions.shape != profile_counts.shape
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(probabilities))
        or np.any(~np.isfinite(null))
        or np.any(probabilities < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(probabilities.sum(axis=1) + null - 1.0)) > 2e-5
        or np.any(full_counts < 0)
        or np.any(maplet_counts < 0)
        or np.any(profile_counts < 0)
        or np.any((profile_fractions < 0.0) | (profile_fractions > 1.0))
        or np.any(~np.isfinite(values[valid]))
        or np.any(np.isfinite(values[~valid]))
        or np.any((probabilities > 0.0) & (full_counts <= 0))
    ):
        raise ValueError("full-track appearance arrays are invalid")
    return {
        "verification_query_ids": query_ids,
        "split_names": split_names,
        "verification_source_row_indices": source_rows,
        "verification_xy": xy,
        "candidate_track_ids": tracks,
        "candidate_probabilities": probabilities,
        "null_probabilities": null,
        "candidate_support_observation_counts": full_counts,
        "source_maplet_support_view_counts": maplet_counts,
        "candidate_summary_features": values,
        "candidate_summary_feature_valid": valid,
        "feature_names": names,
        "profile_names": profiles,
        "candidate_profile_usable_counts": profile_counts,
        "candidate_profile_usable_fractions": profile_fractions,
    }, metadata


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, int], ...]:
    ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    if len(ids) != len(rows):
        raise ValueError("full-track appearance row keys are misaligned")
    return tuple((str(query_id), int(row)) for query_id, row in zip(ids, rows))


def merge_frozen_fulltrack_appearance(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    loaded = [load_frozen_fulltrack_appearance(path) for path in paths]
    reference_arrays, reference_metadata = loaded[0]
    compatibility = {
        "format": reference_metadata.get("format"),
        "version": reference_metadata.get("version"),
        "profiles": reference_metadata.get("profiles"),
        "summary_statistics": reference_metadata.get("summary_statistics"),
        "appearance_config": reference_metadata.get("appearance_config"),
        "support_geometry_index_sha256": reference_metadata.get(
            "support_geometry_index_sha256"
        ),
        "context_cache_sha256": reference_metadata.get("context_cache_sha256"),
        "implementation": reference_metadata.get("implementation"),
    }
    feature_names = reference_arrays["feature_names"]
    profile_names = reference_arrays["profile_names"]
    constant_fields = {"feature_names", "profile_names"}
    fields = tuple(field for field in reference_arrays if field not in constant_fields)
    merged: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    keys: list[tuple[str, int]] = []
    metadata: list[dict[str, Any]] = []
    for path, (arrays, item_metadata) in zip(paths, loaded):
        item_compatibility = {
            "format": item_metadata.get("format"),
            "version": item_metadata.get("version"),
            "profiles": item_metadata.get("profiles"),
            "summary_statistics": item_metadata.get("summary_statistics"),
            "appearance_config": item_metadata.get("appearance_config"),
            "support_geometry_index_sha256": item_metadata.get(
                "support_geometry_index_sha256"
            ),
            "context_cache_sha256": item_metadata.get("context_cache_sha256"),
            "implementation": item_metadata.get("implementation"),
        }
        if (
            item_compatibility != compatibility
            or not np.array_equal(arrays["feature_names"], feature_names)
            or not np.array_equal(arrays["profile_names"], profile_names)
        ):
            raise ValueError(f"{path}: full-track appearance configuration differs")
        keys.extend(_row_keys(arrays))
        for field in fields:
            merged[field].append(np.asarray(arrays[field]))
        metadata.append(item_metadata)
    if len(keys) != len(set(keys)):
        raise ValueError("full-track appearance artifacts overlap query/source-row identities")
    output = {field: np.concatenate(parts, axis=0) for field, parts in merged.items()}
    output["feature_names"] = feature_names.copy()
    output["profile_names"] = profile_names.copy()
    return output, metadata


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    targets = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("average precision inputs are invalid")
    positives = int(np.sum(targets))
    if positives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positives)


def _top_and_positive_rank(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    usable = np.asarray(valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != usable.shape:
        raise ValueError("full-track rank arrays are incompatible")
    if np.any(~np.isfinite(values[usable])):
        raise ValueError("full-track valid candidate score is non-finite")
    ranked = np.where(usable, values, -np.inf)
    order = np.argsort(-ranked, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive & usable, order, axis=1)
    has_positive = np.any(ranked_positive, axis=1)
    rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    rank[~has_positive | ~np.any(usable, axis=1)] = -1
    return order[:, 0].astype(np.int64), rank


def rank_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=bool)
    usable = np.asarray(valid, dtype=bool)
    selected = np.asarray(rows, dtype=bool).reshape(-1).copy()
    if (
        values.shape != targets.shape
        or targets.shape != usable.shape
        or selected.shape != (len(values),)
    ):
        raise ValueError("full-track rank metric inputs are incompatible")
    selected &= np.any(usable, axis=1)
    positives_present = np.any(targets & usable, axis=1)
    top, rank = _top_and_positive_rank(values, targets, usable)
    selected_positive = selected & positives_present
    flat = usable & selected[:, None]
    return {
        "row_count": int(np.sum(selected)),
        "candidate_edge_count": int(np.sum(flat)),
        "candidate_edge_positive_rate": (
            None if not np.any(flat) else float(np.mean(targets[flat]))
        ),
        "candidate_pair_average_precision": (
            None if not np.any(flat) else _average_precision(targets[flat], values[flat])
        ),
        "positive_row_count": int(np.sum(selected_positive)),
        "positive_row_rate": (
            None if not np.any(selected) else float(np.mean(positives_present[selected]))
        ),
        "top1_positive_rate_given_positive": (
            None
            if not np.any(selected_positive)
            else float(np.mean(targets[selected_positive, top[selected_positive]]))
        ),
        "median_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.median(rank[selected_positive]))
        ),
        "p90_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.quantile(rank[selected_positive], 0.9))
        ),
    }


def paired_rank(
    *,
    baseline_scores: np.ndarray,
    probe_scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    probe = np.asarray(probe_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=bool)
    usable = np.asarray(valid, dtype=bool)
    selected = np.asarray(rows, dtype=bool).reshape(-1).copy()
    if (
        baseline.shape != probe.shape
        or probe.shape != targets.shape
        or targets.shape != usable.shape
        or selected.shape != (len(baseline),)
    ):
        raise ValueError("full-track paired rank inputs are incompatible")
    selected &= np.any(targets & usable, axis=1)
    if not np.any(selected):
        return {
            "positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
        }
    _top_base, base_rank = _top_and_positive_rank(baseline, targets, usable)
    _top_probe, probe_rank = _top_and_positive_rank(probe, targets, usable)
    base = base_rank[selected]
    updated = probe_rank[selected]
    return {
        "positive_row_count": int(len(base)),
        "rank_win_count": int(np.sum(updated < base)),
        "rank_loss_count": int(np.sum(updated > base)),
        "rank_tie_count": int(np.sum(updated == base)),
        "top1_rescue_count": int(np.sum((base > 1) & (updated == 1))),
        "top1_harm_count": int(np.sum((base == 1) & (updated > 1))),
        "median_rank_delta_baseline_minus_probe": float(np.median(base - updated)),
    }


def _raw_score_distribution(
    *, scores: np.ndarray, labels: np.ndarray, valid: np.ndarray, rows: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=bool)
    usable = np.asarray(valid, dtype=bool)
    selected = np.asarray(rows, dtype=bool).reshape(-1)
    mask = usable & selected[:, None]
    correct = values[mask & targets]
    incorrect = values[mask & ~targets]
    return {
        "correct_count": int(len(correct)),
        "incorrect_count": int(len(incorrect)),
        "correct_median": None if not len(correct) else float(np.median(correct)),
        "incorrect_median": None if not len(incorrect) else float(np.median(incorrect)),
        "correct_p10": None if not len(correct) else float(np.quantile(correct, 0.1)),
        "incorrect_p90": None if not len(incorrect) else float(np.quantile(incorrect, 0.9)),
    }


def _train_screen(
    *, baseline: Mapping[str, Any], probe: Mapping[str, Any], paired: Mapping[str, Any]
) -> dict[str, Any]:
    """A conservative signal screen, explicitly not a calibration/promotion gate."""

    required = (
        baseline.get("candidate_pair_average_precision"),
        probe.get("candidate_pair_average_precision"),
        baseline.get("top1_positive_rate_given_positive"),
        probe.get("top1_positive_rate_given_positive"),
        baseline.get("p90_first_positive_rank"),
        probe.get("p90_first_positive_rank"),
    )
    comparable = all(value is not None for value in required)
    checks = {
        "comparable_train_rows": comparable,
        "raw_ap_strictly_improved": bool(
            comparable
            and float(probe["candidate_pair_average_precision"])
            > float(baseline["candidate_pair_average_precision"])
        ),
        "raw_top1_not_worse": bool(
            comparable
            and float(probe["top1_positive_rate_given_positive"])
            >= float(baseline["top1_positive_rate_given_positive"])
        ),
        "raw_p90_rank_not_worse": bool(
            comparable
            and float(probe["p90_first_positive_rank"])
            <= float(baseline["p90_first_positive_rank"])
        ),
        "paired_wins_exceed_losses": int(paired["rank_win_count"])
        > int(paired["rank_loss_count"]),
        "top1_rescues_exceed_harms": int(paired["top1_rescue_count"])
        > int(paired["top1_harm_count"]),
    }
    return {
        "policy": (
            "train-only raw separability screen; a pass only permits a separately "
            "frozen validation export and never calibration, fusion, or pose scoring"
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def audit_frozen_fulltrack_candidate_appearance(
    *,
    appearance_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    output_dir: Path,
    audit_splits: Sequence[str],
    registered_identity_radius_px: float,
) -> dict[str, Any]:
    """Join post-hoc registered identities to frozen raw full-track summaries."""

    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite full-track appearance audit output")
    arrays, metadata = merge_frozen_fulltrack_appearance(
        tuple(Path(path) for path in appearance_artifacts)
    )
    selected_splits = tuple(str(value) for value in audit_splits)
    split_names = np.asarray(arrays["split_names"]).astype(str)
    selected = np.isin(split_names, selected_splits)
    if not np.any(selected):
        raise ValueError("full-track appearance audit has no requested split rows")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=arrays["verification_query_ids"],
        query_xy=arrays["verification_xy"],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(arrays["candidate_track_ids"], targets)
    registered_rows = selected & np.asarray(targets.supervised, dtype=bool)
    retrieved_identity_rows = registered_rows & np.any(labels, axis=1)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    features = np.asarray(arrays["candidate_summary_features"], dtype=np.float32)
    feature_valid = np.asarray(arrays["candidate_summary_feature_valid"], dtype=bool)
    result: dict[str, Any] = {
        "stage": "audit_frozen_fulltrack_candidate_appearance",
        "appearance_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in appearance_artifacts
        ],
        "audit_splits": list(selected_splits),
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "protocol": {
            "feature_export_target_free": True,
            "targets_joined_only_after_frozen_export": True,
            "calibration_or_fusion_fitted": False,
            "pose_scoring_performed": False,
            "test_used_for_model_selection": False,
        },
        "target_coverage": summarize_registered_candidate_identity(labels, targets),
        "evaluation_scope": {
            "requested_split_row_count": int(np.sum(selected)),
            "registered_identity_row_count": int(np.sum(registered_rows)),
            "retrieved_identity_row_count": int(np.sum(retrieved_identity_rows)),
            "candidate_ranking_gate_rows": "registered_identity_and_correct_fixed_topl_track",
            "unsupervised_rows_excluded": True,
            "fixed_topl_misses_reported_in_target_coverage": True,
        },
        "support_expansion": {
            "full_support_observation_count": {
                "median": float(np.median(arrays["candidate_support_observation_counts"])),
                "p90": float(np.quantile(arrays["candidate_support_observation_counts"], 0.9)),
                "max": int(np.max(arrays["candidate_support_observation_counts"])),
            },
            "source_maplet_support_view_count": {
                "median": float(np.median(arrays["source_maplet_support_view_counts"])),
                "p90": float(np.quantile(arrays["source_maplet_support_view_counts"], 0.9)),
                "max": int(np.max(arrays["source_maplet_support_view_counts"])),
            },
            "candidate_entry_expanded_rate": float(
                np.mean(
                    arrays["candidate_support_observation_counts"]
                    > arrays["source_maplet_support_view_counts"]
                )
            ),
        },
        "features": {},
        "source_metadata": {
            "support_geometry_index_sha256": metadata[0].get("support_geometry_index_sha256"),
            "context_cache_sha256": metadata[0].get("context_cache_sha256"),
            "profiles": metadata[0].get("profiles"),
            "appearance_config": metadata[0].get("appearance_config"),
        },
    }
    rows_for_csv: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(
        np.asarray(arrays["feature_names"]).astype(str).tolist()
    ):
        valid = (probabilities > 0.0) & feature_valid[..., feature_index]
        scores = features[..., feature_index]
        baseline = rank_metrics(
                scores=probabilities,
                labels=labels,
                valid=valid,
                rows=retrieved_identity_rows,
        )
        probe = rank_metrics(
                scores=scores,
                labels=labels,
                valid=valid,
                rows=retrieved_identity_rows,
        )
        paired = paired_rank(
            baseline_scores=probabilities,
                probe_scores=scores,
                labels=labels,
                valid=valid,
                rows=retrieved_identity_rows,
        )
        payload = {
            "coverage": {
                "candidate_usable_rate": float(
                    np.mean(valid[retrieved_identity_rows])
                ),
                "positive_row_usable_rate": (
                    None
                    if not np.any(retrieved_identity_rows)
                    else float(
                        np.mean(
                            np.any(valid, axis=1)[
                                retrieved_identity_rows
                            ]
                        )
                    )
                ),
            },
            "baseline_common_coverage": baseline,
            "raw_fulltrack_feature": probe,
            "paired_rank": paired,
            "raw_score_distribution": _raw_score_distribution(
                scores=scores,
                labels=labels,
                valid=valid,
                rows=retrieved_identity_rows,
            ),
            "train_only_raw_screen": _train_screen(
                baseline=baseline, probe=probe, paired=paired
            )
            if selected_splits == ("train",)
            else None,
        }
        result["features"][str(feature_name)] = payload
        rows_for_csv.append(
            {
                "feature": str(feature_name),
                "candidate_usable_rate": payload["coverage"]["candidate_usable_rate"],
                "baseline_ap": baseline["candidate_pair_average_precision"],
                "raw_ap": probe["candidate_pair_average_precision"],
                "baseline_top1": baseline["top1_positive_rate_given_positive"],
                "raw_top1": probe["top1_positive_rate_given_positive"],
                "baseline_median_rank": baseline["median_first_positive_rank"],
                "raw_median_rank": probe["median_first_positive_rank"],
                "baseline_p90_rank": baseline["p90_first_positive_rank"],
                "raw_p90_rank": probe["p90_first_positive_rank"],
                "rank_wins": paired["rank_win_count"],
                "rank_losses": paired["rank_loss_count"],
                "top1_rescues": paired["top1_rescue_count"],
                "top1_harms": paired["top1_harm_count"],
                "train_screen_pass": (
                    None
                    if payload["train_only_raw_screen"] is None
                    else payload["train_only_raw_screen"]["passed"]
                ),
            }
        )
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "feature_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_for_csv[0]))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_candidate_appearance(
        appearance_artifacts=_paths(args.appearance_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        audit_splits=_splits(args.audit_splits),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
