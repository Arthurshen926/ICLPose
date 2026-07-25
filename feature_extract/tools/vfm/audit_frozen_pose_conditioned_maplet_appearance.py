"""Target-side audit for frozen candidate-specific 3-D maplet appearance P1.

The scorer deliberately has no pose-target input.  This program is the only
place where its fixed hypothesis rows are joined with pose errors.  It audits
raw visual families and, when supplied, a matched descriptor-permutation
control.  Neither path can train a weight, change a candidate, or promote a
pose; a train-only calibration stage remains required after this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    TARGET_FORMAT,
    _baseline_rows,
    _groups,
    _load_targets,
    _paired,
    _per_query_rows,
    _row_keys,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
    parse_alpha_grid,
    rank_percentiles,
)
from feature_extract.vfm.artifacts import file_sha256_short


SCORE_FORMAT = "frozen_pose_conditioned_maplet_appearance_scores_v1"
AUDIT_FORMAT = "frozen_pose_conditioned_maplet_appearance_target_audit_v1"
_SCORE_STATISTIC_FIELDS = {
    "mean": "family_log_likelihood_means",
    "median": "family_log_likelihood_medians",
    "worst_quartile_mean": "family_log_likelihood_worst_quartile_means",
    "spatial_median_of_means_2x2": "family_spatial_median_of_means_2x2",
}
_FROZEN_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "source_chosen_for_optional_pose",
    "baseline_score_top1",
    "baseline_selection_scores",
    "verification_source_row_indices",
    "verification_xy",
    "fit_query_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "support_view_probabilities",
    "support_image_ids",
)
_STATIC_INPUT_KEYS = (
    "detector_query_cache",
    "proposals",
    "candidate_artifact",
    "fixed_candidate_prior_overlay",
    "fixed_candidate_support_view_overlay",
    "maplet_support_index",
    "support_geometry_index",
    "projected_landmark_bank",
    "neighbor_topology_cache",
    "radio_final_context_cache",
    "radio_intermediate_context_cache",
    "alike_spatial_context_cache",
    "colmap_cameras_bin",
    "colmap_images_bin_camera_ownership_only",
)
_LAYOUT_KEYS = (
    "neighbor_track_ids_sha256",
    "neighbor_xyz_sha256",
    "neighbor_support_xy_sha256",
    "neighbor_valid_sha256",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-artifacts",
        required=True,
        help="comma-separated target-free visual P1 score shards",
    )
    parser.add_argument(
        "--control-score-artifacts",
        default="",
        help="optional comma-separated matched target-free descriptor-control shards",
    )
    parser.add_argument("--target-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--score-statistic",
        choices=tuple(_SCORE_STATISTIC_FIELDS),
        default="mean",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0,0.25,0.5,1",
        help="fixed exploratory non-negative rank-percentile fusion alphas",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str, *, name: str, required: bool) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if required and not paths:
        raise ValueError(f"{name} must be non-empty")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name} contains duplicate paths")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _input_hashes(metadata: Mapping[str, object], keys: Sequence[str]) -> dict[str, str]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("P1 score metadata lacks an input manifest")
    output: dict[str, str] = {}
    for key in keys:
        item = inputs.get(key)
        if not isinstance(item, Mapping) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"P1 score input manifest lacks {key} hash")
        output[key] = str(item["sha256"])
    return output


def _score_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    raw_maplet_config = metadata.get("maplet_config")
    if not isinstance(raw_maplet_config, Mapping):
        raise ValueError("P1 score metadata lacks maplet configuration")
    # The fraction is a recorded per-query consequence of the fixed fit rows
    # and radius, not a scorer hyperparameter.  Keeping it in the cross-shard
    # contract would reject a valid frozen run solely because images have
    # different PnP-fit spatial coverage.
    maplet_config = dict(raw_maplet_config)
    fit_fraction = maplet_config.pop("fit_exclusion_mask_fraction", None)
    if fit_fraction is None or not np.isfinite(float(fit_fraction)) or not 0.0 <= float(fit_fraction) <= 1.0:
        raise ValueError("P1 score fit-exclusion fraction is invalid")
    return {
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "strict_frozen_maplet_appearance_contract": metadata.get(
            "strict_frozen_maplet_appearance_contract"
        ),
        "maplet_config": maplet_config,
        "input_metadata": metadata.get("input_metadata"),
        "static_input_hashes": _input_hashes(metadata, _STATIC_INPUT_KEYS),
    }


def _visual_control_pair_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable portion of a visual/control score contract.

    The two control booleans are intentionally the only differing contract
    fields: a descriptor control must declare that it is a control.  Every
    actual input, topology, feature configuration, and scoring protocol still
    has to match exactly.
    """

    output = _score_contract(metadata)
    strict = output.get("strict_frozen_maplet_appearance_contract")
    if not isinstance(strict, Mapping):
        raise ValueError("P1 score lacks a strict frozen maplet contract")
    normalized = dict(strict)
    normalized.pop("support_descriptor_permutation_control", None)
    normalized.pop("xyz_permutation_control", None)
    output["strict_frozen_maplet_appearance_contract"] = normalized
    return output


def _assert_strict_contract(metadata: Mapping[str, object], *, expected_variant: str) -> None:
    strict = metadata.get("strict_frozen_maplet_appearance_contract")
    required = {
        "heldout_query_rows": True,
        "heldout_query_image_content_excludes_pnp_fit_neighborhoods": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "candidate_support_view_posterior_fixed_before_pose_scoring": True,
        "candidate_group_latent_identity_marginalized": True,
        "candidate_group_topl_denominator_fixed": True,
        "candidate_group_explicit_null": True,
        "support_maplet_center_excluded": True,
        "support_maplet_topology_fixed": True,
        "maplet_neighbor_identity_fixed": True,
        "pose_dependent_correspondence_selection": False,
        "fit_neighborhood_missing_evidence_penalized": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "raw_scores_calibrated_or_promoted": False,
        "raw_scores_must_not_feed_pnp": True,
    }
    if not isinstance(strict, Mapping) or any(strict.get(key) != value for key, value in required.items()):
        raise ValueError("P1 score does not satisfy the strict frozen maplet contract")
    is_descriptor_control = expected_variant == "support_descriptor_permutation_control"
    is_xyz_control = expected_variant == "xyz_permutation_control"
    if (
        strict.get("support_descriptor_permutation_control") is not is_descriptor_control
        or strict.get("xyz_permutation_control") is not is_xyz_control
    ):
        raise ValueError("P1 score control flags disagree with its declared evidence variant")


def _load_score(path: Path, *, expected_variant: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        *_FROZEN_ROW_FIELDS,
        "family_names",
        *tuple(_SCORE_STATISTIC_FIELDS.values()),
        "family_effective_point_counts",
        "family_effective_view_masses",
        "family_projection_visible_fractions",
        "family_fit_masked_visible_fractions",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: incomplete P1 score artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != SCORE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("evidence_variant") != expected_variant
    ):
        raise ValueError(f"{path}: not the required target-free P1 evidence variant")
    runtime = metadata.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("all_frozen_hypotheses") is not True
        or int(runtime.get("hypothesis_limit", -1)) != 0
    ):
        raise ValueError(f"{path}: formal P1 audit rejects a hypothesis-limited score shard")
    _assert_strict_contract(metadata, expected_variant=expected_variant)
    count = int(metadata.get("row_count", -1))
    row_fields = tuple(field for field in _FROZEN_ROW_FIELDS if field not in {
        "verification_source_row_indices", "verification_xy", "fit_query_xy", "candidate_track_ids",
        "candidate_probabilities", "null_probabilities", "support_view_probabilities", "support_image_ids",
    })
    if count <= 0 or any(np.asarray(arrays[field]).shape != (count,) for field in row_fields):
        raise ValueError(f"{path}: P1 row-aligned score fields are inconsistent")
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    if int(metadata.get("query_count", -1)) != 1 or len(set(query_ids.tolist())) != 1:
        raise ValueError(f"{path}: a P1 shard must contain exactly one query")
    if str(metadata.get("query_id")) != str(query_ids[0]):
        raise ValueError(f"{path}: P1 metadata/query row ownership differs")
    family_names = np.asarray(arrays["family_names"]).astype(str).reshape(-1)
    family_count = len(family_names)
    if family_count == 0 or len(set(family_names.tolist())) != family_count:
        raise ValueError(f"{path}: P1 family names are invalid")
    for field in (
        *tuple(_SCORE_STATISTIC_FIELDS.values()),
        "family_effective_point_counts",
        "family_effective_view_masses",
        "family_projection_visible_fractions",
        "family_fit_masked_visible_fractions",
    ):
        values = np.asarray(arrays[field])
        if values.shape != (count, family_count) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {field} is non-finite or family-misaligned")
    if (
        np.asarray(arrays["verification_xy"]).ndim != 2
        or np.asarray(arrays["verification_xy"]).shape[1:] != (2,)
        or np.asarray(arrays["candidate_track_ids"]).shape[0] != len(arrays["verification_xy"])
        or np.asarray(arrays["candidate_track_ids"]).shape[1:] != (20,)
        or np.asarray(arrays["support_view_probabilities"]).shape[:2]
        != np.asarray(arrays["candidate_track_ids"]).shape
    ):
        raise ValueError(f"{path}: P1 fixed top-20 evidence layout is malformed")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: P1 shard repeats query/hypothesis rows")
    _input_hashes(metadata, (*_STATIC_INPUT_KEYS, "hypothesis_artifact", "baseline_score_artifact"))
    return arrays, metadata


def _merge_scores(
    paths: Sequence[Path], *, expected_variant: str
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], tuple[tuple[str, str, str, int], ...]]:
    loaded = [_load_score(path, expected_variant=expected_variant) for path in paths]
    if not loaded:
        raise ValueError("P1 score artifact list is empty")
    family_names = np.asarray(loaded[0][0]["family_names"]).astype(str)
    fingerprint = _canonical_hash(_score_contract(loaded[0][1]))
    rows: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    for arrays, item_metadata in loaded:
        if not np.array_equal(np.asarray(arrays["family_names"]).astype(str), family_names):
            raise ValueError("P1 score shards declare different family orders")
        if _canonical_hash(_score_contract(item_metadata)) != fingerprint:
            raise ValueError("P1 score shards have incompatible frozen maplet contracts")
        rows.append(arrays)
        metadata.append(item_metadata)
    query_ids = [str(item.get("query_id")) for item in metadata]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("P1 score artifacts repeat a query shard")
    merged = {
        field: (
            family_names.copy()
            if field == "family_names"
            else np.concatenate([np.asarray(row[field]) for row in rows], axis=0)
        )
        for field in rows[0]
        if field not in {
            "verification_source_row_indices", "verification_xy", "fit_query_xy", "candidate_track_ids",
            "candidate_probabilities", "null_probabilities", "support_view_probabilities", "support_image_ids",
        }
    }
    keys = _row_keys(merged)
    if len(keys) != len(set(keys)):
        raise ValueError("merged P1 score shards repeat query/hypothesis rows")
    return merged, metadata, keys


def _profile_layout_without_derangement(metadata: Mapping[str, object]) -> dict[str, dict[str, object]]:
    layouts = metadata.get("profile_static_layout")
    if not isinstance(layouts, Mapping) or not layouts:
        raise ValueError("P1 score metadata lacks static profile layouts")
    output: dict[str, dict[str, object]] = {}
    for profile, raw in layouts.items():
        if not isinstance(raw, Mapping) or any(key not in raw for key in _LAYOUT_KEYS):
            raise ValueError("P1 profile layout lacks a static neighbor digest")
        output[str(profile)] = {
            key: raw.get(key)
            for key in (*_LAYOUT_KEYS, "profile", "static_neighbor_count", "static_candidate_view_with_neighbor_count")
        }
    return output


def _assert_visual_control_pair(
    *,
    visual_arrays: Mapping[str, np.ndarray],
    visual_metadata: Mapping[str, object],
    control_arrays: Mapping[str, np.ndarray],
    control_metadata: Mapping[str, object],
) -> None:
    if str(visual_metadata.get("query_id")) != str(control_metadata.get("query_id")):
        raise ValueError("visual/control P1 shards belong to different queries")
    if _canonical_hash(_visual_control_pair_contract(visual_metadata)) != _canonical_hash(
        _visual_control_pair_contract(control_metadata)
    ):
        raise ValueError("visual/control P1 shards have different frozen contracts")
    if visual_metadata.get("evidence_layout_digest") != control_metadata.get("evidence_layout_digest"):
        raise ValueError("visual/control P1 shards use different held-out evidence layouts")
    if _profile_layout_without_derangement(visual_metadata) != _profile_layout_without_derangement(control_metadata):
        raise ValueError("visual/control P1 shards have different static maplet neighbourhoods")
    visual_layout = visual_metadata.get("profile_static_layout")
    control_layout = control_metadata.get("profile_static_layout")
    assert isinstance(visual_layout, Mapping) and isinstance(control_layout, Mapping)
    for profile in visual_layout:
        if visual_layout[profile].get("support_descriptor_derangement") is not None:
            raise ValueError("visual P1 score unexpectedly declares a descriptor derangement")
        if not control_layout[profile].get("support_descriptor_derangement"):
            raise ValueError("descriptor control lacks its deterministic derangement digest")
    for field in _FROZEN_ROW_FIELDS:
        if not np.array_equal(np.asarray(visual_arrays[field]), np.asarray(control_arrays[field])):
            raise ValueError(f"visual/control P1 frozen field differs: {field}")
    if not np.array_equal(visual_arrays["family_names"], control_arrays["family_names"]):
        raise ValueError("visual/control P1 family names differ")
    if not any(
        np.any(np.abs(np.asarray(visual_arrays[field]) - np.asarray(control_arrays[field])) > 1e-9)
        for field in _SCORE_STATISTIC_FIELDS.values()
    ):
        raise ValueError("descriptor permutation control did not change any P1 score")


def _validate_control_collection(
    *,
    visual_paths: Sequence[Path],
    visual_metadata: Sequence[dict[str, object]],
    control_paths: Sequence[Path],
    control_metadata: Sequence[dict[str, object]],
) -> None:
    if len(visual_paths) != len(control_paths):
        raise ValueError("visual/control P1 collections have different shard counts")
    visual = {str(item["query_id"]): (path, item) for path, item in zip(visual_paths, visual_metadata)}
    control = {str(item["query_id"]): (path, item) for path, item in zip(control_paths, control_metadata)}
    if set(visual) != set(control):
        raise ValueError("visual/control P1 collections do not cover the same queries")


def _target_rows_for_keys(
    *, targets: Mapping[str, np.ndarray], keys: Sequence[tuple[str, str, str, int]]
) -> np.ndarray:
    positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        return np.asarray([positions[key] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("a frozen P1 score row is absent from the target artifact") from error


def _validate_target_lineage(
    *, metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    hypothesis_hashes = set(str(value) for value in target_metadata.get("hypothesis_artifact_sha256", []))
    baseline_hashes = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    if not hypothesis_hashes or not baseline_hashes:
        raise ValueError("target artifact lacks frozen S0 lineage hashes")
    for item in metadata:
        hashes = _input_hashes(item, ("hypothesis_artifact", "baseline_score_artifact"))
        if hashes["hypothesis_artifact"] not in hypothesis_hashes or hashes["baseline_score_artifact"] not in baseline_hashes:
            raise ValueError("P1 score/target lineage differs from the frozen S0 source")


def _with_variant(rows: Sequence[Mapping[str, object]], variant: str) -> list[dict[str, object]]:
    return [{**dict(row), "evidence_variant": str(variant)} for row in rows]


def _audit_variant(
    *,
    scores: Mapping[str, np.ndarray],
    keys: Sequence[tuple[str, str, str, int]],
    baseline: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    tie_orders: np.ndarray,
    score_statistic: str,
    alphas: Sequence[float],
    evidence_variant: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, list[dict[str, object]]]]:
    values = np.asarray(scores[_SCORE_STATISTIC_FIELDS[score_statistic]], dtype=np.float64)
    names = np.asarray(scores["family_names"]).astype(str)
    all_rows: list[dict[str, object]] = []
    report: dict[str, object] = {}
    standalone_by_family: dict[str, list[dict[str, object]]] = {}
    for index, family in enumerate(names.tolist()):
        local = values[:, index]
        standalone = _per_query_rows(
            keys=keys,
            scores=local,
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=family,
            score_mode=f"standalone_{score_statistic}",
            alpha=None,
        )
        standalone = _with_variant(standalone, evidence_variant)
        standalone_by_family[family] = standalone
        all_rows.extend(standalone)
        static = {
            "mean_effective_point_count": float(np.mean(np.asarray(scores["family_effective_point_counts"])[:, index])),
            "mean_effective_view_mass": float(np.mean(np.asarray(scores["family_effective_view_masses"])[:, index])),
            "mean_projection_visible_fraction": float(np.mean(np.asarray(scores["family_projection_visible_fractions"])[:, index])),
            "mean_fit_masked_visible_fraction": float(np.mean(np.asarray(scores["family_fit_masked_visible_fractions"])[:, index])),
        }
        family_report: dict[str, object] = {
            "standalone": {
                "splits": _split_summary(standalone),
                "paired_vs_s0": _paired(_with_variant(_baseline_rows(
                    keys=keys,
                    baseline_scores=baseline,
                    translation_m=translation,
                    rotation_deg=rotation,
                    tie_break_orders=tie_orders,
                ), evidence_variant), standalone),
            },
            "rank_percentile_fusion": {},
            "static_evidence": static,
        }
        for alpha in alphas:
            fused = np.empty_like(baseline)
            for _group, rows in _groups(keys).items():
                fused[rows] = rank_percentiles(baseline[rows], tie_orders[rows]) + float(alpha) * rank_percentiles(
                    local[rows], tie_orders[rows]
                )
            fused_rows = _with_variant(_per_query_rows(
                keys=keys,
                scores=fused,
                baseline_scores=baseline,
                translation_m=translation,
                rotation_deg=rotation,
                tie_break_orders=tie_orders,
                family=family,
                score_mode="rank_percentile_fusion",
                alpha=float(alpha),
            ), evidence_variant)
            all_rows.extend(fused_rows)
            family_report["rank_percentile_fusion"][str(alpha)] = {
                "splits": _split_summary(fused_rows),
                "paired_vs_s0": _paired(_with_variant(_baseline_rows(
                    keys=keys,
                    baseline_scores=baseline,
                    translation_m=translation,
                    rotation_deg=rotation,
                    tie_break_orders=tie_orders,
                ), evidence_variant), fused_rows),
                "exploratory_only": True,
            }
        report[family] = family_report
    return all_rows, report, standalone_by_family


def _visual_vs_control_rank_report(
    visual: Mapping[str, Sequence[Mapping[str, object]]], control: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[str, object]:
    if set(visual) != set(control):
        raise ValueError("visual/control P1 family reports differ")
    report: dict[str, object] = {}
    for family in sorted(visual):
        by_key = {
            (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"])): row
            for row in visual[family]
        }
        control_by_key = {
            (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"])): row
            for row in control[family]
        }
        if not by_key or set(by_key) != set(control_by_key):
            raise ValueError("visual/control P1 per-query rank coverage differs")
        deltas = np.asarray(
            [
                int(by_key[key]["oracle_score_rank"]) - int(control_by_key[key]["oracle_score_rank"])
                for key in sorted(by_key)
            ],
            dtype=np.int64,
        )
        report[family] = {
            "visual_lower_oracle_rank_count": int(np.count_nonzero(deltas < 0)),
            "control_lower_oracle_rank_count": int(np.count_nonzero(deltas > 0)),
            "equal_oracle_rank_count": int(np.count_nonzero(deltas == 0)),
            "median_visual_minus_control_oracle_rank": float(np.median(deltas)),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    visual_paths = _paths(args.score_artifacts, name="score artifacts", required=True)
    control_paths = _paths(args.control_score_artifacts, name="control score artifacts", required=False)
    alphas = parse_alpha_grid(args.alpha_grid)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing P1 audit: {summary_path}")
    visual, visual_metadata, keys = _merge_scores(visual_paths, expected_variant="visual")
    controls: dict[str, np.ndarray] | None = None
    control_metadata: list[dict[str, object]] = []
    if control_paths:
        controls, control_metadata, control_keys = _merge_scores(
            control_paths, expected_variant="support_descriptor_permutation_control"
        )
        if keys != control_keys:
            raise ValueError("visual/control P1 rows are not identical")
        _validate_control_collection(
            visual_paths=visual_paths,
            visual_metadata=visual_metadata,
            control_paths=control_paths,
            control_metadata=control_metadata,
        )
        visual_by_query = {str(item["query_id"]): (path, item) for path, item in zip(visual_paths, visual_metadata)}
        control_by_query = {str(item["query_id"]): (path, item) for path, item in zip(control_paths, control_metadata)}
        for query_id in sorted(visual_by_query):
            visual_arrays, visual_item = _load_score(visual_by_query[query_id][0], expected_variant="visual")
            control_arrays, control_item = _load_score(
                control_by_query[query_id][0], expected_variant="support_descriptor_permutation_control"
            )
            _assert_visual_control_pair(
                visual_arrays=visual_arrays,
                visual_metadata=visual_item,
                control_arrays=control_arrays,
                control_metadata=control_item,
            )
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    if not isinstance(target_metadata, Mapping) or target_metadata.get("format") != TARGET_FORMAT:
        raise ValueError("target artifact format is invalid")
    _validate_target_lineage(metadata=visual_metadata, target_metadata=target_metadata)
    if control_metadata:
        _validate_target_lineage(metadata=control_metadata, target_metadata=target_metadata)
    target_rows = _target_rows_for_keys(targets=targets, keys=keys)
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline = np.asarray(visual["baseline_selection_scores"], dtype=np.float64)
    source_top1 = np.asarray(visual["baseline_score_top1"], dtype=bool)
    if not np.isfinite(baseline).all() or translation.shape != baseline.shape or rotation.shape != baseline.shape:
        raise ValueError("frozen P1 score/target arrays are misaligned")
    tie_orders = np.arange(len(keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(source_top1=source_top1, keys=keys, baseline_rows=baseline_rows)
    baseline_rows = _with_variant(baseline_rows, "immutable_s0_baseline")
    visual_rows, visual_families, visual_standalone = _audit_variant(
        scores=visual,
        keys=keys,
        baseline=baseline,
        translation=translation,
        rotation=rotation,
        tie_orders=tie_orders,
        score_statistic=str(args.score_statistic),
        alphas=alphas,
        evidence_variant="visual",
    )
    all_rows: list[dict[str, object]] = [*baseline_rows, *visual_rows]
    control_families: dict[str, object] | None = None
    control_comparison: dict[str, object] | None = None
    if controls is not None:
        if not np.array_equal(baseline, np.asarray(controls["baseline_selection_scores"], dtype=np.float64)):
            raise ValueError("visual/control P1 baseline scores differ")
        control_rows, control_families, control_standalone = _audit_variant(
            scores=controls,
            keys=keys,
            baseline=baseline,
            translation=translation,
            rotation=rotation,
            tie_orders=tie_orders,
            score_statistic=str(args.score_statistic),
            alphas=alphas,
            evidence_variant="support_descriptor_permutation_control",
        )
        all_rows.extend(control_rows)
        control_comparison = _visual_vs_control_rank_report(visual_standalone, control_standalone)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary: dict[str, Any] = {
        "stage": "frozen_pose_conditioned_maplet_appearance_target_audit",
        "format": AUDIT_FORMAT,
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "target_free_p1_scores_validated": True,
            "all_frozen_hypotheses_required": True,
            "fixed_global_topl": True,
            "fixed_candidate_support_view_mixture": True,
            "center_excluded_sfm_maplet": True,
            "pnp_fit_neighborhood_excluded": True,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "rank_percentile_fusion_is_exploratory_only": True,
            "promotion_allowed": False,
        },
        "score_statistic": str(args.score_statistic),
        "alpha_grid": list(alphas),
        "baseline": {"splits": _split_summary(baseline_rows)},
        "visual_families": visual_families,
        "descriptor_permutation_control": None
        if control_families is None
        else {
            "families": control_families,
            "visual_vs_control_oracle_rank": control_comparison,
        },
        "inputs": {
            "score_artifacts": [str(path) for path in visual_paths],
            "score_artifact_sha256": [file_sha256_short(path) for path in visual_paths],
            "control_score_artifacts": [str(path) for path in control_paths],
            "control_score_artifact_sha256": [file_sha256_short(path) for path in control_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_hash": _canonical_hash(_score_contract(visual_metadata[0])),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
