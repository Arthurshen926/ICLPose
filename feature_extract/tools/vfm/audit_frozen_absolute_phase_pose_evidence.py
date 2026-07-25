"""Target-side paired audit for frozen multiscale absolute-phase P1 scores.

Only this program reads pose targets.  It compares real feature evidence to a
fixed support-channel-permutation control under identical hypotheses,
candidates, support views, held-out points, and null denominators.  Passing a
gate authorizes only a later train-only LLR fit; it never promotes raw P1
scores into PnP.
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
    _paired,
    _per_query_rows,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
)
from feature_extract.tools.vfm.audit_v5_dynamic_absolute_context_pose_evidence import (
    _load_targets,
    _row_keys,
)
from feature_extract.tools.vfm.score_frozen_absolute_phase_pose_evidence import (
    CONTEXT_CORRELATION_SCORE_FORMAT,
    FAMILY_COMPONENTS,
    FORMAL_POINT_COUNT,
    POINT_SIDECAR_FORMAT,
    PROFILE_SET_CONFIGS,
    PROFILE_SET_PHASE_V1,
    SCORE_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.mixed_verification_points import (
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
)


_EPSILON = 1e-12
_STATISTIC_FIELDS = {
    "mean": "family_log_likelihood_means",
    "median": "family_log_likelihood_medians",
    "worst_quartile_mean": "family_log_likelihood_worst_quartile_means",
    "spatial_median_of_means_2x2": "family_spatial_median_of_means_2x2",
}
_FORMAL_POINT_COUNTS = {
    POINT_SOURCE_ALIKE: 64,
    POINT_SOURCE_RADIO_INTERMEDIATE: 64,
    POINT_SOURCE_RADIO_FINAL: 64,
}


def _declared_profile_set(
    *, metadata: Mapping[str, object], family_names: np.ndarray
) -> tuple[str, list[str]]:
    """Resolve one immutable scorer profile set from artifact metadata."""

    profile_set = str(metadata.get("profile_set", PROFILE_SET_PHASE_V1))
    config = PROFILE_SET_CONFIGS.get(profile_set)
    if config is None:
        raise ValueError("absolute-context score declares an unknown profile set")
    components = tuple(config["family_components"])
    expected_names = [str(name) for name, _components in components]
    if family_names.astype(str).tolist() != expected_names:
        raise ValueError("absolute-context score family names differ from its immutable profile set")
    expected_format = str(config["score_format"])
    if str(metadata.get("format", "")) != expected_format:
        raise ValueError("absolute-context score format differs from its immutable profile set")
    return profile_set, expected_names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual_score_artifacts", required=True)
    parser.add_argument("--control_score_artifacts", required=True)
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_statistic",
        choices=tuple(_STATISTIC_FIELDS),
        default="spatial_median_of_means_2x2",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("absolute-phase audit paths must be non-empty and unique")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _load_sidecar(
    path: Path,
    *,
    expected_evidence_sha: str,
    expected_variant: str,
    expected_names: Sequence[str],
    expected_row_count: int,
) -> None:
    fields = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "family_names",
        "point_log_ratios",
        "point_active",
        "point_effective_view_masses",
        "point_zero_shift_coverages",
        "verification_xy",
        "verification_point_sources",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = fields - set(payload.files)
        if missing:
            raise ValueError(f"{path}: absolute-phase point sidecar is incomplete ({sorted(missing)})")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        row_count = len(np.asarray(payload["query_ids"]))
        family_names = np.asarray(payload["family_names"]).astype(str)
        point_logs = np.asarray(payload["point_log_ratios"])
        point_active = np.asarray(payload["point_active"])
        point_mass = np.asarray(payload["point_effective_view_masses"])
        point_coverage = np.asarray(payload["point_zero_shift_coverages"])
        point_sources = np.asarray(payload["verification_point_sources"]).astype(str)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != POINT_SIDECAR_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or str(metadata.get("frozen_query_evidence_sha256", "")) != str(expected_evidence_sha)
        or str(metadata.get("evidence_variant", "")) != str(expected_variant)
        or row_count != int(expected_row_count)
        or family_names.tolist() != list(expected_names)
        or point_logs.shape != (row_count, FORMAL_POINT_COUNT, len(family_names))
        or point_active.shape != point_logs.shape
        or point_mass.shape != point_logs.shape
        or point_coverage.shape != point_logs.shape
        or not np.isfinite(point_logs).all()
        or not np.isfinite(point_mass).all()
        or not np.isfinite(point_coverage).all()
        or np.any(point_mass < 0.0)
        or np.any((point_coverage < 0.0) | (point_coverage > 1.0 + 1e-6))
    ):
        raise ValueError(f"{path}: absolute-phase point sidecar violates its target-free contract")
    observed = {source: int(np.count_nonzero(point_sources == source)) for source in set(point_sources)}
    if observed != _FORMAL_POINT_COUNTS:
        raise ValueError(f"{path}: absolute-phase sidecar point sources are not formal 64/64/64")


def _load_score(
    path: Path, *, expected_variant: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "family_names",
        "family_log_likelihood_means",
        "family_log_likelihood_medians",
        "family_log_likelihood_worst_quartile_means",
        "family_spatial_median_of_means_2x2",
        "family_effective_point_counts",
        "family_effective_view_masses",
        "verification_point_sources",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: absolute-phase score artifact is incomplete ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    strict = metadata.get("strict_frozen_evidence_contract") if isinstance(metadata, dict) else None
    required = {
        "heldout_query_rows": True,
        "verification_point_selector_fixed_across_hypotheses": True,
        "formal_p1_mixed_multiscale_points": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "support_view_descriptor_averaging": False,
        "candidate_specific_query_projection": True,
        "source_specific_heldout_feature_roles": True,
        "learned_context_head_used": False,
        "learned_identity_head_used": False,
        "feature_only_bounded_2d_translation_cost_volume": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "raw_scores_calibrated_as_independent_pose_likelihood": False,
    }
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") not in {SCORE_FORMAT, CONTEXT_CORRELATION_SCORE_FORMAT}
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or str(metadata.get("evidence_variant", "")) != str(expected_variant)
        or not isinstance(strict, Mapping)
        or any(strict.get(key) is not value for key, value in required.items())
        or bool(strict.get("support_channel_permutation_control_only"))
        != (str(expected_variant) == "support_channel_permutation_control")
    ):
        raise ValueError(f"{path}: not the declared frozen feature-only absolute-phase P1 score")
    count = int(metadata.get("row_count", -1))
    family_names = np.asarray(arrays["family_names"]).astype(str).reshape(-1)
    profile_set, expected_names = _declared_profile_set(
        metadata=metadata, family_names=family_names
    )
    if profile_set == PROFILE_SET_PHASE_V1:
        expected_bidirectional = False
        expected_multiresolution = False
    else:
        expected_bidirectional = True
        expected_multiresolution = True
    if (
        count <= 0
        or family_names.tolist() != expected_names
        or any(
            np.asarray(arrays[field]).shape[0] != count
            for field in fields
            if field != "family_names" and field != "verification_point_sources"
        )
        or np.asarray(arrays["verification_point_sources"]).shape != (FORMAL_POINT_COUNT,)
    ):
        raise ValueError(f"{path}: absolute-phase score row/family arrays are inconsistent")
    for field in (
        "family_log_likelihood_means",
        "family_log_likelihood_medians",
        "family_log_likelihood_worst_quartile_means",
        "family_spatial_median_of_means_2x2",
        "family_effective_point_counts",
        "family_effective_view_masses",
    ):
        values = np.asarray(arrays[field])
        if values.shape != (count, len(family_names)) or not np.isfinite(values).all():
            raise ValueError(f"{path}: absolute-phase {field} is invalid")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: absolute-phase score repeats query/hypothesis rows")
    selection = metadata.get("verification_point_selection")
    observed = {
        source: int(np.count_nonzero(np.asarray(arrays["verification_point_sources"]).astype(str) == source))
        for source in _FORMAL_POINT_COUNTS
    }
    if (
        not isinstance(selection, Mapping)
        or selection.get("source") != "formal_p1_mixed_multiscale_verification_points_v1"
        or int(selection.get("point_count", -1)) != FORMAL_POINT_COUNT
        or selection.get("point_sources") != _FORMAL_POINT_COUNTS
        or selection.get("candidate_fit_rows_excluded_from_alike") is not True
        or selection.get("dense_point_detector_rows_are_sentinel_minus_one") is not True
        or selection.get("development_test_source_points_excluded") is not True
        or observed != _FORMAL_POINT_COUNTS
        or not str(metadata.get("frozen_query_evidence_sha256", ""))
    ):
        raise ValueError(f"{path}: absolute-phase score lacks formal P1 point provenance")
    if (
        bool(strict.get("feature_only_bidirectional_per_token_2d_correlation", False))
        != expected_bidirectional
        or bool(strict.get("radio_final_multiresolution_absolute_context", False))
        != expected_multiresolution
    ):
        raise ValueError(f"{path}: absolute-context score profile set/contract differs")
    scope = metadata.get("hypothesis_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("all_frozen_hypotheses") is not True
        or int(scope.get("development_prefix_limit", -1)) != 0
    ):
        raise ValueError(f"{path}: formal P1 requires every frozen hypothesis")
    sidecar_path = Path(str(metadata.get("point_sidecar", "")))
    if not sidecar_path.is_file():
        raise ValueError(f"{path}: absolute-phase score has no readable point sidecar")
    _load_sidecar(
        sidecar_path,
        expected_evidence_sha=str(metadata["frozen_query_evidence_sha256"]),
        expected_variant=str(expected_variant),
        expected_names=family_names.tolist(),
        expected_row_count=count,
    )
    return arrays, metadata


def _merge_scores(
    paths: Sequence[Path], *, expected_variant: str
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], tuple[tuple[str, str, str, int], ...]]:
    loaded = [_load_score(path, expected_variant=expected_variant) for path in paths]
    names = np.asarray(loaded[0][0]["family_names"]).astype(str)
    compatibility = {
        "format": loaded[0][1].get("format"),
        "version": loaded[0][1].get("version"),
        "profile_set": loaded[0][1].get("profile_set", PROFILE_SET_PHASE_V1),
        "family_names": names.tolist(),
        "profiles": loaded[0][1].get("profiles"),
        "family_components": loaded[0][1].get("family_components"),
        "strict": loaded[0][1].get("strict_frozen_evidence_contract"),
        "source_cache_lineage": loaded[0][1].get("source_cache_lineage"),
        "score_semantics": loaded[0][1].get("raw_score_semantics"),
    }
    fingerprint = _canonical_hash(compatibility)
    row_fields = tuple(field for field in loaded[0][0] if field not in {"family_names", "verification_point_sources"})
    merged_parts: dict[str, list[np.ndarray]] = {field: [] for field in row_fields}
    metadata: list[dict[str, object]] = []
    for arrays, item_metadata in loaded:
        item = {
            "format": item_metadata.get("format"),
            "version": item_metadata.get("version"),
            "profile_set": item_metadata.get("profile_set", PROFILE_SET_PHASE_V1),
            "family_names": np.asarray(arrays["family_names"]).astype(str).tolist(),
            "profiles": item_metadata.get("profiles"),
            "family_components": item_metadata.get("family_components"),
            "strict": item_metadata.get("strict_frozen_evidence_contract"),
            "source_cache_lineage": item_metadata.get("source_cache_lineage"),
            "score_semantics": item_metadata.get("raw_score_semantics"),
        }
        if _canonical_hash(item) != fingerprint:
            raise ValueError("absolute-phase score shards have incompatible frozen evidence")
        for field in row_fields:
            merged_parts[field].append(np.asarray(arrays[field]))
        metadata.append(item_metadata)
    merged = {field: np.concatenate(parts, axis=0) for field, parts in merged_parts.items()}
    merged["family_names"] = names
    keys = _row_keys(merged)
    if len(keys) != len(set(keys)):
        raise ValueError("merged absolute-phase score shards repeat rows")
    return merged, metadata, keys


def _validate_target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    if target_metadata.get("format") != TARGET_FORMAT:
        raise ValueError("absolute-phase target artifact format differs")
    hypotheses = set(str(value) for value in target_metadata.get("hypothesis_artifact_sha256", []))
    baselines = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("absolute-phase score has no input manifest")
        if (
            str(inputs.get("hypothesis_artifact", {}).get("sha256", "")) not in hypotheses
            or str(inputs.get("baseline_score_artifact", {}).get("sha256", "")) not in baselines
        ):
            raise ValueError("absolute-phase score/target lineage differs from frozen S0")


def _validate_pairing(
    *, visual_metadata: Sequence[Mapping[str, object]], control_metadata: Sequence[Mapping[str, object]]
) -> None:
    visual = {str(item.get("query_id")): item for item in visual_metadata}
    control = {str(item.get("query_id")): item for item in control_metadata}
    if not visual or set(visual) != set(control) or len(visual) != len(visual_metadata):
        raise ValueError("absolute-phase visual/control shards do not cover identical unique queries")

    def normalized_contract(metadata: Mapping[str, object]) -> object:
        strict = metadata.get("strict_frozen_evidence_contract")
        if not isinstance(strict, Mapping):
            return strict
        output = dict(strict)
        output.pop("support_channel_permutation_control_only", None)
        return output

    for query_id in sorted(visual):
        left = visual[query_id]
        right = control[query_id]
        if (
            left.get("frozen_query_evidence_sha256") != right.get("frozen_query_evidence_sha256")
            or left.get("verification_point_selection") != right.get("verification_point_selection")
            or normalized_contract(left) != normalized_contract(right)
            or left.get("profile_set", PROFILE_SET_PHASE_V1)
            != right.get("profile_set", PROFILE_SET_PHASE_V1)
            or left.get("profiles") != right.get("profiles")
            or left.get("family_components") != right.get("family_components")
        ):
            raise ValueError("absolute-phase visual/control inputs differ beyond descriptor control")


def _paired_rank(
    visual_rows: Sequence[Mapping[str, object]], control_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    visual = {key(row): row for row in visual_rows}
    control = {key(row): row for row in control_rows}
    if not visual or set(visual) != set(control):
        raise ValueError("absolute-phase visual/control per-query rows are unpaired")
    delta = np.asarray(
        [
            float(visual[item]["oracle_score_rank"]) - float(control[item]["oracle_score_rank"])
            for item in sorted(visual)
        ],
        dtype=np.float64,
    )
    return {
        "oracle_rank_wins": int(np.count_nonzero(delta < -_EPSILON)),
        "oracle_rank_losses": int(np.count_nonzero(delta > _EPSILON)),
        "oracle_rank_ties": int(np.count_nonzero(np.abs(delta) <= _EPSILON)),
        "median_oracle_rank_delta": float(np.median(delta)),
        "mean_oracle_rank_delta": float(np.mean(delta)),
    }


def _primary_gate(
    *, visual: Mapping[str, object], control: Mapping[str, object], paired_rank: Mapping[str, object]
) -> dict[str, object]:
    checks = {
        "visual_median_oracle_rank_below_control": float(visual["median_oracle_score_rank"])
        < float(control["median_oracle_score_rank"]),
        "visual_p90_oracle_rank_not_worse": float(visual["p90_oracle_score_rank"])
        <= float(control["p90_oracle_score_rank"]),
        "visual_p90_translation_not_worse": float(visual["p90_selected_translation_cm"])
        <= float(control["p90_selected_translation_cm"]),
        "visual_catastrophic_tail_not_worse": int(visual["catastrophic_1m_count"])
        <= int(control["catastrophic_1m_count"]),
        "visual_oracle_rank_wins_exceed_losses": int(paired_rank["oracle_rank_wins"])
        > int(paired_rank["oracle_rank_losses"]),
    }
    return {
        **checks,
        "validation_rank_median_under_35": float(visual["median_oracle_score_rank"]) < 35.0,
        "promotion_rank_median_under_20": float(visual["median_oracle_score_rank"]) < 20.0,
        "uncalibrated_probe_signal_passed": bool(all(checks.values())),
        "eligible_for_pose_or_pnp_promotion": False,
        "reason": (
            "feature-only P1 scores remain uncalibrated; a pass authorizes only a train-only "
            "correct-versus-coherent-wrong pose likelihood-ratio fit"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite absolute-phase audit: {summary_path}")
    visual_paths = _paths(args.visual_score_artifacts)
    control_paths = _paths(args.control_score_artifacts)
    visual, visual_metadata, visual_keys = _merge_scores(visual_paths, expected_variant="visual")
    control, control_metadata, control_keys = _merge_scores(
        control_paths, expected_variant="support_channel_permutation_control"
    )
    if visual_keys != control_keys:
        raise ValueError("absolute-phase visual/control scores have different frozen hypothesis keys")
    _validate_pairing(visual_metadata=visual_metadata, control_metadata=control_metadata)
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _validate_target_lineage(score_metadata=visual_metadata, target_metadata=target_metadata)
    _validate_target_lineage(score_metadata=control_metadata, target_metadata=target_metadata)
    target_positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_positions[key] for key in visual_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("an absolute-phase score row is absent from the target artifact") from error
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline = np.asarray(visual["baseline_selection_scores"], dtype=np.float64)
    if (
        not np.array_equal(baseline, np.asarray(control["baseline_selection_scores"], dtype=np.float64))
        or not np.array_equal(
            np.asarray(visual["baseline_score_top1"], dtype=bool),
            np.asarray(control["baseline_score_top1"], dtype=bool),
        )
        or not np.isfinite(baseline).all()
        or translation.shape != baseline.shape
        or rotation.shape != baseline.shape
    ):
        raise ValueError("absolute-phase visual/control baselines or targets are misaligned")
    tie_orders = np.arange(len(visual_keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=visual_keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(
        source_top1=np.asarray(visual["baseline_score_top1"], dtype=bool),
        keys=visual_keys,
        baseline_rows=baseline_rows,
    )
    field = _STATISTIC_FIELDS[str(args.score_statistic)]
    visual_values = np.asarray(visual[field], dtype=np.float64)
    control_values = np.asarray(control[field], dtype=np.float64)
    family_names = np.asarray(visual["family_names"]).astype(str)
    all_rows: list[dict[str, object]] = list(baseline_rows)
    families: dict[str, object] = {}
    for index, name in enumerate(family_names.tolist()):
        visual_rows = _per_query_rows(
            keys=visual_keys,
            scores=visual_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"visual:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        control_rows = _per_query_rows(
            keys=visual_keys,
            scores=control_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"support_channel_permutation_control:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        all_rows.extend(visual_rows)
        all_rows.extend(control_rows)
        visual_summary = _split_summary(visual_rows)
        control_summary = _split_summary(control_rows)
        paired_rank = _paired_rank(visual_rows, control_rows)
        report: dict[str, object] = {
            "visual": {
                "splits": visual_summary,
                "paired_vs_s0": _paired(baseline_rows, visual_rows),
            },
            "support_channel_permutation_control": {
                "splits": control_summary,
                "paired_vs_s0": _paired(baseline_rows, control_rows),
            },
            "visual_vs_control": {
                "paired_rank": paired_rank,
                "paired_selected_translation": _paired(control_rows, visual_rows),
            },
            "effective_evidence": {
                "visual_mean_effective_point_count": float(
                    np.mean(np.asarray(visual["family_effective_point_counts"])[:, index])
                ),
                "control_mean_effective_point_count": float(
                    np.mean(np.asarray(control["family_effective_point_counts"])[:, index])
                ),
                "visual_mean_effective_view_mass": float(
                    np.mean(np.asarray(visual["family_effective_view_masses"])[:, index])
                ),
                "control_mean_effective_view_mass": float(
                    np.mean(np.asarray(control["family_effective_view_masses"])[:, index])
                ),
            },
        }
        report["predeclared_validation_gate"] = _primary_gate(
            visual=visual_summary["validation"],
            control=control_summary["validation"],
            paired_rank=paired_rank,
        )
        families[name] = report
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary: dict[str, Any] = {
        "stage": "frozen_absolute_phase_pose_target_audit",
        "format": "frozen_absolute_phase_pose_target_audit_v1",
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "visual_support_channel_permutation_control_paired": True,
            "fixed_global_topl": True,
            "explicit_null": True,
            "fixed_support_views": True,
            "candidate_specific_query_projection": True,
            "feature_only_bounded_2d_translation_cost_volume": True,
            "learned_context_head_used": False,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "raw_scores_are_uncalibrated": True,
            "promotion_allowed": False,
        },
        "primary_score_statistic": str(args.score_statistic),
        "baseline": {"splits": _split_summary(baseline_rows)},
        "families": families,
        "inputs": {
            "visual_score_artifacts": [str(path) for path in visual_paths],
            "visual_score_artifact_sha256": [file_sha256_short(path) for path in visual_paths],
            "control_score_artifacts": [str(path) for path in control_paths],
            "control_score_artifact_sha256": [file_sha256_short(path) for path in control_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_hash": _canonical_hash(
                {
                    "strict": visual_metadata[0].get("strict_frozen_evidence_contract"),
                    "profiles": visual_metadata[0].get("profiles"),
                    "family_components": visual_metadata[0].get("family_components"),
                    "sources": visual_metadata[0].get("source_cache_lineage"),
                }
            ),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
