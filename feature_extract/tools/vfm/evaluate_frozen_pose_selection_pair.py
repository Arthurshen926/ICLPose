"""Gate a target-free identity-prior pose selection against a frozen baseline.

Both inputs must be post-hoc GT audits of exactly the same frozen hypothesis
rows.  This command never reselects a score profile or touches inference
artifacts; it only checks whether a predeclared identity prior improves the
already selected poses without worsening the held-out tail.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REQUIRED_PER_QUERY_FIELDS = frozenset(
    {
        "query_id",
        "split_name",
        "evaluation_label",
        "selected_translation_error_m",
        "selected_rotation_error_deg",
        "oracle_score_rank",
        "selected_catastrophic_1m",
    }
)
_HELD_OUT_SPLITS = ("validation", "test")
_COMPARISON_MODES = (
    "identity_prior",
    "verification_selector",
    "score_profile",
    "absolute_likelihood",
    "candidate_topk",
)
_CANDIDATE_TOPK_RANKING = (
    "frozen_overlay_probability_descending_stable_candidate_column_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--probe_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--splits",
        default=",".join(_HELD_OUT_SPLITS),
        help=(
            "comma-separated held-out splits to audit; only validation,test "
            "can support a promotion decision"
        ),
    )
    parser.add_argument(
        "--comparison_mode",
        choices=_COMPARISON_MODES,
        default="identity_prior",
        help=(
            "identity_prior requires an identical fixed verifier set; "
            "verification_selector permits only a declared target-free "
            "verifier-row selection change with the same point budget; "
            "score_profile permits only a declared target-free score-field "
            "reselection over identical frozen score rows; "
            "absolute_likelihood permits only the declared no-RGB to "
            "candidate-specific RGB likelihood semantic change over the same "
            "frozen hypotheses and denominator; candidate_topk permits only a "
            "fixed-posterior top-K truncation whose removed mass is moved to "
            "the explicit null state"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _requested_splits(value: Sequence[str] | str) -> tuple[str, ...]:
    values = tuple(
        item.strip()
        for item in (
            str(value).split(",") if isinstance(value, str) else value
        )
        if str(item).strip()
    )
    if not values or len(set(values)) != len(values):
        raise ValueError("requested held-out splits must be non-empty and unique")
    unknown = sorted(set(values).difference(_HELD_OUT_SPLITS))
    if unknown:
        raise ValueError(f"unsupported held-out splits: {unknown}")
    return values


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: evaluation summary is not an object")
    if payload.get("stage") != "independent_landmark_pose_score_gt_join":
        raise ValueError(f"{path}: unsupported pose-evaluation summary")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping) or any(
        protocol.get(key) is not True
        for key in (
            "score_selection_frozen_before_gt_join",
            "target_free_score_artifact_validated",
            "same_hypothesis_denominator_for_source_and_independent_selection",
        )
    ):
        raise ValueError(f"{path}: pose evaluation lacks the frozen-score contract")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("per_query"), str):
        raise ValueError(f"{path}: pose evaluation has no per-query output")
    return payload


def _parse_bool(value: object, *, context: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{context}: expected a Boolean CSV value")


def _load_per_query(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = frozenset(reader.fieldnames or ())
        missing = sorted(_REQUIRED_PER_QUERY_FIELDS - fields)
        if missing:
            raise ValueError(f"{path}: per-query CSV lacks {missing}")
        output: dict[tuple[str, str, str], dict[str, object]] = {}
        for row in reader:
            key = (
                str(row["split_name"]),
                str(row["evaluation_label"]),
                str(row["query_id"]),
            )
            if key in output:
                raise ValueError(f"{path}: duplicate per-query selection {key}")
            translation = float(row["selected_translation_error_m"])
            rotation = float(row["selected_rotation_error_deg"])
            rank = int(row["oracle_score_rank"])
            if (
                not np.isfinite(translation)
                or not np.isfinite(rotation)
                or translation < 0.0
                or rotation < 0.0
                or rank <= 0
            ):
                raise ValueError(f"{path}: invalid selection result for {key}")
            output[key] = {
                "translation_m": translation,
                "rotation_deg": rotation,
                "oracle_score_rank": rank,
                "catastrophic_1m": _parse_bool(
                    row["selected_catastrophic_1m"], context=f"{path} {key}"
                ),
            }
    if not output:
        raise ValueError(f"{path}: per-query CSV is empty")
    return output


def _score_contract(summary: Mapping[str, Any]) -> dict[str, object]:
    inputs = summary.get("inputs")
    metadata = summary.get("score_metadata")
    if not isinstance(inputs, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("pose evaluation lacks source score provenance")
    return {
        "hypothesis_artifact_sha256": inputs.get("hypothesis_artifact_sha256"),
        "colmap_images_bin_sha256": inputs.get("colmap_images_bin_sha256"),
        "selection": summary.get("selection"),
        "version": metadata.get("version"),
        "hypothesis_compatibility_sha256": metadata.get(
            "hypothesis_compatibility_sha256"
        ),
        "config": metadata.get("config"),
        "query_point_selection": metadata.get("query_point_selection"),
        "hypothesis_scope": metadata.get("hypothesis_scope"),
        "crossfit": metadata.get("crossfit"),
        "strict_absolute_evidence_contract": metadata.get(
            "strict_absolute_evidence_contract"
        ),
    }


def _verification_selector_contract(
    summary: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate the declared point selector from all fixed scoring semantics."""

    contract = _score_contract(summary)
    selection = contract.pop("query_point_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pose evaluation lacks query point-selection provenance")
    point_count = selection.get("verification_point_count")
    if not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("query point-selection has an invalid point budget")
    return contract, dict(selection)


def _validate_verification_selector_ablation(
    baseline_summary: Mapping[str, Any], probe_summary: Mapping[str, Any]
) -> None:
    baseline_contract, baseline_selection = _verification_selector_contract(
        baseline_summary
    )
    probe_contract, probe_selection = _verification_selector_contract(probe_summary)
    if baseline_contract != probe_contract:
        raise ValueError("selector ablation changes score contracts beyond point selection")
    if (
        baseline_selection["verification_point_count"]
        != probe_selection["verification_point_count"]
        or baseline_selection.get("score_splits") != probe_selection.get("score_splits")
    ):
        raise ValueError("selector ablation changes the verifier point budget or splits")
    if probe_selection.get("source") != (
        "target_free_identity_posterior_spatial_quota_selector"
    ):
        raise ValueError("selector probe lacks the declared target-free selector source")
    if not isinstance(
        probe_selection.get("verification_point_selection_artifact"), str
    ) or not probe_selection.get("verification_point_selection_artifact_sha256"):
        raise ValueError("selector probe lacks a frozen selector artifact")
    if baseline_selection.get("verification_point_selection_artifact") is not None:
        raise ValueError("selector baseline must use the legacy verifier-row selection")


def _score_profile_contract(
    summary: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Separate a frozen score-field profile from all scorer semantics."""

    contract = _score_contract(summary)
    selection = contract.pop("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pose evaluation lacks score-profile selection provenance")
    statistic = selection.get("statistic")
    score_field = selection.get("score_field")
    if not isinstance(statistic, str) or not statistic:
        raise ValueError("score-profile selection has no statistic")
    if not isinstance(score_field, str) or not score_field:
        raise ValueError("score-profile selection has no score field")
    metadata = summary.get("score_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("pose evaluation lacks score metadata")
    transform = metadata.get("profile_transform")
    if not isinstance(transform, Mapping):
        raise ValueError(
            "score-profile comparison requires a frozen target-free profile transform"
        )
    required = {
        "target_free": True,
        "source_score_rows_frozen": True,
        "candidate_or_hypothesis_regeneration": False,
    }
    if any(transform.get(key) is not expected for key, expected in required.items()):
        raise ValueError("score-profile transform is not a frozen target-free reselection")
    source_hashes = transform.get("source_score_artifact_sha256")
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ValueError("score-profile transform lacks source score artifact hashes")
    return contract, dict(selection), dict(transform)


def _validate_score_profile_ablation(
    baseline_summary: Mapping[str, Any], probe_summary: Mapping[str, Any]
) -> None:
    """Require a pure target-free statistic change over identical score rows."""

    baseline_contract, baseline_selection, baseline_transform = (
        _score_profile_contract(baseline_summary)
    )
    probe_contract, probe_selection, probe_transform = _score_profile_contract(
        probe_summary
    )
    if baseline_contract != probe_contract:
        raise ValueError("score-profile ablation changes scorer contracts beyond selection")
    if (
        baseline_transform.get("source_score_artifact_sha256")
        != probe_transform.get("source_score_artifact_sha256")
        or baseline_transform.get("source_score_compatibility_sha256")
        != probe_transform.get("source_score_compatibility_sha256")
    ):
        raise ValueError("score-profile baseline/probe do not share frozen source rows")
    if baseline_selection == probe_selection:
        raise ValueError("score-profile baseline/probe select the same score field")


def _absolute_likelihood_invariants(summary: Mapping[str, Any]) -> dict[str, object]:
    """Return everything an RGB-likelihood ablation is not allowed to change.

    This comparison deliberately permits only the transition from no spatial
    modes to the normalized candidate-specific RGB likelihood.  In particular,
    it does not permit an image-retrieval change, a new candidate pool, a new
    verifier set, or a pose-conditioned candidate denominator.
    """

    contract = _score_contract(summary)
    inputs = summary.get("inputs")
    metadata = summary.get("score_metadata")
    if not isinstance(inputs, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("absolute-likelihood summary lacks score provenance")
    strict = metadata.get("strict_absolute_evidence_contract")
    config = metadata.get("config")
    metadata_inputs = metadata.get("inputs")
    if (
        not isinstance(strict, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(metadata_inputs, Mapping)
    ):
        raise ValueError("absolute-likelihood summary lacks strict score semantics")
    required_strict = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
    }
    if any(strict.get(key) != value for key, value in required_strict.items()):
        raise ValueError("absolute-likelihood summary violates frozen-score semantics")
    crossfit = contract.get("crossfit")
    if not isinstance(crossfit, Mapping) or crossfit.get(
        "denominator_fixed_across_hypotheses"
    ) is not True:
        raise ValueError("absolute-likelihood summary lacks a fixed denominator")
    input_keys = (
        "hypothesis_artifact_sha256",
        "colmap_images_bin_sha256",
        "candidate_artifact_sha256",
        "proposals_sha256",
        "fixed_candidate_prior_overlay_sha256",
        "projected_landmark_bank_sha256",
        "independent_verification_landmark_bank_sha256",
        "support_geometry_index_sha256",
        "detector_query_cache_sha256",
    )
    def input_hash(key: str) -> object:
        # The aggregate GT-join summary owns the full hypothesis list, while
        # scorer-only inputs live under score_metadata.  The evaluator already
        # checked that every score shard shares the latter provenance.
        return inputs.get(key, metadata_inputs.get(key))

    missing = [key for key in input_keys if input_hash(key) is None]
    if missing:
        raise ValueError(
            "absolute-likelihood summary lacks immutable input hashes: "
            + ", ".join(missing)
        )
    config_without_spatial = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("candidate_spatial_")
    }
    query_selection = contract.get("query_point_selection")
    selection = contract.get("selection")
    if not isinstance(query_selection, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("absolute-likelihood summary has incomplete selection provenance")
    raw_score_splits = query_selection.get("score_splits")
    if raw_score_splits is None:
        score_splits = ("all",)
    elif isinstance(raw_score_splits, (list, tuple)) and all(
        isinstance(value, str) for value in raw_score_splits
    ):
        score_splits = tuple(sorted(str(value) for value in raw_score_splits))
    else:
        raise ValueError("absolute-likelihood summary has invalid score-split scope")
    return {
        "input_hashes": {key: input_hash(key) for key in input_keys},
        "selection": {
            key: selection.get(key)
            for key in ("statistic", "score_field", "tie_break")
        },
        "query_point_selection": {
            key: query_selection.get(key)
            for key in (
                "verification_point_count",
                "detector_log_merit_weight",
                "merit",
                "source",
                "verification_point_selection_artifact",
                "verification_point_selection_artifact_sha256",
            )
        },
        "score_splits": score_splits,
        "hypothesis_scope": contract.get("hypothesis_scope"),
        "crossfit": dict(crossfit),
        "config_without_spatial": config_without_spatial,
    }


def _validate_absolute_likelihood_ablation(
    baseline_summary: Mapping[str, Any], probe_summary: Mapping[str, Any]
) -> None:
    """Validate the fixed no-RGB -> normalized RGB likelihood experiment."""

    baseline_invariants = _absolute_likelihood_invariants(baseline_summary)
    probe_invariants = _absolute_likelihood_invariants(probe_summary)
    baseline_score_splits = baseline_invariants.pop("score_splits")
    probe_score_splits = probe_invariants.pop("score_splits")
    allowed_heldout_scope_change = {
        tuple(baseline_score_splits),
        tuple(probe_score_splits),
    } == {("all",), ("test", "validation")}
    if (
        baseline_score_splits != probe_score_splits
        and not allowed_heldout_scope_change
    ):
        raise ValueError(
            "absolute-likelihood ablation changes score-split scope beyond the "
            "legacy all-split baseline to held-out RGB materialization"
        )
    if baseline_invariants != probe_invariants:
        raise ValueError(
            "absolute-likelihood ablation changes frozen inputs or score "
            "semantics beyond candidate spatial evidence"
        )
    baseline_strict = dict(
        baseline_summary["score_metadata"]["strict_absolute_evidence_contract"]
    )
    probe_strict = dict(
        probe_summary["score_metadata"]["strict_absolute_evidence_contract"]
    )
    if baseline_strict.get("candidate_specific_rgb_spatial_modes") is not False:
        raise ValueError("absolute-likelihood baseline must omit RGB spatial modes")
    if probe_strict.get("candidate_specific_rgb_spatial_modes") is not True:
        raise ValueError("absolute-likelihood probe must use RGB spatial modes")
    if probe_strict.get("candidate_spatial_dustbin_and_missing_pose_independent") is not True:
        raise ValueError("RGB likelihood probe must keep dustbin/missing pose-independent")
    if probe_strict.get("candidate_spatial_omitted_topk_mass_is_null") is not True:
        raise ValueError("RGB likelihood probe must conserve omitted top-K mass")
    if probe_strict.get("candidate_spatial_semantics") != (
        "per_view_normalized_continuous_gaussian_mixture_relative_to_"
        "grid_uniform_null_v1"
    ):
        raise ValueError("RGB likelihood probe lacks normalized spatial semantics")
    _validate_full_candidate_posterior_if_declared(probe_summary)


def _topk_ablation_metadata(summary: Mapping[str, Any]) -> dict[str, object]:
    metadata = summary.get("score_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("pose evaluation lacks score metadata")
    topk = metadata.get("fixed_candidate_topk_ablation")
    if not isinstance(topk, Mapping):
        raise ValueError("pose evaluation lacks fixed-candidate top-K provenance")
    return dict(topk)


def _validate_full_candidate_posterior_if_declared(
    summary: Mapping[str, Any],
) -> None:
    """Reject a top-K posterior truncation in a no-RGB -> RGB comparison.

    The absolute-likelihood gate answers only whether RGB spatial evidence is
    useful under the full fixed candidate posterior.  A non-full top-K probe
    changes the candidate/null denominator as well, so it must use the
    dedicated ``candidate_topk`` comparison mode below.

    Older score artifacts did not record this provenance.  They remain usable
    for historical diagnostics, while every newly produced declared top-K
    artifact is checked strictly.
    """

    metadata = summary.get("score_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("pose evaluation lacks score metadata")
    if "fixed_candidate_topk_ablation" not in metadata:
        return
    topk = _topk_ablation_metadata(summary)
    if topk.get("applied") is not True:
        raise ValueError("RGB likelihood probe must declare a full frozen candidate posterior")
    count = topk.get("candidate_column_count")
    selected = topk.get("top_k")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected != count
    ):
        raise ValueError(
            "absolute-likelihood RGB probe must retain the full frozen candidate posterior"
        )
    if topk.get("ranking") != _CANDIDATE_TOPK_RANKING:
        raise ValueError("RGB likelihood probe has an unknown candidate top-K ranking")
    if topk.get("removed_candidate_mass_transferred_to_null") is not True:
        raise ValueError("RGB likelihood probe does not conserve removed candidate mass")


def _candidate_topk_invariants(
    summary: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return immutable provenance for a full-RGB fixed-posterior top-K audit."""

    metadata = summary.get("score_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate-topK summary lacks score metadata")
    strict = metadata.get("strict_absolute_evidence_contract")
    config = metadata.get("config")
    metadata_inputs = metadata.get("inputs")
    implementation = metadata.get("implementation")
    if (
        not isinstance(strict, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(metadata_inputs, Mapping)
        or not isinstance(implementation, Mapping)
    ):
        raise ValueError("candidate-topK summary lacks strict scorer provenance")
    required_strict = {
        "candidate_specific_rgb_spatial_modes": True,
        "candidate_spatial_dustbin_and_missing_pose_independent": True,
        "candidate_spatial_omitted_topk_mass_is_null": True,
        "candidate_spatial_semantics": (
            "per_view_normalized_continuous_gaussian_mixture_relative_to_"
            "grid_uniform_null_v1"
        ),
    }
    if any(strict.get(key) != value for key, value in required_strict.items()):
        raise ValueError("candidate-topK summary lacks normalized RGB spatial semantics")
    spatial_hash_keys = (
        "candidate_spatial_likelihood_sha256",
        "candidate_spatial_likelihood_metadata_sha256",
    )
    spatial_hashes: dict[str, object] = {}
    for key in spatial_hash_keys:
        value = metadata_inputs.get(key)
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"candidate-topK summary lacks immutable {key}")
        spatial_hashes[key] = list(value)
    overlay_metadata_hash = metadata_inputs.get(
        "fixed_candidate_prior_overlay_metadata_sha256"
    )
    if not isinstance(overlay_metadata_hash, str) or not overlay_metadata_hash:
        raise ValueError("candidate-topK summary lacks frozen overlay metadata hash")
    invariants = _absolute_likelihood_invariants(summary)
    score_splits = invariants.pop("score_splits")
    topk = _topk_ablation_metadata(summary)
    strict_topk = strict.get("fixed_candidate_topk_ablation")
    if not isinstance(strict_topk, Mapping):
        raise ValueError("candidate-topK summary lacks strict top-K semantics")
    for key in (
        "applied",
        "top_k",
        "ranking",
        "removed_candidate_mass_transferred_to_null",
    ):
        if strict_topk.get(key) != topk.get(key):
            raise ValueError(
                "candidate-topK strict contract disagrees with top-level provenance"
            )
    strict_without_topk = dict(strict)
    strict_without_topk.pop("fixed_candidate_topk_ablation", None)
    return (
        {
            "absolute_invariants": invariants,
            "score_splits": score_splits,
            "strict_without_topk": strict_without_topk,
            "full_config": dict(config),
            "implementation": dict(implementation),
            "score_metadata_inputs": dict(metadata_inputs),
            "spatial_hashes": spatial_hashes,
            "fixed_candidate_prior_overlay_metadata_sha256": overlay_metadata_hash,
        },
        topk,
    )


def _validate_candidate_topk_metadata(
    *,
    baseline_topk: Mapping[str, object],
    probe_topk: Mapping[str, object],
) -> None:
    """Ensure only full-posterior -> truncated-posterior semantics differ."""

    for role, metadata in (("baseline", baseline_topk), ("probe", probe_topk)):
        if metadata.get("applied") is not True:
            raise ValueError(f"candidate-topK {role} must declare a top-K posterior")
        count = metadata.get("candidate_column_count")
        selected = metadata.get("top_k")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected <= 0
            or selected > count
        ):
            raise ValueError(f"candidate-topK {role} has an invalid top-K declaration")
        if metadata.get("ranking") != _CANDIDATE_TOPK_RANKING:
            raise ValueError(f"candidate-topK {role} has an unknown frozen ranking")
        if metadata.get("removed_candidate_mass_transferred_to_null") is not True:
            raise ValueError(
                f"candidate-topK {role} does not transfer removed mass to null"
            )
    baseline_count = int(baseline_topk["candidate_column_count"])
    baseline_selected = int(baseline_topk["top_k"])
    probe_count = int(probe_topk["candidate_column_count"])
    probe_selected = int(probe_topk["top_k"])
    if baseline_count != probe_count:
        raise ValueError("candidate-topK baseline/probe change candidate column count")
    if baseline_selected != baseline_count:
        raise ValueError("candidate-topK baseline must retain the full posterior")
    if probe_selected >= baseline_selected:
        raise ValueError("candidate-topK probe must truncate the full posterior")


def _validate_candidate_topk_ablation(
    baseline_summary: Mapping[str, Any], probe_summary: Mapping[str, Any]
) -> None:
    """Validate a pure fixed-posterior top-K-to-null ablation.

    Both sides must use the exact same candidate-specific RGB modes and all
    frozen score inputs.  The only legal change is retaining fewer overlay
    columns after ranking by that already-frozen posterior.
    """

    baseline_invariants, baseline_topk = _candidate_topk_invariants(baseline_summary)
    probe_invariants, probe_topk = _candidate_topk_invariants(probe_summary)
    if baseline_invariants != probe_invariants:
        raise ValueError(
            "candidate-topK ablation changes immutable RGB evidence, scorer, or frozen inputs"
        )
    _validate_candidate_topk_metadata(
        baseline_topk=baseline_topk,
        probe_topk=probe_topk,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _split_gate(
    baseline: Mapping[tuple[str, str, str], Mapping[str, object]],
    probe: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    split_name: str,
) -> dict[str, Any]:
    base_rows = {
        key: value for key, value in baseline.items() if key[0] == str(split_name)
    }
    probe_rows = {
        key: value for key, value in probe.items() if key[0] == str(split_name)
    }
    if not base_rows or set(base_rows) != set(probe_rows):
        raise ValueError(f"{split_name}: baseline/probe selected-query rows differ")
    keys = sorted(base_rows)
    base_translation = np.asarray(
        [float(base_rows[key]["translation_m"]) for key in keys], dtype=np.float64
    )
    probe_translation = np.asarray(
        [float(probe_rows[key]["translation_m"]) for key in keys], dtype=np.float64
    )
    base_rotation = np.asarray(
        [float(base_rows[key]["rotation_deg"]) for key in keys], dtype=np.float64
    )
    probe_rotation = np.asarray(
        [float(probe_rows[key]["rotation_deg"]) for key in keys], dtype=np.float64
    )
    base_rank = np.asarray(
        [int(base_rows[key]["oracle_score_rank"]) for key in keys], dtype=np.int64
    )
    probe_rank = np.asarray(
        [int(probe_rows[key]["oracle_score_rank"]) for key in keys], dtype=np.int64
    )
    delta = probe_translation - base_translation
    tolerance = 1e-12
    wins = int(np.sum(delta < -tolerance))
    losses = int(np.sum(delta > tolerance))
    baseline_summary = {
        "query_count": int(len(keys)),
        "median_translation_m": _quantile(base_translation, 0.5),
        "p90_translation_m": _quantile(base_translation, 0.9),
        "median_rotation_deg": _quantile(base_rotation, 0.5),
        "p90_rotation_deg": _quantile(base_rotation, 0.9),
        "catastrophic_1m_count": int(
            sum(bool(base_rows[key]["catastrophic_1m"]) for key in keys)
        ),
        "median_oracle_score_rank": _quantile(base_rank, 0.5),
        "p90_oracle_score_rank": _quantile(base_rank, 0.9),
    }
    probe_summary = {
        "query_count": int(len(keys)),
        "median_translation_m": _quantile(probe_translation, 0.5),
        "p90_translation_m": _quantile(probe_translation, 0.9),
        "median_rotation_deg": _quantile(probe_rotation, 0.5),
        "p90_rotation_deg": _quantile(probe_rotation, 0.9),
        "catastrophic_1m_count": int(
            sum(bool(probe_rows[key]["catastrophic_1m"]) for key in keys)
        ),
        "median_oracle_score_rank": _quantile(probe_rank, 0.5),
        "p90_oracle_score_rank": _quantile(probe_rank, 0.9),
    }
    deltas = {
        key: float(probe_summary[key] - baseline_summary[key])
        for key in (
            "median_translation_m",
            "p90_translation_m",
            "median_rotation_deg",
            "p90_rotation_deg",
            "median_oracle_score_rank",
            "p90_oracle_score_rank",
        )
    }
    deltas["catastrophic_1m_count"] = int(
        probe_summary["catastrophic_1m_count"]
        - baseline_summary["catastrophic_1m_count"]
    )
    checks = {
        "median_translation_strictly_improved": deltas["median_translation_m"] < 0.0,
        "p90_translation_not_worse": deltas["p90_translation_m"] <= 0.0,
        "median_rotation_not_worse": deltas["median_rotation_deg"] <= 0.0,
        "p90_rotation_not_worse": deltas["p90_rotation_deg"] <= 0.0,
        "catastrophic_1m_not_worse": deltas["catastrophic_1m_count"] <= 0,
        "median_oracle_score_rank_not_worse": deltas["median_oracle_score_rank"] <= 0.0,
        "p90_oracle_score_rank_not_worse": deltas["p90_oracle_score_rank"] <= 0.0,
        "paired_translation_wins_not_less_than_losses": wins >= losses,
    }
    return {
        "baseline": baseline_summary,
        "probe": probe_summary,
        "deltas": deltas,
        "paired_counts": {
            "translation_wins": wins,
            "translation_losses": losses,
            "translation_ties": int(len(keys) - wins - losses),
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def evaluate_frozen_pose_selection_pair(
    *,
    baseline_summary_path: Path,
    probe_summary_path: Path,
    comparison_mode: str = "identity_prior",
    splits: Sequence[str] = _HELD_OUT_SPLITS,
) -> dict[str, Any]:
    if str(comparison_mode) not in _COMPARISON_MODES:
        raise ValueError(f"unsupported comparison mode: {comparison_mode}")
    requested_splits = _requested_splits(splits)
    baseline_summary = _load_summary(baseline_summary_path)
    probe_summary = _load_summary(probe_summary_path)
    if str(comparison_mode) == "identity_prior":
        baseline_contract = _score_contract(baseline_summary)
        probe_contract = _score_contract(probe_summary)
        if baseline_contract != probe_contract:
            raise ValueError("baseline/probe pose-score contracts differ beyond the prior")
    elif str(comparison_mode) == "verification_selector":
        _validate_verification_selector_ablation(baseline_summary, probe_summary)
    elif str(comparison_mode) == "score_profile":
        _validate_score_profile_ablation(baseline_summary, probe_summary)
    elif str(comparison_mode) == "absolute_likelihood":
        _validate_absolute_likelihood_ablation(baseline_summary, probe_summary)
    else:
        _validate_candidate_topk_ablation(baseline_summary, probe_summary)
    baseline_rows = _load_per_query(
        Path(str(dict(baseline_summary["outputs"])["per_query"]))
    )
    probe_rows = _load_per_query(Path(str(dict(probe_summary["outputs"])["per_query"])))
    return {
        "stage": (
            "frozen_identity_prior_pose_selection_pair_gate"
            if str(comparison_mode) == "identity_prior"
            else (
                "frozen_verification_selector_pose_selection_pair_audit"
                if str(comparison_mode) == "verification_selector"
                else (
                    "frozen_score_profile_pose_selection_pair_audit"
                    if str(comparison_mode) == "score_profile"
                    else (
                        "frozen_absolute_likelihood_pose_selection_pair_audit"
                        if str(comparison_mode) == "absolute_likelihood"
                        else "frozen_candidate_topk_pose_selection_pair_audit"
                    )
                )
            )
        ),
        "protocol": {
            "same_frozen_hypotheses": True,
            "same_target_free_score_configuration": True,
            "comparison_mode": str(comparison_mode),
            "selection_frozen_before_gt_join": True,
            "test_used_for_model_selection": False,
            "requested_splits": list(requested_splits),
            "complete_cross_split_gate": (
                set(requested_splits) == set(_HELD_OUT_SPLITS)
            ),
        },
        "baseline_summary": str(baseline_summary_path),
        "probe_summary": str(probe_summary_path),
        "splits": {
            split_name: _split_gate(
                baseline_rows, probe_rows, split_name=split_name
            )
            for split_name in requested_splits
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=Path(args.baseline_summary),
        probe_summary_path=Path(args.probe_summary),
        comparison_mode=str(args.comparison_mode),
        splits=_requested_splits(args.splits),
    )
    result["pass"] = bool(all(row["pass"] for row in result["splits"].values()))
    result["promotion_eligible"] = bool(
        result["pass"]
        and bool(dict(result["protocol"]).get("complete_cross_split_gate"))
    )
    if not bool(dict(result["protocol"]).get("complete_cross_split_gate")):
        result["next_action"] = "development_only_audit_do_not_promote_or_touch_test"
    elif str(args.comparison_mode) == "identity_prior":
        result["next_action"] = (
            "eligible_for_pose_generation_identity_prior_ablation"
            if result["promotion_eligible"]
            else "keep_identity_prior_as_diagnostic_and_do_not_promote"
        )
    elif str(args.comparison_mode) == "verification_selector":
        result["next_action"] = (
            "eligible_for_same_selector_identity_prior_audit"
            if result["promotion_eligible"]
            else "keep_verification_selector_as_diagnostic_and_do_not_promote"
        )
    elif str(args.comparison_mode) == "score_profile":
        result["next_action"] = (
            "eligible_for_followup_pose_selection_ablation"
            if result["promotion_eligible"]
            else "keep_score_profile_as_diagnostic_and_do_not_promote"
        )
    elif str(args.comparison_mode) == "absolute_likelihood":
        result["next_action"] = (
            "eligible_for_safe_absolute_likelihood_promotion"
            if result["promotion_eligible"]
            else "keep_absolute_likelihood_as_diagnostic_and_do_not_promote"
        )
    else:
        result["next_action"] = (
            "eligible_for_fixed_posterior_topk_promotion"
            if result["promotion_eligible"]
            else "keep_fixed_posterior_topk_as_diagnostic_and_do_not_promote"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
