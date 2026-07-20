from __future__ import annotations

import csv
import json

import pytest

from feature_extract.tools.vfm.evaluate_frozen_pose_selection_pair import (
    evaluate_frozen_pose_selection_pair,
)


def _write_evaluation(
    tmp_path,
    name: str,
    rows: list[dict[str, object]],
    *,
    config=None,
    query_point_selection=None,
    selection=None,
    profile_transform=None,
    strict_absolute_evidence_contract=None,
    input_hashes=None,
    fixed_candidate_topk_ablation=None,
):
    csv_path = tmp_path / f"{name}.csv"
    fields = [
        "query_id",
        "split_name",
        "evaluation_label",
        "selected_translation_error_m",
        "selected_rotation_error_deg",
        "oracle_score_rank",
        "selected_catastrophic_1m",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = tmp_path / f"{name}.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage": "independent_landmark_pose_score_gt_join",
                "protocol": {
                    "score_selection_frozen_before_gt_join": True,
                    "target_free_score_artifact_validated": True,
                    "same_hypothesis_denominator_for_source_and_independent_selection": True,
                },
                "selection": (
                    {"statistic": "mean", "score_field": "score"}
                    if selection is None
                    else selection
                ),
                "inputs": {
                    "hypothesis_artifact_sha256": ["hypothesis"],
                    "colmap_images_bin_sha256": "images",
                    **(
                        {}
                        if input_hashes is None
                        else dict(input_hashes)
                    ),
                },
                "score_metadata": {
                    "version": "v1",
                    "inputs": {} if input_hashes is None else dict(input_hashes),
                    "hypothesis_compatibility_sha256": "compatibility",
                    "implementation": {"scorer_source_sha256": "same-source"},
                    "config": {} if config is None else config,
                    "query_point_selection": (
                        {"verification_point_count": 192, "score_splits": None}
                        if query_point_selection is None
                        else query_point_selection
                    ),
                    "hypothesis_scope": {"mode": "all"},
                    "crossfit": {
                        "query_token_disjoint": True,
                        "denominator_fixed_across_hypotheses": True,
                    },
                    "strict_absolute_evidence_contract": (
                        {"fixed_global_topl": True}
                        if strict_absolute_evidence_contract is None
                        else strict_absolute_evidence_contract
                    ),
                    **(
                        {}
                        if profile_transform is None
                        else {"profile_transform": profile_transform}
                    ),
                    **(
                        {}
                        if fixed_candidate_topk_ablation is None
                        else {
                            "fixed_candidate_topk_ablation": (
                                fixed_candidate_topk_ablation
                            )
                        }
                    ),
                },
                "outputs": {"per_query": str(csv_path)},
            }
        )
    )
    return summary_path


def _rows(*, translation_offset: float = 0.0, rank_offset: int = 0):
    return [
        {
            "query_id": "validation.png",
            "split_name": "validation",
            "evaluation_label": "policy",
            "selected_translation_error_m": 0.30 + translation_offset,
            "selected_rotation_error_deg": 1.0,
            "oracle_score_rank": 4 + rank_offset,
            "selected_catastrophic_1m": False,
        },
        {
            "query_id": "validation2.png",
            "split_name": "validation",
            "evaluation_label": "policy",
            "selected_translation_error_m": 0.50 + translation_offset,
            "selected_rotation_error_deg": 1.5,
            "oracle_score_rank": 5 + rank_offset,
            "selected_catastrophic_1m": False,
        },
        {
            "query_id": "test.png",
            "split_name": "test",
            "evaluation_label": "policy",
            "selected_translation_error_m": 0.25 + translation_offset,
            "selected_rotation_error_deg": 0.9,
            "oracle_score_rank": 3 + rank_offset,
            "selected_catastrophic_1m": False,
        },
        {
            "query_id": "test2.png",
            "split_name": "test",
            "evaluation_label": "policy",
            "selected_translation_error_m": 0.45 + translation_offset,
            "selected_rotation_error_deg": 1.1,
            "oracle_score_rank": 4 + rank_offset,
            "selected_catastrophic_1m": False,
        },
    ]


def test_frozen_pose_pair_gate_requires_cross_split_tail_safe_improvement(tmp_path) -> None:
    baseline = _write_evaluation(tmp_path, "baseline", _rows())
    probe = _write_evaluation(
        tmp_path, "probe", _rows(translation_offset=-0.05, rank_offset=-1)
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline, probe_summary_path=probe
    )

    assert result["splits"]["validation"]["pass"] is True
    assert result["splits"]["test"]["pass"] is True
    assert result["splits"]["test"]["paired_counts"]["translation_wins"] == 2


def test_frozen_pose_pair_gate_rejects_changed_score_contract(tmp_path) -> None:
    baseline = _write_evaluation(tmp_path, "baseline", _rows())
    probe = _write_evaluation(tmp_path, "probe", _rows(translation_offset=-0.05), config={"x": 1})

    with pytest.raises(ValueError, match="contracts differ"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline, probe_summary_path=probe
        )


def test_validation_only_pair_is_explicitly_development_only(tmp_path) -> None:
    baseline = _write_evaluation(tmp_path, "baseline", _rows()[:2])
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05, rank_offset=-1)[:2],
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline,
        probe_summary_path=probe,
        splits=("validation",),
    )

    assert result["splits"]["validation"]["pass"] is True
    assert result["protocol"]["complete_cross_split_gate"] is False


def test_verification_selector_pair_allows_only_fixed_budget_selector_change(tmp_path) -> None:
    baseline = _write_evaluation(tmp_path, "baseline", _rows())
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05, rank_offset=-1),
        query_point_selection={
            "verification_point_count": 192,
            "score_splits": None,
            "source": "target_free_identity_posterior_spatial_quota_selector",
            "verification_point_selection_artifact": "selector.npz",
            "verification_point_selection_artifact_sha256": "selector-hash",
        },
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline,
        probe_summary_path=probe,
        comparison_mode="verification_selector",
    )

    assert result["stage"] == "frozen_verification_selector_pose_selection_pair_audit"
    assert result["splits"]["test"]["pass"] is True


def test_verification_selector_pair_rejects_budget_change(tmp_path) -> None:
    baseline = _write_evaluation(tmp_path, "baseline", _rows())
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05),
        query_point_selection={
            "verification_point_count": 64,
            "score_splits": None,
            "source": "target_free_identity_posterior_spatial_quota_selector",
            "verification_point_selection_artifact": "selector.npz",
            "verification_point_selection_artifact_sha256": "selector-hash",
        },
    )

    with pytest.raises(ValueError, match="point budget"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline,
            probe_summary_path=probe,
            comparison_mode="verification_selector",
        )


def test_score_profile_pair_allows_only_frozen_score_field_reselection(tmp_path) -> None:
    transform = {
        "target_free": True,
        "source_score_rows_frozen": True,
        "candidate_or_hypothesis_regeneration": False,
        "source_score_artifact_sha256": ["same-source"],
        "source_score_compatibility_sha256": "same-compatibility",
    }
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        selection={"statistic": "mean", "score_field": "mean_scores"},
        profile_transform=transform,
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05, rank_offset=-1),
        selection={
            "statistic": "unique_track_assignment_log_joint",
            "score_field": "assignment_scores",
        },
        profile_transform=transform,
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline,
        probe_summary_path=probe,
        comparison_mode="score_profile",
    )

    assert result["stage"] == "frozen_score_profile_pose_selection_pair_audit"
    assert result["splits"]["test"]["pass"] is True


def test_score_profile_pair_rejects_different_frozen_source_rows(tmp_path) -> None:
    common = {
        "target_free": True,
        "source_score_rows_frozen": True,
        "candidate_or_hypothesis_regeneration": False,
        "source_score_compatibility_sha256": "same-compatibility",
    }
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        selection={"statistic": "mean", "score_field": "mean_scores"},
        profile_transform={**common, "source_score_artifact_sha256": ["baseline"]},
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05),
        selection={"statistic": "other", "score_field": "other_scores"},
        profile_transform={**common, "source_score_artifact_sha256": ["probe"]},
    )

    with pytest.raises(ValueError, match="share frozen source rows"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline,
            probe_summary_path=probe,
            comparison_mode="score_profile",
        )


def test_absolute_likelihood_pair_permits_only_declared_rgb_semantic_change(tmp_path) -> None:
    input_hashes = {
        "candidate_artifact_sha256": "candidate",
        "proposals_sha256": "proposals",
        "fixed_candidate_prior_overlay_sha256": "prior",
        "projected_landmark_bank_sha256": "bank",
        "independent_verification_landmark_bank_sha256": "bank",
        "support_geometry_index_sha256": "geometry",
        "detector_query_cache_sha256": "detector",
    }
    common = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "verification_point_selector_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
    }
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        config={"candidate_mode": "fixed_global_topl"},
        strict_absolute_evidence_contract={
            **{
                key: value
                for key, value in common.items()
                if key != "verification_point_selector_fixed_across_hypotheses"
            },
            "candidate_specific_rgb_spatial_modes": False,
        },
        input_hashes=input_hashes,
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05, rank_offset=-1),
        config={
            "candidate_mode": "fixed_global_topl",
            "candidate_spatial_semantics": "normalized",
        },
        strict_absolute_evidence_contract={
            **common,
            "candidate_specific_rgb_spatial_modes": True,
            "candidate_spatial_dustbin_and_missing_pose_independent": True,
            "candidate_spatial_omitted_topk_mass_is_null": True,
            "candidate_spatial_semantics": (
                "per_view_normalized_continuous_gaussian_mixture_relative_to_"
                "grid_uniform_null_v1"
            ),
        },
        input_hashes=input_hashes,
        query_point_selection={
            "verification_point_count": 192,
            "score_splits": ["validation", "test"],
        },
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline,
        probe_summary_path=probe,
        comparison_mode="absolute_likelihood",
    )

    assert result["stage"] == "frozen_absolute_likelihood_pose_selection_pair_audit"
    assert result["splits"]["validation"]["pass"] is True
    assert result["splits"]["test"]["pass"] is True


def test_absolute_likelihood_pair_rejects_changed_fixed_candidate_pool(tmp_path) -> None:
    input_hashes = {
        "candidate_artifact_sha256": "candidate",
        "proposals_sha256": "proposals",
        "fixed_candidate_prior_overlay_sha256": "prior",
        "projected_landmark_bank_sha256": "bank",
        "independent_verification_landmark_bank_sha256": "bank",
        "support_geometry_index_sha256": "geometry",
        "detector_query_cache_sha256": "detector",
    }
    common = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "verification_point_selector_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
    }
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        config={"candidate_mode": "fixed_global_topl"},
        strict_absolute_evidence_contract={
            **common,
            "candidate_specific_rgb_spatial_modes": False,
        },
        input_hashes=input_hashes,
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05),
        config={"candidate_mode": "fixed_global_topl"},
        strict_absolute_evidence_contract={
            **common,
            "candidate_specific_rgb_spatial_modes": True,
            "candidate_spatial_dustbin_and_missing_pose_independent": True,
            "candidate_spatial_omitted_topk_mass_is_null": True,
            "candidate_spatial_semantics": (
                "per_view_normalized_continuous_gaussian_mixture_relative_to_"
                "grid_uniform_null_v1"
            ),
        },
        input_hashes={**input_hashes, "proposals_sha256": "different"},
    )

    with pytest.raises(ValueError, match="changes frozen inputs"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline,
            probe_summary_path=probe,
            comparison_mode="absolute_likelihood",
        )


def _rgb_topk_input_hashes() -> dict[str, object]:
    return {
        "candidate_artifact_sha256": "candidate",
        "proposals_sha256": "proposals",
        "fixed_candidate_prior_overlay_sha256": "prior",
        "fixed_candidate_prior_overlay_metadata_sha256": "prior-metadata",
        "projected_landmark_bank_sha256": "bank",
        "independent_verification_landmark_bank_sha256": "bank",
        "support_geometry_index_sha256": "geometry",
        "detector_query_cache_sha256": "detector",
        "candidate_spatial_likelihood_sha256": ["spatial-a", "spatial-b"],
        "candidate_spatial_likelihood_metadata_sha256": [
            "spatial-metadata-a",
            "spatial-metadata-b",
        ],
    }


def _rgb_topk_strict_contract(top_k: int | None = None) -> dict[str, object]:
    contract: dict[str, object] = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "verification_point_selector_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
        "candidate_specific_rgb_spatial_modes": True,
        "candidate_spatial_dustbin_and_missing_pose_independent": True,
        "candidate_spatial_omitted_topk_mass_is_null": True,
        "candidate_spatial_semantics": (
            "per_view_normalized_continuous_gaussian_mixture_relative_to_"
            "grid_uniform_null_v1"
        ),
    }
    if top_k is not None:
        contract["fixed_candidate_topk_ablation"] = {
            "applied": True,
            "top_k": top_k,
            "ranking": (
                "frozen_overlay_probability_descending_stable_candidate_column_v1"
            ),
            "removed_candidate_mass_transferred_to_null": True,
        }
    return contract


def _fixed_topk_metadata(top_k: int) -> dict[str, object]:
    return {
        "applied": True,
        "top_k": top_k,
        "candidate_column_count": 20,
        "ranking": (
            "frozen_overlay_probability_descending_stable_candidate_column_v1"
        ),
        "removed_candidate_mass_transferred_to_null": True,
        "removed_candidate_mass": {"mean": 0.0, "maximum": 0.0},
        "retained_candidate_slot_count": 80,
    }


def test_candidate_topk_pair_allows_only_fixed_posterior_truncation(tmp_path) -> None:
    input_hashes = _rgb_topk_input_hashes()
    common = {
        "config": {
            "candidate_mode": "fixed_global_topl",
            "spatial_sigma_px": 3.0,
            "candidate_spatial_semantics": "normalized",
        },
        "strict_absolute_evidence_contract": _rgb_topk_strict_contract(20),
        "input_hashes": input_hashes,
        "fixed_candidate_topk_ablation": _fixed_topk_metadata(20),
    }
    baseline = _write_evaluation(tmp_path, "baseline", _rows(), **common)
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05, rank_offset=-1),
        **{
            **common,
            "strict_absolute_evidence_contract": _rgb_topk_strict_contract(5),
            "fixed_candidate_topk_ablation": _fixed_topk_metadata(5),
        },
    )

    result = evaluate_frozen_pose_selection_pair(
        baseline_summary_path=baseline,
        probe_summary_path=probe,
        comparison_mode="candidate_topk",
    )

    assert result["stage"] == "frozen_candidate_topk_pose_selection_pair_audit"
    assert result["splits"]["validation"]["pass"] is True
    assert result["splits"]["test"]["pass"] is True


def test_candidate_topk_pair_rejects_changed_rgb_spatial_artifact(tmp_path) -> None:
    input_hashes = _rgb_topk_input_hashes()
    common = {
        "config": {
            "candidate_mode": "fixed_global_topl",
            "spatial_sigma_px": 3.0,
            "candidate_spatial_semantics": "normalized",
        },
        "strict_absolute_evidence_contract": _rgb_topk_strict_contract(20),
    }
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        input_hashes=input_hashes,
        fixed_candidate_topk_ablation=_fixed_topk_metadata(20),
        **common,
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05),
        input_hashes={
            **input_hashes,
            "candidate_spatial_likelihood_sha256": ["different", "spatial-b"],
        },
        fixed_candidate_topk_ablation=_fixed_topk_metadata(5),
        **{
            **common,
            "strict_absolute_evidence_contract": _rgb_topk_strict_contract(5),
        },
    )

    with pytest.raises(ValueError, match="changes immutable RGB evidence"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline,
            probe_summary_path=probe,
            comparison_mode="candidate_topk",
        )


def test_absolute_likelihood_pair_rejects_declared_truncated_posterior(tmp_path) -> None:
    input_hashes = _rgb_topk_input_hashes()
    baseline = _write_evaluation(
        tmp_path,
        "baseline",
        _rows(),
        config={"candidate_mode": "fixed_global_topl"},
        strict_absolute_evidence_contract={
            **{
                key: value
                for key, value in _rgb_topk_strict_contract().items()
                if not str(key).startswith("candidate_spatial_")
            },
            "candidate_specific_rgb_spatial_modes": False,
        },
        input_hashes=input_hashes,
    )
    probe = _write_evaluation(
        tmp_path,
        "probe",
        _rows(translation_offset=-0.05),
        config={"candidate_mode": "fixed_global_topl"},
        strict_absolute_evidence_contract=_rgb_topk_strict_contract(),
        input_hashes=input_hashes,
        fixed_candidate_topk_ablation=_fixed_topk_metadata(5),
    )

    with pytest.raises(ValueError, match="full frozen candidate posterior"):
        evaluate_frozen_pose_selection_pair(
            baseline_summary_path=baseline,
            probe_summary_path=probe,
            comparison_mode="absolute_likelihood",
        )
