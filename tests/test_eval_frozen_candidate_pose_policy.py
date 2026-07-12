from __future__ import annotations

import pytest
import numpy as np

from feature_extract.tools.vfm.eval_frozen_candidate_pose_policy import (
    _compact_candidate_scores,
    _frozen_global_policy,
    _frozen_score_key,
    _resolve_frozen_global_scores,
    _split_query_ids,
    _validate_ground_truth_join,
)


def test_frozen_score_key_accepts_passed_unconditional_policy() -> None:
    score_key, policy = _frozen_score_key(
        {
            "validation": {
                "passes_stage_gate": True,
                "chosen": {
                    "mode": "unconditional",
                    "strategy": "geometry_p05px_prior_row_confidence",
                    "margin_threshold": None,
                    "action_margin_threshold": None,
                },
            }
        }
    )

    assert score_key == "ensemble__geometry_p05px_prior_row_confidence"
    assert policy["strategy"] == "geometry_p05px_prior_row_confidence"


def test_frozen_score_key_accepts_passed_selective_policy() -> None:
    score_key, policy = _frozen_score_key(
        {
            "validation": {
                "passes_stage_gate": True,
                "chosen": {
                    "mode": "selective",
                    "strategy": "geometry_p05px_prior_row_confidence",
                    "margin_threshold": 0.1,
                    "action_margin_threshold": None,
                },
            }
        }
    )

    assert score_key == "ensemble__geometry_p05px_prior_row_confidence"
    assert policy["margin_threshold"] == 0.1


def test_ground_truth_join_requires_identical_inference_candidates() -> None:
    inference = {
        "query_ids": np.asarray(["q"]),
        "xy": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "candidate_track_ids": np.asarray([[10, 11]], dtype=np.int64),
        "candidate_prototype_ids": np.asarray([[0, 0]], dtype=np.int64),
        "coarse_scores": np.asarray([[0.9, 0.8]], dtype=np.float32),
        "pose_keep_mask": np.asarray([True]),
    }
    ground_truth = {
        **inference,
        "nearest_visible_track_ids": np.asarray([10], dtype=np.int64),
        "nearest_visible_residuals_px": np.asarray([0.5], dtype=np.float32),
        "candidate_gt_residuals_px": np.asarray(
            [[0.5, 5.0]], dtype=np.float32
        ),
    }

    _validate_ground_truth_join(inference, ground_truth)
    changed = {**ground_truth, "candidate_track_ids": np.asarray([[12, 11]])}
    with pytest.raises(ValueError, match="identities or coordinates"):
        _validate_ground_truth_join(inference, changed)


@pytest.mark.parametrize(
    "validation,match",
    [
        (
            {
                "passes_stage_gate": False,
                "chosen": {
                    "mode": "unconditional",
                    "strategy": "geometry_p05px",
                },
            },
            "did not pass",
        ),
        (
            {
                "passes_stage_gate": True,
                "chosen": {
                    "mode": "rescue_action_margin",
                    "strategy": "rescue_policy_resolved",
                },
            },
            "unsupported frozen policy mode",
        ),
        (
            {
                "passes_stage_gate": True,
                "chosen": {
                    "mode": "unconditional",
                    "strategy": "baseline",
                },
            },
            "no learned score strategy",
        ),
    ],
)
def test_frozen_score_key_rejects_non_deployable_policies(
    validation: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _frozen_score_key({"validation": validation})


def _global_summary(**chosen_overrides: object) -> dict[str, object]:
    chosen = {
        "assignment_mode": "whole_image_sparse_bipartite_with_per_query_dustbin",
        "policy_key": "global_bipartite__ensemble__geometry_p02px__dustbin_none",
        "score_key": "ensemble__geometry_p02px",
        "dustbin_score": None,
        "pre_global_assignment_policy": "direct_score",
        "max_matches": 48,
        "selection_mode": "spatial_round_robin",
        "passes_pose_gate": True,
        "selection_fallback_without_strict_pose_gate": False,
        **chosen_overrides,
    }
    return {
        "protocol": {"policy_selected_on_validation_only": True},
        "baseline": {
            "frozen_validation_policy": {
                "policy_key": "baseline_global_max48_score_topk",
                "max_matches": 48,
                "selection_mode": "score_topk",
                "pose": {},
            }
        },
        "validation": {"chosen": chosen},
    }


def test_frozen_global_policy_accepts_complete_strict_policy() -> None:
    policy = _frozen_global_policy(_global_summary())

    assert policy["source_score_key"] == "ensemble__geometry_p02px"
    assert policy["max_matches"] == 48
    assert policy["selection_mode"] == "spatial_round_robin"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"passes_pose_gate": False}, "strict pose gate"),
        (
            {"selection_fallback_without_strict_pose_gate": True},
            "strict pose gate",
        ),
        ({"max_matches": 0}, "match budget"),
        ({"selection_mode": "confidence_magic"}, "pose selection mode"),
    ],
)
def test_frozen_global_policy_rejects_non_deployable_choice(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _frozen_global_policy(_global_summary(**overrides))


def test_frozen_global_policy_allows_only_explicit_baseline_fallback() -> None:
    summary = _global_summary(
        score_key="strategy__alike_support_top2_mean",
        passes_pose_gate=False,
        selection_fallback_without_strict_pose_gate=True,
    )
    summary["inputs"] = {
        "baseline_score_key": "strategy__alike_support_top2_mean"
    }

    policy = _frozen_global_policy(summary, allow_baseline_fallback=True)

    assert policy["baseline_only"] is True


def test_frozen_global_policy_does_not_relax_learned_fallback() -> None:
    summary = _global_summary(
        passes_pose_gate=False,
        selection_fallback_without_strict_pose_gate=True,
    )
    summary["inputs"] = {
        "baseline_score_key": "strategy__alike_support_top2_mean"
    }

    with pytest.raises(ValueError, match="fallback is learned"):
        _frozen_global_policy(summary, allow_baseline_fallback=True)


def test_resolve_frozen_global_scores_enforces_unique_tracks() -> None:
    policy = _frozen_global_policy(_global_summary())
    tracks = np.asarray([[10, 11], [10, 12]], dtype=np.int64)
    source = np.asarray([[0.90, 0.80], [0.85, 0.10]], dtype=np.float32)
    baseline = np.asarray([[0.70, 0.60], [0.65, 0.20]], dtype=np.float32)

    resolved, baseline_resolved, switched = _resolve_frozen_global_scores(
        policy=policy,
        source_scores=source,
        baseline_scores=baseline,
        valid_edges=np.ones_like(tracks, dtype=bool),
        track_ids=tracks,
        query_ids=np.asarray(["q", "q"]),
    )

    assert switched is None
    assert np.argmax(resolved, axis=1).tolist() == [1, 0]
    selected_tracks = tracks[np.arange(2), np.argmax(resolved, axis=1)]
    assert len(np.unique(selected_tracks)) == 2
    baseline_tracks = tracks[np.arange(2), np.argmax(baseline_resolved, axis=1)]
    assert len(np.unique(baseline_tracks)) == 2


def test_split_query_ids_unions_all_list_blocks(tmp_path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        '{"train": ["a", "b"], "validation": ["c"], "metadata": {}}\n'
    )

    assert _split_query_ids(split) == {"a", "b", "c"}


def test_compact_candidate_scores_accepts_full_proposal_matrix() -> None:
    full = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )

    compact = _compact_candidate_scores(
        full,
        selected_rows=np.asarray([1], dtype=np.int64),
        selected_columns=np.asarray([[2, 0]], dtype=np.int64),
        full_candidate_shape=full.shape,
    )

    assert np.allclose(compact, [[0.6, 0.4]])
