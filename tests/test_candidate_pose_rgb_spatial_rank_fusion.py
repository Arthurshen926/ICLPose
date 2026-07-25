from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.fit_candidate_pose_rgb_spatial_rank_fusion import (
    _QueryGroup,
    _oof_coverage,
    _rows_for_alpha,
    _select_alpha,
    _tail_safe_gate,
)
from feature_extract.vfm.localization.candidate_pose_rank_fusion import (
    fuse_rank_percentiles,
    selected_position,
)


def _group(
    *, query_id: str, fold_index: int, baseline_error: float, visual_error: float
) -> _QueryGroup:
    return _QueryGroup(
        split_name="train",
        evaluation_label="fixed",
        query_id=query_id,
        fold_index=fold_index,
        hypothesis_indices=np.asarray([10, 20], dtype=np.int64),
        baseline_scores=np.asarray([2.0, 1.0], dtype=np.float64),
        visual_scores=np.asarray([1.0, 2.0], dtype=np.float64),
        source_baseline_top1=np.asarray([True, False]),
        translation_errors_m=np.asarray([baseline_error, visual_error], dtype=np.float64),
        rotation_errors_deg=np.asarray([0.3, 0.2], dtype=np.float64),
    )


def test_rank_percentile_alpha_zero_reproduces_baseline_order() -> None:
    ties = np.asarray([9, 3, 5], dtype=np.int64)
    baseline = np.asarray([0.1, 0.8, 0.4], dtype=np.float64)
    visual = np.asarray([0.9, 0.2, 0.3], dtype=np.float64)
    fused = fuse_rank_percentiles(
        baseline_scores=baseline,
        visual_scores=visual,
        tie_break_orders=ties,
        alpha=0.0,
    )
    assert selected_position(fused, ties) == selected_position(baseline, ties)


def test_train_oof_alpha_selection_requires_tail_safe_improvement() -> None:
    groups = (
        _group(query_id="q0", fold_index=0, baseline_error=0.20, visual_error=0.05),
        _group(query_id="q1", fold_index=1, baseline_error=0.30, visual_error=0.04),
    )
    alpha, rows, metrics, gate, candidates = _select_alpha(
        groups, alpha_grid=(0.0, 0.5, 1.0, 2.0)
    )
    assert alpha == 2.0
    assert gate["passes"] is True
    assert metrics["paired_translation_wins"] == 2
    assert all(bool(row["selection_changed_from_baseline"]) for row in rows)
    assert len(candidates) == 4


def test_tail_gate_rejects_a_new_catastrophic_promotion() -> None:
    group = _group(query_id="q0", fold_index=0, baseline_error=0.10, visual_error=2.0)
    rows = _rows_for_alpha((group,), alpha=2.0)
    from feature_extract.tools.vfm.fit_candidate_pose_rgb_spatial_rank_fusion import _metrics

    gate = _tail_safe_gate(_metrics(rows), alpha=2.0)
    assert gate["passes"] is False
    assert gate["checks"]["no_new_catastrophic"] is False


def test_partial_oof_coverage_is_explicitly_non_promotable() -> None:
    # Build the manifest through the production helper so the test cannot rely
    # on an ad-hoc hash or a stale partition structure.
    from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
        train_query_partition_manifest,
    )

    partition = train_query_partition_manifest(
        all_query_ids=("q0", "q1", "q2", "q3"),
        inner_train_query_ids=("q1", "q3"),
        inner_validation_query_ids=("q0", "q2"),
        fold_count=2,
        fold_index=0,
    )
    metadata = ({"query_id": "q0", "model_checkpoint_contract": {"oof_train_query_partition": partition}},)
    coverage = _oof_coverage(score_metadata=metadata, scored_query_ids=("q0", "q2"))
    assert coverage["complete"] is False
    assert coverage["missing_query_ids"] == ["q1", "q3"]
