from __future__ import annotations

from feature_extract.vfm.localization.immutable_pose_results import (
    BaselinePoseResult,
    OptionalPoseResult,
    select_optional_or_fallback,
)


def _baseline() -> BaselinePoseResult:
    return BaselinePoseResult.create(
        pose_w2c=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        source_policy="L97",
        match_source_query_rows=[1, 2],
        match_track_ids=[101, 102],
        match_xy=[[10.0, 20.0], [30.0, 40.0]],
        fit_evidence={"inliers": 2.0},
        verification_evidence={"score": 0.7},
        score_schema_version="absolute_pose_evidence_v1",
        artifact_hash="0123456789abcdef",
    )


def _optional(baseline: BaselinePoseResult, hypothesis_id: str) -> OptionalPoseResult:
    return OptionalPoseResult.create(
        parent_baseline_hash=baseline.digest,
        hypothesis_id=hypothesis_id,
        pose_w2c=baseline.pose_w2c,
        absolute_evidence={"score": 0.1},
        fixed_difference_evidence={"score_delta": -0.6},
        artifact_hash="fedcba9876543210",
    )


def test_garbage_optional_pool_cannot_change_fallback_bytes() -> None:
    baseline = _baseline()
    before = baseline.serialized
    hypotheses = [_optional(baseline, f"garbage-{index}") for index in range(20)]
    result = select_optional_or_fallback(
        baseline,
        hypotheses,
        {hypothesis.hypothesis_id: 0.01 for hypothesis in hypotheses},
        promotion_threshold=0.9,
    )
    assert result is baseline
    assert baseline.serialized == before
    assert baseline.digest == _baseline().digest


def test_optional_with_wrong_parent_is_rejected() -> None:
    baseline = _baseline()
    hypothesis = OptionalPoseResult.create(
        parent_baseline_hash="0" * 64,
        hypothesis_id="wrong-parent",
        pose_w2c=baseline.pose_w2c,
        absolute_evidence={},
        fixed_difference_evidence={},
        artifact_hash="fedcba9876543210",
    )
    try:
        select_optional_or_fallback(
            baseline,
            [hypothesis],
            {"wrong-parent": 1.0},
            promotion_threshold=0.9,
        )
    except ValueError as exc:
        assert "different baseline" in str(exc)
    else:
        raise AssertionError("wrong-parent optional hypothesis should fail")
