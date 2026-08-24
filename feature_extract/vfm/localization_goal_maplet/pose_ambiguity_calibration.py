"""Risk-controlled ambiguity gate for set-valued Goal-Maplet pose output.

The pose backend may retain several physically distinct basins.  A single
pose is emitted only when the winning basin is separated from the runner-up
by more than every observed validation error.  This is a finite-validation
control, not a probabilistic correctness certificate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def calibrate_zero_false_accept_margin(
    rows: Sequence[Mapping[str, object]],
    *,
    success_key: str = "final_loose_1m_10deg",
    margin_key: str = "final_distinct_score_margin",
) -> float:
    """Return the smallest strict threshold rejecting all validation errors."""

    margins: list[float] = []
    failures: list[float] = []
    for row in rows:
        if success_key not in row or margin_key not in row:
            raise ValueError("ambiguity calibration row lacks outcome or margin")
        success = row[success_key]
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("ambiguity calibration outcome must be boolean")
        margin = float(row[margin_key])
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("ambiguity calibration margin must be finite and nonnegative")
        margins.append(margin)
        if not bool(success):
            failures.append(margin)
    if not margins or not failures:
        raise ValueError("ambiguity calibration requires both rows and observed errors")
    return float(max(failures))


def evaluate_ambiguity_gate(
    rows: Sequence[Mapping[str, object]],
    threshold: float,
    *,
    success_key: str = "final_loose_1m_10deg",
    strict_key: str = "final_strict_0_5m_5deg",
    set_success_key: str = "final_any_loose_1m_10deg",
    margin_key: str = "final_distinct_score_margin",
) -> dict[str, object]:
    """Evaluate selective single-pose output and unresolved set retention."""

    limit = float(threshold)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("ambiguity threshold must be finite and nonnegative")
    decisions = []
    for row in rows:
        for key in (success_key, strict_key, set_success_key, margin_key):
            if key not in row:
                raise ValueError("ambiguity evaluation row is incomplete")
        margin = float(row[margin_key])
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("ambiguity evaluation margin must be finite and nonnegative")
        accepted = bool(margin > limit)
        decisions.append({
            "image_id": str(row.get("image_id", "")),
            "score_margin": margin,
            "single_pose_accepted": accepted,
            "single_pose_loose_correct": bool(row[success_key]),
            "single_pose_strict_correct": bool(row[strict_key]),
            "retained_set_contains_loose_basin": bool(row[set_success_key]),
        })
    if not decisions:
        raise ValueError("ambiguity evaluation is empty")
    accepted = [row for row in decisions if row["single_pose_accepted"]]
    unresolved = [row for row in decisions if not row["single_pose_accepted"]]
    return {
        "query_count": len(decisions),
        "accepted_single_pose_count": len(accepted),
        "unresolved_multibasin_count": len(unresolved),
        "single_pose_coverage": len(accepted) / len(decisions),
        "accepted_loose_accuracy": (
            sum(row["single_pose_loose_correct"] for row in accepted) / len(accepted)
            if accepted else None
        ),
        "accepted_strict_accuracy": (
            sum(row["single_pose_strict_correct"] for row in accepted) / len(accepted)
            if accepted else None
        ),
        "accepted_loose_error_count": sum(
            not row["single_pose_loose_correct"] for row in accepted
        ),
        "unresolved_set_loose_retention_rate": (
            sum(row["retained_set_contains_loose_basin"] for row in unresolved)
            / len(unresolved) if unresolved else None
        ),
        "rows": decisions,
    }
