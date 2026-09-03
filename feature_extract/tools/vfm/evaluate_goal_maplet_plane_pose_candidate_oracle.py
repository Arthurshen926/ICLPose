"""Post-label candidate-union oracle for plane-pose branch diagnosis only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


THRESHOLDS = (
    ("0.1m_1deg", 0.1, 1.0),
    ("0.25m_2deg", 0.25, 2.0),
    ("0.5m_5deg", 0.5, 5.0),
    ("1m_10deg", 1.0, 10.0),
    ("2m_45deg", 2.0, 45.0),
)


def _rows(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    report = json.loads(path.read_text())
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pose evaluation lacks rows")
    keyed: dict[str, dict[str, object]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in keyed:
            raise ValueError("pose evaluation names are empty or duplicated")
        keyed[name] = row
    return keyed, report


def evaluate_candidate_oracle(
    candidate_rows: list[dict[str, dict[str, object]]],
    candidate_labels: list[str],
    selected_rows: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    if len(candidate_rows) < 2 or len(candidate_rows) != len(candidate_labels):
        raise ValueError("candidate oracle requires at least two labeled branches")
    names = sorted(candidate_rows[0])
    if any(sorted(rows) != names for rows in candidate_rows[1:]):
        raise ValueError("candidate evaluation inventories differ")
    if selected_rows is not None and sorted(selected_rows) != names:
        raise ValueError("selected evaluation inventory differs")

    translation = np.full((len(names), len(candidate_rows)), np.inf, np.float64)
    rotation = np.full_like(translation, np.inf)
    usable = np.zeros_like(translation, bool)
    for column, rows in enumerate(candidate_rows):
        for index, name in enumerate(names):
            row = rows[name]
            valid = bool(row.get("usable", False))
            t = float(row.get("translation_error_m", np.inf))
            r = float(row.get("rotation_error_deg", np.inf))
            usable[index, column] = valid and np.isfinite(t) and np.isfinite(r)
            if usable[index, column]:
                translation[index, column] = t
                rotation[index, column] = r

    # Frozen scalar only chooses one physically attainable branch per query.
    composite = (translation / 0.25) ** 2 + (rotation / 2.0) ** 2
    composite[~usable] = np.inf
    choice = np.argmin(composite, axis=1)
    oracle_usable = np.isfinite(composite[np.arange(len(names)), choice])
    oracle_t = translation[np.arange(len(names)), choice]
    oracle_r = rotation[np.arange(len(names)), choice]
    threshold_hits = {}
    threshold_misses_by_selector = {}
    for key, t_limit, r_limit in THRESHOLDS:
        any_good = np.any(usable & (translation <= t_limit) & (rotation <= r_limit), axis=1)
        threshold_hits[key] = int(np.count_nonzero(any_good))
        if selected_rows is not None:
            selected_good = np.asarray([
                bool(selected_rows[name].get("usable", False))
                and float(selected_rows[name].get("translation_error_m", np.inf)) <= t_limit
                and float(selected_rows[name].get("rotation_error_deg", np.inf)) <= r_limit
                for name in names
            ])
            threshold_misses_by_selector[key] = int(np.count_nonzero(any_good & ~selected_good))
    return {
        "query_count": len(names),
        "candidate_labels": candidate_labels,
        "candidate_usable_counts": {
            label: int(np.count_nonzero(usable[:, index]))
            for index, label in enumerate(candidate_labels)
        },
        "threshold_union_hit_counts": threshold_hits,
        "selected_threshold_recoverable_miss_counts": threshold_misses_by_selector,
        "composite_oracle_semantics": "argmin((translation_m/0.25)^2+(rotation_deg/2)^2)",
        "composite_oracle_usable_count": int(np.count_nonzero(oracle_usable)),
        "composite_oracle_median_translation_m": float(np.median(oracle_t[oracle_usable])),
        "composite_oracle_median_rotation_deg": float(np.median(oracle_r[oracle_usable])),
        "composite_oracle_branch_counts": {
            label: int(np.count_nonzero(oracle_usable & (choice == index)))
            for index, label in enumerate(candidate_labels)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evaluation", type=Path, action="append", required=True)
    parser.add_argument("--candidate_label", action="append", required=True)
    parser.add_argument("--selected_evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite candidate oracle")
    candidate = [_rows(path)[0] for path in args.candidate_evaluation]
    selected = _rows(args.selected_evaluation)[0] if args.selected_evaluation else None
    report = {
        "artifact_type": "goal_maplet_plane_pose_candidate_union_postlabel_oracle_v1",
        "evaluation_role": "POSTLABEL_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "query_pose_or_ground_truth_read": True,
        "selection_or_training_eligible": False,
        "candidate_evaluation_file_sha256_in_order": [
            file_sha256(path) for path in args.candidate_evaluation
        ],
        "selected_evaluation_file_sha256": (
            file_sha256(args.selected_evaluation) if args.selected_evaluation else None
        ),
        **evaluate_candidate_oracle(candidate, args.candidate_label, selected),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
