"""Trajectory-disjoint calibration for V6 Stage-C pose-mode ranking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


FORMAT = "v6_stage_c_pose_score_calibration_v2"
FEATURE_NAMES = (
    "feature_atlas_score",
    "final_fit_score",
    "final_heldout_score",
    "atlas_prerank_local_alignment_score",
    "coarse_score",
)


def _finite_value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    return (
        float(value)
        if value is not None and np.isfinite(float(value))
        else float("-inf")
    )


def candidate_rank_features(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Return tie-aware within-query percentiles for comparable evidence.

    Absolute RADIO likelihood scales vary with query visibility and chart
    support.  Only ordering within the fixed candidate set is calibrated;
    this prevents a high-evidence query from determining the scale learned
    for another query.
    """

    count = len(rows)
    result = np.zeros((count, len(feature_names)), dtype=np.float64)
    if count <= 1:
        return result
    for column, key in enumerate(feature_names):
        values = np.asarray(
            [_finite_value(row, str(key)) for row in rows],
            dtype=np.float64,
        )
        finite = np.isfinite(values)
        result[~finite, column] = 0.0
        if not np.any(finite):
            continue
        finite_values = values[finite]
        order = np.argsort(finite_values, kind="mergesort")
        sorted_values = finite_values[order]
        ranks = np.empty(finite_values.size, dtype=np.float64)
        start = 0
        while start < sorted_values.size:
            stop = start + 1
            while (
                stop < sorted_values.size
                and sorted_values[stop] == sorted_values[start]
            ):
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        result[finite, column] = ranks / max(count - 1, 1)
    return result


@dataclass(frozen=True)
class StageCScoreCalibration:
    feature_names: tuple[str, ...]
    weights: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.feature_names)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if names != FEATURE_NAMES:
            raise ValueError("Stage-C score feature contract differs")
        if weights.shape != (len(names),):
            raise ValueError("Stage-C score weight dimension differs")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Stage-C score weights must be finite/non-negative")
        if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-6):
            raise ValueError("Stage-C score weights must sum to one")
        training = {
            str(value)
            for value in self.metadata.get("calibration_trajectory_ids", ())
        }
        holdout = {
            str(value)
            for value in self.metadata.get("strict_holdout_trajectory_ids", ())
        }
        if not training or not holdout or training & holdout:
            raise ValueError("Stage-C score calibration split is invalid")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "weights", weights)

    def scores(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        features = candidate_rank_features(rows, self.feature_names)
        return features @ self.weights

    def apply(
        self, rows: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        scores = self.scores(rows)
        result = []
        for row, score in zip(rows, scores.tolist()):
            result.append(
                {
                    **dict(row),
                    "calibrated_pose_score": float(score),
                }
            )
        result.sort(
            key=lambda row: (
                -float(row["calibrated_pose_score"]),
                -_finite_value(row, "feature_atlas_score"),
                int(row.get("replay_pool_index", -1)),
            )
        )
        return result

    def validate_report_lineage(self, report: Mapping[str, object]) -> None:
        expected = str(self.metadata.get("radio_atlas_sha256", ""))
        observed = str(report.get("radio_atlas_sha256", ""))
        if not expected or expected != observed:
            raise ValueError("Stage-C score calibration atlas lineage differs")
        trajectory = str(report.get("image_id", "")).split("/", 1)[0]
        training = {
            str(value)
            for value in self.metadata.get("calibration_trajectory_ids", ())
        }
        if trajectory in training:
            raise ValueError(
                "deployment query overlaps Stage-C score calibration"
            )

    def to_json(self, path: Path) -> None:
        payload = {
            "format": FORMAT,
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "metadata": dict(self.metadata),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> "StageCScoreCalibration":
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != FORMAT:
            raise ValueError("invalid Stage-C score calibration format")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            metadata=dict(payload["metadata"]),
        )
