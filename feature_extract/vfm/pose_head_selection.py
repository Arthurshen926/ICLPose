"""Offline calibrated selection among fixed local pose heads.

The selector consumes per-query row JSONL outputs from already-run pose heads
and predicts which head is most likely to be a metric localization success.
It is deliberately a small calibration model over pose diagnostics; it does
not rerun matching or use GT fields at inference time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = (
    "pose_risk",
    "pnp_inlier_count",
    "pnp_inlier_ratio",
    "match_count",
    "pnp_match_count",
    "mean_similarity",
    "visible_landmark_recall",
    "bank_visibility_coverage",
    "pnp_reprojection.pnp_reproj_inlier_median_px",
    "pnp_reprojection.pnp_reproj_inlier_p90_px",
    "pnp_reprojection.pnp_reproj_median_px",
    "pnp_inlier_spatial.grid_4x4_occupancy_frac",
    "pnp_inlier_spatial.convex_hull_area_frac",
    "pnp_inlier_spatial.depth_range_m",
    "pnp_inlier_spatial.xy_pca_minor_major_ratio",
    "pnp_inlier_spatial.xyz_linearity_ratio",
    "pnp_inlier_spatial.xyz_planarity_ratio",
    "patch_offset_refinement.refined_count",
    "patch_offset_refinement.fixed_same_inlier_count",
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must be NxD")
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean[None, :]) / self.std[None, :]


@dataclass(frozen=True)
class PoseHeadSelector:
    weights: np.ndarray
    bias: float
    standardizer: Standardizer
    head_names: tuple[str, ...]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def predict_proba(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float64)
        features = selector_feature_matrix(rows, self.head_names, self.feature_columns)
        logits = self.standardizer.transform(features) @ self.weights.reshape(-1) + float(self.bias)
        return sigmoid(logits)


def finite_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def load_jsonl_rows(path: str | Path, *, head_name: str | None = None) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if head_name is not None:
            row["head_name"] = str(head_name)
        rows.append(row)
    return rows


def write_jsonl_rows(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_json(path: str | Path, data: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def query_split(query_ids: Sequence[str], *, train_fraction: float = 0.5, seed: int = 0) -> tuple[set[str], set[str]]:
    unique = np.asarray(sorted({str(query_id) for query_id in query_ids}), dtype=object)
    if unique.size == 0:
        return set(), set()
    if unique.size == 1:
        only = {str(unique[0])}
        return only, only
    rng = np.random.default_rng(int(seed))
    order = np.arange(unique.shape[0])
    rng.shuffle(order)
    split = int(round(float(train_fraction) * unique.shape[0]))
    split = min(max(split, 1), unique.shape[0] - 1)
    return {str(unique[idx]) for idx in order[:split]}, {str(unique[idx]) for idx in order[split:]}


def common_query_ids(rows_by_head: Mapping[str, Sequence[Mapping[str, object]]]) -> set[str]:
    sets = []
    for rows in rows_by_head.values():
        sets.append({str(row.get("query_id")) for row in rows if row.get("query_id") is not None})
    if not sets:
        return set()
    common = set.intersection(*sets)
    return {query_id for query_id in common if query_id and query_id != "None"}


def _nested_get(row: Mapping[str, object], path: str) -> object:
    value: object = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def feature_value(row: Mapping[str, object], column: str) -> float:
    return finite_float(_nested_get(row, column))


def base_feature_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    return np.asarray([[feature_value(row, column) for column in feature_columns] for row in rows], dtype=np.float64)


def selector_feature_matrix(
    rows: Sequence[Mapping[str, object]],
    head_names: Sequence[str],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    base = base_feature_matrix(rows, feature_columns)
    if not head_names:
        return base
    head_index = {name: idx for idx, name in enumerate(head_names)}
    one_hot = np.zeros((len(rows), len(head_names)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        idx = head_index.get(str(row.get("head_name")))
        if idx is not None:
            one_hot[row_idx, idx] = 1.0
    return np.concatenate([base, one_hot], axis=1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def success_label(
    row: Mapping[str, object],
    *,
    translation_threshold_m: float = 0.25,
    rotation_threshold_deg: float = 10.0,
) -> bool:
    translation = finite_float(row.get("translation_error_m"), default=float("inf"))
    rotation = finite_float(row.get("rotation_error_deg"), default=float("inf"))
    return bool(translation <= float(translation_threshold_m) and rotation <= float(rotation_threshold_deg))


def success_at(row: Mapping[str, object], translation_threshold_m: float, rotation_threshold_deg: float) -> bool:
    return success_label(
        row,
        translation_threshold_m=float(translation_threshold_m),
        rotation_threshold_deg=float(rotation_threshold_deg),
    )


def train_pose_head_selector(
    rows_by_head: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    train_query_ids: set[str],
    translation_threshold_m: float = 0.25,
    rotation_threshold_deg: float = 10.0,
    iterations: int = 500,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> PoseHeadSelector:
    head_names = tuple(sorted(str(name) for name in rows_by_head))
    train_rows = []
    for head_name in head_names:
        for source_row in rows_by_head[head_name]:
            if str(source_row.get("query_id")) not in train_query_ids:
                continue
            row = dict(source_row)
            row["head_name"] = str(head_name)
            train_rows.append(row)
    if not train_rows:
        raise ValueError("no training rows matched train_query_ids")

    x = selector_feature_matrix(train_rows, head_names, feature_columns)
    y = np.asarray(
        [
            1.0
            if success_label(
                row,
                translation_threshold_m=translation_threshold_m,
                rotation_threshold_deg=rotation_threshold_deg,
            )
            else 0.0
            for row in train_rows
        ],
        dtype=np.float64,
    )
    standardizer = Standardizer.fit(x)
    z = standardizer.transform(x)
    weights = np.zeros((z.shape[1],), dtype=np.float64)
    prior = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    bias = float(np.log(prior / (1.0 - prior)))
    for _ in range(int(iterations)):
        pred = sigmoid(z @ weights + bias)
        error = pred - y
        weights -= float(learning_rate) * ((z.T @ error) / max(z.shape[0], 1) + float(l2) * weights)
        bias -= float(learning_rate) * float(np.mean(error))
    return PoseHeadSelector(
        weights=weights.astype(np.float64),
        bias=float(bias),
        standardizer=standardizer,
        head_names=head_names,
        feature_columns=tuple(feature_columns),
    )


def rows_by_query(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if query_id is not None:
            out[str(query_id)] = dict(row)
    return out


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    translations = [finite_float(row.get("translation_error_m"), default=float("nan")) for row in rows]
    rotations = [finite_float(row.get("rotation_error_deg"), default=float("nan")) for row in rows]
    translations = [value for value in translations if math.isfinite(value)]
    rotations = [value for value in rotations if math.isfinite(value)]
    return {
        "query_count": int(len(rows)),
        "pnp_solve_rate": float(np.mean([bool(row.get("pnp_solve", True)) for row in rows])) if rows else 0.0,
        "success_10cm_5deg": float(np.mean([success_at(row, 0.10, 5.0) for row in rows])) if rows else 0.0,
        "success_25cm_10deg": float(np.mean([success_at(row, 0.25, 10.0) for row in rows])) if rows else 0.0,
        "success_50cm_10deg": float(np.mean([success_at(row, 0.50, 10.0) for row in rows])) if rows else 0.0,
        "success_1m_10deg": float(np.mean([success_at(row, 1.00, 10.0) for row in rows])) if rows else 0.0,
        "median_translation_error_m": _median(translations),
        "median_rotation_error_deg": _median(rotations),
        "mean_pnp_inlier_count": _mean_nested(rows, "pnp_inlier_count"),
        "mean_pnp_inlier_patch_at_1": _mean_nested(rows, "patch_geometry.pnp_inlier_patch_at_1"),
        "mean_pnp_reproj_inlier_median_px": _mean_nested(
            rows, "pnp_reprojection.pnp_reproj_inlier_median_px"
        ),
    }


def _mean_nested(rows: Sequence[Mapping[str, object]], path: str) -> float | None:
    values = [finite_float(_nested_get(row, path), default=float("nan")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _best_oracle_row(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("oracle row selection requires at least one row")
    return dict(
        min(
            rows,
            key=lambda row: (
                not success_at(row, 0.25, 10.0),
                finite_float(row.get("translation_error_m"), default=float("inf")),
                finite_float(row.get("rotation_error_deg"), default=float("inf")),
            ),
        )
    )


def _select_query_row(
    query_rows: Sequence[Mapping[str, object]],
    *,
    model: PoseHeadSelector,
) -> dict[str, object]:
    probs = model.predict_proba(query_rows)
    best_idx = int(np.argmax(probs))
    selected = dict(query_rows[best_idx])
    selected["selector_success_probability"] = float(probs[best_idx])
    return selected


def evaluate_pose_head_selection(
    rows_by_head: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    query_ids: set[str] | None = None,
    model: PoseHeadSelector,
    baseline_head: str | None = None,
) -> dict[str, object]:
    if query_ids is None:
        query_ids = common_query_ids(rows_by_head)
    query_ids = {str(query_id) for query_id in query_ids}
    by_head_query = {str(name): rows_by_query(rows) for name, rows in rows_by_head.items()}
    selected_rows = []
    oracle_rows = []
    per_query = []
    for query_id in sorted(query_ids):
        candidates = []
        for head_name in sorted(by_head_query):
            row = by_head_query[head_name].get(query_id)
            if row is None:
                continue
            candidate = dict(row)
            candidate["head_name"] = str(head_name)
            candidates.append(candidate)
        if not candidates:
            continue
        selected = _select_query_row(candidates, model=model)
        oracle = _best_oracle_row(candidates)
        selected_rows.append(selected)
        oracle_rows.append(oracle)
        per_query.append(
            {
                "query_id": query_id,
                "selected_head": selected.get("head_name"),
                "selected_probability": selected.get("selector_success_probability"),
                "selected_translation_error_m": selected.get("translation_error_m"),
                "selected_rotation_error_deg": selected.get("rotation_error_deg"),
                "selected_success_25cm_10deg": success_at(selected, 0.25, 10.0),
                "oracle_head": oracle.get("head_name"),
                "oracle_translation_error_m": oracle.get("translation_error_m"),
                "oracle_rotation_error_deg": oracle.get("rotation_error_deg"),
            }
        )

    summary = _metric_summary(selected_rows)
    summary.update(
        {
            "selected_head_counts": dict(sorted(Counter(str(row.get("head_name")) for row in selected_rows).items())),
            "oracle": _metric_summary(oracle_rows),
            "per_head": {
                str(head_name): _metric_summary(
                    [dict(row, head_name=str(head_name)) for query_id, row in sorted(rows_by_query(rows).items()) if query_id in query_ids]
                )
                for head_name, rows in sorted(rows_by_head.items())
            },
            "rows": per_query,
        }
    )

    if baseline_head is not None:
        baseline_map = by_head_query.get(str(baseline_head), {})
        rescued = 0
        broken = 0
        comparable = 0
        for selected in selected_rows:
            query_id = str(selected.get("query_id"))
            baseline = baseline_map.get(query_id)
            if baseline is None:
                continue
            comparable += 1
            baseline_ok = success_at(baseline, 0.25, 10.0)
            selected_ok = success_at(selected, 0.25, 10.0)
            if selected_ok and not baseline_ok:
                rescued += 1
            elif baseline_ok and not selected_ok:
                broken += 1
        summary["rescue_break_vs_baseline"] = {
            "baseline_head": str(baseline_head),
            "comparable_query_count": int(comparable),
            "rescued": int(rescued),
            "broken": int(broken),
            "ratio": None if broken == 0 else float(rescued / broken),
        }
    return summary


def guarded_pairwise_pose_head_selection(
    rows_by_head: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    baseline_head: str,
    candidate_head: str,
    query_ids: set[str] | None = None,
    min_reproj_improvement_px: float = 0.0,
    max_inlier_count_drop: int = 0,
) -> dict[str, object]:
    """Select a candidate pose head only when no-GT pose diagnostics improve."""

    baseline_head = str(baseline_head)
    candidate_head = str(candidate_head)
    if baseline_head not in rows_by_head:
        raise ValueError(f"baseline_head not found: {baseline_head}")
    if candidate_head not in rows_by_head:
        raise ValueError(f"candidate_head not found: {candidate_head}")
    if query_ids is None:
        query_ids = common_query_ids(rows_by_head)
    query_ids = {str(query_id) for query_id in query_ids}
    by_head_query = {str(name): rows_by_query(rows) for name, rows in rows_by_head.items()}
    selected_rows = []
    oracle_rows = []
    per_query = []
    for query_id in sorted(query_ids):
        baseline = by_head_query[baseline_head].get(query_id)
        candidate = by_head_query[candidate_head].get(query_id)
        if baseline is None or candidate is None:
            continue
        base_residual = feature_value(baseline, "pnp_reprojection.pnp_reproj_inlier_median_px")
        cand_residual = feature_value(candidate, "pnp_reprojection.pnp_reproj_inlier_median_px")
        base_inliers = feature_value(baseline, "pnp_inlier_count")
        cand_inliers = feature_value(candidate, "pnp_inlier_count")
        choose_candidate = bool(
            cand_residual <= base_residual - float(min_reproj_improvement_px)
            and cand_inliers >= base_inliers - float(max_inlier_count_drop)
        )
        selected = dict(candidate if choose_candidate else baseline)
        selected["head_name"] = candidate_head if choose_candidate else baseline_head
        selected["guarded_selection_reason"] = "candidate_lower_residual" if choose_candidate else "baseline_guard"
        selected_rows.append(selected)
        candidates = [dict(baseline, head_name=baseline_head), dict(candidate, head_name=candidate_head)]
        oracle = _best_oracle_row(candidates)
        oracle_rows.append(oracle)
        per_query.append(
            {
                "query_id": query_id,
                "selected_head": selected.get("head_name"),
                "selected_translation_error_m": selected.get("translation_error_m"),
                "selected_rotation_error_deg": selected.get("rotation_error_deg"),
                "selected_success_25cm_10deg": success_at(selected, 0.25, 10.0),
                "baseline_reproj_inlier_median_px": base_residual,
                "candidate_reproj_inlier_median_px": cand_residual,
                "baseline_inlier_count": base_inliers,
                "candidate_inlier_count": cand_inliers,
                "oracle_head": oracle.get("head_name"),
                "oracle_translation_error_m": oracle.get("translation_error_m"),
                "oracle_rotation_error_deg": oracle.get("rotation_error_deg"),
            }
        )

    summary = _metric_summary(selected_rows)
    summary.update(
        {
            "baseline_head": baseline_head,
            "candidate_head": candidate_head,
            "guard": {
                "min_reproj_improvement_px": float(min_reproj_improvement_px),
                "max_inlier_count_drop": int(max_inlier_count_drop),
            },
            "selected_head_counts": dict(sorted(Counter(str(row.get("head_name")) for row in selected_rows).items())),
            "oracle": _metric_summary(oracle_rows),
            "per_head": {
                str(head_name): _metric_summary(
                    [dict(row, head_name=str(head_name)) for query_id, row in sorted(rows_by_query(rows).items()) if query_id in query_ids]
                )
                for head_name, rows in sorted(rows_by_head.items())
            },
            "rows": per_query,
        }
    )
    baseline_map = by_head_query[baseline_head]
    rescued = 0
    broken = 0
    comparable = 0
    for selected in selected_rows:
        query_id = str(selected.get("query_id"))
        baseline = baseline_map.get(query_id)
        if baseline is None:
            continue
        comparable += 1
        baseline_ok = success_at(baseline, 0.25, 10.0)
        selected_ok = success_at(selected, 0.25, 10.0)
        if selected_ok and not baseline_ok:
            rescued += 1
        elif baseline_ok and not selected_ok:
            broken += 1
    summary["rescue_break_vs_baseline"] = {
        "baseline_head": baseline_head,
        "comparable_query_count": int(comparable),
        "rescued": int(rescued),
        "broken": int(broken),
        "ratio": None if broken == 0 else float(rescued / broken),
    }
    return summary


def guarded_multi_pose_head_selection(
    rows_by_head: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    baseline_head: str,
    candidate_heads: Sequence[str],
    query_ids: set[str] | None = None,
    min_reproj_improvement_px: float = 0.0,
    max_inlier_count_drop: int = 0,
) -> dict[str, object]:
    """Select the lowest-residual safe pose among several fixed heads."""

    baseline_head = str(baseline_head)
    candidate_heads = tuple(str(head) for head in candidate_heads)
    if baseline_head not in rows_by_head:
        raise ValueError(f"baseline_head not found: {baseline_head}")
    missing = [head for head in candidate_heads if head not in rows_by_head]
    if missing:
        raise ValueError(f"candidate_heads not found: {', '.join(missing)}")
    if query_ids is None:
        query_ids = common_query_ids(rows_by_head)
    query_ids = {str(query_id) for query_id in query_ids}
    by_head_query = {str(name): rows_by_query(rows) for name, rows in rows_by_head.items()}
    selected_rows = []
    oracle_rows = []
    per_query = []
    considered_heads = (baseline_head,) + candidate_heads
    for query_id in sorted(query_ids):
        baseline = by_head_query[baseline_head].get(query_id)
        if baseline is None:
            continue
        base_residual = feature_value(baseline, "pnp_reprojection.pnp_reproj_inlier_median_px")
        base_inliers = feature_value(baseline, "pnp_inlier_count")
        candidates = [dict(baseline, head_name=baseline_head)]
        safe_candidates = [dict(baseline, head_name=baseline_head)]
        for head_name in candidate_heads:
            row = by_head_query[head_name].get(query_id)
            if row is None:
                continue
            candidate = dict(row, head_name=head_name)
            candidates.append(candidate)
            residual = feature_value(row, "pnp_reprojection.pnp_reproj_inlier_median_px")
            inliers = feature_value(row, "pnp_inlier_count")
            if (
                residual <= base_residual - float(min_reproj_improvement_px)
                and inliers >= base_inliers - float(max_inlier_count_drop)
            ):
                safe_candidates.append(candidate)
        selected = dict(
            min(
                safe_candidates,
                key=lambda row: (
                    feature_value(row, "pnp_reprojection.pnp_reproj_inlier_median_px"),
                    -feature_value(row, "pnp_inlier_count"),
                    str(row.get("head_name")),
                ),
            )
        )
        selected["guarded_selection_reason"] = (
            "lowest_safe_residual" if selected.get("head_name") != baseline_head else "baseline_guard"
        )
        oracle = _best_oracle_row(candidates)
        selected_rows.append(selected)
        oracle_rows.append(oracle)
        per_query.append(
            {
                "query_id": query_id,
                "selected_head": selected.get("head_name"),
                "selected_translation_error_m": selected.get("translation_error_m"),
                "selected_rotation_error_deg": selected.get("rotation_error_deg"),
                "selected_success_25cm_10deg": success_at(selected, 0.25, 10.0),
                "baseline_reproj_inlier_median_px": base_residual,
                "baseline_inlier_count": base_inliers,
                "safe_candidate_count": int(len(safe_candidates)),
                "oracle_head": oracle.get("head_name"),
                "oracle_translation_error_m": oracle.get("translation_error_m"),
                "oracle_rotation_error_deg": oracle.get("rotation_error_deg"),
            }
        )

    summary = _metric_summary(selected_rows)
    summary.update(
        {
            "baseline_head": baseline_head,
            "candidate_heads": list(candidate_heads),
            "guard": {
                "min_reproj_improvement_px": float(min_reproj_improvement_px),
                "max_inlier_count_drop": int(max_inlier_count_drop),
            },
            "selected_head_counts": dict(sorted(Counter(str(row.get("head_name")) for row in selected_rows).items())),
            "oracle": _metric_summary(oracle_rows),
            "per_head": {
                str(head_name): _metric_summary(
                    [
                        dict(row, head_name=str(head_name))
                        for query_id, row in sorted(rows_by_query(rows_by_head[head_name]).items())
                        if query_id in query_ids
                    ]
                )
                for head_name in considered_heads
                if head_name in rows_by_head
            },
            "rows": per_query,
        }
    )
    baseline_map = by_head_query[baseline_head]
    rescued = 0
    broken = 0
    comparable = 0
    for selected in selected_rows:
        query_id = str(selected.get("query_id"))
        baseline = baseline_map.get(query_id)
        if baseline is None:
            continue
        comparable += 1
        baseline_ok = success_at(baseline, 0.25, 10.0)
        selected_ok = success_at(selected, 0.25, 10.0)
        if selected_ok and not baseline_ok:
            rescued += 1
        elif baseline_ok and not selected_ok:
            broken += 1
    summary["rescue_break_vs_baseline"] = {
        "baseline_head": baseline_head,
        "comparable_query_count": int(comparable),
        "rescued": int(rescued),
        "broken": int(broken),
        "ratio": None if broken == 0 else float(rescued / broken),
    }
    return summary


def selector_training_diagnostics(
    rows_by_head: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    query_ids: set[str],
    model: PoseHeadSelector,
) -> dict[str, object]:
    rows = []
    labels = []
    for head_name, source_rows in rows_by_head.items():
        for source_row in source_rows:
            if str(source_row.get("query_id")) not in query_ids:
                continue
            row = dict(source_row)
            row["head_name"] = str(head_name)
            rows.append(row)
            labels.append(1.0 if success_at(row, 0.25, 10.0) else 0.0)
    probs = model.predict_proba(rows)
    labels_array = np.asarray(labels, dtype=np.float64)
    return {
        "row_count": int(len(rows)),
        "success_prior": float(np.mean(labels_array)) if labels else 0.0,
        "auc": roc_auc(probs, labels_array),
        "precision_top20": precision_at_fraction(probs, labels_array, 0.20),
        "precision_top40": precision_at_fraction(probs, labels_array, 0.40),
    }


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    positive = scores_array[labels_array >= 0.5]
    negative = scores_array[labels_array < 0.5]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    wins = 0.0
    total = float(positive.size * negative.size)
    for value in positive:
        wins += float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
    return float(wins / total)


def precision_at_fraction(scores: Sequence[float], labels: Sequence[float], fraction: float) -> float:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    if scores_array.size == 0:
        return float("nan")
    count = max(1, int(math.ceil(float(fraction) * scores_array.size)))
    order = np.argsort(-scores_array)[:count]
    return float(np.mean(labels_array[order] >= 0.5))
