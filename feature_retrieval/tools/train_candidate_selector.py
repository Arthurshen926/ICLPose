from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from feature_retrieval.tools.sweep_loftr_render_selectors import parse_list_cell


DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "hyp_candidate_scores",
    "hyp_candidate_quality_scores",
    "hyp_consensus_support",
    "hyp_retrieval_original_scores_candidates",
    "hyp_retrieval_pnp_success_candidates",
    "hyp_retrieval_pnp_num_inliers_candidates",
    "hyp_retrieval_pnp_num_matches_candidates",
    "hyp_retrieval_pnp_reproj_rmse_candidates",
    "hyp_retrieval_pnp_reproj_median_candidates",
    "hyp_retrieval_pnp_inlier_ratio_candidates",
    "hyp_retrieval_pnp_inlier_conf_mean_candidates",
    "hyp_loftr_render_scores",
    "hyp_loftr_render_inliers",
    "hyp_loftr_render_matches",
    "hyp_loftr_render_raw_matches",
    "hyp_loftr_render_depth_valid",
    "hyp_loftr_render_mean_conf",
    "hyp_loftr_render_consistency_m",
    "hyp_loftr_render_reproj_rmse",
    "hyp_loftr_render_reproj_median",
    "hyp_loftr_render_inlier_ratio",
    "hyp_loftr_render_inlier_conf_mean",
)

_LEAKY_FEATURE_MARKERS = (
    "_err",
    "error",
    "oracle",
    "ground_truth",
    "gt_",
)


@dataclass(frozen=True)
class CandidateDataset:
    features: np.ndarray
    feature_names: List[str]
    sample_ids: np.ndarray
    candidate_indices: np.ndarray
    rot_err_deg: np.ndarray
    trans_err_mm: np.ndarray
    targets: np.ndarray
    image_names: List[str]
    best_indices: List[int]


@dataclass(frozen=True)
class SelectorSummary:
    name: str
    num_samples: int
    num_selected: int
    median_rot_deg: float
    median_trans_mm: float
    mean_rot_deg: float
    mean_trans_mm: float
    success_pose_pct: float
    success_trans_pct: float
    selected_indices: List[int]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return [dict(row) for row in reader]


def _rows_by_image(rows: Sequence[Dict[str, str]], path: Path) -> Dict[str, Dict[str, str]]:
    by_image: Dict[str, Dict[str, str]] = {}
    for row in rows:
        image_name = row.get("image_name", "")
        if not image_name:
            raise ValueError(f"{path} contains a row without image_name")
        if image_name in by_image:
            raise ValueError(f"{path} contains duplicate image_name {image_name!r}")
        by_image[image_name] = row
    return by_image


def _is_leaky_feature_column(column: str) -> bool:
    lowered = column.lower()
    return any(marker in lowered for marker in _LEAKY_FEATURE_MARKERS)


def _array_from_row(row: Dict[str, str], column: str, length: int, invalid_value: float = float("nan")) -> np.ndarray:
    values = parse_list_cell(row.get(column), invalid_value=invalid_value)
    if len(values) < length:
        values = values + [invalid_value] * (length - len(values))
    elif len(values) > length:
        values = values[:length]
    return np.asarray(values, dtype=np.float64)


def _normalize_row(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(finite):
        return out
    finite_values = values[finite]
    lo = float(np.min(finite_values))
    hi = float(np.max(finite_values))
    if hi <= lo:
        out[finite] = 1.0
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    if invert:
        out[finite] = 1.0 - out[finite]
    return out


def _rank_features(length: int) -> Tuple[np.ndarray, List[str]]:
    rank = np.arange(length, dtype=np.float64)
    if length <= 1:
        rank_norm = np.zeros(length, dtype=np.float64)
    else:
        rank_norm = rank / float(length - 1)
    features = np.stack(
        [
            rank,
            rank_norm,
            1.0 / (1.0 + rank),
            (rank == 0.0).astype(np.float64),
        ],
        axis=1,
    )
    return features, ["rank", "rank_norm", "rank_inv", "is_top1"]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _target_values(rot: np.ndarray, trans: np.ndarray, target_mode: str, pose_rot_weight: float) -> np.ndarray:
    if target_mode == "trans":
        return np.log1p(np.maximum(trans, 0.0))
    if target_mode == "pose":
        return np.log1p(np.maximum(trans, 0.0)) + pose_rot_weight * np.log1p(np.maximum(rot, 0.0) * 100.0)
    raise ValueError(f"unknown target_mode {target_mode!r}")


def _best_index(rot: np.ndarray, trans: np.ndarray, target: np.ndarray) -> int:
    valid = np.isfinite(rot) & np.isfinite(trans) & np.isfinite(target)
    if not np.any(valid):
        return -1
    valid_indices = np.flatnonzero(valid)
    order = np.lexsort((rot[valid], target[valid]))
    return int(valid_indices[int(order[0])])


def _candidate_block(
    row: Dict[str, str],
    feature_columns: Sequence[str],
    length: int,
) -> Tuple[np.ndarray, List[str]]:
    rank_block, rank_names = _rank_features(length)
    blocks: List[np.ndarray] = [rank_block]
    names: List[str] = list(rank_names)
    arrays: Dict[str, np.ndarray] = {}

    for column in feature_columns:
        if _is_leaky_feature_column(column):
            continue
        values = _array_from_row(row, column, length)
        arrays[column] = values
        invert = column.endswith("_consistency_m") or column.endswith("_reproj_rmse") or column.endswith("_reproj_median")
        blocks.append(values.reshape(length, 1))
        blocks.append(_normalize_row(values, invert=invert).reshape(length, 1))
        names.extend([column, f"{column}_row01"])

    if "hyp_loftr_render_inliers" in arrays and "hyp_loftr_render_matches" in arrays:
        ratio = _safe_ratio(arrays["hyp_loftr_render_inliers"], arrays["hyp_loftr_render_matches"])
        blocks.append(ratio.reshape(length, 1))
        blocks.append(_normalize_row(ratio).reshape(length, 1))
        names.extend(["loftr_inliers_over_matches", "loftr_inliers_over_matches_row01"])

    if "hyp_retrieval_pnp_num_inliers_candidates" in arrays and "hyp_retrieval_pnp_num_matches_candidates" in arrays:
        ratio = _safe_ratio(
            arrays["hyp_retrieval_pnp_num_inliers_candidates"],
            arrays["hyp_retrieval_pnp_num_matches_candidates"],
        )
        blocks.append(ratio.reshape(length, 1))
        blocks.append(_normalize_row(ratio).reshape(length, 1))
        names.extend(["pnp_inliers_over_matches", "pnp_inliers_over_matches_row01"])

    if "hyp_loftr_render_scores" in arrays and "hyp_loftr_render_consistency_m" in arrays:
        score = arrays["hyp_loftr_render_scores"]
        consistency = arrays["hyp_loftr_render_consistency_m"]
        hybrid = score / (1.0 + np.maximum(consistency, 0.0))
        blocks.append(hybrid.reshape(length, 1))
        blocks.append(_normalize_row(hybrid).reshape(length, 1))
        names.extend(["loftr_score_over_consistency", "loftr_score_over_consistency_row01"])

    return np.concatenate(blocks, axis=1), names


def _finite_feature_matrix(features: np.ndarray) -> np.ndarray:
    out = np.asarray(features, dtype=np.float64).copy()
    for col_idx in range(out.shape[1]):
        column = out[:, col_idx]
        finite = np.isfinite(column)
        fill = float(np.median(column[finite])) if np.any(finite) else 0.0
        column[~finite] = fill
        out[:, col_idx] = column
    return out


def build_candidate_dataset(
    label_csvs: Sequence[str | Path],
    feature_csvs: Optional[Sequence[str | Path]] = None,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_mode: str = "trans",
    pose_rot_weight: float = 0.15,
) -> CandidateDataset:
    if not label_csvs:
        raise ValueError("at least one label CSV is required")
    if feature_csvs is not None and len(feature_csvs) != len(label_csvs):
        raise ValueError("feature_csvs must be omitted or have the same length as label_csvs")

    feature_paths: Sequence[Optional[Path]]
    if feature_csvs is None:
        feature_paths = [None] * len(label_csvs)
    else:
        feature_paths = [Path(path) for path in feature_csvs]

    all_blocks: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    sample_ids: List[int] = []
    candidate_indices: List[int] = []
    rot_values: List[float] = []
    trans_values: List[float] = []
    target_values: List[float] = []
    image_names: List[str] = []
    best_indices: List[int] = []

    for label_path_raw, feature_path in zip(label_csvs, feature_paths):
        label_path = Path(label_path_raw)
        label_rows = _read_rows(label_path)
        feature_by_image: Dict[str, Dict[str, str]] = {}
        if feature_path is not None:
            feature_by_image = _rows_by_image(_read_rows(feature_path), feature_path)

        for label_row in label_rows:
            image_name = label_row.get("image_name", "")
            if not image_name:
                raise ValueError(f"{label_path} contains a row without image_name")
            merged = dict(label_row)
            feature_row = feature_by_image.get(image_name)
            if feature_path is not None and feature_row is None:
                raise ValueError(f"{feature_path} is missing image_name {image_name!r}")
            if feature_row is not None:
                for column in feature_columns:
                    if column in feature_row:
                        merged[column] = feature_row[column]

            rot = np.asarray(parse_list_cell(merged.get("hyp_final_rot_err_deg"), invalid_value=float("nan")), dtype=np.float64)
            trans = np.asarray(parse_list_cell(merged.get("hyp_final_trans_err_mm"), invalid_value=float("nan")), dtype=np.float64)
            length = min(len(rot), len(trans))
            if length <= 0:
                continue
            rot = rot[:length]
            trans = trans[:length]
            target = _target_values(rot, trans, target_mode, pose_rot_weight)
            block, names = _candidate_block(merged, feature_columns, length)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("feature name mismatch while building candidate dataset")

            sample_id = len(image_names)
            all_blocks.append(block)
            sample_ids.extend([sample_id] * length)
            candidate_indices.extend(range(length))
            rot_values.extend(float(v) for v in rot)
            trans_values.extend(float(v) for v in trans)
            target_values.extend(float(v) for v in target)
            image_names.append(image_name)
            best_indices.append(_best_index(rot, trans, target))

    if not all_blocks or feature_names is None:
        raise ValueError("no candidate rows were loaded")

    return CandidateDataset(
        features=_finite_feature_matrix(np.concatenate(all_blocks, axis=0)),
        feature_names=feature_names,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        rot_err_deg=np.asarray(rot_values, dtype=np.float64),
        trans_err_mm=np.asarray(trans_values, dtype=np.float64),
        targets=np.asarray(target_values, dtype=np.float64),
        image_names=image_names,
        best_indices=best_indices,
    )


def _summarize_selection(dataset: CandidateDataset, name: str, selected_indices: Sequence[int]) -> SelectorSummary:
    selected_rot: List[float] = []
    selected_trans: List[float] = []

    for sample_id, selected_idx in enumerate(selected_indices):
        if selected_idx < 0:
            continue
        mask = dataset.sample_ids == sample_id
        row_indices = np.flatnonzero(mask & (dataset.candidate_indices == selected_idx))
        if len(row_indices) == 0:
            continue
        row_idx = int(row_indices[0])
        rot = float(dataset.rot_err_deg[row_idx])
        trans = float(dataset.trans_err_mm[row_idx])
        if math.isfinite(rot) and math.isfinite(trans):
            selected_rot.append(rot)
            selected_trans.append(trans)

    if not selected_rot:
        nan = float("nan")
        return SelectorSummary(name, len(dataset.image_names), 0, nan, nan, nan, nan, nan, nan, list(selected_indices))

    rot_arr = np.asarray(selected_rot, dtype=np.float64)
    trans_arr = np.asarray(selected_trans, dtype=np.float64)
    success_pose = (rot_arr < 0.5) & (trans_arr < 100.0)
    success_trans = trans_arr < 100.0
    return SelectorSummary(
        name=name,
        num_samples=len(dataset.image_names),
        num_selected=len(selected_rot),
        median_rot_deg=float(np.median(rot_arr)),
        median_trans_mm=float(np.median(trans_arr)),
        mean_rot_deg=float(np.mean(rot_arr)),
        mean_trans_mm=float(np.mean(trans_arr)),
        success_pose_pct=float(np.mean(success_pose) * 100.0),
        success_trans_pct=float(np.mean(success_trans) * 100.0),
        selected_indices=[int(v) for v in selected_indices],
    )


def _select_by_scores(dataset: CandidateDataset, scores: np.ndarray, *, maximize: bool) -> List[int]:
    selected: List[int] = []
    for sample_id in range(len(dataset.image_names)):
        rows = np.flatnonzero(dataset.sample_ids == sample_id)
        if len(rows) == 0:
            selected.append(-1)
            continue
        values = scores[rows]
        finite = np.isfinite(values)
        if not np.any(finite):
            selected.append(-1)
            continue
        usable_values = np.where(finite, values, float("-inf") if maximize else float("inf"))
        local = int(np.argmax(usable_values) if maximize else np.argmin(usable_values))
        selected.append(int(dataset.candidate_indices[rows[local]]))
    return selected


_RULE_FEATURES: Dict[str, Tuple[str, bool]] = {
    "rank0": ("rank", False),
    "score_max": ("hyp_loftr_render_scores", True),
    "inliers_max": ("hyp_loftr_render_inliers", True),
    "matches_max": ("hyp_loftr_render_matches", True),
    "quality_max": ("hyp_candidate_quality_scores", True),
    "pnp_inliers_max": ("hyp_retrieval_pnp_num_inliers_candidates", True),
    "pnp_quality_max": ("hyp_retrieval_pnp_inlier_ratio_candidates", True),
    "consistency_min": ("hyp_loftr_render_consistency_m", False),
    "reproj_rmse_min": ("hyp_loftr_render_reproj_rmse", False),
    "loftr_inlier_ratio_max": ("hyp_loftr_render_inlier_ratio", True),
}


def evaluate_rule_selector(
    dataset: CandidateDataset,
    rule: str,
    feature_name: Optional[str] = None,
) -> SelectorSummary:
    if feature_name is None:
        if rule not in _RULE_FEATURES:
            raise ValueError(f"unknown rule selector {rule!r}")
        feature_name, maximize = _RULE_FEATURES[rule]
    else:
        maximize = not (rule.endswith("_min") or feature_name.endswith("_consistency_m"))

    if feature_name not in dataset.feature_names:
        raise ValueError(f"feature {feature_name!r} is not present in dataset")
    feature_idx = dataset.feature_names.index(feature_name)
    selected = _select_by_scores(dataset, dataset.features[:, feature_idx], maximize=maximize)
    return _summarize_selection(dataset, rule, selected)


def _make_regressor(model_name: str, random_state: int):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "etr_l1":
        return ExtraTreesRegressor(
            n_estimators=512,
            min_samples_leaf=1,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "etr_l3":
        return ExtraTreesRegressor(
            n_estimators=512,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "etr_l8":
        return ExtraTreesRegressor(
            n_estimators=512,
            min_samples_leaf=8,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "rf_l3":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.04,
            l2_regularization=0.02,
            random_state=random_state,
        )
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(f"unknown learned model {model_name!r}")


def train_and_evaluate_model(
    train: CandidateDataset,
    test: CandidateDataset,
    model_name: str,
    *,
    random_state: int = 13,
) -> SelectorSummary:
    if train.feature_names != test.feature_names:
        raise ValueError("train/test feature names do not match")
    model = _make_regressor(model_name, random_state=random_state)
    finite_targets = np.isfinite(train.targets)
    if not np.any(finite_targets):
        raise ValueError("training dataset has no finite targets")
    model.fit(train.features[finite_targets], train.targets[finite_targets])
    pred = np.asarray(model.predict(test.features), dtype=np.float64)
    selected = _select_by_scores(test, pred, maximize=False)
    return _summarize_selection(test, model_name, selected)


def _write_summary_csv(summaries: Sequence[SelectorSummary], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "num",
                "med_trans",
                "mean_trans",
                "med_rot",
                "mean_rot",
                "pct100",
                "success_pose_pct",
                "oracle_med",
                "oracle_rank_med",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.name,
                    row.num_selected,
                    row.median_trans_mm,
                    row.mean_trans_mm,
                    row.median_rot_deg,
                    row.mean_rot_deg,
                    row.success_trans_pct,
                    row.success_pose_pct,
                    "",
                    "",
                ]
            )


def _write_indices(summary: SelectorSummary, output_dir: Path) -> None:
    path = output_dir / f"{summary.name}_test50_indices.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.selected_indices, f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate deployable candidate selectors from per-hypothesis CSVs.")
    parser.add_argument("--train_label_csv", action="append", type=Path, required=True)
    parser.add_argument("--train_feature_csv", action="append", type=Path, default=None)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--test_feature_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "rank0",
            "score_max",
            "inliers_max",
            "consistency_min",
            "quality_max",
            "etr_l1",
            "etr_l3",
            "etr_l8",
            "hgb",
            "ridge",
        ],
    )
    parser.add_argument("--target_mode", choices=["trans", "pose"], default="trans")
    parser.add_argument("--pose_rot_weight", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=13)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = build_candidate_dataset(
        args.train_label_csv,
        args.train_feature_csv,
        target_mode=args.target_mode,
        pose_rot_weight=args.pose_rot_weight,
    )
    test_feature_csvs = [args.test_feature_csv] if args.test_feature_csv is not None else None
    test = build_candidate_dataset(
        [args.test_csv],
        test_feature_csvs,
        target_mode=args.target_mode,
        pose_rot_weight=args.pose_rot_weight,
    )

    summaries: List[SelectorSummary] = []
    for model_name in args.models:
        if model_name == "oracle":
            summary = _summarize_selection(test, model_name, test.best_indices)
        elif model_name in _RULE_FEATURES:
            summary = evaluate_rule_selector(test, model_name)
        else:
            summary = train_and_evaluate_model(train, test, model_name, random_state=args.random_state)
        summaries.append(summary)
        _write_indices(summary, args.output_dir)

    _write_summary_csv(summaries, args.output_dir / "summary.csv")
    print(f"wrote {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
