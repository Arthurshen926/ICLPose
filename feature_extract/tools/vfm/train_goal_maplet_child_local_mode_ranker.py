"""Train a lightweight query-grouped ranker for VFM child-local Top-M modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import (
    FEATURE_NAMES,
    ChildLocalModeRankerArtifact,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _load(paths: list[Path], validation: set[str], excluded: set[str]):
    partitions = {"train": [], "validation": []}
    metadata = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            feature = np.asarray(data["features"], dtype=np.float32)
            error = np.asarray(data["target_errors_m"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            image_ids = np.asarray(data["group_image_ids"]).astype(str)
            trajectories = np.asarray(data["group_trajectories"]).astype(str)
            current = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if feature.ndim != 3 or feature.shape[2] != len(FEATURE_NAMES):
            raise ValueError("child-local sample feature contract differs")
        if error.shape != feature.shape[:2] or valid.shape != error.shape:
            raise ValueError("child-local sample arrays differ")
        if image_ids.shape != (feature.shape[0],) or trajectories.shape != image_ids.shape:
            raise ValueError("child-local sample group metadata differs")
        if metadata is None:
            metadata = current
        elif current != metadata:
            raise ValueError("child-local sample shard lineage/config differs")
        for index in range(feature.shape[0]):
            trajectory = str(trajectories[index])
            if trajectory in excluded:
                continue
            split = "validation" if trajectory in validation else "train"
            partitions[split].append({
                "image_id": str(image_ids[index]),
                "trajectory": trajectory,
                "features": feature[index],
                "errors": error[index],
                "valid": valid[index],
            })
    return partitions, metadata


def _fit(rows, estimator):
    x, y, weights = [], [], []
    for row in rows:
        valid_rows = np.flatnonzero(row["valid"])
        if valid_rows.size == 0:
            continue
        target = int(valid_rows[np.argmin(row["errors"][valid_rows])])
        x.append(row["features"])
        label = np.zeros(row["errors"].shape, dtype=np.int64)
        label[target] = 1
        y.append(label)
        weights.append(np.full(label.shape, 1.0 / max(valid_rows.size, 1), dtype=np.float64))
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    weights = np.concatenate(weights, axis=0)
    positive = max(float(np.sum(weights[y == 1])), 1e-8)
    negative = max(float(np.sum(weights[y == 0])), 1e-8)
    weights[y == 1] *= 0.5 / positive
    weights[y == 0] *= 0.5 / negative
    fit_args = (
        {"logisticregression__sample_weight": weights}
        if hasattr(estimator, "steps") else {"sample_weight": weights}
    )
    with threadpool_limits(limits=2):
        estimator.fit(x, y, **fit_args)
    return estimator


def _metrics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median_m": float(np.median(array)),
        "p90_m": float(np.percentile(array, 90.0)),
        "within_0p1m_fraction": float(np.mean(array <= 0.1)),
        "within_0p2m_fraction": float(np.mean(array <= 0.2)),
    }


def _evaluate(rows, estimator):
    selected, default, oracle = [], [], []
    by_image: dict[str, list[float]] = {}
    for row in rows:
        probability = np.asarray(estimator.predict_proba(row["features"])[:, 1], dtype=np.float64)
        probability[~row["valid"]] = -np.inf
        valid_rows = np.flatnonzero(row["valid"])
        if valid_rows.size == 0:
            continue
        choice = int(np.argmax(probability))
        baseline = int(valid_rows[0])
        best = int(valid_rows[np.argmin(row["errors"][valid_rows])])
        selected.append(float(row["errors"][choice]))
        default.append(float(row["errors"][baseline]))
        oracle.append(float(row["errors"][best]))
        by_image.setdefault(row["image_id"], []).append(float(row["errors"][choice]))
    query_medians = [float(np.median(values)) for values in by_image.values()]
    return {
        "selected": _metrics(selected),
        "vfm_top1": _metrics(default),
        "topm_oracle": _metrics(oracle),
        "query_balanced_selected_median": _metrics(query_medians),
        "median_selection_regret_m": float(np.median(np.asarray(selected) - np.asarray(oracle))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq9", "seq10", "seq12", "seq14"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11"])
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite child-local mode ranker")
    validation, excluded = set(args.validation_trajectories), set(args.excluded_trajectories)
    if validation & excluded:
        raise ValueError("validation and excluded trajectories must be disjoint")
    partitions, sample_metadata = _load(
        [Path(value) for value in args.samples], validation, excluded
    )
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("child-local ranker requires trajectory-disjoint splits")
    estimators = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=194917)
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=3,
            min_samples_leaf=32, l2_regularization=1.0, random_state=194917,
        ),
    }
    fitted, reports = {}, {}
    for name, estimator in estimators.items():
        fitted[name] = _fit(partitions["train"], estimator)
        reports[name] = {
            split: _evaluate(partitions[split], fitted[name])
            for split in ("train", "validation")
        }
    selected_name = min(
        reports,
        key=lambda name: (
            reports[name]["validation"]["query_balanced_selected_median"]["median_m"],
            reports[name]["validation"]["selected"]["p90_m"],
        ),
    )
    metadata = {
        "artifact_type": "goal_maplet_child_local_mode_ranker_v1",
        "feature_names": list(FEATURE_NAMES),
        "selected_estimator": selected_name,
        "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "target": "closest_runtime_vfm_topm_primitive_to_exact_child_local_surface",
        "uses_gt_at_runtime": False,
        "uses_absolute_pose_features": False,
        "uses_mapping_rgb": False,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_point_correspondences": False,
        **{
            key: sample_metadata[key] for key in (
                "physical_map_sha256", "canonical_field_sha256",
                "field_feature_contract_sha256", "child_eligibility_sha256",
                "temperature", "maximum_modes",
            )
        },
    }
    ChildLocalModeRankerArtifact(fitted[selected_name], metadata).save(output)
    result = {
        "stage": "train_goal_maplet_child_local_mode_ranker",
        "selected_estimator": selected_name,
        "model": str(output),
        "metadata": metadata,
        "group_count": {split: len(rows) for split, rows in partitions.items()},
        "reports": reports,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
