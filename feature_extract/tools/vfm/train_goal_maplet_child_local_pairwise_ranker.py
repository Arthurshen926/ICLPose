"""Train a query-group pairwise ranker for child-local surface modes."""

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
    ChildLocalPairwiseRankerArtifact,
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
        if metadata is None:
            metadata = current
        else:
            for key in (
                "physical_map_sha256", "canonical_field_sha256",
                "field_feature_contract_sha256", "child_eligibility_sha256",
                "temperature", "maximum_modes",
            ):
                if current.get(key) != metadata.get(key):
                    raise ValueError(f"child-local sample shards differ: {key}")
        for index in range(feature.shape[0]):
            trajectory = str(trajectories[index])
            if trajectory in excluded or int(np.sum(valid[index])) < 2:
                continue
            partitions["validation" if trajectory in validation else "train"].append({
                "image_id": str(image_ids[index]),
                "features": feature[index],
                "errors": error[index],
                "valid": valid[index],
            })
    return partitions, metadata


def _pair_dataset(rows: list[dict], *, maximum_pairs: int, random_seed: int):
    features, labels, weights = [], [], []
    for row in rows:
        valid = np.flatnonzero(row["valid"])
        left, right = np.triu_indices(valid.size, 1)
        left, right = valid[left], valid[right]
        gap = np.abs(row["errors"][left] - row["errors"][right])
        keep = gap >= 0.005
        left, right, gap = left[keep], right[keep], gap[keep]
        if left.size == 0:
            continue
        difference = row["features"][left] - row["features"][right]
        preference = row["errors"][left] < row["errors"][right]
        weight = np.clip(gap / 0.10, 0.10, 5.0)
        features.extend((difference, -difference))
        labels.extend((preference, ~preference))
        weights.extend((weight, weight))
    x = np.concatenate(features, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    weight = np.concatenate(weights, axis=0).astype(np.float64)
    if x.shape[0] > int(maximum_pairs):
        rng = np.random.default_rng(int(random_seed))
        selected = rng.choice(x.shape[0], size=int(maximum_pairs), replace=False)
        x, y, weight = x[selected], y[selected], weight[selected]
    return x, y, weight


def _mode_calibration(rows: list[dict], artifact: ChildLocalPairwiseRankerArtifact) -> dict:
    selected, baseline, oracle, selected_confidence, selected_correct = [], [], [], [], []
    correct_mass, nll = [], []
    resolvable = []
    by_image: dict[str, list[float]] = {}
    feature = np.stack([row["features"] for row in rows], axis=0)
    validity = np.stack([row["valid"] for row in rows], axis=0)
    _, probabilities = artifact.score_modes(feature, validity)
    for row, probability in zip(rows, probabilities):
        valid = np.flatnonzero(row["valid"])
        choice = int(valid[np.argmax(probability[valid])])
        best = int(valid[np.argmin(row["errors"][valid])])
        selected.append(float(row["errors"][choice]))
        baseline.append(float(row["errors"][valid[0]]))
        oracle.append(float(row["errors"][best]))
        selected_confidence.append(float(probability[choice]))
        selected_correct.append(float(row["errors"][choice] <= 0.20))
        correct = row["valid"] & (row["errors"] <= 0.20)
        mass = float(np.sum(probability[correct]))
        correct_mass.append(mass)
        resolvable.append(bool(np.any(correct)))
        if np.any(correct):
            nll.append(float(-np.log(max(mass, 1e-12))))
        by_image.setdefault(row["image_id"], []).append(float(row["errors"][choice]))
    value = np.asarray(selected, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    best = np.asarray(oracle, dtype=np.float64)
    query = np.asarray([np.median(item) for item in by_image.values()], dtype=np.float64)
    confidence = np.asarray(selected_confidence, dtype=np.float64)
    correctness = np.asarray(selected_correct, dtype=np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 if lower < 0.9 else confidence <= 1.0)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(confidence[mask]) - np.mean(correctness[mask])))
    def metrics(array):
        return {
            "count": int(array.size), "median_m": float(np.median(array)),
            "p90_m": float(np.percentile(array, 90.0)),
            "within_0p2m_fraction": float(np.mean(array <= 0.20)),
        }
    return {
        "selected": metrics(value), "vfm_top1": metrics(base), "topm_oracle": metrics(best),
        "query_balanced_selected": metrics(query),
        "median_selection_regret_m": float(np.median(value - best)),
        "correct_mode_mass_mean": float(np.mean(correct_mass)),
        "resolvable_topm_fraction": float(np.mean(resolvable)),
        "local_mode_nll_resolvable": float(np.mean(nll)) if nll else None,
        "selected_mode_ece_10bin": float(ece),
    }


def _fit_score_temperature(rows: list[dict], estimator) -> float:
    neutral = ChildLocalPairwiseRankerArtifact(estimator, {
        "artifact_type": "goal_maplet_child_local_pairwise_ranker_v2",
        "feature_names": list(FEATURE_NAMES), "score_temperature": 1.0,
    })
    feature = np.stack([row["features"] for row in rows], axis=0)
    valid = np.stack([row["valid"] for row in rows], axis=0)
    score, _ = neutral.score_modes(feature, valid)
    errors = np.stack([row["errors"] for row in rows], axis=0)
    best = (float("inf"), 1.0)
    for temperature in np.geomspace(0.03, 2.0, 32):
        logits = np.where(valid, score / float(temperature), -np.inf)
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(np.clip(logits, -60.0, 0.0)) * valid
        probability /= np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)
        correct = valid & (errors <= 0.20)
        resolvable = np.any(correct, axis=1)
        mass = np.sum(probability * correct, axis=1)[resolvable]
        nll = float(np.mean(-np.log(np.maximum(mass, 1e-12))))
        if nll < best[0]:
            best = (nll, float(temperature))
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq9", "seq10", "seq12", "seq14"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11"])
    parser.add_argument("--maximum_pairs", type=int, default=400000)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite child-local pairwise ranker")
    validation, excluded = set(args.validation_trajectories), set(args.excluded_trajectories)
    partitions, sample_metadata = _load([Path(value) for value in args.samples], validation, excluded)
    x, y, weight = _pair_dataset(partitions["train"], maximum_pairs=int(args.maximum_pairs), random_seed=194917)
    estimators = {
        "pairwise_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=194917)
        ),
        "pairwise_hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=3,
            min_samples_leaf=64, l2_regularization=3.0, random_state=194917,
        ),
    }
    fitted, reports = {}, {}
    for name, estimator in estimators.items():
        fit_args = {"logisticregression__sample_weight": weight} if hasattr(estimator, "steps") else {"sample_weight": weight}
        with threadpool_limits(limits=4):
            fitted[name] = estimator.fit(x, y, **fit_args)
        temperature = _fit_score_temperature(partitions["validation"], fitted[name])
        artifact = ChildLocalPairwiseRankerArtifact(fitted[name], {
            "artifact_type": "goal_maplet_child_local_pairwise_ranker_v2",
            "feature_names": list(FEATURE_NAMES), "score_temperature": temperature,
        })
        reports[name] = {
            "score_temperature": temperature,
            "train": _mode_calibration(partitions["train"], artifact),
            "validation": _mode_calibration(partitions["validation"], artifact),
        }
    selected_name = min(reports, key=lambda name: (
        reports[name]["validation"]["query_balanced_selected"]["median_m"],
        reports[name]["validation"]["query_balanced_selected"]["p90_m"],
    ))
    metadata = {
        "artifact_type": "goal_maplet_child_local_pairwise_ranker_v2",
        "feature_names": list(FEATURE_NAMES), "selected_estimator": selected_name,
        "score_temperature": reports[selected_name]["score_temperature"],
        "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "target": "pairwise_continuous_exact_surface_error_m",
        "uses_gt_at_runtime": False, "uses_absolute_pose_features": False,
        "uses_mapping_rgb": False, "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False, "stores_mapping_image_ids": False,
        "stored_downstream_embedding_count": 0, "uses_alike_descriptors": False,
        "uses_radio_intermediate": False, "uses_sfm_points": False,
        "uses_sfm_tracks": False, "uses_point_correspondences": False,
        **{key: sample_metadata[key] for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256", "temperature", "maximum_modes",
        )},
    }
    ChildLocalPairwiseRankerArtifact(fitted[selected_name], metadata).save(output)
    result = {
        "stage": "train_goal_maplet_child_local_pairwise_ranker_v2",
        "selected_estimator": selected_name, "model": str(output), "metadata": metadata,
        "group_count": {key: len(value) for key, value in partitions.items()},
        "pair_count": int(x.shape[0]), "reports": reports,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
