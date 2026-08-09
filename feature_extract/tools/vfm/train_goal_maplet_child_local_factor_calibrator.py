"""Train and calibrate typed-null child-local pose factors by trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from feature_extract.vfm.localization_goal_maplet.child_local_factor import (
    FEATURE_NAMES,
    NULL_TYPES,
    ChildLocalFactorCalibratorArtifact,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _load(paths: list[Path], calibration: set[str], validation: set[str], excluded: set[str]):
    partitions = {"train": [], "calibration": [], "validation": []}
    metadata = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            feature = np.asarray(data["features"], dtype=np.float32)
            target = np.asarray(data["targets"], dtype=np.int64)
            image = np.asarray(data["image_ids"]).astype(str)
            trajectory = np.asarray(data["trajectories"]).astype(str)
            source = np.asarray(data["source_types"]).astype(str)
            current = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if feature.ndim != 2 or feature.shape[1] != len(FEATURE_NAMES):
            raise ValueError("child-local factor feature contract differs")
        if target.shape != (feature.shape[0],) or image.shape != target.shape or trajectory.shape != target.shape:
            raise ValueError("child-local factor sample arrays differ")
        if tuple(current.get("null_types", ())) != NULL_TYPES:
            raise ValueError("child-local factor null contract differs")
        if metadata is None:
            metadata = current
        else:
            for key in (
                "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
                "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
                "physical_instance_readout_sha256",
            ):
                if current.get(key) != metadata.get(key):
                    raise ValueError(f"child-local factor shards differ: {key}")
        for split, mask in (
            ("train", ~np.isin(trajectory, list(calibration | validation | excluded))),
            ("calibration", np.isin(trajectory, list(calibration))),
            ("validation", np.isin(trajectory, list(validation))),
        ):
            partitions[split].append((feature[mask], target[mask], image[mask], source[mask]))
    output = {}
    for split, parts in partitions.items():
        output[split] = tuple(np.concatenate([part[index] for part in parts], axis=0) for index in range(4))
    return output, metadata


def _sample_weight(target: np.ndarray, image: np.ndarray) -> np.ndarray:
    _, inverse, count = np.unique(image, return_inverse=True, return_counts=True)
    weight = 1.0 / count[inverse]
    for label in np.unique(target):
        mask = target == label
        weight[mask] *= 1.0 / max(float(np.sum(weight[mask])), 1e-12)
    weight *= target.size / max(float(np.sum(weight)), 1e-12)
    return weight


def _probability(estimator, feature: np.ndarray, temperature: float) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(feature), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    probability = np.zeros((feature.shape[0], len(NULL_TYPES)), dtype=np.float64)
    probability[:, classes] = raw
    logits = np.log(np.maximum(probability, 1e-12)) / max(float(temperature), 1e-4)
    logits -= np.max(logits, axis=1, keepdims=True)
    probability = np.exp(np.clip(logits, -60.0, 0.0))
    return probability / np.maximum(np.sum(probability, axis=1, keepdims=True), 1e-12)


def _fit_temperature(estimator, feature: np.ndarray, target: np.ndarray) -> float:
    best = (float("inf"), 1.0)
    for temperature in np.geomspace(0.25, 4.0, 41):
        probability = _probability(estimator, feature, float(temperature))
        nll = float(np.mean(-np.log(np.maximum(probability[np.arange(target.size), target], 1e-12))))
        if nll < best[0]:
            best = (nll, float(temperature))
    return best[1]


def _evaluate(estimator, rows, temperature: float) -> dict:
    feature, target, image, source = rows
    probability = _probability(estimator, feature, temperature)
    predicted = np.argmax(probability, axis=1)
    confidence = np.max(probability, axis=1)
    correct = predicted == target
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 if lower < 0.9 else confidence <= 1.0)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(confidence[mask]) - np.mean(correct[mask])))
    ap = {}
    for label, name in enumerate(NULL_TYPES):
        truth = target == label
        ap[name] = float(average_precision_score(truth, probability[:, label])) if np.any(truth) else None
    query_nll = []
    for value in np.unique(image):
        mask = image == value
        query_nll.append(float(np.mean(-np.log(np.maximum(probability[mask, target[mask]], 1e-12)))))
    source_accuracy = {
        value: float(np.mean(correct[source == value])) for value in sorted(set(source.tolist()))
    }
    return {
        "sample_count": int(target.size),
        "query_count": int(np.unique(image).size),
        "nll": float(np.mean(-np.log(np.maximum(probability[np.arange(target.size), target], 1e-12)))),
        "query_balanced_nll": float(np.mean(query_nll)),
        "ece_10bin": float(ece),
        "accuracy": float(np.mean(correct)),
        "valid_auprc": ap["valid"],
        "typed_auprc": ap,
        "source_accuracy": source_accuracy,
        "confusion_matrix": confusion_matrix(target, predicted, labels=np.arange(len(NULL_TYPES))).tolist(),
        "mean_predicted_mass": {
            name: float(np.mean(probability[:, label])) for label, name in enumerate(NULL_TYPES)
        },
        "true_fraction": {
            name: float(np.mean(target == label)) for label, name in enumerate(NULL_TYPES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--calibration_trajectories", nargs="+", default=["seq9", "seq10"])
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq12", "seq14"])
    parser.add_argument("--excluded_trajectories", nargs="+", default=["seq11"])
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_model), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite child-local factor calibrator")
    calibration = set(args.calibration_trajectories)
    validation = set(args.validation_trajectories)
    excluded = set(args.excluded_trajectories)
    if calibration & validation or (calibration | validation) & excluded:
        raise ValueError("factor trajectory partitions must be disjoint")
    partitions, sample_metadata = _load(
        [Path(value) for value in args.samples], calibration, validation, excluded
    )
    x, y, image, _ = partitions["train"]
    weight = _sample_weight(y, image)
    estimators = {
        "multinomial_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.25, max_iter=3000, random_state=194917)
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15, max_depth=3,
            min_samples_leaf=64, l2_regularization=3.0, random_state=194917,
        ),
    }
    fitted, reports, temperatures = {}, {}, {}
    for name, estimator in estimators.items():
        fit_args = {"logisticregression__sample_weight": weight} if hasattr(estimator, "steps") else {"sample_weight": weight}
        with threadpool_limits(limits=4):
            fitted[name] = estimator.fit(x, y, **fit_args)
        temperatures[name] = _fit_temperature(
            fitted[name], partitions["calibration"][0], partitions["calibration"][1]
        )
        reports[name] = {
            "calibration_temperature": temperatures[name],
            "calibration": _evaluate(fitted[name], partitions["calibration"], temperatures[name]),
            "validation": _evaluate(fitted[name], partitions["validation"], temperatures[name]),
        }
    selected_name = min(reports, key=lambda name: (
        reports[name]["validation"]["query_balanced_nll"],
        -reports[name]["validation"]["valid_auprc"],
        reports[name]["validation"]["ece_10bin"],
    ))
    metadata = {
        "artifact_type": "goal_maplet_child_local_factor_calibrator_v2",
        "feature_names": list(FEATURE_NAMES), "null_types": list(NULL_TYPES),
        "selected_estimator": selected_name,
        "calibration_temperature": temperatures[selected_name],
        "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "calibration_trajectories": list(args.calibration_trajectories),
        "validation_trajectories": list(args.validation_trajectories),
        "excluded_trajectories": list(args.excluded_trajectories),
        "training_trajectories": sorted({
            str(value).split("/", 1)[0] for value in partitions["train"][2].tolist()
        }),
        "uses_gt_at_runtime": False, "uses_absolute_pose_features": False,
        "uses_mapping_rgb": False, "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False, "stores_mapping_image_ids": False,
        "stored_downstream_embedding_count": 0, "uses_alike_descriptors": False,
        "uses_radio_intermediate": False, "uses_sfm_points": False,
        "uses_sfm_tracks": False, "uses_point_correspondences": False,
        **{key: sample_metadata[key] for key in (
            "physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256",
            "child_eligibility_sha256", "candidate_pool_sha256", "temperature", "maximum_modes",
        )},
    }
    if sample_metadata.get("physical_instance_readout_sha256") is not None:
        metadata["physical_instance_readout_sha256"] = str(
            sample_metadata["physical_instance_readout_sha256"]
        )
    ChildLocalFactorCalibratorArtifact(fitted[selected_name], metadata).save(output)
    result = {
        "stage": "train_goal_maplet_child_local_factor_calibrator_v2",
        "selected_estimator": selected_name, "model": str(output), "metadata": metadata,
        "partition_count": {key: int(value[1].size) for key, value in partitions.items()},
        "reports": reports,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
