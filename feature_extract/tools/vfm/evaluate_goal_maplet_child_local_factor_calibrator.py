"""Evaluate a frozen typed-null child-local factor calibrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from feature_extract.vfm.localization_goal_maplet.child_local_factor import (
    FEATURE_NAMES,
    NULL_TYPES,
    ChildLocalFactorCalibratorArtifact,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite child-local factor evaluation")
    calibrator = ChildLocalFactorCalibratorArtifact.load(Path(args.calibrator))
    features, targets, images, sources = [], [], [], []
    metadata = None
    for value in args.samples:
        with np.load(Path(value), allow_pickle=False) as data:
            features.append(np.asarray(data["features"], dtype=np.float32))
            targets.append(np.asarray(data["targets"], dtype=np.int64))
            images.append(np.asarray(data["image_ids"]).astype(str))
            sources.append(np.asarray(data["source_types"]).astype(str))
            current = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if metadata is None:
            metadata = current
        else:
            for key in ("physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256", "child_eligibility_sha256"):
                if current.get(key) != metadata.get(key):
                    raise ValueError(f"evaluation sample shards differ: {key}")
    feature = np.concatenate(features)
    target = np.concatenate(targets)
    image = np.concatenate(images)
    source = np.concatenate(sources)
    if feature.ndim != 2 or feature.shape[1] != len(FEATURE_NAMES):
        raise ValueError("child-local factor evaluation feature contract differs")
    for key in ("physical_map_sha256", "canonical_field_sha256", "field_feature_contract_sha256", "child_eligibility_sha256"):
        if calibrator.metadata.get(key) != metadata.get(key):
            raise ValueError(f"calibrator and evaluation samples differ: {key}")
    probability = calibrator.predict_typed_probabilities(feature)
    predicted = np.argmax(probability, axis=1)
    confidence = np.max(probability, axis=1)
    correct = predicted == target
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 if lower < 0.9 else confidence <= 1.0)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(confidence[mask]) - np.mean(correct[mask])))
    query_nll = []
    for value in np.unique(image):
        mask = image == value
        query_nll.append(float(np.mean(-np.log(np.maximum(probability[mask, target[mask]], 1e-12)))))
    typed_ap = {}
    for label, name in enumerate(NULL_TYPES):
        truth = target == label
        typed_ap[name] = float(average_precision_score(truth, probability[:, label])) if np.any(truth) else None
    report = {
        "stage": "evaluate_goal_maplet_child_local_factor_calibrator_v2",
        "sample_count": int(target.size), "query_count": int(np.unique(image).size),
        "samples": list(args.samples), "sample_sha256": [file_sha256(Path(value)) for value in args.samples],
        "calibrator": str(args.calibrator), "calibrator_sha256": file_sha256(Path(args.calibrator)),
        "nll": float(np.mean(-np.log(np.maximum(probability[np.arange(target.size), target], 1e-12)))),
        "query_balanced_nll": float(np.mean(query_nll)),
        "ece_10bin": float(ece), "accuracy": float(np.mean(correct)),
        "typed_auprc": typed_ap,
        "confusion_matrix": confusion_matrix(target, predicted, labels=np.arange(len(NULL_TYPES))).tolist(),
        "source_accuracy": {
            value: float(np.mean(correct[source == value])) for value in sorted(set(source.tolist()))
        },
        "true_fraction": {
            name: float(np.mean(target == label)) for label, name in enumerate(NULL_TYPES)
        },
        "mean_predicted_mass": {
            name: float(np.mean(probability[:, label])) for label, name in enumerate(NULL_TYPES)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
