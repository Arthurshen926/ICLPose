"""Evaluate G19-A phase survival and fixed tiny cross-trajectory probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract.tools.vfm.build_goal_maplet_phase_survival_samples import (
    LEVELS,
    MAP_LEVELS,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _load(paths: list[str]) -> tuple[dict[str, np.ndarray], list[tuple[Path, dict[str, object]]]]:
    values: dict[str, list[np.ndarray]] = {}
    metadata = []
    for name in paths:
        path = Path(name)
        with np.load(path, allow_pickle=False) as data:
            item = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            meta = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if meta.get("artifact_type") != "goal_maplet_phase_survival_samples_v1":
            raise ValueError("not a G19-A phase-survival sample artifact")
        if tuple(meta.get("levels", ())) != LEVELS:
            raise ValueError("G19-A level contract differs")
        metadata.append((path, meta))
        for key, value in item.items():
            values.setdefault(key, []).append(value)
    return {key: np.concatenate(value, axis=0) for key, value in values.items()}, metadata


def _pair_metrics(score: np.ndarray) -> dict[str, float]:
    value = np.asarray(score, dtype=np.float64)
    margin = value[:, 0] - value[:, 1]
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), value.shape[0])
    return {
        "pair_count": int(value.shape[0]),
        "pair_concordance": float(np.mean(margin > 0.0) + 0.5 * np.mean(margin == 0.0)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.percentile(margin, 10.0)),
        "candidate_auc": float(roc_auc_score(labels, value.reshape(-1))),
    }


def _probe_metrics(
    train_descriptor: np.ndarray,
    selection_descriptor: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    train = np.asarray(train_descriptor, dtype=np.float32)
    selection = np.asarray(selection_descriptor, dtype=np.float32)
    train_x = train.reshape(-1, train.shape[-1])
    train_y = np.tile(np.asarray([1, 0], dtype=np.int64), train.shape[0])
    selection_x = selection.reshape(-1, selection.shape[-1])
    selection_y = np.tile(np.asarray([1, 0], dtype=np.int64), selection.shape[0])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=1901,
            solver="liblinear",
        ),
    )
    model.fit(train_x, train_y)

    def report(x: np.ndarray, y: np.ndarray, pair_count: int) -> dict[str, float]:
        probability = model.predict_proba(x)[:, 1].reshape(pair_count, 2)
        margin = probability[:, 0] - probability[:, 1]
        return {
            "candidate_auc": float(roc_auc_score(y, probability.reshape(-1))),
            "pair_concordance": float(
                np.mean(margin > 0.0) + 0.5 * np.mean(margin == 0.0)
            ),
            "probability_margin_median": float(np.median(margin)),
            "probability_margin_p10": float(np.percentile(margin, 10.0)),
        }

    return (
        report(train_x, train_y, train.shape[0]),
        report(selection_x, selection_y, selection.shape[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_samples", nargs="+", required=True)
    parser.add_argument("--selection_samples", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite G19-A evaluation")
    train, train_metadata = _load(list(args.train_samples))
    selection, selection_metadata = _load(list(args.selection_samples))
    train_trajectories = set(train["trajectory_ids"].astype(str).tolist())
    selection_trajectories = set(selection["trajectory_ids"].astype(str).tolist())
    if train_trajectories & selection_trajectories:
        raise ValueError("G19-A train and selection trajectories overlap")
    lineage_keys = (
        "physical_map_sha256", "mapper_field_sha256", "radio_raw_field_sha256",
        "radio_pca_field_sha256",
        "canonical_codec_sha256", "surface_mapper_sha256",
        "physical_instance_readout_sha256",
    )
    metadata_values = [value for _path, value in train_metadata + selection_metadata]
    for key in lineage_keys:
        if len({str(value.get(key)) for value in metadata_values}) != 1:
            raise ValueError(f"G19-A train/selection lineage differs: {key}")

    levels = {}
    for level in LEVELS:
        train_direct = _pair_metrics(train[f"score__{level}"])
        selection_direct = _pair_metrics(selection[f"score__{level}"])
        probe_train, probe_selection = _probe_metrics(
            train[f"descriptor__{level}"], selection[f"descriptor__{level}"],
        )
        level_report = {
            "descriptor_dimension": int(train[f"descriptor__{level}"].shape[-1]),
            "direct_score_train": train_direct,
            "direct_score_selection": selection_direct,
            "fixed_probe_train": probe_train,
            "fixed_probe_selection": probe_selection,
        }
        if level in MAP_LEVELS:
            level_report["token_audit_train"] = {
                "discriminative_fraction_mean": float(np.mean(
                    train[f"discriminative_fraction__{level}"]
                )),
                "positive_discriminative_mass_mean": float(np.mean(
                    train[f"positive_mass__{level}"]
                )),
            }
            level_report["token_audit_selection"] = {
                "discriminative_fraction_mean": float(np.mean(
                    selection[f"discriminative_fraction__{level}"]
                )),
                "positive_discriminative_mass_mean": float(np.mean(
                    selection[f"positive_mass__{level}"]
                )),
            }
        levels[level] = level_report
        print(json.dumps({"level": level, "selection": level_report}, sort_keys=True), flush=True)

    result = {
        "stage": "g19_a_phase_information_survival_audit",
        "train_query_count": int(train["image_ids"].shape[0]),
        "selection_query_count": int(selection["image_ids"].shape[0]),
        "training_trajectory_ids": sorted(train_trajectories),
        "selection_trajectory_ids": sorted(selection_trajectories),
        "phase_pair_selection": {
            "train_translation_m_median": float(np.median(train["phase_translation_m"])),
            "selection_translation_m_median": float(np.median(selection["phase_translation_m"])),
            "train_rotation_deg_median": float(np.median(train["phase_rotation_deg"])),
            "selection_rotation_deg_median": float(np.median(selection["phase_rotation_deg"])),
        },
        "probe_contract": {
            "type": "fixed_standardized_l2_logistic_bilinear_spatial_probe",
            "C": 0.1,
            "selection_used_for_training": False,
            "probe_is_deployment_model": False,
        },
        "radio_level_limitation": metadata_values[0].get("radio_level_limitation"),
        "levels": levels,
        "lineage": {key: metadata_values[0].get(key) for key in lineage_keys},
        "sample_sha256": {
            "train": [file_sha256(path) for path, _meta in train_metadata],
            "selection": [file_sha256(path) for path, _meta in selection_metadata],
        },
        "deployment_contract": {
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "levels"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
