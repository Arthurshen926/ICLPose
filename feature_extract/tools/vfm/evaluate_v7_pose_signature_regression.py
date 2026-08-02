"""Trajectory-held-out pose regression diagnostic for V7 signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
)
from feature_extract.vfm.localization_v7.pose_signature import (
    PoseSignatureBank,
    file_sha256,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_signature_bank", required=True)
    parser.add_argument("--mapping_manifest", required=True)
    parser.add_argument("--query_signatures", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--validation_trajectory", default="seq11")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _vectors(
    identity: np.ndarray,
    mean: np.ndarray,
    extent: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    root = np.sqrt(np.maximum(np.asarray(mass, dtype=np.float32), 0.0))[..., None]
    return np.concatenate(
        [
            np.asarray(identity, dtype=np.float32),
            (root * (np.asarray(mean, dtype=np.float32) - 0.5)).reshape(
                len(identity), -1
            ),
            (root * np.asarray(extent, dtype=np.float32)).reshape(
                len(identity), -1
            ),
        ],
        axis=1,
    )


def _targets(poses: np.ndarray) -> np.ndarray:
    centers = np.stack(
        [camera_center_from_pose_w2c(value) for value in poses]
    )
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    quaternion[quaternion[:, 3] < 0.0] *= -1.0
    return np.concatenate([centers, quaternion], axis=1)


def _errors(prediction: np.ndarray, poses: np.ndarray) -> dict[str, object]:
    target = _targets(poses)
    quaternion = np.asarray(prediction[:, 3:], dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12
    )
    quaternion[quaternion[:, 3] < 0.0] *= -1.0
    translation = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    rotation = np.degrees(
        (
            Rotation.from_quat(quaternion).inv()
            * Rotation.from_quat(target[:, 3:])
        ).magnitude()
    )
    return {
        "query_count": int(poses.shape[0]),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.quantile(translation, 0.90)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.quantile(rotation, 0.90)),
        "within_30cm_3deg": int(
            np.sum((translation <= 0.30) & (rotation <= 3.0))
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V7 regression report")
    bank_path = Path(args.pose_signature_bank)
    manifest_path = Path(args.mapping_manifest)
    query_path = Path(args.query_signatures)
    bank = PoseSignatureBank.load_npz(bank_path)
    if bank.metadata.get("prototype_order") != "sorted_mapping_image_id":
        raise ValueError("pose-signature prototype ordering is not auditable")
    records = sorted(
        TokenBankManifest.from_json(manifest_path).records,
        key=lambda record: record.image_id,
    )
    if len(records) != bank.poses_w2c.shape[0]:
        raise ValueError("mapping manifest and pose prototypes differ")
    trajectory = np.asarray(
        [record.image_id.split("/", 1)[0] for record in records]
    )
    validation = trajectory == str(args.validation_trajectory)
    fit = ~validation
    if not np.any(validation) or not np.any(fit):
        raise ValueError("trajectory-held-out regression split is empty")
    x = _vectors(
        bank.identity,
        bank.layout_mean_xy,
        bank.layout_extent_xy,
        bank.layout_mass,
    )
    y = _targets(bank.poses_w2c)
    with np.load(query_path, allow_pickle=False) as data:
        if not np.array_equal(data["maplet_ids"], bank.maplet_ids):
            raise ValueError("query and mapping signature bases differ")
        query_x = _vectors(
            data["identity"],
            data["layout_mean_xy"],
            data["layout_extent_xy"],
            data["layout_mass"],
        )
        query_poses = np.asarray(data["target_poses_w2c"], dtype=np.float64)

    scaler = StandardScaler().fit(x[fit])
    pca = PCA(
        n_components=128, svd_solver="randomized", random_state=2907
    ).fit(scaler.transform(x[fit]))
    fit_z = pca.transform(scaler.transform(x[fit]))
    validation_z = pca.transform(scaler.transform(x[validation]))
    ridge_validation = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(fit_z, y[fit])
        ridge_validation[str(alpha)] = _errors(
            model.predict(validation_z), bank.poses_w2c[validation]
        )
    selected_alpha = min(
        ridge_validation,
        key=lambda key: (
            ridge_validation[key]["translation_median_m"]
            + 0.1 * ridge_validation[key]["rotation_median_deg"]
        ),
    )

    tree_validation = {}
    for leaf in (1, 2, 4, 8):
        model = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=leaf,
            max_features=0.5,
            n_jobs=-1,
            random_state=2907,
        ).fit(x[fit], y[fit])
        tree_validation[str(leaf)] = _errors(
            model.predict(x[validation]), bank.poses_w2c[validation]
        )
    selected_leaf = min(
        tree_validation,
        key=lambda key: (
            tree_validation[key]["translation_median_m"]
            + 0.1 * tree_validation[key]["rotation_median_deg"]
        ),
    )

    full_scaler = StandardScaler().fit(x)
    full_pca = PCA(
        n_components=128, svd_solver="randomized", random_state=2907
    ).fit(full_scaler.transform(x))
    full_z = full_pca.transform(full_scaler.transform(x))
    query_z = full_pca.transform(full_scaler.transform(query_x))
    ridge = Ridge(alpha=float(selected_alpha)).fit(full_z, y)
    tree = ExtraTreesRegressor(
        n_estimators=512,
        min_samples_leaf=int(selected_leaf),
        max_features=0.5,
        n_jobs=-1,
        random_state=2907,
    ).fit(x, y)
    report = {
        "artifact_type": "v7_pose_signature_regression_diagnostic",
        "status": "diagnostic_not_promoted",
        "selection_protocol": (
            "hyperparameters selected on one complete held-out mapping "
            "trajectory before strict-test evaluation"
        ),
        "validation_trajectory": str(args.validation_trajectory),
        "validation_query_count": int(np.sum(validation)),
        "mapping_fit_count": int(np.sum(fit)),
        "ridge_validation": ridge_validation,
        "ridge_selected_alpha": float(selected_alpha),
        "extra_trees_validation": tree_validation,
        "extra_trees_selected_min_samples_leaf": int(selected_leaf),
        "strict_test": {
            "pca_ridge": _errors(ridge.predict(query_z), query_poses),
            "extra_trees": _errors(tree.predict(query_x), query_poses),
        },
        "artifact_sha256": {
            "pose_signature_bank": file_sha256(bank_path),
            "mapping_manifest": file_sha256(manifest_path),
            "query_signatures": file_sha256(query_path),
        },
        "method_contract": {
            "uses_mapping_pose_ground_truth": True,
            "query_map_feature_interaction_after_stage_a": False,
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids_in_runtime_model": False,
            "stores_mapping_image_paths_in_runtime_model": False,
            "point_correspondence_pnp_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
