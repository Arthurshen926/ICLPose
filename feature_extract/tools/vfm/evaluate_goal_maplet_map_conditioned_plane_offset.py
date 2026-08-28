"""Fit a low-capacity map-region plane-offset calibrator and evaluate a held route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    solve_robust_metric_translation,
    solve_rotation_from_plane_normals,
)


SCHEMA = "goal_maplet_map_conditioned_plane_offset_v1"
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray | dict]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("artifact_type") != "goal_maplet_gt_corresponded_plane_measurements_v1":
        raise ValueError("unexpected plane measurement schema")
    offsets = np.asarray(result["query_offsets"], np.int64)
    if offsets.shape != (len(result["image_ids"]) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid query offsets")
    if offsets[-1] != len(result["region_rows"]):
        raise ValueError("plane measurement rows do not match offsets")
    result["metadata"] = metadata
    return result


def _fit(data: dict, query_mask: np.ndarray, ridge: float) -> dict:
    offsets = data["query_offsets"]
    row_mask = np.zeros(len(data["region_rows"]), bool)
    for query in np.flatnonzero(query_mask).tolist():
        row_mask[int(offsets[query]):int(offsets[query + 1])] = True
    region = np.asarray(data["region_rows"], np.int64)[row_mask]
    unique = np.unique(region)
    region_column = {int(value): index for index, value in enumerate(unique.tolist())}
    x = np.zeros((len(region), 2 + len(unique)), np.float64)
    x[:, 0] = np.asarray(data["query_offset"], np.float64)[row_mask]
    x[:, 1] = 1.0
    x[np.arange(len(region)), 2 + np.asarray([region_column[int(value)] for value in region])] = 1.0
    target = np.asarray(data["ideal_query_offset"], np.float64)[row_mask]
    weight = np.maximum(np.asarray(data["weight"], np.float64)[row_mask], 1.0e-12)
    gram = x.T @ (weight[:, None] * x)
    penalty = np.zeros(x.shape[1], np.float64)
    penalty[2:] = float(ridge)
    coefficient = np.linalg.solve(gram + np.diag(penalty) + np.eye(x.shape[1]) * 1.0e-10, x.T @ (weight * target))
    return {"kind": "region_intercept", "coefficient": coefficient, "regions": unique, "ridge": float(ridge), "training_plane_count": int(len(region))}


def _continuous_features(data: dict, rows, *, include_radio_shape: bool) -> np.ndarray:
    predicted = np.asarray(data["query_offset"], np.float64)[rows]
    query_normal = np.asarray(data["query_normal"], np.float64)[rows]
    parts = [predicted[:, None], query_normal,
             np.asarray(data["map_normal"], np.float64)[rows], np.asarray(data["map_offset"], np.float64)[rows, None]]
    if include_radio_shape:
        shape = np.asarray(data["query_shape"], np.float64)[rows]
        area_fraction = np.maximum(shape[:, 0], 1.0e-6)
        map_area = np.maximum(np.expm1(shape[:, 5]), 1.0e-6)
        incidence = np.maximum(np.abs(query_normal[:, 2]), 1.0e-3)
        # First-principles weak-perspective cue: projected area is proportional
        # to physical area * incidence / depth^2.  Multiplying inferred depth
        # by incidence converts it to camera-to-plane distance approximately.
        polygon_distance_proxy = np.sqrt(map_area * incidence / area_fraction) * incidence
        scale_log_ratio = np.log(map_area) - np.log(area_fraction)
        parts.extend((np.asarray(data["radio_group32"], np.float64)[rows], shape,
                      polygon_distance_proxy[:, None], scale_log_ratio[:, None]))
    return np.concatenate(parts, axis=1)


def _fit_continuous(data: dict, query_mask: np.ndarray, ridge: float, *, include_radio_shape: bool = False) -> dict:
    offsets = np.asarray(data["query_offsets"], np.int64)
    row_mask = np.zeros(len(data["region_rows"]), bool)
    for query in np.flatnonzero(query_mask).tolist():
        row_mask[int(offsets[query]):int(offsets[query + 1])] = True
    raw = _continuous_features(data, row_mask, include_radio_shape=include_radio_shape)
    mean, scale = np.mean(raw, axis=0), np.std(raw, axis=0)
    scale = np.maximum(scale, 1.0e-8)
    x = np.column_stack(((raw - mean) / scale, np.ones(len(raw))))
    target = np.asarray(data["ideal_query_offset"], np.float64)[row_mask]
    weight = np.maximum(np.asarray(data["weight"], np.float64)[row_mask], 1.0e-12)
    penalty = np.full(x.shape[1], float(ridge), np.float64)
    penalty[-1] = 0.0
    coefficient = np.linalg.solve(x.T @ (weight[:, None] * x) + np.diag(penalty) + np.eye(x.shape[1]) * 1.0e-10, x.T @ (weight * target))
    return {"kind": "radio_shape_map_geometry" if include_radio_shape else "continuous_map_geometry", "coefficient": coefficient, "mean": mean, "scale": scale,
            "ridge": float(ridge), "training_plane_count": int(len(raw)), "regions": np.zeros(0, np.int64)}


def _radio_pca_features(data: dict, rows, model: dict | None = None) -> tuple[np.ndarray, dict]:
    base = _continuous_features(data, rows, include_radio_shape=False)
    shape = np.asarray(data["query_shape"], np.float64)[rows]
    query_normal = np.asarray(data["query_normal"], np.float64)[rows]
    area_fraction = np.maximum(shape[:, 0], 1.0e-6)
    map_area = np.maximum(np.expm1(shape[:, 5]), 1.0e-6)
    incidence = np.maximum(np.abs(query_normal[:, 2]), 1.0e-3)
    proxy = np.sqrt(map_area * incidence / area_fraction) * incidence
    log_ratio = np.log(map_area) - np.log(area_fraction)
    radio = np.asarray(data["radio_full1280"], np.float64)[rows]
    radio /= np.maximum(np.linalg.norm(radio, axis=1, keepdims=True), 1.0e-8)
    if model is None:
        radio_mean = np.mean(radio, axis=0)
        _, _, right = np.linalg.svd(radio - radio_mean, full_matrices=False)
        components = right[:min(64, right.shape[0])]
        state = {"radio_mean": radio_mean, "radio_components": components}
    else:
        state = {"radio_mean": model["radio_mean"], "radio_components": model["radio_components"]}
    projected = (radio - state["radio_mean"]) @ state["radio_components"].T
    return np.column_stack((base, shape, proxy, log_ratio, projected)), state


def _fit_radio_pca(data: dict, query_mask: np.ndarray, ridge: float) -> dict:
    offsets = np.asarray(data["query_offsets"], np.int64)
    row_mask = np.zeros(len(data["region_rows"]), bool)
    for query in np.flatnonzero(query_mask).tolist():
        row_mask[int(offsets[query]):int(offsets[query + 1])] = True
    raw, state = _radio_pca_features(data, row_mask)
    mean, scale = np.mean(raw, axis=0), np.maximum(np.std(raw, axis=0), 1.0e-8)
    x = np.column_stack(((raw - mean) / scale, np.ones(len(raw))))
    target = np.asarray(data["ideal_query_offset"], np.float64)[row_mask]
    weight = np.maximum(np.asarray(data["weight"], np.float64)[row_mask], 1.0e-12)
    penalty = np.full(x.shape[1], float(ridge), np.float64); penalty[-1] = 0.0
    coefficient = np.linalg.solve(x.T @ (weight[:, None] * x) + np.diag(penalty) + np.eye(x.shape[1]) * 1.0e-10, x.T @ (weight * target))
    return {"kind": "radio_pca64_polygon", "coefficient": coefficient, "mean": mean, "scale": scale,
            "radio_mean": state["radio_mean"], "radio_components": state["radio_components"], "regions": np.zeros(0, np.int64),
            "ridge": float(ridge), "training_plane_count": int(len(raw))}


def _predict(model: dict, predicted: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model["kind"] == "continuous_map_geometry":
        raise RuntimeError("continuous prediction requires map geometry")
    coefficient = model["coefficient"]
    output = coefficient[0] * predicted + coefficient[1]
    lookup = {int(value): coefficient[2 + index] for index, value in enumerate(model["regions"].tolist())}
    seen = np.asarray([int(value) in lookup for value in region], bool)
    output += np.asarray([lookup.get(int(value), 0.0) for value in region], np.float64)
    return output, seen


def _rotation_error(rotation_c2w: np.ndarray, pose: np.ndarray) -> float:
    relative = rotation_c2w.T @ pose[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def _fit_normal_calibration(data: dict) -> np.ndarray:
    predicted = np.asarray(data["query_normal"], np.float64)
    target = np.asarray(data["ideal_query_normal"], np.float64)
    weight = np.sqrt(np.maximum(np.asarray(data["weight"], np.float64), 1.0e-12))
    matrix, _, _, _ = np.linalg.lstsq(predicted * weight[:, None], target * weight[:, None], rcond=None)
    return matrix


def _evaluate(data: dict, model: dict | None, query_mask: np.ndarray | None = None, normal_matrix: np.ndarray | None = None) -> dict:
    rows = []
    offsets = np.asarray(data["query_offsets"], np.int64)
    if query_mask is None:
        query_mask = np.ones(len(data["image_ids"]), bool)
    for query in np.flatnonzero(query_mask).tolist():
        start, stop = int(offsets[query]), int(offsets[query + 1])
        region = np.asarray(data["region_rows"], np.int64)[start:stop]
        predicted = np.asarray(data["query_offset"], np.float64)[start:stop]
        if model is None:
            corrected, seen = predicted, np.zeros(len(region), bool)
        elif model["kind"] in ("continuous_map_geometry", "radio_shape_map_geometry"):
            raw = _continuous_features(data, slice(start, stop), include_radio_shape=model["kind"] == "radio_shape_map_geometry")
            x = np.column_stack(((raw - model["mean"]) / model["scale"], np.ones(len(raw))))
            corrected, seen = x @ model["coefficient"], np.ones(len(region), bool)
        elif model["kind"] == "radio_pca64_polygon":
            raw, _ = _radio_pca_features(data, slice(start, stop), model)
            x = np.column_stack(((raw - model["mean"]) / model["scale"], np.ones(len(raw))))
            corrected, seen = x @ model["coefficient"], np.ones(len(region), bool)
        else:
            corrected, seen = _predict(model, predicted, region)
        pose = np.asarray(data["poses_w2c"], np.float64)[query]
        row = {"image_id": str(data["image_ids"][query]), "plane_count": int(stop - start), "usable": False,
               "seen_region_fraction": float(np.mean(seen)) if len(seen) else 0.0}
        if stop - start >= 4:
            weights = np.asarray(data["weight"], np.float64)[start:stop]
            query_normal = np.asarray(data["query_normal"], np.float64)[start:stop]
            if normal_matrix is not None:
                query_normal = query_normal @ normal_matrix
                query_normal /= np.maximum(np.linalg.norm(query_normal, axis=1, keepdims=True), 1.0e-12)
            rotation, _ = solve_rotation_from_plane_normals(
                query_normal,
                np.asarray(data["map_normal"], np.float64)[start:stop], weights,
            )
            center, rank, singular, _ = solve_robust_metric_translation(
                np.asarray(data["map_normal"], np.float64)[start:stop],
                np.asarray(data["map_offset"], np.float64)[start:stop], corrected, weights,
            )
            target = camera_center_from_pose_w2c(pose)
            row.update({"usable": rank == 3, "rotation_error_deg": _rotation_error(rotation, pose),
                        "translation_error_m": float(np.linalg.norm(center - target)),
                        "normal_condition": float(singular[0] / max(singular[-1], 1.0e-15))})
        rows.append(row)
    count = max(len(rows), 1)
    usable = [row for row in rows if row["usable"]]
    translation = np.asarray([row["translation_error_m"] for row in usable], np.float64)
    return {"query_count": len(rows), "usable_count": len(usable),
            "metric_1m10_recall": sum(row["usable"] and row["translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0 for row in rows) / count,
            "metric_2m45_recall": sum(row["usable"] and row["translation_error_m"] <= 2.0 and row["rotation_error_deg"] <= 45.0 for row in rows) / count,
            "translation_median_m": float(np.median(translation)) if translation.size else None,
            "translation_p90_m": float(np.quantile(translation, .9)) if translation.size else None,
            "mean_seen_region_fraction": float(np.mean([row["seen_region_fraction"] for row in rows])), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit_measurements", required=True)
    parser.add_argument("--held_measurements", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    fit_path, held_path, output = Path(args.fit_measurements).resolve(), Path(args.held_measurements).resolve(), Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite map-conditioned offset report")
    fit, held = _load(fit_path), _load(held_path)
    if fit["metadata"]["query_route"] == held["metadata"]["query_route"]:
        raise ValueError("fit and held routes must differ")
    query_index = np.arange(len(fit["image_ids"]))
    train_mask, validation_mask = query_index % 3 != 0, query_index % 3 == 0
    candidates = []
    fitters = (("region_intercept", _fit), ("continuous_map_geometry", _fit_continuous),
               ("radio_shape_map_geometry", lambda data, mask, ridge: _fit_continuous(data, mask, ridge, include_radio_shape=True)),
               ("radio_pca64_polygon", _fit_radio_pca))
    for family, fitter in fitters:
        for ridge in LAMBDA_GRID:
            model = fitter(fit, train_mask, ridge)
            score = _evaluate(fit, model, validation_mask)
            candidates.append({"family": family, "ridge": ridge, "validation": score})
    winner = max(candidates, key=lambda row: (row["validation"]["metric_1m10_recall"], row["validation"]["metric_2m45_recall"], -row["ridge"], row["family"]))
    final_fitter = dict((family, fitter) for family, fitter in fitters)[winner["family"]]
    final_model = final_fitter(fit, np.ones(len(fit["image_ids"]), bool), winner["ridge"])
    normal_matrix = _fit_normal_calibration(fit)
    raw = _evaluate(held, None)
    corrected = _evaluate(held, final_model)
    normal_only = _evaluate(held, None, normal_matrix=normal_matrix)
    combined = _evaluate(held, final_model, normal_matrix=normal_matrix)
    report = {"artifact_type": SCHEMA, "fit_route": fit["metadata"]["query_route"], "held_route": held["metadata"]["query_route"],
              "fit_measurements": str(fit_path), "fit_file_sha256": _sha(fit_path), "held_measurements": str(held_path), "held_file_sha256": _sha(held_path),
              "model": "seq9-selected low-capacity region-intercept or continuous map-geometry ridge", "lambda_grid": list(LAMBDA_GRID),
              "selection": "interleaved seq9 validation; metric 1m10, then 2m45, then lower ridge", "candidates": candidates,
              "selected_family": winner["family"], "selected_ridge": winner["ridge"], "training_plane_count": final_model["training_plane_count"],
              "training_region_count": int(len(final_model["regions"])), "raw_held": raw, "map_conditioned_held": corrected,
              "normal_calibration": "weighted 3x3 linear map fit only on fit route", "normal_only_held": normal_only,
              "combined_normal_and_offset_held": combined,
              "target": 0.8, "decision": "GO_TO_PREDICTED_MASK_GATE" if combined["metric_1m10_recall"] >= .8 else "KILL_LOW_CAPACITY_MAP_CONDITIONED_PLANE_PARAMETER_RECOVERY",
              "uses_gt_masks": True, "uses_gt_correspondence": True, "uses_alike": False, "uses_pnp": False, "uses_pose_lattice": False,
              "production_eligible": False}
    report["content_sha256"] = _canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"], "raw": raw["metric_1m10_recall"],
                      "offset_only": corrected["metric_1m10_recall"], "normal_only": normal_only["metric_1m10_recall"],
                      "combined": combined["metric_1m10_recall"]}, indent=2))


if __name__ == "__main__":
    main()
