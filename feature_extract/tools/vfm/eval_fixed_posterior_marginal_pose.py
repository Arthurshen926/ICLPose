"""Select optional poses with a fixed candidate posterior and marginalized reprojection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


STRATEGIES = (
    "frozen_L97_replay",
    "candidate_pgeometry_argmax_DIAGNOSTIC_ONLY",
    "candidate_prior_assignment",
)


def _load_selection(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "candidate_measurement_selection_v1":
        raise ValueError("unsupported candidate measurement selection")
    return arrays, metadata


def _load_probabilities(path: Path) -> dict[tuple[int, int, int], float]:
    output: dict[tuple[int, int, int], float] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["source_query_row"]),
                int(row["track_id"]),
                int(row["prototype_id"]),
            )
            if key in output:
                raise ValueError("duplicate candidate probability identity")
            output[key] = float(row["geometry_probability"])
    return output


def _load_pose_rows(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for strategy in STRATEGIES:
        rows = json.loads((Path(directory) / f"{strategy}.json").read_text())
        output[strategy] = {str(row["query_id"]): dict(row) for row in rows}
    return output


def _pose_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray([float(row["translation_m"]) for row in successes])
    rotations = np.asarray([float(row["rotation_deg"]) for row in successes])
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_rate": 0.0 if not rows else float(len(successes) / len(rows)),
        "median_translation_m_success": None if not len(translations) else float(np.median(translations)),
        "p90_translation_m_success": None if not len(translations) else float(np.quantile(translations, 0.9)),
        "median_rotation_deg_success": None if not len(rotations) else float(np.median(rotations)),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        output[f"recall_{name}"] = float(
            np.mean(
                [
                    bool(row.get("success"))
                    and float(row.get("translation_m") or np.inf) <= distance
                    and float(row.get("rotation_deg") or np.inf) <= angle
                    for row in rows
                ]
            )
        )
    return output


def _marginal_score(
    *,
    pose_w2c: np.ndarray,
    query_xy: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_posterior: np.ndarray,
    camera,
    sigma_px: float,
    outlier_probability: float,
) -> tuple[float, int]:
    total = 0.0
    token_count = 0
    sigma2 = float(sigma_px) ** 2
    for row in range(len(query_xy)):
        valid = np.asarray(candidate_valid[row], dtype=bool)
        if not np.any(valid):
            continue
        xyz = np.asarray(candidate_xyz[row, valid], dtype=np.float64)
        projected = project_world_to_image(xyz, pose_w2c, camera)
        camera_xyz = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3][None]
        visible = np.isfinite(projected).all(axis=1) & (camera_xyz[:, 2] > 1e-6)
        visible &= (projected[:, 0] >= 0.0) & (projected[:, 0] < float(camera.width))
        visible &= (projected[:, 1] >= 0.0) & (projected[:, 1] < float(camera.height))
        residual2 = np.sum((projected - query_xy[row][None]) ** 2, axis=1)
        likelihood = np.full((len(xyz),), float(outlier_probability), dtype=np.float64)
        likelihood[visible] += np.exp(-0.5 * residual2[visible] / sigma2)
        posterior = np.asarray(candidate_posterior[row, valid], dtype=np.float64)
        posterior_sum = float(np.sum(posterior))
        if posterior_sum <= 0.0:
            continue
        posterior /= posterior_sum
        marginal = float(np.sum(posterior * likelihood))
        total += math.log(max(marginal, 1e-12))
        token_count += 1
    return (float("-inf"), 0) if token_count == 0 else (total / token_count, token_count)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_artifact", required=True)
    parser.add_argument("--geometry_probabilities_csv", required=True)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split_name", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fit_partition_modulus", type=int, default=3)
    parser.add_argument("--heldout_partition_residue", type=int, default=0)
    parser.add_argument("--sigma_px", type=float, default=4.0)
    parser.add_argument("--outlier_probability", type=float, default=1e-4)
    parser.add_argument("--promotion_margin", type=float, default=0.0)
    args = parser.parse_args(argv)
    if float(args.sigma_px) <= 0.0:
        raise ValueError("sigma_px must be positive")
    if not 0.0 < float(args.outlier_probability) < 1.0:
        raise ValueError("outlier_probability must be in (0, 1)")

    selection, selection_metadata = _load_selection(Path(args.selection_artifact))
    split_path = Path(args.split_json)
    bank_path = Path(args.projected_landmark_bank)
    if selection_metadata.get("split_json_sha256") != file_sha256_short(split_path):
        raise ValueError("selection and split manifest differ")
    if selection_metadata.get("projected_landmark_bank_sha256") != file_sha256_short(bank_path):
        raise ValueError("selection and landmark bank differ")
    split = json.loads(split_path.read_text())
    allowed = {str(value) for value in split[str(args.split_name)]}
    all_query_ids = np.asarray(selection["query_ids"]).astype(str)
    rows = np.flatnonzero(np.isin(all_query_ids, np.asarray(sorted(allowed), dtype=np.str_)))
    query_ids = all_query_ids[rows]
    source_rows = np.asarray(selection["selected_rows"], dtype=np.int64)[rows]
    query_xy = np.asarray(selection["query_xy"], dtype=np.float64)[rows]
    tracks = np.asarray(selection["candidate_track_ids"], dtype=np.int64)[rows]
    prototypes = np.asarray(selection["candidate_prototype_ids"], dtype=np.int64)[rows]
    valid = np.asarray(selection["candidate_valid"], dtype=bool)[rows]
    heldout = source_rows % int(args.fit_partition_modulus) == int(args.heldout_partition_residue)
    probabilities_by_key = _load_probabilities(Path(args.geometry_probabilities_csv))
    posterior = np.zeros(tracks.shape, dtype=np.float64)
    for row in range(len(rows)):
        for column in np.flatnonzero(valid[row]).tolist():
            posterior[row, column] = max(
                probabilities_by_key.get(
                    (
                        int(source_rows[row]),
                        int(tracks[row, column]),
                        int(prototypes[row, column]),
                    ),
                    0.0,
                ),
                0.0,
            )
        if float(np.sum(posterior[row])) <= 0.0:
            posterior[row, 0] = 1.0
        posterior[row] /= float(np.sum(posterior[row]))

    landmark_index, _ = load_landmark_index_npz(bank_path)
    track_to_row = {
        int(track_id): int(row)
        for row, track_id in enumerate(landmark_index.track_ids.tolist())
    }
    xyz = np.zeros((*tracks.shape, 3), dtype=np.float64)
    for index in zip(*np.nonzero(valid)):
        xyz[index] = landmark_index.xyz[track_to_row[int(tracks[index])]]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    pose_rows = _load_pose_rows(Path(args.pose_dir))

    selected_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for query_id in sorted(allowed):
        query_mask = (query_ids == query_id) & heldout
        image = images_by_name[query_id]
        camera = cameras[int(image.camera_id)]
        scored: list[tuple[float, str, dict[str, object]]] = []
        for strategy in STRATEGIES:
            pose_row = pose_rows[strategy].get(query_id)
            if pose_row is None or not bool(pose_row.get("success")) or pose_row.get("pose_w2c") is None:
                score, count = float("-inf"), 0
            else:
                score, count = _marginal_score(
                    pose_w2c=np.asarray(pose_row["pose_w2c"], dtype=np.float64),
                    query_xy=query_xy[query_mask],
                    candidate_xyz=xyz[query_mask],
                    candidate_valid=valid[query_mask],
                    candidate_posterior=posterior[query_mask],
                    camera=camera,
                    sigma_px=float(args.sigma_px),
                    outlier_probability=float(args.outlier_probability),
                )
            evidence_rows.append(
                {
                    "query_id": query_id,
                    "strategy": strategy,
                    "marginal_log_likelihood_per_token": score,
                    "heldout_token_count": count,
                }
            )
            if pose_row is not None:
                scored.append((score, strategy, pose_row))
        if not scored:
            selected_rows.append({"query_id": query_id, "success": False})
            continue
        baseline_items = [item for item in scored if item[1] == "frozen_L97_replay"]
        optional_items = [item for item in scored if item[1] != "frozen_L97_replay"]
        if len(baseline_items) != 1:
            raise ValueError("marginal verifier requires one frozen baseline pose")
        baseline_item = baseline_items[0]
        best_optional = max(optional_items, key=lambda item: (item[0], item[1]))
        score, strategy, pose_row = (
            best_optional
            if best_optional[0] - baseline_item[0] >= float(args.promotion_margin)
            else baseline_item
        )
        selected_rows.append(
            {
                **dict(pose_row),
                "selected_strategy": strategy,
                "marginal_log_likelihood_per_token": score,
            }
        )
    baseline_rows = [pose_rows["frozen_L97_replay"][query_id] for query_id in sorted(allowed)]
    report = {
        "stage": "fixed_posterior_marginal_pose_verification",
        "split": str(args.split_name),
        "protocol": {
            "candidate_posterior": "fixed_pose_independent_normalized_pgeometry",
            "pose_identity_reselection": False,
            "verification_partition_disjoint_from_fit": True,
            "fit_partition_modulus": int(args.fit_partition_modulus),
            "heldout_partition_residue": int(args.heldout_partition_residue),
            "sigma_px": float(args.sigma_px),
            "outlier_probability": float(args.outlier_probability),
            "promotion_margin": float(args.promotion_margin),
            "GT_pose_used_by_score": False,
        },
        "baseline_pose": _pose_summary(baseline_rows),
        "selected_pose": _pose_summary(selected_rows),
        "selection_counts": dict(
            sorted(
                {
                    strategy: sum(row.get("selected_strategy") == strategy for row in selected_rows)
                    for strategy in STRATEGIES
                }.items()
            )
        ),
        "inputs": {
            "selection_artifact_sha256": file_sha256_short(Path(args.selection_artifact)),
            "geometry_probabilities_sha256": file_sha256_short(Path(args.geometry_probabilities_csv)),
            "pose_summary_sha256": file_sha256_short(Path(args.pose_dir) / "summary.json"),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "split_json_sha256": file_sha256_short(split_path),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence_rows.json").write_text(
        json.dumps(evidence_rows, indent=2, sort_keys=True) + "\n"
    )
    (output / "selected_rows.json").write_text(
        json.dumps(selected_rows, indent=2, sort_keys=True) + "\n"
    )
    report["outputs"] = {
        "evidence_rows": str(output / "evidence_rows.json"),
        "selected_rows": str(output / "selected_rows.json"),
        "summary": str(output / "summary.json"),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
