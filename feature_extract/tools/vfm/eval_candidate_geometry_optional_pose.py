"""Evaluate frozen candidate-geometry policies as optional pose hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import (
    _evaluate_pose_strategy,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
)
from feature_extract.vfm.measurement_v1.candidate_geometry_verifier import (
    CandidateGeometryVerifier,
)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
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
                raise ValueError("duplicate candidate geometry probability identity")
            output[key] = float(row["geometry_probability"])
    return output


def _pose_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successes = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray(
        [float(row["translation_m"]) for row in successes], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_deg"]) for row in successes], dtype=np.float64
    )
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_count": int(len(successes)),
        "success_rate": 0.0 if not rows else float(len(successes) / len(rows)),
        "median_translation_m_success": None if not len(translations) else float(np.median(translations)),
        "p90_translation_m_success": None if not len(translations) else float(np.quantile(translations, 0.9)),
        "median_rotation_deg_success": None if not len(rotations) else float(np.median(rotations)),
        "median_matches": None if not rows else float(np.median([int(row.get("match_count", 0)) for row in rows])),
        "median_inliers_success": None if not successes else float(np.median([int(row.get("inlier_count", 0)) for row in successes])),
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


def _oracle_rows(rows_by_strategy: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    by_strategy_query = {
        strategy: {str(row["query_id"]): row for row in rows}
        for strategy, rows in rows_by_strategy.items()
    }
    query_ids = sorted(
        {query_id for values in by_strategy_query.values() for query_id in values}
    )
    output: list[dict[str, object]] = []
    for query_id in query_ids:
        candidates = [
            (strategy, values[query_id])
            for strategy, values in by_strategy_query.items()
            if query_id in values and bool(values[query_id].get("success"))
        ]
        if not candidates:
            output.append({"query_id": query_id, "success": False})
            continue
        strategy, chosen = min(
            candidates,
            key=lambda item: (
                float(item[1]["translation_m"]),
                float(item[1]["rotation_deg"]),
                item[0],
            ),
        )
        output.append({**dict(chosen), "oracle_strategy_TARGET_ONLY": strategy})
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_artifact", required=True)
    parser.add_argument("--geometry_probabilities_csv", required=True)
    parser.add_argument("--geometry_model", required=True)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split_name", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument("--fit_partition_modulus", type=int, default=0)
    parser.add_argument("--heldout_partition_residue", type=int, default=0)
    args = parser.parse_args(argv)

    selection, selection_metadata = _load_npz(Path(args.selection_artifact))
    policy, policy_metadata = _load_npz(Path(args.policy_artifact))
    if selection_metadata.get("format") != "candidate_measurement_selection_v1":
        raise ValueError("unsupported candidate measurement selection")
    if policy_metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported frozen baseline policy")
    if not np.array_equal(selection["selected_rows"], policy["selected_rows"]):
        raise ValueError("candidate selection and frozen policy rows differ")
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    expected = {
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    for name, value in expected.items():
        if selection_metadata.get(name) != value or policy_metadata.get(name) != value:
            raise ValueError(f"stale optional-pose input: {name}")
    split = json.loads(split_path.read_text())
    allowed = {str(value) for value in split[str(args.split_name)]}
    row_mask = np.isin(
        np.asarray(selection["query_ids"]).astype(str),
        np.asarray(sorted(allowed), dtype=np.str_),
    )
    rows = np.flatnonzero(row_mask)
    query_ids = np.asarray(selection["query_ids"]).astype(str)[rows]
    query_xy = np.asarray(selection["query_xy"], dtype=np.float64)[rows]
    source_rows = np.asarray(selection["selected_rows"], dtype=np.int64)[rows]
    tracks = np.asarray(selection["candidate_track_ids"], dtype=np.int64)[rows]
    prototypes = np.asarray(selection["candidate_prototype_ids"], dtype=np.int64)[rows]
    valid = np.asarray(selection["candidate_valid"], dtype=bool)[rows]
    prior_scores = np.asarray(selection["candidate_scores"], dtype=np.float32)[rows]
    gt_residuals = np.asarray(
        selection["candidate_gt_residuals_px"], dtype=np.float32
    )[rows]
    probabilities_by_key = _load_probabilities(Path(args.geometry_probabilities_csv))
    probabilities = np.full(tracks.shape, np.nan, dtype=np.float32)
    for local_row in range(len(rows)):
        for column in np.flatnonzero(valid[local_row]).tolist():
            key = (
                int(source_rows[local_row]),
                int(tracks[local_row, column]),
                int(prototypes[local_row, column]),
            )
            value = probabilities_by_key.get(key)
            if value is not None:
                probabilities[local_row, column] = float(value)

    landmark_index, _ = load_landmark_index_npz(bank_path)
    track_to_row = {
        int(track_id): int(row)
        for row, track_id in enumerate(landmark_index.track_ids.tolist())
    }
    canonical_rows = np.full(tracks.shape, -1, dtype=np.int64)
    for index in zip(*np.nonzero(valid)):
        canonical_rows[index] = track_to_row[int(tracks[index])]
    candidates = UniqueTrackCandidateSet(
        canonical_rows, tracks, prototypes, prior_scores
    )
    observations = [
        ColmapTrackObservation(
            track_id=int(tracks[row, 0]),
            image_id=str(query_ids[row]),
            point2d_idx=int(source_rows[row]),
            xy=(float(query_xy[row, 0]), float(query_xy[row, 1])),
            xyz=np.zeros((3,), dtype=np.float64),
            track_length=1,
            reprojection_error=0.0,
        )
        for row in range(len(rows))
    ]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    verifier = CandidateGeometryVerifier.from_dict(
        json.loads(Path(args.geometry_model).read_text())
    )

    baseline_scores = np.full(tracks.shape, -np.inf, dtype=np.float32)
    baseline_scores[:, 0] = np.asarray(
        policy["selected_pose_selection_scores"], dtype=np.float32
    )[rows]
    probability_scores = np.where(np.isfinite(probabilities), probabilities, -np.inf)
    missing_baseline = ~np.isfinite(probability_scores[:, 0])
    probability_scores[missing_baseline, 0] = 1.0
    abstaining_scores = np.full_like(probability_scores, -np.inf)
    abstaining_scores[:, 0] = probability_scores[:, 0]
    baseline_probability = probability_scores[:, 0]
    for column in range(1, tracks.shape[1]):
        alternative = probability_scores[:, column]
        promote = (
            np.isfinite(alternative)
            & (alternative >= float(verifier.promotion_probability_min))
            & (
                alternative - baseline_probability
                >= float(verifier.promotion_margin_min)
            )
        )
        abstaining_scores[promote, column] = alternative[promote]
    oracle_scores = np.where(valid & np.isfinite(gt_residuals), -gt_residuals, -np.inf)
    heldout_mask = np.zeros((len(source_rows),), dtype=bool)
    if int(args.fit_partition_modulus) > 1:
        modulus = int(args.fit_partition_modulus)
        residue = int(args.heldout_partition_residue)
        if residue < 0 or residue >= modulus:
            raise ValueError("heldout partition residue must be in [0, modulus)")
        heldout_mask = source_rows % modulus == residue
    strategies = {
        "frozen_L97_replay": baseline_scores,
        "candidate_prior_assignment": np.where(valid, prior_scores, -np.inf),
        "candidate_pgeometry_argmax_DIAGNOSTIC_ONLY": probability_scores,
        "candidate_pgeometry_abstaining": abstaining_scores,
        "candidate_topM_residual_oracle_TARGET_ONLY": oracle_scores,
    }
    common = {
        "candidates": candidates,
        "query_observations": observations,
        "query_ids": query_ids.tolist(),
        "landmark_index": landmark_index,
        "cameras": cameras,
        "images_by_name": images_by_name,
        "reprojection_error_px": float(args.pnp_reprojection_error_px),
        "iterations": int(args.pnp_iterations),
        "max_matches": int(policy_metadata.get("max_matches", 128)),
        "pose_selection_mode": str(policy_metadata.get("selection_mode", "score_topk")),
    }
    summaries: dict[str, object] = {}
    pose_rows: dict[str, list[dict[str, object]]] = {}
    for strategy, score in strategies.items():
        fit_scores = np.array(score, copy=True)
        fit_scores[heldout_mask] = -np.inf
        summary, strategy_rows = _evaluate_pose_strategy(
            strategy=strategy,
            scores=fit_scores,
            selection_scores=fit_scores,
            **common,
        )
        summaries[strategy] = summary
        pose_rows[strategy] = strategy_rows
    oracle_rows = _oracle_rows(
        {
            name: values
            for name, values in pose_rows.items()
            if name != "candidate_topM_residual_oracle_TARGET_ONLY"
        }
    )
    summaries["optional_hypothesis_oracle_TARGET_ONLY"] = _pose_summary(oracle_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for strategy, strategy_rows in pose_rows.items():
        (output / f"{strategy}.json").write_text(
            json.dumps(strategy_rows, indent=2, sort_keys=True) + "\n"
        )
    (output / "optional_hypothesis_oracle_TARGET_ONLY.json").write_text(
        json.dumps(oracle_rows, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "stage": "candidate_geometry_optional_pose",
        "split": str(args.split_name),
        "protocol": {
            "baseline_immutable": True,
            "candidate_pool_frozen": True,
            "pose_features_in_geometry_probability": False,
            "GT_used_only_by_named_oracles": True,
            "fit_partition_modulus": int(args.fit_partition_modulus),
            "heldout_partition_residue": int(args.heldout_partition_residue),
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "measurement_missing_candidate_count": int(np.sum(valid & ~np.isfinite(probabilities))),
        "heldout_token_count": int(np.sum(heldout_mask)),
        "strategy_summaries": summaries,
        "inputs": {
            "selection_artifact_sha256": file_sha256_short(Path(args.selection_artifact)),
            "geometry_probabilities_sha256": file_sha256_short(Path(args.geometry_probabilities_csv)),
            "geometry_model_sha256": file_sha256_short(Path(args.geometry_model)),
            "policy_artifact_sha256": file_sha256_short(Path(args.policy_artifact)),
            **expected,
        },
        "outputs": {"summary": str(output / "summary.json")},
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
