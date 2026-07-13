"""Evaluate held-out multi-hypothesis PnP on frozen global top-L proposals.

All pose hypotheses are selected without ground-truth pose access. Ground
truth is used only after a final pose has been returned, to report validation
metrics. A reused late block is evaluated only under the explicit development
cross-block flag and can never produce a production claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_global_partial_assignment import (
    _load_frozen_baseline_policy,
    _validate_frozen_baseline_pose,
)
from feature_extract.tools.vfm.probe_detector_maplet_geometry import _pose_gate
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    CandidateSpatialLikelihood,
    GroupedCandidatePnPConfig,
    PoseVerificationCandidatePool,
    VerifiedPnPConfig,
    estimate_pose_from_grouped_candidate_pool,
    estimate_pose_with_heldout_verification,
    pose_information_diagnostics,
    resolve_pose_guided_candidate_pool,
    select_geometry_guided_generation_with_immutable_baseline,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


def _positive_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return output


def _positive_float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0.0 or not np.all(np.isfinite(output)):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return output


def _integer_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_keys",
        default="ensemble__geometry_p05px",
        help="comma-separated compact matcher score arrays",
    )
    parser.add_argument("--baseline_score_key", default="baseline_scores")
    parser.add_argument("--frozen_baseline_summary", default=None)
    parser.add_argument(
        "--frozen_baseline_source_score_key",
        default="strategy__alike_support_top2_mean",
        help="baseline score identity recorded by the frozen global sweep",
    )
    parser.add_argument(
        "--assignment_modes", default="row_argmax,global_bipartite"
    )
    parser.add_argument(
        "--fit_match_counts", type=_positive_int_list, default=(24, 32, 48, 64)
    )
    parser.add_argument(
        "--hypothesis_selection_modes",
        default="score_topk,spatial_round_robin,geometry_diverse",
    )
    parser.add_argument(
        "--ransac_thresholds_px", type=_positive_float_list, default=(2.0, 4.0, 8.0)
    )
    parser.add_argument("--rng_seed_offsets", type=_integer_list, default=(0, 1))
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--holdout_folds", type=int, default=4)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--verification_strict_px", type=float, default=2.0)
    parser.add_argument("--verification_loose_px", type=float, default=5.0)
    parser.add_argument("--final_consensus_px", type=float, default=4.0)
    parser.add_argument("--final_refine_f_scale_px", type=float, default=2.0)
    parser.add_argument("--min_final_inliers", type=int, default=6)
    parser.add_argument("--enable_final_refine", action="store_true")
    parser.add_argument(
        "--disable_topl_candidate_pool_verification", action="store_true"
    )
    parser.add_argument("--candidate_pool_residual_sigma_px", type=float, default=2.0)
    parser.add_argument("--candidate_pool_hard_threshold_px", type=float, default=8.0)
    parser.add_argument(
        "--candidate_pool_descriptor_rank_weight", type=float, default=0.02
    )
    parser.add_argument("--candidate_pool_refine_iterations", type=int, default=2)
    parser.add_argument("--candidate_evidence", default="")
    parser.add_argument("--candidate_spatial_likelihood_validation", default="")
    parser.add_argument("--candidate_spatial_likelihood_test", default="")
    parser.add_argument(
        "--candidate_spatial_log_evidence_weight", type=float, default=1.0
    )
    parser.add_argument("--candidate_geometry_probabilities_validation", default="")
    parser.add_argument("--candidate_geometry_probabilities_test", default="")
    parser.add_argument("--candidate_geometry_probabilities_train_oof", default="")
    parser.add_argument("--candidate_update_predictions_validation", default="")
    parser.add_argument("--candidate_update_predictions_test", default="")
    parser.add_argument(
        "--enable_optional_candidate_coordinate_refine", action="store_true"
    )
    parser.add_argument("--candidate_coordinate_min_updates", type=int, default=4)
    parser.add_argument(
        "--candidate_coordinate_min_grid_cells", type=int, default=2
    )
    parser.add_argument(
        "--export_train_grouped_hypotheses", action="store_true"
    )
    parser.add_argument(
        "--candidate_geometry_prior_mix_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_geometry_generation_mix_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--enable_grouped_generation_grid_fallback",
        action="store_true",
        help=(
            "Run an immutable unmixed grouped baseline and promote geometry-guided "
            "generation only when held-out strict grid coverage does not regress."
        ),
    )
    parser.add_argument(
        "--grouped_generation_min_strict_grid_delta", type=int, default=0
    )
    parser.add_argument(
        "--candidate_spatial_geometry_calibration_weight",
        type=float,
        default=0.0,
        help=(
            "Interpolate raw spatial dustbin reliability with the calibrated "
            "true-residual geometry probability without changing identity mass."
        ),
    )
    parser.add_argument("--enable_grouped_candidate_pnp", action="store_true")
    parser.add_argument(
        "--grouped_null_score_key",
        default="ensemble__set_dustbin_probability_DIAGNOSTIC_ONLY",
    )
    parser.add_argument(
        "--grouped_candidate_limits", type=_positive_int_list, default=(1, 3, 5, 10, 20)
    )
    parser.add_argument("--grouped_samples_per_limit", type=int, default=2)
    parser.add_argument(
        "--grouped_sampling_temperatures",
        type=_positive_float_list,
        default=(0.5, 1.0),
    )
    parser.add_argument(
        "--grouped_fit_match_counts", type=_positive_int_list, default=(32, 64)
    )
    parser.add_argument(
        "--grouped_ransac_thresholds_px",
        type=_positive_float_list,
        default=(2.0, 4.0),
    )
    parser.add_argument("--grouped_ransac_iterations", type=int, default=2000)
    parser.add_argument("--grouped_rank_fold_count", type=int, default=1)
    parser.add_argument(
        "--optional_grouped_final_refine_acceptance_policy",
        choices=(
            "same_as_immutable_baseline",
            "strict_count_gain_with_grid_nondecrease",
        ),
        default="same_as_immutable_baseline",
    )
    parser.add_argument("--grouped_min_fit_matches", type=int, default=8)
    parser.add_argument("--grouped_min_fit_grid_cells", type=int, default=4)
    parser.add_argument(
        "--grouped_min_xyz_second_singular_ratio", type=float, default=1e-3
    )
    parser.add_argument("--single_ransac_match_count", type=int, default=32)
    parser.add_argument("--single_ransac_selection_mode", default="score_topk")
    parser.add_argument("--single_ransac_threshold_px", type=float, default=8.0)
    parser.add_argument("--single_ransac_iterations", type=int, default=5000)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help="replay validation-frozen policies on the reused late block",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    valid = columns >= 0
    safe_columns = np.maximum(columns, 0)
    output = np.take_along_axis(np.asarray(values)[rows], safe_columns, axis=1).copy()
    if np.issubdtype(output.dtype, np.floating):
        output[~valid] = -np.inf
    else:
        output[~valid] = -1
    return output


def _selected_columns(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid & np.isfinite(scores), scores, -np.inf)
    selected = np.argmax(safe, axis=1).astype(np.int64)
    selected[~np.any(np.isfinite(safe), axis=1)] = -1
    return selected


def _score_array(
    payload: dict[str, np.ndarray],
    proposals: dict[str, np.ndarray],
    key: str,
    *,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> np.ndarray:
    source = payload.get(str(key), proposals.get(str(key)))
    if source is None:
        raise ValueError(f"score array is missing: {key}")
    values = np.asarray(source)
    if values.shape == selected_columns.shape:
        return values.astype(np.float32)
    proposal_shape = np.asarray(proposals["candidate_track_ids"]).shape
    if values.shape == proposal_shape:
        return _compact(values, selected_rows, selected_columns).astype(np.float32)
    raise ValueError(
        f"score array {key} has shape {values.shape}; expected {selected_columns.shape} "
        f"or {proposal_shape}"
    )


def _query_seed(query_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little")


def _validate_spatial_likelihood_artifact(
    path: Path,
    *,
    split_name: str,
    candidate_evidence_path: Path,
    candidate_evidence_metadata: dict[str, object],
    score_path: Path,
    proposals_path: Path,
    candidate_path: Path,
    bank_path: Path,
) -> dict[str, np.ndarray]:
    payload = _load_npz(path)
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "candidate_spatial_likelihood_v3":
        raise ValueError("unsupported candidate spatial likelihood artifact")
    if str(metadata.get("query_source")) != "real_pair" or bool(
        metadata.get("pose_or_ground_truth_used_for_inference")
    ):
        raise ValueError("candidate spatial likelihood is not pose-free real-image evidence")
    rows_csv = Path(str(metadata.get("rows_csv", "")))
    if not rows_csv.exists() or file_sha256_short(rows_csv) != metadata.get(
        "rows_csv_sha256"
    ):
        raise ValueError("candidate spatial likelihood rows CSV is stale")
    rows_summary_path = rows_csv.with_suffix(".summary.json")
    if not rows_summary_path.exists():
        raise ValueError("candidate spatial likelihood rows summary is missing")
    rows_summary = json.loads(rows_summary_path.read_text())
    if str(rows_summary.get("split")) != str(split_name):
        raise ValueError("candidate spatial likelihood split differs")
    row_inputs = rows_summary.get("inputs")
    if not isinstance(row_inputs, dict) or row_inputs.get(
        "selection_artifact_sha256"
    ) != file_sha256_short(candidate_evidence_path):
        raise ValueError("candidate spatial likelihood references different evidence")
    expected_evidence = {
        "score_artifact_sha256": file_sha256_short(score_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    mismatches = {
        key: {"expected": value, "actual": candidate_evidence_metadata.get(key)}
        for key, value in expected_evidence.items()
        if candidate_evidence_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "candidate evidence is stale or misaligned: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return payload


def _load_candidate_geometry_probability_rows(
    path: Path,
    *,
    spatial_path: Path,
    spatial_payload: dict[str, np.ndarray],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists() and (path.parent / "summary.json").exists():
        summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("candidate geometry probability artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "candidate_geometry_verifier_apply":
        raise ValueError("unsupported candidate geometry probability artifact")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "probabilities_sha256"
    ) != file_sha256_short(path):
        raise ValueError("candidate geometry probability CSV is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("candidate geometry probability inputs are missing")
    model_path = Path(str(inputs.get("model", "")))
    if not model_path.exists() or inputs.get("model_sha256") != file_sha256_short(
        model_path
    ):
        raise ValueError("candidate geometry verifier model is stale")
    model = json.loads(model_path.read_text())
    spatial_metadata = json.loads(str(spatial_payload["metadata_json"].item()))
    if model.get("measurement_checkpoint_sha256") != spatial_metadata.get(
        "measurement_checkpoint_sha256"
    ):
        raise ValueError("candidate geometry verifier checkpoint differs from spatial RGB")
    spatial_summary_path = spatial_path.parent / "summary.json"
    if not spatial_summary_path.exists():
        raise ValueError("candidate spatial diagnostic summary is missing")
    spatial_summary = json.loads(spatial_summary_path.read_text())
    spatial_outputs = spatial_summary.get("outputs")
    if not isinstance(spatial_outputs, dict) or inputs.get(
        "diagnostic_rows_sha256"
    ) != spatial_outputs.get("diagnostic_rows_sha256"):
        raise ValueError("candidate geometry probabilities use different RGB diagnostics")
    required = {
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "geometry_probability",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate geometry probability CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("candidate geometry probability CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("candidate geometry probability CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "geometry_threshold_px": float(model["geometry_threshold_px"]),
    }


def _load_candidate_update_rows(
    path: Path,
    *,
    spatial_path: Path,
    spatial_payload: dict[str, np.ndarray],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists() and (path.parent / "summary.json").exists():
        summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("candidate coordinate update artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") not in {
        "candidate_coordinate_update_verifier_fit",
        "candidate_coordinate_update_verifier_apply",
    }:
        raise ValueError("unsupported candidate coordinate update artifact")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("candidate coordinate update outputs are missing")
    recorded_hash = outputs.get(
        "validation_predictions_sha256"
        if summary.get("stage") == "candidate_coordinate_update_verifier_fit"
        else "predictions_sha256"
    )
    if recorded_hash != file_sha256_short(path):
        raise ValueError("candidate coordinate update CSV is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("candidate coordinate update inputs are missing")
    model_path = Path(
        str(
            outputs.get("model", "")
            if summary.get("stage") == "candidate_coordinate_update_verifier_fit"
            else inputs.get("model", "")
        )
    )
    model_sha = (
        outputs.get("model_sha256")
        if summary.get("stage") == "candidate_coordinate_update_verifier_fit"
        else inputs.get("model_sha256")
    )
    if not model_path.exists() or model_sha != file_sha256_short(model_path):
        raise ValueError("candidate coordinate update verifier model is stale")
    model = json.loads(model_path.read_text())
    if model.get("format") != "candidate_coordinate_update_verifier_v1":
        raise ValueError("unsupported candidate coordinate update model")
    spatial_metadata = json.loads(str(spatial_payload["metadata_json"].item()))
    if model.get("measurement_checkpoint_sha256") != spatial_metadata.get(
        "measurement_checkpoint_sha256"
    ):
        raise ValueError("candidate coordinate update checkpoint differs from spatial RGB")
    spatial_summary_path = spatial_path.parent / "summary.json"
    if not spatial_summary_path.exists():
        raise ValueError("candidate spatial diagnostic summary is missing")
    spatial_summary = json.loads(spatial_summary_path.read_text())
    spatial_outputs = spatial_summary.get("outputs")
    diagnostic_hash = inputs.get(
        "validation_diagnostic_rows_sha256"
        if summary.get("stage") == "candidate_coordinate_update_verifier_fit"
        else "diagnostic_rows_sha256"
    )
    if not isinstance(spatial_outputs, dict) or diagnostic_hash != spatial_outputs.get(
        "diagnostic_rows_sha256"
    ):
        raise ValueError("candidate coordinate updates use different RGB diagnostics")
    required = {
        "candidate_identity_key",
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "update_beneficial_probability",
        "center_x",
        "center_y",
        "refined_x",
        "refined_y",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate coordinate update CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("candidate coordinate update CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("candidate coordinate update CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "update_threshold": float(model["update_threshold"]),
        "candidate_geometry_verifier_sha256": str(
            model["candidate_geometry_verifier_sha256"]
        ),
    }


def _load_train_oof_candidate_geometry_probability_rows(
    path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("train OOF candidate geometry artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "candidate_geometry_verifier_fit":
        raise ValueError("unsupported train OOF candidate geometry artifact")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "train_oof_probabilities_sha256"
    ) != file_sha256_short(path):
        raise ValueError("train OOF candidate geometry CSV is stale")
    model_path = Path(str(outputs.get("model", "")))
    if not model_path.exists() or outputs.get("model_sha256") != file_sha256_short(
        model_path
    ):
        raise ValueError("train OOF candidate geometry verifier model is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("train OOF candidate geometry inputs are missing")
    diagnostic_path = Path(str(inputs.get("train_diagnostic_rows_csv", "")))
    if not diagnostic_path.exists() or inputs.get(
        "train_diagnostic_rows_sha256"
    ) != file_sha256_short(diagnostic_path):
        raise ValueError("train OOF candidate geometry diagnostics are stale")
    model = json.loads(model_path.read_text())
    required = {
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "geometry_probability",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("train OOF candidate geometry CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("train OOF candidate geometry CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("train OOF candidate geometry CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "geometry_threshold_px": float(model["geometry_threshold_px"]),
        "probability_source": "query_grouped_out_of_fold",
        "diagnostic_rows": str(diagnostic_path),
        "diagnostic_rows_sha256": file_sha256_short(diagnostic_path),
    }


def _set_cv2_seed(seed: int) -> None:
    try:
        import cv2

        cv2.setRNGSeed(int(int(seed) % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _pose_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    success = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray(
        [float(row["translation_m"]) for row in success], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_deg"]) for row in success], dtype=np.float64
    )
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_count": int(len(success)),
        "success_rate": 0.0 if not rows else float(len(success) / len(rows)),
        "median_translation_m_success": (
            None if translations.size == 0 else float(np.median(translations))
        ),
        "p90_translation_m_success": (
            None if translations.size == 0 else float(np.percentile(translations, 90))
        ),
        "median_rotation_deg_success": (
            None if rotations.size == 0 else float(np.median(rotations))
        ),
        "median_matches": (
            None
            if not rows
            else float(np.median([int(row.get("match_count", 0)) for row in rows]))
        ),
        "median_inliers_success": (
            None
            if not success
            else float(np.median([int(row.get("inlier_count", 0)) for row in success]))
        ),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        output[f"recall_{name}"] = (
            0.0
            if not rows
            else float(
                np.mean(
                    [
                        bool(row.get("success"))
                        and float(row["translation_m"]) <= distance
                        and float(row["rotation_deg"]) <= angle
                        for row in rows
                    ]
                )
            )
        )
    pre_refine = [
        row
        for row in rows
        if row.get("pre_refine_translation_m") is not None
        and row.get("pre_refine_rotation_deg") is not None
    ]
    oracle = [
        row
        for row in rows
        if row.get("hypothesis_oracle_translation_m") is not None
        and row.get("hypothesis_oracle_rotation_deg") is not None
    ]
    if pre_refine:
        output["pre_refine_median_translation_m"] = float(
            np.median([float(row["pre_refine_translation_m"]) for row in pre_refine])
        )
        output["pre_refine_p90_translation_m"] = float(
            np.percentile(
                [float(row["pre_refine_translation_m"]) for row in pre_refine], 90
            )
        )
        output["pre_refine_median_rotation_deg"] = float(
            np.median([float(row["pre_refine_rotation_deg"]) for row in pre_refine])
        )
        output["median_final_minus_pre_refine_translation_m"] = float(
            np.median(
                [
                    float(row["translation_m"])
                    - float(row["pre_refine_translation_m"])
                    for row in pre_refine
                    if row.get("translation_m") is not None
                ]
            )
        )
    if oracle:
        output["hypothesis_oracle_median_translation_m"] = float(
            np.median(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle]
            )
        )
        output["hypothesis_oracle_p90_translation_m"] = float(
            np.percentile(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle], 90
            )
        )
        output["hypothesis_oracle_median_rotation_deg"] = float(
            np.median([float(row["hypothesis_oracle_rotation_deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_10cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_10cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_5cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_5cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_3cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_3cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_25cm_2deg"] = float(
            np.mean([bool(row["hypothesis_oracle_25cm_2deg"]) for row in oracle])
        )
        output["median_chosen_hypothesis_translation_rank"] = float(
            np.median(
                [float(row["chosen_hypothesis_translation_rank"]) for row in oracle]
            )
        )
        for key in (
            "valid_hypothesis_count",
            "hypothesis_3cm_5deg_count",
            "hypothesis_10cm_5deg_count",
            "hypothesis_25cm_2deg_count",
            "catastrophic_hypothesis_count",
        ):
            output[f"median_{key}"] = float(
                np.median([int(row[key]) for row in oracle])
            )
        output["median_hypothesis_selection_regret_m"] = float(
            np.median(
                [
                    float(row["translation_m"])
                    - float(row["hypothesis_oracle_translation_m"])
                    for row in oracle
                    if row.get("translation_m") is not None
                ]
            )
        )
    return output


def _policy_key(trial: dict[str, object]) -> tuple[float, ...]:
    pose = dict(trial["verified_pose"])
    values = np.asarray(
        [
            float(pose["median_translation_m_success"]),
            float(pose["p90_translation_m_success"]),
            float(pose["median_rotation_deg_success"]),
        ],
        dtype=np.float64,
    )
    geomean = float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))
    return (
        float(pose["success_rate"]),
        -geomean,
        -float(pose["p90_translation_m_success"]),
        -float(pose["median_translation_m_success"]),
        -float(pose["median_rotation_deg_success"]),
        float(pose["recall_10cm_5deg"]),
        float(pose["recall_5cm_5deg"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.development_cross_block_audit) and str(args.evaluation_role) != "development":
        raise ValueError("cross-block replay is development-only")
    if int(args.single_ransac_match_count) < 4:
        raise ValueError("single_ransac_match_count must be at least four")
    if str(args.single_ransac_selection_mode) not in {
        "score_topk",
        "spatial_round_robin",
    }:
        raise ValueError("unsupported single-RANSAC selection mode")
    assignment_modes = tuple(
        item.strip() for item in str(args.assignment_modes).split(",") if item.strip()
    )
    if not assignment_modes or set(assignment_modes) - {
        "row_argmax",
        "global_bipartite",
    }:
        raise ValueError("unsupported assignment mode")
    hypothesis_modes = tuple(
        item.strip()
        for item in str(args.hypothesis_selection_modes).split(",")
        if item.strip()
    )
    score_keys = tuple(
        item.strip() for item in str(args.score_keys).split(",") if item.strip()
    )
    if not score_keys:
        raise ValueError("score_keys cannot be empty")

    proposals_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    score_path = Path(args.score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    score_payload = _load_npz(score_path)
    spatial_paths = {
        "validation": (
            None
            if not str(args.candidate_spatial_likelihood_validation)
            else Path(args.candidate_spatial_likelihood_validation)
        ),
        "test": (
            None
            if not str(args.candidate_spatial_likelihood_test)
            else Path(args.candidate_spatial_likelihood_test)
        ),
    }
    spatial_requested = any(path is not None for path in spatial_paths.values())
    if spatial_requested != bool(str(args.candidate_evidence)):
        raise ValueError(
            "candidate_evidence and candidate spatial likelihoods must be provided together"
        )
    if spatial_requested and any(path is None for path in spatial_paths.values()):
        raise ValueError("both validation and test spatial likelihoods are required")
    if not 0.0 <= float(args.candidate_spatial_log_evidence_weight) <= 1.0:
        raise ValueError("candidate spatial log evidence weight must be in [0, 1]")
    if not 0.0 <= float(args.candidate_geometry_prior_mix_weight) <= 1.0:
        raise ValueError("candidate geometry prior mix weight must be in [0, 1]")
    if not 0.0 <= float(args.candidate_geometry_generation_mix_weight) <= 1.0:
        raise ValueError("candidate geometry generation mix weight must be in [0, 1]")
    if int(args.grouped_generation_min_strict_grid_delta) < 0:
        raise ValueError("grouped generation strict grid delta must be non-negative")
    if bool(args.enable_grouped_generation_grid_fallback) and (
        not bool(args.enable_grouped_candidate_pnp)
        or float(args.candidate_geometry_generation_mix_weight) <= 0.0
    ):
        raise ValueError(
            "grouped generation grid fallback requires grouped candidate PnP and "
            "positive geometry generation mixing"
        )
    if not 0.0 <= float(
        args.candidate_spatial_geometry_calibration_weight
    ) <= 1.0:
        raise ValueError(
            "candidate spatial geometry calibration weight must be in [0, 1]"
        )
    candidate_evidence_path = (
        None if not spatial_requested else Path(args.candidate_evidence)
    )
    candidate_evidence = (
        None if candidate_evidence_path is None else _load_npz(candidate_evidence_path)
    )
    candidate_evidence_metadata = (
        {}
        if candidate_evidence is None
        else json.loads(str(candidate_evidence["metadata_json"].item()))
    )
    spatial_payloads = (
        {}
        if candidate_evidence_path is None
        else {
            split_name: _validate_spatial_likelihood_artifact(
                path,
                split_name=split_name,
                candidate_evidence_path=candidate_evidence_path,
                candidate_evidence_metadata=candidate_evidence_metadata,
                score_path=score_path,
                proposals_path=proposals_path,
                candidate_path=candidate_path,
                bank_path=bank_path,
            )
            for split_name, path in spatial_paths.items()
            if path is not None
        }
    )
    geometry_probability_paths = {
        "validation": (
            None
            if not str(args.candidate_geometry_probabilities_validation)
            else Path(args.candidate_geometry_probabilities_validation)
        ),
        "test": (
            None
            if not str(args.candidate_geometry_probabilities_test)
            else Path(args.candidate_geometry_probabilities_test)
        ),
        "train": (
            None
            if not str(args.candidate_geometry_probabilities_train_oof)
            else Path(args.candidate_geometry_probabilities_train_oof)
        ),
    }
    update_prediction_paths = {
        "validation": (
            None
            if not str(args.candidate_update_predictions_validation)
            else Path(args.candidate_update_predictions_validation)
        ),
        "test": (
            None
            if not str(args.candidate_update_predictions_test)
            else Path(args.candidate_update_predictions_test)
        ),
    }
    update_predictions_requested = any(
        path is not None for path in update_prediction_paths.values()
    )
    if update_predictions_requested and any(
        path is None for path in update_prediction_paths.values()
    ):
        raise ValueError("both validation and test candidate updates are required")
    if bool(args.enable_optional_candidate_coordinate_refine) and not (
        update_predictions_requested
        and spatial_requested
        and bool(args.enable_grouped_candidate_pnp)
    ):
        raise ValueError(
            "candidate coordinate refinement requires grouped PnP plus aligned "
            "spatial and update artifacts"
        )
    if int(args.candidate_coordinate_min_updates) < 4:
        raise ValueError("candidate coordinate refinement needs at least four updates")
    if int(args.candidate_coordinate_min_grid_cells) <= 0:
        raise ValueError("candidate coordinate update grid coverage must be positive")
    if update_predictions_requested and not spatial_requested:
        raise ValueError("candidate coordinate updates require spatial evidence")
    evaluation_geometry_probabilities_requested = any(
        geometry_probability_paths[name] is not None
        for name in ("validation", "test")
    )
    if evaluation_geometry_probabilities_requested and any(
        geometry_probability_paths[name] is None
        for name in ("validation", "test")
    ):
        raise ValueError("both validation and test geometry probabilities are required")
    train_geometry_probabilities_requested = (
        geometry_probability_paths["train"] is not None
    )
    geometry_probabilities_requested = bool(
        evaluation_geometry_probabilities_requested
        or train_geometry_probabilities_requested
    )
    if bool(args.export_train_grouped_hypotheses) and not (
        bool(args.enable_grouped_candidate_pnp)
        and train_geometry_probabilities_requested
    ):
        raise ValueError(
            "train grouped hypothesis export requires grouped candidate PnP and "
            "train OOF geometry probabilities"
        )
    if float(args.candidate_geometry_prior_mix_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "geometry prior mixing requires aligned geometry and spatial artifacts"
        )
    if float(args.candidate_geometry_generation_mix_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "geometry-guided generation requires aligned geometry and spatial artifacts"
        )
    if float(args.candidate_spatial_geometry_calibration_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "spatial geometry calibration requires aligned geometry and spatial artifacts"
        )
    if evaluation_geometry_probabilities_requested and not spatial_requested:
        raise ValueError("evaluation geometry probabilities require spatial evidence")
    geometry_probability_rows: dict[str, list[dict[str, str]]] = {}
    geometry_probability_metadata: dict[str, dict[str, object]] = {}
    if geometry_probabilities_requested:
        for split_name, path in geometry_probability_paths.items():
            if path is None:
                continue
            if split_name == "train":
                rows, metadata = (
                    _load_train_oof_candidate_geometry_probability_rows(path)
                )
            else:
                spatial_path = spatial_paths[split_name]
                if spatial_path is None:
                    raise RuntimeError(
                        "candidate geometry artifact configuration is incomplete"
                    )
                rows, metadata = _load_candidate_geometry_probability_rows(
                    path,
                    spatial_path=spatial_path,
                    spatial_payload=spatial_payloads[split_name],
                )
            geometry_probability_rows[split_name] = rows
            geometry_probability_metadata[split_name] = metadata
    update_prediction_rows: dict[str, list[dict[str, str]]] = {}
    update_prediction_metadata: dict[str, dict[str, object]] = {}
    if update_predictions_requested:
        for split_name, path in update_prediction_paths.items():
            spatial_path = spatial_paths[split_name]
            if path is None or spatial_path is None:
                raise RuntimeError("candidate update artifact configuration is incomplete")
            rows, metadata = _load_candidate_update_rows(
                path,
                spatial_path=spatial_path,
                spatial_payload=spatial_payloads[split_name],
            )
            geometry_metadata = geometry_probability_metadata.get(split_name)
            if geometry_metadata is None or metadata[
                "candidate_geometry_verifier_sha256"
            ] != geometry_metadata.get("model_sha256"):
                raise ValueError(
                    "candidate updates and candidate geometry probabilities differ"
                )
            update_prediction_rows[split_name] = rows
            update_prediction_metadata[split_name] = metadata
        if len(
            {
                (
                    str(metadata["model_sha256"]),
                    float(metadata["update_threshold"]),
                )
                for metadata in update_prediction_metadata.values()
            }
        ) != 1:
            raise ValueError("validation/test candidate updates use different models")
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidate["valid_edges"], dtype=bool)
    if candidate_evidence is not None:
        if not np.array_equal(candidate_evidence["selected_rows"], selected_rows):
            raise ValueError("candidate evidence rows differ from candidate artifact")
        for split_name, spatial_payload in spatial_payloads.items():
            source_rows = np.asarray(
                spatial_payload["source_query_rows"], dtype=np.int64
            )
            if not np.all(np.isin(source_rows, selected_rows)):
                raise ValueError(
                    f"{split_name} spatial likelihood contains unknown proposal rows"
                )
    if selected_columns.ndim != 2 or valid_edges.shape != selected_columns.shape:
        raise ValueError("candidate artifact arrays have incompatible shapes")
    candidate_metadata = json.loads(str(candidate["metadata_json"].item()))
    expected_candidate = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    candidate_mismatches = {
        key: {"expected": value, "actual": candidate_metadata.get(key)}
        for key, value in expected_candidate.items()
        if candidate_metadata.get(key) != value
    }
    if candidate_mismatches:
        raise ValueError(
            "stale candidate artifact: "
            f"{json.dumps(candidate_mismatches, sort_keys=True)}"
        )
    score_summary_path = score_path.parent / "summary.json"
    if not score_summary_path.exists():
        raise ValueError("score artifact requires sibling summary.json")
    score_summary = json.loads(score_summary_path.read_text())
    score_manifest = score_summary.get("data_manifest")
    if not isinstance(score_manifest, dict):
        raise ValueError("score summary is missing data_manifest")
    score_outputs = score_summary.get("outputs")
    recorded_score_hash = (
        None
        if not isinstance(score_outputs, dict)
        else score_outputs.get("scores_sha256")
    )
    actual_score_hash = file_sha256_short(score_path)
    if not recorded_score_hash or str(recorded_score_hash) != actual_score_hash:
        raise ValueError("score artifact hash differs from its sibling summary")
    expected_score = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "feature_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    score_mismatches = {
        key: {"expected": value, "actual": score_manifest.get(key)}
        for key, value in expected_score.items()
        if score_manifest.get(key) != value
    }
    if score_mismatches:
        raise ValueError(
            "stale or misaligned score artifact: "
            f"{json.dumps(score_mismatches, sort_keys=True)}"
        )
    frozen_baseline_source = (
        None
        if args.frozen_baseline_summary is None
        else _load_frozen_baseline_policy(
            Path(args.frozen_baseline_summary),
            proposals_path=proposals_path,
            candidate_path=candidate_path,
            bank_path=bank_path,
            split_path=split_path,
            baseline_score_key=str(args.frozen_baseline_source_score_key),
        )
    )
    if frozen_baseline_source is not None:
        score_protocol = score_summary.get("protocol")
        score_baseline = (
            None
            if not isinstance(score_protocol, dict)
            else score_protocol.get("baseline_strategy")
        )
        normalized_score_baseline = str(score_baseline)
        if not normalized_score_baseline.startswith("strategy__"):
            normalized_score_baseline = f"strategy__{normalized_score_baseline}"
        if normalized_score_baseline != str(args.frozen_baseline_source_score_key):
            raise ValueError(
                "score artifact baseline identity differs from frozen baseline"
            )
        if (
            int(args.single_ransac_match_count)
            != int(frozen_baseline_source["max_matches"])
            or str(args.single_ransac_selection_mode)
            != str(frozen_baseline_source["selection_mode"])
        ):
            raise ValueError(
                "single-RANSAC policy must replay the frozen baseline K/mode"
            )

    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    compact_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    compact_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    canonical_rows = canonical_rows_for_track_candidates(
        compact_tracks, landmark_index.track_ids
    )
    valid_edges &= canonical_rows >= 0
    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    if candidate_evidence is not None and (
        not np.array_equal(candidate_evidence["query_ids"].astype(str), query_ids)
        or not np.allclose(candidate_evidence["query_xy"], query_xy, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("candidate evidence query rows differ from evaluator rows")
    split = json.loads(split_path.read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list) or not split[name]:
            raise ValueError("split JSON requires non-empty train/validation/test lists")
    split_masks = {
        name: np.isin(query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("train", "validation", "test")
    }
    if any(
        np.any(split_masks[left] & split_masks[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("train/validation/late splits overlap")
    geometry_probability_matrices: dict[str, np.ndarray] = {}
    if geometry_probabilities_requested:
        if candidate_evidence is None:
            raise RuntimeError("candidate geometry probabilities require candidate evidence")
        evidence_columns = np.asarray(
            candidate_evidence["candidate_compact_columns"], dtype=np.int64
        )
        compact_row_by_source = {
            int(source_row): int(compact_row)
            for compact_row, source_row in enumerate(selected_rows.tolist())
        }
        for split_name, probability_rows in geometry_probability_rows.items():
            matrix = np.full(valid_edges.shape, np.nan, dtype=np.float64)
            seen: set[tuple[int, int]] = set()
            for probability_row in probability_rows:
                source_row = int(probability_row["source_query_row"])
                compact_row = compact_row_by_source.get(source_row)
                if compact_row is None or not bool(split_masks[split_name][compact_row]):
                    raise ValueError(
                        "candidate geometry probability row has an unknown split/source row"
                    )
                candidate_rank = int(
                    probability_row["candidate_measurement_rank"]
                ) - 1
                if not 0 <= candidate_rank < evidence_columns.shape[1]:
                    raise ValueError("candidate geometry rank is outside evidence top-M")
                column = int(evidence_columns[compact_row, candidate_rank])
                if column < 0 or not bool(valid_edges[compact_row, column]):
                    raise ValueError("candidate geometry probability slot is invalid")
                identity = (compact_row, column)
                if identity in seen:
                    raise ValueError("duplicate candidate geometry probability slot")
                seen.add(identity)
                if (
                    str(probability_row["query_id"]) != str(query_ids[compact_row])
                    or int(probability_row["track_id"])
                    != int(compact_tracks[compact_row, column])
                    or int(probability_row["prototype_id"])
                    != int(compact_prototypes[compact_row, column])
                ):
                    raise ValueError("candidate geometry probability identity differs")
                probability = float(probability_row["geometry_probability"])
                if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("candidate geometry probability is outside [0, 1]")
                matrix[compact_row, column] = probability
            if not seen:
                raise ValueError("candidate geometry probability artifact aligned no rows")
            geometry_probability_matrices[split_name] = matrix
    update_probability_matrices: dict[str, np.ndarray] = {}
    update_refined_xy_matrices: dict[str, np.ndarray] = {}
    if update_predictions_requested:
        if candidate_evidence is None:
            raise RuntimeError("candidate updates require candidate evidence")
        evidence_columns = np.asarray(
            candidate_evidence["candidate_compact_columns"], dtype=np.int64
        )
        compact_row_by_source = {
            int(source_row): int(compact_row)
            for compact_row, source_row in enumerate(selected_rows.tolist())
        }
        for split_name, prediction_rows in update_prediction_rows.items():
            spatial_payload = spatial_payloads[split_name]
            spatial_identity_by_slot: dict[tuple[int, int], str] = {}
            for source_row, measurement_rank, identity in zip(
                np.asarray(spatial_payload["source_query_rows"], dtype=np.int64),
                np.asarray(
                    spatial_payload["candidate_measurement_ranks"], dtype=np.int64
                ),
                np.asarray(spatial_payload["candidate_identity_keys"]).astype(str),
            ):
                key = (int(source_row), int(measurement_rank) - 1)
                previous = spatial_identity_by_slot.setdefault(key, str(identity))
                if previous != str(identity):
                    raise ValueError("spatial support views disagree on candidate identity")
            probability_matrix = np.full(valid_edges.shape, np.nan, dtype=np.float64)
            refined_matrix = np.full(
                (*valid_edges.shape, 2), np.nan, dtype=np.float64
            )
            seen: set[tuple[int, int]] = set()
            for prediction_row in prediction_rows:
                source_row = int(prediction_row["source_query_row"])
                compact_row = compact_row_by_source.get(source_row)
                if compact_row is None or not bool(split_masks[split_name][compact_row]):
                    raise ValueError(
                        "candidate update row has an unknown split/source row"
                    )
                candidate_rank = int(
                    prediction_row["candidate_measurement_rank"]
                ) - 1
                if not 0 <= candidate_rank < evidence_columns.shape[1]:
                    raise ValueError("candidate update rank is outside evidence top-M")
                column = int(evidence_columns[compact_row, candidate_rank])
                if column < 0 or not bool(valid_edges[compact_row, column]):
                    raise ValueError("candidate update probability slot is invalid")
                slot = (compact_row, column)
                if slot in seen:
                    raise ValueError("duplicate candidate update probability slot")
                seen.add(slot)
                expected_identity = spatial_identity_by_slot.get(
                    (source_row, candidate_rank)
                )
                if expected_identity != str(
                    prediction_row["candidate_identity_key"]
                ):
                    raise ValueError("candidate update identity differs from spatial RGB")
                if (
                    str(prediction_row["query_id"]) != str(query_ids[compact_row])
                    or int(prediction_row["track_id"])
                    != int(compact_tracks[compact_row, column])
                    or int(prediction_row["prototype_id"])
                    != int(compact_prototypes[compact_row, column])
                ):
                    raise ValueError("candidate update candidate identity differs")
                center = np.asarray(
                    [
                        float(prediction_row["center_x"]),
                        float(prediction_row["center_y"]),
                    ],
                    dtype=np.float64,
                )
                if not np.allclose(
                    center, query_xy[compact_row], rtol=0.0, atol=1e-4
                ):
                    raise ValueError("candidate update center differs from coarse query xy")
                probability = float(
                    prediction_row["update_beneficial_probability"]
                )
                refined = np.asarray(
                    [
                        float(prediction_row["refined_x"]),
                        float(prediction_row["refined_y"]),
                    ],
                    dtype=np.float64,
                )
                if (
                    not np.isfinite(probability)
                    or not 0.0 <= probability <= 1.0
                    or not np.all(np.isfinite(refined))
                ):
                    raise ValueError("candidate coordinate update values are invalid")
                probability_matrix[compact_row, column] = probability
                refined_matrix[compact_row, column] = refined
            if not seen:
                raise ValueError("candidate update artifact aligned no rows")
            update_probability_matrices[split_name] = probability_matrix
            update_refined_xy_matrices[split_name] = refined_matrix
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    config = VerifiedPnPConfig(
        fit_match_counts=tuple(args.fit_match_counts),
        selection_modes=hypothesis_modes,
        ransac_thresholds_px=tuple(args.ransac_thresholds_px),
        rng_seed_offsets=tuple(args.rng_seed_offsets),
        ransac_iterations=int(args.ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        holdout_fold=int(args.holdout_fold),
        verification_strict_px=float(args.verification_strict_px),
        verification_loose_px=float(args.verification_loose_px),
        final_consensus_px=float(args.final_consensus_px),
        final_refine_f_scale_px=float(args.final_refine_f_scale_px),
        min_final_inliers=int(args.min_final_inliers),
        enable_final_refine=bool(args.enable_final_refine),
        candidate_pool_residual_sigma_px=float(
            args.candidate_pool_residual_sigma_px
        ),
        candidate_pool_hard_threshold_px=float(
            args.candidate_pool_hard_threshold_px
        ),
        candidate_pool_descriptor_rank_weight=float(
            args.candidate_pool_descriptor_rank_weight
        ),
        candidate_pool_refine_iterations=int(args.candidate_pool_refine_iterations),
    )
    grouped_config = None if not bool(args.enable_grouped_candidate_pnp) else GroupedCandidatePnPConfig(
        candidate_limits=tuple(args.grouped_candidate_limits),
        samples_per_limit=int(args.grouped_samples_per_limit),
        sampling_temperatures=tuple(args.grouped_sampling_temperatures),
        fit_match_counts=tuple(args.grouped_fit_match_counts),
        ransac_thresholds_px=tuple(args.grouped_ransac_thresholds_px),
        ransac_iterations=int(args.grouped_ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        verification_fold=int(args.holdout_fold),
        verification_fold_count=int(args.grouped_rank_fold_count),
        final_audit_fold=(
            int(args.holdout_fold) + int(args.grouped_rank_fold_count)
        )
        % int(args.holdout_folds),
        min_fit_matches=int(args.grouped_min_fit_matches),
        min_fit_grid_cells=int(args.grouped_min_fit_grid_cells),
        min_xyz_second_singular_ratio=float(
            args.grouped_min_xyz_second_singular_ratio
        ),
        verification_strict_px=float(args.verification_strict_px),
        verification_loose_px=float(args.verification_loose_px),
        candidate_pool_residual_sigma_px=float(
            args.candidate_pool_residual_sigma_px
        ),
        candidate_pool_hard_threshold_px=float(
            args.candidate_pool_hard_threshold_px
        ),
        candidate_pool_descriptor_rank_weight=float(
            args.candidate_pool_descriptor_rank_weight
        ),
        final_consensus_px=float(args.final_consensus_px),
        final_refine_f_scale_px=float(args.final_refine_f_scale_px),
        min_final_inliers=int(args.min_final_inliers),
        enable_final_refine=bool(args.enable_final_refine),
        final_refine_acceptance_policy="legacy_rank_key",
    )
    optional_grouped_config = grouped_config
    if (
        grouped_config is not None
        and str(args.optional_grouped_final_refine_acceptance_policy)
        != "same_as_immutable_baseline"
    ):
        optional_grouped_config = replace(
            grouped_config,
            final_refine_acceptance_policy=str(
                args.optional_grouped_final_refine_acceptance_policy
            ),
        )
    if grouped_config is not None and bool(
        args.enable_optional_candidate_coordinate_refine
    ):
        optional_grouped_config = replace(
            optional_grouped_config,
            enable_candidate_coordinate_refine=True,
            min_candidate_coordinate_updates=int(
                args.candidate_coordinate_min_updates
            ),
            min_candidate_coordinate_update_grid_cells=int(
                args.candidate_coordinate_min_grid_cells
            ),
        )
    raw_scores = {
        key: _score_array(
            score_payload,
            proposals,
            key,
            selected_rows=selected_rows,
            selected_columns=selected_columns,
        )
        for key in (*score_keys, str(args.baseline_score_key))
    }
    grouped_null_scores = (
        None
        if not bool(args.enable_grouped_candidate_pnp)
        else _score_array(
            score_payload,
            proposals,
            str(args.grouped_null_score_key),
            selected_rows=selected_rows,
            selected_columns=selected_columns,
        )
    )
    for values in raw_scores.values():
        values[~valid_edges] = -np.inf

    policy_scores: dict[str, np.ndarray] = {}
    policy_metadata: dict[str, dict[str, object]] = {}
    for score_key, values in raw_scores.items():
        if "row_argmax" in assignment_modes:
            key = f"row_argmax__{score_key}"
            policy_scores[key] = values
            policy_metadata[key] = {
                "assignment_mode": "row_argmax_then_conflict_resolution",
                "score_key": score_key,
            }
        if "global_bipartite" in assignment_modes:
            key = f"global_bipartite__{score_key}"
            resolved, _selected = global_assignment_score_matrix(
                compact_tracks,
                values,
                query_ids,
                valid_mask=valid_edges,
                dustbin_score=None,
            )
            policy_scores[key] = resolved
            policy_metadata[key] = {
                "assignment_mode": "whole_image_sparse_bipartite_per_query_dustbin",
                "score_key": score_key,
            }

    def matches_by_query(scores: np.ndarray, split_name: str) -> dict[str, list[QueryTo3DMatch]]:
        selected = _selected_columns(scores, valid_edges)
        output: dict[str, list[QueryTo3DMatch]] = {}
        for row in np.flatnonzero(split_masks[split_name]).tolist():
            column = int(selected[row])
            if column < 0:
                continue
            query_id = str(query_ids[row])
            output.setdefault(query_id, []).append(
                QueryTo3DMatch(
                    token_index=int(selected_rows[row]),
                    xy=np.asarray(query_xy[row], dtype=np.float64),
                    track_id=int(compact_tracks[row, column]),
                    xyz=np.asarray(
                        landmark_index.xyz[int(canonical_rows[row, column])],
                        dtype=np.float64,
                    ),
                    similarity=float(scores[row, column]),
                    ratio=0.0,
                    landmark_variance=float(
                        landmark_index.mean_variances[
                            int(canonical_rows[row, column])
                        ]
                    ),
                    source="heldout_pose_hypothesis_eval",
                    prototype_id=int(compact_prototypes[row, column]),
                )
            )
        return output

    def candidate_pools_by_query(
        scores: np.ndarray,
        split_name: str,
        *,
        null_scores: np.ndarray | None = None,
        spatial_payload: dict[str, np.ndarray] | None = None,
        geometry_probabilities: np.ndarray | None = None,
        update_probabilities: np.ndarray | None = None,
        update_refined_xy: np.ndarray | None = None,
        update_threshold: float = 0.5,
    ) -> dict[str, PoseVerificationCandidatePool]:
        output: dict[str, PoseVerificationCandidatePool] = {}
        for query_id in split[split_name]:
            rows = np.flatnonzero(
                split_masks[split_name] & (query_ids == str(query_id))
            )
            xyz = np.zeros((*canonical_rows[rows].shape, 3), dtype=np.float64)
            local_valid = valid_edges[rows]
            xyz[local_valid] = landmark_index.xyz[
                canonical_rows[rows][local_valid]
            ]
            local_null = None
            if null_scores is not None:
                repeated = np.asarray(null_scores[rows], dtype=np.float64)
                if repeated.ndim != 2 or repeated.shape[1] == 0:
                    raise ValueError("grouped null score array has an invalid shape")
                if not np.allclose(
                    repeated, repeated[:, :1], rtol=0.0, atol=2e-5
                ):
                    raise ValueError(
                        "grouped null score must be constant inside each candidate set"
                    )
                local_null = repeated[:, 0]
            spatial_likelihood = None
            if spatial_payload is not None:
                if candidate_evidence is None:
                    raise RuntimeError("spatial likelihood requires candidate evidence")
                offsets = np.asarray(spatial_payload["offsets_xy"], dtype=np.float64)
                source_rows = np.asarray(
                    spatial_payload["source_query_rows"], dtype=np.int64
                )
                source_ranks = np.asarray(
                    spatial_payload["candidate_measurement_ranks"], dtype=np.int64
                ) - 1
                view_ranks = np.asarray(
                    spatial_payload["support_view_ranks"], dtype=np.int64
                )
                max_views = max(
                    1,
                    int(
                        np.max(view_ranks, initial=-1)
                        + 1
                    ),
                )
                shape = (len(rows), valid_edges.shape[1], max_views)
                log_maps = np.full(
                    (*shape, len(offsets)), np.nan, dtype=np.float16
                )
                view_probabilities = np.zeros(shape, dtype=np.float32)
                dustbin_probabilities = np.zeros(shape, dtype=np.float32)
                spatial_valid = np.zeros(shape, dtype=bool)
                query_global_rows = selected_rows[rows]
                local_by_source = {
                    int(source_row): int(local_row)
                    for local_row, source_row in enumerate(query_global_rows.tolist())
                }
                compact_by_source = {
                    int(source_row): int(compact_row)
                    for compact_row, source_row in zip(
                        rows.tolist(), query_global_rows.tolist()
                    )
                }
                relevant = np.flatnonzero(
                    np.isin(source_rows, query_global_rows)
                )
                if len(relevant) == 0:
                    raise ValueError(
                        f"query {query_id} has no aligned spatial likelihood rows"
                    )
                for spatial_row in relevant.tolist():
                    source_row = int(source_rows[spatial_row])
                    local_row = local_by_source[source_row]
                    compact_row = compact_by_source[source_row]
                    if str(spatial_payload["query_ids"][spatial_row]) != str(query_id):
                        raise ValueError("spatial likelihood query identity differs")
                    candidate_rank = int(source_ranks[spatial_row])
                    if not 0 <= candidate_rank < candidate_evidence[
                        "candidate_compact_columns"
                    ].shape[1]:
                        raise ValueError("spatial candidate rank is outside evidence top-M")
                    column = int(
                        candidate_evidence["candidate_compact_columns"][
                            compact_row, candidate_rank
                        ]
                    )
                    view_rank = int(view_ranks[spatial_row])
                    if column < 0 or not 0 <= view_rank < max_views:
                        raise ValueError("spatial likelihood candidate slot is invalid")
                    expected_track = int(compact_tracks[compact_row, column])
                    expected_prototype = int(compact_prototypes[compact_row, column])
                    if (
                        int(spatial_payload["candidate_track_ids"][spatial_row])
                        != expected_track
                        or int(
                            spatial_payload["candidate_prototype_ids"][spatial_row]
                        )
                        != expected_prototype
                    ):
                        raise ValueError("spatial likelihood candidate identity differs")
                    if not np.allclose(
                        spatial_payload["center_xy"][spatial_row],
                        query_xy[compact_row],
                        rtol=0.0,
                        atol=1e-4,
                    ):
                        raise ValueError("spatial likelihood center differs from query xy")
                    if spatial_valid[local_row, column, view_rank]:
                        raise ValueError("duplicate spatial likelihood candidate/view slot")
                    log_maps[local_row, column, view_rank] = spatial_payload[
                        "local_log_probabilities"
                    ][spatial_row]
                    support_priors = np.asarray(
                        candidate_evidence[
                            "candidate_support_view_probabilities"
                        ][compact_row, candidate_rank],
                        dtype=np.float64,
                    )
                    expected_view_probability = (
                        0.0
                        if not 0 <= view_rank < len(support_priors)
                        else float(support_priors[view_rank])
                    )
                    if (
                        not np.isfinite(expected_view_probability)
                        or expected_view_probability < 0.0
                    ):
                        raise ValueError("candidate evidence support prior is invalid")
                    observed_view_probability = float(
                        spatial_payload["support_view_probabilities"][spatial_row]
                    )
                    if (
                        0 <= view_rank < len(support_priors)
                        and np.isfinite(observed_view_probability)
                        and not np.isclose(
                            observed_view_probability,
                            expected_view_probability,
                            rtol=0.0,
                            atol=2e-5,
                        )
                    ):
                        raise ValueError(
                            "spatial likelihood support prior differs from evidence"
                        )
                    view_probabilities[
                        local_row, column, view_rank
                    ] = expected_view_probability
                    dustbin_probabilities[local_row, column, view_rank] = float(
                        spatial_payload["dustbin_probabilities"][spatial_row]
                    )
                    spatial_valid[local_row, column, view_rank] = True
                if not np.any(spatial_valid):
                    raise ValueError(
                        f"query {query_id} produced no valid spatial likelihood slots"
                    )
                spatial_likelihood = CandidateSpatialLikelihood(
                    offsets_xy=offsets,
                    local_log_probabilities=log_maps,
                    view_probabilities=view_probabilities,
                    dustbin_probabilities=dustbin_probabilities,
                    valid_mask=spatial_valid,
                    log_evidence_weight=float(
                        args.candidate_spatial_log_evidence_weight
                    ),
                )
            output[str(query_id)] = PoseVerificationCandidatePool(
                token_indices=selected_rows[rows],
                xy=query_xy[rows],
                track_ids=compact_tracks[rows],
                prototype_ids=compact_prototypes[rows],
                xyz=xyz,
                descriptor_scores=scores[rows],
                valid_mask=local_valid,
                measurement_geometry_probabilities=(
                    None
                    if geometry_probabilities is None
                    else geometry_probabilities[rows]
                ),
                null_scores=local_null,
                spatial_likelihood=spatial_likelihood,
                geometry_prior_mix_weight=float(
                    args.candidate_geometry_prior_mix_weight
                ),
                spatial_geometry_calibration_weight=float(
                    args.candidate_spatial_geometry_calibration_weight
                ),
                geometry_generation_mix_weight=float(
                    args.candidate_geometry_generation_mix_weight
                ),
                candidate_update_probabilities=(
                    None if update_probabilities is None else update_probabilities[rows]
                ),
                candidate_refined_xy=(
                    None if update_refined_xy is None else update_refined_xy[rows]
                ),
                candidate_update_threshold=float(update_threshold),
            )
        return output

    def evaluate(
        scores: np.ndarray,
        split_name: str,
        backend: str,
        *,
        candidate_pool_scores: np.ndarray | None = None,
        candidate_pool_null_scores: np.ndarray | None = None,
        candidate_pool_spatial_payload: dict[str, np.ndarray] | None = None,
        candidate_pool_geometry_probabilities: np.ndarray | None = None,
        candidate_pool_update_probabilities: np.ndarray | None = None,
        candidate_pool_update_refined_xy: np.ndarray | None = None,
        candidate_pool_update_threshold: float = 0.5,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        grouped = matches_by_query(scores, split_name)
        grouped_candidate_pools = (
            {}
            if candidate_pool_scores is None
            else candidate_pools_by_query(
                candidate_pool_scores,
                split_name,
                null_scores=candidate_pool_null_scores,
                spatial_payload=candidate_pool_spatial_payload,
                geometry_probabilities=candidate_pool_geometry_probabilities,
                update_probabilities=candidate_pool_update_probabilities,
                update_refined_xy=candidate_pool_update_refined_xy,
                update_threshold=float(candidate_pool_update_threshold),
            )
        )
        expected_ids = [str(value) for value in split[split_name]]
        rows: list[dict[str, object]] = []
        for query_id in expected_ids:
            image = images_by_name.get(query_id)
            if image is None:
                rows.append(
                    {
                        "query_id": query_id,
                        "success": False,
                        "failure_reason": "missing_colmap_image",
                        "match_count": 0,
                        "inlier_count": 0,
                    }
                )
                continue
            camera = cameras[int(image.camera_id)]
            matches = grouped.get(query_id, [])
            verified_result = None
            if backend == "grouped_candidate_pool":
                pool = grouped_candidate_pools.get(query_id)
                if pool is None:
                    raise ValueError("grouped candidate PnP requires a candidate pool")
                if grouped_config is None or optional_grouped_config is None:
                    raise RuntimeError("grouped candidate PnP configuration is missing")
                optional_result = estimate_pose_from_grouped_candidate_pool(
                    pool,
                    camera,
                    config=optional_grouped_config,
                    query_seed=_query_seed(query_id),
                )
                generation_promotion_audit = None
                if bool(args.enable_grouped_generation_grid_fallback):
                    baseline_result = estimate_pose_from_grouped_candidate_pool(
                        replace(pool, geometry_generation_mix_weight=0.0),
                        camera,
                        config=grouped_config,
                        query_seed=_query_seed(query_id),
                    )
                    verified_result, generation_promotion_audit = (
                        select_geometry_guided_generation_with_immutable_baseline(
                            baseline_result,
                            optional_result,
                            min_strict_grid_cell_delta=int(
                                args.grouped_generation_min_strict_grid_delta
                            ),
                        )
                    )
                else:
                    verified_result = optional_result
                pose = verified_result.pose_w2c
                solver_success = bool(verified_result.success)
                match_count = int(verified_result.match_count)
                inlier_count = int(verified_result.inlier_count)
                backend_summary = verified_result.summary()
                if generation_promotion_audit is not None:
                    backend_summary["geometry_guided_generation_promotion"] = (
                        generation_promotion_audit
                    )
            elif backend == "verified":
                verified_result = estimate_pose_with_heldout_verification(
                    matches,
                    camera,
                    config=config,
                    query_seed=_query_seed(query_id),
                    candidate_pool=grouped_candidate_pools.get(query_id),
                )
                pose = verified_result.pose_w2c
                solver_success = bool(verified_result.success)
                match_count = int(verified_result.match_count)
                inlier_count = int(verified_result.inlier_count)
                backend_summary = verified_result.summary()
            elif backend == "single_ransac":
                selected_matches = select_pose_safe_matches(
                    matches,
                    max_matches=int(args.single_ransac_match_count),
                    image_width=int(camera.width),
                    image_height=int(camera.height),
                    mode=str(args.single_ransac_selection_mode),
                )
                selected_matches = stable_uniform_ransac_order(selected_matches)
                _set_cv2_seed(_query_seed(query_id))
                result = estimate_pose_pnp_ransac(
                    selected_matches,
                    camera,
                    reprojection_error_px=float(args.single_ransac_threshold_px),
                    iterations=int(args.single_ransac_iterations),
                    refine_method="LM",
                )
                pose = result.pose_w2c
                solver_success = bool(result.success)
                match_count = int(result.match_count)
                inlier_count = int(result.inlier_count)
                backend_summary = None
            else:
                raise ValueError(f"unsupported backend: {backend}")
            gt_pose = np.eye(4, dtype=np.float64)
            gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
            gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
            error = pnp_pose_error(pose, gt_pose)
            pre_refine_error = None
            oracle_error = None
            chosen_translation_rank = None
            valid_hypothesis_count = 0
            hypothesis_3cm_count = 0
            hypothesis_10cm_count = 0
            hypothesis_25cm_count = 0
            catastrophic_hypothesis_count = 0
            hypothesis_information_audit: list[dict[str, object]] = []
            if verified_result is not None:
                pre_refine_error = pnp_pose_error(
                    verified_result.pre_refine_pose_w2c, gt_pose
                )
                hypothesis_errors = [
                    pnp_pose_error(hypothesis_pose, gt_pose)
                    for hypothesis_pose in verified_result.hypothesis_poses_w2c
                    if hypothesis_pose is not None
                ]
                finite_hypothesis_errors = [
                    value
                    for value in hypothesis_errors
                    if np.isfinite(value.translation_m)
                    and np.isfinite(value.rotation_deg)
                ]
                if finite_hypothesis_errors:
                    valid_hypothesis_count = int(len(finite_hypothesis_errors))
                    hypothesis_3cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.03
                                and value.rotation_deg <= 5.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    hypothesis_10cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.10
                                and value.rotation_deg <= 5.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    hypothesis_25cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.25
                                and value.rotation_deg <= 2.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    catastrophic_hypothesis_count = int(
                        np.sum(
                            [
                                value.translation_m > 1.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    oracle_error = min(
                        finite_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m), float(value.rotation_deg)
                        ),
                    )
                    chosen_translation_rank = 1 + int(
                        np.sum(
                            [
                                float(value.translation_m)
                                < float(pre_refine_error.translation_m)
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                diagnostic_pool = grouped_candidate_pools.get(query_id)
                for hypothesis_index, (record, hypothesis_pose) in enumerate(
                    zip(
                        verified_result.hypotheses,
                        verified_result.hypothesis_poses_w2c,
                    )
                ):
                    if hypothesis_pose is None:
                        continue
                    if diagnostic_pool is None:
                        diagnostic_matches = matches
                    else:
                        diagnostic_matches, _selected, _residuals = (
                            resolve_pose_guided_candidate_pool(
                                diagnostic_pool,
                                hypothesis_pose,
                                camera,
                                residual_sigma_px=float(
                                    args.candidate_pool_residual_sigma_px
                                ),
                                hard_threshold_px=float(
                                    args.candidate_pool_hard_threshold_px
                                ),
                                descriptor_rank_weight=float(
                                    args.candidate_pool_descriptor_rank_weight
                                ),
                            )
                        )
                    hypothesis_error = pnp_pose_error(hypothesis_pose, gt_pose)
                    verification = record.verification
                    hypothesis_information_audit.append(
                        {
                            "hypothesis_index": int(hypothesis_index),
                            "chosen": bool(
                                hypothesis_index
                                == verified_result.chosen_hypothesis_index
                            ),
                            "fit_match_count_limit": int(
                                record.fit_match_count_limit
                            ),
                            "fit_match_count": int(record.fit_match_count),
                            "fit_inlier_count": int(record.fit_inlier_count),
                            "selection_mode": str(record.selection_mode),
                            "ransac_threshold_px": float(
                                record.ransac_threshold_px
                            ),
                            "rng_seed_offset": int(record.rng_seed_offset),
                            "resolved_match_count": int(len(diagnostic_matches)),
                            "fixed_posterior_log_likelihood_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_mean
                            ),
                            "fixed_posterior_spatial_calibrated_candidate_count": (
                                None
                                if verification is None
                                else int(
                                    verification.fixed_posterior_spatial_calibrated_candidate_count
                                )
                            ),
                            "fixed_posterior_spatial_geometry_calibration_weight": (
                                None
                                if verification is None
                                else float(
                                    verification.fixed_posterior_spatial_geometry_calibration_weight
                                )
                            ),
                            "strict_inlier_count": (
                                None
                                if verification is None
                                else int(verification.strict_inlier_count)
                            ),
                            "strict_grid_cell_count": (
                                None
                                if verification is None
                                else int(verification.strict_grid_cell_count)
                            ),
                            "soft_consensus": (
                                None
                                if verification is None
                                else float(verification.soft_consensus)
                            ),
                            "selected_candidate_fraction": (
                                None
                                if verification is None
                                else float(verification.selected_candidate_fraction)
                            ),
                            "selected_descriptor_score_mean": (
                                None
                                if verification is None
                                else verification.selected_descriptor_score_mean
                            ),
                            "selected_descriptor_margin_mean": (
                                None
                                if verification is None
                                else verification.selected_descriptor_margin_mean
                            ),
                            "selected_assignment_utility_mean": (
                                None
                                if verification is None
                                else verification.selected_assignment_utility_mean
                            ),
                            "selected_reprojection_mean_px": (
                                None
                                if verification is None
                                else verification.selected_reprojection_mean_px
                            ),
                            "selected_reprojection_p90_px": (
                                None
                                if verification is None
                                else verification.selected_reprojection_p90_px
                            ),
                            "measurement_probability_mean": (
                                None
                                if verification is None
                                else verification.measurement_probability_mean
                            ),
                            "measurement_high_confidence_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_fraction
                                )
                            ),
                            "measurement_strict_probability_mass_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_strict_probability_mass_fraction
                                )
                            ),
                            "measurement_loose_probability_mass_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_loose_probability_mass_fraction
                                )
                            ),
                            "measurement_soft_consensus_ratio": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_soft_consensus_ratio
                                )
                            ),
                            "measurement_high_confidence_strict_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_strict_fraction
                                )
                            ),
                            "measurement_high_confidence_loose_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_loose_fraction
                                )
                            ),
                            "measurement_high_confidence_contradiction_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_contradiction_fraction
                                )
                            ),
                            # These target errors are written only after inference
                            # and must never be consumed by a production selector.
                            "translation_m_TARGET_ONLY": (
                                None
                                if not np.isfinite(hypothesis_error.translation_m)
                                else float(hypothesis_error.translation_m)
                            ),
                            "rotation_deg_TARGET_ONLY": (
                                None
                                if not np.isfinite(hypothesis_error.rotation_deg)
                                else float(hypothesis_error.rotation_deg)
                            ),
                            **pose_information_diagnostics(
                                hypothesis_pose,
                                diagnostic_matches,
                                camera,
                                residual_sigma_px=float(
                                    args.candidate_pool_residual_sigma_px
                                ),
                            ),
                        }
                    )
            success = bool(
                solver_success
                and np.isfinite(error.translation_m)
                and np.isfinite(error.rotation_deg)
            )
            rows.append(
                {
                    "query_id": query_id,
                    "success": success,
                    "solver_success": solver_success,
                    "failure_reason": None if success else "pnp_solver_failure",
                    "match_count": match_count,
                    "inlier_count": inlier_count,
                    "translation_m": (
                        None if not np.isfinite(error.translation_m) else float(error.translation_m)
                    ),
                    "rotation_deg": (
                        None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg)
                    ),
                    "pre_refine_translation_m": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.translation_m)
                        else float(pre_refine_error.translation_m)
                    ),
                    "pre_refine_rotation_deg": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.rotation_deg)
                        else float(pre_refine_error.rotation_deg)
                    ),
                    "hypothesis_oracle_translation_m": (
                        None if oracle_error is None else float(oracle_error.translation_m)
                    ),
                    "hypothesis_oracle_rotation_deg": (
                        None if oracle_error is None else float(oracle_error.rotation_deg)
                    ),
                    "hypothesis_oracle_10cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.10
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "hypothesis_oracle_3cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.03
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "hypothesis_oracle_25cm_2deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.25
                            and oracle_error.rotation_deg <= 2.0
                        )
                    ),
                    "hypothesis_oracle_5cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.05
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "chosen_hypothesis_translation_rank": chosen_translation_rank,
                    "valid_hypothesis_count": int(valid_hypothesis_count),
                    "hypothesis_3cm_5deg_count": int(hypothesis_3cm_count),
                    "hypothesis_10cm_5deg_count": int(hypothesis_10cm_count),
                    "hypothesis_25cm_2deg_count": int(hypothesis_25cm_count),
                    "catastrophic_hypothesis_count": int(
                        catastrophic_hypothesis_count
                    ),
                    "hypothesis_information_audit_with_TARGET_ONLY_errors": (
                        hypothesis_information_audit
                    ),
                    "inference_verification": backend_summary,
                }
            )
        return _pose_summary(rows), rows

    validation_trials: list[dict[str, object]] = []
    validation_rows: dict[str, dict[str, object]] = {}
    for policy_key, scores in policy_scores.items():
        source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
        candidate_pool_scores = (
            None
            if bool(args.disable_topl_candidate_pool_verification)
            else source_scores
        )
        verified_pose, verified_rows = evaluate(
            scores,
            "validation",
            "verified",
            candidate_pool_scores=candidate_pool_scores,
        )
        single_pose, single_rows = evaluate(scores, "validation", "single_ransac")
        trial = {
            "trial_id": int(len(validation_trials)),
            "policy_key": policy_key,
            **policy_metadata[policy_key],
            "verified_pose": verified_pose,
            "matched_single_ransac_pose": single_pose,
            "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
            "passes_frozen_baseline_pose_gate": (
                None
                if frozen_baseline_source is None
                else _pose_gate(verified_pose, frozen_baseline_source["pose"])
            ),
        }
        validation_trials.append(trial)
        validation_rows[policy_key] = {
            "verified": verified_rows,
            "matched_single_ransac": single_rows,
        }
    if bool(args.enable_grouped_candidate_pnp):
        if grouped_null_scores is None:
            raise RuntimeError("grouped null scores were not loaded")
        for score_key in score_keys:
            scores = raw_scores[str(score_key)]
            grouped_pose, grouped_rows = evaluate(
                scores,
                "validation",
                "grouped_candidate_pool",
                candidate_pool_scores=scores,
                candidate_pool_null_scores=grouped_null_scores,
                candidate_pool_spatial_payload=spatial_payloads.get("validation"),
                candidate_pool_geometry_probabilities=(
                    geometry_probability_matrices.get("validation")
                ),
                candidate_pool_update_probabilities=(
                    update_probability_matrices.get("validation")
                ),
                candidate_pool_update_refined_xy=(
                    update_refined_xy_matrices.get("validation")
                ),
                candidate_pool_update_threshold=float(
                    update_prediction_metadata.get("validation", {}).get(
                        "update_threshold", 0.5
                    )
                ),
            )
            single_pose, single_rows = evaluate(
                scores, "validation", "single_ransac"
            )
            policy_key = f"grouped_candidate_pool__{str(score_key)}"
            trial = {
                "trial_id": int(len(validation_trials)),
                "policy_key": policy_key,
                "assignment_mode": "grouped_progressive_topl_with_explicit_null",
                "score_key": str(score_key),
                "verified_pose": grouped_pose,
                "matched_single_ransac_pose": single_pose,
                "passes_matched_pose_gate": _pose_gate(grouped_pose, single_pose),
                "passes_frozen_baseline_pose_gate": (
                    None
                    if frozen_baseline_source is None
                    else _pose_gate(grouped_pose, frozen_baseline_source["pose"])
                ),
            }
            validation_trials.append(trial)
            policy_metadata[policy_key] = {
                "assignment_mode": trial["assignment_mode"],
                "score_key": str(score_key),
            }
            validation_rows[policy_key] = {
                "verified": grouped_rows,
                "matched_single_ransac": single_rows,
            }
    if frozen_baseline_source is not None:
        baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
        baseline_trials = [
            trial
            for trial in validation_trials
            if str(trial["policy_key"]) == baseline_policy_key
        ]
        if len(baseline_trials) != 1:
            raise ValueError(
                "frozen baseline replay requires global_bipartite baseline scores"
            )
        _validate_frozen_baseline_pose(
            baseline_trials[0]["matched_single_ransac_pose"],
            frozen_baseline_source,
        )
    finite_trials = [
        trial
        for trial in validation_trials
        if trial["verified_pose"]["median_translation_m_success"] is not None
        and trial["verified_pose"]["p90_translation_m_success"] is not None
    ]
    if not finite_trials:
        raise RuntimeError("no verification policy produced a finite validation pose")
    gate_key = (
        "passes_matched_pose_gate"
        if frozen_baseline_source is None
        else "passes_frozen_baseline_pose_gate"
    )
    gated = [trial for trial in finite_trials if bool(trial[gate_key])]
    chosen = max(gated or finite_trials, key=_policy_key)
    chosen["selection_fallback_without_strict_pose_gate"] = not bool(gated)

    late_trials: list[dict[str, object]] = []
    late_rows: dict[str, dict[str, object]] = {}
    if bool(args.development_cross_block_audit):
        for trial in validation_trials:
            policy_key = str(trial["policy_key"])
            source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
            candidate_pool_scores = (
                None
                if bool(args.disable_topl_candidate_pool_verification)
                else source_scores
            )
            is_grouped = str(policy_metadata[policy_key]["assignment_mode"]).startswith(
                "grouped_progressive"
            )
            evaluation_scores = (
                source_scores if is_grouped else policy_scores[policy_key]
            )
            verified_pose, verified_rows = evaluate(
                evaluation_scores,
                "test",
                "grouped_candidate_pool" if is_grouped else "verified",
                candidate_pool_scores=candidate_pool_scores,
                candidate_pool_null_scores=(
                    grouped_null_scores if is_grouped else None
                ),
                candidate_pool_spatial_payload=(
                    spatial_payloads.get("test") if is_grouped else None
                ),
                candidate_pool_geometry_probabilities=(
                    geometry_probability_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_probabilities=(
                    update_probability_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_refined_xy=(
                    update_refined_xy_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_threshold=float(
                    update_prediction_metadata.get("test", {}).get(
                        "update_threshold", 0.5
                    )
                ),
            )
            single_pose, single_rows = evaluate(
                evaluation_scores, "test", "single_ransac"
            )
            late_trials.append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "policy_key": policy_key,
                    "verified_pose": verified_pose,
                    "matched_single_ransac_pose": single_pose,
                    "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
                    "passes_frozen_baseline_pose_gate": (
                        None
                        if frozen_baseline_source is None
                        or not isinstance(
                            frozen_baseline_source.get("late_development_pose"),
                            dict,
                        )
                        else _pose_gate(
                            verified_pose,
                            frozen_baseline_source["late_development_pose"],
                        )
                    ),
                    "selected_by_late_metrics": False,
                }
            )
            late_rows[policy_key] = {
                "verified": verified_rows,
                "matched_single_ransac": single_rows,
            }
        if frozen_baseline_source is not None:
            expected_late_pose = frozen_baseline_source.get("late_development_pose")
            if not isinstance(expected_late_pose, dict):
                raise ValueError(
                    "frozen baseline summary has no late development replay"
                )
            baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
            baseline_late_trials = [
                trial
                for trial in late_trials
                if str(trial["policy_key"]) == baseline_policy_key
            ]
            if len(baseline_late_trials) != 1:
                raise ValueError("late replay is missing the frozen baseline policy")
            _validate_frozen_baseline_pose(
                baseline_late_trials[0]["matched_single_ransac_pose"],
                {"pose": expected_late_pose},
            )

    train_trials: list[dict[str, object]] = []
    train_rows: dict[str, dict[str, object]] = {}
    if bool(args.export_train_grouped_hypotheses):
        if grouped_null_scores is None:
            raise RuntimeError("train grouped export requires grouped null scores")
        train_geometry = geometry_probability_matrices.get("train")
        if train_geometry is None:
            raise RuntimeError("train grouped export requires aligned OOF geometry")
        for score_key in score_keys:
            scores = raw_scores[str(score_key)]
            grouped_pose, grouped_rows = evaluate(
                scores,
                "train",
                "grouped_candidate_pool",
                candidate_pool_scores=scores,
                candidate_pool_null_scores=grouped_null_scores,
                candidate_pool_geometry_probabilities=train_geometry,
            )
            policy_key = f"grouped_candidate_pool__{str(score_key)}"
            train_trials.append(
                {
                    "policy_key": policy_key,
                    "score_key": str(score_key),
                    "verified_pose": grouped_pose,
                    "target_labels_exported_only_after_inference": True,
                }
            )
            train_rows[policy_key] = {"verified": grouped_rows}

    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "train_oof_geometry": train_rows,
                "validation": validation_rows,
                "late_development": late_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "proposals": str(proposals_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "score_artifact": str(score_path),
        "score_artifact_sha256": file_sha256_short(score_path),
        "projected_landmark_bank": str(bank_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "colmap_model_dir": str(model_dir),
        "colmap_cameras_bin": str(model_dir / "cameras.bin"),
        "colmap_cameras_bin_sha256": file_sha256_short(
            model_dir / "cameras.bin"
        ),
        "colmap_images_bin": str(model_dir / "images.bin"),
        "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        "split_json": str(split_path),
        "split_json_sha256": file_sha256_short(split_path),
        "frozen_baseline_summary": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_path"])
        ),
        "frozen_baseline_summary_sha256": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_sha256"])
        ),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
        "candidate_evidence": (
            None if candidate_evidence_path is None else str(candidate_evidence_path)
        ),
        "candidate_evidence_sha256": (
            None
            if candidate_evidence_path is None
            else file_sha256_short(candidate_evidence_path)
        ),
        "candidate_spatial_likelihood_validation": (
            None if spatial_paths["validation"] is None else str(spatial_paths["validation"])
        ),
        "candidate_spatial_likelihood_validation_sha256": (
            None
            if spatial_paths["validation"] is None
            else file_sha256_short(spatial_paths["validation"])
        ),
        "candidate_spatial_likelihood_test": (
            None if spatial_paths["test"] is None else str(spatial_paths["test"])
        ),
        "candidate_spatial_likelihood_test_sha256": (
            None
            if spatial_paths["test"] is None
            else file_sha256_short(spatial_paths["test"])
        ),
        "candidate_geometry_probabilities": geometry_probability_metadata,
        "candidate_coordinate_updates": update_prediction_metadata,
    }
    summary = {
        "stage": "heldout_multi_hypothesis_pose_verification",
        "inputs": manifest,
        "protocol": {
            "proposal_scope": "full_bank_global_faiss_top_l",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": bool(spatial_requested),
            "candidate_specific_rgb_spatial_likelihood": bool(spatial_requested),
            "candidate_spatial_log_evidence_weight": (
                None
                if not spatial_requested
                else float(args.candidate_spatial_log_evidence_weight)
            ),
            "candidate_geometry_prior_mix_weight": float(
                args.candidate_geometry_prior_mix_weight
            ),
            "candidate_geometry_generation_mix_weight": float(
                args.candidate_geometry_generation_mix_weight
            ),
            "candidate_coordinate_update_refine": bool(
                args.enable_optional_candidate_coordinate_refine
            ),
            "candidate_coordinate_update_policy": (
                "selected_identity_only_then_independent_final_audit"
                if bool(args.enable_optional_candidate_coordinate_refine)
                else None
            ),
            "candidate_geometry_generation_policy": (
                "pose_free_true_residual_probability_within_retained_candidate_mass"
                if float(args.candidate_geometry_generation_mix_weight) > 0.0
                else "original_fixed_candidate_posterior"
            ),
            "immutable_unmixed_generation_baseline": bool(
                args.enable_grouped_generation_grid_fallback
            ),
            "geometry_guided_generation_promotion_rule": (
                "optional_strict_grid_cells_ge_baseline_plus_min_delta"
                if bool(args.enable_grouped_generation_grid_fallback)
                else None
            ),
            "grouped_generation_min_strict_grid_delta": int(
                args.grouped_generation_min_strict_grid_delta
            ),
            "immutable_baseline_final_refine_acceptance_policy": (
                None
                if grouped_config is None
                else str(grouped_config.final_refine_acceptance_policy)
            ),
            "optional_final_refine_acceptance_policy": (
                None
                if optional_grouped_config is None
                else str(optional_grouped_config.final_refine_acceptance_policy)
            ),
            "candidate_spatial_geometry_calibration_weight": float(
                args.candidate_spatial_geometry_calibration_weight
            ),
            "candidate_spatial_reliability_policy": (
                "true_residual_calibrated_geometry_probability_interpolated_with_raw_dustbin"
                if float(args.candidate_spatial_geometry_calibration_weight) > 0.0
                else "raw_measurement_dustbin"
            ),
            "candidate_geometry_prior_mass_policy": (
                "preserve_explicit_null_and_retained_candidate_mass"
                if geometry_probabilities_requested
                else None
            ),
            "ground_truth_available_to_pose_selector": False,
            "pre_pnp_depth_proxy": False,
            "pre_pnp_geometry": "image_bearing_coverage_plus_world_xyz_covariance",
            "post_pose_geometry": (
                "heldout_topl_pose_guided_bipartite_assignment_plus_cheirality_plus_camera_depth"
                if not bool(args.disable_topl_candidate_pool_verification)
                else "heldout_hard_assignment_reprojection_plus_cheirality_plus_camera_depth"
            ),
            "topl_candidate_pool_verification": not bool(
                args.disable_topl_candidate_pool_verification
            ),
            "policy_selected_on_validation_only": True,
            "validation_gate_reference": (
                "matched_single_ransac"
                if frozen_baseline_source is None
                else "externally_frozen_global_baseline_exact_replay"
            ),
            "late_block_is_untouched_test": False,
            "production_promoted": False,
        },
        "config": {
            "verified_pnp": {
                **config.__dict__,
                "fit_match_counts": list(config.fit_match_counts),
                "selection_modes": list(config.selection_modes),
                "ransac_thresholds_px": list(config.ransac_thresholds_px),
                "rng_seed_offsets": list(config.rng_seed_offsets),
            },
            "grouped_candidate_pnp": {
                "enabled": bool(args.enable_grouped_candidate_pnp),
                "null_score_key": str(args.grouped_null_score_key),
                "candidate_spatial_log_evidence_weight": float(
                    args.candidate_spatial_log_evidence_weight
                ),
                "candidate_geometry_prior_mix_weight": float(
                    args.candidate_geometry_prior_mix_weight
                ),
                "candidate_geometry_generation_mix_weight": float(
                    args.candidate_geometry_generation_mix_weight
                ),
                "immutable_unmixed_generation_baseline": bool(
                    args.enable_grouped_generation_grid_fallback
                ),
                "generation_promotion_rule": (
                    "optional_strict_grid_cells_ge_baseline_plus_min_delta"
                    if bool(args.enable_grouped_generation_grid_fallback)
                    else None
                ),
                "generation_min_strict_grid_cell_delta": int(
                    args.grouped_generation_min_strict_grid_delta
                ),
                "immutable_baseline_final_refine_acceptance_policy": (
                    None
                    if grouped_config is None
                    else str(grouped_config.final_refine_acceptance_policy)
                ),
                "optional_final_refine_acceptance_policy": (
                    None
                    if optional_grouped_config is None
                    else str(optional_grouped_config.final_refine_acceptance_policy)
                ),
                "optional_candidate_coordinate_refine": bool(
                    args.enable_optional_candidate_coordinate_refine
                ),
                "candidate_spatial_geometry_calibration_weight": float(
                    args.candidate_spatial_geometry_calibration_weight
                ),
                **(
                    {}
                    if grouped_config is None
                    else {
                        **grouped_config.__dict__,
                        "candidate_limits": list(grouped_config.candidate_limits),
                        "sampling_temperatures": list(
                            grouped_config.sampling_temperatures
                        ),
                        "fit_match_counts": list(grouped_config.fit_match_counts),
                        "ransac_thresholds_px": list(
                            grouped_config.ransac_thresholds_px
                        ),
                    }
                ),
            },
            "single_ransac": {
                "match_count": int(args.single_ransac_match_count),
                "selection_mode": str(args.single_ransac_selection_mode),
                "threshold_px": float(args.single_ransac_threshold_px),
                "iterations": int(args.single_ransac_iterations),
            },
        },
        "validation": {
            "trial_count": int(len(validation_trials)),
            "chosen": chosen,
            "trials": validation_trials,
        },
        "late_development_replay": {
            "enabled": bool(args.development_cross_block_audit),
            "trials": late_trials,
            "development_only": True,
            "used_for_policy_selection": False,
        },
        "train_hypothesis_export": {
            "enabled": bool(args.export_train_grouped_hypotheses),
            "geometry_probabilities_are_query_grouped_oof": bool(
                train_geometry_probabilities_requested
            ),
            "target_labels_written_only_after_pose_inference": True,
            "trials": train_trials,
        },
        "outputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
