"""Fit train-query-only calibration for RGB candidate spatial likelihoods."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.spatial_likelihood_calibration import (
    CandidateSpatialLikelihoodCalibration,
    binary_calibration_metrics,
    fit_dustbin_platt_scaling,
    fit_spatial_temperature,
    spatial_target_nll,
)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise argparse.ArgumentTypeError("expected at least one comma-separated path")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial_artifacts", type=_paths, required=True)
    parser.add_argument("--diagnostic_rows_csvs", type=_paths, required=True)
    parser.add_argument("--base_model_training_rows_csvs", type=_paths, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--audit_fold", type=int, default=4)
    parser.add_argument("--fold_salt", default="candidate_spatial_calibration_v1")
    parser.add_argument("--allow_base_model_query_overlap", action="store_true")
    parser.add_argument(
        "--allow_legacy_identity_dustbin",
        action="store_true",
        help="diagnostic only: calibrate legacy v3 identity-head dustbin artifacts",
    )
    return parser.parse_args(argv)


def _query_manifest_sha256(query_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(str(value) for value in query_ids))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _query_fold(query_id: str, *, folds: int, salt: str) -> int:
    payload = f"{salt}\0{query_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % int(folds)


def _read_query_ids(paths: Sequence[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "query_id" not in reader.fieldnames:
                raise ValueError(f"query_id is missing from {path}")
            output.update(str(row["query_id"]) for row in reader)
    return output


def _load_training_payload(
    spatial_paths: Sequence[Path],
    diagnostic_paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    if len(spatial_paths) != len(diagnostic_paths):
        raise ValueError("spatial artifacts and diagnostic CSVs must be paired")
    arrays: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "local_log_probabilities": [],
        "dustbin_probabilities": [],
        "target_is_dustbin": [],
        "target_offset_xy": [],
        "dustbin_supervision_weight": [],
        "spatial_calibration_supervision_weight": [],
    }
    shared_metadata: dict[str, object] | None = None
    sources: list[dict[str, object]] = []
    compatibility_keys = (
        "format",
        "measurement_checkpoint_sha256",
        "query_source",
        "support_patch_warp",
        "search_radius_px",
        "step_px",
        "context_radius_px",
        "spatial_probability_semantics",
        "dustbin_probability_semantics",
        "target_dustbin_semantics",
        "measurement_success_threshold_px",
    )
    offsets: np.ndarray | None = None
    for spatial_path, diagnostic_path in zip(spatial_paths, diagnostic_paths):
        with np.load(spatial_path, allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        metadata = json.loads(str(payload["metadata_json"].item()))
        artifact_format = str(metadata.get("format", ""))
        if artifact_format not in {
            "candidate_spatial_likelihood_v3",
            "candidate_spatial_likelihood_v4",
            "candidate_spatial_likelihood_v5",
        }:
            raise ValueError("unsupported spatial likelihood training artifact")
        if str(metadata.get("query_source")) != "real_pair" or bool(
            metadata.get("pose_or_ground_truth_used_for_inference")
        ):
            raise ValueError("calibration requires pose-free real-image inference inputs")
        current_shared = {key: metadata.get(key) for key in compatibility_keys}
        if shared_metadata is None:
            shared_metadata = current_shared
        elif current_shared != shared_metadata:
            raise ValueError("spatial calibration shards have incompatible metadata")
        current_offsets = np.asarray(payload["offsets_xy"], dtype=np.float64)
        if offsets is None:
            offsets = current_offsets.copy()
        elif not np.array_equal(current_offsets, offsets):
            raise ValueError("spatial calibration shards use different offset grids")

        with diagnostic_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        count = len(payload["query_ids"])
        if len(rows) != count:
            raise ValueError("spatial artifact and diagnostic CSV row counts differ")
        query_ids = np.asarray([str(row["query_id"]) for row in rows])
        csv_target_is_dustbin = np.asarray(
            [str(row["target_is_dustbin"]).strip().lower() == "true" for row in rows],
            dtype=bool,
        )
        csv_supervision_weight = np.asarray(
            [float(row["dustbin_supervision_weight"]) for row in rows],
            dtype=np.float64,
        )
        if not np.array_equal(query_ids, np.asarray(payload["query_ids"])):
            raise ValueError("spatial artifact and diagnostic query identities differ")
        if artifact_format in {
            "candidate_spatial_likelihood_v4",
            "candidate_spatial_likelihood_v5",
        }:
            required_target_arrays = {
                "target_is_dustbin",
                "measurement_validity_supervision_weight",
                "target_offset_xy",
                "spatial_calibration_supervision_weight",
            }
            missing_target_arrays = required_target_arrays - set(payload)
            if missing_target_arrays:
                raise ValueError(
                    "production spatial artifact lacks target-only GT-pose arrays: "
                    f"{sorted(missing_target_arrays)}"
                )
            target_is_dustbin = np.asarray(
                payload["target_is_dustbin"], dtype=bool
            )
            supervision_weight = np.asarray(
                payload["measurement_validity_supervision_weight"],
                dtype=np.float64,
            )
            spatial_supervision_weight = np.asarray(
                payload["spatial_calibration_supervision_weight"],
                dtype=np.float64,
            )
        else:
            target_is_dustbin = csv_target_is_dustbin
            supervision_weight = csv_supervision_weight
            spatial_supervision_weight = supervision_weight.copy()
            if not np.array_equal(target_is_dustbin, payload["target_is_dustbin"]):
                raise ValueError(
                    "spatial artifact and diagnostic dustbin targets differ"
                )
        if np.any(~np.isfinite(supervision_weight)) or np.any(supervision_weight < 0.0):
            raise ValueError("dustbin supervision weights must be finite and non-negative")
        if np.any(~np.isfinite(spatial_supervision_weight)) or np.any(
            spatial_supervision_weight < 0.0
        ):
            raise ValueError(
                "spatial calibration weights must be finite and non-negative"
            )
        arrays["query_ids"].append(query_ids)
        arrays["local_log_probabilities"].append(
            np.asarray(payload["local_log_probabilities"], dtype=np.float32)
        )
        arrays["dustbin_probabilities"].append(
            np.asarray(payload["dustbin_probabilities"], dtype=np.float64)
        )
        arrays["target_is_dustbin"].append(target_is_dustbin)
        arrays["target_offset_xy"].append(
            np.asarray(payload["target_offset_xy"], dtype=np.float64)
        )
        arrays["dustbin_supervision_weight"].append(supervision_weight)
        arrays["spatial_calibration_supervision_weight"].append(
            spatial_supervision_weight
        )
        sources.append(
            {
                "spatial_artifact": str(spatial_path),
                "spatial_artifact_sha256": file_sha256_short(spatial_path),
                "diagnostic_rows_csv": str(diagnostic_path),
                "diagnostic_rows_sha256": file_sha256_short(diagnostic_path),
                "row_count": int(count),
                "query_count": int(len(set(query_ids.tolist()))),
                "format": artifact_format,
            }
        )
    if shared_metadata is None or offsets is None:
        raise ValueError("spatial calibration inputs are empty")
    output = {
        key: np.concatenate(values, axis=0) for key, values in arrays.items()
    }
    output["offsets_xy"] = offsets
    return output, shared_metadata, sources


def _calibrated_dustbin(
    probabilities: np.ndarray,
    *,
    scale: float,
    bias: float,
) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(values) - np.log1p(-values)
    calibrated_logits = float(scale) * logits + float(bias)
    output = np.empty_like(calibrated_logits)
    positive = calibrated_logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-calibrated_logits[positive]))
    exponential = np.exp(calibrated_logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _split_metrics(
    payload: dict[str, np.ndarray],
    selected: np.ndarray,
    *,
    spatial_temperature: float,
    dustbin_scale: float,
    dustbin_bias: float,
) -> dict[str, object]:
    labels = payload["target_is_dustbin"][selected]
    raw_dustbin = payload["dustbin_probabilities"][selected]
    calibrated_dustbin = _calibrated_dustbin(
        raw_dustbin, scale=dustbin_scale, bias=dustbin_bias
    )
    non_dustbin_rows = (
        selected
        & ~payload["target_is_dustbin"]
        & (payload["spatial_calibration_supervision_weight"] > 0.0)
    )
    raw_spatial_nll = spatial_target_nll(
        payload["local_log_probabilities"][non_dustbin_rows],
        payload["offsets_xy"],
        payload["target_offset_xy"][non_dustbin_rows],
        temperature=1.0,
    )
    calibrated_spatial_nll = spatial_target_nll(
        payload["local_log_probabilities"][non_dustbin_rows],
        payload["offsets_xy"],
        payload["target_offset_xy"][non_dustbin_rows],
        temperature=float(spatial_temperature),
    )
    raw_binary = binary_calibration_metrics(raw_dustbin, labels)
    calibrated_binary = binary_calibration_metrics(calibrated_dustbin, labels)
    non_dustbin_fraction = float(np.mean(~labels))
    return {
        "query_count": int(len(set(payload["query_ids"][selected].tolist()))),
        "reliable_row_count": int(np.count_nonzero(selected)),
        "non_dustbin_row_count": int(np.count_nonzero(non_dustbin_rows)),
        "raw_dustbin": raw_binary,
        "calibrated_dustbin": calibrated_binary,
        "raw_spatial_nll": float(raw_spatial_nll),
        "calibrated_spatial_nll": float(calibrated_spatial_nll),
        "raw_joint_nll": float(raw_binary["nll"] + non_dustbin_fraction * raw_spatial_nll),
        "calibrated_joint_nll": float(
            calibrated_binary["nll"]
            + non_dustbin_fraction * calibrated_spatial_nll
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.folds) < 2 or not 0 <= int(args.audit_fold) < int(args.folds):
        raise ValueError("calibration fold configuration is invalid")
    payload, metadata, sources = _load_training_payload(
        args.spatial_artifacts, args.diagnostic_rows_csvs
    )
    if (
        metadata.get("format")
        not in {
            "candidate_spatial_likelihood_v4",
            "candidate_spatial_likelihood_v5",
        }
        and not bool(args.allow_legacy_identity_dustbin)
    ):
        raise ValueError(
            "production calibration requires v4 or v5 target-only measurement evidence"
        )
    query_ids = np.asarray(payload["query_ids"])
    fold_ids = np.asarray(
        [
            _query_fold(str(query_id), folds=int(args.folds), salt=str(args.fold_salt))
            for query_id in query_ids
        ],
        dtype=np.int64,
    )
    reliable = np.asarray(payload["dustbin_supervision_weight"] > 0.0, dtype=bool)
    fit_mask = reliable & (fold_ids != int(args.audit_fold))
    audit_mask = reliable & (fold_ids == int(args.audit_fold))
    if not np.any(fit_mask) or not np.any(audit_mask):
        raise ValueError("query-grouped calibration split produced an empty fold")
    fit_non_dustbin = (
        fit_mask
        & ~payload["target_is_dustbin"]
        & (payload["spatial_calibration_supervision_weight"] > 0.0)
    )
    if not np.any(fit_non_dustbin) or not np.any(
        audit_mask
        & ~payload["target_is_dustbin"]
        & (payload["spatial_calibration_supervision_weight"] > 0.0)
    ):
        raise ValueError("calibration split needs non-dustbin rows in fit and audit")

    dustbin_fit = fit_dustbin_platt_scaling(
        payload["dustbin_probabilities"][fit_mask],
        payload["target_is_dustbin"][fit_mask],
    )
    spatial_fit = fit_spatial_temperature(
        payload["local_log_probabilities"][fit_non_dustbin],
        payload["offsets_xy"],
        payload["target_offset_xy"][fit_non_dustbin],
    )
    fit_queries = sorted(set(query_ids[fit_mask].tolist()))
    audit_queries = sorted(set(query_ids[audit_mask].tolist()))
    if set(fit_queries) & set(audit_queries):
        raise RuntimeError("fit and audit calibration queries overlap")
    base_model_queries = _read_query_ids(args.base_model_training_rows_csvs)
    calibration_queries = set(fit_queries) | set(audit_queries)
    overlapping_queries = sorted(calibration_queries & base_model_queries)
    data_contract_eligible = len(overlapping_queries) == 0
    if not data_contract_eligible and not bool(args.allow_base_model_query_overlap):
        raise ValueError(
            "calibration queries overlap base-model training queries; use a disjoint "
            "measurement checkpoint or explicitly allow a diagnostic artifact"
        )

    model = CandidateSpatialLikelihoodCalibration(
        spatial_temperature=float(spatial_fit["temperature"]),
        dustbin_logit_scale=float(dustbin_fit["logit_scale"]),
        dustbin_logit_bias=float(dustbin_fit["logit_bias"]),
        measurement_checkpoint_sha256=str(metadata["measurement_checkpoint_sha256"]),
        search_radius_px=float(metadata["search_radius_px"]),
        step_px=float(metadata["step_px"]),
        context_radius_px=float(metadata["context_radius_px"]),
        query_source=str(metadata["query_source"]),
        support_patch_warp=str(metadata["support_patch_warp"]),
        fit_query_manifest_sha256=_query_manifest_sha256(fit_queries),
        audit_query_manifest_sha256=_query_manifest_sha256(audit_queries),
        production_eligible=bool(data_contract_eligible),
        base_model_query_overlap_count=int(len(overlapping_queries)),
        dustbin_probability_semantics=str(
            metadata["dustbin_probability_semantics"]
        ),
    )
    metrics = {
        "fit": _split_metrics(
            payload,
            fit_mask,
            spatial_temperature=model.spatial_temperature,
            dustbin_scale=model.dustbin_logit_scale,
            dustbin_bias=model.dustbin_logit_bias,
        ),
        "audit": _split_metrics(
            payload,
            audit_mask,
            spatial_temperature=model.spatial_temperature,
            dustbin_scale=model.dustbin_logit_scale,
            dustbin_bias=model.dustbin_logit_bias,
        ),
    }
    audit_joint_nll_delta = float(
        metrics["audit"]["calibrated_joint_nll"]
        - metrics["audit"]["raw_joint_nll"]
    )
    calibration_promoted = bool(audit_joint_nll_delta <= 0.0)
    production_eligible = bool(
        data_contract_eligible and calibration_promoted
    )
    if bool(model.production_eligible) != production_eligible:
        model = replace(
            model, production_eligible=production_eligible
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "spatial_likelihood_calibration.json"
    model_path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "candidate_spatial_likelihood_calibration_fit",
        "inputs": {
            "sources": sources,
            "base_model_training_rows": [
                {
                    "path": str(path),
                    "sha256": file_sha256_short(path),
                }
                for path in args.base_model_training_rows_csvs
            ],
        },
        "protocol": {
            "fit_audit_split": "deterministic_query_grouped_hash",
            "folds": int(args.folds),
            "audit_fold": int(args.audit_fold),
            "fold_salt": str(args.fold_salt),
            "fit_audit_query_overlap_count": 0,
            "base_model_query_overlap_count": int(len(overlapping_queries)),
            "base_model_query_overlap_examples": overlapping_queries[:20],
            "production_eligible": bool(production_eligible),
            "data_contract_eligible": bool(data_contract_eligible),
            "calibration_promoted": bool(calibration_promoted),
            "audit_calibrated_minus_raw_joint_nll": float(
                audit_joint_nll_delta
            ),
            "identity_prior_modified": False,
            "null_identity_mass_modified": False,
            "target_use": "calibration_fit_and_audit_only_never_pose_inference",
            "dustbin_probability_semantics": str(
                metadata["dustbin_probability_semantics"]
            ),
            "legacy_identity_dustbin": bool(
                metadata.get("format")
                not in {
                    "candidate_spatial_likelihood_v4",
                    "candidate_spatial_likelihood_v5",
                }
            ),
        },
        "model": model.to_dict(),
        "fit_diagnostics": {
            "dustbin": dustbin_fit,
            "spatial": spatial_fit,
        },
        "metrics": metrics,
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
