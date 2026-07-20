"""Build a provenance-locked RGB identity posterior overlay for pose scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    fuse_candidate_log_likelihood_ratios,
)


OVERLAY_FORMAT = "independent_rgb_candidate_prior_overlay_v1"
PREDICTION_FORMAT = "independent_rgb_candidate_predictions_v1"
PROBABILITY_SEMANTICS = (
    "candidate_identity_probability_plus_explicit_null_equals_one"
)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        if "metadata_json" not in payload.files:
            raise ValueError(f"artifact lacks metadata_json: {path}")
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"artifact metadata must be an object: {path}")
    return arrays, metadata


def _require_arrays(
    arrays: Mapping[str, np.ndarray], required: Sequence[str], *, artifact: Path
) -> None:
    missing = sorted(set(required).difference(arrays))
    if missing:
        raise ValueError(f"{artifact} lacks required arrays: {missing}")


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _validate_base_alignment(
    *,
    base: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> None:
    base_tracks = np.asarray(base["candidate_track_ids"], dtype=np.int64)
    base_probability = np.asarray(base["candidate_probabilities"], dtype=np.float32)
    base_null = np.asarray(base["null_probabilities"], dtype=np.float32).reshape(-1)
    rows = np.asarray(evidence["selected_rows"], dtype=np.int64).reshape(-1)
    source_columns = np.asarray(
        evidence["candidate_source_columns"], dtype=np.int64
    )
    evidence_valid = np.asarray(evidence["candidate_valid"], dtype=bool)
    evidence_tracks = np.asarray(evidence["candidate_track_ids"], dtype=np.int64)
    evidence_prior = np.asarray(
        evidence["candidate_prior_probabilities"], dtype=np.float32
    )
    evidence_unknown = np.asarray(evidence["unknown_probability"], dtype=np.float32)

    if (
        base_tracks.ndim != 2
        or base_probability.shape != base_tracks.shape
        or base_null.shape != (base_tracks.shape[0],)
    ):
        raise ValueError("base overlay candidate arrays have incompatible shapes")
    if (
        source_columns.ndim != 2
        or evidence_valid.shape != source_columns.shape
        or evidence_tracks.shape != source_columns.shape
        or evidence_prior.shape != source_columns.shape
        or evidence_unknown.shape != (len(rows),)
    ):
        raise ValueError("candidate evidence arrays have incompatible shapes")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(
        rows >= base_tracks.shape[0]
    ):
        raise ValueError("candidate evidence selected rows are invalid or repeated")

    expected_valid = source_columns >= 0
    if not np.array_equal(evidence_valid, expected_valid):
        raise ValueError("candidate evidence validity differs from source columns")
    if np.any(source_columns[evidence_valid] >= base_tracks.shape[1]):
        raise ValueError("candidate evidence source column is outside base overlay")

    safe_columns = np.maximum(source_columns, 0)
    base_rows = np.broadcast_to(rows[:, None], source_columns.shape)
    mapped_tracks = base_tracks[base_rows, safe_columns].copy()
    mapped_probability = base_probability[base_rows, safe_columns].copy()
    mapped_tracks[~evidence_valid] = -1
    mapped_probability[~evidence_valid] = 0.0
    if not np.array_equal(mapped_tracks, evidence_tracks):
        raise ValueError("candidate evidence tracks do not map to base overlay columns")
    if not np.allclose(mapped_probability, evidence_prior, rtol=0.0, atol=2e-5):
        raise ValueError("candidate evidence priors differ from base overlay")
    if not np.allclose(base_null[rows], evidence_unknown, rtol=0.0, atol=2e-5):
        raise ValueError("candidate evidence null mass differs from base overlay")

    base_valid = base_tracks[rows] >= 0
    covered = np.zeros_like(base_valid, dtype=bool)
    compact_rows = np.broadcast_to(
        np.arange(len(rows), dtype=np.int64)[:, None], source_columns.shape
    )
    covered[compact_rows[evidence_valid], source_columns[evidence_valid]] = True
    if not np.array_equal(covered, base_valid):
        raise ValueError(
            "candidate evidence must cover every valid base candidate before RGB "
            "identity likelihood can be fused"
        )


def _load_aligned_predictions(
    *,
    paths: Sequence[Path],
    weights: Sequence[float],
    llr_aggregation: str,
    evidence_row_count: int,
    candidate_count: int,
    expected_candidate_hash: str,
    expected_availability_hash: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    if not paths:
        raise ValueError("at least one RGB prediction artifact is required")
    if len(paths) != len(weights):
        raise ValueError("prediction artifact count and LLR weight count differ")
    aggregation = str(llr_aggregation)
    if aggregation not in {"sum", "mean"}:
        raise ValueError("llr_aggregation must be sum or mean")
    for weight in weights:
        if not math.isfinite(float(weight)):
            raise ValueError("RGB LLR weights must be finite")

    expected_positions: np.ndarray | None = None
    expected_measured: np.ndarray | None = None
    combined_llr: np.ndarray | None = None
    manifests: list[dict[str, object]] = []
    for path, weight in zip(paths, weights):
        arrays, metadata = _load_npz(Path(path))
        _require_arrays(
            arrays,
            (
                "evidence_row_indices",
                "candidate_rgb_log_likelihood_ratios",
                "candidate_rgb_measured",
            ),
            artifact=Path(path),
        )
        if metadata.get("format") != PREDICTION_FORMAT:
            raise ValueError(f"unsupported RGB prediction format: {path}")
        if str(metadata.get("candidate_evidence_sha256")) != expected_candidate_hash:
            raise ValueError(f"candidate evidence hash mismatch: {path}")
        if str(metadata.get("availability_evidence_sha256")) != expected_availability_hash:
            raise ValueError(f"availability evidence hash mismatch: {path}")
        positions = np.asarray(
            arrays["evidence_row_indices"], dtype=np.int64
        ).reshape(-1)
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"RGB prediction evidence rows repeat: {path}")
        if np.any(positions < 0) or np.any(positions >= int(evidence_row_count)):
            raise ValueError(f"RGB prediction evidence row is outside the contract: {path}")
        llr = np.asarray(
            arrays["candidate_rgb_log_likelihood_ratios"], dtype=np.float32
        )
        measured = np.asarray(arrays["candidate_rgb_measured"], dtype=bool)
        if (
            llr.shape != (len(positions), int(candidate_count))
            or measured.shape != llr.shape
        ):
            raise ValueError(f"RGB prediction candidate arrays are misaligned: {path}")
        if np.any(~np.isfinite(llr)):
            raise ValueError(f"RGB prediction LLR contains non-finite values: {path}")
        order = np.argsort(positions, kind="stable")
        positions = positions[order]
        llr = llr[order]
        measured = measured[order]
        if expected_positions is None:
            expected_positions = positions
            expected_measured = measured
            combined_llr = np.zeros_like(llr, dtype=np.float32)
        elif not np.array_equal(positions, expected_positions):
            raise ValueError("RGB prediction artifacts cover different candidate rows")
        elif not np.array_equal(measured, expected_measured):
            raise ValueError("RGB prediction artifacts disagree on measurement masks")
        assert combined_llr is not None
        combined_llr += float(weight) * llr
        manifests.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(Path(path)),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "prediction_splits": metadata.get("prediction_splits"),
                "llr_weight": float(weight),
            }
        )
    assert (
        expected_positions is not None
        and expected_measured is not None
        and combined_llr is not None
    )
    if aggregation == "mean":
        combined_llr /= float(len(paths))
    return combined_llr, expected_measured, manifests, expected_positions


def build_independent_rgb_candidate_prior_overlay(
    *,
    base_overlay_path: Path,
    candidate_evidence_path: Path,
    availability_evidence_path: Path,
    prediction_paths: Sequence[Path],
    llr_weights: Sequence[float],
    output_path: Path,
    llr_aggregation: str = "sum",
) -> dict[str, object]:
    """Fuse target-free RGB identity LLRs into a full proposal-row overlay."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing overlay: {output}")
    base, base_metadata = _load_npz(Path(base_overlay_path))
    evidence, evidence_metadata = _load_npz(Path(candidate_evidence_path))
    _require_arrays(
        base,
        ("candidate_track_ids", "candidate_probabilities", "null_probabilities"),
        artifact=Path(base_overlay_path),
    )
    _require_arrays(
        evidence,
        (
            "selected_rows",
            "split_names",
            "candidate_source_columns",
            "candidate_valid",
            "candidate_track_ids",
            "candidate_prior_probabilities",
            "unknown_probability",
        ),
        artifact=Path(candidate_evidence_path),
    )
    if evidence_metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("RGB identity overlay requires candidate evidence V3")
    proposals_hash = str(evidence_metadata.get("proposals_sha256", ""))
    if not proposals_hash or str(base_metadata.get("proposals_sha256")) != proposals_hash:
        raise ValueError("base overlay and candidate evidence use different proposals")
    _validate_base_alignment(base=base, evidence=evidence)

    expected_candidate_hash = file_sha256_short(Path(candidate_evidence_path))
    expected_availability_hash = file_sha256_short(Path(availability_evidence_path))
    rows = np.asarray(evidence["selected_rows"], dtype=np.int64).reshape(-1)
    source_columns = np.asarray(
        evidence["candidate_source_columns"], dtype=np.int64
    )
    valid = np.asarray(evidence["candidate_valid"], dtype=bool)
    prior = np.asarray(evidence["candidate_prior_probabilities"], dtype=np.float32)
    unknown = np.asarray(evidence["unknown_probability"], dtype=np.float32)
    combined_llr, measured, prediction_manifests, positions = _load_aligned_predictions(
        paths=tuple(Path(path) for path in prediction_paths),
        weights=tuple(float(weight) for weight in llr_weights),
        llr_aggregation=str(llr_aggregation),
        evidence_row_count=len(rows),
        candidate_count=prior.shape[1],
        expected_candidate_hash=expected_candidate_hash,
        expected_availability_hash=expected_availability_hash,
    )
    fused, fused_null = fuse_candidate_log_likelihood_ratios(
        torch.from_numpy(prior[positions]),
        torch.from_numpy(unknown[positions]),
        torch.from_numpy(combined_llr),
        measured_mask=torch.from_numpy(measured),
        candidate_valid=torch.from_numpy(valid[positions]),
    )
    fused_probability = fused.numpy().astype(np.float32)
    if not np.allclose(fused_null.numpy(), unknown[positions], rtol=0.0, atol=2e-6):
        raise RuntimeError("RGB fusion changed explicit null mass")

    output_probability = np.array(base["candidate_probabilities"], copy=True)
    output_null = np.array(base["null_probabilities"], copy=True)
    update_rows = rows[positions]
    update_columns = source_columns[positions]
    update_valid = valid[positions]
    source_rows = np.broadcast_to(update_rows[:, None], update_columns.shape)
    output_probability[source_rows[update_valid], update_columns[update_valid]] = (
        fused_probability[update_valid]
    )
    base_tracks = np.asarray(base["candidate_track_ids"], dtype=np.int64)
    mass = np.sum(
        np.where(base_tracks >= 0, output_probability, 0.0), axis=1
    ) + output_null
    maximum_mass_error = float(np.max(np.abs(mass - 1.0)))
    if maximum_mass_error > 3e-5:
        raise RuntimeError("RGB prior overlay no longer conserves probability mass")

    split_names = np.asarray(evidence["split_names"]).astype(str)
    updated_splits = sorted(set(split_names[positions].tolist()))
    aggregation = str(llr_aggregation)
    if aggregation not in {"sum", "mean"}:
        raise ValueError("llr_aggregation must be sum or mean")
    aggregation_divisor = float(len(prediction_manifests)) if aggregation == "mean" else 1.0
    for manifest in prediction_manifests:
        manifest["effective_llr_weight"] = (
            float(manifest["llr_weight"]) / aggregation_divisor
        )
    metadata = {
        "format": OVERLAY_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used_for_prediction": False,
        "probability_semantics": PROBABILITY_SEMANTICS,
        "proposals_sha256": proposals_hash,
        "base_overlay": str(base_overlay_path),
        "base_overlay_sha256": file_sha256_short(Path(base_overlay_path)),
        "candidate_evidence": str(candidate_evidence_path),
        "candidate_evidence_sha256": expected_candidate_hash,
        "availability_evidence": str(availability_evidence_path),
        "availability_evidence_sha256": expected_availability_hash,
        "prediction_artifacts": prediction_manifests,
        "llr_combination": (
            "mean_of_weighted_candidate_log_likelihood_ratios"
            if aggregation == "mean"
            else "weighted_sum_of_calibrated_candidate_log_likelihood_ratios"
        ),
        "llr_aggregation": aggregation,
        "updated_source_row_count": int(len(update_rows)),
        "updated_source_rows_sha256": _array_sha256_short(update_rows),
        "updated_split_names": updated_splits,
        "full_candidate_coverage_required": True,
        "candidate_vs_null_mass_preserved": True,
        "maximum_probability_mass_error": maximum_mass_error,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        candidate_track_ids=base_tracks,
        candidate_probabilities=output_probability.astype(np.float32),
        null_probabilities=output_null.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    result = {
        "stage": OVERLAY_FORMAT,
        "protocol": {
            "target_free": True,
            "full_candidate_coverage_required": True,
            "candidate_vs_null_mass_preserved": True,
            "prediction_rows_only": True,
            "llr_aggregation": aggregation,
        },
        "inputs": metadata,
        "outputs": {
            "overlay": str(output),
            "overlay_sha256": file_sha256_short(output),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
