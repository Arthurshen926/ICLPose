"""Build a target-free learned candidate posterior aligned to proposal rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "candidate_maplet_prior_overlay_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--inference_scores", required=True)
    parser.add_argument("--inference_summary", required=True)
    parser.add_argument(
        "--candidate_probability_key",
        default="ensemble__factorized_set_candidate_probability",
    )
    parser.add_argument(
        "--dustbin_probability_key",
        default=(
            "ensemble__factorized_set_dustbin_probability_DIAGNOSTIC_ONLY"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _metadata(data: Mapping[str, np.ndarray]) -> dict[str, object]:
    if "metadata_json" not in data:
        raise ValueError("artifact has no metadata_json")
    return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _build_overlay_arrays(
    *,
    proposal_track_ids: np.ndarray,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
    valid_edges: np.ndarray,
    compact_candidate_probabilities: np.ndarray,
    compact_dustbin_probabilities: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    tracks = np.asarray(proposal_track_ids, dtype=np.int64)
    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    columns = np.asarray(selected_columns, dtype=np.int64)
    valid = np.asarray(valid_edges, dtype=bool)
    candidate = np.asarray(compact_candidate_probabilities, dtype=np.float32)
    dustbin = np.asarray(compact_dustbin_probabilities, dtype=np.float32)
    if tracks.ndim != 2:
        raise ValueError("proposal candidate tracks must have shape (N, L)")
    if not np.array_equal(rows, np.arange(tracks.shape[0], dtype=np.int64)):
        raise ValueError("prior overlay requires exact full proposal-row coverage")
    if columns.ndim != 2 or columns.shape != valid.shape or columns.shape != candidate.shape:
        raise ValueError("compact candidate arrays have incompatible shapes")
    if not np.all(valid):
        raise ValueError("prior overlay currently requires a fixed valid candidate pool")
    if np.any((columns < 0) | (columns >= tracks.shape[1])):
        raise ValueError("selected candidate columns are out of range")
    if np.any(np.diff(np.sort(columns, axis=1), axis=1) == 0):
        raise ValueError("selected candidate columns repeat within a row")
    if dustbin.ndim == 2:
        if dustbin.shape != candidate.shape:
            raise ValueError("matrix dustbin probabilities must align with candidates")
        row_spread = np.ptp(dustbin.astype(np.float64), axis=1)
        if np.any(row_spread > 1e-6):
            raise ValueError("dustbin probability must be constant across candidate columns")
        null = dustbin[:, 0]
    elif dustbin.ndim == 1 and dustbin.shape == (len(rows),):
        null = dustbin
    else:
        raise ValueError("dustbin probabilities must have shape (N,) or (N, K)")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(null)):
        raise ValueError("candidate posterior contains non-finite probabilities")
    if np.any((candidate < 0.0) | (candidate > 1.0)) or np.any(
        (null < 0.0) | (null > 1.0)
    ):
        raise ValueError("candidate posterior probabilities must be in [0, 1]")
    compact_mass = np.sum(candidate.astype(np.float64), axis=1) + null.astype(
        np.float64
    )
    maximum_mass_error = float(np.max(np.abs(compact_mass - 1.0)))
    if maximum_mass_error > 1e-5:
        raise ValueError("candidate and dustbin probabilities do not sum to one")

    probabilities = np.zeros(tracks.shape, dtype=np.float32)
    probabilities[rows[:, None], columns] = candidate
    if np.any(np.abs(probabilities[tracks < 0]) > 1e-6):
        raise ValueError("invalid proposal candidates received posterior mass")
    output_mass = np.sum(probabilities.astype(np.float64), axis=1) + null.astype(
        np.float64
    )
    maximum_output_mass_error = float(np.max(np.abs(output_mass - 1.0)))
    if maximum_output_mass_error > 1e-5:
        raise RuntimeError("candidate column remapping lost posterior mass")
    return {
        "candidate_track_ids": tracks,
        "candidate_probabilities": probabilities,
        "null_probabilities": np.asarray(null, dtype=np.float32),
    }, {
        "maximum_input_mass_error": maximum_mass_error,
        "maximum_output_mass_error": maximum_output_mass_error,
        "minimum_null_probability": float(np.min(null)),
        "maximum_null_probability": float(np.max(null)),
        "mean_null_probability": float(np.mean(null)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    proposal_path = Path(args.proposals)
    feature_path = Path(args.feature_artifact)
    score_path = Path(args.inference_scores)
    summary_path = Path(args.inference_summary)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    with np.load(proposal_path, allow_pickle=False) as data:
        proposal_tracks = np.asarray(data["candidate_track_ids"], dtype=np.int64)
    with np.load(feature_path, allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError("prior overlay feature artifact must be inference-only")
        feature_metadata = _metadata(data)
        selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
        selected_columns = np.asarray(data["selected_columns"], dtype=np.int64)
        valid_edges = np.asarray(data["valid_edges"], dtype=bool)
    summary = json.loads(summary_path.read_text())
    protocol = dict(summary.get("protocol") or {})
    if not bool(protocol.get("inference_only", False)) or bool(
        protocol.get("supervision_arrays_loaded", True)
    ):
        raise ValueError("candidate-maplet inference summary is not target-free")
    data_manifest = dict(summary.get("data_manifest") or {})
    expected_manifest = {
        "proposals_sha256": file_sha256_short(proposal_path),
        "feature_artifact_sha256": file_sha256_short(feature_path),
    }
    mismatches = {
        key: {"expected": value, "actual": data_manifest.get(key)}
        for key, value in expected_manifest.items()
        if str(data_manifest.get(key)) != str(value)
    }
    if mismatches:
        raise ValueError(
            "candidate-maplet inference inputs differ: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    if str(feature_metadata.get("proposals_sha256")) != str(
        expected_manifest["proposals_sha256"]
    ) or str(feature_metadata.get("supervision_mode")) != "none_inference_only":
        raise ValueError("feature artifact is stale or not inference-only")
    declared_score_hash = dict(summary.get("outputs") or {}).get("scores_sha256")
    actual_score_hash = file_sha256_short(score_path)
    if str(declared_score_hash) != str(actual_score_hash):
        raise ValueError("inference score hash differs from summary")
    with np.load(score_path, allow_pickle=False) as data:
        candidate = np.asarray(data[str(args.candidate_probability_key)])
        dustbin = np.asarray(data[str(args.dustbin_probability_key)])
    arrays, audit = _build_overlay_arrays(
        proposal_track_ids=proposal_tracks,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        valid_edges=valid_edges,
        compact_candidate_probabilities=candidate,
        compact_dustbin_probabilities=dustbin,
    )
    metadata = {
        "format": ARTIFACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "probability_semantics": (
            "candidate_identity_probability_plus_explicit_null_equals_one"
        ),
        "candidate_probability_key": str(args.candidate_probability_key),
        "dustbin_probability_key": str(args.dustbin_probability_key),
        "proposal_row_count": int(proposal_tracks.shape[0]),
        "candidate_count": int(proposal_tracks.shape[1]),
        "proposals_sha256": expected_manifest["proposals_sha256"],
        "feature_artifact_sha256": expected_manifest["feature_artifact_sha256"],
        "inference_scores_sha256": actual_score_hash,
        "inference_summary_sha256": file_sha256_short(summary_path),
        "inference_data_manifest": data_manifest,
        "inference_query_set": dict(summary.get("query_set") or {}),
        "checkpoint_sha256": [
            str(dict(row).get("sha256")) for row in summary.get("checkpoints") or []
        ],
        "audit": audit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": file_sha256_short(output_path),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
