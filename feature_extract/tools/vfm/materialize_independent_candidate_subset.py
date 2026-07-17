"""Materialize the exact held-out candidate rows used by pose verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


def _metadata(payload: np.lib.npyio.NpzFile, *, name: str) -> dict[str, object]:
    if "metadata_json" not in payload.files:
        raise ValueError(f"{name} has no metadata")
    value = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(value, dict):
        raise ValueError(f"{name} metadata must be an object")
    return value


def _heldout_rows(
    *,
    query_ids: np.ndarray,
    image_ids: np.ndarray,
    offsets: np.ndarray,
    coarse_scores: np.ndarray,
    detector_scores: np.ndarray,
    fit_rows: np.ndarray,
    point_count: int,
    detector_log_merit_weight: float,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for image_index, query_id in enumerate(image_ids.astype(str).tolist()):
        all_rows = np.arange(
            int(offsets[image_index]), int(offsets[image_index + 1]), dtype=np.int64
        )
        local_fit = fit_rows[query_ids[fit_rows].astype(str) == str(query_id)]
        unused = np.setdiff1d(all_rows, local_fit, assume_unique=True)
        coarse_reference = np.max(
            np.asarray(coarse_scores[unused], dtype=np.float64), axis=1
        )
        merit = coarse_reference + float(detector_log_merit_weight) * np.log(
            np.maximum(np.asarray(detector_scores[unused], dtype=np.float64), 1e-12)
        )
        kept = unused[
            np.argsort(-merit, kind="mergesort")[: min(int(point_count), len(unused))]
        ]
        selected.append(kept)
    output = np.concatenate(selected).astype(np.int64)
    if len(np.unique(output)) != len(output):
        raise RuntimeError("held-out candidate subset repeats source rows")
    if np.intersect1d(output, fit_rows).size:
        raise RuntimeError("held-out candidate subset overlaps fit rows")
    return output


def materialize_independent_candidate_subset(
    *,
    proposals_path: Path,
    detector_query_cache_path: Path,
    fit_candidate_artifact_path: Path,
    source_candidate_artifact_path: Path,
    source_score_artifact_path: Path,
    source_score_summary_path: Path,
    output_dir: Path,
    point_count: int = 192,
    detector_log_merit_weight: float = 0.01,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    feature_output = output_dir / "features_inference_only.npz"
    score_output = output_dir / "inference_scores.npz"
    summary_output = output_dir / "summary.json"
    if any(path.exists() for path in (feature_output, score_output, summary_output)):
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    with np.load(proposals_path, allow_pickle=False) as payload:
        query_ids = np.asarray(payload["query_ids"]).astype(str)
        coarse_scores = np.asarray(payload["coarse_scores"], dtype=np.float32)
    with np.load(detector_query_cache_path, allow_pickle=False) as payload:
        image_ids = np.asarray(payload["image_ids"]).astype(str)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        detector_scores = np.asarray(payload["detector_scores"], dtype=np.float32)
    with np.load(fit_candidate_artifact_path, allow_pickle=False) as payload:
        fit_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
        fit_metadata = _metadata(payload, name="fit candidate artifact")
    with np.load(source_candidate_artifact_path, allow_pickle=False) as payload:
        forbidden = {"labels", "selected_from_pose_keep"} & set(payload.files)
        if forbidden:
            raise ValueError(f"source candidate artifact contains {sorted(forbidden)}")
        source_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
        source_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
        source_features = np.asarray(payload["features"], dtype=np.float32)
        source_valid = np.asarray(payload["valid_edges"], dtype=bool)
        feature_names = np.asarray(payload["feature_names"])
        source_metadata = _metadata(payload, name="source candidate artifact")
    if str(source_metadata.get("supervision_mode")) != "none_inference_only":
        raise ValueError("source candidate artifact is not inference-only")
    proposal_hash = file_sha256_short(proposals_path)
    if str(source_metadata.get("proposals_sha256")) != proposal_hash:
        raise ValueError("source candidate artifact references different proposals")
    selected = _heldout_rows(
        query_ids=query_ids,
        image_ids=image_ids,
        offsets=offsets,
        coarse_scores=coarse_scores,
        detector_scores=detector_scores,
        fit_rows=fit_rows,
        point_count=int(point_count),
        detector_log_merit_weight=float(detector_log_merit_weight),
    )
    source_position = np.full((len(query_ids),), -1, dtype=np.int64)
    source_position[source_rows] = np.arange(len(source_rows), dtype=np.int64)
    positions = source_position[selected]
    if np.any(positions < 0):
        raise ValueError("source candidate artifact does not cover held-out rows")

    feature_metadata = dict(source_metadata)
    feature_metadata.update(
        {
            "contains_ground_truth": False,
            "contains_pose_derived_selection": False,
            "query_point_selection": "independent_verifier_exact_heldout_merit_v1",
            "query_points_per_image": int(point_count),
            "selected_row_count": int(len(selected)),
            "fit_candidate_artifact_sha256": file_sha256_short(
                fit_candidate_artifact_path
            ),
            "detector_log_merit_weight": float(detector_log_merit_weight),
            "source_inference_feature_artifact_sha256": file_sha256_short(
                source_candidate_artifact_path
            ),
            "supervision_mode": "none_inference_only",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        feature_output,
        selected_rows=selected,
        selected_columns=source_columns[positions],
        features=source_features[positions],
        valid_edges=source_valid[positions],
        feature_names=feature_names,
        metadata_json=np.asarray(json.dumps(feature_metadata, sort_keys=True)),
    )

    with np.load(source_score_artifact_path, allow_pickle=False) as payload:
        score_arrays = {
            key: np.asarray(payload[key])[positions]
            for key in payload.files
            if key != "metadata_json"
        }
    np.savez_compressed(score_output, **score_arrays)
    summary = json.loads(Path(source_score_summary_path).read_text())
    if str((summary.get("outputs") or {}).get("scores_sha256")) != str(
        file_sha256_short(source_score_artifact_path)
    ):
        raise ValueError("source score summary and score artifact differ")
    data_manifest = dict(summary.get("data_manifest") or {})
    if str(data_manifest.get("feature_artifact_sha256")) != str(
        file_sha256_short(source_candidate_artifact_path)
    ):
        raise ValueError("source score summary and candidate artifact differ")
    data_manifest["feature_artifact_sha256"] = file_sha256_short(feature_output)
    summary["data_manifest"] = data_manifest
    summary["outputs"] = {
        **dict(summary.get("outputs") or {}),
        "scores": str(score_output),
        "scores_sha256": file_sha256_short(score_output),
    }
    summary["query_set"] = {
        "selection": "independent_verifier_exact_heldout_merit_v1",
        "query_count": int(len(image_ids)),
        "row_count": int(len(selected)),
        "query_points_per_image": int(point_count),
        "fit_candidate_artifact_sha256": file_sha256_short(
            fit_candidate_artifact_path
        ),
    }
    summary["protocol"] = {
        **dict(summary.get("protocol") or {}),
        "inference_only": True,
        "supervision_arrays_loaded": False,
        "materialized_exact_score_subset": True,
        "source_score_artifact_sha256": file_sha256_short(
            source_score_artifact_path
        ),
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "stage": "independent_candidate_exact_heldout_subset",
        "row_count": int(len(selected)),
        "query_count": int(len(image_ids)),
        "fit_row_count": int(len(fit_rows)),
        "source_candidate_artifact_sha256": file_sha256_short(
            source_candidate_artifact_path
        ),
        "source_score_artifact_sha256": file_sha256_short(source_score_artifact_path),
        "feature_artifact": str(feature_output),
        "feature_artifact_sha256": file_sha256_short(feature_output),
        "score_artifact": str(score_output),
        "score_artifact_sha256": file_sha256_short(score_output),
        "summary": str(summary_output),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--fit_candidate_artifact", required=True)
    parser.add_argument("--source_candidate_artifact", required=True)
    parser.add_argument("--source_score_artifact", required=True)
    parser.add_argument("--source_score_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--point_count", type=int, default=192)
    parser.add_argument("--detector_log_merit_weight", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = materialize_independent_candidate_subset(
        proposals_path=Path(args.proposals),
        detector_query_cache_path=Path(args.detector_query_cache),
        fit_candidate_artifact_path=Path(args.fit_candidate_artifact),
        source_candidate_artifact_path=Path(args.source_candidate_artifact),
        source_score_artifact_path=Path(args.source_score_artifact),
        source_score_summary_path=Path(args.source_score_summary),
        output_dir=Path(args.output_dir),
        point_count=int(args.point_count),
        detector_log_merit_weight=float(args.detector_log_merit_weight),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
