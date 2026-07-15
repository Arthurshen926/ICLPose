"""Join frozen target-free matcher scores to labels in a separate audit process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} has no metadata_json")
    return json.loads(str(np.asarray(payload["metadata_json"]).item()))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("average-precision inputs are invalid")
    positive_count = int(np.count_nonzero(target))
    if positive_count <= 0:
        return float("nan")
    order = np.argsort(-values, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positive_count)


def _identity_metrics(
    probabilities: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
    baseline_probabilities: np.ndarray,
    null_probabilities: np.ndarray | None,
    ranking_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    ranking = (
        probability
        if ranking_scores is None
        else np.asarray(ranking_scores, dtype=np.float64)
    )
    if not (
        probability.shape
        == target.shape
        == candidate_valid.shape
        == baseline.shape
        == ranking.shape
        and selected.shape == probability.shape[:1]
    ):
        raise ValueError("identity audit arrays have incompatible shapes")
    if np.any(~np.isfinite(probability[candidate_valid])) or np.any(
        probability[candidate_valid] < 0.0
    ):
        raise ValueError("identity probabilities must be finite and non-negative")
    if np.any(~np.isfinite(ranking[candidate_valid])):
        raise ValueError("identity ranking scores must be finite")
    rows = np.flatnonzero(selected)
    if len(rows) == 0:
        raise ValueError("identity audit split has no rows")
    split_valid = candidate_valid[rows]
    if np.any(np.sum(split_valid, axis=1) <= 0):
        raise ValueError("identity audit row has no valid candidate")
    split_target = target[rows]
    split_probability = probability[rows]
    split_ranking = ranking[rows]
    split_baseline = baseline[rows]
    mappable = np.any(split_target & split_valid, axis=1)
    predicted = np.argmax(
        np.where(split_valid, split_ranking, -np.inf), axis=1
    )
    baseline_predicted = np.argmax(
        np.where(split_valid, split_baseline, -np.inf), axis=1
    )
    local_rows = np.arange(len(rows), dtype=np.int64)
    predicted_correct = split_target[local_rows, predicted]
    baseline_correct = split_target[local_rows, baseline_predicted]
    switched = predicted != baseline_predicted
    valid_flat = split_valid.reshape(-1)
    pair_ap = _average_precision(
        split_target.reshape(-1)[valid_flat],
        split_ranking.reshape(-1)[valid_flat],
    )
    positive_mass = np.sum(
        np.where(split_target & split_valid, split_probability, 0.0), axis=1
    )
    nll_mask = mappable.copy()
    target_mass = positive_mass
    if null_probabilities is not None:
        null = np.asarray(null_probabilities, dtype=np.float64).reshape(-1)
        if null.shape != selected.shape or np.any(~np.isfinite(null)):
            raise ValueError("null probabilities are invalid")
        target_mass = np.where(mappable, positive_mass, null[rows])
        nll_mask = np.ones_like(mappable)
    nll = float(
        np.mean(-np.log(np.clip(target_mass[nll_mask], 1e-12, 1.0)))
    )
    return {
        "row_count": int(len(rows)),
        "mappable_row_count": int(np.count_nonzero(mappable)),
        "pair_positive_average_precision": pair_ap,
        "conditional_top1_accuracy_mappable": (
            None
            if not np.any(mappable)
            else float(np.mean(predicted_correct[mappable]))
        ),
        "target_nll": nll,
        "transition_vs_baseline": {
            "switch_count": int(np.count_nonzero(switched)),
            "mappable_switch_count": int(np.count_nonzero(switched & mappable)),
            "rescued_count": int(
                np.count_nonzero(
                    switched & mappable & ~baseline_correct & predicted_correct
                )
            ),
            "harmed_count": int(
                np.count_nonzero(
                    switched & mappable & baseline_correct & ~predicted_correct
                )
            ),
            "baseline_correct_count": int(
                np.count_nonzero(baseline_correct & mappable)
            ),
            "candidate_correct_count": int(
                np.count_nonzero(predicted_correct & mappable)
            ),
        },
    }


def _split_masks(
    query_ids: np.ndarray, split_payload: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    split_by_query: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for query_id in split_payload.get(split_name, []):
            previous = split_by_query.setdefault(str(query_id), split_name)
            if previous != split_name:
                raise ValueError("query appears in multiple audit splits")
    assigned = np.asarray(
        [split_by_query.get(str(query_id), "") for query_id in query_ids]
    )
    if np.any(assigned == ""):
        raise ValueError("inference row is absent from the audit split manifest")
    return {name: assigned == name for name in ("train", "validation", "test")}


def audit_candidate_maplet_inference_scores(
    *,
    proposals_path: Path,
    inference_feature_artifact_path: Path,
    supervised_feature_artifact_path: Path,
    inference_scores_path: Path,
    inference_summary_path: Path,
    split_json_path: Path,
    score_prefix: str = "ensemble",
) -> dict[str, Any]:
    summary = json.loads(Path(inference_summary_path).read_text())
    protocol = dict(summary.get("protocol") or {})
    if not bool(protocol.get("inference_only", False)) or bool(
        protocol.get("supervision_arrays_loaded", True)
    ):
        raise ValueError("score summary is not from target-free inference")
    actual_score_hash = file_sha256_short(Path(inference_scores_path))
    if str(dict(summary.get("outputs") or {}).get("scores_sha256")) != str(
        actual_score_hash
    ):
        raise ValueError("inference score artifact differs from its summary")

    with np.load(Path(inference_feature_artifact_path), allow_pickle=False) as payload:
        if "labels" in payload.files:
            raise ValueError("inference feature artifact unexpectedly contains labels")
        inference_metadata = _metadata(payload, context="inference feature artifact")
        inference_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
        inference_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
        inference_valid = np.asarray(payload["valid_edges"], dtype=bool)
    with np.load(Path(supervised_feature_artifact_path), allow_pickle=False) as payload:
        supervised_metadata = _metadata(payload, context="supervised feature artifact")
        supervised_rows = np.asarray(payload["selected_rows"], dtype=np.int64)
        supervised_columns = np.asarray(payload["selected_columns"], dtype=np.int64)
        supervised_valid = np.asarray(payload["valid_edges"], dtype=bool)
        labels = np.asarray(payload["labels"], dtype=bool)
    for name, expected, actual in (
        ("selected_rows", inference_rows, supervised_rows),
        ("selected_columns", inference_columns, supervised_columns),
        ("valid_edges", inference_valid, supervised_valid),
    ):
        if not np.array_equal(expected, actual):
            raise ValueError(f"supervised join changes {name}")
    inference_feature_hash = file_sha256_short(Path(inference_feature_artifact_path))
    if str(supervised_metadata.get("source_inference_feature_artifact_sha256")) != str(
        inference_feature_hash
    ):
        raise ValueError("supervised feature labels are not derived from inference rows")
    if str(inference_metadata.get("supervision_mode")) != "none_inference_only":
        raise ValueError("inference feature artifact has the wrong supervision mode")
    if labels.shape != inference_columns.shape or labels.shape != inference_valid.shape:
        raise ValueError("supervised labels do not align with inference candidates")

    manifest = dict(summary.get("data_manifest") or {})
    expected_manifest = {
        "proposals_sha256": file_sha256_short(Path(proposals_path)),
        "feature_artifact_sha256": inference_feature_hash,
    }
    mismatches = {
        key: {"summary": manifest.get(key), "audit": value}
        for key, value in expected_manifest.items()
        if str(manifest.get(key)) != str(value)
    }
    if mismatches:
        raise ValueError(
            "inference summary lineage mismatch: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    with np.load(Path(proposals_path), allow_pickle=False) as payload:
        proposal_query_ids = np.asarray(payload["query_ids"]).astype(str)
    query_ids = proposal_query_ids[inference_rows]
    split_masks = _split_masks(
        query_ids, json.loads(Path(split_json_path).read_text())
    )
    prefix = str(score_prefix)
    final_key = f"{prefix}__factorized_set_candidate_probability"
    null_key = (
        f"{prefix}__factorized_set_dustbin_probability_DIAGNOSTIC_ONLY"
    )
    direct_key = (
        f"{prefix}__set_identity_evidence_conditional_probability_DIAGNOSTIC_ONLY"
    )
    availability_key = (
        f"{prefix}__factorized_top_l_availability_probability_DIAGNOSTIC_ONLY"
    )
    with np.load(Path(inference_scores_path), allow_pickle=False) as payload:
        required = {
            "selected_columns",
            "baseline_scores",
            final_key,
            null_key,
            direct_key,
            availability_key,
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"inference scores lack arrays: {sorted(missing)}")
        selected_output_columns = np.asarray(
            payload["selected_columns"], dtype=np.int64
        )
        if selected_output_columns.shape != (len(inference_rows),) or np.any(
            (selected_output_columns < 0)
            | (selected_output_columns >= inference_columns.shape[1])
        ):
            raise ValueError("inference selected output columns are invalid")
        baseline_scores = np.asarray(payload["baseline_scores"], dtype=np.float64)
        final_probability = np.asarray(payload[final_key], dtype=np.float64)
        null_matrix = np.asarray(payload[null_key], dtype=np.float64)
        direct_probability = np.asarray(payload[direct_key], dtype=np.float64)
        availability_matrix = np.asarray(payload[availability_key], dtype=np.float64)
    for name, values in (
        ("baseline", baseline_scores),
        ("final", final_probability),
        ("null", null_matrix),
        ("direct", direct_probability),
        ("availability", availability_matrix),
    ):
        if values.shape != labels.shape:
            raise ValueError(f"{name} score shape differs from candidate labels")
    valid_count = np.sum(inference_valid, axis=1)
    baseline_probability = np.where(inference_valid, baseline_scores, -np.inf)
    baseline_probability -= np.max(baseline_probability, axis=1, keepdims=True)
    baseline_probability = np.where(
        inference_valid, np.exp(baseline_probability), 0.0
    )
    baseline_probability /= np.sum(baseline_probability, axis=1, keepdims=True)
    if np.any(valid_count <= 0):
        raise ValueError("inference candidate group is empty")
    if np.any(np.ptp(null_matrix, axis=1) > 1e-6) or np.any(
        np.ptp(availability_matrix, axis=1) > 1e-6
    ):
        raise ValueError("set-level probabilities differ across candidate slots")
    null_probability = null_matrix[:, 0]
    availability_probability = availability_matrix[:, 0]
    mass = np.sum(np.where(inference_valid, final_probability, 0.0), axis=1)
    if not np.allclose(mass + null_probability, 1.0, atol=2e-5, rtol=0.0):
        raise ValueError("final candidate and null probability mass changed")

    split_metrics: dict[str, Any] = {}
    for split_name, row_mask in split_masks.items():
        availability_labels = np.any(labels & inference_valid, axis=1)
        split_metrics[split_name] = {
            "baseline": _identity_metrics(
                baseline_probability,
                labels=labels,
                valid=inference_valid,
                row_mask=row_mask,
                baseline_probabilities=baseline_probability,
                null_probabilities=None,
                ranking_scores=baseline_scores,
            ),
            "direct_independent_evidence": _identity_metrics(
                direct_probability,
                labels=labels,
                valid=inference_valid,
                row_mask=row_mask,
                baseline_probabilities=baseline_probability,
                null_probabilities=None,
            ),
            "factorized_prior_plus_evidence": _identity_metrics(
                final_probability,
                labels=labels,
                valid=inference_valid,
                row_mask=row_mask,
                baseline_probabilities=baseline_probability,
                null_probabilities=null_probability,
            ),
            "top_l_availability": confidence_metrics(
                availability_labels[row_mask], availability_probability[row_mask]
            ),
        }
    return {
        "stage": "candidate_maplet_external_target_join_audit",
        "protocol": {
            "model_inference_and_target_join_are_separate_processes": True,
            "inference_scores_frozen_before_labels_loaded": True,
            "query_pose_loaded_by_model_inference": False,
            "selection_block": "validation_only",
            "test_block_role": "reused_diagnostic_only",
            "positive_semantics": "candidate_gt_projection_residual_le_2px",
        },
        "inputs": {
            "proposals": str(proposals_path),
            "proposals_sha256": expected_manifest["proposals_sha256"],
            "inference_feature_artifact": str(inference_feature_artifact_path),
            "inference_feature_artifact_sha256": inference_feature_hash,
            "supervised_feature_artifact": str(supervised_feature_artifact_path),
            "supervised_feature_artifact_sha256": file_sha256_short(
                Path(supervised_feature_artifact_path)
            ),
            "inference_scores": str(inference_scores_path),
            "inference_scores_sha256": actual_score_hash,
            "inference_summary": str(inference_summary_path),
            "inference_summary_sha256": file_sha256_short(
                Path(inference_summary_path)
            ),
            "split_json": str(split_json_path),
            "split_json_sha256": file_sha256_short(Path(split_json_path)),
        },
        "score_keys": {
            "final": final_key,
            "null": null_key,
            "direct": direct_key,
            "availability": availability_key,
        },
        "metrics": split_metrics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--inference_feature_artifact", required=True)
    parser.add_argument("--supervised_feature_artifact", required=True)
    parser.add_argument("--inference_scores", required=True)
    parser.add_argument("--inference_summary", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--score_prefix", default="ensemble")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = audit_candidate_maplet_inference_scores(
        proposals_path=Path(args.proposals),
        inference_feature_artifact_path=Path(args.inference_feature_artifact),
        supervised_feature_artifact_path=Path(args.supervised_feature_artifact),
        inference_scores_path=Path(args.inference_scores),
        inference_summary_path=Path(args.inference_summary),
        split_json_path=Path(args.split_json),
        score_prefix=str(args.score_prefix),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
