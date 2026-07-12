"""Build and audit a frozen top-M candidate pool for selective RGB measurement."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_measurement_schema import (
    CANDIDATE_MEASUREMENT_SCHEMA_VERSION,
)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = (
            {}
            if "metadata_json" not in payload.files
            else json.loads(str(payload["metadata_json"].item()))
        )
    return arrays, metadata


def _require_arrays(
    arrays: Mapping[str, np.ndarray], required: Sequence[str], *, artifact: str
) -> None:
    missing = set(required) - set(arrays)
    if missing:
        raise ValueError(f"{artifact} is missing arrays: {sorted(missing)}")


def _sequence_name(query_id: str) -> str:
    match = re.search(r"(?:^|/)(seq[^/]+)/", str(query_id))
    return "unknown" if match is None else str(match.group(1))


def _rank_columns_descending(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid & np.isfinite(scores), scores, -np.inf)
    return np.argsort(-safe, axis=1, kind="stable")


def _select_candidate_columns(
    *,
    scores: np.ndarray,
    valid: np.ndarray,
    frozen_selected_columns: np.ndarray,
    candidates_per_token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the frozen candidate, then fill from the frozen score order."""

    row_count, candidate_count = scores.shape
    requested = int(candidates_per_token)
    if requested <= 0 or requested > candidate_count:
        raise ValueError("candidates_per_token must be in [1, candidate_top_k]")
    selected = np.full((row_count, requested), -1, dtype=np.int64)
    roles = np.full((row_count, requested), "invalid", dtype="<U24")
    ranked = _rank_columns_descending(scores, valid)
    for row in range(row_count):
        frozen = int(frozen_selected_columns[row])
        if frozen < 0 or frozen >= candidate_count or not bool(valid[row, frozen]):
            raise ValueError("frozen selected candidate is invalid")
        chosen = [frozen]
        for column in ranked[row].tolist():
            if int(column) in chosen or not bool(valid[row, int(column)]):
                continue
            chosen.append(int(column))
            if len(chosen) >= requested:
                break
        selected[row, : len(chosen)] = np.asarray(chosen, dtype=np.int64)
        roles[row, 0] = "frozen_selected"
        for rank in range(1, len(chosen)):
            roles[row, rank] = f"score_alternative_{rank}"
    return selected, roles


def _take_2d(values: np.ndarray, columns: np.ndarray, *, fill: Any) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != columns.shape[0]:
        raise ValueError("candidate array and selected columns are not row-aligned")
    safe = np.maximum(columns, 0)
    output = np.take_along_axis(values, safe, axis=1)
    output = np.array(output, copy=True)
    output[columns < 0] = fill
    return output


def _metric_block(
    residuals: np.ndarray,
    pool_residuals: np.ndarray,
    valid: np.ndarray,
    *,
    threshold_px: float,
) -> dict[str, object]:
    measured_correct = valid & np.isfinite(residuals) & (residuals <= float(threshold_px))
    selected_correct = measured_correct[:, 0]
    any_measured_correct = np.any(measured_correct, axis=1)
    pool_correct = np.any(
        np.isfinite(pool_residuals) & (pool_residuals <= float(threshold_px)), axis=1
    )
    rescue_eligible = (~selected_correct) & pool_correct
    rescued = rescue_eligible & any_measured_correct
    first_rank = np.full((len(residuals),), -1, dtype=np.int64)
    for rank in range(residuals.shape[1]):
        take = (first_rank < 0) & measured_correct[:, rank]
        first_rank[take] = int(rank + 1)
    return {
        "threshold_px": float(threshold_px),
        "token_count": int(len(residuals)),
        "selected_correct_count": int(np.sum(selected_correct)),
        "selected_correct_rate": float(np.mean(selected_correct)),
        "measured_top_m_correct_count": int(np.sum(any_measured_correct)),
        "measured_top_m_correct_rate": float(np.mean(any_measured_correct)),
        "oracle_pool_correct_count": int(np.sum(pool_correct)),
        "oracle_pool_correct_rate": float(np.mean(pool_correct)),
        "rescue_eligible_count": int(np.sum(rescue_eligible)),
        "rescued_count": int(np.sum(rescued)),
        "rescue_recall": (
            None
            if not np.any(rescue_eligible)
            else float(np.sum(rescued) / np.sum(rescue_eligible))
        ),
        "first_correct_candidate_rank_counts": {
            str(rank): int(np.sum(first_rank == rank))
            for rank in range(1, residuals.shape[1] + 1)
        },
    }


def audit_candidate_measurement_selection(
    *,
    query_ids: np.ndarray,
    split_names: np.ndarray,
    selected_residuals: np.ndarray,
    pool_residuals: np.ndarray,
    valid: np.ndarray,
    thresholds_px: Sequence[float] = (1.0, 2.0, 5.0),
    oracle_pool_definition: str = "frozen_top_l",
) -> dict[str, object]:
    """Report rescue availability without using GT to select candidates."""

    query_ids = np.asarray(query_ids).astype(str)
    split_names = np.asarray(split_names).astype(str)
    blocks: dict[str, object] = {}
    for split_name in sorted(set(split_names.tolist())):
        mask = split_names == split_name
        blocks[split_name] = {
            str(float(threshold)): _metric_block(
                selected_residuals[mask],
                pool_residuals[mask],
                valid[mask],
                threshold_px=float(threshold),
            )
            for threshold in thresholds_px
        }
    query_token_counts = Counter(query_ids.tolist())
    sequence_rows: dict[str, int] = defaultdict(int)
    for query_id in query_ids.tolist():
        sequence_rows[_sequence_name(query_id)] += 1
    return {
        "oracle_pool_definition": str(oracle_pool_definition),
        "split_metrics": blocks,
        "coverage": {
            "query_count": int(len(query_token_counts)),
            "token_count": int(len(query_ids)),
            "candidate_count": int(np.sum(valid)),
            "candidates_per_token_median": float(np.median(np.sum(valid, axis=1))),
            "tokens_per_query_min": int(min(query_token_counts.values(), default=0)),
            "tokens_per_query_median": float(
                np.median(list(query_token_counts.values())) if query_token_counts else 0.0
            ),
            "tokens_per_query_max": int(max(query_token_counts.values(), default=0)),
            "token_count_by_sequence": dict(sorted(sequence_rows.items())),
        },
    }


def build_candidate_measurement_selection(
    *,
    proposals_path: Path,
    candidate_artifact_path: Path,
    score_artifact_path: Path,
    policy_artifact_path: Path,
    split_json_path: Path,
    output_path: Path,
    candidate_score_key: str,
    candidates_per_token: int = 3,
) -> dict[str, object]:
    """Create a pose-free frozen top-M candidate set and its rescue oracle audit."""

    proposals, _ = _load_npz(Path(proposals_path))
    candidates, candidate_metadata = _load_npz(Path(candidate_artifact_path))
    scores, _ = _load_npz(Path(score_artifact_path))
    policy, policy_metadata = _load_npz(Path(policy_artifact_path))
    _require_arrays(
        proposals,
        (
            "query_ids",
            "xy",
            "bank_row_indices",
            "candidate_track_ids",
            "candidate_prototype_ids",
            "coarse_scores",
            "candidate_gt_residuals_px",
        ),
        artifact="proposals",
    )
    _require_arrays(
        candidates, ("selected_rows", "selected_columns", "valid_edges"), artifact="candidates"
    )
    _require_arrays(
        policy,
        (
            "selected_rows",
            "candidate_pool_columns",
            "selected_compact_columns",
            "selected_track_ids",
            "selected_prototype_ids",
        ),
        artifact="policy",
    )
    if str(candidate_score_key) not in scores:
        raise ValueError(f"candidate score key is absent: {candidate_score_key}")

    actual_hashes = {
        "proposals_sha256": file_sha256_short(Path(proposals_path)),
        "candidate_artifact_sha256": file_sha256_short(Path(candidate_artifact_path)),
        "score_artifact_sha256": file_sha256_short(Path(score_artifact_path)),
        "split_json_sha256": file_sha256_short(Path(split_json_path)),
    }
    expected_pairs = {
        "candidate_metadata.proposals_sha256": (
            candidate_metadata.get("proposals_sha256"), actual_hashes["proposals_sha256"]
        ),
        "policy.proposals_sha256": (
            policy_metadata.get("proposals_sha256"), actual_hashes["proposals_sha256"]
        ),
        "policy.candidate_artifact_sha256": (
            policy_metadata.get("candidate_artifact_sha256"),
            actual_hashes["candidate_artifact_sha256"],
        ),
        "policy.score_artifact_sha256": (
            policy_metadata.get("score_artifact_sha256"), actual_hashes["score_artifact_sha256"]
        ),
        "policy.split_json_sha256": (
            policy_metadata.get("split_json_sha256"), actual_hashes["split_json_sha256"]
        ),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in expected_pairs.items()
        if expected != actual
    }
    if mismatches:
        raise ValueError(f"candidate measurement inputs are stale: {json.dumps(mismatches, sort_keys=True)}")

    selected_rows = np.asarray(candidates["selected_rows"], dtype=np.int64)
    compact_to_source = np.asarray(candidates["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidates["valid_edges"], dtype=bool)
    if not np.array_equal(selected_rows, np.asarray(policy["selected_rows"], dtype=np.int64)):
        raise ValueError("candidate and policy selected rows differ")
    if not np.array_equal(
        compact_to_source, np.asarray(policy["candidate_pool_columns"], dtype=np.int64)
    ):
        raise ValueError("candidate and policy compact-to-source columns differ")
    score_values = np.asarray(scores[str(candidate_score_key)], dtype=np.float32)
    if score_values.shape != compact_to_source.shape or valid_edges.shape != compact_to_source.shape:
        raise ValueError("score, validity, and candidate pool shapes differ")
    frozen_columns = np.asarray(policy["selected_compact_columns"], dtype=np.int64)
    measured_columns, candidate_roles = _select_candidate_columns(
        scores=score_values,
        valid=valid_edges,
        frozen_selected_columns=frozen_columns,
        candidates_per_token=int(candidates_per_token),
    )
    ranked_columns = _rank_columns_descending(score_values, valid_edges)
    score_rank_by_column = np.empty_like(ranked_columns)
    score_rank_by_column[
        np.arange(len(ranked_columns))[:, None], ranked_columns
    ] = np.arange(1, ranked_columns.shape[1] + 1, dtype=np.int64)[None, :]
    candidate_score_ranks = _take_2d(
        score_rank_by_column, measured_columns, fill=-1
    ).astype(np.int64)
    source_columns = _take_2d(compact_to_source, measured_columns, fill=-1).astype(np.int64)
    source_rows = selected_rows[:, None]
    safe_sources = np.maximum(source_columns, 0)
    safe_rows = np.broadcast_to(source_rows, source_columns.shape)
    candidate_valid = measured_columns >= 0

    def proposal_take(name: str, *, fill: Any) -> np.ndarray:
        values = np.asarray(proposals[name])
        result = values[safe_rows, safe_sources]
        result = np.array(result, copy=True)
        result[~candidate_valid] = fill
        return result

    track_ids = proposal_take("candidate_track_ids", fill=-1).astype(np.int64)
    prototype_ids = proposal_take("candidate_prototype_ids", fill=-1).astype(np.int64)
    bank_rows = proposal_take("bank_row_indices", fill=-1).astype(np.int64)
    coarse_scores = proposal_take("coarse_scores", fill=np.nan).astype(np.float32)
    gt_residuals = proposal_take("candidate_gt_residuals_px", fill=np.inf).astype(np.float32)
    selected_scores = _take_2d(score_values, measured_columns, fill=np.nan).astype(np.float32)
    score_prefix = str(candidate_score_key).split("__", 1)[0]
    geometry_keys = tuple(
        key
        for key in (
            f"{score_prefix}__geometry_p01px",
            f"{score_prefix}__geometry_p02px",
            f"{score_prefix}__geometry_p05px",
        )
        if key in scores
    )
    support_view_keys = tuple(
        sorted(
            (
                key
                for key in scores
                if key.startswith(f"{score_prefix}__support_view_probability_")
            ),
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        )
    )
    if len(geometry_keys) != 3:
        raise ValueError("candidate score artifact lacks the three geometry probabilities")
    if not support_view_keys:
        raise ValueError("candidate score artifact lacks support-view probabilities")
    geometry_probabilities = np.stack(
        [
            _take_2d(np.asarray(scores[key], dtype=np.float32), measured_columns, fill=np.nan)
            for key in geometry_keys
        ],
        axis=2,
    ).astype(np.float32)
    support_view_probabilities = np.stack(
        [
            _take_2d(np.asarray(scores[key], dtype=np.float32), measured_columns, fill=np.nan)
            for key in support_view_keys
        ],
        axis=2,
    ).astype(np.float32)
    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    if not np.array_equal(query_ids, np.asarray(policy.get("query_ids", query_ids)).astype(str)):
        raise ValueError("policy query ids do not align with proposal rows")
    if "query_xy" in policy and not np.allclose(
        query_xy, np.asarray(policy["query_xy"], dtype=np.float32), rtol=0.0, atol=1e-5
    ):
        raise ValueError("policy query coordinates do not align with proposal rows")
    selected_track = track_ids[np.arange(len(track_ids)), 0]
    selected_prototype = prototype_ids[np.arange(len(prototype_ids)), 0]
    if not np.array_equal(selected_track, np.asarray(policy["selected_track_ids"], dtype=np.int64)):
        raise ValueError("frozen selected track identity changed while building top-M")
    if not np.array_equal(
        selected_prototype, np.asarray(policy["selected_prototype_ids"], dtype=np.int64)
    ):
        raise ValueError("frozen selected prototype identity changed while building top-M")

    split_payload = json.loads(Path(split_json_path).read_text())
    split_by_query: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for query_id in split_payload.get(split_name, []):
            previous = split_by_query.setdefault(str(query_id), split_name)
            if previous != split_name:
                raise ValueError("query appears in more than one split")
    split_names = np.asarray(
        [split_by_query.get(str(query_id), "unassigned") for query_id in query_ids],
        dtype="<U16",
    )
    if np.any(split_names == "unassigned"):
        raise ValueError("candidate query is absent from the split manifest")

    pool_gt_residuals = proposal_take("candidate_gt_residuals_px", fill=np.inf)
    # The helper above indexes measured columns; pool supervision must retain all top-L.
    pool_gt_residuals = np.take_along_axis(
        np.asarray(proposals["candidate_gt_residuals_px"], dtype=np.float32)[selected_rows],
        compact_to_source,
        axis=1,
    )
    audit = audit_candidate_measurement_selection(
        query_ids=query_ids,
        split_names=split_names,
        selected_residuals=gt_residuals,
        pool_residuals=pool_gt_residuals,
        valid=candidate_valid,
        oracle_pool_definition=f"frozen_top_l_{compact_to_source.shape[1]}",
    )
    metadata = {
        "format": "candidate_measurement_selection_v1",
        "format_version": CANDIDATE_MEASUREMENT_SCHEMA_VERSION,
        "selection_strategy": "frozen_selected_then_candidate_score_descending",
        "candidate_score_key": str(candidate_score_key),
        "geometry_probability_keys": list(geometry_keys),
        "support_view_probability_keys": list(support_view_keys),
        "candidates_per_token": int(candidates_per_token),
        "ground_truth_used_for_selection": False,
        "pose_used_for_selection": False,
        "render": False,
        "image_retrieval": False,
        "submap": False,
        **actual_hashes,
        "policy_artifact_sha256": file_sha256_short(Path(policy_artifact_path)),
        "descriptor_space_id": policy_metadata.get("descriptor_space_id"),
        "support_geometry_index_sha256": policy_metadata.get(
            "support_geometry_index_sha256"
        ),
        "maplet_support_index_sha256": policy_metadata.get(
            "maplet_support_index_sha256"
        ),
        "projected_landmark_bank_sha256": policy_metadata.get(
            "projected_landmark_bank_sha256"
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        selected_rows=selected_rows,
        query_ids=query_ids,
        query_xy=query_xy,
        split_names=split_names,
        candidate_compact_columns=measured_columns,
        candidate_source_columns=source_columns,
        candidate_roles=candidate_roles,
        candidate_valid=candidate_valid,
        candidate_track_ids=track_ids,
        candidate_prototype_ids=prototype_ids,
        candidate_bank_rows=bank_rows,
        candidate_scores=selected_scores,
        candidate_score_ranks=candidate_score_ranks,
        candidate_coarse_similarities=coarse_scores,
        candidate_geometry_probabilities=geometry_probabilities,
        candidate_support_view_probabilities=support_view_probabilities,
        candidate_gt_residuals_px=gt_residuals,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "candidate_measurement_selection",
        "protocol": metadata,
        "audit": audit,
        "outputs": {
            "selection_artifact": str(output),
            "selection_artifact_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
