"""Build the pose-free candidate evidence contract used by latent localization.

The V3 contract preserves probability mass.  Candidate probabilities and the
set dustbin come from one joint softmax; when only top-M candidates are kept,
the omitted candidate mass is therefore unknown, not evidence for the retained
candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


CANDIDATE_EVIDENCE_V3_FORMAT = "candidate_evidence_v3"


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


def _take(values: np.ndarray, columns: np.ndarray, *, fill: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != columns.shape[0]:
        raise ValueError("candidate array and selected columns are not row-aligned")
    safe = np.maximum(columns, 0)
    output = np.take_along_axis(values, safe, axis=1)
    output = np.array(output, copy=True)
    output[columns < 0] = fill
    return output


def _rank_top_m(
    probabilities: np.ndarray, valid: np.ndarray, *, top_m: int
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if probabilities.shape != valid.shape or probabilities.ndim != 2:
        raise ValueError("candidate probabilities and validity must be aligned 2D arrays")
    if int(top_m) <= 0 or int(top_m) > probabilities.shape[1]:
        raise ValueError("top_m must be in [1, candidate_top_l]")
    safe = np.where(valid & np.isfinite(probabilities), probabilities, -np.inf)
    ranked = np.argsort(-safe, axis=1, kind="stable")
    selected = ranked[:, : int(top_m)].astype(np.int64)
    selected_valid = np.take_along_axis(valid, selected, axis=1)
    selected[~selected_valid] = -1
    return selected, selected_valid


def _validate_probability_contract(
    candidate_probabilities: np.ndarray,
    dustbin_probabilities: np.ndarray,
    valid: np.ndarray,
    *,
    atol: float = 2e-5,
) -> None:
    candidates = np.asarray(candidate_probabilities, dtype=np.float64)
    dustbin = np.asarray(dustbin_probabilities, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if candidates.shape != valid.shape or dustbin.shape != candidates.shape:
        raise ValueError("candidate, dustbin, and validity shapes differ")
    repeated_dustbin = dustbin[:, :1]
    if not np.allclose(dustbin, repeated_dustbin, rtol=0.0, atol=float(atol)):
        raise ValueError("set dustbin probability is not constant within a candidate set")
    if np.any(candidates[valid] < -float(atol)) or np.any(candidates[valid] > 1.0 + float(atol)):
        raise ValueError("candidate probability lies outside [0, 1]")
    if np.any(repeated_dustbin < -float(atol)) or np.any(
        repeated_dustbin > 1.0 + float(atol)
    ):
        raise ValueError("dustbin probability lies outside [0, 1]")
    invalid_mass = np.sum(np.where(valid, 0.0, candidates), axis=1)
    if np.any(np.abs(invalid_mass) > float(atol)):
        raise ValueError("invalid candidate slots carry non-zero probability mass")
    total = np.sum(np.where(valid, candidates, 0.0), axis=1) + repeated_dustbin[:, 0]
    if not np.allclose(total, 1.0, rtol=0.0, atol=float(atol)):
        raise ValueError("candidate probabilities and dustbin do not form a joint softmax")


def _split_names(
    query_ids: np.ndarray, split_payload: Mapping[str, Any]
) -> np.ndarray:
    by_query: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for query_id in split_payload.get(split_name, []):
            previous = by_query.setdefault(str(query_id), split_name)
            if previous != split_name:
                raise ValueError("query appears in more than one split")
    result = np.asarray(
        [by_query.get(str(query_id), "unassigned") for query_id in query_ids],
        dtype="<U16",
    )
    if np.any(result == "unassigned"):
        raise ValueError("candidate query is absent from the split manifest")
    return result


def _audit(
    *,
    split_names: np.ndarray,
    candidate_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    residuals_px: np.ndarray,
    pool_residuals_px: np.ndarray,
) -> dict[str, object]:
    blocks: dict[str, object] = {}
    for split_name in sorted(set(np.asarray(split_names).astype(str).tolist())):
        mask = np.asarray(split_names).astype(str) == split_name
        split_block: dict[str, object] = {}
        for threshold in (1.0, 2.0, 5.0):
            correct = np.isfinite(residuals_px[mask]) & (
                residuals_px[mask] <= float(threshold)
            )
            pool_correct = np.any(
                np.isfinite(pool_residuals_px[mask])
                & (pool_residuals_px[mask] <= float(threshold)),
                axis=1,
            )
            top_m_correct = np.any(correct, axis=1)
            posterior_correct = np.sum(
                np.where(correct, candidate_probabilities[mask], 0.0), axis=1
            )
            target_probability = np.where(
                top_m_correct, posterior_correct, unknown_probabilities[mask]
            )
            first_correct = np.full((int(np.sum(mask)),), -1, dtype=np.int64)
            for rank in range(correct.shape[1]):
                take = (first_correct < 0) & correct[:, rank]
                first_correct[take] = rank + 1
            split_block[str(int(threshold))] = {
                "top_m_correct_rate": float(np.mean(top_m_correct)),
                "pool_top_l_correct_rate": float(np.mean(pool_correct)),
                "rescue_recall_from_top_m": (
                    None
                    if not np.any(pool_correct)
                    else float(np.sum(top_m_correct & pool_correct) / np.sum(pool_correct))
                ),
                "identity_target_nll": float(
                    np.mean(-np.log(np.clip(target_probability, 1e-12, 1.0)))
                ),
                "mean_correct_probability_mass_when_present": (
                    None
                    if not np.any(top_m_correct)
                    else float(np.mean(posterior_correct[top_m_correct]))
                ),
                "first_correct_rank_counts": {
                    str(rank): int(np.sum(first_correct == rank))
                    for rank in range(1, correct.shape[1] + 1)
                },
            }
        blocks[split_name] = split_block
    return blocks


def build_candidate_evidence_v3(
    *,
    proposals_path: Path,
    candidate_artifact_path: Path,
    score_artifact_path: Path,
    score_summary_path: Path,
    split_json_path: Path,
    colmap_model_dir: Path,
    projected_landmark_bank_path: Path,
    maplet_support_index_path: Path,
    support_geometry_index_path: Path,
    output_path: Path,
    score_prefix: str = "ensemble",
    candidates_per_token: int = 5,
) -> dict[str, object]:
    """Create a top-M identity posterior with an explicit unknown state."""

    proposals, _ = _load_npz(Path(proposals_path))
    candidates, candidate_metadata = _load_npz(Path(candidate_artifact_path))
    scores, score_metadata = _load_npz(Path(score_artifact_path))
    score_summary = json.loads(Path(score_summary_path).read_text())
    split_payload = json.loads(Path(split_json_path).read_text())
    landmark_arrays, landmark_metadata = _load_npz(Path(projected_landmark_bank_path))
    _maplet_arrays, maplet_metadata = _load_npz(Path(maplet_support_index_path))
    _geometry_arrays, _geometry_metadata = _load_npz(Path(support_geometry_index_path))

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
        candidates,
        ("selected_rows", "selected_columns", "valid_edges"),
        artifact="candidate feature artifact",
    )
    _require_arrays(landmark_arrays, ("track_ids", "features"), artifact="landmark bank")

    prefix = str(score_prefix)
    candidate_key = f"{prefix}__set_candidate_probability"
    dustbin_key = f"{prefix}__set_dustbin_probability_DIAGNOSTIC_ONLY"
    geometry_keys = tuple(
        f"{prefix}__geometry_p{threshold:02d}px" for threshold in (1, 2, 5)
    )
    support_keys = tuple(
        sorted(
            (
                key
                for key in scores
                if key.startswith(f"{prefix}__support_view_probability_")
            ),
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        )
    )
    _require_arrays(
        scores,
        (candidate_key, dustbin_key, *geometry_keys, *support_keys),
        artifact="score artifact",
    )
    if not support_keys:
        raise ValueError("score artifact has no support-view probabilities")

    actual_hashes = {
        "proposals_sha256": file_sha256_short(Path(proposals_path)),
        "candidate_artifact_sha256": file_sha256_short(Path(candidate_artifact_path)),
        "score_artifact_sha256": file_sha256_short(Path(score_artifact_path)),
        "score_summary_sha256": file_sha256_short(Path(score_summary_path)),
        "split_json_sha256": file_sha256_short(Path(split_json_path)),
        "colmap_images_sha256": file_sha256_short(
            Path(colmap_model_dir) / "images.bin"
        ),
        "colmap_cameras_sha256": file_sha256_short(
            Path(colmap_model_dir) / "cameras.bin"
        ),
        "colmap_points3d_sha256": file_sha256_short(
            Path(colmap_model_dir) / "points3D.bin"
        ),
        "projected_landmark_bank_sha256": file_sha256_short(
            Path(projected_landmark_bank_path)
        ),
        "maplet_support_index_sha256": file_sha256_short(
            Path(maplet_support_index_path)
        ),
        "support_geometry_index_sha256": file_sha256_short(
            Path(support_geometry_index_path)
        ),
    }
    data_manifest = dict(score_summary.get("data_manifest") or {})
    expected = {
        "candidate_metadata.proposals_sha256": (
            candidate_metadata.get("proposals_sha256"),
            actual_hashes["proposals_sha256"],
        ),
        "candidate_metadata.projected_landmark_bank_sha256": (
            candidate_metadata.get("projected_landmark_bank_sha256"),
            actual_hashes["projected_landmark_bank_sha256"],
        ),
        "candidate_metadata.maplet_support_index_sha256": (
            candidate_metadata.get("maplet_support_index_sha256"),
            actual_hashes["maplet_support_index_sha256"],
        ),
        "candidate_metadata.support_geometry_index_sha256": (
            candidate_metadata.get("support_geometry_index_sha256"),
            actual_hashes["support_geometry_index_sha256"],
        ),
        "score_summary.proposals_sha256": (
            data_manifest.get("proposals_sha256"), actual_hashes["proposals_sha256"]
        ),
        "score_summary.feature_artifact_sha256": (
            data_manifest.get("feature_artifact_sha256"),
            actual_hashes["candidate_artifact_sha256"],
        ),
        "score_summary.projected_landmark_bank_sha256": (
            data_manifest.get("projected_landmark_bank_sha256"),
            actual_hashes["projected_landmark_bank_sha256"],
        ),
        "score_summary.maplet_support_index_sha256": (
            data_manifest.get("maplet_support_index_sha256"),
            actual_hashes["maplet_support_index_sha256"],
        ),
        "score_summary.support_geometry_index_sha256": (
            data_manifest.get("support_geometry_index_sha256"),
            actual_hashes["support_geometry_index_sha256"],
        ),
    }
    declared_score_hash = (score_summary.get("outputs") or {}).get("scores_sha256")
    if declared_score_hash is not None:
        expected["score_summary.outputs.scores_sha256"] = (
            declared_score_hash,
            actual_hashes["score_artifact_sha256"],
        )
    mismatches = {
        key: {"declared": declared, "actual": actual}
        for key, (declared, actual) in expected.items()
        if declared != actual
    }
    if mismatches:
        raise ValueError(
            f"candidate evidence inputs are stale: {json.dumps(mismatches, sort_keys=True)}"
        )

    selected_rows = np.asarray(candidates["selected_rows"], dtype=np.int64)
    compact_to_source = np.asarray(candidates["selected_columns"], dtype=np.int64)
    valid = np.asarray(candidates["valid_edges"], dtype=bool)
    candidate_probabilities = np.asarray(scores[candidate_key], dtype=np.float32)
    dustbin_probabilities = np.asarray(scores[dustbin_key], dtype=np.float32)
    if candidate_probabilities.shape != compact_to_source.shape:
        raise ValueError("score and compact candidate shapes differ")
    _validate_probability_contract(candidate_probabilities, dustbin_probabilities, valid)
    selected_columns, selected_valid = _rank_top_m(
        candidate_probabilities, valid, top_m=int(candidates_per_token)
    )
    selected_probabilities = _take(
        candidate_probabilities, selected_columns, fill=0.0
    ).astype(np.float32)
    source_dustbin = dustbin_probabilities[:, 0].astype(np.float32)
    retained_mass = np.sum(selected_probabilities, axis=1).astype(np.float32)
    full_candidate_mass = np.sum(
        np.where(valid, candidate_probabilities, 0.0), axis=1
    ).astype(np.float32)
    omitted_mass = np.maximum(full_candidate_mass - retained_mass, 0.0).astype(
        np.float32
    )
    unknown_probability = (source_dustbin + omitted_mass).astype(np.float32)
    total = retained_mass + unknown_probability
    if not np.allclose(total, 1.0, rtol=0.0, atol=3e-5):
        raise RuntimeError("retained candidate and unknown probability mass changed")

    source_columns = _take(compact_to_source, selected_columns, fill=-1).astype(np.int64)
    safe_source = np.maximum(source_columns, 0)
    source_rows = np.broadcast_to(selected_rows[:, None], source_columns.shape)

    def proposal_take(name: str, *, fill: Any) -> np.ndarray:
        values = np.asarray(proposals[name])
        output = values[source_rows, safe_source]
        output = np.array(output, copy=True)
        output[~selected_valid] = fill
        return output

    geometry = np.stack(
        [
            _take(np.asarray(scores[key]), selected_columns, fill=np.nan)
            for key in geometry_keys
        ],
        axis=2,
    ).astype(np.float32)
    support = np.stack(
        [
            _take(np.asarray(scores[key]), selected_columns, fill=np.nan)
            for key in support_keys
        ],
        axis=2,
    ).astype(np.float32)
    support_sum = np.nansum(support, axis=2)
    if not np.allclose(support_sum[selected_valid], 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("support-view probabilities do not sum to one")

    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    split_names = _split_names(query_ids, split_payload)
    summary_split = score_summary.get("split")
    if isinstance(summary_split, Mapping):
        summary_split_names = _split_names(query_ids, summary_split)
        if not np.array_equal(summary_split_names, split_names):
            raise ValueError("score summary and split manifest disagree")

    residuals = proposal_take("candidate_gt_residuals_px", fill=np.inf).astype(
        np.float32
    )
    pool_residuals = np.take_along_axis(
        np.asarray(proposals["candidate_gt_residuals_px"], dtype=np.float32)[
            selected_rows
        ],
        compact_to_source,
        axis=1,
    )
    audit = _audit(
        split_names=split_names,
        candidate_probabilities=selected_probabilities,
        unknown_probabilities=unknown_probability,
        residuals_px=residuals,
        pool_residuals_px=pool_residuals,
    )

    roles = np.asarray(
        [
            [f"identity_rank_{rank + 1}" for rank in range(int(candidates_per_token))]
            for _ in range(len(selected_rows))
        ],
        dtype="<U24",
    )
    roles[~selected_valid] = "invalid"
    checkpoint_manifest = [
        {
            "sha256": item.get("sha256"),
            "format": item.get("format"),
            "seed": item.get("seed"),
        }
        for item in score_summary.get("checkpoints", [])
    ]
    metadata = {
        "format": CANDIDATE_EVIDENCE_V3_FORMAT,
        "format_version": 3,
        "selection_strategy": "pose_free_set_identity_probability_descending",
        "candidate_probability_semantics": "joint_softmax_with_set_dustbin",
        "unknown_probability_semantics": "set_dustbin_plus_omitted_top_l_mass",
        "candidate_score_key": candidate_key,
        "dustbin_score_key": dustbin_key,
        "geometry_probability_keys": list(geometry_keys),
        "support_view_probability_keys": list(support_keys),
        "candidates_per_token": int(candidates_per_token),
        "source_candidate_top_l": int(candidate_probabilities.shape[1]),
        "ground_truth_used_for_selection": False,
        "ground_truth_arrays_target_only": ["candidate_target_gt_residuals_px"],
        "pose_used_for_selection": False,
        "render": False,
        "image_retrieval": False,
        "submap": False,
        "score_artifact_embedded_metadata_present": bool(score_metadata),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
        "descriptor_space_manifest": landmark_metadata.get("descriptor_space_manifest"),
        "score_checkpoints": checkpoint_manifest,
        "maplet_source_descriptor_space_id": maplet_metadata.get(
            "source_descriptor_space_id"
        ),
        "colmap_model_dir": str(Path(colmap_model_dir)),
        **actual_hashes,
    }
    if metadata["descriptor_space_id"] != metadata["maplet_source_descriptor_space_id"]:
        raise ValueError("landmark and maplet descriptor spaces differ")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        selected_rows=selected_rows,
        query_ids=query_ids,
        query_xy=query_xy,
        split_names=split_names,
        candidate_compact_columns=selected_columns,
        candidate_source_columns=source_columns,
        candidate_roles=roles,
        candidate_valid=selected_valid,
        candidate_track_ids=proposal_take("candidate_track_ids", fill=-1).astype(np.int64),
        candidate_prototype_ids=proposal_take(
            "candidate_prototype_ids", fill=-1
        ).astype(np.int64),
        candidate_bank_rows=proposal_take("bank_row_indices", fill=-1).astype(np.int64),
        candidate_prior_probabilities=selected_probabilities,
        candidate_score_ranks=np.broadcast_to(
            np.arange(1, int(candidates_per_token) + 1, dtype=np.int64)[None, :],
            selected_columns.shape,
        ),
        candidate_coarse_similarities=proposal_take(
            "coarse_scores", fill=np.nan
        ).astype(np.float32),
        candidate_geometry_probabilities=geometry,
        candidate_support_view_probabilities=support,
        source_set_dustbin_probability=source_dustbin,
        retained_candidate_probability_mass=retained_mass,
        omitted_candidate_probability_mass=omitted_mass,
        unknown_probability=unknown_probability,
        candidate_target_gt_residuals_px=residuals,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    result = {
        "stage": CANDIDATE_EVIDENCE_V3_FORMAT,
        "protocol": metadata,
        "probability_mass": {
            "source_dustbin_mean": float(np.mean(source_dustbin)),
            "retained_candidate_mass_mean": float(np.mean(retained_mass)),
            "omitted_candidate_mass_mean": float(np.mean(omitted_mass)),
            "unknown_probability_mean": float(np.mean(unknown_probability)),
            "mass_sum_max_abs_error": float(np.max(np.abs(total - 1.0))),
        },
        "identity_audit": audit,
        "outputs": {
            "candidate_evidence": str(output),
            "candidate_evidence_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
