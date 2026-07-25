"""Attribute target-only lifted-LoFTR tail pairs by group, candidate and view.

The input score artifacts are generated without targets.  A separate
target-only manifest names the wrong selected and frozen oracle hypotheses.
This audit explains their difference; it never emits a replacement score or
an inference-time decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_frozen_lifted_loftr_tail_manifest import (
    MANIFEST_FORMAT,
)
from feature_extract.tools.vfm.score_frozen_lifted_loftr_map_to_query_pose_evidence import (
    EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW,
    SCORE_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short


REPORT_FORMAT = "frozen_lifted_loftr_candidate_view_tail_attribution_v1"
_DIAGNOSTIC_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "profile_names",
    "candidate_group_track_indices",
    "candidate_group_reference_xy",
    "candidate_group_active_mask",
    "candidate_group_identity_probabilities",
    "candidate_group_null_probabilities",
    "candidate_group_support_slot_indices",
    "candidate_group_support_image_ids",
    "candidate_group_support_view_probabilities",
    "canonical_track_ids",
    "support_slot_track_indices",
    "support_slot_image_ids",
    "mode_offsets",
    "mode_query_xy",
    "mode_weights",
    "mode_confidence_sums",
    "support_reliabilities",
    "support_reference_xy",
    "diagnostic_support_view_slot_log_ratios",
    "diagnostic_candidate_log_ratios",
    "diagnostic_group_log_ratios",
    "diagnostic_projected_xy",
    "diagnostic_projection_valid",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-score-artifacts", required=True)
    parser.add_argument("--tail-manifest", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--top-candidates-per-group", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("diagnostic score paths must be non-empty and unique")
    return paths


def _metadata(path: Path, payload: Mapping[str, np.ndarray]) -> dict[str, object]:
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: diagnostic score metadata is malformed") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: diagnostic score metadata must be an object")
    return metadata


def _input_hashes(metadata: Mapping[str, object]) -> dict[str, str]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("diagnostic score lacks a target-free input manifest")
    output: dict[str, str] = {}
    for name, item in inputs.items():
        if not isinstance(item, Mapping) or not str(item.get("sha256", "")):
            raise ValueError("diagnostic score input manifest is incomplete")
        output[str(name)] = str(item["sha256"])
    return output


def _load_diagnostic_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(_DIAGNOSTIC_FIELDS).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: missing diagnostic field {missing[0] if missing else 'metadata_json'}")
        arrays = {name: np.asarray(payload[name]).copy() for name in _DIAGNOSTIC_FIELDS}
        metadata_probe = {"metadata_json": np.asarray(payload["metadata_json"]).copy()}
    metadata = _metadata(Path(path), metadata_probe)
    runtime = metadata.get("runtime")
    if (
        metadata.get("format") != SCORE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("evidence_layout") != EVIDENCE_LAYOUT_CANDIDATE_SUPPORT_VIEW
        or not isinstance(runtime, Mapping)
        or runtime.get("diagnostic_hypothesis_selection") is not True
        or runtime.get("diagnostic_group_terms_dumped") is not True
        or runtime.get("all_frozen_hypotheses") is not False
    ):
        raise ValueError(f"{path}: not an explicit target-only diagnostic scorer output")
    requested = np.asarray(runtime.get("diagnostic_hypothesis_indices", []), dtype=np.int64)
    hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1)
    row_count = int(metadata.get("row_count", -1))
    names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
    slots = np.asarray(arrays["support_slot_track_indices"], dtype=np.int64).reshape(-1)
    tracks = np.asarray(arrays["canonical_track_ids"], dtype=np.int64).reshape(-1)
    if (
        row_count <= 0
        or hypotheses.shape != (row_count,)
        or not np.array_equal(hypotheses, requested)
        or len(np.unique(hypotheses)) != len(hypotheses)
        or len(names) == 0
        or len(slots) == 0
        or len(tracks) == 0
        or np.asarray(arrays["diagnostic_support_view_slot_log_ratios"]).shape
        != (row_count, len(names), len(slots))
        or np.asarray(arrays["diagnostic_candidate_log_ratios"]).shape
        != (row_count, len(names), 128, 20)
        or np.asarray(arrays["diagnostic_group_log_ratios"]).shape
        != (row_count, len(names), 128)
        or np.asarray(arrays["diagnostic_projected_xy"]).shape
        != (row_count, len(tracks), 2)
        or np.asarray(arrays["diagnostic_projection_valid"]).shape != (row_count, len(tracks))
    ):
        raise ValueError(f"{path}: diagnostic term shapes or immutable row identity are invalid")
    for name in (
        "diagnostic_support_view_slot_log_ratios",
        "diagnostic_candidate_log_ratios",
        "diagnostic_group_log_ratios",
        "diagnostic_projected_xy",
    ):
        if not np.isfinite(np.asarray(arrays[name], dtype=np.float64)).all():
            raise ValueError(f"{path}: diagnostic field {name} is non-finite")
    return arrays, metadata


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: tail manifest is malformed") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("target_only") is not True
        or manifest.get("must_not_be_used_for_inference_or_formal_p1_audit") is not True
        or not isinstance(manifest.get("tail_pairs_TARGET_ONLY"), list)
    ):
        raise ValueError("tail manifest is incompatible")
    return manifest


def _model_size(metadata: Mapping[str, object]) -> tuple[float, float]:
    contract = metadata.get("coordinate_contract")
    query = contract.get("query") if isinstance(contract, Mapping) else None
    size = query.get("model_image_size") if isinstance(query, Mapping) else None
    if (
        not isinstance(size, list)
        or len(size) != 2
        or float(size[0]) <= 1.0
        or float(size[1]) <= 1.0
    ):
        raise ValueError("diagnostic score lacks a valid query model size")
    return float(size[0]), float(size[1])


def _spatial_blocks(
    *,
    references: np.ndarray,
    active: np.ndarray,
    selected_scores: np.ndarray,
    oracle_scores: np.ndarray,
    width: float,
    height: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for y_block in range(2):
        for x_block in range(2):
            mask = active & (references[:, 0] >= x_block * width / 2.0) & (
                references[:, 0] < (x_block + 1) * width / 2.0
            ) & (references[:, 1] >= y_block * height / 2.0) & (
                references[:, 1] < (y_block + 1) * height / 2.0
            )
            if not np.any(mask):
                rows.append(
                    {
                        "block": f"{x_block},{y_block}",
                        "active_group_count": 0,
                        "selected_mean_log_ratio": None,
                        "oracle_mean_log_ratio": None,
                        "oracle_minus_selected_mean": None,
                    }
                )
                continue
            selected_mean = float(np.mean(selected_scores[mask]))
            oracle_mean = float(np.mean(oracle_scores[mask]))
            rows.append(
                {
                    "block": f"{x_block},{y_block}",
                    "active_group_count": int(np.count_nonzero(mask)),
                    "selected_mean_log_ratio": selected_mean,
                    "oracle_mean_log_ratio": oracle_mean,
                    "oracle_minus_selected_mean": oracle_mean - selected_mean,
                }
            )
    return rows


def _candidate_summary(
    *,
    group: int,
    candidate: int,
    arrays: Mapping[str, np.ndarray],
    selected_row: int,
    oracle_row: int,
    profile_index: int,
) -> dict[str, object] | None:
    track_indices = np.asarray(arrays["candidate_group_track_indices"], dtype=np.int64)
    track_index = int(track_indices[group, candidate])
    if track_index < 0:
        return None
    track_ids = np.asarray(arrays["canonical_track_ids"], dtype=np.int64)
    slots = np.asarray(arrays["candidate_group_support_slot_indices"], dtype=np.int64)
    support_ids = np.asarray(arrays["candidate_group_support_image_ids"]).astype(str)
    view_probability = np.asarray(
        arrays["candidate_group_support_view_probabilities"], dtype=np.float64
    )
    slot_ids = np.asarray(arrays["support_slot_image_ids"]).astype(str)
    slot_track_indices = np.asarray(arrays["support_slot_track_indices"], dtype=np.int64)
    mode_offsets = np.asarray(arrays["mode_offsets"], dtype=np.int64)
    mode_xy = np.asarray(arrays["mode_query_xy"], dtype=np.float32)
    reliability = np.asarray(arrays["support_reliabilities"], dtype=np.float64)
    support_log = np.asarray(
        arrays["diagnostic_support_view_slot_log_ratios"], dtype=np.float64
    )
    candidate_log = np.asarray(arrays["diagnostic_candidate_log_ratios"], dtype=np.float64)
    projected = np.asarray(arrays["diagnostic_projected_xy"], dtype=np.float64)
    projected_valid = np.asarray(arrays["diagnostic_projection_valid"], dtype=bool)
    views: list[dict[str, object]] = []
    for view, slot in enumerate(slots[group, candidate].tolist()):
        if int(slot) < 0:
            views.append(
                {
                    "view_index": int(view),
                    "available": False,
                    "support_image_id": str(support_ids[group, candidate, view]),
                    "fixed_probability": float(view_probability[group, candidate, view]),
                    "neutral_missing_slot": True,
                }
            )
            continue
        start, stop = int(mode_offsets[slot]), int(mode_offsets[slot + 1])
        views.append(
            {
                "view_index": int(view),
                "available": True,
                "support_slot": int(slot),
                "support_image_id": str(slot_ids[slot]),
                "fixed_probability": float(view_probability[group, candidate, view]),
                "support_log_ratio_selected": float(support_log[selected_row, profile_index, slot]),
                "support_log_ratio_oracle": float(support_log[oracle_row, profile_index, slot]),
                "support_reliability": float(reliability[slot]),
                "mode_count": int(stop - start),
                "mode_query_xy": mode_xy[start:stop].astype(float).tolist(),
            }
        )
    return {
        "candidate_index": int(candidate),
        "track_id": int(track_ids[track_index]),
        "candidate_identity_probability": float(
            np.asarray(arrays["candidate_group_identity_probabilities"], dtype=np.float64)[
                group, candidate
            ]
        ),
        "candidate_log_ratio_selected": float(candidate_log[selected_row, profile_index, group, candidate]),
        "candidate_log_ratio_oracle": float(candidate_log[oracle_row, profile_index, group, candidate]),
        "candidate_ratio_oracle_minus_selected": float(
            candidate_log[oracle_row, profile_index, group, candidate]
            - candidate_log[selected_row, profile_index, group, candidate]
        ),
        "projected_xy_selected": projected[selected_row, track_index].astype(float).tolist(),
        "projected_xy_oracle": projected[oracle_row, track_index].astype(float).tolist(),
        "projection_valid_selected": bool(projected_valid[selected_row, track_index]),
        "projection_valid_oracle": bool(projected_valid[oracle_row, track_index]),
        "support_views": views,
        "support_slot_track_consistent": bool(
            all(
                int(slot) < 0 or int(slot_track_indices[int(slot)]) == track_index
                for slot in slots[group, candidate].tolist()
            )
        ),
    }


def _top_group_rows(
    *,
    arrays: Mapping[str, np.ndarray],
    selected_row: int,
    oracle_row: int,
    profile_index: int,
    top_groups: int,
    top_candidates_per_group: int,
    direction: str,
) -> list[dict[str, object]]:
    group_log = np.asarray(arrays["diagnostic_group_log_ratios"], dtype=np.float64)
    active = np.asarray(arrays["candidate_group_active_mask"], dtype=bool)
    delta = group_log[oracle_row, profile_index] - group_log[selected_row, profile_index]
    indices = np.flatnonzero(active)
    if direction == "wrong_pose":
        order = indices[np.argsort(delta[indices], kind="stable")]
    elif direction == "oracle":
        order = indices[np.argsort(-delta[indices], kind="stable")]
    else:
        raise ValueError("tail group direction is invalid")
    candidate_log = np.asarray(arrays["diagnostic_candidate_log_ratios"], dtype=np.float64)
    identity = np.asarray(arrays["candidate_group_identity_probabilities"], dtype=np.float64)
    null = np.asarray(arrays["candidate_group_null_probabilities"], dtype=np.float64)
    references = np.asarray(arrays["candidate_group_reference_xy"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for group in order[: int(top_groups)].tolist():
        weighted = identity[group] * np.exp(
            np.clip(candidate_log[selected_row, profile_index, group], -30.0, 30.0)
        )
        candidates = np.argsort(-weighted, kind="stable")[: int(top_candidates_per_group)]
        candidate_rows = [
            item
            for candidate in candidates.tolist()
            if (
                item := _candidate_summary(
                    group=int(group),
                    candidate=int(candidate),
                    arrays=arrays,
                    selected_row=selected_row,
                    oracle_row=oracle_row,
                    profile_index=profile_index,
                )
            )
            is not None
        ]
        rows.append(
            {
                "group_index": int(group),
                "reference_xy": references[group].astype(float).tolist(),
                "null_probability_fixed": float(null[group]),
                "selected_group_log_ratio": float(group_log[selected_row, profile_index, group]),
                "oracle_group_log_ratio": float(group_log[oracle_row, profile_index, group]),
                "oracle_minus_selected_group_log_ratio": float(delta[group]),
                "top_candidates_by_selected_fixed_prior_times_ratio": candidate_rows,
            }
        )
    return rows


def _correlated_evidence_reuse(
    *,
    arrays: Mapping[str, np.ndarray],
    selected_row: int,
    oracle_row: int,
    profile_index: int,
    top_items: int,
) -> dict[str, list[dict[str, object]]]:
    """Expose repeated latent evidence that a mean cannot treat as independent."""

    active = np.asarray(arrays["candidate_group_active_mask"], dtype=bool)
    track_indices = np.asarray(arrays["candidate_group_track_indices"], dtype=np.int64)
    track_ids = np.asarray(arrays["canonical_track_ids"], dtype=np.int64)
    identity = np.asarray(arrays["candidate_group_identity_probabilities"], dtype=np.float64)
    view = np.asarray(
        arrays["candidate_group_support_view_probabilities"], dtype=np.float64
    )
    slots = np.asarray(arrays["candidate_group_support_slot_indices"], dtype=np.int64)
    slot_ids = np.asarray(arrays["support_slot_image_ids"]).astype(str)
    candidate_log = np.asarray(arrays["diagnostic_candidate_log_ratios"], dtype=np.float64)
    support_log = np.asarray(
        arrays["diagnostic_support_view_slot_log_ratios"], dtype=np.float64
    )
    track_rows: dict[int, dict[str, object]] = {}
    slot_rows: dict[int, dict[str, object]] = {}
    for group in np.flatnonzero(active).tolist():
        for candidate, track_index in enumerate(track_indices[group].tolist()):
            prior = float(identity[group, candidate])
            if int(track_index) < 0 or prior <= 0.0:
                continue
            selected_ratio = float(
                np.exp(np.clip(candidate_log[selected_row, profile_index, group, candidate], -30.0, 30.0))
            )
            oracle_ratio = float(
                np.exp(np.clip(candidate_log[oracle_row, profile_index, group, candidate], -30.0, 30.0))
            )
            wrong_excess = prior * (selected_ratio - oracle_ratio)
            row = track_rows.setdefault(
                int(track_index),
                {
                    "track_id": int(track_ids[int(track_index)]),
                    "candidate_group_count": 0,
                    "wrong_favoring_group_count": 0,
                    "sum_selected_minus_oracle_prior_weighted_ratio": 0.0,
                    "max_selected_minus_oracle_prior_weighted_ratio": -np.inf,
                    "group_indices": [],
                },
            )
            row["candidate_group_count"] = int(row["candidate_group_count"]) + 1
            row["sum_selected_minus_oracle_prior_weighted_ratio"] = float(
                row["sum_selected_minus_oracle_prior_weighted_ratio"]
            ) + wrong_excess
            row["max_selected_minus_oracle_prior_weighted_ratio"] = max(
                float(row["max_selected_minus_oracle_prior_weighted_ratio"]), wrong_excess
            )
            if wrong_excess > 0.0:
                row["wrong_favoring_group_count"] = int(row["wrong_favoring_group_count"]) + 1
                row["group_indices"].append(int(group))
            for view_index, slot in enumerate(slots[group, candidate].tolist()):
                if int(slot) < 0:
                    continue
                view_prior = prior * float(view[group, candidate, view_index])
                selected_view_ratio = float(
                    np.exp(
                        np.clip(support_log[selected_row, profile_index, int(slot)], -30.0, 30.0)
                    )
                )
                oracle_view_ratio = float(
                    np.exp(
                        np.clip(support_log[oracle_row, profile_index, int(slot)], -30.0, 30.0)
                    )
                )
                view_excess = view_prior * (selected_view_ratio - oracle_view_ratio)
                slot_row = slot_rows.setdefault(
                    int(slot),
                    {
                        "support_slot": int(slot),
                        "support_image_id": str(slot_ids[int(slot)]),
                        "candidate_group_count": 0,
                        "wrong_favoring_group_count": 0,
                        "sum_selected_minus_oracle_prior_weighted_ratio": 0.0,
                        "max_selected_minus_oracle_prior_weighted_ratio": -np.inf,
                        "group_indices": [],
                    },
                )
                slot_row["candidate_group_count"] = int(slot_row["candidate_group_count"]) + 1
                slot_row["sum_selected_minus_oracle_prior_weighted_ratio"] = float(
                    slot_row["sum_selected_minus_oracle_prior_weighted_ratio"]
                ) + view_excess
                slot_row["max_selected_minus_oracle_prior_weighted_ratio"] = max(
                    float(slot_row["max_selected_minus_oracle_prior_weighted_ratio"]), view_excess
                )
                if view_excess > 0.0:
                    slot_row["wrong_favoring_group_count"] = int(
                        slot_row["wrong_favoring_group_count"]
                    ) + 1
                    slot_row["group_indices"].append(int(group))

    def ordered(values: Mapping[int, dict[str, object]]) -> list[dict[str, object]]:
        rows = list(values.values())
        for row in rows:
            row["group_indices"] = sorted(set(int(value) for value in row["group_indices"]))
        return sorted(
            rows,
            key=lambda row: (
                -float(row["sum_selected_minus_oracle_prior_weighted_ratio"]),
                -int(row["candidate_group_count"]),
                str(row.get("track_id", row.get("support_slot"))),
            ),
        )[: int(top_items)]

    return {
        "tracks_by_wrong_selected_excess": ordered(track_rows),
        "support_views_by_wrong_selected_excess": ordered(slot_rows),
    }


def audit_tail_pairs(
    *,
    diagnostic_score_artifacts: Sequence[Path],
    tail_manifest: Mapping[str, object],
    profile_name: str,
    top_groups: int,
    top_candidates_per_group: int,
) -> dict[str, object]:
    if int(top_groups) <= 0 or int(top_candidates_per_group) <= 0 or not str(profile_name):
        raise ValueError("tail attribution configuration is invalid")
    expected_pairs = {
        str(item["query_id"]): item
        for item in tail_manifest["tail_pairs_TARGET_ONLY"]
        if isinstance(item, Mapping)
    }
    if not expected_pairs:
        raise ValueError("tail manifest has no selected/oracle pairs")
    reports: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    for path in diagnostic_score_artifacts:
        arrays, metadata = _load_diagnostic_score(Path(path))
        query_ids = np.unique(np.asarray(arrays["query_ids"]).astype(str))
        if len(query_ids) != 1:
            raise ValueError(f"{path}: diagnostic score must own exactly one query")
        query_id = str(query_ids[0])
        manifest_row = expected_pairs.get(query_id)
        if manifest_row is None or query_id in seen_queries:
            raise ValueError(f"{path}: diagnostic query is absent or duplicated in tail manifest")
        manifest_inputs = manifest_row.get("source_score_inputs_target_free")
        if not isinstance(manifest_inputs, Mapping):
            raise ValueError("tail manifest lacks target-free input lineage")
        expected_hashes = {
            str(name): str(item.get("sha256", ""))
            for name, item in manifest_inputs.items()
            if isinstance(item, Mapping)
        }
        if _input_hashes(metadata) != expected_hashes:
            raise ValueError(f"{path}: diagnostic score input lineage is stale")
        names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
        if str(profile_name) not in names.tolist():
            raise ValueError(f"{path}: requested profile is absent")
        profile_index = int(np.flatnonzero(names == str(profile_name))[0])
        hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)
        selected_hypothesis = int(manifest_row["selected_hypothesis_index_TARGET_ONLY"])
        oracle_hypothesis = int(manifest_row["oracle_hypothesis_index_TARGET_ONLY"])
        rows = {int(hypothesis): row for row, hypothesis in enumerate(hypotheses.tolist())}
        if selected_hypothesis not in rows or oracle_hypothesis not in rows:
            raise ValueError(f"{path}: diagnostic score does not cover selected/oracle hypotheses")
        selected_row, oracle_row = rows[selected_hypothesis], rows[oracle_hypothesis]
        references = np.asarray(arrays["candidate_group_reference_xy"], dtype=np.float64)
        active = np.asarray(arrays["candidate_group_active_mask"], dtype=bool)
        group_log = np.asarray(arrays["diagnostic_group_log_ratios"], dtype=np.float64)
        width, height = _model_size(metadata)
        reports.append(
            {
                "query_id": query_id,
                "split_name": str(np.asarray(arrays["split_names"]).astype(str)[0]),
                "evaluation_label": str(
                    np.asarray(arrays["evaluation_labels"]).astype(str)[0]
                ),
                "profile_name": str(profile_name),
                "selected_oracle_target_only": {
                    key: manifest_row[key]
                    for key in (
                        "selected_hypothesis_index_TARGET_ONLY",
                        "selected_translation_error_m_TARGET_ONLY",
                        "selected_rotation_error_deg_TARGET_ONLY",
                        "selected_score_target_free",
                        "oracle_hypothesis_index_TARGET_ONLY",
                        "oracle_translation_error_m_TARGET_ONLY",
                        "oracle_rotation_error_deg_TARGET_ONLY",
                        "oracle_score_target_free",
                        "oracle_score_rank_TARGET_ONLY",
                    )
                },
                "fixed_group_prior_summary": {
                    "active_group_count": int(np.count_nonzero(active)),
                    "mean_null_probability": float(
                        np.mean(
                            np.asarray(
                                arrays["candidate_group_null_probabilities"], dtype=np.float64
                            )[active]
                        )
                    ),
                    "median_null_probability": float(
                        np.median(
                            np.asarray(
                                arrays["candidate_group_null_probabilities"], dtype=np.float64
                            )[active]
                        )
                    ),
                    "note": "null and identity priors are frozen across the two poses",
                },
                "projection_visibility": {
                    "selected_track_fraction": float(
                        np.mean(
                            np.asarray(arrays["diagnostic_projection_valid"], dtype=bool)[
                                selected_row
                            ]
                        )
                    ),
                    "oracle_track_fraction": float(
                        np.mean(
                            np.asarray(arrays["diagnostic_projection_valid"], dtype=bool)[
                                oracle_row
                            ]
                        )
                    ),
                },
                "spatial_blocks": _spatial_blocks(
                    references=references,
                    active=active,
                    selected_scores=group_log[selected_row, profile_index],
                    oracle_scores=group_log[oracle_row, profile_index],
                    width=width,
                    height=height,
                ),
                "groups_favoring_wrong_selected_pose": _top_group_rows(
                    arrays=arrays,
                    selected_row=selected_row,
                    oracle_row=oracle_row,
                    profile_index=profile_index,
                    top_groups=int(top_groups),
                    top_candidates_per_group=int(top_candidates_per_group),
                    direction="wrong_pose",
                ),
                "groups_favoring_oracle_pose": _top_group_rows(
                    arrays=arrays,
                    selected_row=selected_row,
                    oracle_row=oracle_row,
                    profile_index=profile_index,
                    top_groups=int(top_groups),
                    top_candidates_per_group=int(top_candidates_per_group),
                    direction="oracle",
                ),
                "correlated_evidence_reuse": _correlated_evidence_reuse(
                    arrays=arrays,
                    selected_row=selected_row,
                    oracle_row=oracle_row,
                    profile_index=profile_index,
                    top_items=max(int(top_groups), int(top_candidates_per_group)),
                ),
                "diagnostic_score_artifact": {
                    "path": str(path),
                    "sha256": file_sha256_short(Path(path)),
                },
            }
        )
        seen_queries.add(query_id)
    missing = sorted(set(expected_pairs).difference(seen_queries))
    if missing:
        raise ValueError("tail manifest pairs have no diagnostic score: " + ", ".join(missing))
    return {
        "format": REPORT_FORMAT,
        "stage": "target_only_frozen_lifted_loftr_candidate_view_tail_attribution",
        "target_only": True,
        "must_not_be_used_for_inference_or_pnp": True,
        "score_terms_are_target_free_but_pair_selection_is_target_only": True,
        "profile_name": str(profile_name),
        "tail_manifest": {
            "path": str(tail_manifest.get("_source_path", "")),
            "sha256": str(tail_manifest.get("_source_sha256", "")),
        },
        "tail_reports_TARGET_ONLY": reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output = output_dir / "tail_attribution.json"
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite tail attribution: {output}")
    manifest_path = Path(args.tail_manifest)
    manifest = _load_manifest(manifest_path)
    manifest["_source_path"] = str(manifest_path)
    manifest["_source_sha256"] = file_sha256_short(manifest_path)
    report = audit_tail_pairs(
        diagnostic_score_artifacts=_paths(args.diagnostic_score_artifacts),
        tail_manifest=manifest,
        profile_name=str(args.profile_name),
        top_groups=int(args.top_groups),
        top_candidates_per_group=int(args.top_candidates_per_group),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "tail_report_count": len(report["tail_reports_TARGET_ONLY"]),
                "profile_name": report["profile_name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
