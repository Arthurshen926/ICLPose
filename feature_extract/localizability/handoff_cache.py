"""Export POFD-FS handoff selections as standard retrieval init caches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from data.radio_loc_retrieval_dataset import (
    OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS,
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)
from feature_extract.localizability.score_calibrator import load_candidate_table_jsonl
from feature_extract.localizability.solver_handoff import select_handoff_rows


_OPTIONAL_DEFAULTS = {
    "retrieval_original_scores_candidates": 0.0,
    "retrieval_pnp_success_candidates": 0.0,
    "retrieval_pnp_num_inliers_candidates": 0.0,
    "retrieval_pnp_num_matches_candidates": 0.0,
    "retrieval_pnp_reproj_rmse_candidates": float("inf"),
    "retrieval_pnp_reproj_median_candidates": float("inf"),
    "retrieval_pnp_inlier_ratio_candidates": 0.0,
    "retrieval_pnp_inlier_conf_mean_candidates": 0.0,
}


def _entry_keys(entry: dict) -> tuple[str, ...]:
    image_name = str(entry["query_image_name"])
    image_stem = str(entry["query_image_stem"])
    plain_stem = Path(image_name).stem
    no_suffix = Path(image_name).with_suffix("").as_posix()
    return (image_name, image_stem, plain_stem, no_suffix)


def _selected_rows_by_sample(rows: Sequence[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        out[str(row["sample_name"])] = row
    return out


def _find_selected_row(entry: dict, selected_by_sample: dict[str, dict]) -> dict:
    for key in _entry_keys(entry):
        if key in selected_by_sample:
            return selected_by_sample[key]
    raise ValueError(
        "missing selected candidate row for cache entry "
        f"{entry['query_image_name']} / {entry['query_image_stem']}"
    )


def _candidate_float(entry: dict, row: dict, key: str, candidate_idx: int) -> float:
    if key in entry:
        values = np.asarray(entry[key], dtype=np.float32).reshape(-1)
        if candidate_idx >= values.shape[0]:
            raise ValueError(f"{key} length {values.shape[0]} does not contain candidate {candidate_idx}")
        return float(values[candidate_idx])
    if key in row:
        return float(row[key])
    return float(_OPTIONAL_DEFAULTS[key])


def export_selected_init_cache(
    *,
    source_cache_path: str | Path,
    candidate_table_path: str | Path,
    save_path: str | Path,
    topk: int = 1,
    selection_mode: str = "pofd_score",
    source_name: str | None = None,
) -> tuple[list[dict], dict]:
    """Write a one-candidate init cache selected by a POFD-FS handoff policy.

    The input cache supplies the actual candidate poses and retrieval metadata.
    The candidate table supplies POFD-FS scores plus optional solver-quality
    fields used to choose one candidate from POFD topK.
    """
    entries, source_stats = load_retrieval_init_entries(str(source_cache_path))
    rows = load_candidate_table_jsonl(candidate_table_path)
    selected_rows = select_handoff_rows(rows, topk=topk, selection_mode=selection_mode)
    if not selected_rows:
        raise ValueError(f"no selected rows from candidate table: {candidate_table_path}")
    selected_by_sample = _selected_rows_by_sample(selected_rows)
    resolved_source = str(source_name or f"pofd_fs_top{int(topk)}_{selection_mode}")

    exported_entries: list[dict] = []
    selected_indices: list[int] = []
    selected_costs: list[float] = []
    selected_scores: list[float] = []
    for entry in entries:
        row = _find_selected_row(entry, selected_by_sample)
        candidate_idx = int(row["candidate_idx"])
        poses = np.asarray(entry["pose_init_candidates"], dtype=np.float32)
        if candidate_idx < 0 or candidate_idx >= poses.shape[0]:
            raise ValueError(
                f"candidate_idx {candidate_idx} out of range for {entry['query_image_name']} "
                f"with {poses.shape[0]} candidates"
            )
        valid_mask = np.asarray(entry["candidate_valid_mask"], dtype=bool).reshape(-1)
        frame_ids = np.asarray(entry["retrieval_frame_ids_candidates"], dtype=np.int64).reshape(-1)
        names = list(entry["retrieval_image_names_candidates"])
        scores = np.asarray(entry["retrieval_scores_candidates"], dtype=np.float32).reshape(-1)
        if candidate_idx >= len(valid_mask) or candidate_idx >= len(frame_ids) or candidate_idx >= len(names):
            raise ValueError(f"candidate metadata missing index {candidate_idx} for {entry['query_image_name']}")
        selected_pose = poses[candidate_idx].astype(np.float32)
        selected_valid = bool(valid_mask[candidate_idx]) and bool(row.get("valid", True))
        selected_score = float(scores[candidate_idx]) if candidate_idx < len(scores) else float(row.get("score", 0.0))

        exported = {
            "query_img_id": int(entry["query_img_id"]),
            "query_image_name": str(entry["query_image_name"]),
            "query_image_stem": str(entry["query_image_stem"]),
            "pose_init": selected_pose.copy(),
            "init_source": resolved_source,
            "retrieval_frame_id": int(frame_ids[candidate_idx]),
            "retrieval_image_name": str(names[candidate_idx]),
            "retrieval_score": selected_score,
            "pose_init_candidates": selected_pose[None].astype(np.float32),
            "candidate_valid_mask": np.asarray([selected_valid], dtype=bool),
            "retrieval_frame_ids_candidates": np.asarray([int(frame_ids[candidate_idx])], dtype=np.int64),
            "retrieval_image_names_candidates": np.asarray([str(names[candidate_idx])]),
            "retrieval_scores_candidates": np.asarray([selected_score], dtype=np.float32),
        }
        for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
            exported[key] = np.asarray([_candidate_float(entry, row, key, candidate_idx)], dtype=np.float32)
        exported_entries.append(exported)
        selected_indices.append(candidate_idx)
        selected_costs.append(float(row.get("pose_cost_m", float("nan"))))
        selected_scores.append(float(row.get("score", float("nan"))))

    stats = dict(source_stats or {})
    stats.update(
        {
            "selection_source": "pofd_fs_handoff_cache",
            "source_cache": str(source_cache_path),
            "candidate_table": str(candidate_table_path),
            "selection_mode": str(selection_mode),
            "topk": int(topk),
            "source_name": resolved_source,
            "num_entries": len(exported_entries),
            "selected_candidate_indices": selected_indices,
            "selected_pose_cost_mean_m": float(np.nanmean(np.asarray(selected_costs, dtype=np.float64))),
            "selected_score_mean": float(np.nanmean(np.asarray(selected_scores, dtype=np.float64))),
        }
    )
    save_retrieval_init_entries(exported_entries, stats, str(save_path))
    return exported_entries, stats
