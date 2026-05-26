"""Adapters from normalized candidate tables into VFM hypothesis records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost


METADATA_FIELDS = {
    "candidate_source",
    "generator",
    "identity_delta_m",
    "in_basin",
    "is_oracle",
    "pnp_inlier_conf_mean",
    "pnp_inlier_ratio",
    "pnp_inliers",
    "pnp_num_matches",
    "pnp_reproj_median",
    "pnp_reproj_rmse",
    "pnp_score",
    "pose_cost_m",
    "reprojection_median",
    "retrieval_rank",
    "retrieval_score",
    "rot_cost_weight",
    "scene",
    "score_rank",
    "source_cache",
}


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value}")


def _as_python(value: object) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        return _as_python(value.item())
    if isinstance(value, np.generic):
        return _as_python(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _array_value(data: Mapping[str, np.ndarray], key: str, i: int, j: int | None = None) -> object | None:
    if key not in data:
        return None
    array = data[key]
    value = array[i] if j is None else array[i, j]
    return _as_python(value)


def _string_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(_as_python(value))
    if text == "":
        return None
    return text


def _pose_cost_from_record(record: Mapping[str, object]) -> PoseCost | None:
    if record.get("pose_error") is not None:
        data = dict(record["pose_error"])
        return PoseCost(
            translation_m=float(data["translation_m"]),
            rotation_deg=float(data["rotation_deg"]),
        )
    translation = _optional_float(record.get("translation_m"))
    rotation = _optional_float(record.get("rotation_deg"))
    if translation is None and rotation is None:
        return None
    if translation is None or rotation is None:
        raise ValueError("translation_m and rotation_deg must be provided together")
    return PoseCost(translation_m=translation, rotation_deg=rotation)


def _metadata_from_record(record: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        str(key): _as_python(value)
        for key, value in dict(record.get("metadata", {})).items()
        if value not in (None, "")
    }
    for field in METADATA_FIELDS:
        if field not in record or record[field] in (None, ""):
            continue
        if field in {"retrieval_rank", "pnp_inliers", "pnp_num_matches", "score_rank"}:
            metadata[field] = _optional_int(record[field])
        elif field in {
            "identity_delta_m",
            "pnp_inlier_conf_mean",
            "pnp_inlier_ratio",
            "pnp_reproj_median",
            "pnp_reproj_rmse",
            "pnp_score",
            "pose_cost_m",
            "reprojection_median",
            "retrieval_score",
            "rot_cost_weight",
        }:
            metadata[field] = _optional_float(record[field])
        elif field in {"in_basin", "is_oracle"}:
            metadata[field] = _optional_bool(record[field])
        else:
            metadata[field] = str(record[field])
    return metadata


def candidate_from_normalized_record(record: Mapping[str, object]) -> CandidateHypothesis:
    return CandidateHypothesis(
        query_id=None if record.get("query_id") in (None, "") else str(record["query_id"]),
        candidate_id=str(record["candidate_id"]),
        candidate_type=str(record["candidate_type"]),
        pose_error=_pose_cost_from_record(record),
        prior_score=_optional_float(record.get("prior_score")),
        pose=record.get("pose"),
        reference_image=None if record.get("reference_image") in (None, "") else str(record["reference_image"]),
        submap_id=None if record.get("submap_id") in (None, "") else str(record["submap_id"]),
        solver_success=_optional_bool(record.get("solver_success")),
        hard_case_type=None
        if record.get("hard_case_type") in (None, "")
        else str(record["hard_case_type"]),
        metadata=_metadata_from_record(record),
    )


def load_candidate_records(path: Path) -> list[CandidateHypothesis]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError("candidate JSON must contain a list")
        return [candidate_from_normalized_record(item) for item in payload]
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return [candidate_from_normalized_record(row) for row in reader]
    if suffix == ".jsonl":
        return load_score_table_jsonl_records(path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
        if {"pose_init_candidates", "candidate_valid_mask"} <= keys:
            return load_pose_init_npz_records(path)
        if {"candidates", "trans_err_m", "rot_err_deg", "valid_mask"} <= keys:
            return load_reference_pose_bank_npz_records(path)
        raise ValueError(f"unsupported candidate NPZ fields in {path}")
    raise ValueError(f"unsupported candidate table format: {path.suffix}")


def load_score_table_jsonl_records(path: Path) -> list[CandidateHypothesis]:
    """Load labeled candidates from a generic offline score-table JSONL."""

    records: list[CandidateHypothesis] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "header":
                continue
            if row.get("record_type") == "candidate":
                from feature_extract.vfm.hypothesis_io import candidate_from_dict

                records.append(candidate_from_dict(row))
                continue
            query_id = row.get("sample_name") or row.get("query_id")
            candidate_idx = int(row.get("candidate_idx", row.get("row", len(records))))
            record = {
                "query_id": query_id,
                "candidate_id": f"{Path(str(query_id)).stem}:score:{candidate_idx:03d}",
                "candidate_type": "score_table_candidate",
                "translation_m": row.get("trans_err_m", row.get("translation_m")),
                "rotation_deg": row.get("rot_err_deg", row.get("rotation_deg")),
                "prior_score": row.get("score"),
                "score_rank": row.get("score_rank"),
                "pose_cost_m": row.get("pose_cost_m"),
                "in_basin": row.get("in_basin"),
                "is_oracle": row.get("is_oracle"),
                "retrieval_rank": row.get(
                    "retrieval_rank",
                    None if row.get("candidate_idx") is None else int(row["candidate_idx"]) + 1,
                ),
                "identity_delta_m": row.get("delta_trans_m"),
                "pnp_inliers": row.get(
                    "retrieval_pnp_num_inliers",
                    row.get("retrieval_pnp_num_inliers_candidates"),
                ),
                "pnp_num_matches": row.get(
                    "retrieval_pnp_num_matches",
                    row.get("retrieval_pnp_num_matches_candidates"),
                ),
                "pnp_reproj_median": row.get(
                    "retrieval_pnp_reproj_median",
                    row.get("retrieval_pnp_reproj_median_candidates"),
                ),
                "pnp_reproj_rmse": row.get(
                    "retrieval_pnp_reproj_rmse",
                    row.get("retrieval_pnp_reproj_rmse_candidates"),
                ),
                "pnp_inlier_ratio": row.get(
                    "retrieval_pnp_inlier_ratio",
                    row.get("retrieval_pnp_inlier_ratio_candidates"),
                ),
                "source_cache": str(path),
            }
            records.append(candidate_from_normalized_record(record))
    if not records:
        raise ValueError(f"no candidate rows found in {path}")
    return records


def _pose_init_candidate_type(init_source: str | None) -> str:
    if init_source is not None and "hloc" in init_source.lower():
        return "hloc_pose"
    return "real_retrieval"


def _candidate_id(query_id: str, prefix: str, candidate_index: int) -> str:
    query_stem = Path(query_id).stem if query_id else "query"
    return f"{query_stem}:{prefix}:{candidate_index:03d}"


def load_pose_init_npz_records(path: Path) -> list[CandidateHypothesis]:
    """Load real-retrieval pose-init candidates from an NPZ export."""

    records: list[CandidateHypothesis] = []
    with np.load(path, allow_pickle=True) as data:
        poses = data["pose_init_candidates"]
        valid = data["candidate_valid_mask"].astype(bool)
        query_count, candidate_count = valid.shape
        for i in range(query_count):
            query_id = _string_value(
                _array_value(data, "query_image_names", i)
                or _array_value(data, "query_image_stems", i)
                or _array_value(data, "query_img_ids", i)
            )
            init_source = _string_value(_array_value(data, "init_sources", i))
            for j in range(candidate_count):
                if not bool(valid[i, j]):
                    continue
                retrieval_score = _array_value(data, "retrieval_scores_candidates", i, j)
                pnp_inliers = _array_value(data, "retrieval_pnp_num_inliers_candidates", i, j)
                pnp_matches = _array_value(data, "retrieval_pnp_num_matches_candidates", i, j)
                pnp_rmse = _array_value(data, "retrieval_pnp_reproj_rmse_candidates", i, j)
                pnp_median = _array_value(data, "retrieval_pnp_reproj_median_candidates", i, j)
                pnp_ratio = _array_value(data, "retrieval_pnp_inlier_ratio_candidates", i, j)
                pnp_conf = _array_value(data, "retrieval_pnp_inlier_conf_mean_candidates", i, j)
                pnp_success = _array_value(data, "retrieval_pnp_success_candidates", i, j)
                record = {
                    "query_id": query_id,
                    "candidate_id": _candidate_id(query_id or "", "retrieval", j),
                    "candidate_type": _pose_init_candidate_type(init_source),
                    "pose": poses[i, j].astype(float).tolist(),
                    "reference_image": _string_value(
                        _array_value(data, "retrieval_image_names_candidates", i, j)
                    ),
                    "prior_score": retrieval_score,
                    "retrieval_rank": j + 1,
                    "retrieval_score": retrieval_score,
                    "pnp_inliers": pnp_inliers,
                    "pnp_num_matches": pnp_matches,
                    "pnp_reproj_rmse": pnp_rmse,
                    "pnp_reproj_median": pnp_median,
                    "reprojection_median": pnp_median,
                    "pnp_inlier_ratio": pnp_ratio,
                    "pnp_inlier_conf_mean": pnp_conf,
                    "generator": init_source or "pose_init_npz",
                    "solver_success": pnp_success,
                    "source_cache": str(path),
                }
                records.append(candidate_from_normalized_record(record))
    return records


def load_reference_pose_bank_npz_records(path: Path) -> list[CandidateHypothesis]:
    """Load labeled reference-pose candidates from a localizability NPZ bank."""

    records: list[CandidateHypothesis] = []
    with np.load(path, allow_pickle=True) as data:
        poses = data["candidates"]
        valid = data["valid_mask"].astype(bool)
        query_count, candidate_count = valid.shape
        scene = _string_value(data["scene"]) if "scene" in data else None
        source = _string_value(data["candidate_source"]) if "candidate_source" in data else None
        rot_cost_weight = _as_python(data["rot_cost_weight"]) if "rot_cost_weight" in data else None
        for i in range(query_count):
            query_id = _string_value(_array_value(data, "sample_names", i))
            for j in range(candidate_count):
                if not bool(valid[i, j]):
                    continue
                pose_cost = _array_value(data, "pose_cost_m", i, j)
                record = {
                    "query_id": query_id,
                    "candidate_id": _candidate_id(query_id or "", "reference", j),
                    "candidate_type": "reference_pose",
                    "pose": poses[i, j].astype(float).tolist(),
                    "reference_image": _string_value(_array_value(data, "reference_names", i, j)),
                    "translation_m": _array_value(data, "trans_err_m", i, j),
                    "rotation_deg": _array_value(data, "rot_err_deg", i, j),
                    "retrieval_rank": j + 1,
                    "pose_cost_m": pose_cost,
                    "scene": scene,
                    "candidate_source": source,
                    "rot_cost_weight": rot_cost_weight,
                    "source_cache": str(path),
                }
                records.append(candidate_from_normalized_record(record))
    return records
