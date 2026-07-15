"""Strict target-free real-RGB candidate inference data path."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    coordinate_space_from_evidence,
    image_root_manifest,
    validate_sampling_dimensions,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _crop_cached_rgb_windows_grouped,
    _load_query_rgb,
)


INFERENCE_EVIDENCE_FORMAT = "candidate_rgb_inference_evidence_v1"
INFERENCE_ROWS_STAGE = "candidate_specific_real_rgb_inference_rows"

INFERENCE_EVIDENCE_ARRAYS = (
    "selected_rows",
    "query_ids",
    "query_xy",
    "split_names",
    "candidate_compact_columns",
    "candidate_source_columns",
    "candidate_roles",
    "candidate_valid",
    "candidate_track_ids",
    "candidate_prototype_ids",
    "candidate_bank_rows",
    "candidate_prior_probabilities",
    "candidate_score_ranks",
    "candidate_coarse_similarities",
    "candidate_support_view_probabilities",
    "source_set_dustbin_probability",
    "retained_candidate_probability_mass",
    "omitted_candidate_probability_mass",
    "unknown_probability",
)

INFERENCE_ROW_FIELDS = (
    "query_id",
    "support_image_id",
    "track_id",
    "support_track_id",
    "track_length",
    "support_x",
    "support_y",
    "render_x",
    "render_y",
    "center_x",
    "center_y",
    "support_reprojection_error",
    "support_frame_gap",
    "candidate_identity_key",
    "support_view_set_id",
    "candidate_measurement_rank",
    "candidate_score_rank",
    "candidate_role",
    "candidate_prototype_id",
    "candidate_bank_row",
    "candidate_assignment_probability",
    "candidate_retrieval_similarity",
    "source_query_row",
    "split",
    "support_view_rank",
    "support_view_probability",
    "supervision_source_row_index",
)


def _read_metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    return json.loads(str(np.asarray(payload["metadata_json"]).item()))


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INFERENCE_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def materialize_candidate_rgb_inference_inputs(
    *,
    candidate_evidence: Path,
    rows_by_split: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Strip every target field before the production RGB inference process."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    expected_splits = {"train", "validation", "test"}
    if set(rows_by_split) != expected_splits:
        raise ValueError("RGB inference materialization requires train/validation/test rows")

    source_path = Path(candidate_evidence)
    with np.load(source_path, allow_pickle=False) as payload:
        source_metadata = _read_metadata(payload, context="candidate evidence")
        missing = set(INFERENCE_EVIDENCE_ARRAYS) - set(payload.files)
        if missing:
            raise ValueError(f"candidate evidence lacks inference arrays: {sorted(missing)}")
        evidence_arrays = {
            key: np.asarray(payload[key]) for key in INFERENCE_EVIDENCE_ARRAYS
        }
    if source_metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("RGB inference input requires candidate evidence V3")
    if bool(source_metadata.get("ground_truth_used_for_selection")) or bool(
        source_metadata.get("pose_used_for_selection")
    ):
        raise ValueError("candidate evidence selection used pose or ground truth")
    if bool(source_metadata.get("image_retrieval")) or bool(
        source_metadata.get("submap")
    ) or bool(source_metadata.get("render")):
        raise ValueError("production RGB inference requires global real-image candidates")

    source_hash = file_sha256_short(source_path)
    metadata = {
        key: source_metadata[key]
        for key in (
            "candidate_probability_semantics",
            "candidate_score_key",
            "candidates_per_token",
            "colmap_cameras_sha256",
            "colmap_images_sha256",
            "colmap_model_dir",
            "colmap_points3d_sha256",
            "descriptor_space_id",
            "descriptor_space_manifest",
            "maplet_support_index_sha256",
            "projected_landmark_bank_sha256",
            "proposals_sha256",
            "score_artifact_sha256",
            "split_json_sha256",
            "support_geometry_index_sha256",
            "unknown_probability_semantics",
        )
        if key in source_metadata
    }
    metadata.update(
        {
            "format": INFERENCE_EVIDENCE_FORMAT,
            "format_version": 1,
            "source_candidate_evidence": str(source_path),
            "source_candidate_evidence_sha256": source_hash,
            "contains_ground_truth": False,
            "contains_pose_derived_selection": False,
            "ground_truth_loaded_by_inference": False,
            "allowed_arrays": list(INFERENCE_EVIDENCE_ARRAYS),
            "image_retrieval": False,
            "submap": False,
            "render": False,
        }
    )
    output.mkdir(parents=True)
    evidence_output = output / "candidate_rgb_inference_evidence_v1.npz"
    np.savez_compressed(
        evidence_output,
        **evidence_arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    row_outputs: dict[str, dict[str, object]] = {}
    for split in sorted(expected_splits):
        source_rows_path = Path(rows_by_split[split])
        source_summary_path = source_rows_path.with_suffix(".summary.json")
        if not source_summary_path.exists():
            raise ValueError(f"RGB source rows lack summary: {source_summary_path}")
        source_summary = json.loads(source_summary_path.read_text())
        if source_summary.get("stage") != "candidate_specific_real_rgb_measurement_rows":
            raise ValueError(f"unsupported RGB source row stage: {source_rows_path}")
        if str(source_summary.get("split")) != split:
            raise ValueError(f"RGB source row split mismatch: {source_rows_path}")
        outputs = dict(source_summary.get("outputs", {}))
        if outputs.get("rows_csv_sha256") != file_sha256_short(source_rows_path):
            raise ValueError(f"RGB source rows are stale: {source_rows_path}")
        protocol = dict(source_summary.get("protocol", {}))
        if bool(protocol.get("pose_used_for_candidate_or_support_selection")) or bool(
            protocol.get("pose_derived_features_exposed_to_rgb_scorer")
        ):
            raise ValueError("RGB row selection exposes pose-derived inference inputs")
        row_inputs = dict(source_summary.get("inputs", {}))
        if row_inputs.get("selection_artifact_sha256") != source_hash:
            raise ValueError("RGB source rows reference different candidate evidence")

        source_fields, source_rows = _read_csv(source_rows_path)
        missing_fields = set(INFERENCE_ROW_FIELDS) - {
            *source_fields,
            "supervision_source_row_index",
        }
        if missing_fields:
            raise ValueError(f"RGB source rows lack inference fields: {sorted(missing_fields)}")
        sanitized_rows: list[dict[str, object]] = []
        for source_index, row in enumerate(source_rows):
            if str(row.get("split", "")) != split:
                raise ValueError(f"RGB source row crosses split: {source_rows_path}")
            clean = {key: row.get(key, "") for key in INFERENCE_ROW_FIELDS}
            clean["supervision_source_row_index"] = int(source_index)
            sanitized_rows.append(clean)
        rows_output = output / f"{split}.csv"
        _write_csv(rows_output, sanitized_rows)
        rows_hash = file_sha256_short(rows_output)
        sanitized_summary = {
            "stage": INFERENCE_ROWS_STAGE,
            "split": split,
            "coordinate_space": source_summary.get("coordinate_space"),
            "protocol": {
                "contains_ground_truth": False,
                "contains_pose_derived_features": False,
                "ground_truth_loaded_by_inference": False,
                "query_source": "real_pair",
                "render": False,
                "image_retrieval": False,
                "submap": False,
                "same_image_support_forbidden": True,
                "support_view_aggregation": "none_rows_remain_multimodal",
            },
            "inputs": {
                "selection_artifact": str(source_path),
                "selection_artifact_sha256": source_hash,
                "inference_evidence": str(evidence_output),
                "inference_evidence_sha256": file_sha256_short(evidence_output),
                "supervision_rows": str(source_rows_path),
                "supervision_rows_sha256": file_sha256_short(source_rows_path),
            },
            "schema": {
                "allowed_fields": list(INFERENCE_ROW_FIELDS),
                "source_row_join_key": "supervision_source_row_index",
            },
            "outputs": {
                "rows_csv": str(rows_output),
                "rows_csv_sha256": rows_hash,
            },
            "output_rows": int(len(sanitized_rows)),
        }
        summary_output = rows_output.with_suffix(".summary.json")
        summary_output.write_text(
            json.dumps(sanitized_summary, indent=2, sort_keys=True) + "\n"
        )
        row_outputs[split] = {
            "rows_csv": str(rows_output),
            "rows_csv_sha256": rows_hash,
            "summary": str(summary_output),
            "row_count": int(len(sanitized_rows)),
        }

    summary = {
        "stage": "candidate_rgb_target_free_inference_input_materialization",
        "protocol": {
            "production_inference_consumes_only_sanitized_outputs": True,
            "target_columns_copied": False,
            "pose_derived_columns_copied": False,
            "allowed_evidence_arrays": list(INFERENCE_EVIDENCE_ARRAYS),
            "allowed_row_fields": list(INFERENCE_ROW_FIELDS),
        },
        "inputs": {
            "candidate_evidence": str(source_path),
            "candidate_evidence_sha256": source_hash,
        },
        "outputs": {
            "inference_evidence": str(evidence_output),
            "inference_evidence_sha256": file_sha256_short(evidence_output),
            "rows": row_outputs,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _float(
    row: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        if default is None:
            raise ValueError(f"RGB inference row {key} is empty")
        value = float(default)
    else:
        value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"RGB inference row {key} is not finite")
    return value


class CandidateRGBInferenceData:
    """Read only sanitized evidence and rows; fail closed on extra columns."""

    def __init__(
        self,
        *,
        inference_evidence: Path,
        rows_by_split: Mapping[str, Path],
        max_views: int,
    ) -> None:
        self.evidence_path = Path(inference_evidence)
        with np.load(self.evidence_path, allow_pickle=False) as payload:
            if set(payload.files) != {*INFERENCE_EVIDENCE_ARRAYS, "metadata_json"}:
                raise ValueError("RGB inference evidence contains unapproved arrays")
            self.metadata = _read_metadata(payload, context="RGB inference evidence")
            arrays = {key: np.asarray(payload[key]) for key in INFERENCE_EVIDENCE_ARRAYS}
        if self.metadata.get("format") != INFERENCE_EVIDENCE_FORMAT:
            raise ValueError("unsupported RGB inference evidence format")
        if bool(self.metadata.get("contains_ground_truth")) or bool(
            self.metadata.get("contains_pose_derived_selection")
        ):
            raise ValueError("RGB inference evidence is not target-free")
        self.selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
        self.query_ids = np.asarray(arrays["query_ids"]).astype(str)
        self.query_xy = np.asarray(arrays["query_xy"], dtype=np.float32)
        self.split_names = np.asarray(arrays["split_names"]).astype(str)
        self.candidate_valid = np.asarray(arrays["candidate_valid"], dtype=bool)
        self.track_ids = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        self.prototype_ids = np.asarray(
            arrays["candidate_prototype_ids"], dtype=np.int64
        )
        self.prior = np.asarray(
            arrays["candidate_prior_probabilities"], dtype=np.float32
        )
        self.unknown = np.asarray(arrays["unknown_probability"], dtype=np.float32)
        n = len(self.query_ids)
        if (
            self.selected_rows.shape != (n,)
            or self.query_xy.shape != (n, 2)
            or self.split_names.shape != (n,)
            or self.candidate_valid.shape != self.track_ids.shape
            or self.prototype_ids.shape != self.track_ids.shape
            or self.prior.shape != self.track_ids.shape
            or self.unknown.shape != (n,)
        ):
            raise ValueError("RGB inference evidence arrays have incompatible shapes")
        mass = np.sum(np.where(self.candidate_valid, self.prior, 0.0), axis=1)
        if not np.allclose(mass + self.unknown, 1.0, rtol=0.0, atol=2e-5):
            raise ValueError("RGB inference candidate probability mass is invalid")
        self.max_views = int(max_views)
        if self.max_views <= 0:
            raise ValueError("max_views must be positive")
        self.coordinate_space = coordinate_space_from_evidence(self.metadata)
        self.source_candidate_evidence_sha256 = str(
            self.metadata.get("source_candidate_evidence_sha256", "")
        )
        if not self.source_candidate_evidence_sha256:
            raise ValueError("RGB inference evidence lacks source candidate hash")

        key_to_index: dict[tuple[str, int], int] = {}
        for index, key in enumerate(zip(self.query_ids.tolist(), self.selected_rows.tolist())):
            normalized = (str(key[0]), int(key[1]))
            if normalized in key_to_index:
                raise ValueError("RGB inference evidence group key is not unique")
            key_to_index[normalized] = int(index)
        candidate_count = int(self.track_ids.shape[1])
        self.rows_by_group: dict[int, list[list[dict[str, str]]]] = {
            index: [[] for _ in range(candidate_count)] for index in range(n)
        }
        self.rows_paths = {key: Path(value) for key, value in rows_by_split.items()}
        if set(self.rows_paths) != {"train", "validation", "test"}:
            raise ValueError("RGB inference data requires train/validation/test rows")
        self.image_ids: set[str] = set()
        for split, path in self.rows_paths.items():
            summary_path = path.with_suffix(".summary.json")
            if not summary_path.exists():
                raise ValueError(f"RGB inference rows lack summary: {path}")
            summary = json.loads(summary_path.read_text())
            if summary.get("stage") != INFERENCE_ROWS_STAGE or str(
                summary.get("split")
            ) != split:
                raise ValueError(f"unsupported RGB inference row artifact: {path}")
            if dict(summary.get("outputs", {})).get(
                "rows_csv_sha256"
            ) != file_sha256_short(path):
                raise ValueError(f"RGB inference rows are stale: {path}")
            inputs = dict(summary.get("inputs", {}))
            if inputs.get("inference_evidence_sha256") != file_sha256_short(
                self.evidence_path
            ) or inputs.get(
                "selection_artifact_sha256"
            ) != self.source_candidate_evidence_sha256:
                raise ValueError("RGB inference rows reference different evidence")
            fields, rows = _read_csv(path)
            if fields != INFERENCE_ROW_FIELDS:
                raise ValueError("RGB inference rows contain unapproved columns")
            for row in rows:
                if str(row.get("split", "")) != split:
                    raise ValueError("RGB inference row crosses split")
                key = (str(row.get("query_id", "")), int(row["source_query_row"]))
                group = key_to_index.get(key)
                if group is None or self.split_names[group] != split:
                    raise ValueError("RGB inference row does not map to evidence")
                rank = int(row["candidate_measurement_rank"]) - 1
                if not 0 <= rank < candidate_count:
                    raise ValueError("RGB inference candidate rank is invalid")
                if int(row["track_id"]) != int(self.track_ids[group, rank]):
                    raise ValueError("RGB inference row track differs from evidence")
                if int(row["candidate_prototype_id"]) != int(
                    self.prototype_ids[group, rank]
                ):
                    raise ValueError("RGB inference row prototype differs from evidence")
                center = np.asarray(
                    [_float(row, "center_x"), _float(row, "center_y")],
                    dtype=np.float32,
                )
                if not np.allclose(center, self.query_xy[group], rtol=0.0, atol=1e-4):
                    raise ValueError("RGB inference row center differs from evidence")
                self.image_ids.update(
                    [str(row["query_id"]), str(row["support_image_id"])]
                )
                self.rows_by_group[group][rank].append(row)
        for candidates in self.rows_by_group.values():
            for rows in candidates:
                rows.sort(
                    key=lambda row: (
                        int(row["support_view_rank"]),
                        str(row["support_image_id"]),
                    )
                )
                del rows[self.max_views :]
        self.indices_by_split = {
            split: np.flatnonzero(self.split_names == split).astype(np.int64)
            for split in ("train", "validation", "test")
        }
        self.has_any_rgb = np.asarray(
            [any(self.rows_by_group[index]) for index in range(n)], dtype=bool
        )

    def runtime_contract(
        self, *, image_root: Path, image_width: int, image_height: int
    ) -> dict[str, Any]:
        validate_sampling_dimensions(
            self.coordinate_space,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        return {
            "version": 1,
            "coordinate_space": dict(self.coordinate_space),
            "image_source": image_root_manifest(
                Path(image_root), sorted(self.image_ids)
            ),
            "candidate_evidence_sha256": self.source_candidate_evidence_sha256,
            "availability_evidence_sha256": "runtime_not_required",
            "rows_sha256": {
                split: file_sha256_short(path)
                for split, path in sorted(self.rows_paths.items())
            },
        }


def prepare_candidate_rgb_inference_batch(
    data: CandidateRGBInferenceData,
    evidence_indices: Sequence[int],
    *,
    image_root: Path,
    image_width: int,
    image_height: int,
    image_cache: TensorImageLRUCache,
    image_cache_device: torch.device | None,
    crop_radius_px: float,
    step_px: float,
) -> dict[str, object]:
    flat_rows: list[dict[str, str]] = []
    pair_groups: list[int] = []
    pair_candidates: list[int] = []
    pair_slots: list[int] = []
    pair_view_probabilities: list[float] = []
    query_specs: list[tuple[str, Any]] = []
    query_centers: list[list[float]] = []
    for group, evidence_index in enumerate(evidence_indices):
        query_id = str(data.query_ids[int(evidence_index)])
        query_path = Path(image_root) / query_id
        query_specs.append(
            (str(query_path), lambda path=query_path: _load_query_rgb(path))
        )
        query_centers.append(data.query_xy[int(evidence_index)].tolist())
        for candidate, rows in enumerate(data.rows_by_group[int(evidence_index)]):
            for slot, row in enumerate(rows):
                flat_rows.append(row)
                pair_groups.append(int(group))
                pair_candidates.append(int(candidate))
                pair_slots.append(int(slot))
                probability = _float(
                    row, "support_view_probability", default=0.0
                )
                if probability < 0.0:
                    raise ValueError("RGB inference support-view probability is negative")
                pair_view_probabilities.append(probability)
    if not flat_rows:
        raise ValueError("RGB inference batch has no measurable views")
    support_specs: list[tuple[str, Any]] = []
    support_centers: list[list[float]] = []
    for row in flat_rows:
        support_path = Path(image_root) / str(row["support_image_id"])
        support_specs.append(
            (str(support_path), lambda path=support_path: _load_query_rgb(path))
        )
        support_centers.append(
            [
                _float(row, "support_x")
                if str(row.get("support_x", "")).strip()
                else _float(row, "render_x"),
                _float(row, "support_y")
                if str(row.get("support_y", "")).strip()
                else _float(row, "render_y"),
            ]
        )
    query_patches = _crop_cached_rgb_windows_grouped(
        query_specs,
        query_centers,
        cache=image_cache,
        cache_device=image_cache_device,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    support_patches = _crop_cached_rgb_windows_grouped(
        support_specs,
        support_centers,
        cache=image_cache,
        cache_device=image_cache_device,
        radius_px=float(crop_radius_px),
        step_px=float(step_px),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    indices = np.asarray(evidence_indices, dtype=np.int64)
    return {
        "evidence_indices": indices,
        "query_patches_by_group": query_patches,
        "support_patches": support_patches,
        "pair_group_indices": torch.tensor(pair_groups, dtype=torch.long),
        "pair_candidate_indices": torch.tensor(pair_candidates, dtype=torch.long),
        "pair_view_slots": torch.tensor(pair_slots, dtype=torch.long),
        "pair_view_probabilities": torch.tensor(
            pair_view_probabilities, dtype=torch.float32
        ),
        "candidate_valid": torch.from_numpy(data.candidate_valid[indices]),
        "candidate_prior": torch.from_numpy(data.prior[indices]),
        "unknown_probability": torch.from_numpy(data.unknown[indices]),
        "flat_rows": flat_rows,
    }
