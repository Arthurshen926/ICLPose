"""Build target-free, hypothesis-disjoint mixed verification points.

The artifact combines held-out ALIKE detail points with spatially stratified
RADIO-intermediate and RADIO-final context points.  Every point receives one
immutable full-bank FAISS top-L candidate set.  No pose, reprojection error,
render, image retrieval, or submap is used while building it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_detector_global_landmark_proposals import (
    _load_feature_map,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.global_landmark_ann import (
    FaissIVFConfig,
    build_or_load_faiss_ivf_index,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
    fixed_topl_coarse_posterior,
    mixed_verification_points_scoring_compatibility,
    select_detector_rows_spatial_quota,
    select_disjoint_lattice_points,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    sample_dense_feature_points,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


_ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class _ImagePlan:
    record_index: int
    record: TokenBankRecord
    split_name: str
    image_width: int
    image_height: int
    xy: np.ndarray
    sources: np.ndarray
    detector_rows: np.ndarray
    dense_xy: np.ndarray
    dense_output_rows: np.ndarray
    diagnostics: Mapping[str, object]


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not splits or len(set(splits)) != len(splits) or set(splits) - _ALLOWED_SPLITS:
        raise ValueError("splits must be a non-empty unique subset of train,validation,test")
    return splits


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a non-empty unique list")
    return devices


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--faiss_index_cache", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--mapper_batch_size", type=int, default=4)
    parser.add_argument("--alike_points_per_image", type=int, default=64)
    parser.add_argument("--radio_intermediate_grid_rows", type=int, default=8)
    parser.add_argument("--radio_intermediate_grid_columns", type=int, default=8)
    parser.add_argument("--radio_final_grid_rows", type=int, default=8)
    parser.add_argument("--radio_final_grid_columns", type=int, default=8)
    parser.add_argument("--detector_grid_rows", type=int, default=4)
    parser.add_argument("--detector_grid_columns", type=int, default=4)
    parser.add_argument("--fit_exclusion_radius_px", type=float, default=8.0)
    parser.add_argument("--proposal_top_l", type=int, default=20)
    parser.add_argument("--faiss_nlist", type=int, default=1024)
    parser.add_argument("--faiss_train_samples", type=int, default=100000)
    parser.add_argument("--faiss_nprobe", type=int, default=128)
    parser.add_argument("--faiss_oversample_factor", type=int, default=4)
    parser.add_argument("--coarse_temperature", type=float, default=0.04)
    parser.add_argument("--diagnostic_null_probability", type=float, default=0.10)
    return parser.parse_args(argv)


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata_json is not an object")
    return value


def _load_detector(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "image_ids",
        "offsets",
        "xy",
        "global_descriptors",
        "detector_scores",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"detector cache lacks {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        metadata = _metadata(payload, context="detector cache")
    if metadata.get("format") != "alike_detector_mapped_radio_query_cache_v1":
        raise ValueError("unsupported detector cache format")
    image_ids = np.asarray(arrays["image_ids"]).astype(str).reshape(-1)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    descriptors = np.asarray(arrays["global_descriptors"], dtype=np.float32)
    scores = np.asarray(arrays["detector_scores"], dtype=np.float32).reshape(-1)
    if (
        len(image_ids) == 0
        or len(set(image_ids.tolist())) != len(image_ids)
        or offsets.shape != (len(image_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(xy)
        or np.any(offsets[1:] < offsets[:-1])
        or xy.shape != (len(scores), 2)
        or descriptors.shape[0] != len(xy)
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(descriptors))
        or np.any(~np.isfinite(scores))
    ):
        raise ValueError("detector cache arrays are invalid")
    return {
        "image_ids": image_ids,
        "offsets": offsets,
        "xy": xy,
        "global_descriptors": descriptors,
        "detector_scores": scores,
    }, metadata


def _load_candidate_fit_rows(
    path: Path,
    *,
    detector_row_count: int,
) -> tuple[dict[str, list[int]], dict[str, str], dict[str, Any]]:
    """Read only target-free row ownership from the candidate artifact."""

    required = {"selected_rows", "query_ids", "split_names"}
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"candidate evidence lacks {sorted(missing)}")
        selected = np.asarray(payload["selected_rows"], dtype=np.int64).reshape(-1)
        query_ids = np.asarray(payload["query_ids"]).astype(str).reshape(-1)
        split_names = np.asarray(payload["split_names"]).astype(str).reshape(-1)
        metadata = _metadata(payload, context="candidate evidence")
    if metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("mixed verification requires candidate_evidence_v3")
    if (
        len(selected) == 0
        or len(np.unique(selected)) != len(selected)
        or not (query_ids.shape == split_names.shape == selected.shape)
        or np.any((selected < 0) | (selected >= int(detector_row_count)))
        or set(split_names.tolist()) - _ALLOWED_SPLITS
        or bool(metadata.get("pose_used_for_selection", True))
        or bool(metadata.get("image_retrieval", True))
        or bool(metadata.get("render", True))
    ):
        raise ValueError("candidate evidence violates the target-free fixed-candidate protocol")
    rows_by_query: dict[str, list[int]] = {}
    split_by_query: dict[str, str] = {}
    for row, query_id, split in zip(selected.tolist(), query_ids.tolist(), split_names.tolist()):
        previous = split_by_query.setdefault(str(query_id), str(split))
        if previous != str(split):
            raise ValueError(f"candidate evidence assigns multiple splits to {query_id}")
        rows_by_query.setdefault(str(query_id), []).append(int(row))
    return rows_by_query, split_by_query, metadata


def _detector_rows_for_image(query_id: str, detector: Mapping[str, np.ndarray]) -> np.ndarray:
    positions = np.flatnonzero(np.asarray(detector["image_ids"]).astype(str) == str(query_id))
    if len(positions) != 1:
        raise ValueError(f"detector cache does not uniquely contain {query_id}")
    offsets = np.asarray(detector["offsets"], dtype=np.int64)
    image_row = int(positions[0])
    return np.arange(int(offsets[image_row]), int(offsets[image_row + 1]), dtype=np.int64)


def _normalize_descriptors(values: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(values, dtype=np.float32)
    if descriptors.ndim != 2 or np.any(~np.isfinite(descriptors)):
        raise ValueError("descriptor rows are invalid")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("descriptor rows contain a zero vector")
    return descriptors / norms


def _build_image_plans(
    *,
    records: Sequence[TokenBankRecord],
    selected_splits: set[str],
    detector: Mapping[str, np.ndarray],
    fit_rows_by_query: Mapping[str, Sequence[int]],
    split_by_query: Mapping[str, str],
    image_sizes: Mapping[str, tuple[int, int]],
    alike_points_per_image: int,
    detector_grid_rows: int,
    detector_grid_columns: int,
    intermediate_grid_rows: int,
    intermediate_grid_columns: int,
    final_grid_rows: int,
    final_grid_columns: int,
    fit_exclusion_radius_px: float,
) -> list[_ImagePlan]:
    plans: list[_ImagePlan] = []
    for record_index, record in enumerate(records):
        query_id = str(record.image_id)
        split = split_by_query.get(query_id)
        if split is None:
            raise ValueError(f"candidate evidence has no split ownership for {query_id}")
        # The token manifest describes the extraction storage partition.  This
        # OldHospital manifest intentionally keeps all query-token files under
        # its train source tree, while candidate_evidence_v3 owns the frozen
        # evaluation split.  Never promote token storage metadata into a
        # validation/test label source.
        if split not in selected_splits:
            continue
        if query_id not in image_sizes:
            raise KeyError(f"COLMAP model has no camera size for {query_id}")
        width, height = image_sizes[query_id]
        all_rows = _detector_rows_for_image(query_id, detector)
        fit_rows = np.asarray(fit_rows_by_query.get(query_id, ()), dtype=np.int64)
        if len(fit_rows) == 0 or len(np.unique(fit_rows)) != len(fit_rows):
            raise ValueError(f"{query_id}: candidate fit rows are invalid")
        if np.any(~np.isin(fit_rows, all_rows)):
            raise ValueError(f"{query_id}: candidate fit rows are not detector-owned")
        detector_rows = select_detector_rows_spatial_quota(
            source_rows=all_rows,
            xy=np.asarray(detector["xy"], dtype=np.float32)[all_rows],
            scores=np.asarray(detector["detector_scores"], dtype=np.float32)[all_rows],
            excluded_rows=fit_rows,
            point_count=int(alike_points_per_image),
            image_width=int(width),
            image_height=int(height),
            grid_rows=int(detector_grid_rows),
            grid_columns=int(detector_grid_columns),
        )
        fit_xy = np.asarray(detector["xy"], dtype=np.float32)[fit_rows]
        intermediate_xy, intermediate_diag = select_disjoint_lattice_points(
            image_width=int(width),
            image_height=int(height),
            grid_rows=int(intermediate_grid_rows),
            grid_columns=int(intermediate_grid_columns),
            primary_phase=(0.25, 0.75),
            alternate_phases=((0.75, 0.25), (0.2, 0.25), (0.8, 0.75)),
            excluded_xy=fit_xy,
            minimum_distance_px=float(fit_exclusion_radius_px),
        )
        final_xy, final_diag = select_disjoint_lattice_points(
            image_width=int(width),
            image_height=int(height),
            grid_rows=int(final_grid_rows),
            grid_columns=int(final_grid_columns),
            primary_phase=(0.5, 0.5),
            alternate_phases=((0.25, 0.25), (0.75, 0.75), (0.2, 0.8)),
            excluded_xy=fit_xy,
            minimum_distance_px=float(fit_exclusion_radius_px),
        )
        alike_xy = np.asarray(detector["xy"], dtype=np.float32)[detector_rows]
        xy = np.concatenate((alike_xy, intermediate_xy, final_xy), axis=0).astype(np.float32)
        sources = np.concatenate(
            (
                np.full(
                    (len(alike_xy),),
                    POINT_SOURCE_ALIKE,
                    dtype=f"<U{len(POINT_SOURCE_ALIKE)}",
                ),
                np.full(
                    (len(intermediate_xy),),
                    POINT_SOURCE_RADIO_INTERMEDIATE,
                    dtype=f"<U{len(POINT_SOURCE_RADIO_INTERMEDIATE)}",
                ),
                np.full(
                    (len(final_xy),),
                    POINT_SOURCE_RADIO_FINAL,
                    dtype=f"<U{len(POINT_SOURCE_RADIO_FINAL)}",
                ),
            )
        )
        point_detector_rows = np.concatenate(
            (
                detector_rows,
                np.full((len(intermediate_xy) + len(final_xy),), -1, dtype=np.int64),
            )
        )
        dense_xy = np.concatenate((intermediate_xy, final_xy), axis=0).astype(np.float32)
        dense_output_rows = np.arange(len(alike_xy), len(xy), dtype=np.int64)
        if (
            len(np.unique(detector_rows)) != len(detector_rows)
            or np.any(np.isin(detector_rows, fit_rows))
            or xy.shape != (len(sources), 2)
            or point_detector_rows.shape != (len(xy),)
            or len(dense_xy) != len(dense_output_rows)
        ):
            raise RuntimeError(f"{query_id}: mixed verification point plan is invalid")
        plans.append(
            _ImagePlan(
                record_index=int(record_index),
                record=record,
                split_name=str(split),
                image_width=int(width),
                image_height=int(height),
                xy=xy,
                sources=sources,
                detector_rows=point_detector_rows,
                dense_xy=dense_xy,
                dense_output_rows=dense_output_rows,
                diagnostics={
                    "fit_point_count": int(len(fit_rows)),
                    "alike_heldout_point_count": int(len(alike_xy)),
                    "radio_intermediate": intermediate_diag,
                    "radio_final": final_diag,
                },
            )
        )
    if not plans:
        raise ValueError("no query images remain after split filtering")
    return plans


def _map_dense_descriptors(
    *,
    plans: Sequence[_ImagePlan],
    checkpoint: Path,
    feature_key: str,
    devices: Sequence[str],
    batch_size: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    if int(batch_size) <= 0:
        raise ValueError("mapper batch size must be positive")
    partitions = [list(range(index, len(plans), len(devices))) for index in range(len(devices))]

    def worker(worker_index: int) -> tuple[dict[int, np.ndarray], dict[str, object]]:
        device = str(devices[worker_index])
        run = load_matcha_joint_model(checkpoint, device=device)
        mapper = JointFeatureMapper(run.model, device=device)
        assigned = partitions[worker_index]
        output: dict[int, np.ndarray] = {}
        started = time.monotonic()
        for start in range(0, len(assigned), int(batch_size)):
            positions = assigned[start : start + int(batch_size)]
            local_plans = [plans[position] for position in positions]
            raw_maps = [
                _load_feature_map(plan.record.token_path, key=str(feature_key))
                for plan in local_plans
            ]
            shapes = {tuple(value.shape) for value in raw_maps}
            if len(shapes) != 1:
                raise ValueError("mixed verification mapper batch has incompatible raw map shapes")
            mapped_maps = mapper.project_batch(np.stack(raw_maps, axis=0))
            for plan, mapped in zip(local_plans, mapped_maps):
                sampled, _ = sample_dense_feature_points(
                    mapped.coarse_descriptors,
                    plan.dense_xy,
                    image_width=int(plan.image_width),
                    image_height=int(plan.image_height),
                )
                output[int(plan.record_index)] = _normalize_descriptors(sampled)
            completed = min(start + len(positions), len(assigned))
            print(
                json.dumps(
                    {
                        "stage": "mixed_verification_dense_mapper",
                        "device": device,
                        "completed": int(completed),
                        "assigned": int(len(assigned)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return output, {
            "device": device,
            "image_count": int(len(assigned)),
            "elapsed_seconds": float(time.monotonic() - started),
        }

    if len(devices) == 1:
        values = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            values = list(executor.map(worker, range(len(devices))))
    output: dict[int, np.ndarray] = {}
    worker_summaries: list[dict[str, object]] = []
    for local, summary in values:
        if set(output) & set(local):
            raise RuntimeError("mixed verification mapper workers overlapped records")
        output.update(local)
        worker_summaries.append(summary)
    expected = {int(plan.record_index) for plan in plans}
    if set(output) != expected:
        raise RuntimeError("mixed verification mapper did not emit every image")
    return output, worker_summaries


def build_mixed_multiscale_verification_points(
    *,
    query_manifest: Path,
    detector_query_cache: Path,
    candidate_evidence: Path,
    matcha_joint_checkpoint: Path,
    projected_landmark_bank: Path,
    faiss_index_cache: Path,
    colmap_model_dir: Path,
    output: Path,
    summary_json: Path,
    splits: Sequence[str],
    feature_key: str,
    devices: Sequence[str],
    mapper_batch_size: int,
    alike_points_per_image: int,
    detector_grid_rows: int,
    detector_grid_columns: int,
    intermediate_grid_rows: int,
    intermediate_grid_columns: int,
    final_grid_rows: int,
    final_grid_columns: int,
    fit_exclusion_radius_px: float,
    proposal_top_l: int,
    faiss_nlist: int,
    faiss_train_samples: int,
    faiss_nprobe: int,
    faiss_oversample_factor: int,
    coarse_temperature: float,
    diagnostic_null_probability: float,
) -> dict[str, object]:
    selected_splits = set(splits)
    if (
        not selected_splits
        or selected_splits - _ALLOWED_SPLITS
        or int(alike_points_per_image) <= 0
        or min(
            int(detector_grid_rows),
            int(detector_grid_columns),
            int(intermediate_grid_rows),
            int(intermediate_grid_columns),
            int(final_grid_rows),
            int(final_grid_columns),
            int(proposal_top_l),
            int(faiss_nlist),
            int(faiss_train_samples),
            int(faiss_nprobe),
            int(faiss_oversample_factor),
        )
        <= 0
        or not np.isfinite(float(fit_exclusion_radius_px))
        or float(fit_exclusion_radius_px) < 0.0
    ):
        raise ValueError("mixed verification build arguments are invalid")
    output = Path(output)
    summary_json = Path(summary_json)
    if output.exists() or summary_json.exists():
        raise FileExistsError("refusing to overwrite mixed verification outputs")
    checkpoint = Path(matcha_joint_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    started = time.monotonic()
    manifest = TokenBankManifest.from_json(Path(query_manifest))
    manifest.validate(verify_checksums=False)
    records = tuple(manifest.records)
    detector, detector_metadata = _load_detector(Path(detector_query_cache))
    if str(detector_metadata.get("query_manifest_sha256", "")) != str(
        file_sha256_short(Path(query_manifest))
    ):
        raise ValueError("detector cache and query manifest provenance differs")
    fit_rows_by_query, split_by_query, candidate_metadata = _load_candidate_fit_rows(
        Path(candidate_evidence), detector_row_count=len(detector["xy"])
    )
    detector_query_ids = set(np.asarray(detector["image_ids"]).astype(str).tolist())
    manifest_query_ids = {str(record.image_id) for record in records}
    if detector_query_ids != manifest_query_ids or set(split_by_query) != manifest_query_ids:
        raise ValueError("detector, candidate evidence, and query manifest image ownership differs")
    bank, bank_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
    descriptor_space_id = str(bank_metadata.get("descriptor_space_id", ""))
    descriptor_manifest = bank_metadata.get("descriptor_space_manifest")
    if not descriptor_space_id or not isinstance(descriptor_manifest, Mapping):
        raise ValueError("projected landmark bank lacks a descriptor-space manifest")
    expected_checkpoint = str(descriptor_manifest.get("checkpoint_sha256", ""))
    actual_checkpoint = file_sha256_short(checkpoint)
    if not expected_checkpoint or expected_checkpoint != actual_checkpoint:
        raise ValueError("mixed verification mapper checkpoint differs from landmark bank")
    if str(detector_metadata.get("descriptor_space_id", "")) != descriptor_space_id:
        raise ValueError("detector and projected landmark bank descriptor spaces differ")
    if str(detector_metadata.get("matcha_joint_checkpoint_sha256", "")) != actual_checkpoint:
        raise ValueError("detector and mixed verification mapper checkpoints differ")

    model_dir = Path(colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    image_sizes = {
        str(image.image_name): (
            int(cameras[int(image.camera_id)].width),
            int(cameras[int(image.camera_id)].height),
        )
        for image in images.values()
    }
    plans = _build_image_plans(
        records=records,
        selected_splits=selected_splits,
        detector=detector,
        fit_rows_by_query=fit_rows_by_query,
        split_by_query=split_by_query,
        image_sizes=image_sizes,
        alike_points_per_image=int(alike_points_per_image),
        detector_grid_rows=int(detector_grid_rows),
        detector_grid_columns=int(detector_grid_columns),
        intermediate_grid_rows=int(intermediate_grid_rows),
        intermediate_grid_columns=int(intermediate_grid_columns),
        final_grid_rows=int(final_grid_rows),
        final_grid_columns=int(final_grid_columns),
        fit_exclusion_radius_px=float(fit_exclusion_radius_px),
    )
    dense_descriptors, worker_summaries = _map_dense_descriptors(
        plans=plans,
        checkpoint=checkpoint,
        feature_key=str(feature_key),
        devices=tuple(str(value) for value in devices),
        batch_size=int(mapper_batch_size),
    )

    query_ids: list[np.ndarray] = []
    split_names: list[np.ndarray] = []
    xy: list[np.ndarray] = []
    point_sources: list[np.ndarray] = []
    source_detector_rows: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    plan_diagnostics: dict[str, object] = {}
    for plan in plans:
        values = np.empty((len(plan.xy), int(bank.feature_dim)), dtype=np.float32)
        alike_mask = plan.detector_rows >= 0
        values[alike_mask] = np.asarray(detector["global_descriptors"], dtype=np.float32)[
            plan.detector_rows[alike_mask]
        ]
        values[plan.dense_output_rows] = dense_descriptors[int(plan.record_index)]
        values = _normalize_descriptors(values)
        query_id = str(plan.record.image_id)
        split_name = str(plan.split_name)
        query_ids.append(
            np.full((len(plan.xy),), query_id, dtype=f"<U{len(query_id)}")
        )
        split_names.append(
            np.full((len(plan.xy),), split_name, dtype=f"<U{len(split_name)}")
        )
        xy.append(plan.xy)
        point_sources.append(plan.sources)
        source_detector_rows.append(plan.detector_rows)
        descriptors.append(values)
        plan_diagnostics[str(plan.record.image_id)] = dict(plan.diagnostics)
    query_ids_array = np.concatenate(query_ids)
    split_names_array = np.concatenate(split_names)
    xy_array = np.concatenate(xy).astype(np.float32, copy=False)
    sources_array = np.concatenate(point_sources)
    detector_rows_array = np.concatenate(source_detector_rows)
    descriptor_array = _normalize_descriptors(np.concatenate(descriptors)).astype(np.float32)
    source_point_ids = np.arange(len(query_ids_array), dtype=np.int64)
    if set(split_names_array.tolist()) != selected_splits:
        raise RuntimeError("mixed verification output does not contain exactly requested splits")

    faiss_index, faiss_cache_hit = build_or_load_faiss_ivf_index(
        bank,
        landmark_bank_path=Path(projected_landmark_bank),
        descriptor_space_id=descriptor_space_id,
        cache_path=Path(faiss_index_cache),
        config=FaissIVFConfig(
            nlist=int(faiss_nlist),
            train_samples=int(faiss_train_samples),
            seed=0,
        ),
    )
    candidates = faiss_index.search_unique_tracks(
        descriptor_array,
        bank,
        proposal_top_l=int(proposal_top_l),
        nprobe=int(faiss_nprobe),
        oversample_factor=int(faiss_oversample_factor),
    )
    valid = np.asarray(candidates.track_ids, dtype=np.int64) >= 0
    probabilities, null = fixed_topl_coarse_posterior(
        candidates.scores,
        valid,
        temperature=float(coarse_temperature),
        null_probability=float(diagnostic_null_probability),
    )
    if np.any(valid & (candidates.bank_row_indices < 0)):
        raise RuntimeError("full-bank FAISS emitted an invalid candidate row")
    point_count_by_source = {
        source: int(np.count_nonzero(sources_array == source))
        for source in (POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_INTERMEDIATE, POINT_SOURCE_RADIO_FINAL)
    }
    candidate_fit_artifact_sha = candidate_metadata.get("candidate_artifact_sha256")
    if candidate_fit_artifact_sha is not None and not str(candidate_fit_artifact_sha):
        raise ValueError("candidate evidence has an empty candidate-fit artifact hash")
    metadata: dict[str, object] = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "global_landmark_ann_used": True,
        "global_landmark_ann_scope": "full_projected_landmark_bank_only",
        "candidate_reselection": False,
        "render": False,
        "query_manifest": str(Path(query_manifest)),
        "query_manifest_sha256": file_sha256_short(Path(query_manifest)),
        "detector_query_cache": str(Path(detector_query_cache)),
        "detector_query_cache_sha256": file_sha256_short(Path(detector_query_cache)),
        "candidate_evidence": str(Path(candidate_evidence)),
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
        "candidate_evidence_format": candidate_metadata.get("format"),
        # candidate_evidence_v3 owns the target-free split/fit-row partition;
        # when present, this ties that partition back to the feature artifact
        # used by grouped hypothesis generation.
        "candidate_fit_artifact_sha256": candidate_fit_artifact_sha,
        "candidate_fit_row_count": int(sum(len(rows) for rows in fit_rows_by_query.values())),
        "matcha_joint_checkpoint": str(checkpoint),
        "matcha_joint_checkpoint_sha256": actual_checkpoint,
        "feature_key": str(feature_key),
        "projected_landmark_bank": str(Path(projected_landmark_bank)),
        "projected_landmark_bank_sha256": file_sha256_short(Path(projected_landmark_bank)),
        "descriptor_space_id": descriptor_space_id,
        "projection_space_id": descriptor_manifest.get("projection_space_id"),
        "faiss_index_cache": str(Path(faiss_index_cache)),
        "faiss_index_cache_sha256": file_sha256_short(Path(faiss_index_cache)),
        "faiss_index_metadata_sha256": file_sha256_short(
            Path(faiss_index_cache).with_suffix(Path(faiss_index_cache).suffix + ".json")
        ),
        "faiss_cache_hit": bool(faiss_cache_hit),
        "faiss_nprobe": int(candidates.nprobe),
        "faiss_search_k": int(candidates.search_k),
        "candidate_set": "fixed_full_global_faiss_top_l_unique_tracks",
        "candidate_top_k": int(proposal_top_l),
        "candidate_prior_semantics": "fixed_top20_coarse_softmax_with_explicit_diagnostic_null_v1",
        "candidate_prior_temperature": float(coarse_temperature),
        "candidate_null_probability": float(diagnostic_null_probability),
        "candidate_prior_pose_calibrated": False,
        "diagnostic_only_until_train_only_prior_calibration": True,
        "exported_splits": sorted(selected_splits),
        "query_split_source": "candidate_evidence_v3_split_names",
        "token_manifest_split_role": "feature_storage_only_not_evaluation_split",
        "test_source_points_materialized": bool("test" in selected_splits),
        "test_target_labels_materialized": False,
        "point_sources": {
            POINT_SOURCE_ALIKE: {
                "count_per_image": int(alike_points_per_image),
                "selection": "heldout_detector_score_spatial_round_robin_quota_v1",
                "fit_detector_rows_excluded": True,
            },
            POINT_SOURCE_RADIO_INTERMEDIATE: {
                "grid_rows": int(intermediate_grid_rows),
                "grid_columns": int(intermediate_grid_columns),
                "selection": "uniform_lattice_with_fit_distance_fallback_v1",
                "fit_exclusion_radius_px": float(fit_exclusion_radius_px),
            },
            POINT_SOURCE_RADIO_FINAL: {
                "grid_rows": int(final_grid_rows),
                "grid_columns": int(final_grid_columns),
                "selection": "uniform_lattice_with_fit_distance_fallback_v1",
                "fit_exclusion_radius_px": float(fit_exclusion_radius_px),
            },
        },
        "mapper_workers": worker_summaries,
        "colmap_model_dir": str(model_dir),
        "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
        "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
    }
    metadata["scoring_compatibility"] = mixed_verification_points_scoring_compatibility(
        metadata
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_point_ids=source_point_ids,
            query_ids=query_ids_array,
            split_names=split_names_array,
            xy=xy_array,
            point_sources=sources_array,
            source_detector_rows=detector_rows_array,
            descriptors=descriptor_array.astype(np.float16),
            candidate_bank_rows=np.asarray(candidates.bank_row_indices, dtype=np.int64),
            candidate_track_ids=np.asarray(candidates.track_ids, dtype=np.int64),
            candidate_prototype_ids=np.asarray(candidates.prototype_ids, dtype=np.int64),
            candidate_coarse_similarities=np.asarray(candidates.scores, dtype=np.float32),
            candidate_prior_probabilities=probabilities.astype(np.float32),
            null_probabilities=null.astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output)
    summary: dict[str, object] = {
        "stage": "build_mixed_multiscale_verification_points",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "query_count": int(len(set(query_ids_array.tolist()))),
        "point_count": int(len(query_ids_array)),
        "point_count_by_source": point_count_by_source,
        "candidate_top_k": int(proposal_top_l),
        "plan_diagnostics": plan_diagnostics,
        "protocol": {
            "hypothesis_detector_rows_excluded_from_alike": True,
            "dense_points_spatially_separated_from_fit_rows_when_possible": True,
            "fixed_full_global_faiss_top_l": True,
            "candidate_reselection": False,
            "pose_or_ground_truth_used": False,
            "test_target_labels_materialized": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_mixed_multiscale_verification_points(
        query_manifest=Path(args.query_manifest),
        detector_query_cache=Path(args.detector_query_cache),
        candidate_evidence=Path(args.candidate_evidence),
        matcha_joint_checkpoint=Path(args.matcha_joint_checkpoint),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        faiss_index_cache=Path(args.faiss_index_cache),
        colmap_model_dir=Path(args.colmap_model_dir),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        splits=_parse_splits(args.splits),
        feature_key=str(args.feature_key),
        devices=_parse_devices(args.devices),
        mapper_batch_size=int(args.mapper_batch_size),
        alike_points_per_image=int(args.alike_points_per_image),
        detector_grid_rows=int(args.detector_grid_rows),
        detector_grid_columns=int(args.detector_grid_columns),
        intermediate_grid_rows=int(args.radio_intermediate_grid_rows),
        intermediate_grid_columns=int(args.radio_intermediate_grid_columns),
        final_grid_rows=int(args.radio_final_grid_rows),
        final_grid_columns=int(args.radio_final_grid_columns),
        fit_exclusion_radius_px=float(args.fit_exclusion_radius_px),
        proposal_top_l=int(args.proposal_top_l),
        faiss_nlist=int(args.faiss_nlist),
        faiss_train_samples=int(args.faiss_train_samples),
        faiss_nprobe=int(args.faiss_nprobe),
        faiss_oversample_factor=int(args.faiss_oversample_factor),
        coarse_temperature=float(args.coarse_temperature),
        diagnostic_null_probability=float(args.diagnostic_null_probability),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
