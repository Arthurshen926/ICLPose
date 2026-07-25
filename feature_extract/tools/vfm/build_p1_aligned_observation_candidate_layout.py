"""Build a train-only, P1-shaped observation layout for identity training.

The prior broad identity pretrain used one same-track support plus generic
RADIO-PCA ANN negatives.  That is useful for appearance initialization, but
it does not reproduce the candidate/support distribution seen by P1: mapped
full-bank top-L tracks with fixed maplet support views.  This builder closes
that gap without changing runtime P1:

1. choose interior SfM observations from train query images only;
2. map their real RADIO feature maps with the production JointFeatureMapper;
3. retrieve the same global FAISS top-L landmark tracks used by P1;
4. attach the same fixed maplet support observations and retain two views;
5. apply a static, target-free support/coarse selector before any target join.

The saved arrays contain no track label, pose, residual, or projection.  The
choice of query anchor originates from train-only SfM observations, therefore
the artifact is explicitly forbidden as a runtime layout.  Exact identities
and coherent-wrong pose targets are attached later by separate train-only
builders.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_layout import (
    _array_sha256_short,
)
from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_observation_pairs import (
    _query_observation_candidates,
)
from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _fixed_support8_layout,
    _load_maplet_support_index,
)
from feature_extract.tools.vfm.eval_detector_global_landmark_proposals import (
    _load_feature_map,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
    CandidatePoseRGBSpatialLayout,
    _truncate_resolved_rgb_support_views,
    load_candidate_pose_rgb_spatial_layout,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    TARGET_FREE_SELECTOR_POLICIES,
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.global_landmark_ann import (
    FaissIVFConfig,
    build_or_load_faiss_ivf_index,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    fixed_topl_coarse_posterior,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    sample_dense_feature_points,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


_POINT_SOURCE = "train_sfm_observation"
_RUNTIME_FORBIDDEN_LAYOUT_REASON = "train_only_sfm_observation_anchor_selection"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-query-layout", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--matcha-joint-checkpoint", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--faiss-index-cache", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--feature-key", default="radio_final")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--mapper-batch-size", type=int, default=4)
    parser.add_argument("--max-observations-per-query", type=int, default=512)
    parser.add_argument("--border-margin-px", type=float, default=48.0)
    parser.add_argument(
        "--anchor-jitter-radius-px",
        type=float,
        default=0.0,
        help=(
            "Train-only target-free perturbation radius applied to selected SfM "
            "observation anchors before mapper projection and global top-L retrieval."
        ),
    )
    parser.add_argument(
        "--anchor-jitter-copies",
        type=int,
        default=0,
        help=(
            "Number of deterministic nonzero jittered copies per source observation. "
            "The unperturbed observation is always retained."
        ),
    )
    parser.add_argument("--points-per-query", type=int, default=64)
    parser.add_argument(
        "--selector-policy",
        choices=("uniform", "coarse_max_probability", "coarse_margin", "support_coverage", "coarse_margin_support_coverage"),
        default="support_coverage",
    )
    parser.add_argument("--selector-grid-rows", type=int, default=4)
    parser.add_argument("--selector-grid-columns", type=int, default=4)
    parser.add_argument("--proposal-top-l", type=int, default=20)
    parser.add_argument("--support-views-per-candidate", type=int, default=2)
    parser.add_argument("--faiss-nlist", type=int, default=1024)
    parser.add_argument("--faiss-train-samples", type=int, default=100000)
    parser.add_argument("--faiss-nprobe", type=int, default=128)
    parser.add_argument("--faiss-oversample-factor", type=int, default=4)
    parser.add_argument("--coarse-temperature", type=float, default=0.04)
    parser.add_argument("--diagnostic-null-probability", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a non-empty unique list")
    return devices


def _validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    values = (
        float(args.border_margin_px),
        float(args.anchor_jitter_radius_px),
        float(args.coarse_temperature),
        float(args.diagnostic_null_probability),
    )
    if (
        int(args.mapper_batch_size) <= 0
        or int(args.max_observations_per_query) <= 0
        or int(args.anchor_jitter_copies) < 0
        or int(args.points_per_query) <= 0
        or int(args.selector_grid_rows) <= 0
        or int(args.selector_grid_columns) <= 0
        or int(args.proposal_top_l) <= 0
        or int(args.support_views_per_candidate) <= 0
        or int(args.faiss_nlist) <= 0
        or int(args.faiss_train_samples) <= 0
        or int(args.faiss_nprobe) <= 0
        or int(args.faiss_oversample_factor) <= 0
        or not all(np.isfinite(value) for value in values)
        or float(args.border_margin_px) < 0.0
        or float(args.anchor_jitter_radius_px) < 0.0
        or float(args.coarse_temperature) <= 0.0
        or not 0.0 <= float(args.diagnostic_null_probability) < 1.0
        or str(args.selector_policy) not in TARGET_FREE_SELECTOR_POLICIES
    ):
        raise ValueError("P1-aligned observation-layout arguments are invalid")
    if (float(args.anchor_jitter_radius_px) == 0.0) != (int(args.anchor_jitter_copies) == 0):
        raise ValueError("P1-aligned anchor jitter requires both positive radius and copy count")
    return _parse_devices(str(args.devices))


def _array_manifest_sha256(*arrays: np.ndarray) -> str:
    """Hash train-only anchor coordinates without serializing their tracks."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()[:16]


def _stable_anchor_jitter_seed(
    *, seed: int, query_id: str, source_index: int, copy_index: int
) -> int:
    """Return a deterministic train-only perturbation seed without serializing a track."""

    if int(seed) < 0 or not str(query_id) or int(source_index) < 0 or int(copy_index) <= 0:
        raise ValueError("P1-aligned anchor jitter seed inputs are invalid")
    digest = hashlib.sha256(
        f"p1_aligned_anchor_jitter_v1\0{int(seed)}\0{query_id}\0{int(source_index)}\0{int(copy_index)}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _expand_train_observation_anchor_jitter(
    *,
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    query_tracks: np.ndarray,
    radius_px: float,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add reproducible local anchor variants before target-free retrieval.

    The input track IDs select only the original train-only observations.  They
    are returned solely so the caller can preserve row alignment during layout
    construction, then discarded before serialization.  Jitter itself depends
    only on the frozen source coordinate, query image, copy index, and seed.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float32)
    tracks = np.asarray(query_tracks, dtype=np.int64).reshape(-1)
    radius = float(radius_px)
    copy_count = int(copies)
    if (
        len(ids) == 0
        or xy.shape != (len(ids), 2)
        or tracks.shape != (len(ids),)
        or np.any(ids == "")
        or np.any(tracks < 0)
        or not np.isfinite(xy).all()
        or not np.isfinite(radius)
        or radius < 0.0
        or copy_count < 0
        or int(seed) < 0
        or (radius == 0.0) != (copy_count == 0)
    ):
        raise ValueError("P1-aligned train-observation jitter inputs are invalid")
    if copy_count == 0:
        return ids, xy, tracks
    output_ids = [ids]
    output_xy = [xy]
    output_tracks = [tracks]
    for copy_index in range(1, copy_count + 1):
        offsets = np.empty_like(xy)
        for source_index, query_id in enumerate(ids.tolist()):
            generator = np.random.default_rng(
                _stable_anchor_jitter_seed(
                    seed=int(seed),
                    query_id=str(query_id),
                    source_index=int(source_index),
                    copy_index=int(copy_index),
                )
            )
            # Uniform radius, rather than uniform area, puts its median at
            # half the configured radius and matches detector-anchor offsets.
            distance = float(generator.uniform(0.0, radius))
            angle = float(generator.uniform(0.0, 2.0 * np.pi))
            offsets[source_index] = (
                distance * np.cos(angle),
                distance * np.sin(angle),
            )
        output_ids.append(ids)
        output_xy.append((xy + offsets).astype(np.float32, copy=False))
        output_tracks.append(tracks)
    return (
        np.concatenate(output_ids, axis=0),
        np.concatenate(output_xy, axis=0),
        np.concatenate(output_tracks, axis=0),
    )


def _train_query_ids(layout: CandidatePoseRGBSpatialLayout) -> tuple[str, ...]:
    if (
        layout.metadata.get("contains_ground_truth") is not False
        or layout.metadata.get("pose_or_ground_truth_used") is not False
        or layout.metadata.get("render") is not False
        or layout.metadata.get("image_retrieval_or_submap_used") is not False
    ):
        raise ValueError("reference P1 layout is not target-free")
    rows = np.flatnonzero(np.asarray(layout.split_names).astype(str) == "train")
    result = tuple(sorted(set(np.asarray(layout.query_ids)[rows].astype(str).tolist())))
    if not result:
        raise ValueError("reference P1 layout has no train queries")
    return result


def _mapper_descriptors_at_observations(
    *,
    records_by_id: Mapping[str, TokenBankRecord],
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    image_sizes: Mapping[str, tuple[int, int]],
    checkpoint: Path,
    feature_key: str,
    devices: Sequence[str],
    mapper_batch_size: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Project full maps then sample the production descriptor space at anchors."""

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float32)
    if len(ids) == 0 or xy.shape != (len(ids), 2):
        raise ValueError("observation mapper inputs are invalid")
    rows_by_query = {
        query_id: np.flatnonzero(ids == query_id)
        for query_id in sorted(set(ids.tolist()))
    }
    if set(rows_by_query) - set(records_by_id) or set(rows_by_query) - set(image_sizes):
        raise ValueError("observation mapper lacks a query record or image size")
    partitions = [
        tuple(sorted(rows_by_query)[position :: len(devices)])
        for position in range(len(devices))
    ]

    def worker(worker_index: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        device = str(devices[worker_index])
        mapper = JointFeatureMapper(load_matcha_joint_model(checkpoint, device=device).model, device=device)
        output: dict[str, np.ndarray] = {}
        assigned = partitions[worker_index]
        started = time.monotonic()
        for start in range(0, len(assigned), int(mapper_batch_size)):
            batch_ids = assigned[start : start + int(mapper_batch_size)]
            raw_maps = [
                _load_feature_map(records_by_id[query_id].token_path, key=str(feature_key))
                for query_id in batch_ids
            ]
            if len({tuple(value.shape) for value in raw_maps}) != 1:
                raise ValueError("P1-aligned mapper batch has incompatible raw map shapes")
            mapped_maps = mapper.project_batch(np.stack(raw_maps, axis=0))
            for query_id, mapped in zip(batch_ids, mapped_maps):
                width, height = image_sizes[query_id]
                sampled, _ = sample_dense_feature_points(
                    mapped.coarse_descriptors,
                    xy[rows_by_query[query_id]],
                    image_width=int(width),
                    image_height=int(height),
                )
                output[query_id] = np.asarray(sampled, dtype=np.float32)
        return output, {
            "device": device,
            "query_image_count": int(len(assigned)),
            "elapsed_seconds": float(time.monotonic() - started),
        }

    if len(devices) == 1:
        values = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            values = list(executor.map(worker, range(len(devices))))
    result = np.empty((len(ids), 0), dtype=np.float32)
    summaries: list[dict[str, object]] = []
    seen: set[str] = set()
    for local, summary in values:
        if seen & set(local):
            raise RuntimeError("P1-aligned mapper workers overlapped query images")
        seen.update(local)
        summaries.append(summary)
        for query_id, descriptors in local.items():
            rows = rows_by_query[query_id]
            values_array = np.asarray(descriptors, dtype=np.float32)
            if values_array.ndim != 2 or values_array.shape[0] != len(rows):
                raise RuntimeError("P1-aligned mapper emitted an invalid descriptor block")
            if result.shape[1] == 0:
                result = np.empty((len(ids), values_array.shape[1]), dtype=np.float32)
            if values_array.shape[1] != result.shape[1]:
                raise RuntimeError("P1-aligned mapper descriptor dimensions differ")
            result[rows] = values_array
    if seen != set(rows_by_query) or result.shape[1] == 0 or not np.isfinite(result).all():
        raise RuntimeError("P1-aligned mapper did not emit every requested observation")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise RuntimeError("P1-aligned mapper emitted a zero descriptor")
    return (result / norms).astype(np.float32, copy=False), summaries


def _select_static_rows(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    points_per_query: int,
    policy: str,
    grid_rows: int,
    grid_columns: int,
    coordinate_image_size: tuple[int, int],
    border_margin_px: float,
) -> np.ndarray:
    """Freeze a selector-derived subset before target labels are joined."""

    selected_parts: list[np.ndarray] = []
    for query_id in sorted(set(layout.query_ids.tolist())):
        rows = np.flatnonzero(layout.query_ids == str(query_id))
        if len(rows) < int(points_per_query):
            raise ValueError("a train query has fewer observation anchors than point budget")
        selector_input = selector_input_from_target_free_layout(
            layout=layout,
            rows=rows,
            rgb_context_radius_px=float(border_margin_px),
            coordinate_image_size=coordinate_image_size,
        )
        positions = select_target_free_spatial_quota(
            selector_input=selector_input,
            quality_scores=target_free_selector_scores(
                selector_input=selector_input, policy=str(policy)
            ),
            point_budget=int(points_per_query),
            grid_rows=int(grid_rows),
            grid_columns=int(grid_columns),
            image_size=coordinate_image_size,
        )
        selected_parts.append(rows[np.asarray(positions, dtype=np.int64)])
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise RuntimeError("target-free observation preselection is invalid")
    return selected


def _subset_layout(
    layout: CandidatePoseRGBSpatialLayout, rows: np.ndarray, metadata: Mapping[str, object]
) -> CandidatePoseRGBSpatialLayout:
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray(layout.source_point_ids[selected], dtype=np.int64),
        query_ids=np.asarray(layout.query_ids[selected]).astype(str),
        split_names=np.asarray(layout.split_names[selected]).astype(str),
        xy=np.asarray(layout.xy[selected], dtype=np.float32),
        point_sources=np.asarray(layout.point_sources[selected]).astype(str),
        candidate_track_ids=np.asarray(layout.candidate_track_ids[selected], dtype=np.int64),
        candidate_bank_rows=np.asarray(layout.candidate_bank_rows[selected], dtype=np.int64),
        candidate_coarse_similarities=np.asarray(
            layout.candidate_coarse_similarities[selected], dtype=np.float32
        ),
        candidate_prior_probabilities=np.asarray(
            layout.candidate_prior_probabilities[selected], dtype=np.float32
        ),
        null_probabilities=np.asarray(layout.null_probabilities[selected], dtype=np.float32),
        support_image_ids=np.asarray(layout.support_image_ids[selected]).astype(str),
        support_xy=np.asarray(layout.support_xy[selected], dtype=np.float32),
        support_view_valid=np.asarray(layout.support_view_valid[selected], dtype=bool),
        support_view_weights=np.asarray(layout.support_view_weights[selected], dtype=np.float32),
        support_coverage_counts=np.asarray(layout.support_coverage_counts[selected], dtype=np.int32),
        metadata=dict(metadata),
    )


def build_p1_aligned_observation_candidate_layout(args: argparse.Namespace) -> dict[str, object]:
    """Create a train-only P1-shaped observation pool without target arrays."""

    devices = _validate_args(args)
    output = Path(args.output)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite P1-aligned observation layout")
    reference_path = Path(args.train_query_layout)
    reference = load_candidate_pose_rgb_spatial_layout(reference_path)
    train_query_ids = _train_query_ids(reference)
    if int(reference.candidate_count) != int(args.proposal_top_l):
        raise ValueError("P1-aligned candidate count differs from reference P1 layout")
    if int(reference.support_view_count) != int(args.support_views_per_candidate):
        raise ValueError("P1-aligned support-view count differs from reference P1 layout")

    manifest_path = Path(args.query_manifest)
    manifest = TokenBankManifest.from_json(manifest_path)
    manifest.validate(verify_checksums=False)
    records_by_id = {str(record.image_id): record for record in manifest.records}
    if not set(train_query_ids).issubset(records_by_id):
        raise ValueError("query manifest lacks a train query used by P1")

    model_dir = Path(args.colmap_model_dir)
    images_path = model_dir / "images.bin"
    cameras_path = model_dir / "cameras.bin"
    images = read_colmap_images_binary(images_path)
    cameras = read_colmap_cameras_binary(cameras_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    image_sizes = {
        str(image.image_name): (int(cameras[int(image.camera_id)].width), int(cameras[int(image.camera_id)].height))
        for image in images.values()
    }
    if not set(train_query_ids).issubset(image_sizes):
        raise ValueError("COLMAP model lacks a train query used by P1")
    unique_sizes = {image_sizes[query_id] for query_id in train_query_ids}
    if len(unique_sizes) != 1:
        raise ValueError("P1-aligned observation layout requires a common query resolution")
    coordinate_image_size = next(iter(unique_sizes))

    # The track ids returned here choose train-only observed anchor locations;
    # they are deliberately discarded before the target-free layout is made.
    query_ids, query_xy, _query_tracks = _query_observation_candidates(
        query_ids=train_query_ids,
        images_by_name=images_by_name,
        max_anchors_per_query=int(args.max_observations_per_query),
        seed=int(args.seed),
        sizes_by_image={
            query_id: np.asarray(image_sizes[query_id], dtype=np.int64)
            for query_id in train_query_ids
        },
        # Every jittered copy must remain inside the same RGB crop interior;
        # target-free retrieval then happens at the actual perturbed coordinate.
        border_margin_px=(
            float(args.border_margin_px) + float(args.anchor_jitter_radius_px)
        ),
    )
    base_observation_row_count = int(len(query_ids))
    query_ids, query_xy, _query_tracks = _expand_train_observation_anchor_jitter(
        query_ids=query_ids,
        query_xy=query_xy,
        query_tracks=_query_tracks,
        radius_px=float(args.anchor_jitter_radius_px),
        copies=int(args.anchor_jitter_copies),
        seed=int(args.seed),
    )
    descriptor_manifest_hash = _array_manifest_sha256(query_ids.astype(str), query_xy)
    checkpoint = Path(args.matcha_joint_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    descriptors, mapper_summaries = _mapper_descriptors_at_observations(
        records_by_id=records_by_id,
        query_ids=query_ids,
        query_xy=query_xy,
        image_sizes=image_sizes,
        checkpoint=checkpoint,
        feature_key=str(args.feature_key),
        devices=devices,
        mapper_batch_size=int(args.mapper_batch_size),
    )

    bank_path = Path(args.projected_landmark_bank)
    bank, bank_metadata = load_landmark_index_npz(bank_path)
    descriptor_space_id = str(bank_metadata.get("descriptor_space_id", ""))
    descriptor_space_manifest = bank_metadata.get("descriptor_space_manifest")
    if not descriptor_space_id or not isinstance(descriptor_space_manifest, Mapping):
        raise ValueError("projected landmark bank lacks a descriptor-space manifest")
    expected_checkpoint = str(descriptor_space_manifest.get("checkpoint_sha256", ""))
    if not expected_checkpoint or expected_checkpoint != file_sha256_short(checkpoint):
        raise ValueError("P1-aligned mapper checkpoint differs from landmark descriptor space")
    index, index_cache_hit = build_or_load_faiss_ivf_index(
        bank,
        landmark_bank_path=bank_path,
        descriptor_space_id=descriptor_space_id,
        cache_path=Path(args.faiss_index_cache),
        config=FaissIVFConfig(
            nlist=int(args.faiss_nlist),
            train_samples=int(args.faiss_train_samples),
            seed=0,
        ),
    )
    candidates = index.search_unique_tracks(
        descriptors,
        bank,
        proposal_top_l=int(args.proposal_top_l),
        nprobe=int(args.faiss_nprobe),
        oversample_factor=int(args.faiss_oversample_factor),
    )
    candidate_valid = np.asarray(candidates.track_ids, dtype=np.int64) >= 0
    coarse = np.where(candidate_valid, candidates.scores, 0.0).astype(np.float32)
    priors, null = fixed_topl_coarse_posterior(
        coarse,
        candidate_valid,
        temperature=float(args.coarse_temperature),
        null_probability=float(args.diagnostic_null_probability),
    )

    maplet_path = Path(args.maplet_support_index)
    maplet, maplet_metadata = _load_maplet_support_index(maplet_path)
    if str(maplet_metadata.get("source_descriptor_space_id", "")) != descriptor_space_id:
        raise ValueError("P1-aligned maplet support index differs from descriptor space")
    all_support_ids, all_support_valid, all_coverage = _fixed_support8_layout(
        layout={
            "candidate_track_ids": np.asarray(candidates.track_ids, dtype=np.int64),
            "candidate_canonical_rows": np.asarray(candidates.bank_row_indices, dtype=np.int64),
        },
        maplet=maplet,
    )
    geometry_path = Path(args.support_geometry_index)
    support_geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("P1-aligned layout requires real SfM support observation xy")
    cache_ids = np.unique(
        np.concatenate((query_ids.astype(str), all_support_ids[all_support_valid]))
    )
    runtime = build_fixed_candidate_context_runtime(
        query_ids=query_ids,
        query_xy=query_xy,
        candidate_track_ids=np.asarray(candidates.track_ids, dtype=np.int64),
        candidate_support_image_ids=all_support_ids,
        candidate_view_valid=all_support_valid,
        cache_image_ids=cache_ids,
        support_geometry=support_geometry,
    )
    support_ids, support_xy, support_valid, support_weights, support_coverage = (
        _truncate_resolved_rgb_support_views(
            query_ids=query_ids,
            support_image_ids=all_support_ids,
            support_xy=runtime.support_xy,
            support_view_valid=runtime.view_valid,
            support_coverage_counts=all_coverage,
            support_views_per_candidate=int(args.support_views_per_candidate),
        )
    )

    metadata: dict[str, object] = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_set": "fixed_full_global_faiss_top_l_tracks",
        "candidate_reselection": False,
        "candidate_top_k": int(args.proposal_top_l),
        "candidate_prior_semantics": "fixed_top20_coarse_softmax_with_explicit_diagnostic_null_v1",
        "candidate_prior_pose_calibrated": False,
        "support_view_selection": "fixed_maplet_coverage_rank_excluding_query_image_v1",
        "support_view_weight_semantics": "selected_maplet_coverage_normalized_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "support_views_per_candidate": int(args.support_views_per_candidate),
        "verification_points_sha256": descriptor_manifest_hash,
        "maplet_support_index": str(maplet_path.resolve()),
        "maplet_support_index_sha256": file_sha256_short(maplet_path),
        "support_geometry_index": str(geometry_path.resolve()),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "projected_landmark_bank": str(bank_path.resolve()),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "projection_space_id": descriptor_space_manifest.get("projection_space_id"),
        "descriptor_space_id": descriptor_space_id,
        "matcha_joint_checkpoint": str(checkpoint.resolve()),
        "matcha_joint_checkpoint_sha256": file_sha256_short(checkpoint),
        "feature_key": str(args.feature_key),
        "faiss_index_cache": str(Path(args.faiss_index_cache).resolve()),
        "faiss_index_cache_sha256": file_sha256_short(Path(args.faiss_index_cache)),
        "faiss_index_metadata_sha256": file_sha256_short(
            Path(args.faiss_index_cache).with_suffix(Path(args.faiss_index_cache).suffix + ".json")
        ),
        "faiss_nprobe": int(candidates.nprobe),
        "faiss_search_k": int(candidates.search_k),
        "source_train_query_layout": str(reference_path.resolve()),
        "source_train_query_layout_sha256": file_sha256_short(reference_path),
        "query_manifest": str(manifest_path.resolve()),
        "query_manifest_sha256": file_sha256_short(manifest_path),
        "colmap_images_sha256": file_sha256_short(images_path),
        "colmap_cameras_sha256": file_sha256_short(cameras_path),
        "training_only_anchor_layout": True,
        "training_only_anchor_source": "train_sfm_observation_xy",
        "training_only_anchor_track_ids_serialized": False,
        "training_only_anchor_jitter": {
            "radius_px": float(args.anchor_jitter_radius_px),
            "jittered_copies_per_source_observation": int(args.anchor_jitter_copies),
            "distribution": "uniform_radius_uniform_angle_v1",
            "applied_before_mapper_projection_and_global_faiss_topl": True,
            "source_track_ids_serialized": False,
            "target_free_after_layout_freeze": True,
        },
        "runtime_scorer_must_not_load_this_layout": True,
        "runtime_forbidden_reason": _RUNTIME_FORBIDDEN_LAYOUT_REASON,
        "source_target_arrays_read": False,
        "source_target_arrays_excluded": ["pose", "residual", "registered_track_identity"],
    }
    full_layout = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.arange(len(query_ids), dtype=np.int64),
        query_ids=query_ids,
        split_names=np.full((len(query_ids),), "train", dtype="<U5"),
        xy=query_xy,
        point_sources=np.full((len(query_ids),), _POINT_SOURCE, dtype=f"<U{len(_POINT_SOURCE)}"),
        candidate_track_ids=np.asarray(candidates.track_ids, dtype=np.int64),
        candidate_bank_rows=np.asarray(candidates.bank_row_indices, dtype=np.int64),
        candidate_coarse_similarities=coarse,
        candidate_prior_probabilities=priors,
        null_probabilities=null,
        support_image_ids=support_ids,
        support_xy=support_xy,
        support_view_valid=support_valid,
        support_view_weights=support_weights,
        support_coverage_counts=support_coverage,
        metadata=metadata,
    )
    selected_rows = _select_static_rows(
        layout=full_layout,
        points_per_query=int(args.points_per_query),
        policy=str(args.selector_policy),
        grid_rows=int(args.selector_grid_rows),
        grid_columns=int(args.selector_grid_columns),
        coordinate_image_size=coordinate_image_size,
        border_margin_px=float(args.border_margin_px),
    )
    selected_metadata = {
        **metadata,
        "anchor_preselection": {
            "policy": str(args.selector_policy),
            "point_budget_per_query": int(args.points_per_query),
            "grid_rows": int(args.selector_grid_rows),
            "grid_columns": int(args.selector_grid_columns),
            "target_free": True,
            "selected_row_count": int(len(selected_rows)),
            "preselection_source_row_count": int(full_layout.row_count),
        },
        "point_candidate_support_sha256": _array_sha256_short(
            np.asarray(full_layout.source_point_ids[selected_rows], dtype=np.int64),
            np.asarray(full_layout.candidate_track_ids[selected_rows], dtype=np.int64),
            np.asarray(full_layout.support_image_ids[selected_rows]).astype(str),
            np.asarray(full_layout.support_xy[selected_rows], dtype=np.float32),
        ),
    }
    layout = _subset_layout(full_layout, selected_rows, selected_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_candidate_pose_rgb_spatial_layout(layout, output)
    summary: dict[str, object] = {
        "stage": "build_p1_aligned_observation_candidate_layout",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "base_observation_row_count": int(base_observation_row_count),
        "preselection_row_count": int(full_layout.row_count),
        "row_count": int(layout.row_count),
        "query_count": int(len(train_query_ids)),
        "candidate_top_k": int(layout.candidate_count),
        "support_views_per_candidate": int(layout.support_view_count),
        "valid_support_view_count": int(np.count_nonzero(layout.support_view_valid)),
        "faiss_cache_hit": bool(index_cache_hit),
        "mapper_workers": mapper_summaries,
        "protocol": {
            "train_query_only": True,
            "output_contains_target_arrays": False,
            "runtime_scorer_must_not_load_this_layout": True,
            "candidate_reselection": False,
            "global_full_bank_faiss_only": True,
            "anchor_jitter_retrieves_new_global_topl_candidates": bool(
                int(args.anchor_jitter_copies) > 0
            ),
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    summary = build_p1_aligned_observation_candidate_layout(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
