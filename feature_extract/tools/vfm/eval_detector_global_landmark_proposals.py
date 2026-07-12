"""Evaluate deployable ALIKE query points against a full global landmark bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
    nearest_visible_landmarks,
    summarize_detector_proposal_geometry,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.global_landmark_ann import (
    FaissIVFConfig,
    audit_ann_against_exact_candidates,
    build_or_load_faiss_ivf_index,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.pipeline import _load_feature_map
from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
    DetectedImageFeatures,
    SPATIAL_DETECTION_SELECTION_VERSION,
    sample_dense_feature_points,
    spatially_diverse_detection_indices,
)
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.tokens import TokenBankManifest


def _int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or any(item <= 0 for item in output):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return output


def _float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or any(item <= 0.0 for item in output):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--faiss_index_cache", required=True)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--exact_probe_arrays", default=None)
    parser.add_argument("--exact_query_global_cache", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--mapper_device", default="cuda:0")
    parser.add_argument("--detector_devices", default="cuda:0,cuda:1")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--detector_top_k", type=int, default=512)
    parser.add_argument("--detector_candidate_top_k", type=int, default=4096)
    parser.add_argument("--detector_nms_radius_px", type=float, default=4.0)
    parser.add_argument("--detector_grid_rows", type=int, default=4)
    parser.add_argument("--detector_grid_cols", type=int, default=4)
    parser.add_argument("--detector_min_score", type=float, default=None)
    parser.add_argument("--mapper_batch_size", type=int, default=8)
    parser.add_argument("--proposal_top_l", type=int, default=20)
    parser.add_argument("--faiss_nlist", type=int, default=1024)
    parser.add_argument("--faiss_train_samples", type=int, default=100000)
    parser.add_argument("--faiss_nprobe_values", type=_int_list, default=(16, 32, 64, 128))
    parser.add_argument("--faiss_oversample_factor", type=int, default=4)
    parser.add_argument("--ann_min_positive_retention", type=float, default=0.99)
    parser.add_argument("--ann_min_top1_agreement", type=float, default=0.95)
    parser.add_argument(
        "--frozen_faiss_nprobe",
        type=int,
        default=None,
        help="reuse a validation-selected nprobe and skip ANN oracle selection",
    )
    parser.add_argument("--geometry_thresholds_px", type=_float_list, default=(1.0, 2.0, 5.0, 8.0))
    parser.add_argument("--pose_max_points", type=int, default=300)
    parser.add_argument("--pose_nms_radius_px", type=float, default=8.0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help=(
            "write detector/global top-L proposals without using query poses, "
            "GT residuals, or running pose evaluation"
        ),
    )
    return parser.parse_args(argv)


def _source_image_manifest_hash(records, image_root: Path) -> str:
    lines = []
    for record in records:
        path = Path(image_root) / str(record.image_id)
        if not path.exists():
            raise FileNotFoundError(path)
        lines.append(f"{record.image_id}:{file_sha256_short(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf8")).hexdigest()[:16]


def _alike_checkpoint(matcha_repo: Path, model_name: str) -> Path:
    suffix = str(model_name).split("-")[-1]
    path = Path(matcha_repo) / "third_party" / "alike" / "models" / f"alike-{suffix}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _detector_cache_expected(
    *,
    args: argparse.Namespace,
    records,
    descriptor_space_id: str,
) -> dict[str, object]:
    return {
        "format": "alike_detector_mapped_radio_query_cache_v1",
        "query_manifest": str(Path(args.query_manifest)),
        "query_manifest_sha256": file_sha256_short(Path(args.query_manifest)),
        "source_image_manifest_sha256": _source_image_manifest_hash(records, Path(args.image_root)),
        "matcha_joint_checkpoint": str(Path(args.matcha_joint_checkpoint)),
        "matcha_joint_checkpoint_sha256": file_sha256_short(Path(args.matcha_joint_checkpoint)),
        "descriptor_space_id": str(descriptor_space_id),
        "feature_key": str(args.feature_key),
        "alike_model_name": str(args.alike_model_name),
        "alike_checkpoint_sha256": file_sha256_short(
            _alike_checkpoint(Path(args.matcha_repo), str(args.alike_model_name))
        ),
        "detector_top_k": int(args.detector_top_k),
        "detector_candidate_top_k": int(args.detector_candidate_top_k),
        "detector_nms_radius_px": float(args.detector_nms_radius_px),
        "detector_grid_rows": int(args.detector_grid_rows),
        "detector_grid_cols": int(args.detector_grid_cols),
        "detector_min_score": args.detector_min_score,
        "detector_selection_version": SPATIAL_DETECTION_SELECTION_VERSION,
        "coordinate_convention": "sfm_pixel_endpoint_subpixel_v1",
    }


def _load_detector_cache(path: Path, expected: dict[str, object]):
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"stale detector query cache: {json.dumps(mismatches, sort_keys=True)}")
        payload = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    return payload, metadata


def _extract_alike_detections(
    records,
    *,
    images_by_name,
    cameras,
    args: argparse.Namespace,
) -> tuple[list[DetectedImageFeatures], dict[str, object]]:
    devices = tuple(value.strip() for value in str(args.detector_devices).split(",") if value.strip())
    if not devices:
        raise ValueError("at least one detector device is required")
    partitions = [list(range(index, len(records), len(devices))) for index in range(len(devices))]

    def worker(worker_index: int):
        extractor = AlikeDenseObservationExtractor(
            device=devices[worker_index],
            matcha_repo=Path(args.matcha_repo),
            model_name=str(args.alike_model_name),
        )
        values = []
        for record_index in partitions[worker_index]:
            record = records[record_index]
            image = images_by_name.get(str(record.image_id))
            if image is None:
                raise KeyError(f"query image missing from COLMAP model: {record.image_id}")
            camera = cameras[int(image.camera_id)]
            detected = extractor.detect(
                Path(args.image_root) / str(record.image_id),
                image_width=int(camera.width),
                image_height=int(camera.height),
                top_k=int(args.detector_top_k),
                candidate_top_k=int(args.detector_candidate_top_k),
                nms_radius_px=float(args.detector_nms_radius_px),
                grid_rows=int(args.detector_grid_rows),
                grid_cols=int(args.detector_grid_cols),
                min_score=args.detector_min_score,
                sub_pixel=True,
            )
            values.append((int(record_index), detected))
        return values, extractor.metadata

    if len(devices) == 1:
        worker_outputs = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            worker_outputs = list(executor.map(worker, range(len(devices))))
    detections: list[DetectedImageFeatures | None] = [None] * len(records)
    for values, _metadata in worker_outputs:
        for index, detected in values:
            detections[int(index)] = detected
    if any(value is None for value in detections):
        raise RuntimeError("detector workers did not return every query image")
    return [value for value in detections if value is not None], {
        **dict(worker_outputs[0][1]),
        "devices": list(devices),
        "partition_image_counts": [len(partition) for partition in partitions],
    }


def _build_detector_cache(
    *,
    records,
    images_by_name,
    cameras,
    args: argparse.Namespace,
    expected: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    detections, detector_metadata = _extract_alike_detections(
        records,
        images_by_name=images_by_name,
        cameras=cameras,
        args=args,
    )
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=str(args.mapper_device))
    mapper = JointFeatureMapper(joint_run.model, device=str(args.mapper_device))
    mapped_descriptors: list[np.ndarray] = []
    for start in range(0, len(records), int(args.mapper_batch_size)):
        batch_records = records[start : start + int(args.mapper_batch_size)]
        raw_maps = np.stack(
            [_load_feature_map(record.token_path, key=str(args.feature_key)) for record in batch_records],
            axis=0,
        )
        mapped = mapper.project_batch(raw_maps)
        for local_index, mapped_map in enumerate(mapped):
            global_index = start + local_index
            image = images_by_name[str(records[global_index].image_id)]
            camera = cameras[int(image.camera_id)]
            sampled, _scores = sample_dense_feature_points(
                mapped_map.coarse_descriptors,
                detections[global_index].xy,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
            mapped_descriptors.append(sampled)
    offsets = np.zeros((len(records) + 1,), dtype=np.int64)
    for index, detected in enumerate(detections):
        offsets[index + 1] = offsets[index] + len(detected.xy)
    payload = {
        "image_ids": np.asarray([str(record.image_id) for record in records], dtype=np.str_),
        "offsets": offsets,
        "xy": np.concatenate([value.xy for value in detections], axis=0).astype(np.float32),
        "local_descriptors": np.concatenate([value.descriptors for value in detections], axis=0).astype(np.float16),
        "global_descriptors": np.concatenate(mapped_descriptors, axis=0).astype(np.float16),
        "detector_scores": np.concatenate([value.scores for value in detections], axis=0).astype(np.float32),
        "detector_dispersions": np.concatenate([value.dispersions for value in detections], axis=0).astype(np.float32),
    }
    metadata = {
        **expected,
        "query_count": int(len(records)),
        "detector_point_count": int(len(payload["xy"])),
        "local_descriptor_dimension": int(payload["local_descriptors"].shape[1]),
        "global_descriptor_dimension": int(payload["global_descriptors"].shape[1]),
        "detector": detector_metadata,
        "image_sha256_by_id": {
            str(record.image_id): str(detected.image_sha256)
            for record, detected in zip(records, detections)
        },
    }
    cache_path = Path(args.detector_query_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **payload,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    return payload, metadata


def _query_ids_per_point(payload: dict[str, np.ndarray]) -> np.ndarray:
    image_ids = np.asarray(payload["image_ids"]).astype(str)
    offsets = np.asarray(payload["offsets"], dtype=np.int64)
    return np.repeat(image_ids, np.diff(offsets))


def _pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def _pose_point_mask(
    *,
    payload: dict[str, np.ndarray],
    coarse_scores: np.ndarray,
    args: argparse.Namespace,
    cameras,
    images_by_name,
) -> np.ndarray:
    offsets = np.asarray(payload["offsets"], dtype=np.int64)
    xy = np.asarray(payload["xy"], dtype=np.float32)
    detector_scores = np.asarray(payload["detector_scores"], dtype=np.float32)
    image_ids = np.asarray(payload["image_ids"]).astype(str)
    keep = np.zeros((len(xy),), dtype=bool)
    for image_index, query_id in enumerate(image_ids):
        start, end = int(offsets[image_index]), int(offsets[image_index + 1])
        image = images_by_name[str(query_id)]
        camera = cameras[int(image.camera_id)]
        score = np.asarray(coarse_scores[start:end, 0], dtype=np.float32)
        local_detector = detector_scores[start:end]
        detector_scale = max(float(np.nanmax(local_detector) - np.nanmin(local_detector)), 1e-6)
        normalized_detector = (local_detector - float(np.nanmin(local_detector))) / detector_scale
        quality = score + 0.1 * normalized_detector
        selected = spatially_diverse_detection_indices(
            xy[start:end],
            quality,
            top_k=min(int(args.pose_max_points), end - start),
            nms_radius_px=float(args.pose_nms_radius_px),
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_rows=int(args.detector_grid_rows),
            grid_cols=int(args.detector_grid_cols),
        )
        keep[start + selected] = True
    return keep


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    descriptor_space_id = str(landmark_metadata.get("descriptor_space_id", ""))
    if not descriptor_space_id:
        raise ValueError("projected landmark bank is missing descriptor_space_id")

    cache_expected = _detector_cache_expected(
        args=args,
        records=records,
        descriptor_space_id=descriptor_space_id,
    )
    detector_cache_path = Path(args.detector_query_cache)
    detector_cache_hit = detector_cache_path.exists()
    if detector_cache_hit:
        detector_payload, detector_metadata = _load_detector_cache(detector_cache_path, cache_expected)
    else:
        detector_payload, detector_metadata = _build_detector_cache(
            records=records,
            images_by_name=images_by_name,
            cameras=cameras,
            args=args,
            expected=cache_expected,
        )

    faiss_index, faiss_cache_hit = build_or_load_faiss_ivf_index(
        landmark_index,
        landmark_bank_path=Path(args.projected_landmark_bank),
        descriptor_space_id=descriptor_space_id,
        cache_path=Path(args.faiss_index_cache),
        config=FaissIVFConfig(
            nlist=int(args.faiss_nlist),
            train_samples=int(args.faiss_train_samples),
            seed=0,
        ),
    )
    ann_trials: list[dict[str, object]] = []
    if args.frozen_faiss_nprobe is not None:
        if int(args.frozen_faiss_nprobe) <= 0:
            raise ValueError("frozen FAISS nprobe must be positive")
        chosen_nprobe = int(args.frozen_faiss_nprobe)
        ann_gate_passed = True
        ann_selection = "frozen_validation_value"
    else:
        if args.exact_probe_arrays is None or args.exact_query_global_cache is None:
            raise ValueError(
                "ANN selection requires exact probe arrays and query descriptors"
            )
        with np.load(Path(args.exact_probe_arrays), allow_pickle=False) as data:
            exact_track_ids = np.asarray(
                data["candidate_track_ids"], dtype=np.int64
            )
            exact_correct_track_ids = np.asarray(
                data["correct_track_ids"], dtype=np.int64
            )
            exact_bank_rows = np.asarray(data["bank_row_indices"], dtype=np.int64)
        valid_exact = exact_bank_rows >= 0
        if not np.array_equal(
            exact_track_ids[valid_exact],
            landmark_index.track_ids[exact_bank_rows[valid_exact]],
        ):
            raise ValueError(
                "exact proposal artifact is not aligned with the requested landmark bank"
            )
        with np.load(Path(args.exact_query_global_cache), allow_pickle=False) as data:
            exact_query_features = np.asarray(
                data["query_global_features"], dtype=np.float32
            )
            exact_query_metadata = json.loads(str(data["metadata_json"].item()))
        if (
            str(exact_query_metadata.get("descriptor_space_id", ""))
            != descriptor_space_id
        ):
            raise ValueError(
                "exact query cache and landmark bank descriptor spaces differ"
            )
        if exact_query_features.shape[0] != exact_track_ids.shape[0]:
            raise ValueError("exact query feature and proposal row counts differ")
        if exact_track_ids.shape[1] != int(args.proposal_top_l):
            raise ValueError("exact proposal top-L differs from --proposal_top_l")
        for nprobe in args.faiss_nprobe_values:
            result = faiss_index.search_unique_tracks(
                exact_query_features,
                landmark_index,
                proposal_top_l=int(args.proposal_top_l),
                nprobe=int(nprobe),
                oversample_factor=int(args.faiss_oversample_factor),
            )
            audit = audit_ann_against_exact_candidates(
                result.track_ids,
                exact_track_ids,
                exact_correct_track_ids,
            )
            passes = (
                float(audit["correct_track_retention_given_exact"])
                >= float(args.ann_min_positive_retention)
                and float(audit["top1_track_agreement"])
                >= float(args.ann_min_top1_agreement)
            )
            ann_trials.append(
                {"nprobe": int(nprobe), "passes_gate": bool(passes), **audit}
            )
        passing = [
            index
            for index, trial in enumerate(ann_trials)
            if bool(trial["passes_gate"])
        ]
        chosen_ann_index = passing[0] if passing else len(ann_trials) - 1
        chosen_nprobe = int(ann_trials[chosen_ann_index]["nprobe"])
        ann_gate_passed = bool(ann_trials[chosen_ann_index]["passes_gate"])
        ann_selection = "exact_validation_oracle_audit"

    detector_search = faiss_index.search_unique_tracks(
        np.asarray(detector_payload["global_descriptors"], dtype=np.float32),
        landmark_index,
        proposal_top_l=int(args.proposal_top_l),
        nprobe=chosen_nprobe,
        oversample_factor=int(args.faiss_oversample_factor),
    )
    candidates = UniqueTrackCandidateSet(
        detector_search.bank_row_indices,
        detector_search.track_ids,
        detector_search.prototype_ids,
        detector_search.scores,
    )
    query_ids = _query_ids_per_point(detector_payload)
    xy = np.asarray(detector_payload["xy"], dtype=np.float32)
    offsets = np.asarray(detector_payload["offsets"], dtype=np.int64)
    pose_keep = _pose_point_mask(
        payload=detector_payload,
        coarse_scores=candidates.coarse_scores,
        args=args,
        cameras=cameras,
        images_by_name=images_by_name,
    )
    geometry: dict[str, object] = {}
    pose: dict[str, object] = {}
    pose_rows: dict[str, object] = {}
    supervision_payload: dict[str, np.ndarray] = {}
    if not bool(args.inference_only):
        nearest_rows = np.full((len(xy),), -1, dtype=np.int64)
        nearest_tracks = np.full((len(xy),), -1, dtype=np.int64)
        nearest_residuals = np.full((len(xy),), np.inf, dtype=np.float32)
        candidate_residuals = np.full(
            candidates.bank_row_indices.shape, np.inf, dtype=np.float32
        )
        for image_index, query_id in enumerate(
            np.asarray(detector_payload["image_ids"]).astype(str)
        ):
            row_start, row_end = (
                int(offsets[image_index]),
                int(offsets[image_index + 1]),
            )
            image = images_by_name[str(query_id)]
            camera = cameras[int(image.camera_id)]
            pose_w2c = _pose_w2c(image)
            rows, tracks, distances = nearest_visible_landmarks(
                xy[row_start:row_end],
                landmark_index,
                pose_w2c,
                camera,
            )
            nearest_rows[row_start:row_end] = rows
            nearest_tracks[row_start:row_end] = tracks
            nearest_residuals[row_start:row_end] = distances
            candidate_residuals[row_start:row_end] = (
                candidate_reprojection_residuals(
                    xy[row_start:row_end],
                    candidates.bank_row_indices[row_start:row_end],
                    landmark_index,
                    pose_w2c,
                    camera,
                )
            )
        geometry = summarize_detector_proposal_geometry(
            nearest_landmark_residuals=nearest_residuals,
            candidate_residuals=candidate_residuals,
            query_ids=query_ids,
            thresholds_px=args.geometry_thresholds_px,
            top_ls=(1, 5, 10, int(args.proposal_top_l)),
        )
        observations = [
            ColmapTrackObservation(
                track_id=int(nearest_tracks[row]),
                image_id=str(query_ids[row]),
                point2d_idx=int(row),
                xy=(float(xy[row, 0]), float(xy[row, 1])),
                xyz=np.zeros((3,), dtype=np.float64),
                track_length=1,
                reprojection_error=0.0,
            )
            for row in range(len(xy))
        ]
        coarse_pose_scores = candidates.coarse_scores.copy()
        coarse_pose_scores[~pose_keep] = -np.inf
        pose["coarse_top1"], pose_rows["coarse_top1"] = (
            _evaluate_pose_strategy(
                strategy="detector_global_coarse_top1",
                scores=coarse_pose_scores,
                candidates=candidates,
                query_observations=observations,
                query_ids=query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )
        )
        for threshold in (2.0, 5.0):
            proposal_oracle_scores = -candidate_residuals
            proposal_oracle_scores[
                (candidate_residuals > threshold) | (~pose_keep[:, None])
            ] = -np.inf
            name = f"proposal_geometry_oracle_{threshold:g}px"
            pose[name], pose_rows[name] = _evaluate_pose_strategy(
                strategy=name,
                scores=proposal_oracle_scores,
                candidates=candidates,
                query_observations=observations,
                query_ids=query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )
            map_rows = nearest_rows[:, None]
            valid_map = map_rows[:, 0] >= 0
            map_tracks = np.full_like(map_rows, -1)
            map_prototypes = np.full_like(map_rows, -1)
            map_scores = np.full(map_rows.shape, -np.inf, dtype=np.float32)
            map_tracks[valid_map, 0] = landmark_index.track_ids[
                map_rows[valid_map, 0]
            ]
            map_prototypes[valid_map, 0] = landmark_index.prototype_ids[
                map_rows[valid_map, 0]
            ]
            accepted_map = valid_map & pose_keep & (
                nearest_residuals <= threshold
            )
            map_scores[accepted_map, 0] = -nearest_residuals[accepted_map]
            map_candidates = UniqueTrackCandidateSet(
                map_rows, map_tracks, map_prototypes, map_scores
            )
            map_name = f"detector_map_geometry_oracle_{threshold:g}px"
            pose[map_name], pose_rows[map_name] = _evaluate_pose_strategy(
                strategy=map_name,
                scores=map_scores,
                candidates=map_candidates,
                query_observations=observations,
                query_ids=query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )
        supervision_payload = {
            "nearest_visible_bank_rows": nearest_rows,
            "nearest_visible_track_ids": nearest_tracks,
            "nearest_visible_residuals_px": nearest_residuals,
            "candidate_gt_residuals_px": candidate_residuals,
        }

    proposal_path = output_dir / "detector_global_proposals.npz"
    np.savez(
        proposal_path,
        query_ids=query_ids,
        xy=xy,
        detector_scores=np.asarray(detector_payload["detector_scores"], dtype=np.float32),
        detector_dispersions=np.asarray(detector_payload["detector_dispersions"], dtype=np.float32),
        bank_row_indices=candidates.bank_row_indices,
        candidate_track_ids=candidates.track_ids,
        candidate_prototype_ids=candidates.prototype_ids,
        coarse_scores=candidates.coarse_scores,
        pose_keep_mask=pose_keep,
        **supervision_payload,
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "s4_l2_detector_global_landmark_proposal_audit",
        "protocol": {
            "query_point_source": "alike_subpixel_detector_real_rgb",
            "landmark_search": "full_bank_global_faiss_ivf_flat",
            "image_retrieval": False,
            "submap": False,
            "ratio_test": False,
            "render": False,
            "proposal_top_l": int(args.proposal_top_l),
            "pose_max_points": int(args.pose_max_points),
            "inference_only": bool(args.inference_only),
            "ground_truth_pose_used": not bool(args.inference_only),
        },
        "descriptor_space_id": descriptor_space_id,
        "detector_cache": {
            "cache_hit": bool(detector_cache_hit),
            "path": str(detector_cache_path),
            "sha256": file_sha256_short(detector_cache_path),
            "metadata": detector_metadata,
        },
        "faiss": {
            "cache_hit": bool(faiss_cache_hit),
            "path": str(args.faiss_index_cache),
            "metadata": faiss_index.metadata,
            "exact_audit_trials": ann_trials,
            "nprobe_selection": ann_selection,
            "gate_passed": bool(ann_gate_passed),
            "chosen_nprobe": int(chosen_nprobe),
            "detector_search_k": int(detector_search.search_k),
        },
        "geometry": geometry,
        "pose": pose,
        "runtime_seconds": float(time.time() - start_time),
        "limitations": [
            "this stage audits detector/global-proposal behavior before support-view assignment",
            *(
                ["GT geometry and pose evaluation were disabled in inference-only mode"]
                if bool(args.inference_only)
                else [
                    "geometric oracle rows use GT pose only to localize the remaining error source"
                ]
            ),
            "the fixed ALIKE detector is not yet trained for SfM-track mappability",
            "pixel measurement and keep/update/drop are not active",
        ],
        "outputs": {
            "detector_query_cache": str(detector_cache_path),
            "faiss_index_cache": str(args.faiss_index_cache),
            "proposals": str(proposal_path),
            "pose_rows": str(pose_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
