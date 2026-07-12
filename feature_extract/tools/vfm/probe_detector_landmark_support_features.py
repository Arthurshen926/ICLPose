"""Rerank detector landmark proposals with real multi-view ALIKE support features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    summarize_ranked_detector_proposal_geometry,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
    score_candidate_support_feature_pooling,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    extract_real_image_observation_features,
)
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--detector_proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--support_track_observations_jsonl", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument(
        "--full_support_feature_cache",
        default=None,
        help="optional validated full-observation ALIKE cache used to materialize the candidate subset",
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="write every reranking score without running the diagnostic strategy/PnP sweep",
    )
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--geometry_thresholds_px", default="1,2,5,8")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _alike_checkpoint(matcha_repo: Path, model_name: str) -> Path:
    suffix = str(model_name).split("-")[-1]
    path = Path(matcha_repo) / "third_party" / "alike" / "models" / f"alike-{suffix}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _image_manifest_hash(image_ids: Sequence[str], image_root: Path) -> str:
    lines = []
    for image_id in sorted(set(str(value) for value in image_ids)):
        path = Path(image_root) / image_id
        if not path.exists():
            raise FileNotFoundError(path)
        lines.append(f"{image_id}:{file_sha256_short(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf8")).hexdigest()[:16]


def _blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    output = np.full(lhs.shape, -np.inf, dtype=np.float32)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    output[valid] = float(left_weight) * lhs[valid] + (1.0 - float(left_weight)) * rhs[valid]
    return output


def _load_query_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    return payload, metadata


def _load_proposals(path: Path):
    with np.load(Path(path), allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    candidates = UniqueTrackCandidateSet(
        payload["bank_row_indices"],
        payload["candidate_track_ids"],
        payload["candidate_prototype_ids"],
        payload["coarse_scores"],
    )
    return payload, candidates


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = Path(args.detector_proposals)
    proposal_payload, candidates = _load_proposals(proposal_path)
    query_cache_path = Path(args.detector_query_cache)
    query_cache, query_metadata = _load_query_cache(query_cache_path)
    query_ids = np.asarray(proposal_payload["query_ids"]).astype(str)
    query_xy = np.asarray(proposal_payload["xy"], dtype=np.float32)
    if not np.array_equal(query_xy, np.asarray(query_cache["xy"], dtype=np.float32)):
        raise ValueError("detector proposal and query cache coordinates differ")
    cache_query_ids = np.repeat(
        np.asarray(query_cache["image_ids"]).astype(str),
        np.diff(np.asarray(query_cache["offsets"], dtype=np.int64)),
    )
    if not np.array_equal(query_ids, cache_query_ids):
        raise ValueError("detector proposal and query cache image rows differ")
    query_features = np.asarray(query_cache["local_descriptors"], dtype=np.float32)
    if query_features.shape[0] != candidates.query_count:
        raise ValueError("query local descriptor count differs from proposal rows")

    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    if str(query_metadata.get("descriptor_space_id", "")) != str(
        landmark_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("detector query cache and landmark bank descriptor spaces differ")
    valid_candidates = candidates.valid_mask
    if not np.array_equal(
        candidates.track_ids[valid_candidates],
        landmark_index.track_ids[candidates.bank_row_indices[valid_candidates]],
    ):
        raise ValueError("detector proposal rows are not aligned with landmark bank")

    candidate_track_ids = {
        int(value) for value in candidates.track_ids[valid_candidates].tolist()
    }
    support_path = Path(args.support_track_observations_jsonl)
    support_observations = load_colmap_track_observations_jsonl(
        support_path,
        track_ids=candidate_track_ids,
    )
    support_image_manifest = _image_manifest_hash(
        [observation.image_id for observation in support_observations],
        Path(args.image_root),
    )
    expected_cache = {
        "format": "detector_candidate_alike_support_features_v1",
        "detector_proposals": str(proposal_path),
        "detector_proposals_sha256": file_sha256_short(proposal_path),
        "support_track_observations_jsonl": str(support_path),
        "support_track_observations_sha256": file_sha256_short(support_path),
        "candidate_track_count": int(len(candidate_track_ids)),
        "support_observation_count": int(len(support_observations)),
        "support_source_image_manifest_sha256": support_image_manifest,
        "alike_model_name": str(args.alike_model_name),
        "alike_checkpoint_sha256": file_sha256_short(
            _alike_checkpoint(Path(args.matcha_repo), str(args.alike_model_name))
        ),
    }
    support_cache_path = Path(args.support_feature_cache)
    support_cache_hit = support_cache_path.exists()
    if support_cache_hit:
        with np.load(support_cache_path, allow_pickle=False) as data:
            support_metadata = json.loads(str(data["metadata_json"].item()))
            mismatches = {
                key: {"expected": value, "actual": support_metadata.get(key)}
                for key, value in expected_cache.items()
                if support_metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"stale ALIKE support cache: {json.dumps(mismatches, sort_keys=True)}")
            support_features = np.asarray(data["support_features"], dtype=np.float32)
            support_detector_scores = np.asarray(data["support_detector_scores"], dtype=np.float32)
            support_track_ids = np.asarray(data["support_track_ids"], dtype=np.int64)
    elif args.full_support_feature_cache:
        full_cache_path = Path(args.full_support_feature_cache)
        with np.load(full_cache_path, allow_pickle=False) as data:
            full_metadata = json.loads(str(data["metadata_json"].item()))
            full_track_ids = np.asarray(data["track_ids"], dtype=np.int64)
            full_features = np.asarray(data["descriptors"], dtype=np.float32)
            full_scores = np.asarray(data["detector_scores"], dtype=np.float32)
        expected_full = {
            "format": "full_support_alike_observation_features_v1",
            "support_track_observations_sha256": file_sha256_short(support_path),
            "model_checkpoint_sha256": file_sha256_short(
                _alike_checkpoint(Path(args.matcha_repo), str(args.alike_model_name))
            ),
        }
        full_mismatches = {
            key: {"expected": value, "actual": full_metadata.get(key)}
            for key, value in expected_full.items()
            if full_metadata.get(key) != value
        }
        if full_mismatches:
            raise ValueError(
                "stale full ALIKE support cache: "
                f"{json.dumps(full_mismatches, sort_keys=True)}"
            )
        keep = np.isin(
            full_track_ids,
            np.asarray(sorted(candidate_track_ids), dtype=np.int64),
            assume_unique=False,
        )
        support_track_ids = full_track_ids[keep]
        support_features = full_features[keep]
        support_detector_scores = full_scores[keep]
        observation_tracks = np.asarray(
            [observation.track_id for observation in support_observations],
            dtype=np.int64,
        )
        if not np.array_equal(support_track_ids, observation_tracks):
            raise ValueError(
                "full ALIKE support cache order differs from filtered support observations"
            )
        support_metadata = {
            **expected_cache,
            "feature_type": "alike_dense_descriptor",
            "feature_dimension": int(support_features.shape[1]),
            "cache_dtype": str(args.cache_dtype),
            "source_full_support_feature_cache": str(full_cache_path),
            "source_full_support_feature_cache_sha256": file_sha256_short(
                full_cache_path
            ),
            "materialization": "track_subset_from_full_support_cache",
        }
        cache_dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
        support_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            support_cache_path,
            support_features=np.asarray(support_features, dtype=cache_dtype),
            support_detector_scores=np.asarray(
                support_detector_scores, dtype=np.float32
            ),
            support_track_ids=support_track_ids,
            metadata_json=np.asarray(
                json.dumps(support_metadata, sort_keys=True), dtype=np.str_
            ),
        )
    else:
        devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
        extracted = extract_real_image_observation_features(
            support_observations,
            image_root=Path(args.image_root),
            devices=devices,
            feature_type="alike",
            alike_model_name=str(args.alike_model_name),
            matcha_repo=Path(args.matcha_repo),
            image_batch_size=1,
        )
        support_features = extracted.descriptors
        support_detector_scores = extracted.detector_scores
        support_track_ids = np.asarray(
            [observation.track_id for observation in support_observations],
            dtype=np.int64,
        )
        support_metadata = {**expected_cache, **dict(extracted.metadata), "cache_dtype": str(args.cache_dtype)}
        cache_dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
        support_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            support_cache_path,
            support_features=np.asarray(support_features, dtype=cache_dtype),
            support_detector_scores=np.asarray(support_detector_scores, dtype=np.float32),
            support_track_ids=support_track_ids,
            metadata_json=np.asarray(json.dumps(support_metadata, sort_keys=True), dtype=np.str_),
        )

    local_scores = score_candidate_support_feature_pooling(
        query_features=query_features,
        candidates=candidates,
        support_track_ids=support_track_ids,
        support_features=support_features,
        prefix="alike",
    )
    coarse = np.asarray(candidates.coarse_scores, dtype=np.float32)
    top4 = local_scores["alike_support_top4_mean"]
    strategies = {
        "coarse_prototype": coarse,
        **local_scores,
        "coarse_alike_top4_75_25": _blend(coarse, top4, 0.75),
        "coarse_alike_top4_50_50": _blend(coarse, top4, 0.50),
        "coarse_alike_top4_25_75": _blend(coarse, top4, 0.25),
    }
    pose: dict[str, object] = {}
    pose_rows: dict[str, object] = {}
    if bool(args.build_only):
        geometry: dict[str, object] = {}
    else:
        thresholds = tuple(
            float(value.strip())
            for value in str(args.geometry_thresholds_px).split(",")
            if value.strip()
        )
        candidate_residuals = np.asarray(
            proposal_payload["candidate_gt_residuals_px"], dtype=np.float32
        )
        nearest_residuals = np.asarray(
            proposal_payload["nearest_visible_residuals_px"], dtype=np.float32
        )
        geometry = {
            name: summarize_ranked_detector_proposal_geometry(
                nearest_landmark_residuals=nearest_residuals,
                candidate_residuals=candidate_residuals,
                candidate_scores=scores,
                query_ids=query_ids,
                thresholds_px=thresholds,
                top_ls=(1, 5, 10, candidates.top_l),
            )
            for name, scores in strategies.items()
        }

        model_dir = Path(args.colmap_model_dir)
        cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
        images = read_colmap_images_binary(model_dir / "images.bin")
        images_by_name = {str(image.image_name): image for image in images.values()}
        nearest_tracks = np.asarray(
            proposal_payload["nearest_visible_track_ids"], dtype=np.int64
        )
        observations = [
            ColmapTrackObservation(
                track_id=int(nearest_tracks[row]),
                image_id=str(query_ids[row]),
                point2d_idx=int(row),
                xy=(float(query_xy[row, 0]), float(query_xy[row, 1])),
                xyz=np.zeros((3,), dtype=np.float64),
                track_length=1,
                reprojection_error=0.0,
            )
            for row in range(len(query_xy))
        ]
        pose_keep = np.asarray(proposal_payload["pose_keep_mask"], dtype=bool)
        for name, scores in strategies.items():
            pose_scores = np.asarray(scores, dtype=np.float32).copy()
            pose_scores[~pose_keep] = -np.inf
            pose[name], pose_rows[name] = _evaluate_pose_strategy(
                strategy=name,
                scores=pose_scores,
                candidates=candidates,
                query_observations=observations,
                query_ids=query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )

    unique_support_tracks, support_counts = np.unique(support_track_ids, return_counts=True)
    positions = np.searchsorted(unique_support_tracks, candidates.track_ids)
    support_count_matrix = np.zeros(candidates.track_ids.shape, dtype=np.int32)
    valid_positions = candidates.valid_mask & (positions < len(unique_support_tracks))
    valid_positions &= unique_support_tracks[np.minimum(positions, len(unique_support_tracks) - 1)] == candidates.track_ids
    support_count_matrix[valid_positions] = support_counts[positions[valid_positions]].astype(np.int32)
    extended_path = output_dir / "detector_support_reranked_proposals.npz"
    np.savez(
        extended_path,
        **proposal_payload,
        support_observation_counts=support_count_matrix,
        **{f"strategy__{name}": values.astype(np.float32) for name, values in strategies.items()},
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "s4_l2_detector_real_support_feature_probe",
        "protocol": {
            "query_point_source": "alike_subpixel_detector_real_rgb",
            "support_source": "real_train_images_excluding_all_query_images",
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "proposal_pool_frozen": True,
            "build_only": bool(args.build_only),
        },
        "support_cache": {
            "cache_hit": bool(support_cache_hit),
            "path": str(support_cache_path),
            "sha256": file_sha256_short(support_cache_path),
            "metadata": support_metadata,
        },
        "geometry": geometry,
        "pose": pose,
        "runtime_seconds": float(time.time() - start_time),
        "limitations": [
            "support pooling scores each detector proposal independently and does not yet enforce partial assignment",
            "all strategies retain the same global top20 pool, so missing positives cannot be recovered",
            "pose uses detector coordinates without a learned local measurement update",
            *( ["strategy geometry and PnP audits were skipped in build-only mode"] if bool(args.build_only) else [] ),
        ],
        "outputs": {
            "support_feature_cache": str(support_cache_path),
            "extended_proposals": str(extended_path),
            "pose_rows": str(pose_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
