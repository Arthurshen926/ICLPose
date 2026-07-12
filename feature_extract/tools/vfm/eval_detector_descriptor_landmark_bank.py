"""Evaluate detector descriptors against a same-space full landmark bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation, qvec_to_rotmat, read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
    nearest_visible_landmarks,
    summarize_detector_proposal_geometry,
)
from feature_extract.vfm.localization.global_landmark_ann import (
    FaissIVFConfig,
    build_or_load_faiss_ivf_index,
    collapse_unique_track_search,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.query_to_3d_matching import normalize_rows


def _int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or any(item <= 0 for item in output):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector_query_cache", required=True)
    parser.add_argument("--detector_base_proposals", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--faiss_index_cache", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_descriptor_key", default="local_descriptors")
    parser.add_argument("--proposal_top_l", type=int, default=20)
    parser.add_argument("--faiss_nlist", type=int, default=1024)
    parser.add_argument("--faiss_train_samples", type=int, default=100000)
    parser.add_argument("--faiss_nprobe_values", type=_int_list, default=(16, 32, 64, 128))
    parser.add_argument("--faiss_oversample_factor", type=int, default=4)
    parser.add_argument("--exact_audit_query_count", type=int, default=1024)
    parser.add_argument("--ann_min_top1_agreement", type=float, default=0.99)
    parser.add_argument("--ann_min_top_l_overlap", type=float, default=0.99)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _exact_unique_tracks(query_features, landmark_index, *, top_l: int, oversample_factor: int):
    try:
        import faiss
    except Exception as exc:
        raise RuntimeError("FAISS is required for exact audit") from exc
    features, valid = normalize_rows(landmark_index.features)
    if not np.all(valid):
        raise ValueError("landmark bank contains invalid descriptors")
    queries, valid_query = normalize_rows(query_features)
    if not np.all(valid_query):
        raise ValueError("query audit contains invalid descriptors")
    index = faiss.IndexFlatIP(int(features.shape[1]))
    index.add(np.ascontiguousarray(features.astype(np.float32)))
    search_k = min(len(landmark_index), int(top_l) * int(oversample_factor))
    scores, rows = index.search(np.ascontiguousarray(queries.astype(np.float32)), search_k)
    return collapse_unique_track_search(
        rows,
        scores,
        landmark_index.track_ids,
        landmark_index.prototype_ids,
        proposal_top_l=int(top_l),
    )


def _ann_overlap(ann_tracks: np.ndarray, exact_tracks: np.ndarray) -> dict[str, float]:
    ann = np.asarray(ann_tracks, dtype=np.int64)
    exact = np.asarray(exact_tracks, dtype=np.int64)
    overlaps = []
    for left, right in zip(ann, exact):
        left_set = {int(value) for value in left if int(value) >= 0}
        right_set = {int(value) for value in right if int(value) >= 0}
        overlaps.append(len(left_set & right_set) / max(len(right_set), 1))
    return {
        "top1_track_agreement": float(np.mean(ann[:, 0] == exact[:, 0])),
        "mean_exact_top_l_track_overlap": float(np.mean(overlaps)),
    }


def _pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.detector_query_cache), allow_pickle=False) as data:
        query_metadata = json.loads(str(data["metadata_json"].item()))
        query_features = np.asarray(data[str(args.query_descriptor_key)], dtype=np.float32)
        query_xy = np.asarray(data["xy"], dtype=np.float32)
        image_ids = np.asarray(data["image_ids"]).astype(str)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        detector_scores = np.asarray(data["detector_scores"], dtype=np.float32)
        detector_dispersions = np.asarray(data["detector_dispersions"], dtype=np.float32)
    query_ids = np.repeat(image_ids, np.diff(offsets))
    with np.load(Path(args.detector_base_proposals), allow_pickle=False) as data:
        base_query_ids = np.asarray(data["query_ids"]).astype(str)
        base_xy = np.asarray(data["xy"], dtype=np.float32)
        pose_keep = np.asarray(data["pose_keep_mask"], dtype=bool)
    if not np.array_equal(query_ids, base_query_ids) or not np.array_equal(query_xy, base_xy):
        raise ValueError("detector cache and base proposal rows differ")
    landmark_index, bank_metadata = load_landmark_index_npz(Path(args.landmark_bank))
    bank_manifest = dict(bank_metadata.get("descriptor_space_manifest") or {})
    if int(query_features.shape[1]) != int(landmark_index.feature_dim):
        raise ValueError("query and landmark descriptor dimensions differ")
    detector_metadata = dict(query_metadata.get("detector") or {})
    if str(detector_metadata.get("model_checkpoint_sha256", "")) != str(
        bank_manifest.get("alike_checkpoint_sha256", "")
    ):
        raise ValueError("query and landmark ALIKE checkpoints differ")

    faiss_index, faiss_cache_hit = build_or_load_faiss_ivf_index(
        landmark_index,
        landmark_bank_path=Path(args.landmark_bank),
        descriptor_space_id=str(bank_metadata.get("descriptor_space_id", "")),
        cache_path=Path(args.faiss_index_cache),
        config=FaissIVFConfig(
            nlist=int(args.faiss_nlist),
            train_samples=int(args.faiss_train_samples),
            seed=0,
        ),
    )
    audit_count = min(int(args.exact_audit_query_count), len(query_features))
    audit_rows = np.linspace(0, len(query_features) - 1, audit_count, dtype=np.int64)
    _exact_rows, exact_tracks, _exact_prototypes, _exact_scores = _exact_unique_tracks(
        query_features[audit_rows],
        landmark_index,
        top_l=int(args.proposal_top_l),
        oversample_factor=int(args.faiss_oversample_factor),
    )
    trials = []
    for nprobe in args.faiss_nprobe_values:
        ann = faiss_index.search_unique_tracks(
            query_features[audit_rows],
            landmark_index,
            proposal_top_l=int(args.proposal_top_l),
            nprobe=int(nprobe),
            oversample_factor=int(args.faiss_oversample_factor),
        )
        metrics = _ann_overlap(ann.track_ids, exact_tracks)
        trials.append(
            {
                "nprobe": int(nprobe),
                **metrics,
                "passes_gate": (
                    metrics["top1_track_agreement"] >= float(args.ann_min_top1_agreement)
                    and metrics["mean_exact_top_l_track_overlap"] >= float(args.ann_min_top_l_overlap)
                ),
            }
        )
    passing = [index for index, trial in enumerate(trials) if bool(trial["passes_gate"])]
    chosen_index = passing[0] if passing else len(trials) - 1
    chosen_nprobe = int(trials[chosen_index]["nprobe"])
    result = faiss_index.search_unique_tracks(
        query_features,
        landmark_index,
        proposal_top_l=int(args.proposal_top_l),
        nprobe=chosen_nprobe,
        oversample_factor=int(args.faiss_oversample_factor),
    )
    candidates = UniqueTrackCandidateSet(
        result.bank_row_indices,
        result.track_ids,
        result.prototype_ids,
        result.scores,
    )

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    nearest_rows = np.full((len(query_xy),), -1, dtype=np.int64)
    nearest_tracks = np.full((len(query_xy),), -1, dtype=np.int64)
    nearest_residuals = np.full((len(query_xy),), np.inf, dtype=np.float32)
    candidate_residuals = np.full(candidates.bank_row_indices.shape, np.inf, dtype=np.float32)
    for image_index, query_id in enumerate(image_ids):
        start, end = int(offsets[image_index]), int(offsets[image_index + 1])
        image = images_by_name[str(query_id)]
        camera = cameras[int(image.camera_id)]
        pose = _pose_w2c(image)
        rows, tracks, distances = nearest_visible_landmarks(
            query_xy[start:end], landmark_index, pose, camera
        )
        nearest_rows[start:end] = rows
        nearest_tracks[start:end] = tracks
        nearest_residuals[start:end] = distances
        candidate_residuals[start:end] = candidate_reprojection_residuals(
            query_xy[start:end], candidates.bank_row_indices[start:end], landmark_index, pose, camera
        )
    geometry = summarize_detector_proposal_geometry(
        nearest_landmark_residuals=nearest_residuals,
        candidate_residuals=candidate_residuals,
        query_ids=query_ids,
        thresholds_px=(1.0, 2.0, 5.0, 8.0),
        top_ls=(1, 5, 10, int(args.proposal_top_l)),
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
    pose = {}
    pose_rows = {}
    top1_scores = candidates.coarse_scores.copy()
    top1_scores[~pose_keep] = -np.inf
    pose["descriptor_top1"], pose_rows["descriptor_top1"] = _evaluate_pose_strategy(
        strategy="detector_descriptor_top1",
        scores=top1_scores,
        candidates=candidates,
        query_observations=observations,
        query_ids=query_ids,
        landmark_index=landmark_index,
        cameras=cameras,
        images_by_name=images_by_name,
        reprojection_error_px=float(args.pnp_reprojection_error_px),
        iterations=int(args.pnp_iterations),
    )
    for threshold in (2.0, 5.0):
        oracle_scores = -candidate_residuals
        oracle_scores[(candidate_residuals > threshold) | (~pose_keep[:, None])] = -np.inf
        name = f"proposal_geometry_oracle_{threshold:g}px"
        pose[name], pose_rows[name] = _evaluate_pose_strategy(
            strategy=name,
            scores=oracle_scores,
            candidates=candidates,
            query_observations=observations,
            query_ids=query_ids,
            landmark_index=landmark_index,
            cameras=cameras,
            images_by_name=images_by_name,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
    proposal_path = output_dir / "detector_descriptor_proposals.npz"
    np.savez(
        proposal_path,
        query_ids=query_ids,
        xy=query_xy,
        detector_scores=detector_scores,
        detector_dispersions=detector_dispersions,
        bank_row_indices=candidates.bank_row_indices,
        candidate_track_ids=candidates.track_ids,
        candidate_prototype_ids=candidates.prototype_ids,
        coarse_scores=candidates.coarse_scores,
        nearest_visible_bank_rows=nearest_rows,
        nearest_visible_track_ids=nearest_tracks,
        nearest_visible_residuals_px=nearest_residuals,
        candidate_gt_residuals_px=candidate_residuals,
        pose_keep_mask=pose_keep,
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "detector_same_space_global_landmark_retrieval",
        "protocol": {
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "query_descriptor_key": str(args.query_descriptor_key),
            "proposal_top_l": int(args.proposal_top_l),
        },
        "descriptor_space_id": bank_metadata.get("descriptor_space_id"),
        "faiss": {
            "cache_hit": bool(faiss_cache_hit),
            "exact_audit_query_count": int(audit_count),
            "trials": trials,
            "chosen_nprobe": int(chosen_nprobe),
            "gate_passed": bool(trials[chosen_index]["passes_gate"]),
        },
        "geometry": geometry,
        "pose": pose,
        "runtime_seconds": float(time.time() - start_time),
        "outputs": {
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
