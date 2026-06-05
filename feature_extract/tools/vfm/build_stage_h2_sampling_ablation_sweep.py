"""Build Stage H2 raw Gaussian maps for sampling ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
    _load_feature,
    _parse_default_camera,
    _select_records,
    _vote_stats,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_raw_landmarks import (
    RawGaussianFeatureAggregationConfig,
    VfmGaussianAnchorVoteConfig,
    _project_xyz_to_grid,
    aggregate_raw_vfm_features_to_gaussian_anchors,
    sample_gaussian_indices_from_votes,
    subset_gaussian_anchor_map_by_source_indices,
    vote_gaussians_from_vfm_token_saliency_configs,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, GaussianVFMSource, load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest


def _read_gray(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint voting") from exc
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _detect_keypoints(image_gray: np.ndarray, detector: str, max_keypoints: int) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for keypoint voting") from exc
    mode = str(detector).lower()
    if mode == "sift" and hasattr(cv2, "SIFT_create"):
        extractor = cv2.SIFT_create(nfeatures=int(max_keypoints))
    elif mode == "orb":
        extractor = cv2.ORB_create(nfeatures=int(max_keypoints))
    else:
        raise ValueError(f"unsupported detector or unavailable extractor: {detector}")
    keypoints = extractor.detect(image_gray, None)
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32)
    keypoints = sorted(keypoints, key=lambda item: float(item.response), reverse=True)[: int(max_keypoints)]
    return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)


def _vote_gaussians_from_image_keypoints(
    source: GaussianVFMSource,
    views: Sequence[GaussianVFMFeatureView],
    *,
    image_root: Path,
    detector: str,
    max_keypoints: int,
    vote_radius_px: float,
    max_views: int,
    min_opacity: float,
) -> tuple[np.ndarray, dict[str, object]]:
    votes = np.zeros((source.xyz.shape[0],), dtype=np.int64)
    rows = []
    selected_views = list(views)
    if int(max_views) > 0 and len(selected_views) > int(max_views):
        indices = np.linspace(0, len(selected_views) - 1, int(max_views), dtype=np.int64)
        selected_views = [selected_views[int(idx)] for idx in indices.tolist()]
    for view in selected_views:
        image = _read_gray(Path(image_root) / view.image_id)
        if image is None:
            rows.append({"image_id": view.image_id, "loaded": False, "keypoints": 0, "vote_hits": 0})
            continue
        keypoints = _detect_keypoints(image, detector=detector, max_keypoints=int(max_keypoints))
        if keypoints.shape[0] == 0:
            rows.append({"image_id": view.image_id, "loaded": True, "keypoints": 0, "vote_hits": 0})
            continue
        uv, depth = _project_xyz_to_grid(source.xyz, view.pose_w2c, view.camera, int(view.camera.width), int(view.camera.height))
        valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
        valid &= depth > 1e-8
        valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] < float(view.camera.width))
        valid &= (uv[:, 1] >= 0.0) & (uv[:, 1] < float(view.camera.height))
        if float(min_opacity) > 0.0:
            valid &= source.opacity >= float(min_opacity)
        projected_rows = np.flatnonzero(valid)
        if projected_rows.size == 0:
            rows.append({"image_id": view.image_id, "loaded": True, "keypoints": int(keypoints.shape[0]), "vote_hits": 0})
            continue
        tree = cKDTree(keypoints.astype(np.float64, copy=False))
        distances, _indices = tree.query(uv[projected_rows].astype(np.float64, copy=False), k=1)
        hit_rows = projected_rows[np.asarray(distances <= float(vote_radius_px), dtype=bool)]
        if hit_rows.size:
            votes[np.unique(hit_rows)] += 1
        rows.append(
            {
                "image_id": view.image_id,
                "loaded": True,
                "keypoints": int(keypoints.shape[0]),
                "projected_gaussians": int(projected_rows.size),
                "vote_hits": int(np.unique(hit_rows).size),
            }
        )
    return votes, {
        "stage": "image_keypoint_gaussian_votes",
        "detector": str(detector),
        "max_keypoints": int(max_keypoints),
        "vote_radius_px": float(vote_radius_px),
        "max_views": int(max_views),
        "view_count": int(len(rows)),
        "voted_gaussian_count": int(np.sum(votes > 0)),
        "max_votes": int(np.max(votes)) if votes.size else 0,
        "mean_keypoints": None if not rows else float(np.mean([row.get("keypoints", 0) for row in rows])),
        "mean_vote_hits": None if not rows else float(np.mean([row.get("vote_hits", 0) for row in rows])),
        "views": rows,
    }


def _default_saliency_configs(top5_opacity: float) -> dict[str, VfmGaussianAnchorVoteConfig]:
    return {
        "norm_top5": VfmGaussianAnchorVoteConfig(
            top_token_fraction=0.05,
            saliency_mode="norm",
            min_owner_opacity=float(top5_opacity),
        ),
        "norm_top10": VfmGaussianAnchorVoteConfig(
            top_token_fraction=0.10,
            saliency_mode="norm",
            min_owner_opacity=float(top5_opacity),
        ),
        "local_contrast_top5": VfmGaussianAnchorVoteConfig(
            top_token_fraction=0.05,
            saliency_mode="local_contrast",
            min_owner_opacity=float(top5_opacity),
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage H2 sampling ablation raw Gaussian maps")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--vote_min_owner_opacity", type=float, default=0.1)
    parser.add_argument("--max_anchors", type=int, default=20000)
    parser.add_argument("--min_votes", type=int, default=2)
    parser.add_argument("--sample_opacity_power", type=float, default=1.0)
    parser.add_argument("--sample_min_opacity", type=float, default=0.05)
    parser.add_argument("--nms_voxel_size", type=float, default=0.03)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--aggregation_owner_min_opacity", type=float, default=0.1)
    parser.add_argument("--no_l2_normalize_observations", action="store_true")
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--include_keypoint_vote", action="store_true")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--keypoint_detector", default="sift", choices=("sift", "orb"))
    parser.add_argument("--keypoint_max_views", type=int, default=0)
    parser.add_argument("--keypoint_max_keypoints", type=int, default=2048)
    parser.add_argument("--keypoint_vote_radius_px", type=float, default=4.0)
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args(argv)

    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_by_image = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    records = [record for record in manifest.records if record.image_id in pose_by_image]
    records = _select_records(records, int(args.max_views), args.view_selection)
    views = [
        GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=_load_feature(Path(record.token_path), args.layer_name),
            pose_w2c=pose_by_image[record.image_id].pose_w2c,
            camera=camera_by_image.get(record.image_id, fallback_camera),
        )
        for record in records
    ]
    if not views:
        raise ValueError("no reference views with both tokens and poses")
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=int(args.max_gaussians))

    saliency_configs = _default_saliency_configs(float(args.vote_min_owner_opacity))
    votes_by_name, multi_vote_summary = vote_gaussians_from_vfm_token_saliency_configs(source, views, saliency_configs)
    vote_summaries: dict[str, object] = {
        name: dict(multi_vote_summary["configs"][name])
        for name in saliency_configs
    }
    if args.include_keypoint_vote:
        if not args.image_root:
            raise ValueError("--image_root is required when --include_keypoint_vote is set")
        keypoint_votes, keypoint_summary = _vote_gaussians_from_image_keypoints(
            source,
            views,
            image_root=Path(args.image_root),
            detector=args.keypoint_detector,
            max_keypoints=int(args.keypoint_max_keypoints),
            vote_radius_px=float(args.keypoint_vote_radius_px),
            max_views=int(args.keypoint_max_views),
            min_opacity=float(args.sample_min_opacity),
        )
        keypoint_name = f"{args.keypoint_detector}_keypoint_vote"
        votes_by_name[keypoint_name] = keypoint_votes
        vote_summaries[keypoint_name] = keypoint_summary

    sampled_by_name = {}
    for name, votes in votes_by_name.items():
        sampled_by_name[name] = sample_gaussian_indices_from_votes(
            source,
            votes,
            max_anchors=int(args.max_anchors),
            min_votes=int(args.min_votes),
            nms_voxel_size=float(args.nms_voxel_size),
            opacity_power=float(args.sample_opacity_power),
            min_opacity=float(args.sample_min_opacity),
        )
    non_empty = [sampled for sampled in sampled_by_name.values() if sampled.size > 0]
    if not non_empty:
        raise ValueError("all sampling ablation configs produced empty sampled sets")
    union_sampled = np.unique(np.concatenate(non_empty).astype(np.int64, copy=False))

    aggregation_config = RawGaussianFeatureAggregationConfig(
        min_observations=int(args.min_observations),
        l2_normalize_observations=not bool(args.no_l2_normalize_observations),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
        require_token_owner_visibility=True,
        owner_min_opacity=float(args.aggregation_owner_min_opacity),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    union_anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        source,
        union_sampled,
        views,
        aggregation_config,
        metadata={
            "stage": "stage_h2_sampling_ablation_union",
            "sampled_gaussian_count": int(union_sampled.size),
            "sampling_ablation_names": sorted(sampled_by_name),
        },
    )
    union_anchor_map.save_npz(output_root / "union_raw_anchor_map.npz")
    np.savez_compressed(
        output_root / "votes.npz",
        **{f"votes_{name}": votes.astype(np.int64, copy=False) for name, votes in votes_by_name.items()},
        **{f"sampled_{name}": sampled.astype(np.int64, copy=False) for name, sampled in sampled_by_name.items()},
        union_sampled_indices=union_sampled.astype(np.int64, copy=False),
    )

    settings = []
    for name, sampled in sampled_by_name.items():
        setting_dir = output_root / name
        setting_dir.mkdir(parents=True, exist_ok=True)
        anchor_map = subset_gaussian_anchor_map_by_source_indices(
            union_anchor_map,
            source,
            sampled,
            metadata={
                "stage": "stage_h2_sampling_ablation_raw_gaussian_anchor_map",
                "sampling_name": str(name),
                "sampled_gaussian_count": int(sampled.size),
                "union_sampled_gaussian_count": int(union_sampled.size),
            },
        )
        map_path = setting_dir / "raw_anchor_map.npz"
        summary_path = setting_dir / "summary.json"
        anchor_map.save_npz(map_path)
        summary = {
            "stage": "stage_h2_sampling_ablation_setting",
            "sampling_name": str(name),
            "source_gaussian_count": int(source.xyz.shape[0]),
            "sampled_gaussian_count": int(sampled.size),
            "union_sampled_gaussian_count": int(union_sampled.size),
            "anchor_count": int(len(anchor_map)),
            "feature_dim": int(anchor_map.feature_dim),
            "view_count": int(len(views)),
            "min_votes": int(args.min_votes),
            "selected_vote_stats": _vote_stats(votes_by_name[name][sampled]),
            "vote_summary": vote_summaries[name],
            "sampling_config": {
                "max_anchors": int(args.max_anchors),
                "min_votes": int(args.min_votes),
                "nms_voxel_size": float(args.nms_voxel_size),
                "opacity_power": float(args.sample_opacity_power),
                "min_opacity": float(args.sample_min_opacity),
            },
            "aggregation_config": aggregation_config.to_dict(),
            "outputs": {
                "anchor_map": str(map_path),
                "summary": str(summary_path),
                "votes": str(output_root / "votes.npz"),
            },
            "missing_after_union_aggregation": int(anchor_map.metadata.get("missing_after_union_aggregation", 0)),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        settings.append(summary)

    availability = {
        "saliency_norm_top5": "run",
        "saliency_norm_top10": "run",
        "saliency_local_contrast_top5": "run",
        "keypoint_vote": "run" if args.include_keypoint_vote else "not_requested",
        "mask_filtered_saliency": "unavailable_no_per_view_masks_found",
        "render_contribution_vote": "unavailable_no_per_view_contribution_npz_found",
    }
    sweep_summary = {
        "stage": "stage_h2_sampling_ablation_sweep",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "union_sampled_gaussian_count": int(union_sampled.size),
        "union_anchor_count": int(len(union_anchor_map)),
        "feature_dim": int(union_anchor_map.feature_dim),
        "view_count": int(len(views)),
        "min_votes": int(args.min_votes),
        "availability": availability,
        "aggregation_config": aggregation_config.to_dict(),
        "settings": settings,
        "outputs": {
            "union_anchor_map": str(output_root / "union_raw_anchor_map.npz"),
            "votes": str(output_root / "votes.npz"),
        },
    }
    (output_root / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
