"""Build raw high-dimensional VFM features on sampled Gaussian anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera, read_colmap_cameras_binary, read_colmap_images_binary
from feature_extract.vfm.gaussian_raw_landmarks import (
    GaussianTokenContributionView,
    RawGaussianFeatureAggregationConfig,
    VfmGaussianAnchorVoteConfig,
    aggregate_raw_vfm_features_from_token_contributions,
    aggregate_raw_vfm_features_from_contribution_visibility,
    aggregate_raw_vfm_features_to_gaussian_anchors,
    sample_gaussian_indices_from_votes,
    vote_gaussians_from_token_contribution_views,
    vote_gaussians_from_vfm_token_saliency,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_default_camera(text: str) -> ColmapCamera:
    values = [float(item) for item in text.split(",") if item.strip()]
    if len(values) < 6:
        raise ValueError("--default_camera must be 'model_id,width,height,param0,param1,...'")
    return ColmapCamera(
        camera_id=-1,
        model_id=int(values[0]),
        width=int(values[1]),
        height=int(values[2]),
        params=tuple(float(item) for item in values[3:]),
    )


def _load_camera_by_image(model_dir: str) -> dict[str, ColmapCamera]:
    if not model_dir:
        return {}
    model_path = Path(model_dir)
    cameras = read_colmap_cameras_binary(model_path / "cameras.bin")
    images = read_colmap_images_binary(model_path / "images.bin")
    result = {
        image.image_name: cameras[image.camera_id]
        for image in images.values()
        if image.camera_id in cameras
    }
    # Strict MAtCha datasets flatten ``seqN/frame.png`` to
    # ``seqN__frame.png`` because their RGB directory is single-level.  Keep
    # the original Cambridge ID as a deterministic alias for downstream VFM
    # manifests; no pose or image content is duplicated here.
    for name, camera in list(result.items()):
        if "/" not in name and "__" in name and name.startswith("seq"):
            route, image = name.split("__", 1)
            result.setdefault(f"{route}/{image}", camera)
    return result


def _load_feature(path: Path, layer_name: str) -> np.ndarray:
    with np.load(Path(path)) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        return np.asarray(data[layer_name], dtype=np.float32)


def _safe_image_stem(image_id: str) -> str:
    stem = str(image_id).replace("\\", "/").strip("/")
    stem = stem.replace("/", "__")
    return stem.replace(" ", "_")


def _find_contribution_npz(root: Path, image_id: str) -> Path | None:
    image_path = Path(str(image_id))
    candidates = [
        root / f"{image_id}.npz",
        root / image_path.with_suffix(".npz"),
        root / f"{_safe_image_stem(image_id)}.npz",
        root / f"{_safe_image_stem(str(image_path.with_suffix('')))}.npz",
        root / f"{image_path.name}.npz",
        root / f"{image_path.stem}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_contribution_view(
    path: Path,
    image_id: str,
    feature_map: np.ndarray,
    top_contributor_key: str,
    top_alpha_key: str,
    alpha_entropy_key: str,
) -> GaussianTokenContributionView:
    with np.load(path) as data:
        if top_contributor_key not in data:
            raise ValueError(f"{top_contributor_key!r} not found in contribution map {path}")
        top_alpha = np.asarray(data[top_alpha_key], dtype=np.float32) if top_alpha_key and top_alpha_key in data else None
        alpha_entropy = (
            np.asarray(data[alpha_entropy_key], dtype=np.float32)
            if alpha_entropy_key and alpha_entropy_key in data
            else None
        )
        return GaussianTokenContributionView(
            image_id=str(image_id),
            feature_map=np.asarray(feature_map, dtype=np.float32),
            top_contributor=np.asarray(data[top_contributor_key], dtype=np.int64),
            top_alpha=top_alpha,
            alpha_entropy=alpha_entropy,
        )


def _select_records(records: list, max_views: int, mode: str) -> list:
    if int(max_views) <= 0 or len(records) <= int(max_views):
        return records
    if mode == "prefix":
        return records[: int(max_views)]
    if mode == "uniform":
        indices = np.linspace(0, len(records) - 1, int(max_views), dtype=np.int64)
        return [records[int(idx)] for idx in indices.tolist()]
    raise ValueError("view_selection must be prefix or uniform")


def _vote_stats(votes: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(votes, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0,
            "vote_eq1_fraction": 0.0,
            "vote_le2_fraction": 0.0,
        }
    return {
        "count": int(values.size),
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": int(np.max(values)),
        "vote_eq1_fraction": float(np.mean(values == 1.0)),
        "vote_le2_fraction": float(np.mean(values <= 2.0)),
    }


def _numeric_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _source_geometry_stats(source, source_rows: np.ndarray) -> dict[str, object]:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    rows = rows[(rows >= 0) & (rows < int(source.xyz.shape[0]))]
    if rows.size == 0:
        return {
            "count": 0,
            "opacity": _numeric_stats(np.asarray([], dtype=np.float32)),
            "scale": _numeric_stats(np.asarray([], dtype=np.float32)),
            "anisotropy_ratio": _numeric_stats(np.asarray([], dtype=np.float32)),
            "normal_available": False,
            "normal_available_fraction": 0.0,
        }
    scale_xyz = np.asarray(source.scale_xyz, dtype=np.float32)[rows]
    anisotropy = np.max(scale_xyz, axis=1) / np.maximum(np.min(scale_xyz, axis=1), 1e-8)
    normal = getattr(source, "normal", None)
    normal_available_fraction = 0.0
    if normal is not None:
        normal_rows = np.asarray(normal, dtype=np.float32)[rows]
        normal_available_fraction = float(np.mean(np.linalg.norm(normal_rows, axis=1) > 0.5))
    return {
        "count": int(rows.size),
        "opacity": _numeric_stats(np.asarray(source.opacity, dtype=np.float32)[rows]),
        "scale": _numeric_stats(np.asarray(source.scale, dtype=np.float32)[rows]),
        "anisotropy_ratio": _numeric_stats(anisotropy),
        "normal_available": bool(normal_available_fraction > 0.0),
        "normal_available_fraction": normal_available_fraction,
    }


def _source_rows_for_gaussian_indices(source, gaussian_indices: np.ndarray) -> np.ndarray:
    row_by_gaussian_index = {
        int(gaussian_index): int(row)
        for row, gaussian_index in enumerate(np.asarray(source.gaussian_indices, dtype=np.int64).tolist())
    }
    rows = [
        row_by_gaussian_index[int(gaussian_index)]
        for gaussian_index in np.asarray(gaussian_indices, dtype=np.int64).reshape(-1).tolist()
        if int(gaussian_index) in row_by_gaussian_index
    ]
    return np.asarray(rows, dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage H2 raw VFM Gaussian anchor map")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--vote_top_token_fraction", type=float, default=0.05)
    parser.add_argument("--vote_min_saliency", type=float, default=0.0)
    parser.add_argument("--vote_saliency_mode", default="norm", choices=("norm", "local_contrast"))
    parser.add_argument("--vote_min_owner_opacity", type=float, default=0.0)
    parser.add_argument("--max_anchors", type=int, default=10000)
    parser.add_argument("--min_votes", type=int, default=3)
    parser.add_argument("--sample_opacity_power", type=float, default=0.0)
    parser.add_argument("--sample_min_opacity", type=float, default=0.0)
    parser.add_argument("--nms_voxel_size", type=float, default=0.03)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--require_token_owner_visibility", action="store_true")
    parser.add_argument("--disable_token_owner_visibility", action="store_true")
    parser.add_argument("--aggregation_owner_min_opacity", type=float, default=None)
    parser.add_argument("--contribution_map_dir", default="")
    parser.add_argument("--contribution_top_contributor_key", default="top_contributor")
    parser.add_argument("--contribution_top_alpha_key", default="top_alpha")
    parser.add_argument("--contribution_alpha_entropy_key", default="alpha_entropy")
    parser.add_argument("--use_contribution_votes", action="store_true")
    parser.add_argument("--contribution_visibility_center_sampling", action="store_true")
    parser.add_argument("--min_contribution_alpha", type=float, default=0.0)
    parser.add_argument("--max_contribution_entropy", type=float, default=None)
    parser.add_argument("--no_l2_normalize_observations", action="store_true")
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--votes_npz", default="")
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
    vote_config = VfmGaussianAnchorVoteConfig(
        top_token_fraction=float(args.vote_top_token_fraction),
        min_saliency=float(args.vote_min_saliency),
        saliency_mode=args.vote_saliency_mode,
        min_owner_opacity=float(args.vote_min_owner_opacity),
    )
    aggregation_config = RawGaussianFeatureAggregationConfig(
        min_observations=int(args.min_observations),
        l2_normalize_observations=not bool(args.no_l2_normalize_observations),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
        require_token_owner_visibility=bool(args.require_token_owner_visibility) or not bool(args.disable_token_owner_visibility),
        owner_min_opacity=(
            float(args.aggregation_owner_min_opacity)
            if args.aggregation_owner_min_opacity is not None
            else float(args.vote_min_owner_opacity)
        ),
        min_contribution_alpha=float(args.min_contribution_alpha),
        max_contribution_entropy=(
            None if args.max_contribution_entropy is None else float(args.max_contribution_entropy)
        ),
    )
    contribution_summary: dict[str, object] | None = None
    contribution_views: list[GaussianTokenContributionView] = []
    missing_contribution_maps: list[str] = []
    if args.contribution_map_dir:
        contribution_root = Path(args.contribution_map_dir)
        for view in views:
            contribution_path = _find_contribution_npz(contribution_root, view.image_id)
            if contribution_path is None:
                missing_contribution_maps.append(str(view.image_id))
                continue
            contribution_views.append(
                _load_contribution_view(
                    contribution_path,
                    view.image_id,
                    view.feature_map,
                    top_contributor_key=args.contribution_top_contributor_key,
                    top_alpha_key=args.contribution_top_alpha_key,
                    alpha_entropy_key=args.contribution_alpha_entropy_key,
                )
            )
        if not contribution_views:
            raise ValueError("no contribution maps were found for selected reference views")
        contribution_summary = {
            "mode": "render_contribution",
            "contribution_map_dir": str(contribution_root),
            "loaded_view_count": int(len(contribution_views)),
            "missing_view_count": int(len(missing_contribution_maps)),
            "missing_views_preview": missing_contribution_maps[:10],
            "top_contributor_key": str(args.contribution_top_contributor_key),
            "top_alpha_key": str(args.contribution_top_alpha_key),
            "alpha_entropy_key": str(args.contribution_alpha_entropy_key),
        }
    if bool(args.use_contribution_votes):
        if not contribution_views:
            raise ValueError("--use_contribution_votes requires --contribution_map_dir with at least one map")
        votes, vote_summary = vote_gaussians_from_token_contribution_views(
            source_gaussian_count=int(source.xyz.shape[0]),
            contribution_views=contribution_views,
            vote_config=vote_config,
            min_contribution_alpha=float(args.min_contribution_alpha),
            max_contribution_entropy=(
                None if args.max_contribution_entropy is None else float(args.max_contribution_entropy)
            ),
        )
    else:
        votes, vote_summary = vote_gaussians_from_vfm_token_saliency(source, views, vote_config)
    sampled = sample_gaussian_indices_from_votes(
        source,
        votes,
        max_anchors=int(args.max_anchors),
        min_votes=int(args.min_votes),
        nms_voxel_size=float(args.nms_voxel_size),
        opacity_power=float(args.sample_opacity_power),
        min_opacity=float(args.sample_min_opacity),
    )
    if contribution_views:
        if bool(args.contribution_visibility_center_sampling):
            anchor_map = aggregate_raw_vfm_features_from_contribution_visibility(
                source,
                sampled,
                contribution_views,
                views,
                aggregation_config,
                metadata={
                    "vote_summary": vote_summary,
                    "sampled_gaussian_count": int(sampled.size),
                    "contribution_summary": contribution_summary,
                    "contribution_aggregation_mode": "center_sampling_visibility_gate",
                },
            )
        else:
            anchor_map = aggregate_raw_vfm_features_from_token_contributions(
                source,
                sampled,
                contribution_views,
                aggregation_config,
                metadata={
                    "vote_summary": vote_summary,
                    "sampled_gaussian_count": int(sampled.size),
                    "contribution_summary": contribution_summary,
                    "contribution_aggregation_mode": "token_top_contributor_average",
                },
            )
    else:
        contribution_summary = {"mode": "owner_projection_fallback"}
        anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
            source,
            sampled,
            views,
            aggregation_config,
            metadata={
                "vote_summary": vote_summary,
                "sampled_gaussian_count": int(sampled.size),
                "contribution_summary": contribution_summary,
            },
        )
    anchor_map.save_npz(Path(args.output_npz))
    if args.votes_npz:
        Path(args.votes_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.votes_npz, vote_counts=votes.astype(np.int64, copy=False), sampled_indices=sampled)
    summary = {
        "stage": "stage_h2_raw_gaussian_anchor_map",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "sampled_gaussian_count": int(sampled.size),
        "anchor_count": int(len(anchor_map)),
        "feature_dim": int(anchor_map.feature_dim),
        "view_count": int(len(views)),
        "vote_config": vote_config.to_dict(),
        "sampling_config": {
            "max_anchors": int(args.max_anchors),
            "min_votes": int(args.min_votes),
            "nms_voxel_size": float(args.nms_voxel_size),
            "opacity_power": float(args.sample_opacity_power),
            "min_opacity": float(args.sample_min_opacity),
        },
        "selected_vote_stats": _vote_stats(votes[sampled]),
        "vote_source": "render_contribution" if bool(args.use_contribution_votes) else "owner_projection",
        "source_geometry_stats": _source_geometry_stats(source, sampled),
        "anchor_geometry_stats": _source_geometry_stats(
            source,
            _source_rows_for_gaussian_indices(source, anchor_map.source_gaussian_indices),
        ),
        "aggregation_config": aggregation_config.to_dict(),
        "contribution_summary": contribution_summary,
        "vote_summary": vote_summary,
        "inputs": {
            "gaussian_ply": args.gaussian_ply,
            "reference_manifest": args.reference_manifest,
            "reference_pose_file": args.reference_pose_file,
            "camera_model_dir": args.camera_model_dir,
            "layer_name": args.layer_name,
            "view_selection": args.view_selection,
        },
        "outputs": {"anchor_map": args.output_npz, "votes": args.votes_npz},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
