"""Evaluate the local basin of the runtime RADIO-final maplet atlas.

Ground truth chooses visible chart identities and creates a bounded pose
perturbation.  Everything after initialization is the production
render/correlate/SE(3)/held-out-verification path.  This is a Stage-C component
diagnostic, not end-to-end localization and not a deployable use of GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _select_oracle_charts,
    _visible_charts,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.atlas_pose_alignment import (
    AtlasAlignmentLevel,
    _translation_search_step,
    build_radio_final_feature_pyramid,
    refine_pose_with_maplet_atlases,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    load_surface_spatial_projection,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


FEATURE_LEVEL = "coarse"
FEATURE_STRIDE = 16


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument(
        "--frame_spatial_projection_checkpoint", default=""
    )
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument(
        "--spatial_query_token_dir",
        default="",
        help=(
            "Optional RADIO-final token directory used only by atlas "
            "alignment; filenames are image IDs with '/' replaced by '__'."
        ),
    )
    parser.add_argument(
        "--spatial_feature_stride",
        type=int,
        choices=(8, 16),
        default=16,
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument(
        "--image_id",
        default="",
        help="Optional single-query component diagnostic.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_m", type=float, default=0.05)
    parser.add_argument("--rotation_deg", type=float, default=1.0 / 3.0)
    parser.add_argument("--visible_charts", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2907)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    resolved = float(value)
    return resolved if np.isfinite(resolved) else None


def _trajectory_ids_from_mapper(
    metadata: Mapping[str, object],
) -> set[str]:
    result = set()
    for key in ("training_images", "validation_images"):
        for image_id in metadata.get(key, []):
            result.add(str(image_id).split("/", 1)[0])
    return result


def _perturbation(
    image_id: str,
    *,
    translation_m: float,
    rotation_deg: float,
    seed: int,
) -> np.ndarray:
    digest = hashlib.sha256(
        f"{int(seed)}:{image_id}".encode("utf-8")
    ).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(local_seed)
    translation_axis = rng.normal(size=3)
    translation_axis /= max(float(np.linalg.norm(translation_axis)), 1e-8)
    rotation_axis = rng.normal(size=3)
    rotation_axis /= max(float(np.linalg.norm(rotation_axis)), 1e-8)
    return np.r_[
        rotation_axis * np.deg2rad(float(rotation_deg)),
        translation_axis * float(translation_m),
    ]


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    initial_t = np.asarray(
        [float(row["initial_translation_m"]) for row in rows],
        dtype=np.float64,
    )
    final_t = np.asarray(
        [float(row["final_translation_m"]) for row in rows],
        dtype=np.float64,
    )
    final_r = np.asarray(
        [float(row["final_rotation_deg"]) for row in rows],
        dtype=np.float64,
    )
    flow = np.asarray(
        [float(row["initial_pixel_flow_median_px"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "query_count": len(rows),
        "accepted_update_fraction": (
            float(
                np.mean(
                    [
                        int(row["accepted_step_count"]) > 0
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
        ),
        "translation_improved_fraction": (
            # A strict floating-point comparison previously counted unchanged
            # 5/20 cm poses as improvements at ~1e-15 m.  Require a meaningful
            # one-millimetre reduction for this basin diagnostic.
            float(np.mean(final_t < initial_t - 1e-3)) if rows else 0.0
        ),
        "success_4cm_1deg": (
            float(np.mean((final_t <= 0.04) & (final_r <= 1.0)))
            if rows
            else 0.0
        ),
        "success_20cm_3deg": (
            float(np.mean((final_t <= 0.20) & (final_r <= 3.0)))
            if rows
            else 0.0
        ),
        "success_30cm_3deg": (
            float(np.mean((final_t <= 0.30) & (final_r <= 3.0)))
            if rows
            else 0.0
        ),
        "final_translation_median_m": (
            float(np.median(final_t)) if rows else None
        ),
        "final_translation_p90_m": (
            float(np.quantile(final_t, 0.90)) if rows else None
        ),
        "final_rotation_median_deg": (
            float(np.median(final_r)) if rows else None
        ),
        "final_rotation_p90_deg": (
            float(np.quantile(final_r, 0.90)) if rows else None
        ),
        "initial_pixel_flow_median_px": (
            float(np.median(flow)) if rows else None
        ),
        "initial_pixel_flow_p90_px": (
            float(np.quantile(flow, 0.90)) if rows else None
        ),
    }


def _initial_flow_statistics(
    atlas: MapletFeatureAtlasBank,
    selected_ids: np.ndarray,
    initial_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    camera: object,
    *,
    maximum_points: int = 4096,
) -> dict[str, object]:
    row_by_id = {
        int(value): row for row, value in enumerate(atlas.maplet_ids.tolist())
    }
    rows = np.asarray(
        [row_by_id[int(value)] for value in selected_ids if int(value) in row_by_id],
        dtype=np.int64,
    )
    if rows.size == 0:
        return {
            "initial_pixel_flow_sample_count": 0,
            "initial_pixel_flow_median_px": 0.0,
            "initial_pixel_flow_p90_px": 0.0,
            "selected_surface_depth_median_m": 0.0,
        }
    xyz = np.asarray(atlas.xyz[rows], dtype=np.float64)
    valid = np.asarray(atlas.valid_mask[rows], dtype=bool)
    points = xyz[valid]
    if points.shape[0] > int(maximum_points):
        indices = np.linspace(
            0, points.shape[0] - 1, int(maximum_points), dtype=np.int64
        )
        points = points[indices]
    target_xy, target_depth = project_world_points(
        points, target_pose_w2c, camera
    )
    initial_xy, initial_depth = project_world_points(
        points, initial_pose_w2c, camera
    )
    keep = (
        np.isfinite(target_xy).all(axis=1)
        & np.isfinite(initial_xy).all(axis=1)
        & np.isfinite(target_depth)
        & np.isfinite(initial_depth)
        & (target_depth > 0.10)
        & (initial_depth > 0.10)
        & (target_xy[:, 0] >= 0.0)
        & (target_xy[:, 0] < float(camera.width))
        & (target_xy[:, 1] >= 0.0)
        & (target_xy[:, 1] < float(camera.height))
    )
    flow = np.linalg.norm(initial_xy[keep] - target_xy[keep], axis=1)
    depth = target_depth[keep]
    return {
        "initial_pixel_flow_sample_count": int(flow.size),
        "initial_pixel_flow_median_px": (
            float(np.median(flow)) if flow.size else 0.0
        ),
        "initial_pixel_flow_p90_px": (
            float(np.quantile(flow, 0.90)) if flow.size else 0.0
        ),
        "selected_surface_depth_median_m": (
            float(np.median(depth)) if depth.size else 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas_path = Path(args.radio_atlas)
    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    has_mapper = bool(str(args.surface_mapper_checkpoint))
    has_spatial_projection = bool(
        str(args.frame_spatial_projection_checkpoint)
    )
    if has_mapper == has_spatial_projection:
        raise ValueError(
            "provide exactly one query feature transform checkpoint"
        )
    if has_mapper:
        transform_path = Path(args.surface_mapper_checkpoint)
        mapper, transform_metadata = load_surface_maplet_mapper(
            transform_path, device=str(args.device)
        )
        spatial_projection = None
        transform_kind = "surface_maplet_mapper"
    else:
        transform_path = Path(
            args.frame_spatial_projection_checkpoint
        )
        spatial_projection, loaded_metadata = (
            load_surface_spatial_projection(
                transform_path, device=str(args.device)
            )
        )
        spatial_projection.eval()
        transform_metadata = dict(loaded_metadata)
        mapper = None
        transform_kind = "surface_spatial_projection"
    metadata = dict(atlas.metadata or {})
    if str(metadata.get("query_feature_transform", "")) != transform_kind:
        raise ValueError("atlas and query feature transforms differ")
    if str(metadata.get("query_feature_transform_sha256", "")) != (
        _sha256(transform_path)
    ):
        raise ValueError("query transform checkpoint differs from baked atlas")
    views = _load_views(
        Path(args.query_contributor_dir),
        atlas,
        Path(args.image_root),
    )
    if str(args.image_id):
        views = [
            view
            for view in views
            if str(view.image_id) == str(args.image_id)
        ]
        if len(views) != 1:
            raise ValueError(
                f"expected one basin view for {args.image_id!r}, "
                f"got {len(views)}"
            )
    if (
        int(args.query_shard_count) <= 0
        or not 0
        <= int(args.query_shard_index)
        < int(args.query_shard_count)
    ):
        raise ValueError("invalid query shard")
    views = [
        view
        for row, view in enumerate(views)
        if row % int(args.query_shard_count)
        == int(args.query_shard_index)
    ]
    if not views:
        raise ValueError("query shard is empty")
    query_trajectories = {
        str(view.trajectory_id) for view in views
    }
    mapping_trajectories = {
        str(value)
        for value in metadata.get("mapping_trajectory_ids", [])
    }
    if has_mapper:
        transform_trajectories = _trajectory_ids_from_mapper(
            transform_metadata
        )
    else:
        transform_trajectories = {
            str(value)
            for key in (
                "reference_trajectory_ids",
                "train_query_trajectory_ids",
                "validation_trajectory_ids",
            )
            for value in transform_metadata.get(key, [])
        }
    if query_trajectories & mapping_trajectories:
        raise ValueError("strict queries overlap atlas mapping")
    if query_trajectories & transform_trajectories:
        raise ValueError(
            "strict queries overlap feature-transform references"
        )
    base_stride = int(args.spatial_feature_stride)
    level_names = (
        ("coarse", "middle", "fine")
        if base_stride == 16
        else ("middle", "fine")
    )
    level_strides = {"coarse": 16, "middle": 8, "fine": 4}
    level_radii = (6, 4, 3)
    levels = []
    for iteration in range(max(int(args.rounds), 1)):
        level_name = level_names[min(iteration, len(level_names) - 1)]
        levels.append(
            AtlasAlignmentLevel(
                name=level_name,
                feature_stride=level_strides[level_name],
                correlation_radius=level_radii[
                    min(iteration, len(level_radii) - 1)
                ],
                maximum_translation_step_m=max(
                    0.03, 0.08 - 0.025 * iteration
                ),
                maximum_rotation_step_deg=max(
                    1.0, 4.0 - 1.25 * iteration
                ),
                maximum_points=3072,
                translation_search_step_m=max(
                    0.08, 0.25 - 0.07 * iteration
                ),
                direct_rotation_step_deg=max(
                    0.25, 1.0 * (0.5 ** iteration)
                ),
            )
        )
    rows = []
    for view in views:
        if str(args.spatial_query_token_dir):
            token_path = (
                Path(args.spatial_query_token_dir)
                / f"{view.image_id.replace('/', '__')}.npz"
            )
            if not token_path.exists():
                raise FileNotFoundError(
                    f"spatial query token is missing: {token_path}"
                )
            with np.load(token_path, allow_pickle=False) as token_data:
                raw_radio = np.asarray(
                    token_data["radio_final"], dtype=np.float32
                )
        else:
            raw_radio = np.asarray(view.radio, dtype=np.float32)
        if mapper is not None:
            mapped = mapper.project(
                raw_radio
            ).measurement_context.astype(np.float32)
        else:
            if spatial_projection is None:
                raise RuntimeError(
                    "spatial projection transform is unavailable"
                )
            with torch.no_grad():
                raw = torch.from_numpy(raw_radio).to(str(args.device))
                mapped = (
                    spatial_projection(raw.permute(1, 2, 0))
                    .permute(2, 0, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        query_pyramid = build_radio_final_feature_pyramid(
            mapped,
            base_stride=base_stride,
            strides=(
                (16, 8, 4) if base_stride == 16 else (8, 4)
            ),
        )
        alignment_atlases = {
            level: atlas for level in query_pyramid
        }
        visible_ids, visible_counts = _visible_charts(view, atlas)
        selected = _select_oracle_charts(
            visible_ids,
            visible_counts,
            atlas,
            int(args.visible_charts),
        )
        delta = _perturbation(
            view.image_id,
            translation_m=float(args.translation_m),
            rotation_deg=float(args.rotation_deg),
            seed=int(args.seed),
        )
        initial_pose = se3_exp(delta) @ view.pose_w2c
        flow_statistics = _initial_flow_statistics(
            atlas,
            selected,
            initial_pose,
            view.pose_w2c,
            view.camera,
        )
        alignment = refine_pose_with_maplet_atlases(
            alignment_atlases,
            query_pyramid,
            {level: None for level in query_pyramid},
            selected,
            initial_pose,
            view.camera,
            levels,
            device=str(args.device),
            minimum_fit_gain=0.01,
            minimum_heldout_gain=0.0,
            maximum_committed_translation_updates=1,
        )
        initial_error = pnp_pose_error(initial_pose, view.pose_w2c)
        final_error = pnp_pose_error(
            alignment.refined_pose_w2c, view.pose_w2c
        )
        gt_center = (
            -view.pose_w2c[:3, :3].T @ view.pose_w2c[:3, 3]
        )
        initial_center = (
            -initial_pose[:3, :3].T @ initial_pose[:3, 3]
        )
        final_center = (
            -alignment.refined_pose_w2c[:3, :3].T
            @ alignment.refined_pose_w2c[:3, 3]
        )
        row = {
            "image_id": view.image_id,
            "selected_chart_ids": list(alignment.selected_chart_ids),
            "initial_pose_w2c": initial_pose.tolist(),
            "final_pose_w2c": alignment.refined_pose_w2c.tolist(),
            "initial_translation_error_world": (
                initial_center - gt_center
            ).tolist(),
            "initial_translation_error_camera": (
                view.pose_w2c[:3, :3] @ (initial_center - gt_center)
            ).tolist(),
            "final_translation_error_world": (
                final_center - gt_center
            ).tolist(),
            "initial_translation_m": float(initial_error.translation_m),
            "initial_rotation_deg": float(initial_error.rotation_deg),
            "final_translation_m": float(final_error.translation_m),
            "final_rotation_deg": float(final_error.rotation_deg),
            "accepted_step_count": int(alignment.accepted_step_count),
            **flow_statistics,
            "atlas_score": (
                float(alignment.score)
                if np.isfinite(alignment.score)
                else None
            ),
            "steps": [
                {
                    **step.__dict__,
                    "condition_number": _finite_or_none(
                        step.condition_number
                    ),
                    "normalized_update_disagreement": _finite_or_none(
                        step.normalized_update_disagreement
                    ),
                    "fit_before": _finite_or_none(step.fit_before),
                    "fit_after": _finite_or_none(step.fit_after),
                    "heldout_before": _finite_or_none(
                        step.heldout_before
                    ),
                    "heldout_after": _finite_or_none(step.heldout_after),
                }
                for step in alignment.steps
            ],
        }
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    report = {
        "stage": "v6_radio_atlas_oracle_identity_local_basin",
        "diagnostic_scope": (
            "GT chooses visible chart identities and the initial perturbation "
            "only; atlas alignment itself receives no GT"
        ),
        "artifact_sha256": {
            "radio_atlas": _sha256(atlas_path),
            "query_feature_transform": _sha256(transform_path),
        },
        "protocol": {
            "atlas_mapping": sorted(mapping_trajectories),
            "feature_transform_reference": sorted(
                transform_trajectories
            ),
            "strict_test": sorted(query_trajectories),
            "strict_test_disjoint_from_atlas_mapping": True,
            "strict_test_disjoint_from_feature_transform": True,
        },
        "configuration": {
            "translation_m": float(args.translation_m),
            "rotation_deg": float(args.rotation_deg),
            "visible_charts": int(args.visible_charts),
            "rounds": int(args.rounds),
            "feature_levels": list(
                dict.fromkeys(level.name for level in levels)
            ),
            "feature_strides": (
                [16, 8, 4] if base_stride == 16 else [8, 4]
            ),
            "feature_pyramid_source": (
                (
                    "native stride16 RADIO-final plus pixel-centre-aligned "
                    "interpolation"
                    if base_stride == 16
                    else "phase-interleaved stride8 RADIO-final; stride4 interpolation"
                )
            ),
            "spatial_query_token_dir": (
                str(args.spatial_query_token_dir)
                if str(args.spatial_query_token_dir)
                else None
            ),
            "analytic_translation_steps_m": [
                _translation_search_step(level)
                for level in levels
            ],
            "direct_atlas_translation_steps_m": [
                float(
                    level.translation_search_step_m
                    if level.translation_search_step_m is not None
                    else level.maximum_translation_step_m
                )
                for level in levels
            ],
            "se3_optimization_order": (
                "per_level_unified_joint_rotation_direct_fit_selection_"
                "then_translation_pyramid"
            ),
            "minimum_rotation_fit_log_gain": 0.10,
            "minimum_translation_fit_log_gain": 0.05,
            "maximum_committed_translation_updates": 1,
            "verification_weighting": (
                "fixed_chart_fit_then_disjoint_heldout_acceptance"
            ),
            "query_shard_count": int(args.query_shard_count),
            "query_shard_index": int(args.query_shard_index),
        },
        "map_contract": {
            "query_feature_source": (
                f"RADIO-final through frozen {transform_kind}"
            ),
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_point_correspondence_pnp": False,
            "final_pose_estimator": (
                "rendered_maplet_atlas_correlation_iterative_se3"
            ),
        },
        "summary": _aggregate(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
