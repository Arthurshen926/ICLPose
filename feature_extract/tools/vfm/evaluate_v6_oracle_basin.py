"""Evaluate V6 oracle-maplet correlation and analytic SE(3) basin gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.atlas_renderer import (
    render_selected_maplet_atlases,
)
from feature_extract.vfm.localization_v6.heldout_verifier import (
    accept_pose_update,
    split_fit_heldout_maplets,
)
from feature_extract.vfm.localization_v6.local_correlation import (
    CorrelationDistribution,
    local_correlation_distribution,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
)
from feature_extract.vfm.localization_v6.se3_update import (
    se3_exp,
    solve_correlation_se3_hypotheses,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--atlas_middle", default="")
    parser.add_argument("--atlas_coarse", default="")
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--metric_encoder", required=True)
    parser.add_argument("--validation_contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--translation_buckets", nargs="+", type=float, default=[0.05, 0.20, 0.30])
    parser.add_argument("--maximum_views", type=int, default=4)
    parser.add_argument("--visible_maplets", type=int, default=18)
    parser.add_argument("--wrong_maplets", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _average_precision(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not np.any(labels):
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=np.float64))
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision * ranked) / np.sum(ranked))


def _subset_correlation(
    value: CorrelationDistribution, keep: np.ndarray
) -> CorrelationDistribution:
    return CorrelationDistribution(
        pixel_xy=value.pixel_xy[keep],
        xyz=value.xyz[keep],
        maplet_ids=value.maplet_ids[keep],
        offsets_xy=value.offsets_xy,
        probabilities=value.probabilities[keep],
        null_probability=value.null_probability[keep],
        mean_displacement=value.mean_displacement[keep],
        covariance=value.covariance[keep],
        entropy=value.entropy[keep],
        matchability=value.matchability[keep],
        surface_ids=(
            value.surface_ids[keep] if value.surface_ids is not None else None
        ),
    )


def _metrics(
    correlation: CorrelationDistribution,
    gt_pose: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    positive_maplets: np.ndarray,
    positive_surface_ids: np.ndarray | None = None,
) -> dict[str, float]:
    if positive_surface_ids is not None and correlation.surface_ids is not None:
        # Visibility is a property of a canonical surface texel, not of an
        # entire maplet.  Treating every rendered cell from a visible maplet as
        # positive incorrectly labels its occluded/back-side regions.
        positive = np.isin(
            correlation.surface_ids,
            np.asarray(positive_surface_ids, dtype=np.int64),
        )
    else:
        positive = np.isin(correlation.maplet_ids, positive_maplets)
    gt_xy, depth = project_world_points(correlation.xyz, gt_pose, camera)
    gt_grid = np.stack(
        [
            (gt_xy[:, 0] + 0.5) * width / camera.width - 0.5,
            (gt_xy[:, 1] + 0.5) * height / camera.height - 0.5,
        ],
        axis=1,
    )
    displacement = gt_grid - correlation.pixel_xy
    visible = (
        positive
        & (depth > 0.0)
        & np.isfinite(displacement).all(axis=1)
        & (np.max(np.abs(displacement), axis=1)
           <= np.max(np.abs(correlation.offsets_xy)) + 0.5)
    )
    if np.any(visible):
        predicted = correlation.mean_displacement[visible]
        target = displacement[visible]
        residual = predicted - target
        epe = np.linalg.norm(residual, axis=1)
        target_norm = np.linalg.norm(target, axis=1)
        predicted_norm = np.linalg.norm(predicted, axis=1)
        directional = target_norm > 1e-4
        direction_cosine = (
            float(
                np.mean(
                    np.sum(predicted[directional] * target[directional], axis=1)
                    / np.maximum(
                        predicted_norm[directional] * target_norm[directional],
                        1e-8,
                    )
                )
            )
            if np.any(directional)
            else float("nan")
        )
        unit_target = target / np.maximum(target_norm[:, None], 1e-8)
        parallel_error = np.sum(residual * unit_target, axis=1)
        orthogonal_error = residual - parallel_error[:, None] * unit_target
        mode = correlation.offsets_xy[
            np.argmax(correlation.probabilities[visible], axis=1)
        ]
        recall = np.mean(np.linalg.norm(mode - displacement[visible], axis=1) <= 1.0)
    else:
        epe = np.asarray([np.inf])
        recall = 0.0
        direction_cosine = float("nan")
        parallel_error = np.asarray([np.nan])
        orthogonal_error = np.asarray([[np.nan, np.nan]])
    null_labels = ~positive
    return {
        "flow_epe": float(np.mean(epe)),
        "correct_mode_recall": float(recall),
        "flow_direction_cosine": direction_cosine,
        "flow_parallel_bias": float(np.nanmean(parallel_error)),
        "flow_orthogonal_epe": float(
            np.nanmean(np.linalg.norm(orthogonal_error, axis=1))
        ),
        "null_auprc": _average_precision(
            null_labels, correlation.null_probability
        ),
        "null_prevalence": float(np.mean(null_labels)),
        "positive_point_count": int(np.sum(visible)),
        "negative_point_count": int(np.sum(null_labels)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_json)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite V6 basin report")
    rng = np.random.default_rng(int(args.seed))
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas))
    atlas_by_level = {
        "fine": atlas,
        "middle": (
            MapletFeatureAtlasBank.load_npz(Path(args.atlas_middle))
            if str(args.atlas_middle)
            else atlas
        ),
        "coarse": (
            MapletFeatureAtlasBank.load_npz(Path(args.atlas_coarse))
            if str(args.atlas_coarse)
            else atlas
        ),
    }
    geometry = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    views = _load_views(
        Path(args.validation_contributor_dir), geometry, Path(args.image_root)
    )[: int(args.maximum_views)]
    model, metadata = load_v6_metric_encoder(
        Path(args.metric_encoder), device=str(args.device)
    )
    encoder_sha256 = hashlib.sha256(
        Path(args.metric_encoder).read_bytes()
    ).hexdigest()
    compatible_map_encoder_sha256 = str(
        metadata.get("compatible_map_encoder_sha256", "")
    )
    uses_asymmetric_encoder_lineage = bool(
        compatible_map_encoder_sha256
        and compatible_map_encoder_sha256 != encoder_sha256
    )
    if uses_asymmetric_encoder_lineage and (
        not bool(metadata.get("map_encoder_is_frozen_teacher", False))
        or not bool(
            metadata.get(
                "compatible_map_encoder_verified_at_training", False
            )
        )
    ):
        raise ValueError(
            "query checkpoint declares an unverified frozen-map encoder"
        )
    expected_map_encoder_sha256 = (
        compatible_map_encoder_sha256 or encoder_sha256
    )
    lineage_verified = True
    exact_same_encoder_lineage = True
    for level, level_atlas in atlas_by_level.items():
        if (
            not np.array_equal(level_atlas.maplet_ids, atlas.maplet_ids)
            or level_atlas.height != atlas.height
            or level_atlas.width != atlas.width
            or not np.array_equal(
                level_atlas.primitive_ids, atlas.primitive_ids
            )
            or not np.allclose(level_atlas.xyz, atlas.xyz, atol=1e-6)
        ):
            raise ValueError(
                f"{level} atlas does not share canonical geometry/order"
            )
        atlas_encoder_sha256 = str(
            (level_atlas.metadata or {}).get("metric_encoder_sha256", "")
        )
        if (
            atlas_encoder_sha256
            and atlas_encoder_sha256 != expected_map_encoder_sha256
        ):
            raise ValueError(
                "atlas encoder is not the query checkpoint's compatible "
                "frozen-map encoder"
            )
        lineage_verified &= bool(atlas_encoder_sha256)
        exact_same_encoder_lineage &= bool(
            atlas_encoder_sha256 == encoder_sha256
        )
        declared_level = str(
            (level_atlas.metadata or {}).get("metric_feature_level", "")
        )
        if declared_level and declared_level != level:
            raise ValueError(f"{level} atlas uses {declared_level} features")
    model.eval()
    evaluation_trajectories = sorted(
        {view.trajectory_id for view in views}
    )
    training_trajectories = sorted(
        str(value) for value in metadata.get("training_trajectories", [])
    )
    mapping_trajectories = sorted(
        {
            str(value)
            for level_atlas in atlas_by_level.values()
            for value in (level_atlas.metadata or {}).get(
                "mapping_trajectory_ids", []
            )
        }
    )
    encoder_trajectory_disjoint = not bool(
        set(evaluation_trajectories) & set(training_trajectories)
    )
    atlas_trajectory_disjoint = not bool(
        set(evaluation_trajectories) & set(mapping_trajectories)
    )
    if not encoder_trajectory_disjoint:
        raise ValueError(
            "evaluation trajectory overlaps metric-encoder training"
        )
    if not atlas_trajectory_disjoint:
        raise ValueError(
            "evaluation trajectory overlaps metric-atlas baking"
        )
    cell_area = atlas.height * atlas.width
    support_by_maplet = np.sum(atlas.valid_mask, axis=(1, 2))
    cases = []
    gt_reconstruction = []
    with torch.no_grad():
        for view in views:
            output = model(
                view.radio[None].to(str(args.device)),
                view.rgb[None].to(str(args.device)),
            )
            query_features = {
                "fine": output["fine"][0].cpu().numpy(),
                "middle": output["middle"][0].cpu().numpy(),
                "coarse": output["coarse"][0].cpu().numpy(),
            }
            query_matchability = {
                key: torch.nn.functional.interpolate(
                    output["matchability"],
                    size=query_features[key].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy()
                for key in query_features
            }
            visible_maplet_rows = (
                view.visible_rows // cell_area
            ).astype(np.int64)
            unique, counts = np.unique(visible_maplet_rows, return_counts=True)
            order = unique[np.argsort(-counts)]
            order = order[atlas.valid_mask[order].any(axis=(1, 2))]
            positive_rows = order[: int(args.visible_maplets)]
            if positive_rows.size < 2:
                continue
            center_xy, center_depth = project_world_points(
                atlas.centers, view.pose_w2c, view.camera
            )
            center_in_view = (
                np.isfinite(center_xy).all(axis=1)
                & np.isfinite(center_depth)
                & (center_depth > 0.0)
                & (center_xy[:, 0] >= 0.0)
                & (center_xy[:, 0] < view.camera.width)
                & (center_xy[:, 1] >= 0.0)
                & (center_xy[:, 1] < view.camera.height)
            )
            wrong_candidates = np.flatnonzero(
                (support_by_maplet > 0)
                & center_in_view
                & ~np.isin(np.arange(len(atlas)), unique)
            )
            wrong_rows = wrong_candidates[
                np.argsort(-support_by_maplet[wrong_candidates])
            ][: int(args.wrong_maplets)]
            selected_rows = np.r_[positive_rows, wrong_rows]
            selected_ids = atlas.maplet_ids[selected_rows]
            positive_ids = atlas.maplet_ids[positive_rows]
            fit_ids, heldout_ids = split_fit_heldout_maplets(positive_ids)
            for key, radius in (
                ("coarse", 8),
                ("middle", 6),
                ("fine", 4),
            ):
                level_atlas = atlas_by_level[key]
                height, width = query_features[key].shape[-2:]
                gt_render = render_selected_maplet_atlases(
                    level_atlas,
                    selected_ids,
                    view.pose_w2c,
                    view.camera,
                    width=width,
                    height=height,
                )
                gt_correlation = local_correlation_distribution(
                    gt_render,
                    query_features[key],
                    radius=radius,
                    query_matchability=query_matchability[key],
                    maximum_points=4096,
                    device=str(args.device),
                )
                reconstruction_metrics = _metrics(
                    gt_correlation,
                    view.pose_w2c,
                    view.camera,
                    width=width,
                    height=height,
                    positive_maplets=positive_ids,
                    positive_surface_ids=view.visible_rows,
                )
                reprojection_xy, reprojection_depth = project_world_points(
                    gt_correlation.xyz, view.pose_w2c, view.camera
                )
                reprojection_grid = np.stack(
                    [
                        (reprojection_xy[:, 0] + 0.5)
                        * width
                        / view.camera.width
                        - 0.5,
                        (reprojection_xy[:, 1] + 0.5)
                        * height
                        / view.camera.height
                        - 0.5,
                    ],
                    axis=1,
                )
                reprojection_valid = (
                    np.isfinite(reprojection_grid).all(axis=1)
                    & np.isfinite(reprojection_depth)
                    & (reprojection_depth > 0.0)
                )
                gt_reconstruction.append(
                    {
                        "image_id": view.image_id,
                        "scale": key,
                        "rendered_point_count": int(
                            gt_correlation.xyz.shape[0]
                        ),
                        "raster_xyz_reprojection_rms_cells": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (
                                            reprojection_grid[
                                                reprojection_valid
                                            ]
                                            - gt_correlation.pixel_xy[
                                                reprojection_valid
                                            ]
                                        )
                                        ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        )
                        if np.any(reprojection_valid)
                        else None,
                        **reconstruction_metrics,
                    }
                )
            for translation_m in args.translation_buckets:
                axis = rng.normal(size=3)
                axis /= max(np.linalg.norm(axis), 1e-8)
                translation = axis * float(translation_m)
                rotation_axis = rng.normal(size=3)
                rotation_axis /= max(np.linalg.norm(rotation_axis), 1e-8)
                rotation = rotation_axis * np.deg2rad(2.0 * translation_m / 0.30)
                pose = se3_exp(np.r_[rotation, translation]) @ view.pose_w2c
                initial_error = pnp_pose_error(pose, view.pose_w2c)
                accepted_steps = 0
                first_metrics = None
                step_diagnostics = []
                for key, radius, stride in (
                    ("coarse", 8, 16),
                    ("middle", 6, 8),
                    ("fine", 4, 4),
                ):
                    level_atlas = atlas_by_level[key]
                    height, width = query_features[key].shape[-2:]
                    before_render = render_selected_maplet_atlases(
                        level_atlas,
                        selected_ids,
                        pose,
                        view.camera,
                        width=width,
                        height=height,
                    )
                    before = local_correlation_distribution(
                        before_render,
                        query_features[key],
                        radius=radius,
                        query_matchability=query_matchability[key],
                        maximum_points=4096,
                        device=str(args.device),
                    )
                    if first_metrics is None and before.xyz.shape[0] > 0:
                        first_metrics = _metrics(
                            before,
                            view.pose_w2c,
                            view.camera,
                            width=width,
                            height=height,
                            positive_maplets=positive_ids,
                            positive_surface_ids=view.visible_rows,
                        )
                    scale_metrics = _metrics(
                        before,
                        view.pose_w2c,
                        view.camera,
                        width=width,
                        height=height,
                        positive_maplets=positive_ids,
                        positive_surface_ids=view.visible_rows,
                    )
                    fit_before = _subset_correlation(
                        before, np.isin(before.maplet_ids, positive_ids)
                    )
                    updates = solve_correlation_se3_hypotheses(
                        fit_before,
                        pose,
                        view.camera,
                        fit_maplet_ids=fit_ids,
                        maximum_entropy=3.0,
                        minimum_points=6,
                        displacement_scale_xy=(stride, stride),
                        maximum_translation_step_m={
                            "coarse": 0.10,
                            "middle": 0.06,
                            "fine": 0.03,
                        }[key],
                        maximum_rotation_step_deg={
                            "coarse": 2.0,
                            "middle": 1.0,
                            "fine": 0.5,
                        }[key],
                    )
                    if not updates:
                        step_diagnostics.append(
                            {
                                "scale": key,
                                "accepted": False,
                                "solver_success": False,
                                "used_point_count": 0,
                                "hypothesis_count": 0,
                                **scale_metrics,
                            }
                        )
                        continue
                    verified = []
                    for hypothesis_index, update in enumerate(updates):
                        after_render = render_selected_maplet_atlases(
                            level_atlas,
                            selected_ids,
                            update.updated_pose_w2c,
                            view.camera,
                            width=width,
                            height=height,
                        )
                        after = local_correlation_distribution(
                            after_render,
                            query_features[key],
                            radius=radius,
                            query_matchability=query_matchability[key],
                            maximum_points=4096,
                            device=str(args.device),
                        )
                        accepted, evidence = accept_pose_update(
                            before,
                            after,
                            fit_maplet_ids=fit_ids,
                            heldout_maplet_ids=heldout_ids,
                            minimum_fit_gain=0.10,
                            minimum_heldout_gain=0.10,
                        )
                        score = (
                            evidence["heldout_after"]
                            + 0.25 * evidence["fit_after"]
                            if np.isfinite(evidence["heldout_after"])
                            and np.isfinite(evidence["fit_after"])
                            else float("-inf")
                        )
                        verified.append(
                            (bool(accepted), score, hypothesis_index, update, evidence)
                        )
                    accepted_rows = [row for row in verified if row[0]]
                    chosen = max(
                        accepted_rows if accepted_rows else verified,
                        key=lambda row: row[1],
                    )
                    accepted, _score, hypothesis_index, update, _evidence = chosen
                    if bool(accepted):
                        pose = update.updated_pose_w2c
                        accepted_steps += 1
                    step_diagnostics.append(
                        {
                            "scale": key,
                            "accepted": bool(accepted),
                            "solver_success": True,
                            "hypothesis_count": len(updates),
                            "selected_hypothesis_index": int(hypothesis_index),
                            "used_point_count": update.used_point_count,
                            "delta_rotation_deg": float(
                                np.degrees(np.linalg.norm(update.delta[:3]))
                            ),
                            "delta_translation_m": float(
                                np.linalg.norm(update.delta[3:])
                            ),
                            "evidence": _evidence,
                            **scale_metrics,
                        }
                    )
                final_error = pnp_pose_error(pose, view.pose_w2c)
                cases.append(
                    {
                        "image_id": view.image_id,
                        "initial_translation_m": float(initial_error.translation_m),
                        "initial_rotation_deg": float(initial_error.rotation_deg),
                        "target_bucket_m": float(translation_m),
                        "final_translation_m": float(final_error.translation_m),
                        "final_rotation_deg": float(final_error.rotation_deg),
                        "accepted_steps": accepted_steps,
                        "steps": step_diagnostics,
                        **(first_metrics or {}),
                    }
                )
                print(json.dumps(cases[-1]), flush=True)
    gates = {}
    for bucket in args.translation_buckets:
        rows = [row for row in cases if row["target_bucket_m"] == float(bucket)]
        gates[str(bucket)] = {
            "case_count": len(rows),
            "success_4cm_1deg": float(
                np.mean(
                    [
                        row["final_translation_m"] <= 0.04
                        and row["final_rotation_deg"] <= 1.0
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0,
            "median_final_translation_m": float(
                np.median([row["final_translation_m"] for row in rows])
            )
            if rows
            else None,
            "median_final_rotation_deg": float(
                np.median([row["final_rotation_deg"] for row in rows])
            )
            if rows
            else None,
        }
    thresholds = {0.05: 0.90, 0.20: 0.80, 0.30: 0.60}
    g2_pass = all(
        gates[str(bucket)]["success_4cm_1deg"] >= thresholds.get(float(bucket), 1.0)
        for bucket in args.translation_buckets
    )
    reconstruction_summary = {}
    for level in ("coarse", "middle", "fine"):
        rows = [row for row in gt_reconstruction if row["scale"] == level]
        level_summary = {"view_count": len(rows)}
        for key in (
            "flow_epe",
            "correct_mode_recall",
            "flow_direction_cosine",
            "null_auprc",
            "null_prevalence",
            "raster_xyz_reprojection_rms_cells",
        ):
            level_summary[key] = (
                float(
                    np.nanmean(
                        [
                            float(row[key])
                            for row in rows
                            if row.get(key) is not None
                        ]
                    )
                )
                if any(row.get(key) is not None for row in rows)
                else None
            )
        reconstruction_summary[level] = level_summary
    report = {
        "stage": "v6_g1_g2_oracle_maplet_correlation",
        "metric_encoder": str(args.metric_encoder),
        "metric_encoder_best_step": metadata.get("best_step", -1),
        "metric_encoder_sha256": encoder_sha256,
        "compatible_map_encoder_sha256": expected_map_encoder_sha256,
        "atlas_encoder_lineage_verified": bool(lineage_verified),
        "atlas_uses_exact_query_encoder": bool(exact_same_encoder_lineage),
        "uses_verified_teacher_map_student_query_lineage": bool(
            lineage_verified and uses_asymmetric_encoder_lineage
        ),
        "uses_scale_specific_atlases": bool(
            str(args.atlas_middle) and str(args.atlas_coarse)
        ),
        "trajectory_ids": evaluation_trajectories,
        "encoder_training_trajectory_ids": training_trajectories,
        "atlas_mapping_trajectory_ids": mapping_trajectories,
        "trajectory_disjoint_from_encoder_training": bool(
            encoder_trajectory_disjoint
        ),
        "trajectory_disjoint_from_atlas_baking": bool(
            atlas_trajectory_disjoint
        ),
        "case_count": len(cases),
        "gates": gates,
        "g2_pass": bool(g2_pass),
        "gt_pose_feature_reconstruction_summary": reconstruction_summary,
        "gt_pose_feature_reconstruction": gt_reconstruction,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
