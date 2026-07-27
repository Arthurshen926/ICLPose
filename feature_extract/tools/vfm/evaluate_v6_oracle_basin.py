"""Evaluate V6 oracle-maplet correlation and analytic SE(3) basin gates."""

from __future__ import annotations

import argparse
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
    solve_correlation_se3_update,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
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
    )


def _metrics(
    correlation: CorrelationDistribution,
    gt_pose: np.ndarray,
    camera,
    *,
    width: int,
    height: int,
    positive_maplets: np.ndarray,
) -> dict[str, float]:
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
        epe = np.linalg.norm(
            correlation.mean_displacement[visible] - displacement[visible], axis=1
        )
        mode = correlation.offsets_xy[
            np.argmax(correlation.probabilities[visible], axis=1)
        ]
        recall = np.mean(np.linalg.norm(mode - displacement[visible], axis=1) <= 1.0)
    else:
        epe = np.asarray([np.inf])
        recall = 0.0
    null_labels = ~positive
    return {
        "flow_epe": float(np.mean(epe)),
        "correct_mode_recall": float(recall),
        "null_auprc": _average_precision(
            null_labels, correlation.null_probability
        ),
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
    geometry = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    views = _load_views(
        Path(args.validation_contributor_dir), geometry, Path(args.image_root)
    )[: int(args.maximum_views)]
    depth_by_image = {}
    for cache_path in Path(args.validation_contributor_dir).glob("*.npz"):
        with np.load(cache_path, allow_pickle=False) as data:
            cache_metadata = json.loads(str(data["metadata_json"].item()))
            depth_by_image[str(cache_metadata["image_id"])] = np.asarray(
                data["dominant_depth"], dtype=np.float32
            )
    model, metadata = load_v6_metric_encoder(
        Path(args.metric_encoder), device=str(args.device)
    )
    model.eval()
    cell_area = atlas.height * atlas.width
    support_by_maplet = np.sum(atlas.valid_mask, axis=(1, 2))
    cases = []
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
            query_null = {
                key: torch.nn.functional.interpolate(
                    output["null_probability"],
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
            wrong_candidates = np.flatnonzero(
                (support_by_maplet > 0) & ~np.isin(np.arange(len(atlas)), unique)
            )
            wrong_rows = wrong_candidates[
                np.argsort(-support_by_maplet[wrong_candidates])
            ][: int(args.wrong_maplets)]
            selected_rows = np.r_[positive_rows, wrong_rows]
            selected_ids = atlas.maplet_ids[selected_rows]
            positive_ids = atlas.maplet_ids[positive_rows]
            fit_ids, heldout_ids = split_fit_heldout_maplets(positive_ids)
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
                    height, width = query_features[key].shape[-2:]
                    full_depth = torch.nn.functional.interpolate(
                        torch.from_numpy(depth_by_image[view.image_id])[
                            None, None
                        ],
                        size=(height, width),
                        mode="nearest",
                    )[0, 0].numpy()
                    before_render = render_selected_maplet_atlases(
                        atlas,
                        selected_ids,
                        pose,
                        view.camera,
                        width=width,
                        height=height,
                        full_scene_depth=full_depth,
                        occlusion_epsilon=0.10,
                    )
                    before = local_correlation_distribution(
                        before_render,
                        query_features[key],
                        radius=radius,
                        query_matchability=query_matchability[key],
                        query_null_probability=query_null[key],
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
                        )
                    scale_metrics = _metrics(
                        before,
                        view.pose_w2c,
                        view.camera,
                        width=width,
                        height=height,
                        positive_maplets=positive_ids,
                    )
                    fit_before = _subset_correlation(
                        before, np.isin(before.maplet_ids, positive_ids)
                    )
                    update = solve_correlation_se3_update(
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
                    if not update.success:
                        step_diagnostics.append(
                            {
                                "scale": key,
                                "accepted": False,
                                "solver_success": False,
                                "used_point_count": update.used_point_count,
                                **scale_metrics,
                            }
                        )
                        continue
                    after_render = render_selected_maplet_atlases(
                        atlas,
                        selected_ids,
                        update.updated_pose_w2c,
                        view.camera,
                        width=width,
                        height=height,
                        full_scene_depth=full_depth,
                        occlusion_epsilon=0.10,
                    )
                    after = local_correlation_distribution(
                        after_render,
                        query_features[key],
                        radius=radius,
                        query_matchability=query_matchability[key],
                        query_null_probability=query_null[key],
                        maximum_points=4096,
                        device=str(args.device),
                    )
                    accepted, _evidence = accept_pose_update(
                        before,
                        after,
                        fit_maplet_ids=fit_ids,
                        heldout_maplet_ids=heldout_ids,
                        minimum_fit_gain=0.10,
                        minimum_heldout_gain=0.10,
                    )
                    if accepted:
                        pose = update.updated_pose_w2c
                        accepted_steps += 1
                    step_diagnostics.append(
                        {
                            "scale": key,
                            "accepted": bool(accepted),
                            "solver_success": True,
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
    report = {
        "stage": "v6_g1_g2_oracle_maplet_correlation",
        "metric_encoder": str(args.metric_encoder),
        "metric_encoder_best_step": metadata.get("best_step", -1),
        "trajectory_ids": sorted({view.trajectory_id for view in views}),
        "trajectory_disjoint_from_encoder_training": True,
        "case_count": len(cases),
        "gates": gates,
        "g2_pass": bool(g2_pass),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
