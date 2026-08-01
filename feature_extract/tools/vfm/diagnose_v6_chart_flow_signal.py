"""Measure whether local RADIO chart correlations contain the GT flow signal.

This is an observation-level diagnostic for the V6 production path.  It
replays the exact global chart search on held-out trajectories, then measures
the rank of the geometrically correct local displacement in each chart cell's
RADIO-final correlation patch.  It does not create point identities or a map
artifact and never uses mapping/reference RGB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.tools.vfm.train_v6_structured_frame_adapter import (
    _prepare_views,
)
from feature_extract.tools.vfm.train_v6_structured_frame_refiner import (
    _SurfaceSpatialProjectionMapper,
    _build_deployment_replay,
    _canonical_for_rows,
    _project_homography,
    _sha256,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    structured_frame_correlation,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    load_surface_spatial_projection,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    estimate_chart_volume_residual,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--frame_spatial_projection_checkpoint", default="")
    parser.add_argument("--contributor_dir", required=True)
    parser.add_argument("--trajectory_ids", nargs="*", default=("seq11",))
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--charts_per_view", type=int, default=4)
    parser.add_argument("--wrong_charts_per_view", type=int, default=1)
    parser.add_argument("--modes_per_chart", type=int, default=8)
    parser.add_argument("--correlation_radius", type=int, default=4)
    parser.add_argument("--maximum_modes", type=int, default=0)
    parser.add_argument("--maximum_views", type=int, default=0)
    parser.add_argument(
        "--run_structured_optimization", action="store_true"
    )
    parser.add_argument("--optimization_cells", type=int, default=256)
    parser.add_argument(
        "--optimization_topk_per_cell", type=int, default=9
    )
    parser.add_argument("--optimization_proposals", type=int, default=384)
    parser.add_argument("--optimization_iterations", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _summarize_channel(
    score_rows: list[np.ndarray],
    displacement_rows: list[np.ndarray],
    offsets: np.ndarray,
) -> dict[str, object]:
    if not score_rows:
        return {"cell_count": 0}
    scores = np.concatenate(score_rows, axis=0)
    displacement = np.concatenate(displacement_rows, axis=0)
    nearest = np.argmin(
        np.linalg.norm(
            offsets[None].astype(np.float32) - displacement[:, None],
            axis=2,
        ),
        axis=1,
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.empty_like(order)
    inverse[
        np.arange(order.shape[0])[:, None],
        order,
    ] = np.arange(order.shape[1])[None]
    rank = inverse[np.arange(order.shape[0]), nearest] + 1
    top1_offset = offsets[order[:, 0]].astype(np.float32)
    endpoint = np.linalg.norm(top1_offset - displacement, axis=1)
    target_score = scores[np.arange(scores.shape[0]), nearest]
    top1_score = scores[np.arange(scores.shape[0]), order[:, 0]]
    result: dict[str, object] = {
        "cell_count": int(scores.shape[0]),
        "target_rank_median": float(np.median(rank)),
        "target_rank_p90": float(np.percentile(rank, 90)),
        "top1_endpoint_error_median_cells": float(np.median(endpoint)),
        "top1_endpoint_error_p90_cells": float(np.percentile(endpoint, 90)),
        "target_minus_top1_score_median": float(
            np.median(target_score - top1_score)
        ),
    }
    for k in (1, 3, 5, 9):
        result[f"target_bin_recall_at_{k}"] = float(np.mean(rank <= k))
        oracle = np.min(
            np.linalg.norm(
                offsets[order[:, :k]].astype(np.float32)
                - displacement[:, None],
                axis=2,
            ),
            axis=1,
        )
        result[f"oracle_top{k}_endpoint_median_cells"] = float(
            np.median(oracle)
        )
        result[f"oracle_top{k}_within_1cell_fraction"] = float(
            np.mean(oracle <= 1.0)
        )
    return result


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.radio_atlas))
    mapper_path = str(args.surface_mapper_checkpoint)
    projection_path = str(args.frame_spatial_projection_checkpoint)
    if bool(mapper_path) == bool(projection_path):
        raise ValueError(
            "provide exactly one feature transform checkpoint"
        )
    if mapper_path:
        feature_transform_path = Path(mapper_path)
        feature_transform_kind = "surface_maplet_mapper"
        mapper, _metadata = load_surface_maplet_mapper(
            feature_transform_path, device=str(args.device)
        )
    else:
        feature_transform_path = Path(projection_path)
        feature_transform_kind = "surface_spatial_projection"
        projection, _metadata = load_surface_spatial_projection(
            feature_transform_path, device=str(args.device)
        )
        projection.eval()
        mapper = _SurfaceSpatialProjectionMapper(
            projection, str(args.device)
        )
    atlas_metadata = dict(atlas.metadata or {})
    if str(atlas_metadata.get("query_feature_transform", "")) != str(
        feature_transform_kind
    ):
        raise ValueError("diagnostic feature transform differs from atlas")
    if str(
        atlas_metadata.get("query_feature_transform_sha256", "")
    ) != _sha256(feature_transform_path):
        raise ValueError("diagnostic feature-transform lineage differs")
    raw = _load_views(
        Path(args.contributor_dir),
        atlas,
        Path(args.image_root),
        tuple(str(value) for value in args.trajectory_ids),
    )
    views = _prepare_views(raw, atlas, mapper)
    del raw
    if int(args.maximum_views) > 0:
        views = views[: int(args.maximum_views)]
    replay, replay_summary = _build_deployment_replay(
        views,
        atlas,
        charts_per_view=int(args.charts_per_view),
        wrong_charts_per_view=int(args.wrong_charts_per_view),
        modes_per_chart=int(args.modes_per_chart),
        maximum_fitted_error_cells=0.75,
        maximum_linear_residual=3.2,
        linear_parameterization="additive",
        update_parameterization="canonical_residual",
        seed=int(args.seed),
        device=str(args.device),
    )
    positives = [value for value in replay if value.usable]
    if int(args.maximum_modes) > 0:
        positives = positives[: int(args.maximum_modes)]
    radius = int(args.correlation_radius)
    values = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    offsets = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    cell_count = int(atlas.height * atlas.width)
    score_rows: dict[str, list[np.ndarray]] = {
        "mean": [],
        "appearance_mode": [],
        "maximum": [],
        "average": [],
    }
    displacement_rows: list[np.ndarray] = []
    mode_coverages: list[float] = []
    mode_max_displacements: list[float] = []
    before_errors: list[float] = []
    optimized_after_errors: list[float] = []
    optimized_evidence_gains: list[float] = []
    optimized_rows: list[dict[str, object]] = []
    for mode_index, value in enumerate(positives):
        view = views[int(value.view_index)]
        chart_row = int(value.chart_row)
        local = np.flatnonzero(
            np.asarray(atlas.valid_mask[chart_row], bool)
            & (np.asarray(atlas.support_count[chart_row]) > 0)
        )
        rows = chart_row * cell_count + local
        _row, canonical = _canonical_for_rows(rows, atlas)
        candidate = _project_homography(
            value.canonical_homography, canonical
        )
        target = _project_homography(value.target_homography, canonical)
        displacement = target - candidate
        y = local // atlas.width
        x = local % atlas.width
        mean = atlas.features[chart_row, :, y, x][None]
        mode_feature = (
            atlas.mode_features[chart_row, :, :, y, x][None]
            if atlas.mode_features is not None
            else None
        )
        mode_weight = (
            atlas.mode_weights[chart_row, :, y, x][None]
            if atlas.mode_weights is not None
            else None
        )
        mode_valid = (
            atlas.mode_valid_mask[chart_row, :, y, x][None]
            if atlas.mode_valid_mask is not None
            else None
        )
        correlation = structured_frame_correlation(
            view.query_feature[None].to(
                device=str(args.device), dtype=torch.float32
            ),
            torch.from_numpy(mean).to(
                device=str(args.device), dtype=torch.float32
            ),
            (
                torch.from_numpy(mode_feature).to(
                    device=str(args.device), dtype=torch.float32
                )
                if mode_feature is not None
                else None
            ),
            (
                torch.from_numpy(mode_weight).to(
                    device=str(args.device), dtype=torch.float32
                )
                if mode_weight is not None
                else None
            ),
            (
                torch.from_numpy(mode_valid).to(device=str(args.device))
                if mode_valid is not None
                else None
            ),
            torch.from_numpy(candidate[None]).to(
                device=str(args.device), dtype=torch.float32
            ),
            radius=radius,
        )[0].cpu().numpy()
        patch_cells = int(offsets.shape[0])
        mean_score = correlation[:, :patch_cells]
        mode_score = correlation[:, patch_cells:]
        score_rows["mean"].append(mean_score)
        score_rows["appearance_mode"].append(mode_score)
        score_rows["maximum"].append(np.maximum(mean_score, mode_score))
        score_rows["average"].append(0.5 * (mean_score + mode_score))
        displacement_rows.append(displacement)
        infinity = np.max(np.abs(displacement), axis=1)
        mode_coverages.append(float(np.mean(infinity <= radius)))
        mode_max_displacements.append(float(np.max(infinity)))
        before_errors.append(
            float(np.mean(np.linalg.norm(displacement, axis=1)))
        )
        if bool(args.run_structured_optimization):
            optimization_cells = min(
                int(args.optimization_cells), int(local.size)
            )
            selection = np.linspace(
                0,
                local.size - 1,
                optimization_cells,
                dtype=np.int64,
            )
            linear, translation, evidence, baseline_evidence = (
                estimate_chart_volume_residual(
                    torch.from_numpy(correlation[selection]).to(
                        device=str(args.device), dtype=torch.float32
                    ),
                    torch.from_numpy(canonical[selection]).to(
                        device=str(args.device), dtype=torch.float32
                    ),
                    radius_cells=radius,
                    topk_per_cell=int(args.optimization_topk_per_cell),
                    random_proposals=int(args.optimization_proposals),
                    iterations=int(args.optimization_iterations),
                    seed=int(args.seed) + 1009 * mode_index,
                )
            )
            predicted = (
                candidate
                + canonical @ linear.T
                + translation[None]
            )
            after_error = float(
                np.mean(np.linalg.norm(predicted - target, axis=1))
            )
            optimized_after_errors.append(after_error)
            optimized_evidence_gains.append(
                float(evidence - baseline_evidence)
            )
            optimized_rows.append(
                {
                    "view_index": int(value.view_index),
                    "chart_row": int(value.chart_row),
                    "before_error_cells": before_errors[-1],
                    "after_error_cells": after_error,
                    "evidence": float(evidence),
                    "baseline_evidence": float(baseline_evidence),
                }
            )
        print(
            json.dumps(
                {
                    "progress": f"{mode_index + 1}/{len(positives)}",
                    "cells": int(local.size),
                    "radius_coverage": mode_coverages[-1],
                }
            ),
            flush=True,
        )
    all_displacement = (
        np.concatenate(displacement_rows, axis=0)
        if displacement_rows
        else np.empty((0, 2), dtype=np.float32)
    )
    report = {
        "stage": "v6_chart_flow_signal_diagnostic",
        "trajectory_ids": sorted(
            {view.trajectory_id for view in views}
        ),
        "view_count": int(len(views)),
        "positive_mode_count": int(len(positives)),
        "replay": replay_summary,
        "correlation_radius_cells": radius,
        "query_feature_transform": feature_transform_kind,
        "query_feature_transform_sha256": _sha256(
            feature_transform_path
        ),
        "observability": {
            "cell_count": int(all_displacement.shape[0]),
            "cell_target_inside_radius_fraction": (
                float(
                    np.mean(
                        np.max(np.abs(all_displacement), axis=1) <= radius
                    )
                )
                if all_displacement.size
                else None
            ),
            "mode_full_cell_coverage_fraction": float(
                np.mean(np.asarray(mode_coverages) >= 1.0)
            )
            if mode_coverages
            else None,
            "mode_radius_coverage_median": _percentile(
                mode_coverages, 50
            ),
            "mode_max_displacement_median_cells": _percentile(
                mode_max_displacements, 50
            ),
            "mode_max_displacement_p90_cells": _percentile(
                mode_max_displacements, 90
            ),
            "mode_before_error_median_cells": _percentile(
                before_errors, 50
            ),
        },
        "channels": {
            name: _summarize_channel(values, displacement_rows, offsets)
            for name, values in score_rows.items()
        },
        "structured_chart_volume": (
            {
                "mode_count": int(len(optimized_after_errors)),
                "before_error_median_cells": _percentile(
                    before_errors, 50
                ),
                "after_error_median_cells": _percentile(
                    optimized_after_errors, 50
                ),
                "after_error_p90_cells": _percentile(
                    optimized_after_errors, 90
                ),
                "improvement_fraction": float(
                    np.mean(
                        np.asarray(optimized_after_errors)
                        < np.asarray(before_errors)
                    )
                ),
                "recall_within_1cell": float(
                    np.mean(np.asarray(optimized_after_errors) <= 1.0)
                ),
                "recall_within_1_5cells": float(
                    np.mean(np.asarray(optimized_after_errors) <= 1.5)
                ),
                "evidence_gain_median": _percentile(
                    optimized_evidence_gains, 50
                ),
                "per_mode": optimized_rows,
            }
            if bool(args.run_structured_optimization)
            else None
        ),
        "map_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_pairwise_image_matching": False,
            "uses_point_correspondence_pnp": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summary": report}), flush=True)


if __name__ == "__main__":
    main()
