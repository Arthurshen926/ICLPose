"""Audit proposal recall and PnP oracle limits on the exact local-matcher pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from feature_extract.tools.vfm.probe_local_assignment_support_views import _evaluate_pose_strategy
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    geometry_oracle_scores,
    rank_candidate_pool,
    summarize_detector_proposal_geometry,
    summarize_query_proposal_difficulty,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)


def _positive_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or min(parsed) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _positive_float_list(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or min(parsed) <= 0.0:
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--feature_artifact", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_strategy", default="alike_support_top2_mean")
    parser.add_argument("--full_pool_top_ls", type=_positive_int_list, default=(1, 5, 10, 20))
    parser.add_argument("--matcher_pool_top_ls", type=_positive_int_list, default=(1, 5, 10))
    parser.add_argument("--geometry_thresholds_px", type=_positive_float_list, default=(1.0, 2.0, 5.0, 8.0))
    parser.add_argument("--oracle_thresholds_px", type=_positive_float_list, default=(2.0, 5.0))
    parser.add_argument("--pose_splits", default="validation,test")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 2 or source.shape[0] <= int(np.max(rows, initial=-1)):
        raise ValueError("candidate source and selected rows are incompatible")
    return np.take_along_axis(source[rows], columns, axis=1)


def _difficulty_aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {"query_count": int(len(rows)), "thresholds_px": {}}
    if not rows:
        return output
    threshold_names = tuple(dict(rows[0]["thresholds_px"]).keys())
    for threshold in threshold_names:
        values = [dict(dict(row["thresholds_px"])[threshold]) for row in rows]
        positive = np.asarray([row["positive_point_count"] for row in values], dtype=np.float64)
        unique = np.asarray([row["positive_unique_track_count"] for row in values], dtype=np.float64)
        coverage = np.asarray([row["grid_coverage"] for row in values], dtype=np.float64)
        output["thresholds_px"][threshold] = {
            "median_positive_points_per_query": float(np.median(positive)),
            "minimum_positive_points_per_query": int(np.min(positive)),
            "query_with_at_least_4_positive_rate": float(np.mean(positive >= 4.0)),
            "median_unique_positive_tracks_per_query": float(np.median(unique)),
            "minimum_unique_positive_tracks_per_query": int(np.min(unique)),
            "median_grid_coverage_4x4": float(np.median(coverage)),
            "minimum_grid_coverage_4x4": float(np.min(coverage)),
        }
    return output


def _finite_correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return {"count": int(len(x)), "pearson": None, "spearman": None}
    spearman = spearmanr(x, y).statistic
    return {
        "count": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": None if not np.isfinite(spearman) else float(spearman),
    }


def _pose_difficulty_correlation(
    difficulty_rows: Sequence[Mapping[str, object]],
    baseline_pose_rows: Sequence[Mapping[str, object]],
    oracle_pose_rows: Sequence[Mapping[str, object]],
    *,
    threshold: str,
) -> dict[str, object]:
    baseline = {str(row["query_id"]): row for row in baseline_pose_rows}
    oracle = {str(row["query_id"]): row for row in oracle_pose_rows}
    joined = []
    for item in difficulty_rows:
        query_id = str(item["query_id"])
        baseline_row = baseline.get(query_id)
        oracle_row = oracle.get(query_id)
        if baseline_row is None or oracle_row is None:
            continue
        baseline_error = baseline_row.get("translation_m")
        oracle_error = oracle_row.get("translation_m")
        if baseline_error is None or oracle_error is None:
            continue
        geometry = dict(dict(item["thresholds_px"])[str(threshold)])
        joined.append(
            {
                "query_id": query_id,
                "baseline_translation_m": float(baseline_error),
                "oracle_translation_m": float(oracle_error),
                "selected_vs_oracle_regret_m": float(baseline_error) - float(oracle_error),
                **geometry,
            }
        )
    baseline_errors = [float(row["baseline_translation_m"]) for row in joined]
    regrets = [float(row["selected_vs_oracle_regret_m"]) for row in joined]
    correlations = {}
    for metric in ("positive_point_count", "positive_unique_track_count", "grid_coverage"):
        metric_values = [float(row[metric]) for row in joined]
        correlations[metric] = {
            "vs_baseline_translation": _finite_correlation(metric_values, baseline_errors),
            "vs_selected_oracle_regret": _finite_correlation(metric_values, regrets),
        }
    return {"query_rows": joined, "correlations": correlations}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = Path(args.proposals)
    feature_path = Path(args.feature_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    proposals = _load_npz(proposals_path)
    features = _load_npz(feature_path)
    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    split = json.loads(split_path.read_text())
    pose_splits = tuple(value.strip() for value in str(args.pose_splits).split(",") if value.strip())
    if not pose_splits or any(name not in split for name in pose_splits):
        raise ValueError("pose_splits must name blocks in split_json")

    required = {
        "query_ids",
        "xy",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "candidate_gt_residuals_px",
        "nearest_visible_track_ids",
        "nearest_visible_residuals_px",
        f"strategy__{args.baseline_strategy}",
    }
    missing = sorted(required - set(proposals))
    if missing:
        raise ValueError(f"proposal artifact is missing arrays: {missing}")
    query_ids_all = np.asarray(proposals["query_ids"]).astype(str)
    xy_all = np.asarray(proposals["xy"], dtype=np.float32)
    candidate_tracks_all = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    candidate_prototypes_all = np.asarray(proposals["candidate_prototype_ids"], dtype=np.int64)
    candidate_residuals_all = np.asarray(proposals["candidate_gt_residuals_px"], dtype=np.float32)
    baseline_all = np.asarray(
        proposals[f"strategy__{args.baseline_strategy}"], dtype=np.float32
    )
    canonical_all = canonical_rows_for_track_candidates(
        candidate_tracks_all, landmark_index.track_ids
    )
    nearest_residuals_all = np.asarray(
        proposals["nearest_visible_residuals_px"], dtype=np.float32
    )
    nearest_tracks_all = np.asarray(proposals["nearest_visible_track_ids"], dtype=np.int64)
    pose_keep_all = np.asarray(
        proposals.get("pose_keep_mask", np.ones((len(query_ids_all),), dtype=bool)),
        dtype=bool,
    )

    selected_rows = np.asarray(features["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(features["selected_columns"], dtype=np.int64)
    if np.any(selected_columns < 0):
        raise ValueError("this audit requires a fixed valid matcher candidate pool")
    pools = {
        "full_detector_pool": {
            "rows": np.arange(len(query_ids_all), dtype=np.int64),
            "scores": baseline_all,
            "tracks": candidate_tracks_all,
            "prototypes": candidate_prototypes_all,
            "canonical": canonical_all,
            "residuals": candidate_residuals_all,
            "pose_keep": pose_keep_all,
            "top_ls": tuple(args.full_pool_top_ls),
        },
        "matcher_input_pool": {
            "rows": selected_rows,
            "scores": _compact(baseline_all, selected_rows, selected_columns),
            "tracks": _compact(candidate_tracks_all, selected_rows, selected_columns),
            "prototypes": _compact(candidate_prototypes_all, selected_rows, selected_columns),
            "canonical": _compact(canonical_all, selected_rows, selected_columns),
            "residuals": _compact(candidate_residuals_all, selected_rows, selected_columns),
            "pose_keep": pose_keep_all[selected_rows],
            "top_ls": tuple(args.matcher_pool_top_ls),
        },
    }

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    image_sizes = {
        str(image.image_name): (
            int(cameras[int(image.camera_id)].width),
            int(cameras[int(image.camera_id)].height),
        )
        for image in images.values()
    }

    summary_pools: dict[str, object] = {}
    all_query_diagnostics: dict[str, object] = {}
    all_pose_rows: dict[str, object] = {}
    for pool_name, raw_pool in pools.items():
        rows = np.asarray(raw_pool["rows"], dtype=np.int64)
        pool_query_ids = query_ids_all[rows]
        pool_xy = xy_all[rows]
        nearest_residuals = nearest_residuals_all[rows]
        nearest_tracks = nearest_tracks_all[rows]
        pose_keep = np.asarray(raw_pool["pose_keep"], dtype=bool)
        observations = [
            ColmapTrackObservation(
                track_id=int(nearest_tracks[index]),
                image_id=str(pool_query_ids[index]),
                point2d_idx=int(rows[index]),
                xy=(float(pool_xy[index, 0]), float(pool_xy[index, 1])),
                xyz=np.zeros((3,), dtype=np.float64),
                track_length=1,
                reprojection_error=0.0,
            )
            for index in range(len(rows))
        ]
        pool_summary: dict[str, object] = {
            "point_count": int(len(rows)),
            "points_per_query": float(len(rows) / max(len(set(pool_query_ids.tolist())), 1)),
            "top_l": {},
        }
        map_canonical = canonical_rows_for_track_candidates(
            nearest_tracks[:, None], landmark_index.track_ids
        )
        map_prototypes = np.full(map_canonical.shape, -1, dtype=np.int64)
        map_valid = map_canonical[:, 0] >= 0
        map_prototypes[map_valid, 0] = landmark_index.prototype_ids[
            map_canonical[map_valid, 0]
        ]
        map_candidates = UniqueTrackCandidateSet(
            map_canonical,
            nearest_tracks[:, None],
            map_prototypes,
            np.zeros(map_canonical.shape, dtype=np.float32),
        )
        pool_summary["map_geometry_oracle_pose"] = {}
        for split_name in pose_splits:
            split_mask = np.isin(pool_query_ids, np.asarray(split[split_name], dtype=np.str_))
            split_rows = np.flatnonzero(split_mask)
            subset = UniqueTrackCandidateSet(
                map_candidates.bank_row_indices[split_rows],
                map_candidates.track_ids[split_rows],
                map_candidates.prototype_ids[split_rows],
                map_candidates.coarse_scores[split_rows],
            )
            pool_summary["map_geometry_oracle_pose"][split_name] = {}
            for threshold in tuple(args.oracle_thresholds_px):
                map_scores = np.full((len(split_rows), 1), -np.inf, dtype=np.float32)
                accepted = (
                    map_valid[split_rows]
                    & pose_keep[split_rows]
                    & (nearest_residuals[split_rows] <= float(threshold))
                )
                map_scores[accepted, 0] = -nearest_residuals[split_rows][accepted]
                map_pose, map_rows = _evaluate_pose_strategy(
                    strategy=f"{pool_name}_map_geometry_oracle_{threshold:g}px",
                    scores=map_scores,
                    candidates=subset,
                    query_observations=[observations[int(index)] for index in split_rows],
                    query_ids=pool_query_ids[split_rows].tolist(),
                    landmark_index=landmark_index,
                    cameras=cameras,
                    images_by_name=images_by_name,
                    reprojection_error_px=float(args.pnp_reprojection_error_px),
                    iterations=int(args.pnp_iterations),
                )
                threshold_name = f"{float(threshold):g}"
                pool_summary["map_geometry_oracle_pose"][split_name][threshold_name] = map_pose
                all_pose_rows[
                    f"{pool_name}/map_geometry_oracle/{split_name}/{threshold_name}px"
                ] = map_rows
        pool_query_output: dict[str, object] = {}
        for top_l in tuple(raw_pool["top_ls"]):
            ranked = rank_candidate_pool(
                candidate_scores=np.asarray(raw_pool["scores"]),
                top_l=int(top_l),
                arrays=(
                    np.asarray(raw_pool["tracks"]),
                    np.asarray(raw_pool["prototypes"]),
                    np.asarray(raw_pool["canonical"]),
                    np.asarray(raw_pool["residuals"]),
                ),
            )
            ranked_scores, ranked_tracks, ranked_prototypes, ranked_canonical, ranked_residuals = ranked
            candidates = UniqueTrackCandidateSet(
                ranked_canonical,
                ranked_tracks,
                ranked_prototypes,
                ranked_scores,
            )
            top_summary: dict[str, object] = {"splits": {}}
            top_query_output: dict[str, object] = {}
            for split_name in ("train", "validation", "test", "all"):
                split_mask = (
                    np.ones((len(rows),), dtype=bool)
                    if split_name == "all"
                    else np.isin(pool_query_ids, np.asarray(split[split_name], dtype=np.str_))
                )
                split_rows = np.flatnonzero(split_mask)
                difficulty = summarize_query_proposal_difficulty(
                    nearest_landmark_residuals=nearest_residuals[split_mask],
                    ranked_candidate_residuals=ranked_residuals[split_mask],
                    ranked_candidate_track_ids=ranked_tracks[split_mask],
                    query_ids=pool_query_ids[split_mask],
                    query_xy=pool_xy[split_mask],
                    image_sizes=image_sizes,
                    thresholds_px=tuple(args.geometry_thresholds_px),
                )
                split_summary: dict[str, object] = {
                    "geometry": summarize_detector_proposal_geometry(
                        nearest_landmark_residuals=nearest_residuals[split_mask],
                        candidate_residuals=ranked_residuals[split_mask],
                        query_ids=pool_query_ids[split_mask].tolist(),
                        thresholds_px=tuple(args.geometry_thresholds_px),
                        top_ls=(int(top_l),),
                    ),
                    "difficulty": _difficulty_aggregate(difficulty),
                }
                top_query_output[split_name] = difficulty
                if split_name in pose_splits:
                    subset = UniqueTrackCandidateSet(
                        candidates.bank_row_indices[split_rows],
                        candidates.track_ids[split_rows],
                        candidates.prototype_ids[split_rows],
                        candidates.coarse_scores[split_rows],
                    )
                    subset_observations = [observations[int(index)] for index in split_rows]
                    subset_query_ids = pool_query_ids[split_rows].tolist()
                    baseline_pose, baseline_rows = _evaluate_pose_strategy(
                        strategy=f"{pool_name}_l{top_l}_baseline",
                        scores=np.where(
                            pose_keep[split_rows, None], ranked_scores[split_rows], -np.inf
                        ),
                        candidates=subset,
                        query_observations=subset_observations,
                        query_ids=subset_query_ids,
                        landmark_index=landmark_index,
                        cameras=cameras,
                        images_by_name=images_by_name,
                        reprojection_error_px=float(args.pnp_reprojection_error_px),
                        iterations=int(args.pnp_iterations),
                    )
                    split_summary["baseline_pose"] = baseline_pose
                    all_pose_rows[f"{pool_name}/L{top_l}/{split_name}/baseline"] = baseline_rows
                    oracle_rows_by_threshold = {}
                    split_summary["geometry_oracle_pose"] = {}
                    for threshold in tuple(args.oracle_thresholds_px):
                        oracle_scores = geometry_oracle_scores(
                            ranked_residuals[split_rows], threshold_px=float(threshold)
                        )
                        oracle_scores[~pose_keep[split_rows]] = -np.inf
                        oracle_pose, oracle_rows = _evaluate_pose_strategy(
                            strategy=f"{pool_name}_l{top_l}_geometry_oracle_{threshold:g}px",
                            scores=oracle_scores,
                            candidates=subset,
                            query_observations=subset_observations,
                            query_ids=subset_query_ids,
                            landmark_index=landmark_index,
                            cameras=cameras,
                            images_by_name=images_by_name,
                            reprojection_error_px=float(args.pnp_reprojection_error_px),
                            iterations=int(args.pnp_iterations),
                        )
                        threshold_name = f"{float(threshold):g}"
                        split_summary["geometry_oracle_pose"][threshold_name] = oracle_pose
                        oracle_rows_by_threshold[threshold_name] = oracle_rows
                        all_pose_rows[
                            f"{pool_name}/L{top_l}/{split_name}/geometry_oracle_{threshold_name}px"
                        ] = oracle_rows
                    correlation_threshold = "2" if "2" in oracle_rows_by_threshold else next(iter(oracle_rows_by_threshold))
                    correlation = _pose_difficulty_correlation(
                        difficulty,
                        baseline_rows,
                        oracle_rows_by_threshold[correlation_threshold],
                        threshold=correlation_threshold,
                    )
                    split_summary["difficulty_pose_correlation"] = correlation["correlations"]
                    top_query_output[f"{split_name}_pose_join"] = correlation["query_rows"]
                top_summary["splits"][split_name] = split_summary
            pool_summary["top_l"][str(int(top_l))] = top_summary
            pool_query_output[str(int(top_l))] = top_query_output
        summary_pools[pool_name] = pool_summary
        all_query_diagnostics[pool_name] = pool_query_output

    summary = {
        "stage": "candidate_pool_pose_oracle_audit",
        "protocol": {
            "diagnostic_gt_geometry_only": True,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "baseline_strategy": str(args.baseline_strategy),
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
            "pnp_iterations": int(args.pnp_iterations),
        },
        "inputs": {
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "feature_artifact": str(feature_path),
            "feature_artifact_sha256": file_sha256_short(feature_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": file_sha256_short(bank_path),
            "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
            "split_json": str(split_path),
            "split_json_sha256": file_sha256_short(split_path),
            "colmap_model_dir": str(model_dir),
        },
        "pools": summary_pools,
    }
    (output_dir / "query_diagnostics.json").write_text(
        json.dumps(all_query_diagnostics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "pose_rows.json").write_text(
        json.dumps(all_pose_rows, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
