"""Run no-submap full-bank projected-landmark aggregation and measurement sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_AGGREGATION_METHODS = (
    "mean",
    "cosine_weighted_mean",
    "geometry_weighted",
    "robust_trimmed_mean",
    "view_consistent",
    "geometric_median",
    "medoid",
    "ulf_geometry_weighted",
)


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv_strings(value))


def _python_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def build_projected_bank_command(common: Mapping[str, object], *, method: str, output_index: str, summary_json: str) -> list[str]:
    cmd = _python_module("feature_extract.tools.vfm.build_projected_observation_landmark_bank")
    cmd.extend(
        [
            "--track_observations",
            str(common["track_observations_jsonl"]),
            "--token_manifest",
            str(common["token_manifest"]),
            "--matcha_joint_checkpoint",
            str(common["matcha_joint_checkpoint"]),
            "--method",
            str(method),
            "--utility_mode",
            str(common.get("utility_mode", "inverse_reprojection_center_view")),
            "--sample_mode",
            str(common.get("sample_mode", "bilinear")),
            "--min_observations",
            str(int(common.get("min_observations", 2))),
            "--output_index",
            str(output_index),
            "--summary_json",
            str(summary_json),
            "--device",
            str(common.get("device", "cuda")),
        ]
    )
    if bool(common.get("l2_normalize_observations", False)):
        cmd.append("--l2_normalize_observations")
    return cmd


def build_eval_command(common: Mapping[str, object], *, measurement_k: int) -> list[str]:
    cmd = _python_module("feature_extract.tools.vfm.eval_real_radio_landmark_hybrid")
    cmd.extend(
        [
            "--query_manifest",
            str(common["query_manifest"]),
            "--landmark_bank",
            str(common["landmark_bank"]),
            "--track_observations_jsonl",
            str(common["track_observations_jsonl"]),
            "--image_root",
            str(common["image_root"]),
            "--colmap_model_dir",
            str(common["colmap_model_dir"]),
            "--query_pose_file",
            str(common["query_pose_file"]),
            "--matcha_joint_checkpoint",
            str(common["matcha_joint_checkpoint"]),
            "--projected_landmark_cache",
            str(common["projected_landmark_cache"]),
            "--output_dir",
            str(common["output_dir"]),
            "--landmark_search_backend",
            "faiss",
            "--submap_mode",
            "none",
            "--query_token_selection",
            "heatmap",
            "--query_heatmap_top_k",
            str(int(common.get("query_heatmap_top_k", 1000))),
            "--query_heatmap_nms_radius",
            str(int(common.get("query_heatmap_nms_radius", 1))),
            "--query_heatmap_grid_rows",
            str(int(common.get("query_heatmap_grid_rows", 4))),
            "--query_heatmap_grid_cols",
            str(int(common.get("query_heatmap_grid_cols", 8))),
            "--top_k",
            str(int(common.get("top_k", 2))),
            "--max_matches",
            str(int(common.get("max_matches", 300))),
            "--enable_quality_rescore",
            "--pnp_min_soft_score",
            str(float(common.get("pnp_min_soft_score", 0.4))),
            "--progress_interval_queries",
            str(int(common.get("progress_interval_queries", 20))),
            "--device",
            str(common.get("device", "cuda")),
        ]
    )
    max_queries = int(common.get("max_queries", 0))
    if max_queries > 0:
        cmd.extend(["--max_queries", str(max_queries)])
    if int(measurement_k) > 0:
        cmd.extend(
            [
                "--measurement_mode",
                "owner_rgb",
                "--measurement_checkpoint",
                str(common.get("measurement_checkpoint") or common["matcha_joint_checkpoint"]),
                "--measurement_max_matches",
                str(int(measurement_k)),
                "--measurement_selection_strategy",
                str(common.get("measurement_selection_strategy", "score_spatial")),
                "--measurement_batch_size",
                str(int(common.get("measurement_batch_size", 256))),
                "--measurement_query_batch_size",
                str(int(common.get("measurement_query_batch_size", 8))),
                "--measurement_score_mode",
                str(common.get("measurement_score_mode", "measurement_quality")),
                "--measurement_confidence_temperature",
                str(float(common.get("measurement_confidence_temperature", 4.0))),
                "--measurement_tensor_cache_size",
                str(int(common.get("measurement_tensor_cache_size", 64))),
                "--min_measurement_confidence",
                str(float(common.get("min_measurement_confidence", 0.5))),
                "--max_measurement_uncertainty_px",
                str(float(common.get("max_measurement_uncertainty_px", 5.0))),
            ]
        )
        if bool(common.get("measurement_amp", True)):
            cmd.append("--measurement_amp")
    return cmd


def _summary_score(path: Path, metric: str) -> tuple[float, float]:
    data = json.loads(path.read_text())
    pose = dict(data.get("pose") or {})
    if metric == "recall_0_5m_5deg":
        return -float(pose.get("recall_0_5m_5deg", 0.0)), float(pose.get("median_translation_error_m", 1e9))
    if metric == "median_translation_error_m":
        return float(pose.get("median_translation_error_m", 1e9)), float(pose.get("median_rotation_error_deg", 1e9))
    raise ValueError("selection_metric must be 'recall_0_5m_5deg' or 'median_translation_error_m'")


def select_top_methods(output_root: Path, methods: Sequence[str], *, top_n: int, metric: str) -> tuple[str, ...]:
    scored: list[tuple[tuple[float, float], str]] = []
    for method in methods:
        summary = output_root / "eval" / method / "meas0" / "summary.json"
        if summary.exists():
            scored.append((_summary_score(summary, metric), str(method)))
    scored.sort(key=lambda item: item[0])
    return tuple(method for _score, method in scored[: max(0, int(top_n))])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_observations_jsonl", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--measurement_checkpoint", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_AGGREGATION_METHODS))
    parser.add_argument("--measurement_ks", default="0")
    parser.add_argument("--measurement_selection_strategy", default="score_spatial", choices=("score_spatial", "coarse_pnp_inliers"))
    parser.add_argument("--select_top_n", type=int, default=0)
    parser.add_argument("--selection_metric", default="recall_0_5m_5deg", choices=("recall_0_5m_5deg", "median_translation_error_m"))
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print(" ".join(str(item) for item in cmd), flush=True)
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    methods = parse_csv_strings(str(args.methods))
    measurement_ks = parse_csv_ints(str(args.measurement_ks))
    common = vars(args).copy()
    common["measurement_checkpoint"] = str(args.measurement_checkpoint or args.matcha_joint_checkpoint)
    common["measurement_amp"] = True
    common["output_root"] = str(output_root)

    for method in methods:
        bank_path = output_root / "banks" / f"projected_observations_{method}.npz"
        bank_summary = output_root / "banks" / f"projected_observations_{method}.summary.json"
        if not bool(args.skip_build) and (bool(args.rebuild) or not bank_path.exists()):
            _run(
                build_projected_bank_command(common, method=method, output_index=str(bank_path), summary_json=str(bank_summary)),
                dry_run=bool(args.dry_run),
            )
        eval_common = {**common, "projected_landmark_cache": str(bank_path)}
        eval_dir = output_root / "eval" / method / "meas0"
        eval_common["output_dir"] = str(eval_dir)
        _run(build_eval_command(eval_common, measurement_k=0), dry_run=bool(args.dry_run))

    methods_for_measurement = methods
    if int(args.select_top_n) > 0 and not bool(args.dry_run):
        methods_for_measurement = select_top_methods(
            output_root,
            methods,
            top_n=int(args.select_top_n),
            metric=str(args.selection_metric),
        )
    for method in methods_for_measurement:
        for measurement_k in measurement_ks:
            if int(measurement_k) <= 0:
                continue
            bank_path = output_root / "banks" / f"projected_observations_{method}.npz"
            eval_common = {**common, "projected_landmark_cache": str(bank_path)}
            eval_common["output_dir"] = str(output_root / "eval" / method / f"meas{int(measurement_k)}")
            _run(build_eval_command(eval_common, measurement_k=int(measurement_k)), dry_run=bool(args.dry_run))


if __name__ == "__main__":  # pragma: no cover
    main()
