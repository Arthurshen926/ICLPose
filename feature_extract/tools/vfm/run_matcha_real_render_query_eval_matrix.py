"""Run the full real render-query RADIO-MATCHA evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_TRANSLATION_OFFSETS_M = (0.05, 0.10, 0.25, 0.50)
DEFAULT_ROTATION_OFFSETS_DEG = (1.0, 3.0, 6.0, 10.0)
REFERENCE_MODES = ("reference_top1", "reference_top5", "reference_top10")


def _format_xyz(x: float, y: float, z: float) -> str:
    return f"{float(x):.3f},{float(y):.3f},{float(z):.3f}"


def _offset_label(value: float, suffix: str) -> str:
    return f"{float(value):.3f}".replace(".", "p") + suffix


def _normalize_negative_csv_option_args(
    argv: Sequence[str] | None,
    *,
    option_names: set[str],
) -> list[str] | None:
    if argv is None:
        return None
    items = list(argv)
    normalized: list[str] = []
    idx = 0
    while idx < len(items):
        item = str(items[idx])
        if item in option_names and idx + 1 < len(items):
            value = str(items[idx + 1])
            if value.startswith("-") and "," in value:
                normalized.append(f"{item}={value}")
                idx += 2
                continue
        normalized.append(item)
        idx += 1
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--matcha_joint_checkpoint", default="")
    parser.add_argument("--matcha_adapter_checkpoint", default="")
    parser.add_argument("--matcha_eval_preset", default="none")
    parser.add_argument("--feature_mode", default="radio_dual")
    parser.add_argument("--match_mode", default="matcha_c2f")
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--query_token_cache_dir", default="")
    parser.add_argument("--render_token_cache_root", default="")
    parser.add_argument("--render_rgb_depth_cache_root", default="")
    parser.add_argument("--extract_query_features_from_image", action="store_true")
    parser.add_argument("--skip_existing_query_tokens", action="store_true")
    parser.add_argument("--skip_existing_render_tokens", action="store_true")
    parser.add_argument("--skip_existing_render_rgb_depth", action="store_true")
    parser.add_argument("--extra_eval_arg", action="append", default=[])
    parser.add_argument("--renderer", default="official_2dgs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--pose_update_iterations", type=int, default=1)
    parser.add_argument("--rotation_search_offsets_deg", default="")
    parser.add_argument("--rotation_search_axis", default="y", choices=("x", "y", "z"))
    parser.add_argument("--labels", default="")
    parser.add_argument("--resume_eval_rows", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--diagnostic_labels", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--summary_tsv", default="")
    parser.add_argument("--comparison_baseline_root", default="")
    parser.add_argument("--comparison_candidate_root", default="")
    parser.add_argument("--comparison_json", default="")
    parser.add_argument("--comparison_tsv", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(
        _normalize_negative_csv_option_args(
            argv,
            option_names={"--rotation_search_offsets_deg"},
        )
    )
    if bool(args.execute) and bool(args.dry_run):
        raise ValueError("Use either --dry_run or --execute, not both")
    if not bool(args.dry_run) and not bool(args.execute):
        args.dry_run = True
    if not str(args.candidate_bank):
        raise ValueError("--candidate_bank is required for the reference top1/top5/top10 evaluation matrix")
    if str(args.matcha_joint_checkpoint) and str(args.matcha_adapter_checkpoint):
        raise ValueError("Use either --matcha_joint_checkpoint or --matcha_adapter_checkpoint, not both")
    return args


def _selected_label_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = [("gt", ["--render_pose_mode", "gt"])]
    for value in DEFAULT_TRANSLATION_OFFSETS_M:
        jobs.append(
            (
                f"gt_offset_{_offset_label(value, 'm')}",
                [
                    "--render_pose_mode",
                    "gt_offset",
                    "--render_pose_world_offset",
                    _format_xyz(value, 0.0, 0.0),
                ],
            )
        )
    rotation_search_args: list[str] = []
    if str(args.rotation_search_offsets_deg):
        rotation_search_args = [
            f"--render_pose_rotation_search_offsets_deg={str(args.rotation_search_offsets_deg)}",
            "--render_pose_rotation_search_axis",
            str(args.rotation_search_axis),
        ]
    for value in DEFAULT_ROTATION_OFFSETS_DEG:
        jobs.append(
            (
                f"gt_rotation_y_{_offset_label(value, 'deg')}",
                [
                    "--render_pose_mode",
                    "gt_rotation_offset",
                    "--render_pose_rotation_offset_deg",
                    _format_xyz(0.0, value, 0.0),
                ]
                + rotation_search_args,
            )
        )
    for mode in REFERENCE_MODES:
        jobs.append(
            (
                mode,
                [
                    "--render_pose_mode",
                    mode,
                    "--candidate_bank",
                    str(args.candidate_bank),
                ],
            )
        )

    requested_labels = {item.strip() for item in str(args.labels).split(",") if item.strip()}
    if requested_labels:
        known_labels = {label for label, _mode_args in jobs}
        unknown = sorted(requested_labels.difference(known_labels))
        if unknown:
            raise ValueError(f"unknown eval matrix label(s): {', '.join(unknown)}")
        jobs = [(label, mode_args) for label, mode_args in jobs if label in requested_labels]
    return jobs


def build_eval_matrix_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    base = [
        sys.executable,
        "-m",
        "feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose",
        "--query_manifest",
        str(args.query_manifest),
        "--query_pose_file",
        str(args.query_pose_file),
        "--image_root",
        str(args.image_root),
        "--gaussian_rgb_ply",
        str(args.gaussian_rgb_ply),
        "--renderer",
        str(args.renderer),
        "--device",
        str(args.device),
        "--feature_mode",
        str(args.feature_mode),
        "--match_mode",
        str(args.match_mode),
        "--matcha_eval_preset",
        str(args.matcha_eval_preset),
        "--max_queries",
        str(int(args.max_queries)),
        "--start_index",
        str(int(args.start_index)),
        "--view_selection",
        str(args.view_selection),
        "--pose_update_iterations",
        str(int(args.pose_update_iterations)),
    ]
    if str(args.matcha_joint_checkpoint):
        base.extend(["--matcha_joint_checkpoint", str(args.matcha_joint_checkpoint)])
    if str(args.matcha_adapter_checkpoint):
        base.extend(["--matcha_adapter_checkpoint", str(args.matcha_adapter_checkpoint)])
    if str(args.query_token_cache_dir):
        base.extend(["--query_token_cache_dir", str(args.query_token_cache_dir)])
    if bool(args.extract_query_features_from_image):
        base.append("--extract_query_features_from_image")
    if bool(args.skip_existing_query_tokens):
        base.append("--skip_existing_query_tokens")
    if bool(args.skip_existing_render_tokens):
        base.append("--skip_existing_render_tokens")
    if bool(args.skip_existing_render_rgb_depth):
        base.append("--skip_existing_render_rgb_depth")
    if bool(args.resume_eval_rows):
        base.extend(["--stream_rows", "--resume_existing_rows"])
    base.extend(str(item) for item in list(args.extra_eval_arg or []))

    output_root = Path(args.output_root)
    diagnostic_labels = {item.strip() for item in str(args.diagnostic_labels).split(",") if item.strip()}
    eval_jobs: list[tuple[str, list[str]]] = []
    for label, mode_args in _selected_label_jobs(args):
        cache_args: list[str] = []
        if str(args.render_token_cache_root):
            cache_args.extend(["--render_token_cache_dir", str(Path(args.render_token_cache_root) / label)])
        if str(args.render_rgb_depth_cache_root):
            cache_args.extend(["--render_rgb_depth_cache_dir", str(Path(args.render_rgb_depth_cache_root) / label)])
        diagnostic_args: list[str] = []
        if label in diagnostic_labels:
            diagnostic_args = [
                "--save_pose_candidate_table",
                "--save_match_table",
                "--match_table_stage",
                "candidate",
                "--save_coarse_oracle_table",
                "--save_coarse_oracle_candidate_table",
            ]
        eval_jobs.append((label, base + ["--output_dir", str(output_root / label)] + cache_args + diagnostic_args + mode_args))
    return eval_jobs


def build_eval_matrix_commands(args: argparse.Namespace) -> list[list[str]]:
    return [command for _label, command in build_eval_matrix_jobs(args)]


def write_matrix_summary(
    output_root: Path,
    *,
    labels: Sequence[str],
    summary_json: Path | None = None,
    summary_tsv: Path | None = None,
) -> dict[str, object]:
    root = Path(output_root)
    rows: list[dict[str, object]] = []
    for label in labels:
        path = root / str(label) / "summary.json"
        if not path.exists():
            rows.append({"label": str(label), "status": "missing", "summary_path": str(path)})
            continue
        payload = json.loads(path.read_text())
        metrics = dict(payload.get("metrics", {}))
        config = dict(payload.get("config", {}))
        rows.append(
            {
                "label": str(label),
                "status": "ok",
                "summary_path": str(path),
                "render_pose_mode": config.get("render_pose_mode", ""),
                **metrics,
            }
        )
    payload = {"stage": "matcha_real_render_query_eval_matrix", "output_root": str(root), "rows": rows}
    json_path = Path(summary_json) if summary_json is not None else root / "matrix_summary.json"
    tsv_path = Path(summary_tsv) if summary_tsv is not None else root / "matrix_summary.tsv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return payload


def _load_label_summary(root: Path, label: str) -> dict[str, object] | None:
    path = Path(root) / str(label) / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_controlled_comparison(
    baseline_root: Path,
    candidate_root: Path,
    *,
    labels: Sequence[str],
    output_json: Path,
    output_tsv: Path,
) -> dict[str, object]:
    metric_names = (
        "query_count",
        "median_translation_error_m",
        "median_rotation_error_deg",
        "success_10cm_5deg",
        "success_25cm_10deg",
        "mean_pnp_inlier_gt_precision_16px",
        "median_render_translation_error_m",
        "locked_to_render_within_3cm_rate",
    )
    rows: list[dict[str, object]] = []
    for label in labels:
        baseline = _load_label_summary(Path(baseline_root), str(label))
        candidate = _load_label_summary(Path(candidate_root), str(label))
        row: dict[str, object] = {
            "label": str(label),
            "baseline_status": "missing" if baseline is None else "ok",
            "candidate_status": "missing" if candidate is None else "ok",
        }
        baseline_metrics = {} if baseline is None else dict(baseline.get("metrics", {}))
        candidate_metrics = {} if candidate is None else dict(candidate.get("metrics", {}))
        for name in metric_names:
            b_value = baseline_metrics.get(name)
            c_value = candidate_metrics.get(name)
            row[f"baseline_{name}"] = b_value
            row[f"candidate_{name}"] = c_value
            try:
                row[f"{name}_delta"] = float(c_value) - float(b_value)
            except (TypeError, ValueError):
                row[f"{name}_delta"] = None
        rows.append(row)
    payload = {
        "stage": "matcha_real_render_query_controlled_comparison",
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_eval_matrix_jobs(args)
    labels = [label for label, _command in jobs]
    if bool(args.summarize_only):
        write_matrix_summary(
            Path(args.output_root),
            labels=labels,
            summary_json=Path(args.summary_json) if str(args.summary_json) else None,
            summary_tsv=Path(args.summary_tsv) if str(args.summary_tsv) else None,
        )
        if str(args.comparison_baseline_root) and str(args.comparison_candidate_root):
            write_controlled_comparison(
                Path(args.comparison_baseline_root),
                Path(args.comparison_candidate_root),
                labels=labels,
                output_json=Path(args.comparison_json)
                if str(args.comparison_json)
                else Path(args.output_root) / "old_vs_new_controlled_comparison.json",
                output_tsv=Path(args.comparison_tsv)
                if str(args.comparison_tsv)
                else Path(args.output_root) / "old_vs_new_controlled_comparison.tsv",
            )
        return
    for label, command in jobs:
        if bool(args.dry_run):
            print(shlex.join(command))
        else:
            if bool(args.skip_completed) and (Path(args.output_root) / label / "summary.json").exists():
                continue
            subprocess.run(command, check=True)
    if bool(args.execute):
        write_matrix_summary(
            Path(args.output_root),
            labels=labels,
            summary_json=Path(args.summary_json) if str(args.summary_json) else None,
            summary_tsv=Path(args.summary_tsv) if str(args.summary_tsv) else None,
        )
        if str(args.comparison_baseline_root) and str(args.comparison_candidate_root):
            write_controlled_comparison(
                Path(args.comparison_baseline_root),
                Path(args.comparison_candidate_root),
                labels=labels,
                output_json=Path(args.comparison_json)
                if str(args.comparison_json)
                else Path(args.output_root) / "old_vs_new_controlled_comparison.json",
                output_tsv=Path(args.comparison_tsv)
                if str(args.comparison_tsv)
                else Path(args.output_root) / "old_vs_new_controlled_comparison.tsv",
            )


if __name__ == "__main__":
    main()
