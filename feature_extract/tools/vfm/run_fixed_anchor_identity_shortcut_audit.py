"""Run full GT-render identity-shortcut controls for the fixed-anchor baseline."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SHIFT_MAGNITUDES_CELLS = (1, 2, 4)


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _mean(values: Sequence[float | None]) -> float | None:
    good = [float(value) for value in values if value is not None]
    return float(sum(good) / len(good)) if good else None


def _median(values: Sequence[float | None]) -> float | None:
    good = [float(value) for value in values if value is not None]
    return float(statistics.median(good)) if good else None


def _success_from_rows(rows: Sequence[dict[str, str]], *, max_t_m: float, max_r_deg: float) -> float | None:
    flags: list[bool] = []
    for row in rows:
        t = _float_or_none(row.get("translation_error_m"))
        r = _float_or_none(row.get("rotation_error_deg"))
        if t is None or r is None:
            continue
        flags.append(float(t) <= float(max_t_m) and float(r) <= float(max_r_deg))
    return float(sum(1 for item in flags if item) / len(flags)) if flags else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--query_token_cache_dir", default="")
    parser.add_argument("--render_rgb_depth_cache_dir", default="")
    parser.add_argument("--feature_mode", default="radio_dual")
    parser.add_argument("--renderer", default="official_2dgs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--labels", default="")
    parser.add_argument("--extra_eval_arg", action="append", default=[])
    parser.add_argument("--resume_eval_rows", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.execute) and bool(args.dry_run):
        raise ValueError("Use either --execute or --dry_run, not both")
    if not bool(args.execute) and not bool(args.dry_run):
        args.dry_run = True
    return args


def _control_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"label": "learned_gt", "mode": "learned", "shift": (0, 0)},
        {"label": "same_cell_gt", "mode": "same_cell_identity", "shift": (0, 0)},
        {"label": "constant_gt", "mode": "constant", "shift": (0, 0)},
        {"label": "random_gt", "mode": "random_normalized", "shift": (0, 0)},
        {"label": "spatial_shuffle_gt", "mode": "spatial_permute_query", "shift": (0, 0)},
    ]
    for axis, vector in (("x", (1, 0)), ("y", (0, 1))):
        for magnitude in SHIFT_MAGNITUDES_CELLS:
            for sign_name, sign in (("pos", 1), ("neg", -1)):
                dx = int(vector[0] * magnitude * sign)
                dy = int(vector[1] * magnitude * sign)
                specs.append(
                    {
                        "label": f"shift_{axis}{sign_name}_{magnitude}cell_gt",
                        "mode": "shift_query_cells",
                        "shift": (dx, dy),
                    }
                )
    return specs


def build_audit_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    requested = {item.strip() for item in str(args.labels).split(",") if item.strip()}
    output_root = Path(args.output_root)
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
        "matcha_c2f",
        "--matcha_eval_preset",
        "fixed_anchor_coarse_only_v1",
        "--matcha_joint_checkpoint",
        str(args.matcha_joint_checkpoint),
        "--render_pose_mode",
        "gt",
        "--max_queries",
        str(int(args.max_queries)),
        "--view_selection",
        str(args.view_selection),
        "--pose_update_iterations",
        "1",
        "--max_keypoints",
        "1000",
        "--visualize_limit",
        "0",
    ]
    if str(args.query_token_cache_dir):
        base.extend(
            [
                "--query_token_cache_dir",
                str(args.query_token_cache_dir),
                "--extract_query_features_from_image",
                "--skip_existing_query_tokens",
            ]
        )
    if str(args.render_rgb_depth_cache_dir):
        base.extend(["--render_rgb_depth_cache_dir", str(args.render_rgb_depth_cache_dir), "--skip_existing_render_rgb_depth"])
    if bool(args.resume_eval_rows):
        base.extend(["--stream_rows", "--resume_existing_rows"])
    base.extend(str(item) for item in list(args.extra_eval_arg or []))
    jobs: list[tuple[str, list[str]]] = []
    for spec in _control_specs():
        label = str(spec["label"])
        if requested and label not in requested:
            continue
        mode = str(spec["mode"])
        dx, dy = spec["shift"]  # type: ignore[misc]
        cmd = base + [
            "--output_dir",
            str(output_root / label),
            "--coarse_control_mode",
            mode,
            f"--coarse_control_shift_cells={int(dx)},{int(dy)}",
        ]
        jobs.append((label, cmd))
    return jobs


def summarize(output_root: Path, labels: Sequence[str]) -> dict[str, object]:
    rows = []
    for label in labels:
        summary_path = Path(output_root) / label / "summary.json"
        rows_path = Path(output_root) / label / "rows.csv"
        if not summary_path.exists():
            rows.append({"label": label, "status": "missing", "summary": str(summary_path)})
            continue
        payload = json.loads(summary_path.read_text())
        metrics = dict(payload.get("metrics", {}))
        eval_rows: list[dict[str, str]] = []
        if rows_path.exists():
            with rows_path.open(newline="") as f:
                eval_rows = list(csv.DictReader(f))
        mean_dx = _mean([_float_or_none(row.get("coarse_mean_cell_delta_x")) for row in eval_rows])
        mean_dy = _mean([_float_or_none(row.get("coarse_mean_cell_delta_y")) for row in eval_rows])
        shift_dx = _mean([_float_or_none(row.get("coarse_control_shift_dx_cells")) for row in eval_rows])
        shift_dy = _mean([_float_or_none(row.get("coarse_control_shift_dy_cells")) for row in eval_rows])
        slope_x = (-float(mean_dx) / float(shift_dx)) if mean_dx is not None and shift_dx not in (None, 0.0) else None
        slope_y = (-float(mean_dy) / float(shift_dy)) if mean_dy is not None and shift_dy not in (None, 0.0) else None
        rows.append(
            {
                "label": label,
                "status": "ok",
                "query_count": metrics.get("query_count"),
                "median_translation_error_m": metrics.get("median_translation_error_m"),
                "median_rotation_error_deg": metrics.get("median_rotation_error_deg"),
                "success_3cm_1deg": _success_from_rows(eval_rows, max_t_m=0.03, max_r_deg=1.0),
                "success_10cm_5deg": metrics.get("success_10cm_5deg"),
                "pnp_solve_rate": metrics.get("pnp_solve_rate"),
                "mean_gt_precision_5px": metrics.get("mean_gt_precision_5px"),
                "mean_gt_precision_10px": metrics.get("mean_gt_precision_10px"),
                "mean_gt_precision_16px": metrics.get("mean_gt_precision_16px"),
                "mean_coarse_same_cell_fraction": _mean(
                    [_float_or_none(row.get("coarse_same_cell_fraction")) for row in eval_rows]
                ),
                "median_coarse_same_cell_fraction": _median(
                    [_float_or_none(row.get("coarse_same_cell_fraction")) for row in eval_rows]
                ),
                "mean_coarse_cell_delta_x": mean_dx,
                "mean_coarse_cell_delta_y": mean_dy,
                "shift_dx_cells": shift_dx,
                "shift_dy_cells": shift_dy,
                "shift_tracking_slope_x": slope_x,
                "shift_tracking_slope_y": slope_y,
                "summary": str(summary_path),
            }
        )
    report = {
        "stage": "fixed_anchor_identity_shortcut_audit",
        "output_root": str(output_root),
        "rows": rows,
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "identity_shortcut_audit_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if rows:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with (Path(output_root) / "identity_shortcut_audit_summary.tsv").open("w") as f:
            f.write("\t".join(keys) + "\n")
            for row in rows:
                f.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in keys) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_audit_jobs(args)
    if bool(args.summarize_only):
        print(json.dumps(summarize(Path(args.output_root), [label for label, _ in jobs]), indent=2, sort_keys=True))
        return
    for label, cmd in jobs:
        if bool(args.skip_completed) and (Path(args.output_root) / label / "summary.json").exists():
            continue
        if bool(args.dry_run):
            print(shlex.join(cmd))
        else:
            subprocess.run(cmd, check=True)
    print(json.dumps(summarize(Path(args.output_root), [label for label, _ in jobs]), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
