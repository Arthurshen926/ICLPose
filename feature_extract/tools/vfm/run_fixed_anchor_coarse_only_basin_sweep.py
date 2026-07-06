"""Run pose-basin sweep for the fixed-anchor coarse-only baseline."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_TRANSLATION_MAGNITUDES_M = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_ROTATION_MAGNITUDES_DEG = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_TRANSLATION_DIRECTIONS = {
    "xpos": (1.0, 0.0, 0.0),
    "xneg": (-1.0, 0.0, 0.0),
    "ypos": (0.0, 1.0, 0.0),
    "yneg": (0.0, -1.0, 0.0),
    "zpos": (0.0, 0.0, 1.0),
    "zneg": (0.0, 0.0, -1.0),
}
DEFAULT_ROTATION_AXES = ("x", "y", "z")


def _csv_floats(text: str, defaults: Sequence[float]) -> tuple[float, ...]:
    if not str(text).strip():
        return tuple(float(v) for v in defaults)
    return tuple(float(item) for item in str(text).split(",") if item.strip())


def _format_xyz(values: Sequence[float]) -> str:
    return ",".join(f"{float(value):.6g}" for value in values)


def _mag_label(value: float, suffix: str) -> str:
    return f"{float(value):.3f}".replace(".", "p") + suffix


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--query_token_cache_dir", default="")
    parser.add_argument("--render_rgb_depth_cache_root", default="")
    parser.add_argument("--feature_mode", default="radio_dual")
    parser.add_argument("--renderer", default="official_2dgs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--translation_magnitudes_m", default="")
    parser.add_argument("--rotation_magnitudes_deg", default="")
    parser.add_argument("--translation_directions", default="xpos,xneg,ypos,yneg,zpos,zneg")
    parser.add_argument("--rotation_axes", default="x,y,z")
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


def build_basin_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    translation_mags = _csv_floats(str(args.translation_magnitudes_m), DEFAULT_TRANSLATION_MAGNITUDES_M)
    rotation_mags = _csv_floats(str(args.rotation_magnitudes_deg), DEFAULT_ROTATION_MAGNITUDES_DEG)
    direction_names = [item.strip() for item in str(args.translation_directions).split(",") if item.strip()]
    axis_names = [item.strip() for item in str(args.rotation_axes).split(",") if item.strip()]
    requested = {item.strip() for item in str(args.labels).split(",") if item.strip()}
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
        base.extend(["--query_token_cache_dir", str(args.query_token_cache_dir), "--extract_query_features_from_image", "--skip_existing_query_tokens"])
    if bool(args.resume_eval_rows):
        base.extend(["--stream_rows", "--resume_existing_rows"])
    base.extend(str(item) for item in list(args.extra_eval_arg or []))
    jobs: list[tuple[str, list[str]]] = []
    output_root = Path(args.output_root)
    for mag in translation_mags:
        if float(mag) == 0.0:
            label = "gt"
            cmd = base + ["--output_dir", str(output_root / label), "--render_pose_mode", "gt"]
            if str(args.render_rgb_depth_cache_root):
                cmd.extend(["--render_rgb_depth_cache_dir", str(Path(args.render_rgb_depth_cache_root) / label), "--skip_existing_render_rgb_depth"])
            jobs.append((label, cmd))
            continue
        for name in direction_names:
            if name not in DEFAULT_TRANSLATION_DIRECTIONS:
                raise ValueError(f"unknown translation direction: {name}")
            direction = DEFAULT_TRANSLATION_DIRECTIONS[name]
            offset = tuple(float(mag) * float(component) for component in direction)
            label = f"trans_{name}_{_mag_label(float(mag), 'm')}"
            cmd = base + [
                "--output_dir",
                str(output_root / label),
                "--render_pose_mode",
                "gt_offset",
                f"--render_pose_world_offset={_format_xyz(offset)}",
            ]
            if str(args.render_rgb_depth_cache_root):
                cmd.extend(["--render_rgb_depth_cache_dir", str(Path(args.render_rgb_depth_cache_root) / label), "--skip_existing_render_rgb_depth"])
            jobs.append((label, cmd))
    for mag in rotation_mags:
        if float(mag) == 0.0:
            continue
        for axis in axis_names:
            if axis not in DEFAULT_ROTATION_AXES:
                raise ValueError(f"unknown rotation axis: {axis}")
            for sign_name, sign in (("pos", 1.0), ("neg", -1.0)):
                xyz = {
                    "x": (sign * float(mag), 0.0, 0.0),
                    "y": (0.0, sign * float(mag), 0.0),
                    "z": (0.0, 0.0, sign * float(mag)),
                }[axis]
                label = f"rot_{axis}{sign_name}_{_mag_label(float(mag), 'deg')}"
                cmd = base + [
                    "--output_dir",
                    str(output_root / label),
                    "--render_pose_mode",
                    "gt_rotation_offset",
                    f"--render_pose_rotation_offset_deg={_format_xyz(xyz)}",
                ]
                if str(args.render_rgb_depth_cache_root):
                    cmd.extend(["--render_rgb_depth_cache_dir", str(Path(args.render_rgb_depth_cache_root) / label), "--skip_existing_render_rgb_depth"])
                jobs.append((label, cmd))
    if requested:
        jobs = [(label, cmd) for label, cmd in jobs if label in requested]
    return jobs


def summarize(output_root: Path, labels: Sequence[str]) -> dict[str, object]:
    rows = []
    for label in labels:
        summary_path = Path(output_root) / label / "summary.json"
        if not summary_path.exists():
            rows.append({"label": label, "status": "missing", "summary": str(summary_path)})
            continue
        payload = json.loads(summary_path.read_text())
        metrics = dict(payload.get("metrics", {}))
        eval_rows_path = Path(output_root) / label / "rows.csv"
        success_3cm_1deg = metrics.get("success_3cm_1deg")
        if success_3cm_1deg is None and eval_rows_path.exists():
            with eval_rows_path.open(newline="") as f:
                eval_rows = list(csv.DictReader(f))
            ok = []
            for eval_row in eval_rows:
                try:
                    t = float(eval_row.get("translation_error_m", "nan"))
                    r = float(eval_row.get("rotation_error_deg", "nan"))
                except ValueError:
                    continue
                if np_isfinite(t) and np_isfinite(r):
                    ok.append(t <= 0.03 and r <= 1.0)
            success_3cm_1deg = (sum(1 for item in ok if item) / len(ok)) if ok else None
        protocol = _protocol_summary_from_rows(eval_rows_path, label=label)
        rows.append(
            {
                "label": label,
                "status": "ok",
                "query_count": metrics.get("query_count"),
                "median_translation_error_m": metrics.get("median_translation_error_m"),
                "median_rotation_error_deg": metrics.get("median_rotation_error_deg"),
                "success_3cm_1deg": success_3cm_1deg,
                "success_10cm_5deg": metrics.get("success_10cm_5deg"),
                "mean_gt_precision_10px": metrics.get("mean_gt_precision_10px"),
                "mean_gt_precision_16px": metrics.get("mean_gt_precision_16px"),
                "pnp_solve_rate": metrics.get("pnp_solve_rate"),
                "summary": str(summary_path),
                **protocol,
            }
        )
    report = {
        "stage": "fixed_anchor_coarse_only_pose_basin_sweep",
        "output_root": str(output_root),
        "rows": rows,
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "basin_sweep_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if rows:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with (Path(output_root) / "basin_sweep_summary.tsv").open("w") as f:
            f.write("\t".join(keys) + "\n")
            for row in rows:
                f.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in keys) + "\n")
    return report


def _float_from_cell(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np_isfinite(out) else None


def _requested_from_label(label: str) -> tuple[float | None, float | None]:
    text = str(label)
    if text == "gt":
        return 0.0, 0.0
    if text.startswith("trans_") and text.endswith("m"):
        amount = text.rsplit("_", 1)[-1][:-1].replace("p", ".")
        try:
            return float(amount), 0.0
        except ValueError:
            return None, None
    if text.startswith("rot_") and text.endswith("deg"):
        amount = text.rsplit("_", 1)[-1][:-3].replace("p", ".")
        try:
            return 0.0, float(amount)
        except ValueError:
            return None, None
    return None, None


def _protocol_summary_from_rows(rows_path: Path, *, label: str = "") -> dict[str, object]:
    if not Path(rows_path).exists():
        return {}
    with Path(rows_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def values(name: str) -> list[float]:
        out = []
        for row in rows:
            item = _float_from_cell(row.get(name))
            if item is not None:
                out.append(float(item))
        return out

    realized_t = values("realized_initial_translation_error_m") or values("render_translation_error_m")
    realized_r = values("realized_initial_rotation_error_deg") or values("render_rotation_error_deg")
    requested_t = values("requested_translation_m")
    requested_r = values("requested_rotation_deg")
    fallback_t, fallback_r = _requested_from_label(str(label))
    if not requested_t and fallback_t is not None:
        requested_t = [float(fallback_t)]
    if not requested_r and fallback_r is not None:
        requested_r = [float(fallback_r)]
    output: dict[str, object] = {
        "requested_translation_m": None if not requested_t else float(max(requested_t)),
        "requested_rotation_deg": None if not requested_r else float(max(requested_r)),
        "max_realized_initial_translation_error_m": None if not realized_t else float(max(realized_t)),
        "max_realized_initial_rotation_error_deg": None if not realized_r else float(max(realized_r)),
    }
    if requested_r and max(requested_r) > 0.0 and requested_t and max(requested_t) <= 1e-12:
        output["protocol_pure_rotation_center_pass"] = bool(realized_t and max(realized_t) <= 1e-8)
    if requested_t and max(requested_t) > 0.0 and requested_r and max(requested_r) <= 1e-12:
        output["protocol_pure_translation_rotation_pass"] = bool(realized_r and max(realized_r) <= 1e-8)
    return output


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_basin_jobs(args)
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
