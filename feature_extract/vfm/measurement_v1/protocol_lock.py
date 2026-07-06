from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    fieldnames = sorted({str(key) for row in values for key in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_float(value: object) -> float | None:
    try:
        item = float(value)
    except (TypeError, ValueError):
        return None
    if item != item:
        return None
    return item


def _metric_summary(rows: Sequence[dict[str, str]]) -> dict[str, float | int | None]:
    t_values = [_maybe_float(row.get("translation_error_m")) for row in rows]
    r_values = [_maybe_float(row.get("rotation_error_deg")) for row in rows]
    t = [float(value) for value in t_values if value is not None]
    r = [float(value) for value in r_values if value is not None]
    success_10cm = [
        1.0
        for tv, rv in zip(t_values, r_values)
        if tv is not None and rv is not None and float(tv) <= 0.10 and float(rv) <= 5.0
    ]
    valid_pose = [1.0 for tv, rv in zip(t_values, r_values) if tv is not None and rv is not None]
    return {
        "query_count": int(len({str(row.get("query_id", "")) for row in rows if str(row.get("query_id", "")).strip()})),
        "candidate_count": int(len(rows)),
        "median_translation_error_m": None if not t else float(median(t)),
        "median_rotation_error_deg": None if not r else float(median(r)),
        "success_10cm_5deg": None if not valid_pose else float(sum(success_10cm) / len(valid_pose)),
    }


def _choose_rank0(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: int(float(row.get("candidate_rank", 10**9) or 10**9)))[0]


def _choose_score(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -float(_maybe_float(row.get("pose_score")) or _maybe_float(row.get("learned_pose_score")) or -1e18),
            int(float(row.get("candidate_rank", 10**9) or 10**9)),
        ),
    )[0]


def _choose_oracle(rows: Sequence[dict[str, str]], key: str) -> dict[str, str] | None:
    scored = [(float(value), row) for row in rows if (value := _maybe_float(row.get(key))) is not None]
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0])[0][1]


def _group_by_query(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if query_id:
            grouped.setdefault(query_id, []).append(row)
    return grouped


def _replay_topk(candidate_rows: Sequence[dict[str, str]], topk: int) -> dict[str, Any]:
    selected_rows: dict[str, list[dict[str, str]]] = {"rank0": [], "score_selected": [], "oracle_solver_best": [], "oracle_initial_best": []}
    for _query_id, rows in sorted(_group_by_query(candidate_rows).items()):
        filtered = [
            row
            for row in rows
            if (_maybe_float(row.get("candidate_rank")) is not None and int(float(row.get("candidate_rank", 0))) < int(topk))
        ]
        choices = {
            "rank0": _choose_rank0(filtered),
            "score_selected": _choose_score(filtered),
            "oracle_solver_best": _choose_oracle(filtered, "translation_error_m"),
            "oracle_initial_best": _choose_oracle(filtered, "render_translation_error_m"),
        }
        for name, row in choices.items():
            if row is not None:
                selected_rows[name].append(row)
    return {name: _metric_summary(rows) for name, rows in selected_rows.items()}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "sha256": _sha256(path),
        "row_count": len(_read_csv(path)) if path.exists() and path.suffix == ".csv" else None,
    }


def _derive_query_summary(source: Path, output: Path) -> Path:
    source_query_summary = source / "query_summary.csv"
    if source_query_summary.exists():
        return source_query_summary
    rows_path = source / "rows.csv"
    target = output / "query_summary.csv"
    if rows_path.exists():
        rows = _read_csv(rows_path)
        _write_csv(target, rows)
    return target


def _derive_measurement_table(source: Path, output: Path) -> Path:
    source_measurement = source / "measurement_table.csv"
    if source_measurement.exists():
        return source_measurement
    match_table = source / "match_table.csv"
    target = output / "measurement_table.csv"
    if not match_table.exists():
        return target
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(_read_csv(match_table)):
        query_id = str(row.get("query_id", "")).strip()
        match_index = row.get("match_index", idx)
        render_index = row.get("render_index", row.get("anchor_id", ""))
        rows.append(
            {
                "measurement_id": f"{query_id}:{match_index}",
                "query_id": query_id,
                "match_index": match_index,
                "anchor_id": render_index,
                "render_index": render_index,
                "candidate_id": row.get("candidate_id", ""),
                "candidate_render_index": row.get("candidate_render_index", ""),
                "candidate_rank": row.get("candidate_rank", row.get("match_rank", "")),
                "query_x": row.get("query_x", ""),
                "query_y": row.get("query_y", ""),
                "query_mean_x": row.get("query_x", ""),
                "query_mean_y": row.get("query_y", ""),
                "render_x": row.get("render_x", ""),
                "render_y": row.get("render_y", ""),
                "world_x": row.get("world_x", ""),
                "world_y": row.get("world_y", ""),
                "world_z": row.get("world_z", ""),
                "p_assignment": row.get("confidence", ""),
                "p_visible": row.get("render_alpha", ""),
                "sigma_xx": row.get("measurement_sigma_xx", ""),
                "sigma_xy": row.get("measurement_sigma_xy", ""),
                "sigma_yy": row.get("measurement_sigma_yy", ""),
                "gt_reproj_error_px": row.get("gt_reproj_error_px", ""),
                "gt_correct_5px": row.get("gt_correct_5px", ""),
                "gt_correct_10px": row.get("gt_correct_10px", ""),
                "pnp_inlier": row.get("pnp_inlier", ""),
                "anchor_xyz_change_m": row.get("anchor_xyz_change_m", ""),
                "render_depth_change_m": row.get("render_depth_change_m", ""),
                "surface_switch_flag": row.get("surface_switch_flag", ""),
                "render_depth_gradient": row.get("render_depth_gradient", ""),
            }
        )
    _write_csv(target, rows)
    return target


def build_protocol_lock_report(
    *,
    eval_dir: Path,
    output_dir: Path,
    topks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    source = Path(eval_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    config = dict(summary.get("config", {}))
    inputs = dict(summary.get("inputs", {}))
    pose_candidate_table = source / "pose_candidate_table.csv"
    candidate_rows = _read_csv(pose_candidate_table) if pose_candidate_table.exists() else []
    replay = {f"top{int(k)}": _replay_topk(candidate_rows, int(k)) for k in topks}
    query_summary_path = _derive_query_summary(source, output)
    measurement_table_path = _derive_measurement_table(source, output)
    report = {
        "stage": "measurement_v1_protocol_lock",
        "source_eval_dir": str(source),
        "protocol": {
            "git_sha": _current_git_sha(Path.cwd()),
            "checkpoint": inputs.get("matcha_joint_checkpoint") or inputs.get("matcha_adapter_checkpoint") or "",
            "checkpoint_sha256": _sha256(Path(str(inputs.get("matcha_joint_checkpoint", ""))))
            if str(inputs.get("matcha_joint_checkpoint", ""))
            else None,
            "cache_schema_version": "measurement_v1_protocol_lock_v1",
            "query_manifest": inputs.get("query_manifest", ""),
            "render_pose_mode": config.get("render_pose_mode", ""),
            "matcher": {
                "match_mode": config.get("match_mode", ""),
                "matcha_eval_preset": config.get("matcha_eval_preset", ""),
                "feature_mode": config.get("feature_mode", ""),
            },
            "pnp": {
                "measurement_sigma_px": config.get("measurement_sigma_px", ""),
                "pnp_soft_order_mode": config.get("pnp_soft_order_mode", ""),
                "pnp_soft_order_top_n": config.get("pnp_soft_order_top_n", ""),
            },
            "scorer_enabled": bool(config.get("enable_pose_rescore", False) or config.get("pose_scorer_model", "")),
        },
        "artifacts": {
            "summary": _artifact(summary_path),
            "rows": _artifact(source / "rows.csv"),
            "match_table": _artifact(source / "match_table.csv"),
            "measurement_table": _artifact(measurement_table_path),
            "pose_candidate_table": _artifact(pose_candidate_table),
            "query_summary": _artifact(query_summary_path),
            "render_cache_manifest": _artifact(source / "render_cache_manifest.csv"),
        },
        "deterministic_replay": replay,
    }
    (output / "protocol_lock.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _current_git_sha(cwd: Path) -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None
