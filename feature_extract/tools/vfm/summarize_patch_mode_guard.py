"""Evaluate per-query guarded selection across patch matching mode outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Optional, Sequence


def _parse_run(text: str) -> tuple[str, str, Path]:
    parts = [part.strip() for part in text.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--run must be formatted as scene,method,evaluation_rows_jsonl")
    return parts[0], parts[1], Path(parts[2])


def _load_rows(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _nested_float(row: dict[str, object], outer: str, inner: str, default: float = 0.0) -> float:
    payload = row.get(outer)
    if not isinstance(payload, dict):
        return default
    value = payload.get(inner)
    return default if value is None else float(value)


def _success(row: dict[str, object], key: str) -> bool:
    return bool(row.get(key))


def _pose_cost(row: dict[str, object]) -> float:
    translation = row.get("translation_error_m")
    rotation = row.get("rotation_error_deg")
    if translation is None or rotation is None:
        return float("inf")
    return float(translation) + 0.01 * float(rotation)


def _policy_key(policy: str, row: dict[str, object]) -> tuple[float, ...]:
    if policy == "inlier_count":
        return (_float(row, "pnp_inlier_count"), _float(row, "pnp_inlier_ratio"))
    if policy == "inlier_ratio":
        return (_float(row, "pnp_inlier_ratio"), _float(row, "pnp_inlier_count"))
    if policy == "low_residual":
        return (
            -_nested_float(row, "pnp_reprojection", "pnp_reproj_inlier_median_px", default=float("inf")),
            _float(row, "pnp_inlier_count"),
        )
    if policy == "oracle_pose":
        return (-_pose_cost(row),)
    raise ValueError("unsupported guard policy")


def _choose(policy: str, candidates: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    if not candidates:
        raise ValueError("no candidates to choose from")
    method, row = max(candidates.items(), key=lambda item: _policy_key(policy, item[1]))
    return method, row


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _summarize(scene: str, policy: str, chosen: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    rows = [row for _method, row in chosen]
    methods = [method for method, _row in chosen]
    method_counts = {method: methods.count(method) for method in sorted(set(methods))}
    return {
        "scene": scene,
        "policy": policy,
        "query_count": len(rows),
        "method_counts": method_counts,
        "success_25cm_10deg": _mean([1.0 if _success(row, "success_25cm_10deg") else 0.0 for row in rows]),
        "success_50cm_10deg": _mean([1.0 if _success(row, "success_50cm_10deg") else 0.0 for row in rows]),
        "median_translation_error_m": _median([_float(row, "translation_error_m", default=float("nan")) for row in rows if row.get("translation_error_m") is not None]),
        "median_rotation_error_deg": _median([_float(row, "rotation_error_deg", default=float("nan")) for row in rows if row.get("rotation_error_deg") is not None]),
        "mean_pnp_inlier_count": _mean([_float(row, "pnp_inlier_count") for row in rows]),
        "mean_pnp_inlier_patch_at_1": _mean(
            [_nested_float(row, "patch_geometry", "pnp_inlier_patch_at_1") for row in rows]
        ),
    }


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ",".join(f"{key}:{value[key]}" for key in sorted(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scene",
        "policy",
        "query_count",
        "method_counts",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "mean_pnp_inlier_count",
        "mean_pnp_inlier_patch_at_1",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize guarded per-query patch mode selection")
    parser.add_argument("--run", action="append", required=True, help="scene,method,evaluation_rows_jsonl")
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        choices=("inlier_count", "inlier_ratio", "low_residual", "oracle_pose"),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    by_scene: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for item in args.run:
        scene, method, path = _parse_run(item)
        by_scene.setdefault(scene, {})[method] = _load_rows(path)

    policies = args.policy or ["inlier_count", "inlier_ratio", "low_residual", "oracle_pose"]
    rows = []
    for scene, methods in sorted(by_scene.items()):
        common = set.intersection(*(set(rows_by_query) for rows_by_query in methods.values()))
        for policy in policies:
            chosen = []
            for query_id in sorted(common):
                candidates = {method: rows_by_query[query_id] for method, rows_by_query in methods.items()}
                chosen.append(_choose(policy, candidates))
            rows.append(_summarize(scene, policy, chosen))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"stage": "patch_mode_guard_summary", "rows": rows}, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_markdown(rows, Path(args.output_md))


if __name__ == "__main__":
    main()
