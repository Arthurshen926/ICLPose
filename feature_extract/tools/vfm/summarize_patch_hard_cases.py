"""Summarize patch-to-3D hard-case subsets from per-query evaluation rows."""

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
    rows: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row["query_id"])
        rows[query_id] = row
    return rows


def _float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def _nested_float(row: dict[str, object], outer: str, inner: str) -> float | None:
    payload = row.get(outer)
    if not isinstance(payload, dict):
        return None
    value = payload.get(inner)
    if value is None:
        return None
    return float(value)


def _success(row: dict[str, object], key: str = "success_25cm_10deg") -> bool:
    return bool(row.get(key))


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = float(q) * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _subset_masks(anchor: dict[str, dict[str, object]], raw: dict[str, dict[str, object]] | None) -> dict[str, set[str]]:
    query_ids = set(anchor)
    visible_values = [
        value
        for row in anchor.values()
        for value in [_float(row, "visible_landmark_recall")]
        if value is not None
    ]
    landmark_values = [
        value
        for row in anchor.values()
        for value in [_float(row, "submap_gt_visible_tracks")]
        if value is not None
    ]
    inlier_values = [
        value
        for row in anchor.values()
        for value in [_float(row, "pnp_inlier_count")]
        if value is not None
    ]
    visible_q25 = _quantile(visible_values, 0.25)
    landmark_q25 = _quantile(landmark_values, 0.25)
    inlier_q25 = _quantile(inlier_values, 0.25)
    subsets: dict[str, set[str]] = {"all": set(query_ids)}
    subsets["reference_top1_far"] = {
        query_id
        for query_id, row in anchor.items()
        if (
            (_nested_float(row, "reference_prior", "top1_translation_error_m") or 0.0) > 1.0
            or (_nested_float(row, "reference_prior", "top1_rotation_error_deg") or 0.0) > 10.0
        )
    }
    if visible_q25 is not None:
        subsets["reference_top10_weak_coverage_q25"] = {
            query_id
            for query_id, row in anchor.items()
            if (_float(row, "visible_landmark_recall") is not None and _float(row, "visible_landmark_recall") <= visible_q25)
        }
    if landmark_q25 is not None:
        subsets["low_visible_landmark_count_q25"] = {
            query_id
            for query_id, row in anchor.items()
            if (_float(row, "submap_gt_visible_tracks") is not None and _float(row, "submap_gt_visible_tracks") <= landmark_q25)
        }
    if inlier_q25 is not None:
        subsets["low_random128_inlier_count_q25"] = {
            query_id
            for query_id, row in anchor.items()
            if (_float(row, "pnp_inlier_count") is not None and _float(row, "pnp_inlier_count") <= inlier_q25)
        }
    subsets["random128_fail_s25"] = {query_id for query_id, row in anchor.items() if not _success(row)}
    subsets["random128_large_pose_error"] = {
        query_id
        for query_id, row in anchor.items()
        if (
            (_float(row, "translation_error_m") is None or (_float(row, "translation_error_m") or 0.0) > 0.5)
            or (_float(row, "rotation_error_deg") is None or (_float(row, "rotation_error_deg") or 0.0) > 10.0)
        )
    }
    if raw is not None:
        subsets["raw_fail_s25"] = {
            query_id for query_id, row in raw.items() if query_id in query_ids and not _success(row)
        }
    return subsets


def _summarize_method(
    scene: str,
    method: str,
    rows: dict[str, dict[str, object]],
    subset_name: str,
    query_ids: set[str],
) -> dict[str, object]:
    selected = [rows[query_id] for query_id in sorted(query_ids) if query_id in rows]
    translations = [_float(row, "translation_error_m") for row in selected]
    rotations = [_float(row, "rotation_error_deg") for row in selected]
    return {
        "scene": scene,
        "subset": subset_name,
        "method": method,
        "query_count": len(selected),
        "success_25cm_10deg": _mean([1.0 if _success(row, "success_25cm_10deg") else 0.0 for row in selected]),
        "success_50cm_10deg": _mean([1.0 if _success(row, "success_50cm_10deg") else 0.0 for row in selected]),
        "median_translation_error_m": _median([value for value in translations if value is not None]),
        "median_rotation_error_deg": _median([value for value in rotations if value is not None]),
        "mean_pnp_inlier_count": _mean(
            [value for row in selected for value in [_float(row, "pnp_inlier_count")] if value is not None]
        ),
        "mean_pnp_inlier_patch_at_1": _mean(
            [
                value
                for row in selected
                for value in [_nested_float(row, "patch_geometry", "pnp_inlier_patch_at_1")]
                if value is not None
            ]
        ),
    }


def _pair_rows(
    scene: str,
    baseline_name: str,
    baseline: dict[str, dict[str, object]],
    method_name: str,
    method: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    common = sorted(set(baseline).intersection(method))
    baseline_fail = {query_id for query_id in common if not _success(baseline[query_id])}
    baseline_success = {query_id for query_id in common if _success(baseline[query_id])}
    rescued = {query_id for query_id in baseline_fail if _success(method[query_id])}
    worsened = {query_id for query_id in baseline_success if not _success(method[query_id])}
    return [
        {
            "scene": scene,
            "baseline": baseline_name,
            "method": method_name,
            "subset": f"{baseline_name}_fail_s25",
            "query_count": len(baseline_fail),
            "rescued_count": len(rescued),
            "rescued_rate": None if not baseline_fail else float(len(rescued) / len(baseline_fail)),
            "worsened_count": len(worsened),
            "worsened_rate": None if not baseline_success else float(len(worsened) / len(baseline_success)),
        }
    ]


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(rows: Sequence[dict[str, object]], pair_rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scene",
        "subset",
        "method",
        "query_count",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "mean_pnp_inlier_count",
        "mean_pnp_inlier_patch_at_1",
    )
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    if pair_rows:
        pair_fields = ("scene", "baseline", "method", "subset", "query_count", "rescued_count", "rescued_rate", "worsened_count", "worsened_rate")
        lines.extend(["", "| " + " | ".join(pair_fields) + " |", "| " + " | ".join(["---"] * len(pair_fields)) + " |"])
        for row in pair_rows:
            lines.append("| " + " | ".join(_format(row.get(field)) for field in pair_fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize patch-to-3D hard-case subsets")
    parser.add_argument("--run", action="append", required=True, help="scene,method,evaluation_rows_jsonl")
    parser.add_argument("--anchor_method", default="random128")
    parser.add_argument("--raw_method", default="raw1280")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    by_scene: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for item in args.run:
        scene, method, path = _parse_run(item)
        by_scene.setdefault(scene, {})[method] = _load_rows(path)

    rows: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    for scene, methods in sorted(by_scene.items()):
        if args.anchor_method not in methods:
            raise ValueError(f"anchor method {args.anchor_method!r} missing for {scene}")
        anchor = methods[args.anchor_method]
        raw = methods.get(args.raw_method)
        subsets = _subset_masks(anchor, raw)
        for subset_name, query_ids in sorted(subsets.items()):
            if not query_ids:
                continue
            for method, method_rows in sorted(methods.items()):
                rows.append(_summarize_method(scene, method, method_rows, subset_name, query_ids))
        for method, method_rows in sorted(methods.items()):
            if method == args.anchor_method:
                continue
            pairwise.extend(_pair_rows(scene, args.anchor_method, anchor, method, method_rows))

    report = {"stage": "patch_hard_case_summary", "anchor_method": args.anchor_method, "rows": rows, "pairwise": pairwise}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_markdown(rows, pairwise, Path(args.output_md))


if __name__ == "__main__":
    main()
