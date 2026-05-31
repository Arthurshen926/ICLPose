#!/usr/bin/env python3
"""Build the canonical sparse VFM patch-to-3D local pipeline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from feature_extract.vfm.local_pipeline_reporting import (
    failure_bucket_summary,
    load_jsonl_rows,
    metric_row_from_summary,
)


SCENE_KEYS = {
    "OldHospital": "oldhospital",
    "ShopFacade": "shopfacade",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _mean_std_value(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    if value is None:
        return None
    return float(value)


def _stage_group_row(scene: str, method: str, group: dict[str, Any], note: str) -> dict[str, Any]:
    query_count = 182 if scene == "OldHospital" else 103
    elapsed = group.get("total_elapsed_sec_mean")
    runtime = None if elapsed is None else float(elapsed) / float(query_count)
    return {
        "scene": scene,
        "method": method,
        "protocol_note": note,
        "query_count": query_count,
        "seed_count": group.get("seed_count"),
        "success_10cm_5deg": None,
        "success_25cm_10deg": group.get("success_25cm_10deg_mean"),
        "success_50cm_10deg": group.get("success_50cm_10deg_mean"),
        "median_translation_error_m": group.get("median_translation_error_m_mean"),
        "median_rotation_error_deg": group.get("median_rotation_error_deg_mean"),
        "mean_pnp_inlier_patch_at_1": group.get("mean_pnp_inlier_patch_at_1_mean"),
        "mean_pnp_inlier_count": group.get("mean_pnp_inlier_count_mean"),
        "pnp_solve_rate": None,
        "runtime_sec_per_query": runtime,
        "storage_ratio_vs_raw": group.get("storage_ratio_vs_raw_mean"),
    }


def _c2s_row(scene: str, block: dict[str, Any]) -> dict[str, Any]:
    metrics = block["c2s_conservative128_mean_std"]
    return {
        "scene": scene,
        "method": "C2-S",
        "protocol_note": "C2-S conservative128 3-seed mean, no LM refine",
        "query_count": 182 if scene == "OldHospital" else 103,
        "seed_count": 3,
        "success_10cm_5deg": None,
        "success_25cm_10deg": _mean_std_value(metrics, "success_25cm_10deg"),
        "success_50cm_10deg": _mean_std_value(metrics, "success_50cm_10deg"),
        "median_translation_error_m": _mean_std_value(metrics, "median_translation_error_m"),
        "median_rotation_error_deg": _mean_std_value(metrics, "median_rotation_error_deg"),
        "mean_pnp_inlier_patch_at_1": _mean_std_value(metrics, "mean_pnp_inlier_patch_at_1"),
        "mean_pnp_inlier_count": _mean_std_value(metrics, "mean_pnp_inlier_count"),
        "pnp_solve_rate": None,
        "runtime_sec_per_query": None,
        "storage_ratio_vs_raw": None,
    }


def _best_row_from_c25(scene: str, block: dict[str, Any]) -> dict[str, Any]:
    row = metric_row_from_summary(scene, "C2.5", block["best"])
    row["protocol_note"] = f"{block.get('best_name', 'C2.5 best')}, no LM refine"
    row["storage_ratio_vs_raw"] = None
    return row


def _summary_row(scene: str, method: str, path: Path, note: str) -> dict[str, Any] | None:
    summary = _load_json(path)
    if summary is None:
        return None
    row = metric_row_from_summary(scene, method, summary)
    row["protocol_note"] = note
    row["storage_ratio_vs_raw"] = None
    return row


def _index_c1_groups(c1: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for group in c1.get("groups", []):
        if isinstance(group, dict):
            out[(str(group.get("scene")), str(group.get("method")), int(group.get("output_dim", -1)))] = group
    return out


def build_canonical_table(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c1 = _load_json(repo / "output/vfm/stage_c1_patch_selector_final/stage_c1_summary.json") or {}
    c2s = _load_json(repo / "output/vfm/stage_c2s_precision_rescue/c2s_summary.json") or {}
    c25 = _load_json(repo / "output/vfm/stage_c25_pose_refinement/c25_summary.json") or {}
    c1_groups = _index_c1_groups(c1)

    c1_specs = [
        ("raw1280", ("raw", 1280)),
        ("random128", ("random", 128)),
        ("PCA512", ("pca", 512)),
        ("C1 learned128", ("learned", 128)),
    ]
    for scene in ("OldHospital", "ShopFacade"):
        for label, key in c1_specs:
            group = c1_groups.get((scene, key[0], key[1]))
            if group is not None:
                rows.append(_stage_group_row(scene, label, group, "Stage C1 aggregate, no LM refine"))

        scene_key = SCENE_KEYS[scene]
        c2s_scene = c2s.get("scenes", {}).get(scene_key)
        if isinstance(c2s_scene, dict) and "c2s_conservative128_mean_std" in c2s_scene:
            rows.append(_c2s_row(scene, c2s_scene))

        c25_scene = c25.get("scenes", {}).get(scene_key)
        if isinstance(c25_scene, dict) and "best" in c25_scene:
            rows.append(_best_row_from_c25(scene, c25_scene))

    lm_paths = {
        "OldHospital": repo
        / "output/vfm/stage_c26_oldhospital_robust_matching/oldhospital/colmap_pnp_thr075_refinelm_summary.json",
        "ShopFacade": repo
        / "output/vfm/stage_c26_oldhospital_robust_matching/shopfacade_c25_margin01_refinelm_summary.json",
    }
    for scene, path in lm_paths.items():
        row = _summary_row(
            scene,
            "C2.5+LM",
            path,
            "canonical C2.5 descriptor, top10, MNN top1, 0.75 stride, EPNP+LM",
        )
        if row is not None:
            rows.append(row)
    return rows


def build_upper_bound_table(repo: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "OldHospital",
            "oracle_patch_positives_top10",
            repo
            / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/oracle_patch_positives_top10_summary.json",
            "GT patch-positive correspondences in reference top10 pool",
        ),
        (
            "OldHospital",
            "oracle_filter_gt_correct_top10",
            repo
            / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/oracle_filter_gt_correct_top10_summary.json",
            "current C2.5 matches filtered to GT patch-correct matches",
        ),
        (
            "OldHospital",
            "gt_visible_capped20k",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/c25_lm_gtvisible_summary.json",
            "GT-visible submap diagnostic capped to 20k landmarks, not a true uncapped oracle",
        ),
    ]
    rows = []
    for scene, method, path, note in specs:
        row = _summary_row(scene, method, path, note)
        if row is not None:
            rows.append(row)
    return rows


def build_solver_table(repo: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "OldHospital",
            "EPNP+LM",
            repo
            / "output/vfm/stage_c26_oldhospital_robust_matching/oldhospital/colmap_pnp_thr075_refinelm_summary.json",
            "canonical default",
        ),
        (
            "OldHospital",
            "AP3P+LM",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/solver_ap3p_lm_summary.json",
            "backend sweep",
        ),
        (
            "OldHospital",
            "SQPNP+LM",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/solver_sqpnp_lm_summary.json",
            "backend sweep",
        ),
        (
            "OldHospital",
            "EPNP+VVS",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/solver_epnp_vvs_summary.json",
            "backend sweep",
        ),
    ]
    rows = []
    for scene, method, path, note in specs:
        row = _summary_row(scene, method, path, note)
        if row is not None:
            rows.append(row)
    return rows


def build_soft_order_table(repo: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "OldHospital",
            "soft_composite_top800",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/soft_composite_top800_summary.json",
            "soft composite ordering plus top800 PnP preselection",
        ),
        (
            "OldHospital",
            "soft_composite_top600",
            repo / "output/vfm/stage_c27_canonical_local_pipeline/oldhospital/soft_composite_top600_summary.json",
            "soft composite ordering plus top600 PnP preselection",
        ),
    ]
    rows = []
    for scene, method, path, note in specs:
        row = _summary_row(scene, method, path, note)
        if row is not None:
            rows.append(row)
    return rows


def build_failure_analysis(repo: Path) -> dict[str, Any] | None:
    current_path = repo / "output/vfm/stage_c26_oldhospital_robust_matching/oldhospital/colmap_pnp_thr075_refinelm_rows.jsonl"
    baseline_path = repo / "output/vfm/stage_c26_oldhospital_robust_matching/oldhospital/colmap_baseline_thr075_rows.jsonl"
    if not current_path.exists() or not baseline_path.exists():
        return None
    return failure_bucket_summary(
        load_jsonl_rows(current_path),
        baseline_rows=load_jsonl_rows(baseline_path),
        success_key="success_25cm_10deg",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    metric_cols = [
        "scene",
        "method",
        "success_10cm_5deg",
        "success_25cm_10deg",
        "success_50cm_10deg",
        "median_translation_error_m",
        "median_rotation_error_deg",
        "mean_pnp_inlier_patch_at_1",
        "mean_pnp_inlier_count",
        "pnp_solve_rate",
        "runtime_sec_per_query",
    ]
    lines = [
        "# Canonical Sparse VFM Local Pipeline",
        "",
        "Protocol: `feature_extract/configs/vfm/protocol/canonical_local_pipeline.yaml`.",
        "",
        "## Canonical Table",
        "",
        _markdown_table(report["canonical_table"], metric_cols),
        "",
        "## OldHospital Upper Bounds",
        "",
        _markdown_table(report["upper_bounds"], metric_cols + ["protocol_note"]),
        "",
        "## OldHospital Solver Sweep",
        "",
        _markdown_table(report["solver_sweep"], metric_cols + ["protocol_note"]),
        "",
        "## OldHospital Soft PnP Ordering",
        "",
        _markdown_table(report["soft_order"], metric_cols + ["protocol_note"]),
        "",
        "## Failure Buckets",
        "",
    ]
    failure = report.get("failure_analysis") or {}
    buckets = failure.get("buckets", {})
    bucket_rows = [
        {"bucket": name, "count": data.get("count"), "examples": ", ".join(data.get("examples", [])[:5])}
        for name, data in buckets.items()
    ]
    lines.append(_markdown_table(bucket_rows, ["bucket", "count", "examples"]))
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- C2.5+LM is the fixed local baseline for the next stage.",
            "- Oracle patch positives in the same top10 pool reach near-perfect S@25, so OldHospital is not limited by landmark geometry or patch uncertainty alone.",
            "- Filtering current matches to GT-correct correspondences is much stronger than current matching, so match confidence and false correspondence rejection remain the main bottleneck.",
            "- The capped GT-visible run is a diagnostic only; it is not a clean oracle submap upper bound because the 20k cap changes the candidate pool.",
            "- AP3P is worse; SQPNP+LM and EPNP+VVS are effectively tied with EPNP+LM, so the default remains EPNP+LM.",
            "- Soft top-N PnP preselection increases inlier Patch@1 slightly but hurts S@25/S@50, so it is recorded as a negative diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/vfm/stage_c27_canonical_local_pipeline"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "stage": "stage_c27_canonical_local_pipeline",
        "protocol": "feature_extract/configs/vfm/protocol/canonical_local_pipeline.yaml",
        "canonical_table": build_canonical_table(repo),
        "upper_bounds": build_upper_bound_table(repo),
        "solver_sweep": build_solver_table(repo),
        "soft_order": build_soft_order_table(repo),
        "failure_analysis": build_failure_analysis(repo),
    }
    (out_dir / "canonical_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, out_dir / "canonical_summary.md")
    print(f"wrote {out_dir / 'canonical_summary.json'}")
    print(f"wrote {out_dir / 'canonical_summary.md'}")


if __name__ == "__main__":
    main()
