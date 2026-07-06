"""Run a fixed RGB-patch pose-proxy matrix over methods and residual bins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.tools.vfm.eval_rgb_patch_pose_proxy import evaluate_rgb_patch_pose_proxy


METHODS = {
    "measurement": ("query_pred_x", "query_pred_y"),
    "center": ("center_x", "center_y"),
    "oracle_gt": ("query_gt_x", "query_gt_y"),
}

MATRIX_FIELDNAMES = [
    "bin_label",
    "method",
    "min_baseline_epe_px",
    "max_baseline_epe_px",
    "query_count",
    "success_count",
    "success_rate",
    "kept_measurement_count",
    "matched_measurement_count",
    "missing_xyz_count",
    "missing_query_count",
    "missing_camera_count",
    "median_match_count",
    "match_count_p10",
    "median_translation_error_m",
    "translation_error_p90_m",
    "median_rotation_error_deg",
    "rotation_error_p90_deg",
    "rate_3cm_1deg",
    "rate_5cm_2deg",
    "rate_10cm_5deg",
    "summary_path",
]

COMPARISON_FIELDNAMES = [
    "bin_label",
    "comparison",
    "candidate_method",
    "reference_method",
    "translation_median_improvement_m",
    "translation_p90_improvement_m",
    "rotation_median_improvement_deg",
    "rotation_p90_improvement_deg",
    "rate_3cm_1deg_delta",
    "rate_5cm_2deg_delta",
    "rate_10cm_5deg_delta",
    "candidate_better_median_translation",
    "candidate_better_p90_translation",
    "candidate_better_rate_3cm_1deg",
]


def _bin_label(value: float) -> str:
    return f"bin_{float(value):.3f}".replace(".", "p")


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in MATRIX_FIELDNAMES})


def _write_comparison_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in COMPARISON_FIELDNAMES})


def _summary_row(
    *,
    bin_label: str,
    method: str,
    min_baseline_epe_px: float | None,
    max_baseline_epe_px: float | None,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "bin_label": str(bin_label),
        "method": str(method),
        "min_baseline_epe_px": "" if min_baseline_epe_px is None else float(min_baseline_epe_px),
        "max_baseline_epe_px": "" if max_baseline_epe_px is None else float(max_baseline_epe_px),
        "query_count": int(summary.get("query_count", 0)),
        "success_count": int(summary.get("success_count", 0)),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "kept_measurement_count": int(summary.get("kept_measurement_count", 0)),
        "matched_measurement_count": int(summary.get("matched_measurement_count", 0)),
        "missing_xyz_count": int(summary.get("missing_xyz_count", 0)),
        "missing_query_count": int(summary.get("missing_query_count", 0)),
        "missing_camera_count": int(summary.get("missing_camera_count", 0)),
        "median_match_count": float(summary.get("median_match_count", 0.0)),
        "match_count_p10": float(summary.get("match_count_p10", 0.0)),
        "median_translation_error_m": float(summary.get("median_translation_error_m", float("inf"))),
        "translation_error_p90_m": float(summary.get("translation_error_p90_m", float("inf"))),
        "median_rotation_error_deg": float(summary.get("median_rotation_error_deg", float("inf"))),
        "rotation_error_p90_deg": float(summary.get("rotation_error_p90_deg", float("inf"))),
        "rate_3cm_1deg": float(summary.get("rate_3cm_1deg", 0.0)),
        "rate_5cm_2deg": float(summary.get("rate_5cm_2deg", 0.0)),
        "rate_10cm_5deg": float(summary.get("rate_10cm_5deg", 0.0)),
        "summary_path": str(summary.get("outputs", {}).get("summary", "")),
    }


def _metric(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value)


def _comparison_row(
    *,
    bin_label: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_method = str(candidate["method"])
    reference_method = str(reference["method"])
    median_t_improvement = _metric(reference, "median_translation_error_m") - _metric(
        candidate, "median_translation_error_m"
    )
    p90_t_improvement = _metric(reference, "translation_error_p90_m") - _metric(
        candidate, "translation_error_p90_m"
    )
    median_r_improvement = _metric(reference, "median_rotation_error_deg") - _metric(
        candidate, "median_rotation_error_deg"
    )
    p90_r_improvement = _metric(reference, "rotation_error_p90_deg") - _metric(
        candidate, "rotation_error_p90_deg"
    )
    rate_3_delta = _metric(candidate, "rate_3cm_1deg") - _metric(reference, "rate_3cm_1deg")
    rate_5_delta = _metric(candidate, "rate_5cm_2deg") - _metric(reference, "rate_5cm_2deg")
    rate_10_delta = _metric(candidate, "rate_10cm_5deg") - _metric(reference, "rate_10cm_5deg")
    return {
        "bin_label": str(bin_label),
        "comparison": f"{candidate_method}_vs_{reference_method}",
        "candidate_method": candidate_method,
        "reference_method": reference_method,
        "translation_median_improvement_m": float(median_t_improvement),
        "translation_p90_improvement_m": float(p90_t_improvement),
        "rotation_median_improvement_deg": float(median_r_improvement),
        "rotation_p90_improvement_deg": float(p90_r_improvement),
        "rate_3cm_1deg_delta": float(rate_3_delta),
        "rate_5cm_2deg_delta": float(rate_5_delta),
        "rate_10cm_5deg_delta": float(rate_10_delta),
        "candidate_better_median_translation": bool(median_t_improvement > 0.0),
        "candidate_better_p90_translation": bool(p90_t_improvement > 0.0),
        "candidate_better_rate_3cm_1deg": bool(rate_3_delta > 0.0),
    }


def _build_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_bin_method: dict[tuple[str, str], Mapping[str, Any]] = {}
    bin_labels: list[str] = []
    for row in rows:
        bin_label = str(row["bin_label"])
        method = str(row["method"])
        by_bin_method[(bin_label, method)] = row
        if bin_label not in bin_labels:
            bin_labels.append(bin_label)

    comparison_rows: list[dict[str, Any]] = []
    for bin_label in bin_labels:
        candidate = by_bin_method.get((bin_label, "measurement"))
        if candidate is None:
            continue
        for reference_method in ("center", "oracle_gt"):
            reference = by_bin_method.get((bin_label, reference_method))
            if reference is None:
                continue
            comparison_rows.append(_comparison_row(bin_label=bin_label, candidate=candidate, reference=reference))
    return comparison_rows


def run_rgb_patch_pose_proxy_matrix(
    *,
    prediction_rows_csv: Path,
    model_dir: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    baseline_bins_px: Sequence[float],
    bin_tolerance_px: float = 0.02,
    include_all: bool = True,
    dustbin_threshold: float = 0.5,
    reprojection_error_px: float = 8.0,
    min_inliers: int = 4,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bins: list[tuple[str, float | None, float | None]] = []
    if bool(include_all):
        bins.append(("all", None, None))
    for value in baseline_bins_px:
        center = float(value)
        tol = float(bin_tolerance_px)
        bins.append((_bin_label(center), center - tol, center + tol))

    rows: list[dict[str, Any]] = []
    for label, min_epe, max_epe in bins:
        for method, (x_key, y_key) in METHODS.items():
            run_dir = output / str(label) / str(method)
            summary = evaluate_rgb_patch_pose_proxy(
                prediction_rows_csv=Path(prediction_rows_csv),
                model_dir=Path(model_dir),
                output_dir=run_dir,
                image_width=int(image_width),
                image_height=int(image_height),
                prediction_x_key=x_key,
                prediction_y_key=y_key,
                min_baseline_epe_px=min_epe,
                max_baseline_epe_px=max_epe,
                dustbin_threshold=float(dustbin_threshold),
                reprojection_error_px=float(reprojection_error_px),
                min_inliers=int(min_inliers),
            )
            rows.append(
                _summary_row(
                    bin_label=label,
                    method=method,
                    min_baseline_epe_px=min_epe,
                    max_baseline_epe_px=max_epe,
                    summary=summary,
                )
            )

    comparison_rows = _build_comparison_rows(rows)
    _write_tsv(output / "matrix_summary.tsv", rows)
    _write_comparison_tsv(output / "matrix_comparison.tsv", comparison_rows)
    summary = {
        "stage": "measurement_v1_rgb_patch_pose_proxy_matrix",
        "prediction_rows_csv": str(prediction_rows_csv),
        "model_dir": str(model_dir),
        "output_dir": str(output),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "baseline_bins_px": [float(value) for value in baseline_bins_px],
        "bin_tolerance_px": float(bin_tolerance_px),
        "include_all": bool(include_all),
        "row_count": int(len(rows)),
        "comparison_row_count": int(len(comparison_rows)),
        "rows": rows,
        "comparison_rows": comparison_rows,
        "outputs": {
            "matrix_summary_tsv": str(output / "matrix_summary.tsv"),
            "matrix_summary_json": str(output / "matrix_summary.json"),
            "matrix_comparison_tsv": str(output / "matrix_comparison.tsv"),
            "matrix_comparison_json": str(output / "matrix_comparison.json"),
        },
    }
    (output / "matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    comparison_summary = {
        "stage": "measurement_v1_rgb_patch_pose_proxy_matrix_comparison",
        "source_summary_json": str(output / "matrix_summary.json"),
        "row_count": int(len(comparison_rows)),
        "rows": comparison_rows,
    }
    (output / "matrix_comparison.json").write_text(
        json.dumps(comparison_summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_rows_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--baseline_bins_px", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--bin_tolerance_px", type=float, default=0.02)
    parser.add_argument("--no_include_all", action="store_true")
    parser.add_argument("--dustbin_threshold", type=float, default=0.5)
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--min_inliers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_rgb_patch_pose_proxy_matrix(
        prediction_rows_csv=Path(args.prediction_rows_csv),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        baseline_bins_px=[float(value) for value in args.baseline_bins_px],
        bin_tolerance_px=float(args.bin_tolerance_px),
        include_all=not bool(args.no_include_all),
        dustbin_threshold=float(args.dustbin_threshold),
        reprojection_error_px=float(args.reprojection_error_px),
        min_inliers=int(args.min_inliers),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
