"""Filter Stage F VFM-native landmark banks by reliability diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.map_lifting import load_selected_track_bank_npz, save_selected_track_bank_npz
from feature_extract.vfm.vfm_aware_landmarks import (
    aggregate_track_match_reliability,
    filter_selected_track_bank_by_ids,
)


def _load_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _track_geometry_summary(track_observation_rows: list[dict[str, object]]) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in track_observation_rows:
        if row.get("track_id") is None:
            continue
        grouped.setdefault(int(row["track_id"]), []).append(row)
    result: dict[int, dict[str, float | int]] = {}
    for track_id, rows in grouped.items():
        reproj = [float(row.get("reprojection_error", 0.0)) for row in rows if row.get("reprojection_error") is not None]
        angles = [
            float(row.get("triangulation_angle_deg", 0.0))
            for row in rows
            if row.get("triangulation_angle_deg") is not None
        ]
        similarities = [
            float(row.get("mean_pair_similarity", 0.0))
            for row in rows
            if row.get("mean_pair_similarity") is not None
        ]
        result[int(track_id)] = {
            "track_id": int(track_id),
            "track_length": int(len(rows)),
            "max_reprojection_error": 0.0 if not reproj else float(np.max(reproj)),
            "mean_reprojection_error": 0.0 if not reproj else float(np.mean(reproj)),
            "triangulation_angle_deg": 0.0 if not angles else float(np.mean(angles)),
            "mean_pair_similarity": 0.0 if not similarities else float(np.mean(similarities)),
        }
    return result


def _passes_geometry(row: dict[str, float | int], args: argparse.Namespace) -> bool:
    if int(row.get("track_length", 0)) < int(args.min_track_length):
        return False
    if float(row.get("max_reprojection_error", 0.0)) > float(args.max_track_reprojection_error):
        return False
    if float(row.get("triangulation_angle_deg", 0.0)) < float(args.min_triangulation_angle_deg):
        return False
    if float(row.get("mean_pair_similarity", 0.0)) < float(args.min_mean_pair_similarity):
        return False
    return True


def _passes_oracle(row: dict[str, float | int], args: argparse.Namespace) -> bool:
    if int(row.get("match_count", 0)) < int(args.min_match_count):
        return False
    if float(row.get("patch_precision", 0.0)) < float(args.min_patch_precision):
        return False
    if float(row.get("stride_precision", 0.0)) < float(args.min_stride_precision):
        return False
    if float(row.get("pnp_inlier_rate", 0.0)) < float(args.min_pnp_inlier_rate):
        return False
    if float(row.get("median_gt_reproj_error_px", 0.0)) > float(args.max_median_gt_reproj_error_px):
        return False
    return True


def _passes_proxy(row: dict[str, float | int], args: argparse.Namespace) -> bool:
    if int(row.get("match_count", 0)) < int(args.min_match_count):
        return False
    if float(row.get("mean_similarity", 0.0)) < float(args.min_mean_similarity):
        return False
    if float(row.get("mean_landmark_reprojection_error", 0.0)) > float(args.max_landmark_reprojection_error):
        return False
    if float(row.get("mean_landmark_ambiguity", 0.0)) > float(args.max_landmark_ambiguity):
        return False
    if float(row.get("mean_observation_count", 0.0)) < float(args.min_observation_count):
        return False
    return True


def _limit_keep_fraction(
    keep_ids: list[int],
    scores: dict[int, float],
    keep_fraction: float,
) -> list[int]:
    if keep_fraction <= 0.0 or keep_fraction >= 1.0:
        return keep_ids
    keep_count = max(1, int(np.ceil(float(keep_fraction) * len(keep_ids))))
    return sorted(keep_ids, key=lambda track_id: (-float(scores.get(track_id, 0.0)), int(track_id)))[:keep_count]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Filter Stage F VFM-native landmark bank")
    parser.add_argument("--input_bank", required=True)
    parser.add_argument("--input_track_observations", required=True)
    parser.add_argument("--match_table", default="")
    parser.add_argument("--mode", choices=("track_geometry", "oracle_match_reliability", "proxy_match_quality"), required=True)
    parser.add_argument("--min_track_length", type=int, default=3)
    parser.add_argument("--max_track_reprojection_error", type=float, default=6.0)
    parser.add_argument("--min_triangulation_angle_deg", type=float, default=0.25)
    parser.add_argument("--min_mean_pair_similarity", type=float, default=0.0)
    parser.add_argument("--min_match_count", type=int, default=1)
    parser.add_argument("--min_patch_precision", type=float, default=0.0)
    parser.add_argument("--min_stride_precision", type=float, default=0.0)
    parser.add_argument("--min_pnp_inlier_rate", type=float, default=0.0)
    parser.add_argument("--max_median_gt_reproj_error_px", type=float, default=1e9)
    parser.add_argument("--min_mean_similarity", type=float, default=0.0)
    parser.add_argument("--max_landmark_reprojection_error", type=float, default=1e9)
    parser.add_argument("--max_landmark_ambiguity", type=float, default=1e9)
    parser.add_argument("--min_observation_count", type=float, default=0.0)
    parser.add_argument("--keep_fraction", type=float, default=0.0)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--output_track_observations", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    bank = load_selected_track_bank_npz(Path(args.input_bank))
    track_rows = _load_jsonl(args.input_track_observations)
    geometry = _track_geometry_summary(track_rows)

    if args.mode == "track_geometry":
        keep_ids = [track_id for track_id, row in geometry.items() if _passes_geometry(row, args)]
        scores = {track_id: float(row.get("triangulation_angle_deg", 0.0)) for track_id, row in geometry.items()}
        reliability = {}
    else:
        if not args.match_table:
            raise ValueError("match_table is required for match-reliability filtering")
        reliability = aggregate_track_match_reliability(_load_jsonl(args.match_table))
        if args.mode == "oracle_match_reliability":
            keep_ids = [track_id for track_id, row in reliability.items() if _passes_oracle(row, args)]
            scores = {track_id: float(row.get("stride_precision", 0.0)) for track_id, row in reliability.items()}
        else:
            keep_ids = [track_id for track_id, row in reliability.items() if _passes_proxy(row, args)]
            scores = {track_id: float(row.get("mean_similarity", 0.0)) for track_id, row in reliability.items()}

    keep_ids = [track_id for track_id in keep_ids if int(track_id) in bank.tracks]
    keep_ids = _limit_keep_fraction(keep_ids, scores, float(args.keep_fraction))
    keep_set = {int(track_id) for track_id in keep_ids}
    filtered = filter_selected_track_bank_by_ids(bank, keep_set)

    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(filtered, output_bank)

    output_obs = Path(args.output_track_observations)
    output_obs.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    with output_obs.open("w") as handle:
        for row in track_rows:
            if int(row.get("track_id", -1)) in keep_set:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written_rows += 1

    selected_reliability = [reliability[track_id] for track_id in keep_set if track_id in reliability]
    selected_geometry = [geometry[track_id] for track_id in keep_set if track_id in geometry]
    summary = {
        "stage": "stage_f_source_aware_vfm_landmark_filter",
        "mode": args.mode,
        "inputs": {
            "input_bank": str(args.input_bank),
            "input_track_observations": str(args.input_track_observations),
            "match_table": str(args.match_table),
        },
        "outputs": {
            "bank": str(output_bank),
            "track_observations": str(output_obs),
        },
        "input_track_count": int(len(bank.tracks)),
        "selected_track_count": int(len(filtered.tracks)),
        "selected_track_observation_rows": int(written_rows),
        "config": vars(args),
        "selected_mean_patch_precision": None
        if not selected_reliability
        else float(np.mean([row["patch_precision"] for row in selected_reliability])),
        "selected_mean_stride_precision": None
        if not selected_reliability
        else float(np.mean([row["stride_precision"] for row in selected_reliability])),
        "selected_mean_pnp_inlier_rate": None
        if not selected_reliability
        else float(np.mean([row["pnp_inlier_rate"] for row in selected_reliability])),
        "selected_mean_max_reprojection_error": None
        if not selected_geometry
        else float(np.mean([row["max_reprojection_error"] for row in selected_geometry])),
        "selected_mean_triangulation_angle_deg": None
        if not selected_geometry
        else float(np.mean([row["triangulation_angle_deg"] for row in selected_geometry])),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
