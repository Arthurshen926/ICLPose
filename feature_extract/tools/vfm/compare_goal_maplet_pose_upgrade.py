"""Paired post-label audit of two already-frozen pose inventories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.statistics import mcnemar_exact_pvalue, paired_bootstrap_delta_ci


THRESHOLDS = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 10.0), (2.0, 45.0))


def _threshold_hits(translation, rotation, threshold_t, threshold_r):
    # Each method's failure belongs to that method, not to the common subset.
    return (np.isfinite(translation) & np.isfinite(rotation)
            & (translation <= threshold_t) & (rotation <= threshold_r))


def _temporal_block_ci(names, delta, block_length, resamples, seed):
    """Route-stratified circular moving blocks, conditional on the observed routes."""
    names = np.asarray(names).astype(str)
    delta = np.asarray(delta, np.float64)
    if delta.shape != names.shape or block_length < 1 or resamples < 1:
        raise ValueError("invalid temporal bootstrap input")
    routes = {}
    for row, name in enumerate(names):
        match = re.fullmatch(r"(.+)__frame(\d+)\.png\.npz", name)
        if match is None:
            raise ValueError("temporal bootstrap requires explicit route/frame names")
        routes.setdefault(match[1], []).append((int(match[2]), row))
    groups = []
    for entries in routes.values():
        entries = sorted(entries)
        if len({frame for frame, _ in entries}) != len(entries):
            raise ValueError("duplicate route/frame identity")
        groups.append(np.asarray([row for _, row in entries], np.int64))
    if not np.isfinite(delta).any():
        raise ValueError("no finite temporal deltas")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        selected = []
        for rows in groups:
            size = min(block_length, len(rows))
            starts = rng.integers(0, len(rows), size=(len(rows) + size - 1) // size)
            indices = ((starts[:, None] + np.arange(size)) % len(rows)).ravel()[:len(rows)]
            selected.append(rows[indices])
        value = delta[np.concatenate(selected)]
        if np.isfinite(value).any():
            samples.append(float(np.mean(value[np.isfinite(value)])))
    if not samples:
        raise ValueError("no usable temporal bootstrap samples")
    lower, upper = np.quantile(samples, [.025, .975])
    return {"estimate": float(np.mean(delta[np.isfinite(delta)])),
            "lower": float(lower), "upper": float(upper),
            "valid_resamples": len(samples), "route_count": len(groups),
            "block_length_observed_frames": block_length,
            "semantics": "route_stratified_circular_moving_blocks_sorted_numeric_frame;not_unseen_route_inference"}


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    if (
        not {"names", "pose_w2c", "usable"}.issubset(arrays)
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or arrays["pose_w2c"].shape != (len(arrays["names"]), 4, 4)
    ):
        raise ValueError("frozen pose inventory differs")
    return arrays, metadata


def _errors(arrays: dict[str, np.ndarray], contributors: Path) -> tuple[np.ndarray, np.ndarray]:
    translation = np.full(len(arrays["names"]), np.inf, np.float64)
    rotation = np.full(len(arrays["names"]), np.inf, np.float64)
    for row, name in enumerate(arrays["names"].astype(str).tolist()):
        if not bool(arrays["usable"][row]):
            continue
        with np.load(contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        pose = np.asarray(arrays["pose_w2c"][row], np.float64)
        center = -pose[:3, :3].T @ pose[:3, 3]
        gt_center = -gt[:3, :3].T @ gt[:3, 3]
        translation[row] = np.linalg.norm(center - gt_center)
        rotation[row] = Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude() * 180.0 / np.pi
    return translation, rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=260907)
    parser.add_argument("--temporal_block_lengths", type=int, nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite paired pose audit")
    baseline, baseline_meta = _load(args.baseline)
    method, method_meta = _load(args.method)
    if not np.array_equal(baseline["names"].astype(str), method["names"].astype(str)):
        raise ValueError("paired pose names differ")
    baseline_t, baseline_r = _errors(baseline, args.query_contributors)
    method_t, method_r = _errors(method, args.query_contributors)
    finite = np.isfinite(baseline_t) & np.isfinite(baseline_r) & np.isfinite(method_t) & np.isfinite(method_r)
    if not np.any(finite):
        raise ValueError("paired pose audit has no jointly usable rows")
    threshold_rows: dict[str, object] = {}
    for translation, rotation in THRESHOLDS:
        base = _threshold_hits(baseline_t, baseline_r, translation, rotation)
        new = _threshold_hits(method_t, method_r, translation, rotation)
        key = f"{translation:g}m_{rotation:g}deg"
        threshold_rows[key] = {
            "baseline_hits": int(np.sum(base)),
            "method_hits": int(np.sum(new)),
            "gains": int(np.sum(~base & new)),
            "losses": int(np.sum(base & ~new)),
            "net_gain": int(np.sum(new) - np.sum(base)),
            "mcnemar_exact_two_sided_p": mcnemar_exact_pvalue(base, new),
        }
    t_delta = paired_bootstrap_delta_ci(
        method_t[finite], baseline_t[finite], int(args.bootstrap_resamples), int(args.bootstrap_seed),
    )
    r_delta = paired_bootstrap_delta_ci(
        method_r[finite], baseline_r[finite], int(args.bootstrap_resamples), int(args.bootstrap_seed) + 1,
    )
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_frozen_pose_upgrade_paired_postlabel_audit_v1",
        "pose_inventories_frozen_before_labels_opened": True,
        "query_count": int(len(baseline_t)),
        "jointly_usable_count": int(np.sum(finite)),
        "thresholds": threshold_rows,
        "baseline_median_translation_m": float(np.median(baseline_t[finite])),
        "method_median_translation_m": float(np.median(method_t[finite])),
        "baseline_median_rotation_deg": float(np.median(baseline_r[finite])),
        "method_median_rotation_deg": float(np.median(method_r[finite])),
        "mean_translation_delta_m_bootstrap_95ci": {
            "estimate": t_delta[0], "lower": t_delta[1], "upper": t_delta[2],
        },
        "mean_rotation_delta_deg_bootstrap_95ci": {
            "estimate": r_delta[0], "lower": r_delta[1], "upper": r_delta[2],
        },
        "translation_improved_count": int(np.sum(method_t[finite] < baseline_t[finite])),
        "rotation_improved_count": int(np.sum(method_r[finite] < baseline_r[finite])),
        "baseline_file_sha256": file_sha256(args.baseline),
        "baseline_content_sha256": baseline_meta.get("content_sha256"),
        "method_file_sha256": file_sha256(args.method),
        "method_content_sha256": method_meta.get("content_sha256"),
        "query_pose_or_ground_truth_opened": True,
        "production_eligible": False,
    }
    if args.temporal_block_lengths:
        td = np.full(len(finite), np.nan); rd = td.copy()
        td[finite] = method_t[finite] - baseline_t[finite]
        rd[finite] = method_r[finite] - baseline_r[finite]
        report["temporal_block_bootstrap"] = {
            str(length): {
                "translation_m": _temporal_block_ci(baseline["names"], td, length,
                                                     args.bootstrap_resamples, args.bootstrap_seed),
                "rotation_deg": _temporal_block_ci(baseline["names"], rd, length,
                                                   args.bootstrap_resamples, args.bootstrap_seed + 1),
            } for length in args.temporal_block_lengths
        }
    report["threshold_population"] = "all_queries_independent_method_usability"
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
