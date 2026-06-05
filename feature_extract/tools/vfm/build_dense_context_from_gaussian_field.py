"""Build sparse-anchor dense context from a scene-level Gaussian VFM field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import _load_track_stats
from feature_extract.vfm.dense_gaussian_field_diagnostics import gaussian_field_to_semidense_anchor_map
from feature_extract.vfm.dense_patch_context import DensePatchContextConfig, build_dense_patch_context_bank
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _summary(bank, sparse_count: int, gaussian_count: int, args: argparse.Namespace) -> dict[str, object]:
    support = np.asarray(bank.support_counts, dtype=np.float32)
    reliability = np.asarray(bank.reliability_scores, dtype=np.float32)
    return {
        "stage": "dense_context_from_gaussian_vfm_field",
        "sparse_anchor_count": int(sparse_count),
        "gaussian_field_anchor_count": int(gaussian_count),
        "context_anchor_count": int(len(bank)),
        "feature_dim": int(bank.feature_dim),
        "prototype_count": int(bank.prototype_count),
        "mean_support_count": float(np.mean(support)) if support.size else 0.0,
        "median_support_count": float(np.median(support)) if support.size else 0.0,
        "zero_support_fraction": float(np.mean(support == 0.0)) if support.size else 1.0,
        "support_at_least_4_fraction": float(np.mean(support >= 4.0)) if support.size else 0.0,
        "support_at_least_8_fraction": float(np.mean(support >= 8.0)) if support.size else 0.0,
        "mean_support_radius_m": float(np.mean(bank.support_radius)) if len(bank) else 0.0,
        "mean_feature_variance": float(np.mean(bank.feature_variance)) if len(bank) else 0.0,
        "mean_reliability": float(np.mean(reliability)) if reliability.size else 0.0,
        "median_reliability": float(np.median(reliability)) if reliability.size else 0.0,
        "inputs": {
            "sparse_landmark_bank": args.sparse_landmark_bank,
            "gaussian_field": args.gaussian_field,
            "track_observations": args.track_observations,
        },
        "outputs": {"context_bank": args.output_npz},
        "config": {
            "max_radius_m": float(args.max_radius_m),
            "max_support": int(args.max_support),
            "min_support": int(args.min_support),
            "prototype_count": int(args.prototype_count),
            "include_radius_fallback": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build dense context bank from a full Gaussian VFM field")
    parser.add_argument("--sparse_landmark_bank", required=True)
    parser.add_argument("--gaussian_field", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--max_radius_m", type=float, default=0.10)
    parser.add_argument("--max_support", type=int, default=32)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--prototype_count", type=int, default=4)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    xyz_by_track, reprojection_by_track = _load_track_stats(Path(args.track_observations))
    sparse = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.sparse_landmark_bank)),
        xyz_by_track,
        reprojection_by_track,
    )
    field = GaussianVFMField.load_npz(Path(args.gaussian_field))
    dense_source = gaussian_field_to_semidense_anchor_map(field)
    bank = build_dense_patch_context_bank(
        sparse,
        dense_source,
        DensePatchContextConfig(
            max_radius_m=float(args.max_radius_m),
            max_support=int(args.max_support),
            min_support=int(args.min_support),
            prototype_count=int(args.prototype_count),
            include_radius_fallback=True,
        ),
    )
    bank.save_npz(Path(args.output_npz))
    summary = _summary(bank, len(sparse), len(dense_source), args)
    path = Path(args.summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
