"""Merge landmark-associated and ray-contributed Gaussian VFM fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField, merge_gaussian_vfm_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a primary-preferred hybrid Gaussian VFM field")
    parser.add_argument("--primary_field", required=True)
    parser.add_argument("--fallback_field", required=True)
    parser.add_argument("--l2_normalize_features", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    primary = GaussianVFMField.load_npz(Path(args.primary_field))
    fallback = GaussianVFMField.load_npz(Path(args.fallback_field))
    hybrid = merge_gaussian_vfm_fields(
        primary,
        fallback,
        l2_normalize_features=bool(args.l2_normalize_features),
    )
    hybrid.save_npz(Path(args.output))
    primary_set = set(int(idx) for idx in primary.gaussian_indices.tolist())
    fallback_set = set(int(idx) for idx in fallback.gaussian_indices.tolist())
    summary = {
        "stage": "hybrid_gaussian_vfm_field",
        "primary_count": int(len(primary)),
        "fallback_count": int(len(fallback)),
        "overlap_count": int(len(primary_set.intersection(fallback_set))),
        "fallback_added_count": int(dict(hybrid.metadata).get("fallback_added_count", 0)),
        "hybrid_count": int(len(hybrid)),
        "feature_dim": int(hybrid.feature_dim),
        "mean_support_count": 0.0 if len(hybrid) == 0 else float(np.mean(hybrid.support_counts)),
        "inputs": {
            "primary_field": args.primary_field,
            "fallback_field": args.fallback_field,
        },
        "outputs": {"field": str(args.output)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
