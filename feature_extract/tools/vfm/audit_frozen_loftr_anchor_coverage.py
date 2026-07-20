"""Report target-free LoFTR anchor coverage for every frozen top-L rank band.

This answers a prerequisite for interpreting a top-5/top-10/top-20 comparison:
the added ranks must actually have pairwise anchor evidence and retain their
original posterior mass.  The audit contains no track identity label, pose,
or hypothesis score.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_loftr_anchor_manifest import (
    _load_query_artifact,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    LOFTR_ANCHOR_FEATURE_NAMES,
)


FROZEN_LOFTR_ANCHOR_COVERAGE_AUDIT_FORMAT = "frozen_loftr_anchor_rank_coverage_audit_v1"
RANK_BANDS: Mapping[str, tuple[int, int]] = {
    "rank_1_5": (0, 5),
    "rank_6_10": (5, 10),
    "rank_11_20": (10, 20),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-artifact-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _paths(pattern: str) -> tuple[Path, ...]:
    paths = tuple(Path(value) for value in sorted(glob.glob(str(pattern))))
    if not paths or len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("LoFTR anchor artifact glob must resolve unique existing files")
    return paths


def _quantiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) == 0 or np.any(~np.isfinite(data)):
        raise ValueError("coverage summary values are invalid")
    return {
        "min": float(np.min(data)),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(np.max(data)),
    }


def rank_band_row_statistics(
    *,
    candidate_probabilities: np.ndarray,
    candidate_view_weights: np.ndarray,
    candidate_view_usable: np.ndarray,
    candidate_view_pair_match_counts: np.ndarray,
    candidate_usable_view_weight_mass: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    """Return per-row fixed-posterior coverage for one positional rank interval."""

    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    weights = np.asarray(candidate_view_weights, dtype=np.float32)
    usable = np.asarray(candidate_view_usable, dtype=bool)
    counts = np.asarray(candidate_view_pair_match_counts, dtype=np.int32)
    usable_mass = np.asarray(candidate_usable_view_weight_mass, dtype=np.float32)
    if (
        probabilities.ndim != 2
        or weights.shape[:2] != probabilities.shape
        or usable.shape != weights.shape
        or counts.shape != weights.shape
        or usable_mass.shape != probabilities.shape
        or not 0 <= int(start) < int(stop) <= probabilities.shape[1]
        or np.any(probabilities < 0.0)
        or np.any(weights < 0.0)
        or np.any(counts[usable] <= 0)
        or not np.allclose(
            usable_mass, (weights * usable.astype(np.float32)).sum(axis=2), atol=2e-5
        )
    ):
        raise ValueError("LoFTR rank-band coverage arrays are invalid")
    candidate = probabilities[:, start:stop]
    view_weights = weights[:, start:stop]
    view_usable = usable[:, start:stop]
    view_counts = counts[:, start:stop]
    mass = usable_mass[:, start:stop]
    prior_mass = candidate.sum(axis=1, dtype=np.float64)
    observed_mass = (candidate * mass).sum(axis=1, dtype=np.float64)
    nonzero = prior_mass > 0.0
    coverage = np.divide(
        observed_mass,
        prior_mass,
        out=np.zeros_like(observed_mass),
        where=nonzero,
    )
    fixed_views = view_weights > 0.0
    usable_views = fixed_views & view_usable
    candidate_any = np.any((mass > 0.0) & (candidate > 0.0), axis=1)
    per_row_pair_matches = np.where(usable_views, view_counts, np.nan)
    pair_median = np.nanmedian(per_row_pair_matches.reshape(len(candidate), -1), axis=1)
    pair_median = np.where(np.isfinite(pair_median), pair_median, 0.0)
    return {
        "prior_mass": prior_mass.astype(np.float32),
        "observed_mass": observed_mass.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "candidate_any_usable": candidate_any.astype(bool),
        "fixed_view_count": fixed_views.sum(axis=(1, 2), dtype=np.int32),
        "usable_view_count": usable_views.sum(axis=(1, 2), dtype=np.int32),
        "usable_pair_match_median": pair_median.astype(np.float32),
    }


def _summarize_band(stats: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "prior_mass": _quantiles(stats["prior_mass"]),
        "observed_prior_mass": _quantiles(stats["observed_mass"]),
        "fixed_view_mass_coverage": _quantiles(stats["coverage"]),
        "rows_with_any_usable_candidate_rate": float(
            np.mean(np.asarray(stats["candidate_any_usable"], dtype=bool))
        ),
        "fixed_view_count": _quantiles(stats["fixed_view_count"]),
        "usable_view_count": _quantiles(stats["usable_view_count"]),
        "usable_pair_match_median": _quantiles(stats["usable_pair_match_median"]),
    }


def audit_frozen_loftr_anchor_coverage(
    *, anchor_artifacts: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    """Audit all rank bands without altering candidates or posterior mass."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite LoFTR coverage audit: {output}")
    records = tuple(_load_query_artifact(Path(path), anchor=True) for path in anchor_artifacts)
    if len({record.query_id for record in records}) != len(records):
        raise ValueError("LoFTR coverage audit artifacts repeat a query id")
    if set(record.split_name for record in records) - {"train", "validation"}:
        raise ValueError("LoFTR coverage audit accepts only train/validation artifacts")
    query_ids = []
    split_names = []
    per_band = {name: [] for name in RANK_BANDS}
    null_mass = []
    for record in records:
        arrays = record.arrays
        query_ids.append(record.query_id)
        split_names.append(record.split_name)
        null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
        null_mass.append(float(np.mean(null)))
        for name, (start, stop) in RANK_BANDS.items():
            per_band[name].append(
                rank_band_row_statistics(
                    candidate_probabilities=arrays["candidate_probabilities"],
                    candidate_view_weights=arrays["candidate_view_weights"],
                    candidate_view_usable=arrays["candidate_view_usable"],
                    candidate_view_pair_match_counts=arrays[
                        "candidate_view_pair_match_counts"
                    ],
                    candidate_usable_view_weight_mass=arrays[
                        "candidate_usable_view_weight_mass"
                    ],
                    start=start,
                    stop=stop,
                )
            )
    output.mkdir(parents=True, exist_ok=False)
    query_ids_array = np.asarray(query_ids, dtype=np.str_)
    split_names_array = np.asarray(split_names, dtype=np.str_)
    band_payload: dict[str, np.ndarray] = {}
    split_summary: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        selected_queries = split_names_array == split_name
        if not np.any(selected_queries):
            continue
        split_summary[split_name] = {
            "query_count": int(np.sum(selected_queries)),
            "mean_null_probability": float(
                np.mean(np.asarray(null_mass, dtype=np.float32)[selected_queries])
            ),
            "rank_bands": {},
        }
        for name in RANK_BANDS:
            merged = {
                key: np.concatenate([item[key] for item in per_band[name]], axis=0)
                for key in per_band[name][0]
            }
            rows_per_query = len(per_band[name][0]["prior_mass"])
            row_mask = np.repeat(selected_queries, rows_per_query)
            split_summary[split_name]["rank_bands"][name] = _summarize_band(
                {key: value[row_mask] for key, value in merged.items()}
            )
    for name in RANK_BANDS:
        for key in per_band[name][0]:
            band_payload[f"{name}_{key}"] = np.stack(
                [item[key] for item in per_band[name]], axis=0
            )
    np.savez_compressed(
        output / "per_query_rank_coverage.npz",
        query_ids=query_ids_array,
        split_names=split_names_array,
        rank_band_names=np.asarray(tuple(RANK_BANDS), dtype=np.str_),
        **band_payload,
    )
    summary = {
        "format": FROZEN_LOFTR_ANCHOR_COVERAGE_AUDIT_FORMAT,
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": 20,
            "rank_bands": {name: [start + 1, stop] for name, (start, stop) in RANK_BANDS.items()},
            "candidate_posterior_re_normalized": False,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "identity_or_pose_labels_loaded": False,
        },
        "anchor_feature_names": list(LOFTR_ANCHOR_FEATURE_NAMES),
        "anchor_artifacts": [
            {"path": str(record.path), "sha256": record.sha256} for record in records
        ],
        "split_summary": split_summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_frozen_loftr_anchor_coverage(
        anchor_artifacts=_paths(args.anchor_artifact_glob), output_dir=Path(args.output_dir)
    )
    print(json.dumps(summary["split_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
