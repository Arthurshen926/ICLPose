"""Summarize frozen fine-support ablations with block-bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.vfm.tokens import compute_file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _content_sha256(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bootstrap_interval(
    rows: list[dict[str, object]],
    metric: str,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in sorted(rows, key=lambda value: str(value["image_id"])):
        sequence = str(row["image_id"]).split("/", 1)[0]
        surface = row["surface_sets"]["child_returned_set"]
        grouped.setdefault(sequence, []).append(float(surface[metric]))
    blocks: dict[str, list[np.ndarray]] = {}
    for sequence, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        blocks[sequence] = [
            array[start : start + int(block_length)]
            for start in range(0, array.size, int(block_length))
        ]
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(repetitions),), dtype=np.float64)
    for repeat in range(int(repetitions)):
        selected: list[np.ndarray] = []
        for sequence in sorted(blocks):
            current = blocks[sequence]
            chosen = rng.integers(0, len(current), size=len(current))
            selected.extend(current[int(index)] for index in chosen)
        samples[repeat] = float(np.mean(np.concatenate(selected)))
    return {
        "mean": float(np.mean([
            float(row["surface_sets"]["child_returned_set"][metric])
            for row in rows
        ])),
        "block_bootstrap_95_low": float(np.quantile(samples, 0.025)),
        "block_bootstrap_95_high": float(np.quantile(samples, 0.975)),
    }


def _paired_bootstrap_delta(
    baseline_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    metric: str,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    baseline = {str(row["image_id"]): row for row in baseline_rows}
    variant = {str(row["image_id"]): row for row in variant_rows}
    if sorted(baseline) != sorted(variant):
        raise ValueError("paired ablations require identical query inventory")
    delta_rows: list[dict[str, object]] = []
    for image_id in sorted(baseline):
        left = float(
            baseline[image_id]["surface_sets"]["child_returned_set"][metric]
        )
        right = float(
            variant[image_id]["surface_sets"]["child_returned_set"][metric]
        )
        delta_rows.append(
            {
                "image_id": image_id,
                "surface_sets": {"child_returned_set": {metric: right - left}},
            }
        )
    return _bootstrap_interval(
        delta_rows,
        metric,
        block_length=block_length,
        repetitions=repetitions,
        seed=seed,
    )


def _mean_surface(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([
        float(row["surface_sets"]["child_returned_set"][key]) for row in rows
    ]))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite fine-support ablation summary")
    if int(args.block_length) <= 0 or int(args.bootstrap_repetitions) <= 0:
        raise ValueError("bootstrap controls must be positive")
    variants: dict[str, object] = {}
    rows_by_variant: dict[str, list[dict[str, object]]] = {}
    for value in args.evaluation:
        path = Path(value).resolve()
        report = _json_without_duplicates(path)
        if report.get("artifact_type") != "goal_maplet_pure_retrieval_surface_evaluation_v1":
            raise ValueError("not a pure retrieval surface evaluation")
        rows = list(report.get("rows", []))
        if len(rows) != 530:
            raise ValueError("fine-support ablation requires the frozen 530-query set")
        label = path.parent.name
        if label in variants:
            raise ValueError("ablation labels are duplicated")
        rows_by_variant[label] = rows
        variants[label] = {
            "evaluation": str(path),
            "evaluation_sha256": compute_file_sha256(path),
            "query_count": len(rows),
            "exact_visible_mass_recall": _bootstrap_interval(
                rows,
                "exact_visible_mass_recall",
                block_length=int(args.block_length),
                repetitions=int(args.bootstrap_repetitions),
                seed=int(args.seed),
            ),
            "tolerant_visible_mass_recall_0.5m": _bootstrap_interval(
                rows,
                "tolerant_visible_mass_recall_0.5m",
                block_length=int(args.block_length),
                repetitions=int(args.bootstrap_repetitions),
                seed=int(args.seed) + 1,
            ),
            **{
                key: _mean_surface(rows, key)
                for key in (
                    "tolerant_visible_mass_recall_0.1m",
                    "tolerant_visible_mass_recall_0.25m",
                    "tolerant_visible_mass_recall_1m",
                    "weighted_distance_median_m",
                    "weighted_distance_p90_m",
                    "weighted_distance_p95_m",
                    "visible_primitive_area_precision",
                    "predicted_surface_area_m2",
                    "predicted_primitive_count",
                    "selected_child_count",
                    "connected_component_count",
                )
            },
            "sequence": report.get("sequence", {}),
            "claim_scope": {
                "retrieval_only": True,
                "not_pose_recall": True,
                "development_530_not_untouched_test": True,
            },
        }
    labels = list(variants)
    paired_deltas: dict[str, object] = {}
    if labels:
        baseline_label = labels[0]
        for offset, label in enumerate(labels[1:], start=1):
            paired_deltas[label] = {
                "relative_to": baseline_label,
                "exact_visible_mass_recall": _paired_bootstrap_delta(
                    rows_by_variant[baseline_label], rows_by_variant[label],
                    "exact_visible_mass_recall",
                    block_length=int(args.block_length),
                    repetitions=int(args.bootstrap_repetitions),
                    seed=int(args.seed) + 100 + offset,
                ),
                "tolerant_visible_mass_recall_0.5m": _paired_bootstrap_delta(
                    rows_by_variant[baseline_label], rows_by_variant[label],
                    "tolerant_visible_mass_recall_0.5m",
                    block_length=int(args.block_length),
                    repetitions=int(args.bootstrap_repetitions),
                    seed=int(args.seed) + 200 + offset,
                ),
            }
    payload: dict[str, object] = {
        "artifact_type": "goal_maplet_fine_support_ablation_summary_v1",
        "block_bootstrap": {
            "unit": "contiguous_frames_within_sequence",
            "block_length": int(args.block_length),
            "repetitions": int(args.bootstrap_repetitions),
            "seed": int(args.seed),
        },
        "variants": variants,
        "paired_deltas": paired_deltas,
    }
    payload["content_sha256"] = _content_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
