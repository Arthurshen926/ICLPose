"""Audit candidate and stored-hypothesis pose ceilings after localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm import parse_cambridge_pose_file
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground_truth_pose_file", required=True)
    parser.add_argument(
        "--candidate_results",
        nargs="+",
        required=True,
        help="Named JSONL inputs in NAME=PATH form.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args(argv)


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = str(value).partition("=")
        if not separator or not name or not path:
            raise ValueError("candidate inputs must use NAME=PATH")
        if name in output:
            raise ValueError(f"duplicate candidate name: {name}")
        output[name] = Path(path)
    return output


def _load_rows(path: Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = dict(json.loads(line))
        image_id = str(row["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate query in {path}: {image_id}")
        output[image_id] = row
    return output


def _error(
    pose_w2c: object,
    ground_truth_w2c: np.ndarray,
) -> tuple[float, float]:
    error = pnp_pose_error(
        np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
        ground_truth_w2c,
    )
    return float(error.translation_m), float(error.rotation_deg)


def _best(errors: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not errors:
        return None
    # A fixed metric is needed only to choose one member of an oracle set.
    # Translation remains primary; 1 degree is equated with 2 cm.
    return min(errors, key=lambda value: value[0] + 0.02 * value[1])


def _metrics(
    errors: Sequence[tuple[float, float] | None],
) -> dict[str, object]:
    valid = np.asarray(
        [value for value in errors if value is not None],
        dtype=np.float64,
    )
    query_count = len(errors)
    if len(valid) == 0:
        return {
            "query_count": query_count,
            "available_count": 0,
            "coverage": 0.0,
        }
    return {
        "query_count": query_count,
        "available_count": int(len(valid)),
        "coverage": float(len(valid) / max(query_count, 1)),
        "median_translation_m": float(np.median(valid[:, 0])),
        "p90_translation_m": float(np.percentile(valid[:, 0], 90.0)),
        "median_rotation_deg": float(np.median(valid[:, 1])),
        "p90_rotation_deg": float(np.percentile(valid[:, 1], 90.0)),
        "recall_4cm_1deg_all_queries": float(
            np.sum((valid[:, 0] <= 0.04) & (valid[:, 1] <= 1.0))
            / max(query_count, 1)
        ),
        "recall_5cm_5deg_all_queries": float(
            np.sum((valid[:, 0] <= 0.05) & (valid[:, 1] <= 5.0))
            / max(query_count, 1)
        ),
        "recall_10cm_5deg_all_queries": float(
            np.sum((valid[:, 0] <= 0.10) & (valid[:, 1] <= 5.0))
            / max(query_count, 1)
        ),
        "recall_25cm_10deg_all_queries": float(
            np.sum((valid[:, 0] <= 0.25) & (valid[:, 1] <= 10.0))
            / max(query_count, 1)
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = _named_paths(args.candidate_results)
    candidates = {name: _load_rows(path) for name, path in paths.items()}
    ground_truth = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(
            Path(args.ground_truth_pose_file)
        )
    }
    image_ids = list(ground_truth)
    for name, rows in candidates.items():
        missing = set(image_ids) - set(rows)
        if missing:
            raise ValueError(
                f"candidate {name} misses queries: {sorted(missing)[:3]}"
            )

    selected_by_name: dict[str, list[tuple[float, float] | None]] = {
        name: [] for name in candidates
    }
    candidate_oracle: list[tuple[float, float] | None] = []
    hypothesis_oracle: list[tuple[float, float] | None] = []
    for image_id in image_ids:
        selected_errors: list[tuple[float, float]] = []
        hypothesis_errors: list[tuple[float, float]] = []
        for name, rows in candidates.items():
            row = rows[image_id]
            selected_error = None
            if bool(row.get("success")) and row.get("pose_w2c") is not None:
                selected_error = _error(
                    row["pose_w2c"], ground_truth[image_id]
                )
                selected_errors.append(selected_error)
            selected_by_name[name].append(selected_error)
            for hypothesis in row.get("hypotheses") or ():
                pose = dict(hypothesis).get("pose_w2c")
                if pose is not None:
                    hypothesis_errors.append(
                        _error(pose, ground_truth[image_id])
                    )
        candidate_oracle.append(_best(selected_errors))
        hypothesis_oracle.append(
            _best(hypothesis_errors + selected_errors)
        )

    summary = {
        "stage": "audit_surface_pose_oracles",
        "oracle_selection_cost": "translation_m + 0.02 * rotation_deg",
        "selected_results": {
            name: _metrics(errors)
            for name, errors in selected_by_name.items()
        },
        "candidate_pose_oracle": _metrics(candidate_oracle),
        "stored_hypothesis_oracle": _metrics(hypothesis_oracle),
        "frozen_inputs": {
            "ground_truth_pose_file": str(args.ground_truth_pose_file),
            "ground_truth_pose_file_sha256": file_sha256_short(
                Path(args.ground_truth_pose_file)
            ),
            "candidate_results": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256_short(path),
                }
                for name, path in paths.items()
            },
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
