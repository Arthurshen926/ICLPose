"""Merge independently executed candidate-maplet inference members.

The member jobs must describe the same inference-only query set and model
contract.  This tool only ensembles already exported scores; it does not read
pose labels, tune a strategy, or run localization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member_dirs",
        required=True,
        help="Comma-separated directories containing summary.json and inference_scores.npz",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _load_member(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary_path = directory / "summary.json"
    scores_path = directory / "inference_scores.npz"
    summary = json.loads(summary_path.read_text())
    if not bool(dict(summary.get("protocol") or {}).get("inference_only", False)):
        raise ValueError(f"member is not inference-only: {directory}")
    expected_hash = dict(summary.get("outputs") or {}).get("scores_sha256")
    actual_hash = file_sha256_short(scores_path)
    if str(expected_hash) != str(actual_hash):
        raise ValueError(f"member score hash differs from summary: {directory}")
    with np.load(scores_path, allow_pickle=False) as data:
        scores = {key: np.asarray(data[key]) for key in data.files}
    return summary, scores


def _finite_mean(arrays: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(
        [np.asarray(array, dtype=np.float32) for array in arrays], axis=0
    )
    finite = np.isfinite(stacked)
    count = np.sum(finite, axis=0)
    result = np.full(stacked.shape[1:], -np.inf, dtype=np.float32)
    accepted = count > 0
    result[accepted] = (
        np.sum(np.where(finite, stacked, 0.0), axis=0)[accepted]
        / count[accepted]
    ).astype(np.float32)
    return result


def _merge_members(
    members: Sequence[tuple[dict[str, object], dict[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if len(members) < 2:
        raise ValueError("at least two inference members are required")
    first_summary, first_scores = members[0]
    contract_keys = ("data_manifest", "query_set", "model_config", "protocol")
    for member_index, (summary, _scores) in enumerate(members[1:], start=1):
        mismatched = [
            key for key in contract_keys if summary.get(key) != first_summary.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"member {member_index} contract differs: {', '.join(mismatched)}"
            )

    ensemble_prefix = "ensemble__"
    strategy_names = {
        key[len(ensemble_prefix) :]
        for key in first_scores
        if key.startswith(ensemble_prefix)
    }
    if not strategy_names:
        raise ValueError("member has no ensemble score arrays")
    baseline = np.asarray(first_scores["baseline_scores"], dtype=np.float32)
    selected = np.asarray(first_scores["selected_columns"], dtype=np.int64)
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_hashes: set[str] = set()
    radio_lineages: list[dict[str, object]] = []
    for member_index, (summary, scores) in enumerate(members):
        member_strategies = {
            key[len(ensemble_prefix) :]
            for key in scores
            if key.startswith(ensemble_prefix)
        }
        if member_strategies != strategy_names:
            raise ValueError(f"member {member_index} score strategies differ")
        if not np.array_equal(
            baseline, np.asarray(scores["baseline_scores"], dtype=np.float32)
        ):
            raise ValueError(f"member {member_index} baseline scores differ")
        if not np.array_equal(
            selected, np.asarray(scores["selected_columns"], dtype=np.int64)
        ):
            raise ValueError(f"member {member_index} selected columns differ")
        rows = list(summary.get("checkpoints") or [])
        if len(rows) != 1:
            raise ValueError(
                "parallel member merge requires exactly one checkpoint per input"
            )
        checkpoint = dict(rows[0])
        checkpoint_hash = str(checkpoint.get("sha256", ""))
        if not checkpoint_hash or checkpoint_hash in checkpoint_hashes:
            raise ValueError("member checkpoints must have unique non-empty hashes")
        checkpoint_hashes.add(checkpoint_hash)
        checkpoint_rows.append(checkpoint)
        member_lineages = list(summary.get("radio_projection_lineages") or [])
        if len(member_lineages) != 1:
            raise ValueError(
                "parallel member merge requires one RADIO lineage per input"
            )
        radio_lineages.append(dict(member_lineages[0]))

    output_arrays: dict[str, np.ndarray] = {
        "baseline_scores": baseline,
        "selected_columns": selected,
    }
    for member_index, (_summary, scores) in enumerate(members):
        for strategy in sorted(strategy_names):
            values = np.asarray(scores[f"ensemble__{strategy}"], dtype=np.float32)
            single_member = scores.get(f"member_0__{strategy}")
            if single_member is not None and not np.array_equal(
                values, np.asarray(single_member, dtype=np.float32)
            ):
                raise ValueError(
                    f"member {member_index} single-member and ensemble scores differ"
                )
            output_arrays[f"member_{member_index}__{strategy}"] = values
    for strategy in sorted(strategy_names):
        output_arrays[f"ensemble__{strategy}"] = _finite_mean(
            [
                scores[f"ensemble__{strategy}"]
                for _summary, scores in members
            ]
        )

    summary = {
        "stage": "candidate_maplet_external_query_inference_merged_ensemble",
        "protocol": dict(first_summary["protocol"]),
        "data_manifest": dict(first_summary["data_manifest"]),
        "query_set": dict(first_summary["query_set"]),
        "checkpoints": checkpoint_rows,
        "member_score_prefixes": [
            f"member_{index}" for index in range(len(members))
        ],
        "model_config": dict(first_summary["model_config"]),
        "radio_projection_lineages": radio_lineages,
        "merge": {
            "member_count": len(members),
            "finite_value_mean": True,
            "strategy_names": sorted(strategy_names),
        },
    }
    return output_arrays, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    member_dirs = tuple(
        Path(value.strip())
        for value in str(args.member_dirs).split(",")
        if value.strip()
    )
    members = tuple(_load_member(directory) for directory in member_dirs)
    output_arrays, summary = _merge_members(members)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "inference_scores.npz"
    np.savez(scores_path, **output_arrays)
    summary["inputs"] = [
        {
            "directory": str(directory),
            "summary_sha256": file_sha256_short(directory / "summary.json"),
            "scores_sha256": file_sha256_short(directory / "inference_scores.npz"),
        }
        for directory in member_dirs
    ]
    summary["outputs"] = {
        "scores": str(scores_path),
        "scores_sha256": file_sha256_short(scores_path),
        "summary": str(output_dir / "summary.json"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
