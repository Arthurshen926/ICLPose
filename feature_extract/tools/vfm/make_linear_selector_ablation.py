"""Create channel-group ablations for a learned linear VFM selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.feature_compression import FeatureCompressionTransform


def _group_slices(input_dim: int, group_size: int) -> list[slice]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [slice(start, min(start + group_size, input_dim)) for start in range(0, input_dim, group_size)]


def _group_energy(matrix: np.ndarray, group_size: int) -> np.ndarray:
    groups = _group_slices(int(matrix.shape[0]), int(group_size))
    return np.asarray([float(np.sum(np.square(matrix[group, :]))) for group in groups], dtype=np.float32)


def _select_groups(energy: np.ndarray, policy: str, remove_count: int, seed: int) -> np.ndarray:
    if not 0 < int(remove_count) <= int(energy.shape[0]):
        raise ValueError("remove_count must be in (0, group_count]")
    if policy == "top":
        order = np.lexsort((np.arange(energy.size), -energy))
        return order[:remove_count].astype(np.int64)
    if policy == "bottom":
        order = np.lexsort((np.arange(energy.size), energy))
        return order[:remove_count].astype(np.int64)
    if policy == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(energy.size, size=int(remove_count), replace=False)).astype(np.int64)
    raise ValueError("policy must be one of: top, bottom, random")


def ablate_transform(
    transform: FeatureCompressionTransform,
    *,
    policy: str,
    group_size: int,
    remove_fraction: float,
    seed: int,
) -> tuple[FeatureCompressionTransform, dict[str, object]]:
    if transform.matrix is None:
        raise ValueError("linear selector ablation requires a matrix transform")
    matrix = np.asarray(transform.matrix, dtype=np.float32).copy()
    energy = _group_energy(matrix, int(group_size))
    remove_count = max(1, int(round(float(remove_fraction) * energy.shape[0])))
    remove_count = min(remove_count, int(energy.shape[0]))
    selected_groups = _select_groups(energy, policy, remove_count, seed)
    groups = _group_slices(int(transform.input_dim), int(group_size))
    removed_channels: list[int] = []
    for group_idx in selected_groups.tolist():
        group = groups[int(group_idx)]
        matrix[group, :] = 0.0
        removed_channels.extend(range(group.start, group.stop))
    channel_scores = np.asarray(transform.channel_scores, dtype=np.float32) if transform.channel_scores is not None else None
    ablated = FeatureCompressionTransform(
        method=f"{transform.method}_ablate_{policy}",
        input_dim=int(transform.input_dim),
        output_dim=int(transform.output_dim),
        mean=transform.mean,
        matrix=matrix,
        selected_channels=None,
        channel_scores=channel_scores,
        l2_normalize=bool(transform.l2_normalize),
    )
    summary = {
        "stage": "linear_selector_channel_group_ablation",
        "source_method": transform.method,
        "policy": policy,
        "group_size": int(group_size),
        "group_count": int(energy.shape[0]),
        "remove_fraction": float(remove_fraction),
        "remove_count": int(remove_count),
        "removed_groups": [int(value) for value in selected_groups.tolist()],
        "removed_channel_count": int(len(removed_channels)),
        "removed_channel_head": [int(value) for value in removed_channels[:64]],
        "energy_mean": float(np.mean(energy)),
        "energy_max": float(np.max(energy)),
        "removed_energy_sum": float(np.sum(energy[selected_groups])),
        "removed_energy_fraction": float(np.sum(energy[selected_groups]) / max(float(np.sum(energy)), 1e-12)),
    }
    return ablated, summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create top/bottom/random channel-group ablations for a linear selector")
    parser.add_argument("--input_transform", required=True)
    parser.add_argument("--policy", required=True, choices=("top", "bottom", "random"))
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--remove_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_transform", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    transform = FeatureCompressionTransform.from_npz(Path(args.input_transform))
    ablated, summary = ablate_transform(
        transform,
        policy=args.policy,
        group_size=int(args.group_size),
        remove_fraction=float(args.remove_fraction),
        seed=int(args.seed),
    )
    summary["input_transform"] = str(args.input_transform)
    summary["output_transform"] = str(args.output_transform)
    ablated.to_npz(Path(args.output_transform), metadata=summary)
    output_summary = Path(args.summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
