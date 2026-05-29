"""Export descriptors from a trained Stage C2 safe selector checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.tools.vfm.train_stage_c2_safe_selector import (
    _transform_track_bank_with_safe_selector,
    _write_safe_query_manifest,
)
from feature_extract.vfm.patch_selector_training import SafePatchSelectorTrainingRun, load_safe_patch_selector_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest


def _group_slices(input_dim: int, group_size: int) -> list[slice]:
    if int(group_size) <= 0:
        return [slice(0, int(input_dim))]
    return [slice(start, min(start + int(group_size), int(input_dim))) for start in range(0, int(input_dim), int(group_size))]


def _projection_group_energy(matrix: np.ndarray, group_size: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return np.asarray(
        [float(np.sum(np.square(values[group, :]))) for group in _group_slices(values.shape[0], int(group_size))],
        dtype=np.float32,
    )


def active_group_mask_for_checkpoint(run: SafePatchSelectorTrainingRun, keep_fraction: float, min_groups: int = 1) -> np.ndarray:
    if not 0.0 < float(keep_fraction) <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if int(min_groups) <= 0:
        raise ValueError("min_groups must be positive")
    model = run.model
    gates = model.group_gates().detach().cpu().numpy().astype(np.float32)
    weights = model.projection.weight.detach().cpu().numpy().astype(np.float32).T
    energy = _projection_group_energy(weights, int(model.group_size))
    scores = gates * energy
    group_count = int(scores.shape[0])
    keep_count = min(group_count, max(int(min_groups), int(np.ceil(group_count * float(keep_fraction)))))
    order = np.lexsort((np.arange(group_count), -scores))
    mask = np.zeros((group_count,), dtype=np.float32)
    mask[order[:keep_count]] = 1.0
    return mask


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export Stage C2 safe selector descriptors with an optional hard group gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--hard_gate_keep_fraction", type=float, default=1.0)
    parser.add_argument("--hard_gate_min_groups", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_rows", type=int, default=65536)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    parser.add_argument("--output_query_dir", required=True)
    parser.add_argument("--output_query_manifest", required=True)
    parser.add_argument("--output_landmark_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    loaded = load_safe_patch_selector_checkpoint(Path(args.checkpoint), device=args.device)
    active_mask = active_group_mask_for_checkpoint(
        loaded,
        keep_fraction=float(args.hard_gate_keep_fraction),
        min_groups=int(args.hard_gate_min_groups),
    )
    run = SafePatchSelectorTrainingRun(
        model=loaded.model,
        summary=loaded.summary,
        active_group_mask=active_mask,
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    output_layer_name = args.output_layer_name or args.layer_name
    mapability = _transform_track_bank_with_safe_selector(
        Path(args.landmark_bank),
        Path(args.output_landmark_bank),
        run,
        device=args.device,
        batch_rows=int(args.batch_rows),
    )
    compressed_manifest, query_count = _write_safe_query_manifest(
        manifest,
        run,
        layer_name=args.layer_name,
        output_query_dir=Path(args.output_query_dir),
        output_layer_name=output_layer_name,
        device=args.device,
        batch_tokens=int(args.batch_tokens),
    )
    compressed_manifest.to_json(Path(args.output_query_manifest))
    summary = {
        "stage": "stage_c2_safe_selector_export",
        "checkpoint": str(args.checkpoint),
        "query_manifest": str(args.query_manifest),
        "landmark_bank": str(args.landmark_bank),
        "output_query_manifest": str(args.output_query_manifest),
        "output_landmark_bank": str(args.output_landmark_bank),
        "query_record_count": int(query_count),
        "hard_gate_keep_fraction": float(args.hard_gate_keep_fraction),
        "group_count": int(active_mask.shape[0]),
        "active_group_count": int(np.sum(active_mask > 0.0)),
        "active_group_fraction": float(np.mean(active_mask > 0.0)) if active_mask.size else 0.0,
        "active_group_mask": [float(value) for value in active_mask.tolist()],
        "mapability": mapability,
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
