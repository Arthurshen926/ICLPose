"""Extract train-only registered track targets for V5's identity side head.

The main geometric target is deliberately set-valued at a tight reprojection
threshold.  Registered SfM point identity has a different role and is stored
separately, so fitting never has to open validation/test annotations or force
the two meanings into one posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
)
from feature_extract.tools.vfm.fit_multiscale_context_attention_probe import (
    EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT,
    _load_contract,
    _train_identity_targets_with_explicit_null,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_frozen_layout,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def build_context_attention_exact_identity_train_targets(
    *,
    contract_path: Path,
    colmap_model_dir: Path,
    registered_identity_radius_px: float,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Write a lineage-bound cache containing supervised train rows only."""

    target_path = Path(output)
    summary_path = Path(summary_json)
    if (target_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite exact identity train target cache")
    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    contract = _load_contract(Path(contract_path))
    layout_path = Path(str(contract["frozen_layout_features"]))
    layout, _layout_metadata = load_context_attention_frozen_layout(layout_path)
    layout_rows, target_classes, _candidate_membership, audit = (
        _train_identity_targets_with_explicit_null(
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            query_xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_tracks=np.asarray(layout["candidate_track_ids"], dtype=np.int64),
            split_names=np.asarray(layout["split_names"]).astype(str),
            colmap_model_dir=Path(colmap_model_dir),
            radius_px=float(registered_identity_radius_px),
        )
    )
    source_rows = np.asarray(layout["source_row_indices"], dtype=np.int64)[layout_rows]
    candidate_tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)[layout_rows]
    full_splits = np.asarray(layout["split_names"]).astype(str)
    candidate_count = int(candidate_tracks.shape[1])
    if (
        layout_rows.ndim != 1
        or layout_rows.size == 0
        or np.unique(layout_rows).size != len(layout_rows)
        or np.any(full_splits[layout_rows] != "train")
        or source_rows.shape != layout_rows.shape
        or candidate_tracks.shape != (len(layout_rows), candidate_count)
        or target_classes.shape != layout_rows.shape
        or np.any(target_classes < 0)
        or np.any(target_classes > candidate_count)
    ):
        raise RuntimeError("exact identity train target arrays are invalid")
    metadata = {
        "format": EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "training_split": "train",
        "contract_sha256": file_sha256_short(Path(contract_path)),
        "frozen_layout_features_sha256": contract["frozen_layout_features_sha256"],
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "layout_row_indices_sha256": _array_sha256_short(layout_rows),
        "source_row_indices_sha256": _array_sha256_short(source_rows),
        "candidate_track_ids_sha256": _array_sha256_short(candidate_tracks),
        "target_class_semantics": "registered_exact_track_if_in_fixed_topl_else_explicit_null",
        "fit_must_not_open_colmap_identity_model": True,
        "audit": audit,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        layout_row_indices=layout_rows,
        source_row_indices=source_rows,
        candidate_track_ids=candidate_tracks,
        target_classes=np.asarray(target_classes, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_context_attention_exact_identity_train_targets",
        "output": str(target_path),
        "output_sha256": file_sha256_short(target_path),
        "row_count": int(len(layout_rows)),
        "target_audit": audit,
        "protocol": {
            "training_split": "train",
            "contains_validation_or_test_targets": False,
            "fit_must_not_open_colmap_identity_model": True,
            "registered_identity_radius_px": float(registered_identity_radius_px),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_context_attention_exact_identity_train_targets(
        contract_path=Path(args.contract),
        colmap_model_dir=Path(args.colmap_model_dir),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
