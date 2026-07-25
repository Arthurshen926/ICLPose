"""Extract train-only geometric-set targets for the context-attention probe.

The immutable proposal archive contains candidate residuals for every split,
which is useful for post-hoc evaluation but must not be opened by the fitting
job.  This command reads only memberships at frozen train source rows and
writes a compact cache with no validation/test target entries.  The fitter
validates its full contract/proposal/layout lineage before using it.
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
from feature_extract.tools.vfm.fit_multiscale_candidate_probe import _load_proposal_tracks
from feature_extract.tools.vfm.fit_multiscale_context_attention_probe import (
    GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT,
    _PROPOSAL_OVERLAY_CANDIDATE_INPUT,
    _load_contract,
    _train_geometric_targets_with_explicit_null,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_frozen_layout,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--geometric_positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def build_context_attention_geometric_train_targets(
    *,
    contract_path: Path,
    proposals_path: Path,
    geometric_positive_threshold_px: float,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Write a lineage-bound cache of train-only candidate memberships."""

    target_path = Path(output)
    summary_path = Path(summary_json)
    if (target_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite geometric train target cache")
    contract = _load_contract(Path(contract_path))
    if str(contract.get("candidate_input_kind", _PROPOSAL_OVERLAY_CANDIDATE_INPUT)) != (
        _PROPOSAL_OVERLAY_CANDIDATE_INPUT
    ):
        raise ValueError("geometric train targets require a proposal-backed frozen contract")
    proposal_file = Path(proposals_path)
    if not proposal_file.is_file() or file_sha256_short(proposal_file) != str(
        contract.get("proposals_sha256", "")
    ):
        raise ValueError("geometric train targets differ from frozen proposal lineage")
    layout_path = Path(str(contract["frozen_layout_features"]))
    layout, _layout_metadata = load_context_attention_frozen_layout(layout_path)
    source_rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    candidate_tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    split_names = np.asarray(layout["split_names"]).astype(str)
    proposal_tracks = _load_proposal_tracks(proposal_file)
    if (
        np.any(source_rows < 0)
        or np.any(source_rows >= len(proposal_tracks))
        or not np.array_equal(proposal_tracks[source_rows], candidate_tracks)
    ):
        raise ValueError("geometric train targets do not align with frozen candidates")
    train_rows, membership, audit = _train_geometric_targets_with_explicit_null(
        proposals_path=proposal_file,
        source_rows=source_rows,
        candidate_tracks=candidate_tracks,
        split_names=split_names,
        positive_threshold_px=float(geometric_positive_threshold_px),
    )
    expected_train_rows = np.flatnonzero(split_names == "train").astype(np.int64)
    if not np.array_equal(train_rows, expected_train_rows):
        raise RuntimeError("geometric train target rows differ from frozen train split")
    if membership.dtype != np.dtype(bool) or membership.shape != (
        len(train_rows),
        candidate_tracks.shape[1] + 1,
    ):
        raise RuntimeError("geometric train target membership is invalid")
    train_source_rows = source_rows[train_rows]
    train_tracks = candidate_tracks[train_rows]
    metadata = {
        "format": GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "training_split": "train",
        "contract_sha256": file_sha256_short(Path(contract_path)),
        "frozen_layout_features_sha256": contract["frozen_layout_features_sha256"],
        "proposals_sha256": file_sha256_short(proposal_file),
        "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
        "layout_row_indices_sha256": _array_sha256_short(train_rows),
        "source_row_indices_sha256": _array_sha256_short(train_source_rows),
        "candidate_track_ids_sha256": _array_sha256_short(train_tracks),
        "source_residual_memberships_used": "frozen_train_source_rows_only",
        "fit_must_not_open_full_residual_archive": True,
        "audit": audit,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        layout_row_indices=train_rows,
        source_row_indices=train_source_rows,
        candidate_track_ids=train_tracks,
        target_membership=np.asarray(membership, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_context_attention_geometric_train_targets",
        "output": str(target_path),
        "output_sha256": file_sha256_short(target_path),
        "row_count": int(len(train_rows)),
        "target_audit": audit,
        "protocol": {
            "training_split": "train",
            "contains_validation_or_test_targets": False,
            "fit_must_not_open_full_residual_archive": True,
            "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_context_attention_geometric_train_targets(
        contract_path=Path(args.contract),
        proposals_path=Path(args.proposals),
        geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
