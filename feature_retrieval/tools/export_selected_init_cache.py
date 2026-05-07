#!/usr/bin/env python3
"""Export an init cache whose active init pose comes from selected candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    load_retrieval_init_entries,
    save_retrieval_init_entries,
)


def _copy_entry(entry: Dict) -> Dict:
    copied: Dict = {}
    for key, value in entry.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _coerce_index(value, *, entry_idx: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"selected index for entry {entry_idx} must be an integer, got bool")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"selected index for entry {entry_idx} must be an integer")
    return int(value)


def export_selected_init_cache(
    init_cache_path: str,
    selected_indices: Sequence[int],
    save_path: str,
    source_name: str = "selected_candidate",
) -> Tuple[List[Dict], Dict]:
    """Select one candidate per query from an existing retrieval init cache.

    The candidate arrays remain intact; only the active ``pose_init`` and
    top-level retrieval metadata are switched to the selected candidate.
    """
    entries, stats = load_retrieval_init_entries(str(init_cache_path))
    if len(selected_indices) != len(entries):
        raise ValueError(
            f"selected_indices length {len(selected_indices)} does not match entries length {len(entries)}"
        )

    exported_entries: List[Dict] = []
    for entry_idx, (entry, raw_idx) in enumerate(zip(entries, selected_indices)):
        selected_idx = _coerce_index(raw_idx, entry_idx=entry_idx)
        poses = np.asarray(entry["pose_init_candidates"], dtype=np.float32)
        valid_mask = np.asarray(entry["candidate_valid_mask"], dtype=bool).reshape(-1)
        if selected_idx < 0 or selected_idx >= len(poses):
            raise ValueError(
                f"selected index {selected_idx} out of range for entry {entry_idx} with {len(poses)} candidates"
            )
        if selected_idx >= len(valid_mask) or not bool(valid_mask[selected_idx]):
            raise ValueError(f"selected index {selected_idx} is not valid for entry {entry_idx}")

        frame_ids = np.asarray(entry["retrieval_frame_ids_candidates"], dtype=np.int64)
        image_names = list(entry["retrieval_image_names_candidates"])
        scores = np.asarray(entry["retrieval_scores_candidates"], dtype=np.float32)
        if selected_idx >= len(frame_ids) or selected_idx >= len(image_names) or selected_idx >= len(scores):
            raise ValueError(f"selected index {selected_idx} out of range for retrieval metadata in entry {entry_idx}")

        exported = _copy_entry(entry)
        exported["pose_init"] = poses[selected_idx].astype(np.float32).copy()
        exported["retrieval_frame_id"] = int(frame_ids[selected_idx])
        exported["retrieval_image_name"] = str(image_names[selected_idx])
        exported["retrieval_score"] = float(scores[selected_idx])
        exported["init_source"] = str(source_name)
        exported_entries.append(exported)

    exported_stats = dict(stats or {})
    exported_stats.update(
        {
            "selection_source": "selected_candidate",
            "init_cache": str(init_cache_path),
            "num_selected": len(exported_entries),
            "source_name": str(source_name),
        }
    )
    save_retrieval_init_entries(exported_entries, exported_stats, str(save_path))
    return exported_entries, exported_stats


def _parse_indices_json(indices_json: str) -> List[int]:
    try:
        parsed = json.loads(indices_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--indices_json must be a JSON list of integer candidate indices") from exc
    if not isinstance(parsed, list):
        raise ValueError("--indices_json must be a JSON list of integer candidate indices")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_cache", required=True, help="Path to the source retrieval init cache .npz")
    parser.add_argument("--indices_json", required=True, help="JSON list with one selected candidate index per query")
    parser.add_argument("--save_path", required=True, help="Path for the exported init cache .npz")
    parser.add_argument("--source_name", default="selected_candidate", help="init_source value for exported entries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        selected_indices = _parse_indices_json(args.indices_json)
        exported_entries, _stats = export_selected_init_cache(
            args.init_cache,
            selected_indices,
            args.save_path,
            source_name=args.source_name,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Exported {len(exported_entries)} selected init entries to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
