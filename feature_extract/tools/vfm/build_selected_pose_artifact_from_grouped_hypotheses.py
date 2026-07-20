"""Extract a fixed external source pose from target-free grouped hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.selected_pose_artifact import (
    selected_pose_rows_from_grouped_hypotheses,
    write_selected_pose_artifact,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis-artifacts",
        required=True,
        help="comma-separated complete grouped-hypothesis shard set",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-evaluation-label",
        default="",
        help="optional grouped-hypothesis label; required only for multi-label inputs",
    )
    parser.add_argument(
        "--split-json",
        default="",
        help="require exact train/validation/test source coverage",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _expected_query_ids(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    required = ("train", "validation", "test")
    if any(not isinstance(payload.get(name), list) for name in required):
        raise ValueError("split JSON requires train, validation, and test lists")
    return {name: [str(query_id) for query_id in payload[name]] for name in required}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = tuple(
        Path(value.strip())
        for value in str(args.hypothesis_artifacts).split(",")
        if value.strip()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "selected_pose_inference_only_v1.npz"
    summary_path = output_dir / "summary.json"
    if (artifact_path.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite selected pose artifact output")
    split_path = None if not str(args.split_json) else Path(args.split_json)
    rows, source_manifest = selected_pose_rows_from_grouped_hypotheses(
        paths,
        evaluation_label=str(args.source_evaluation_label),
        expected_query_ids=_expected_query_ids(split_path),
    )
    write_selected_pose_artifact(
        artifact_path,
        rows,
        source_manifest=source_manifest,
    )
    summary = {
        "stage": "selected_pose_from_grouped_hypotheses",
        "protocol": source_manifest["protocol"],
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in paths],
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in paths],
            "split_json": None if split_path is None else str(split_path),
            "source_evaluation_label": str(rows[0]["evaluation_label"]),
        },
        "outputs": {
            "selected_pose_artifact": str(artifact_path),
            "selected_pose_artifact_sha256": file_sha256_short(artifact_path),
            "query_count": int(len(rows)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
