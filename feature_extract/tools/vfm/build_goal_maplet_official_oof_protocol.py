"""Build and validate the official-train GoalMaplet OOF protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.official_oof_protocol import build_stmarys_protocol


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_pose_file", required=True)
    parser.add_argument("--test_pose_file", required=True)
    parser.add_argument("--train_token_manifest", required=True)
    parser.add_argument("--test_token_manifest", required=True)
    parser.add_argument("--mapping_camera_manifest", required=True)
    parser.add_argument("--mapping_depth_bank", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _camera_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("mapping camera manifest has no cameras object")
    return list(cameras)


def _depth_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("mapping depth bank has no records list")
    return [str(record["image_id"]) for record in records]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite protocol: {output}")
    protocol = build_stmarys_protocol(
        train_pose_file=Path(args.train_pose_file),
        test_pose_file=Path(args.test_pose_file),
        train_token_manifest=Path(args.train_token_manifest),
        test_token_manifest=Path(args.test_token_manifest),
        extra_train_manifests={
            "mapping_camera_manifest": _camera_ids(
                Path(args.mapping_camera_manifest)
            ),
            "mapping_depth_bank": _depth_ids(Path(args.mapping_depth_bank)),
        },
    )
    protocol["integrity"]["mapping_camera_manifest"] = str(
        args.mapping_camera_manifest
    )
    protocol["integrity"]["mapping_depth_bank"] = str(args.mapping_depth_bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
