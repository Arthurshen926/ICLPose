"""Materialize route-clean RADIO and pose inputs for strict map rebuilds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.cambridge_pose_lattice import (
    parse_cambridge_pose_file,
    write_cambridge_pose_file,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.tokens import TokenBankManifest


def _fold_definition(protocol: dict[str, object], fold_id: str) -> dict[str, object]:
    if str(fold_id) == "final_alltrain":
        return {
            "fold_id": "final_alltrain",
            "held_query_trajectories": [],
            "mapping_trajectories": sorted(
                protocol["official_train"]["trajectory_counts"]
            ),
            "mapping_count": int(protocol["official_train"]["count"]),
            "mapping_image_ids_sha256": protocol["official_train"][
                "image_ids_sha256"
            ],
        }
    matches = [
        value for value in protocol["development"]["folds"]
        if str(value["fold_id"]) == str(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError("fold_id is not unique in protocol")
    return matches[0]


def build_inputs(
    *,
    protocol_path: Path,
    fold_id: str,
    output_manifest: Path,
    output_pose_file: Path,
    output_json: Path,
    force: bool = False,
) -> dict[str, object]:
    outputs = (Path(output_manifest), Path(output_pose_file), Path(output_json))
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite strict mapping inputs")
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    fold = _fold_definition(protocol, str(fold_id))
    mapping_routes = set(str(value) for value in fold["mapping_trajectories"])
    held_routes = set(str(value) for value in fold["held_query_trajectories"])
    if mapping_routes & held_routes:
        raise ValueError("mapping and held routes overlap")

    source_pose = Path(protocol["official_train"]["pose_file"])
    source_manifest = Path(protocol["official_train"]["token_manifest"])
    poses = [
        value for value in parse_cambridge_pose_file(source_pose)
        if value.image_id.split("/", 1)[0] in mapping_routes
    ]
    manifest = TokenBankManifest.from_json(source_manifest)
    records = tuple(
        value for value in manifest.records
        if value.image_id.split("/", 1)[0] in mapping_routes
    )
    pose_ids = {value.image_id for value in poses}
    token_ids = {value.image_id for value in records}
    expected_count = int(fold["mapping_count"])
    expected_hash = str(fold["mapping_image_ids_sha256"])
    if (
        pose_ids != token_ids
        or len(pose_ids) != expected_count
        or ordered_id_sha256(pose_ids) != expected_hash
    ):
        raise ValueError("strict mapping inputs differ from protocol fold")
    if any(value.split("/", 1)[0] in held_routes for value in pose_ids):
        raise ValueError("strict mapping input contains a held route")

    output_manifest = Path(output_manifest)
    output_pose_file = Path(output_pose_file)
    TokenBankManifest(records).to_json(output_manifest)
    write_cambridge_pose_file(poses, output_pose_file)
    payload = {
        "artifact_type": "goal_maplet_strict_mapping_inputs_v1",
        "fold_id": str(fold_id),
        "mapping_trajectories": sorted(mapping_routes),
        "held_query_trajectories": sorted(held_routes),
        "mapping_image_count": len(pose_ids),
        "mapping_image_ids_sha256": ordered_id_sha256(pose_ids),
        "contains_held_query": False,
        "contains_official_test": False,
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_pose_file": str(source_pose),
        "source_pose_file_sha256": file_sha256(source_pose),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": file_sha256(output_manifest),
        "output_pose_file": str(output_pose_file),
        "output_pose_file_sha256": file_sha256(output_pose_file),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold_id", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_pose_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    payload = build_inputs(
        protocol_path=Path(args.protocol), fold_id=str(args.fold_id),
        output_manifest=Path(args.output_manifest),
        output_pose_file=Path(args.output_pose_file),
        output_json=Path(args.output_json), force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
