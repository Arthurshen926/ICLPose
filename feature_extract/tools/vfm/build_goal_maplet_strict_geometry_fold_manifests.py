"""Build one fold's geometry-head train/eval manifests from its strict 2DGS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def build_manifests(
    *, geometry_manifests: Sequence[Path], protocol_path: Path,
    fold_id: str, output_dir: Path, force: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.glob("*.json")) and not force:
        raise FileExistsError("refusing to overwrite strict geometry manifests")
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if str(fold_id) == "final_alltrain":
        fold = {
            "fold_id": "final_alltrain", "held_query_trajectories": [],
            "mapping_count": int(protocol["official_train"]["count"]),
            "mapping_image_ids_sha256": protocol["official_train"]["image_ids_sha256"],
        }
    else:
        found = [
            value for value in protocol["development"]["folds"]
            if str(value["fold_id"]) == str(fold_id)
        ]
        if len(found) != 1:
            raise ValueError("fold_id is not unique in protocol")
        fold = found[0]
    source_paths = [Path(value) for value in geometry_manifests]
    records = []
    source_geometry_hashes: set[str] = set()
    for path in source_paths:
        payload = json.loads(path.read_text())
        records.extend(payload["records"])
        source_geometry_hashes.update(
            str(value) for value in payload["inputs"]["contributor_geometry_source_sha256"]
        )
    image_ids = [str(value["image_id"]) for value in records]
    official = protocol["official_train"]
    if (
        len(image_ids) != len(set(image_ids))
        or len(image_ids) != int(official["count"])
        or ordered_id_sha256(image_ids) != str(official["image_ids_sha256"])
        or len(source_geometry_hashes) != 1
    ):
        raise ValueError("strict geometry labels do not cover one geometry/all train")
    held = set(str(value) for value in fold["held_query_trajectories"])
    train = [
        value for value in records
        if str(value["image_id"]).split("/", 1)[0] not in held
    ]
    evaluation = [
        value for value in records
        if str(value["image_id"]).split("/", 1)[0] in held
    ]
    if (
        len(train) != int(fold["mapping_count"])
        or ordered_id_sha256(value["image_id"] for value in train)
        != str(fold["mapping_image_ids_sha256"])
    ):
        raise ValueError("strict geometry training split differs from map split")
    source = {
        "geometry_manifests": [str(value) for value in source_paths],
        "geometry_manifest_sha256": [file_sha256(value) for value in source_paths],
        "strict_geometry_source_sha256": next(iter(source_geometry_hashes)),
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, values in (("train", train), ("eval", evaluation)):
        payload = {
            "artifact_type": "goal_maplet_strict_geometry_route_fold_manifest_v1",
            "fold_id": str(fold_id), "role": role,
            "held_query_trajectories": sorted(held),
            "record_count": len(values),
            "image_ids_sha256": ordered_id_sha256(
                str(value["image_id"]) for value in values
            ),
            "source": source, "records": values,
        }
        (output_dir / f"{role}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    report = {
        "artifact_type": "goal_maplet_strict_geometry_fold_manifest_summary_v1",
        "fold_id": str(fold_id), "training_count": len(train),
        "evaluation_count": len(evaluation),
        "held_query_trajectories": sorted(held), "source": source,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_manifests", nargs="+", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = build_manifests(
        geometry_manifests=[Path(value) for value in args.geometry_manifests],
        protocol_path=Path(args.protocol), fold_id=str(args.fold_id),
        output_dir=Path(args.output_dir), force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
