"""Merge geometry-label shards into route-grouped OOF train/eval manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_manifests", required=True, nargs="+")
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and any(output.glob("*.json")) and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite geometry folds: {output}")
    protocol_path = Path(args.protocol_json)
    protocol = json.loads(protocol_path.read_text())
    source_paths = [Path(value) for value in args.geometry_manifests]
    records = []
    for path in source_paths:
        records.extend(json.loads(path.read_text())["records"])
    image_ids = [str(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("geometry label shards overlap")
    expected = protocol["official_train"]
    if (
        len(image_ids) != int(expected["count"])
        or ordered_id_sha256(image_ids) != str(expected["image_ids_sha256"])
    ):
        raise ValueError("geometry labels do not cover official train exactly")
    source = {
        "geometry_manifests": [str(path) for path in source_paths],
        "geometry_manifest_sha256": [file_sha256(path) for path in source_paths],
        "protocol_json": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for fold in protocol["development"]["folds"]:
        held = set(str(value) for value in fold["held_query_trajectories"])
        evaluation = [
            record for record in records
            if str(record["image_id"]).split("/", 1)[0] in held
        ]
        training = [
            record for record in records
            if str(record["image_id"]).split("/", 1)[0] not in held
        ]
        for role, values in (("train", training), ("eval", evaluation)):
            payload = {
                "artifact_type": "goal_maplet_geometry_route_fold_manifest_v1",
                "fold_id": str(fold["fold_id"]),
                "role": role,
                "held_query_trajectories": sorted(held),
                "record_count": len(values),
                "image_ids_sha256": ordered_id_sha256(
                    str(record["image_id"]) for record in values
                ),
                "source": source,
                "records": values,
            }
            (output / f"{fold['fold_id']}_{role}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        summary.append({
            "fold_id": str(fold["fold_id"]),
            "training_count": len(training),
            "evaluation_count": len(evaluation),
            "held_query_trajectories": sorted(held),
        })
    all_train = {
        "artifact_type": "goal_maplet_geometry_all_official_train_manifest_v1",
        "role": "final_fit_or_fit_diagnostic",
        "record_count": len(records),
        "image_ids_sha256": ordered_id_sha256(image_ids),
        "source": source,
        "records": records,
    }
    (output / "all_official_train.json").write_text(
        json.dumps(all_train, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "artifact_type": "goal_maplet_geometry_oof_manifest_summary_v1",
        "official_train_count": len(records),
        "official_train_image_ids_sha256": ordered_id_sha256(image_ids),
        "folds": summary,
        "all_official_train_manifest": str(output / "all_official_train.json"),
        "source": source,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
