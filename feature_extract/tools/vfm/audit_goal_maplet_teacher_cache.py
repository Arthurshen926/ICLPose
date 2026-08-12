"""Audit complete offline teacher coverage without retaining mapping RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument("--shard_summaries", nargs="*", default=[])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite teacher audit: {output}")
    protocol = json.loads(Path(args.protocol_json).read_text())
    expected = protocol["official_train"]
    paths = sorted(Path(args.teacher_cache).glob("*.npz"))
    image_ids = []
    malformed = []
    forbidden_metadata = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                image_id = str(metadata["image_id"])
                xy = np.asarray(data["token_xy"])
                teacher = [
                    np.asarray(data[name])
                    for name in ("dino_v3_7b", "sam3", "siglip2-g")
                ]
                summaries = [
                    np.asarray(data[name + "_summary"])
                    for name in ("dino_v3_7b", "sam3", "siglip2-g")
                ]
            count = int(xy.shape[0]) if xy.ndim == 2 and xy.shape[1] == 2 else -1
            if (
                count <= 0
                or any(value.ndim != 2 or value.shape[0] != count for value in teacher)
                or any(value.ndim != 1 for value in summaries)
                or not all(np.all(np.isfinite(value)) for value in [xy, *teacher, *summaries])
            ):
                malformed.append(image_id)
            if bool(metadata.get("stores_rgb")) or bool(metadata.get("stores_image_path")):
                forbidden_metadata.append(image_id)
            image_ids.append(image_id)
        except Exception:
            malformed.append(path.name)
    shard_ids = []
    shard_counts = []
    for summary_name in args.shard_summaries:
        summary = json.loads(Path(summary_name).read_text())
        shard_counts.append(int(summary["processed"]))
        shard_ids.extend(str(value) for value in summary["image_ids"])
    duplicate_ids = sorted(
        value for value in set(image_ids) if image_ids.count(value) != 1
    )
    checks = {
        "file_count_matches_official_train": len(paths) == int(expected["count"]),
        "unique_image_ids": not duplicate_ids and len(image_ids) == len(paths),
        "image_ids_match_protocol_hash": (
            ordered_id_sha256(image_ids) == str(expected["image_ids_sha256"])
        ),
        "teacher_arrays_well_formed": not malformed,
        "no_rgb_or_image_path_stored": not forbidden_metadata,
        "shard_summaries_partition_cache": (
            not args.shard_summaries
            or (
                len(shard_ids) == len(image_ids)
                and len(set(shard_ids)) == len(image_ids)
                and set(shard_ids) == set(image_ids)
            )
        ),
    }
    result = {
        "artifact_type": "goal_maplet_offline_teacher_cache_audit_v1",
        "teacher_cache": str(args.teacher_cache),
        "protocol_json": str(args.protocol_json),
        "file_count": len(paths),
        "image_ids_sha256": ordered_id_sha256(image_ids),
        "shard_counts": shard_counts,
        "duplicates": duplicate_ids,
        "malformed": malformed,
        "forbidden_metadata": forbidden_metadata,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not result["pass"]:
        raise ValueError(json.dumps(result, indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
