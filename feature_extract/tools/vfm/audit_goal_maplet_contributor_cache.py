"""Audit completeness and lineage of a GoalMaplet contributor-cache directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument(
        "--split", choices=("official_train", "official_test"),
        default="official_train",
    )
    parser.add_argument("--shard_summaries", nargs="*", default=[])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    protocol = json.loads(Path(args.protocol_json).read_text())
    expected = protocol[str(args.split)]
    paths = sorted(Path(args.contributors).glob("*.npz"))
    image_ids: list[str] = []
    geometry_hashes: set[str] = set()
    clean_geometry_hashes: set[str] = set()
    clean_index_hashes: set[str] = set()
    token_missing: list[str] = []
    shape_failures: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            ids = np.asarray(data["topk_ids"])
            weights = np.asarray(data["topk_weights"])
            depth = np.asarray(data["dominant_depth"])
        image_id = str(metadata["image_id"])
        image_ids.append(image_id)
        geometry_hashes.add(str(metadata["geometry_source_sha256"]))
        clean_geometry_hashes.add(str(metadata["clean_geometry_source_sha256"]))
        clean_index_hashes.add(str(metadata["clean_source_index_sha256"]))
        token_path = Path(str(metadata["token_path"]))
        if not token_path.exists():
            token_missing.append(image_id)
        expected_shape = (
            int(metadata["height"]),
            int(metadata["width"]),
            int(metadata["top_k"]),
        )
        if ids.shape != expected_shape or weights.shape != expected_shape:
            shape_failures.append(image_id)
        if depth.shape != expected_shape[:2]:
            shape_failures.append(image_id)
    duplicates = sorted(
        image_id
        for image_id in set(image_ids)
        if image_ids.count(image_id) != 1
    )
    actual_hash = ordered_id_sha256(image_ids)
    shard_rows: list[str] = []
    shard_counts = []
    for value in args.shard_summaries:
        payload = json.loads(Path(value).read_text())
        shard_counts.append(int(payload["view_count"]))
        shard_rows.extend(str(row["image_id"]) for row in payload["records"])
    shard_complete = (
        not args.shard_summaries
        or (
            len(shard_rows) == len(image_ids)
            and len(set(shard_rows)) == len(image_ids)
            and set(shard_rows) == set(image_ids)
        )
    )
    checks = {
        "file_count_matches_protocol": len(paths) == int(expected["count"]),
        "unique_image_ids": not duplicates and len(image_ids) == len(paths),
        "image_ids_match_protocol_hash": actual_hash
        == str(expected["image_ids_sha256"]),
        "one_geometry_source": len(geometry_hashes) == 1,
        "one_clean_geometry_source": len(clean_geometry_hashes) == 1,
        "one_clean_source_index_set": len(clean_index_hashes) == 1,
        "all_token_paths_exist": not token_missing,
        "all_array_shapes_match_metadata": not shape_failures,
        "shard_summaries_partition_cache": shard_complete,
    }
    report = {
        "artifact_type": "goal_maplet_contributor_cache_audit_v1",
        "contributors": str(args.contributors),
        "protocol_json": str(args.protocol_json),
        "split": str(args.split),
        "file_count": len(paths),
        "image_ids_sha256": actual_hash,
        "geometry_source_sha256": sorted(geometry_hashes),
        "clean_geometry_source_sha256": sorted(clean_geometry_hashes),
        "clean_source_index_sha256": sorted(clean_index_hashes),
        "shard_counts": shard_counts,
        "duplicate_image_ids": duplicates,
        "missing_token_image_ids": token_missing,
        "shape_failure_image_ids": sorted(set(shape_failures)),
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not report["pass"]:
        raise ValueError(json.dumps(report, indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
