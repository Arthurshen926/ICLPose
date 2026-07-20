"""Write a PCA training-image manifest from a frozen disjoint maplet index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    _load_maplet_support_index,
)
from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "maplet_support_image_manifest_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument(
        "--excluded_query_split_json",
        required=True,
        help="frozen split containing every train/validation/test query image",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _excluded_query_ids(path: Path) -> tuple[set[str], dict[str, int]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "stratified_landmark_query_split_v1":
        raise ValueError("excluded query split has an unsupported format")
    excluded: set[str] = set()
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        values = payload.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"excluded query split has no {split} images")
        ids = {str(value).strip() for value in values if str(value).strip()}
        if len(ids) != len(values):
            raise ValueError(f"excluded query split has duplicate or empty {split} IDs")
        overlap = excluded & ids
        if overlap:
            raise ValueError(
                f"excluded query split crosses partitions: {sorted(overlap)[:5]}"
            )
        excluded.update(ids)
        counts[split] = int(len(ids))
    return excluded, counts


def build_maplet_support_image_manifest(
    *,
    maplet_support_index: Path,
    excluded_query_split_json: Path,
    output: Path,
    summary_json: Path,
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite maplet support manifest outputs")
    maplet, metadata = _load_maplet_support_index(Path(maplet_support_index))
    support_ids = sorted(set(maplet["support_image_ids"].astype(str).tolist()))
    if not support_ids:
        raise ValueError("maplet support index has no support images")
    excluded_ids, excluded_counts = _excluded_query_ids(Path(excluded_query_split_json))
    overlap = sorted(set(support_ids) & excluded_ids)
    if overlap:
        raise ValueError(
            "maplet support images overlap frozen query images: "
            f"{overlap[:10]}"
        )
    payload = {
        "format": ARTIFACT_FORMAT,
        "records": [{"image_id": image_id} for image_id in support_ids],
        "metadata": {
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "maplet_support_index": str(Path(maplet_support_index)),
            "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
            "maplet_support_index_format": metadata.get("format"),
            "excluded_query_split_json": str(Path(excluded_query_split_json)),
            "excluded_query_split_sha256": file_sha256_short(
                Path(excluded_query_split_json)
            ),
            "excluded_query_counts": excluded_counts,
            "excluded_query_image_count": int(len(excluded_ids)),
            "support_image_count": int(len(support_ids)),
            "support_query_overlap_count": 0,
            "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "build_maplet_support_image_manifest",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "metadata": payload["metadata"],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_maplet_support_image_manifest(
        maplet_support_index=Path(args.maplet_support_index),
        excluded_query_split_json=Path(args.excluded_query_split_json),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
