"""Run disjoint full-map LoFTR cache and anchor-evidence shards.

This worker consumes only target-free frozen S0 appearance artifacts. Each
selected query is paired with every mapping support image, then the resulting
cache is sampled only at the already-fixed candidate support observations.
There is no image shortlist, no submap, and no label or pose input here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.tools.vfm.build_frozen_loftr_candidate_anchor_evidence import (
    build_frozen_loftr_candidate_anchor_evidence,
)
from feature_extract.tools.vfm.build_frozen_loftr_pair_cache import (
    build_frozen_loftr_pair_cache,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_ARTIFACT_FORMAT,
    load_frozen_appearance_probe_features,
)


@dataclass(frozen=True)
class QueryArtifact:
    path: Path
    query_id: str
    split_name: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--appearance-artifacts")
    source.add_argument("--appearance-artifact-glob")
    parser.add_argument("--mapping-support-manifest", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--hloc-root", default="third_party/Hierarchical-Localization")
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--pair-batch-size", type=int, default=6)
    parser.add_argument("--match-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def _paths(*, value: str | None, pattern: str | None) -> tuple[Path, ...]:
    if value is not None:
        paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    elif pattern is not None:
        paths = tuple(Path(item) for item in sorted(glob.glob(str(pattern))))
    else:  # pragma: no cover - argparse enforces one source.
        raise ValueError("appearance artifacts are required")
    if not paths or len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("appearance artifacts must be unique existing files")
    return paths


def _query_artifacts(paths: Sequence[Path]) -> tuple[QueryArtifact, ...]:
    result: list[QueryArtifact] = []
    for path in paths:
        features = load_frozen_appearance_probe_features([Path(path)])
        metadata = dict(features.metadata)
        if (
            metadata.get("format") != FROZEN_APPEARANCE_ARTIFACT_FORMAT
            or len(features.query_ids) != 192
            or len(set(features.query_ids.tolist())) != 1
            or len(set(features.split_names.tolist())) != 1
        ):
            raise ValueError(f"{path}: worker requires one complete direct target-free query artifact")
        result.append(
            QueryArtifact(
                path=Path(path),
                query_id=str(features.query_ids[0]),
                split_name=str(features.split_names[0]),
            )
        )
    ordered = tuple(sorted(result, key=lambda item: (item.query_id, item.split_name)))
    if len({item.query_id for item in ordered}) != len(ordered):
        raise ValueError("appearance artifacts repeat a query id")
    return ordered


def partition_query_artifacts(
    artifacts: Sequence[QueryArtifact], *, shard_count: int, shard_index: int
) -> tuple[tuple[int, QueryArtifact], ...]:
    if (
        int(shard_count) <= 0
        or int(shard_index) < 0
        or int(shard_index) >= int(shard_count)
    ):
        raise ValueError("LoFTR worker shard arguments are invalid")
    return tuple(
        (index, artifact)
        for index, artifact in enumerate(artifacts)
        if index % int(shard_count) == int(shard_index)
    )


def _query_slug(*, index: int, query_id: str) -> str:
    digest = hashlib.sha256(str(query_id).encode("utf8")).hexdigest()[:10]
    readable = str(query_id).replace("/", "_").replace(".", "_")
    return f"{int(index):03d}_{readable}_{digest}"


def run_frozen_loftr_pair_cache_shard(
    *,
    appearance_artifacts: Sequence[Path],
    mapping_support_manifest: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    colmap_model_dir: Path,
    image_root: Path,
    hloc_root: Path,
    loftr_checkpoint: Path,
    pair_batch_size: int,
    match_chunk_size: int,
    device: str,
    shard_count: int,
    shard_index: int,
    output_root: Path,
) -> dict[str, object]:
    """Run one deterministic, label-free shard and write no shared manifest."""

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"refusing to reuse LoFTR worker output root: {root}")
    artifacts = _query_artifacts(tuple(Path(path) for path in appearance_artifacts))
    selected = partition_query_artifacts(
        artifacts, shard_count=int(shard_count), shard_index=int(shard_index)
    )
    if not selected:
        raise ValueError("LoFTR worker shard selects no queries")
    if int(pair_batch_size) <= 0 or int(match_chunk_size) <= 0:
        raise ValueError("LoFTR worker batch sizes must be positive")
    root.mkdir(parents=True)
    cache_root = root / "pair_caches"
    anchor_root = root / "anchor_evidence"
    cache_root.mkdir()
    anchor_root.mkdir()
    completed = []
    started = time.monotonic()
    for position, artifact in selected:
        slug = _query_slug(index=position, query_id=artifact.query_id)
        cache_path = cache_root / f"{slug}.npz"
        anchor_path = anchor_root / f"{slug}.npz"
        anchor_summary = anchor_root / f"{slug}.summary.json"
        pair_summary = build_frozen_loftr_pair_cache(
            query_id=artifact.query_id,
            mapping_support_manifest=Path(mapping_support_manifest),
            maplet_support_index=Path(maplet_support_index),
            image_root=Path(image_root),
            hloc_root=Path(hloc_root),
            loftr_checkpoint=Path(loftr_checkpoint),
            loftr_weights="outdoor",
            match_threshold=0.2,
            resize_width=960,
            resize_height=540,
            pair_batch_size=int(pair_batch_size),
            device=str(device),
            output=cache_path,
        )
        anchor_summary_data = build_frozen_loftr_candidate_anchor_evidence(
            appearance_artifact=artifact.path,
            loftr_pair_cache=cache_path,
            maplet_support_index=Path(maplet_support_index),
            support_geometry_index=Path(support_geometry_index),
            colmap_model_dir=Path(colmap_model_dir),
            image_root=Path(image_root),
            loftr_checkpoint=Path(loftr_checkpoint),
            match_chunk_size=int(match_chunk_size),
            device=str(device),
            output=anchor_path,
            summary_json=anchor_summary,
        )
        completed.append(
            {
                "global_index": int(position),
                "query_id": artifact.query_id,
                "split_name": artifact.split_name,
                "appearance_artifact": {
                    "path": str(artifact.path),
                    "sha256": file_sha256_short(artifact.path),
                },
                "pair_cache": pair_summary,
                "anchor_evidence": anchor_summary_data,
            }
        )
        print(
            json.dumps(
                {
                    "event": "query_complete",
                    "global_index": int(position),
                    "query_id": artifact.query_id,
                    "split_name": artifact.split_name,
                    "pair_cache_sha256": pair_summary["output_sha256"],
                    "anchor_evidence_sha256": anchor_summary_data["output_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    summary = {
        "stage": "run_frozen_loftr_pair_cache_shard",
        "shard": {"index": int(shard_index), "count": int(shard_count)},
        "query_count": int(len(completed)),
        "completed": completed,
        "elapsed_seconds": float(time.monotonic() - started),
        "protocol": {
            "target_free": True,
            "full_mapping_pair_cache": True,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_frozen_loftr_pair_cache_shard(
        appearance_artifacts=_paths(
            value=args.appearance_artifacts, pattern=args.appearance_artifact_glob
        ),
        mapping_support_manifest=Path(args.mapping_support_manifest),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        colmap_model_dir=Path(args.colmap_model_dir),
        image_root=Path(args.image_root),
        hloc_root=Path(args.hloc_root),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        pair_batch_size=int(args.pair_batch_size),
        match_chunk_size=int(args.match_chunk_size),
        device=str(args.device),
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "shard": summary["shard"],
                "query_count": summary["query_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
