"""Export complete frozen LoFTR global-alignment overlays safely.

The LoFTR pair cache has already matched every query to every mapping image.
This launcher never invokes LoFTR and never chooses a reference image.  It
only joins each immutable S0 top-20 candidate shard to the cache entry for the
same query, then fits target-free support-to-query homographies for the
support observations that are already attached to those candidates.

The export manifest is intentionally strict: a resume is accepted only when
the complete source/cache manifests and every dependency hash are unchanged.
This prevents an old pair cache from being silently overlaid on a newer frozen
candidate layout.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from feature_extract.tools.vfm.build_frozen_loftr_global_alignment_evidence import (
    build_frozen_loftr_global_alignment_evidence,
)
from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "frozen_loftr_global_alignment_export_v1"
ALIGNMENT_FILENAME = "frozen_loftr_global_alignment_evidence_v1.npz"
SUMMARY_FILENAME = "summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--source-glob",
        default=(
            "s88[56]_s1_absresidual_full_*/"
            "frozen_multiscale_candidate_appearance_v1.npz"
        ),
    )
    parser.add_argument("--pair-cache-root", required=True)
    parser.add_argument("--pair-cache-glob", default="**/pair_caches/*.npz")
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-shards", type=int, default=84)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _single_string(array: np.ndarray, *, field: str, path: Path) -> str:
    values = np.asarray(array).astype(str).reshape(-1)
    if len(values) != 1 or not str(values[0]):
        raise ValueError(f"{path}: {field} must contain exactly one non-empty query ID")
    return str(values[0])


def _source_query_id(path: Path) -> str:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "verification_query_ids" not in payload.files:
            raise ValueError(f"{path}: frozen appearance artifact lacks query IDs")
        values = np.asarray(payload["verification_query_ids"]).astype(str).reshape(-1)
    if len(values) != 192 or len(set(values.tolist())) != 1 or not str(values[0]):
        raise ValueError(f"{path}: frozen appearance artifact must contain one 192-row query")
    return str(values[0])


def _cache_query_id(path: Path) -> str:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "query_id" not in payload.files:
            raise ValueError(f"{path}: frozen LoFTR cache lacks query ID")
        return _single_string(payload["query_id"], field="query_id", path=Path(path))


def _index_paths_by_query_id(
    paths: Sequence[Path], *, kind: str
) -> dict[str, Path]:
    if not paths:
        raise ValueError(f"{kind} paths are empty")
    reader = _source_query_id if str(kind) == "source" else _cache_query_id
    indexed: dict[str, Path] = {}
    for path in tuple(Path(item) for item in paths):
        query_id = reader(path)
        if query_id in indexed:
            raise ValueError(f"duplicate {kind} query ID: {query_id}")
        indexed[query_id] = path
    return indexed


def _source_cache_manifest(
    *, source_by_query: Mapping[str, Path], cache_by_query: Mapping[str, Path]
) -> list[dict[str, str]]:
    if set(source_by_query) != set(cache_by_query):
        source_only = sorted(set(source_by_query).difference(cache_by_query))
        cache_only = sorted(set(cache_by_query).difference(source_by_query))
        raise ValueError(
            "frozen appearance and LoFTR cache query sets differ: "
            f"source_only={source_only[:3]}, cache_only={cache_only[:3]}"
        )
    return [
        {
            "query_id": query_id,
            "source": str(source_by_query[query_id].resolve()),
            "source_sha256": file_sha256_short(source_by_query[query_id]),
            "pair_cache": str(cache_by_query[query_id].resolve()),
            "pair_cache_sha256": file_sha256_short(cache_by_query[query_id]),
        }
        for query_id in sorted(source_by_query)
    ]


def _run_config(
    *,
    source_cache_manifest: Sequence[Mapping[str, str]],
    maplet_support_index: Path,
    support_geometry_index: Path,
    image_root: Path,
    loftr_checkpoint: Path,
    workers: int,
) -> dict[str, Any]:
    return {
        "format": ARTIFACT_FORMAT,
        "version": 1,
        "source_cache_manifest": [dict(item) for item in source_cache_manifest],
        "maplet_support_index": str(Path(maplet_support_index).resolve()),
        "maplet_support_index_sha256": file_sha256_short(Path(maplet_support_index)),
        "support_geometry_index": str(Path(support_geometry_index).resolve()),
        "support_geometry_index_sha256": file_sha256_short(
            Path(support_geometry_index)
        ),
        "image_root": str(Path(image_root).resolve()),
        "loftr_checkpoint": str(Path(loftr_checkpoint).resolve()),
        "loftr_checkpoint_sha256": file_sha256_short(Path(loftr_checkpoint)),
        "workers": int(workers),
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_or_ground_truth_used": False,
            "all_mapping_images_pair_cached": True,
            "global_alignment_model_per_candidate": False,
        },
    }


def _load_resume(path: Path, *, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("format") != ARTIFACT_FORMAT
        or payload.get("config") != dict(config)
        or not isinstance(payload.get("completed"), dict)
        or not isinstance(payload.get("failed"), dict)
    ):
        raise ValueError("LoFTR global-alignment resume manifest is stale or incompatible")
    return payload


def _completed_output_is_valid(entry: Mapping[str, Any]) -> bool:
    required = (
        "query_id",
        "source",
        "source_sha256",
        "pair_cache",
        "pair_cache_sha256",
        "output",
        "output_sha256",
        "summary",
        "summary_sha256",
    )
    if any(not isinstance(entry.get(key), str) or not str(entry[key]) for key in required):
        return False
    for path_key, hash_key in (
        ("source", "source_sha256"),
        ("pair_cache", "pair_cache_sha256"),
        ("output", "output_sha256"),
        ("summary", "summary_sha256"),
    ):
        path = Path(str(entry[path_key]))
        if not path.is_file() or file_sha256_short(path) != str(entry[hash_key]):
            return False
    return True


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _export_one(
    *,
    query_id: str,
    source: Path,
    pair_cache: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    image_root: Path,
    loftr_checkpoint: Path,
    output_dir: Path,
) -> dict[str, str]:
    output = Path(output_dir) / ALIGNMENT_FILENAME
    summary = Path(output_dir) / SUMMARY_FILENAME
    result = build_frozen_loftr_global_alignment_evidence(
        appearance_artifact=Path(source),
        loftr_pair_cache=Path(pair_cache),
        maplet_support_index=Path(maplet_support_index),
        support_geometry_index=Path(support_geometry_index),
        image_root=Path(image_root),
        loftr_checkpoint=Path(loftr_checkpoint),
        output=output,
        summary_json=summary,
    )
    if str(result.get("query_id", "")) != str(query_id):
        raise RuntimeError("LoFTR global-alignment exporter changed its query identity")
    return {
        "query_id": str(query_id),
        "source": str(Path(source).resolve()),
        "source_sha256": file_sha256_short(Path(source)),
        "pair_cache": str(Path(pair_cache).resolve()),
        "pair_cache_sha256": file_sha256_short(Path(pair_cache)),
        "output": str(output.resolve()),
        "output_sha256": file_sha256_short(output),
        "summary": str(summary.resolve()),
        "summary_sha256": file_sha256_short(summary),
    }


def export_frozen_loftr_global_alignment_shards(
    *,
    source_root: Path,
    source_glob: str,
    pair_cache_root: Path,
    pair_cache_glob: str,
    maplet_support_index: Path,
    support_geometry_index: Path,
    image_root: Path,
    loftr_checkpoint: Path,
    output_root: Path,
    expected_shards: int,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    source_paths = tuple(sorted(Path(source_root).glob(str(source_glob))))
    cache_paths = tuple(sorted(Path(pair_cache_root).glob(str(pair_cache_glob))))
    if (
        len(source_paths) != int(expected_shards)
        or len(cache_paths) != int(expected_shards)
        or int(expected_shards) <= 0
        or int(workers) <= 0
    ):
        raise ValueError("LoFTR global-alignment export inputs are incomplete")
    source_by_query = _index_paths_by_query_id(source_paths, kind="source")
    cache_by_query = _index_paths_by_query_id(cache_paths, kind="cache")
    source_cache_manifest = _source_cache_manifest(
        source_by_query=source_by_query, cache_by_query=cache_by_query
    )
    config = _run_config(
        source_cache_manifest=source_cache_manifest,
        maplet_support_index=Path(maplet_support_index),
        support_geometry_index=Path(support_geometry_index),
        image_root=Path(image_root),
        loftr_checkpoint=Path(loftr_checkpoint),
        workers=int(workers),
    )
    destination = Path(output_root)
    manifest_path = destination / "export_summary.json"
    if destination.exists() and not bool(resume):
        raise FileExistsError("LoFTR global-alignment output root exists; use --resume")
    destination.mkdir(parents=True, exist_ok=True)
    if bool(resume) and not manifest_path.exists() and any(destination.iterdir()):
        raise ValueError("cannot resume LoFTR global alignment without its manifest")
    state = (
        _load_resume(manifest_path, config=config)
        if bool(resume) and manifest_path.exists()
        else {"format": ARTIFACT_FORMAT, "config": config, "completed": {}, "failed": {}}
    )
    scheduled: list[tuple[str, Path, Path, Path]] = []
    for query_id in sorted(source_by_query):
        key = source_by_query[query_id].parent.name
        completed = state["completed"].get(key)
        if isinstance(completed, Mapping) and _completed_output_is_valid(completed):
            if str(completed["query_id"]) != query_id:
                raise ValueError(f"{key}: resume query identity differs from source")
            continue
        if key in state["completed"]:
            raise ValueError(f"{key}: recorded LoFTR global-alignment output is stale")
        shard_dir = destination / key
        if shard_dir.exists():
            raise FileExistsError(f"{key}: output directory exists without a valid resume entry")
        scheduled.append((query_id, source_by_query[query_id], cache_by_query[query_id], shard_dir))
    # OpenCV otherwise launches a worker pool per RANSAC fit.  Shards are
    # independent and this outer pool gives bounded, reproducible CPU usage.
    cv2.setNumThreads(1)
    worker_count = min(int(workers), len(scheduled)) if scheduled else 1
    futures: dict[Future[dict[str, str]], tuple[str, Path]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for query_id, source, pair_cache, shard_dir in scheduled:
            futures[
                executor.submit(
                    _export_one,
                    query_id=query_id,
                    source=source,
                    pair_cache=pair_cache,
                    maplet_support_index=Path(maplet_support_index),
                    support_geometry_index=Path(support_geometry_index),
                    image_root=Path(image_root),
                    loftr_checkpoint=Path(loftr_checkpoint),
                    output_dir=shard_dir,
                )
            ] = (query_id, shard_dir)
        failures: list[tuple[str, BaseException]] = []
        for future in as_completed(futures):
            query_id, shard_dir = futures[future]
            key = shard_dir.name
            try:
                state["completed"][key] = future.result()
                state["failed"].pop(key, None)
            except Exception as error:
                state["failed"][key] = {
                    "query_id": query_id,
                    "source": str(source_by_query[query_id].resolve()),
                    "pair_cache": str(cache_by_query[query_id].resolve()),
                    "error": repr(error),
                }
                failures.append((key, error))
            _write_state(manifest_path, state)
            if not failures:
                print(
                    json.dumps(
                        {
                            "completed": len(state["completed"]),
                            "total": len(source_paths),
                            "query_id": query_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if failures:
            failed_keys = ", ".join(key for key, _error in failures)
            raise RuntimeError(
                "LoFTR global-alignment shards failed; successful sibling shards "
                f"were recorded for resume: {failed_keys}"
            ) from failures[0][1]
    if len(state["completed"]) != len(source_paths) or state["failed"]:
        raise RuntimeError("LoFTR global-alignment export did not complete every shard")
    state["complete"] = True
    _write_state(manifest_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state = export_frozen_loftr_global_alignment_shards(
        source_root=Path(args.source_root),
        source_glob=str(args.source_glob),
        pair_cache_root=Path(args.pair_cache_root),
        pair_cache_glob=str(args.pair_cache_glob),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        image_root=Path(args.image_root),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        output_root=Path(args.output_root),
        expected_shards=int(args.expected_shards),
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "complete": state.get("complete") is True,
                "completed_count": len(state["completed"]),
                "failed_count": len(state["failed"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
