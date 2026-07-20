"""Reliably export every frozen full-track shard for sparse-maplet v2.

The launcher is deliberately sequential because each shard uses both GPUs for
disjoint CSR edge ranges.  It is resumable only when the immutable source
manifest, cache manifest, and per-shard output hashes still agree; otherwise
it refuses to mix stale descriptors with a current run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.tools.vfm.build_frozen_fulltrack_per_view_sparse_maplet_transport import (
    build_frozen_fulltrack_per_view_sparse_maplet_transport,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _parse_devices,
)
from feature_extract.vfm.artifacts import file_sha256_short


ARTIFACT_FORMAT = "frozen_fulltrack_sparse_maplet_transport_export_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--source-glob",
        default="s937_s1_fulltrack_perview_*/frozen_fulltrack_candidate_per_view_appearance_v1.npz",
    )
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-pca256-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--neighbor-topology-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--expected-shards", type=int, default=84)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _source_manifest(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": file_sha256_short(path)}
        for path in paths
    ]


def _run_config(
    *,
    source_manifest: Sequence[dict[str, str]],
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    neighbor_topology_cache: Path,
    devices: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    return {
        "format": ARTIFACT_FORMAT,
        "version": 1,
        "source_manifest": list(source_manifest),
        "support_geometry_index": str(Path(support_geometry_index).resolve()),
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "radio_final_context_cache": str(Path(radio_final_context_cache).resolve()),
        "radio_final_context_cache_sha256": file_sha256_short(
            Path(radio_final_context_cache)
        ),
        "radio_intermediate_pca256_context_cache": str(
            Path(radio_intermediate_pca256_context_cache).resolve()
        ),
        "radio_intermediate_pca256_context_cache_sha256": file_sha256_short(
            Path(radio_intermediate_pca256_context_cache)
        ),
        "alike_spatial_context_cache": str(Path(alike_spatial_context_cache).resolve()),
        "alike_spatial_context_cache_sha256": file_sha256_short(
            Path(alike_spatial_context_cache)
        ),
        "neighbor_topology_cache": str(Path(neighbor_topology_cache).resolve()),
        "neighbor_topology_cache_sha256": file_sha256_short(Path(neighbor_topology_cache)),
        "devices": [str(value) for value in devices],
        "batch_size": int(batch_size),
        "protocol": {
            "target_free": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_or_ground_truth_used": False,
            "matched_topology_control": True,
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
        raise ValueError("sparse-maplet resume manifest is stale or incompatible")
    return payload


def _completed_output_is_valid(entry: Mapping[str, Any]) -> bool:
    required = (
        "visual_output",
        "visual_output_sha256",
        "control_output",
        "control_output_sha256",
        "summary",
        "summary_sha256",
        "control_summary",
        "control_summary_sha256",
    )
    if any(not isinstance(entry.get(key), str) or not str(entry[key]) for key in required):
        return False
    for path_key, hash_key in (
        ("visual_output", "visual_output_sha256"),
        ("control_output", "control_output_sha256"),
        ("summary", "summary_sha256"),
        ("control_summary", "control_summary_sha256"),
    ):
        path = Path(str(entry[path_key]))
        if not path.is_file() or file_sha256_short(path) != str(entry[hash_key]):
            return False
    return True


def export_sparse_maplet_transport_shards(
    *,
    source_root: Path,
    source_glob: str,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_pca256_context_cache: Path,
    alike_spatial_context_cache: Path,
    neighbor_topology_cache: Path,
    output_root: Path,
    devices: Sequence[str],
    batch_size: int,
    expected_shards: int,
    resume: bool,
) -> dict[str, Any]:
    root = Path(source_root)
    paths = tuple(sorted(root.glob(str(source_glob))))
    if (
        not paths
        or len(paths) != int(expected_shards)
        or len(set(paths)) != len(paths)
        or int(batch_size) <= 0
        or not devices
    ):
        raise ValueError("sparse-maplet shard export inputs are invalid or incomplete")
    source_manifest = _source_manifest(paths)
    config = _run_config(
        source_manifest=source_manifest,
        support_geometry_index=Path(support_geometry_index),
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        neighbor_topology_cache=Path(neighbor_topology_cache),
        devices=devices,
        batch_size=int(batch_size),
    )
    destination = Path(output_root)
    manifest_path = destination / "export_summary.json"
    if destination.exists() and not bool(resume):
        raise FileExistsError("sparse-maplet output root exists; use --resume only for same manifest")
    destination.mkdir(parents=True, exist_ok=True)
    state = (
        _load_resume(manifest_path, config=config)
        if bool(resume) and manifest_path.exists()
        else {"format": ARTIFACT_FORMAT, "config": config, "completed": {}, "failed": {}}
    )
    if bool(resume) and not manifest_path.exists() and any(destination.iterdir()):
        raise ValueError("cannot resume sparse-maplet export without its manifest")
    for source_path in paths:
        key = source_path.parent.name
        completed = state["completed"].get(key)
        if isinstance(completed, dict) and _completed_output_is_valid(completed):
            continue
        if key in state["completed"]:
            raise ValueError(f"{key}: recorded sparse-maplet output is stale")
        shard_dir = destination / key
        if shard_dir.exists():
            raise FileExistsError(f"{key}: output directory exists without a valid resume entry")
        try:
            result = build_frozen_fulltrack_per_view_sparse_maplet_transport(
                source_per_view_artifact=source_path,
                support_geometry_index=Path(support_geometry_index),
                radio_final_context_cache=Path(radio_final_context_cache),
                radio_intermediate_pca256_context_cache=Path(
                    radio_intermediate_pca256_context_cache
                ),
                alike_spatial_context_cache=Path(alike_spatial_context_cache),
                output=shard_dir
                / "frozen_fulltrack_candidate_per_view_sparse_maplet_transport_v2.npz",
                summary_json=shard_dir / "summary.json",
                topology_control_output=shard_dir
                / "frozen_fulltrack_candidate_per_view_sparse_maplet_topology_control_v2.npz",
                topology_control_summary_json=shard_dir / "topology_control_summary.json",
                neighbor_topology_cache=Path(neighbor_topology_cache),
                devices=devices,
                batch_size=int(batch_size),
                force=False,
            )
        except Exception as exc:
            state["failed"][key] = {"source": str(source_path.resolve()), "error": repr(exc)}
            manifest_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            raise
        visual = Path(str(result["output"]))
        control = Path(str(result["topology_control_output"]))
        summary = shard_dir / "summary.json"
        control_summary = shard_dir / "topology_control_summary.json"
        state["completed"][key] = {
            "source": str(source_path.resolve()),
            "visual_output": str(visual.resolve()),
            "visual_output_sha256": file_sha256_short(visual),
            "control_output": str(control.resolve()),
            "control_output_sha256": file_sha256_short(control),
            "summary": str(summary.resolve()),
            "summary_sha256": file_sha256_short(summary),
            "control_summary": str(control_summary.resolve()),
            "control_summary_sha256": file_sha256_short(control_summary),
        }
        state["failed"].pop(key, None)
        manifest_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "completed": len(state["completed"]),
                    "total": len(paths),
                    "query_shard": key,
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(state["completed"]) != len(paths) or state["failed"]:
        raise RuntimeError("sparse-maplet export did not complete every immutable shard")
    state["complete"] = True
    manifest_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state = export_sparse_maplet_transport_shards(
        source_root=Path(args.source_root),
        source_glob=str(args.source_glob),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_pca256_context_cache=Path(
            args.radio_intermediate_pca256_context_cache
        ),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        neighbor_topology_cache=Path(args.neighbor_topology_cache),
        output_root=Path(args.output_root),
        devices=_parse_devices(args.devices),
        batch_size=int(args.batch_size),
        expected_shards=int(args.expected_shards),
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
