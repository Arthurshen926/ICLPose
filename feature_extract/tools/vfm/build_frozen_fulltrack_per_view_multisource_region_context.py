"""Attach multiscale landmark-centred context to immutable raw CSR edges.

This S1 exporter starts with the current raw full-track per-view artifact.
For every already-frozen ``(query point, candidate track, real SfM support
observation)`` edge, it compares landmark-centred image context in three
separately manifested descriptor spaces:

* RADIO-final grid16, windows 7 and 11;
* RADIO-intermediate grid16, windows 7 and 11; and
* ALIKE grid32, windows 7 and 15.

Support observations are never selected, averaged, or re-enumerated here.
The downstream diagnostic is responsible for the only allowed log-sum-exp
support-view marginalization.  This exporter is target-free and does not read
query pose, pose hypotheses, retrieval results, renderings, or identities.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_frozen_fulltrack_global_context import (
    _array_sha256_short,
    _geometry_image_indices,
    _load_current_raw_per_view_source,
)
from feature_extract.tools.vfm.build_landmark_region_prototype_candidate_probe_features import (
    _cache_image_indices,
    _parse_devices,
)
from feature_extract.tools.vfm.build_multisource_landmark_region_prototype_candidate_probe_features import (
    _compute_partition,
    _load_sources,
    _source_metadata,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_APPEARANCE_FORMAT,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_multisource_region_context_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="devices receiving disjoint immutable CSR edge ranges",
    )
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _edge_offsets(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("candidate support observation counts are invalid")
    return np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(values.reshape(-1), dtype=np.int64),
        )
    )


def _edge_context_inputs(*, geometry: object, edges: object, cache_image_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Resolve fixed CSR edges to query rows and real support image samples."""

    candidate_shape = tuple(getattr(edges, "candidate_shape"))
    if len(candidate_shape) != 2 or candidate_shape[0] <= 0 or candidate_shape[1] <= 0:
        raise ValueError("full-track CSR candidate shape is invalid")
    geometry_rows = np.asarray(getattr(edges, "geometry_rows"), dtype=np.int64)
    candidate_indices = np.asarray(
        getattr(edges, "edge_candidate_indices"), dtype=np.int64
    )
    if (
        geometry_rows.ndim != 1
        or candidate_indices.shape != geometry_rows.shape
        or np.any(geometry_rows < 0)
        or np.any(geometry_rows >= len(np.asarray(getattr(geometry, "track_ids"))))
        or np.any(candidate_indices < 0)
        or np.any(candidate_indices >= candidate_shape[0] * candidate_shape[1])
    ):
        raise ValueError("full-track CSR edge inputs are invalid")
    image_indices = _geometry_image_indices(geometry)
    support_ids = np.asarray(getattr(geometry, "image_ids")).astype(str)[
        image_indices[geometry_rows]
    ]
    support_cache_rows = _cache_image_indices(
        cache_image_ids=np.asarray(cache_image_ids).astype(str),
        image_ids=support_ids,
        context="real full-track support observation",
    )
    edge_rows = candidate_indices // int(candidate_shape[1])
    return {
        "edge_rows": edge_rows.astype(np.int64, copy=False),
        "support_cache_rows": support_cache_rows.astype(np.int64, copy=False),
        "support_xy": np.asarray(getattr(geometry, "xy"), dtype=np.float32)[
            geometry_rows
        ],
    }


def _context_source_contract(sources: Sequence[object]) -> dict[str, Any]:
    payload = [_source_metadata(source) for source in sources]
    if len(payload) != 3:
        raise ValueError("multisource region context needs final/intermediate/ALIKE")
    return {
        "mode": "candidate_specific_landmark_centered_per_real_sfm_observation_v1",
        "support_coordinate_source": "sfm_observation_xy",
        "view_aggregation": "none_before_learned_logsumexp_mixture_v1",
        "feature_names": list(FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES),
        "sources": payload,
    }


def build_frozen_fulltrack_per_view_multisource_region_context(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    devices: Sequence[str],
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """Copy one immutable raw CSR shard and attach multiscale edge evidence."""

    if int(batch_size) <= 0:
        raise ValueError("multisource region-context batch size must be positive")
    selected_devices = tuple(str(value) for value in devices)
    if not selected_devices or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("multisource region-context devices must be unique and non-empty")
    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    final_path = Path(radio_final_context_cache)
    intermediate_path = Path(radio_intermediate_context_cache)
    alike_path = Path(alike_spatial_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite per-view region-context outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("multisource region context requires real SfM observation xy")
    (
        source,
        source_metadata,
        edges,
        source_maplet_counts,
        source_lineage,
    ) = _load_current_raw_per_view_source(
        source_path=source_path,
        support_geometry_index=geometry_path,
        geometry=geometry,
    )
    source_context_hashes = source_metadata.get("context_cache_sha256")
    if (
        not isinstance(source_context_hashes, dict)
        or str(source_context_hashes.get("radio_final", ""))
        != file_sha256_short(final_path)
    ):
        raise ValueError("raw per-view source and RADIO-final spatial context differ")
    sources = _load_sources(
        radio_final_context_cache=final_path,
        radio_intermediate_context_cache=intermediate_path,
        alike_spatial_context_cache=alike_path,
    )
    cache_ids = np.asarray(getattr(sources[0], "image_ids")).astype(str)
    query_cache_rows = _cache_image_indices(
        cache_image_ids=cache_ids,
        image_ids=np.asarray(source["verification_query_ids"]).astype(str),
        context="frozen query",
    )
    edge = _edge_context_inputs(
        geometry=geometry,
        edges=edges,
        cache_image_ids=cache_ids,
    )
    edge_count = int(getattr(edges, "edge_count"))
    if edge_count <= 0 or edge_count != len(edge["edge_rows"]):
        raise ValueError("multisource region context has no immutable CSR edges")
    partitions = [
        (
            edge_count * index // len(selected_devices),
            edge_count * (index + 1) // len(selected_devices),
        )
        for index in range(len(selected_devices))
    ]
    if any(end <= begin for begin, end in partitions):
        raise ValueError("more devices than immutable full-track CSR edges")
    with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
        futures = [
            executor.submit(
                _compute_partition,
                device_name=device,
                sources=sources,
                query_cache_rows=query_cache_rows,
                query_xy=np.asarray(source["verification_xy"], dtype=np.float32),
                edge_rows=edge["edge_rows"],
                edge_support_cache_rows=edge["support_cache_rows"],
                edge_support_xy=edge["support_xy"],
                begin=begin,
                end=end,
                batch_size=int(batch_size),
            )
            for device, (begin, end) in zip(selected_devices, partitions)
        ]
        computed = [future.result() for future in futures]
    scores = np.concatenate([values for values, _worker in computed], axis=0)
    if scores.shape != (
        edge_count,
        len(FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES),
    ):
        raise RuntimeError("multisource region-context edge score shape drifted")
    pooled_columns = np.asarray(
        [
            index
            for index, name in enumerate(MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES)
            if name.endswith("_pool_cosine")
        ],
        dtype=np.int64,
    )
    coverage_columns = np.asarray(
        [
            index
            for index, name in enumerate(MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES)
            if name.endswith("_common_cell_fraction")
        ],
        dtype=np.int64,
    )
    if (
        not len(pooled_columns)
        or not len(coverage_columns)
        or np.any(np.isinf(scores))
        or np.any(~np.isfinite(scores[:, pooled_columns]))
        or np.any(~np.isfinite(scores[:, coverage_columns]))
    ):
        raise RuntimeError("multisource region-context edge scores are invalid")
    valid = np.isfinite(scores)
    if np.any(~np.isfinite(scores[valid])) or np.any(np.isfinite(scores[~valid])):
        raise RuntimeError("multisource region-context missingness is invalid")
    offsets = _edge_offsets(np.asarray(edges.candidate_observation_counts, dtype=np.int64))
    if (
        offsets[-1] != edge_count
        or not np.array_equal(
            np.repeat(
                np.arange(edges.candidate_shape[0] * edges.candidate_shape[1]),
                np.diff(offsets),
            ),
            np.asarray(edges.edge_candidate_indices, dtype=np.int64),
        )
    ):
        raise RuntimeError("multisource region context changed CSR edge order")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all real support observations")
    query_id = str(source["verification_query_ids"][0])
    source_contract = _context_source_contract(sources)
    metadata: dict[str, Any] = {
        "format": FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_APPEARANCE_FORMAT,
        "version": ARTIFACT_VERSION,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "query_count": 1,
        "source_frozen_appearance_artifact": source_lineage[
            "source_frozen_appearance_artifact"
        ],
        "source_frozen_appearance_artifact_sha256": source_lineage[
            "source_frozen_appearance_artifact_sha256"
        ],
        "source_fulltrack_per_view_artifact": source_lineage[
            "source_fulltrack_per_view_artifact"
        ],
        "source_fulltrack_per_view_artifact_sha256": source_lineage[
            "source_fulltrack_per_view_artifact_sha256"
        ],
        "source_edge_candidate_offsets_sha256": source_lineage[
            "source_edge_candidate_offsets_sha256"
        ],
        "source_edge_geometry_rows_sha256": source_lineage[
            "source_edge_geometry_rows_sha256"
        ],
        "source_edge_feature_semantics": source_lineage[
            "source_edge_feature_semantics"
        ],
        "source_fulltrack_edge_contract": source_lineage["source_kind"],
        "source_candidate_tracks_sha256": _array_sha256_short(
            source["candidate_track_ids"]
        ),
        "source_candidate_probabilities_sha256": _array_sha256_short(
            source["candidate_probabilities"]
        ),
        "source_null_probabilities_sha256": _array_sha256_short(
            source["null_probabilities"]
        ),
        "source_verification_rows_sha256": _array_sha256_short(
            source["verification_source_row_indices"]
        ),
        "support_geometry_index": str(geometry_path),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "context_cache_sha256": {
            "radio_final": file_sha256_short(final_path),
            "radio_intermediate": file_sha256_short(intermediate_path),
            "alike": file_sha256_short(alike_path),
        },
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": source_contract["sources"],
        "multisource_region_context_contract": source_contract,
        "appearance_config": {
            "candidate_specific": True,
            "per_view": True,
            "feature_definition": (
                "landmark_centered_multiscale_region_cosine_per_real_sfm_observation_v1"
            ),
            "support_view_marginalization": "not_aggregated_export_per_view_v1",
            "whole_image_retrieval_or_candidate_reselection": False,
        },
        "strict_fulltrack_appearance_contract": {
            "candidate_identity_fixed": True,
            "candidate_posterior_preserved": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations_enumerated": True,
            "support_view_count_cap": None,
            "candidate_3d_projection_or_pose_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "heldout_s0_verification_rows": True,
            "raw_region_context_not_calibrated_likelihood": True,
            "source_fulltrack_csr_edges_preserved": True,
            "support_view_features_averaged_before_inference": False,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "multisource_region_helper_sha256": file_sha256_short(
                Path(__file__).with_name(
                    "build_multisource_landmark_region_prototype_candidate_probe_features.py"
                )
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            verification_query_ids=source["verification_query_ids"],
            split_names=source["split_names"],
            verification_source_row_indices=source["verification_source_row_indices"],
            verification_xy=source["verification_xy"],
            candidate_track_ids=source["candidate_track_ids"],
            candidate_probabilities=candidate,
            null_probabilities=source["null_probabilities"],
            candidate_support_observation_counts=edges.candidate_observation_counts,
            source_maplet_support_view_counts=source_maplet_counts,
            profile_names=np.asarray(
                FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES,
                dtype=np.str_,
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
            edge_profile_scores=scores.astype(np.float16, copy=False),
            edge_profile_valid=valid,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_multisource_region_context",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": edge_count,
        "feature_count": int(scores.shape[1]),
        "edge_joint_feature_coverage": float(np.mean(np.all(valid, axis=1))),
        "devices": list(selected_devices),
        "workers": [worker for _values, worker in computed],
        "elapsed_seconds": float(time.monotonic() - started),
        "protocol": {
            "fixed_global_topl": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "source_csr_edges_preserved": True,
            "all_real_sfm_track_observations": True,
            "support_view_features_averaged_before_inference": False,
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "diagnostic_only": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_per_view_multisource_region_context(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        devices=_parse_devices(str(args.devices)),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
