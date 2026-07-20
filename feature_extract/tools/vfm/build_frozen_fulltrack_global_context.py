"""Export all-observation soft global RADIO context for frozen candidates.

This is a deliberately narrow diagnostic.  For every fixed top-20 landmark
candidate, it marginalizes full-image RADIO-final context cosine over *all*
real SfM observations of that track.  It neither retrieves support images nor
changes a candidate, posterior, or query anchor.  The result is raw,
target-free separability evidence only; it is not a calibrated likelihood or
pose scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_frozen_fulltrack_candidate_appearance import (
    _geometry_image_indices,
    _source_cache_lineage,
    load_frozen_source_appearance,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.full_track_support_view_probe import (
    FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT,
    FULL_TRACK_VIEW_STATISTIC_NAMES,
    FullTrackCandidateEdges,
    aggregate_full_track_view_scores,
    build_full_track_candidate_edges,
    build_track_observation_lookup,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS,
    _load_artifact as _load_frozen_fulltrack_per_view_artifact,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)


ARTIFACT_FORMAT = FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT
ARTIFACT_VERSION = "frozen_fulltrack_global_context_summary_v1"
RADIO_FINAL_CONTEXT_FORMAT = "radio_final_context_pca_v1"
PROFILE_NAMES = ("radio_final_global", "radio_final_summary")
_STATISTIC_SUFFIXES = {
    "uniform_mean_ncc": "uniform_mean_cosine",
    "uniform_logmeanexp_tau0p05_ncc": "uniform_logmeanexp_tau0p05_cosine",
    "uniform_top4_mean_ncc": "uniform_top4_mean_cosine",
    "uniform_max_ncc": "uniform_max_cosine",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-appearance-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    payload = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(payload, dict):
        raise ValueError(f"{context} metadata is not an object")
    return payload


def _normalize_rows(values: np.ndarray, *, context: str) -> np.ndarray:
    descriptors = np.asarray(values, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0 or descriptors.shape[1] == 0:
        raise ValueError(f"{context} descriptors are invalid")
    if np.any(~np.isfinite(descriptors)):
        raise ValueError(f"{context} descriptors are non-finite")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError(f"{context} descriptors contain a zero-norm row")
    return descriptors / norms


def load_radio_final_full_image_context(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load two scoped full-image RADIO descriptors without any image search."""

    required = {"image_ids", "global_descriptors", "summary_descriptors"}
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"RADIO-final context cache lacks {sorted(missing)}")
        arrays = {name: np.asarray(data[name]).copy() for name in required}
        metadata = _metadata(data, context="RADIO-final context cache")
    if (
        metadata.get("format") != RADIO_FINAL_CONTEXT_FORMAT
        or metadata.get("pose_or_ground_truth_used") is not False
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or metadata.get("pca_fit_scope")
        not in {
            "mapping_train_images_only",
            "mapping_support_images_excluding_all_query_splits_v1",
        }
    ):
        raise ValueError("RADIO-final full-image context cache violates the probe contract")
    image_ids = np.asarray(arrays["image_ids"]).astype(str).reshape(-1)
    if len(image_ids) == 0 or len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("RADIO-final context image identities are invalid")
    global_descriptors = _normalize_rows(
        arrays["global_descriptors"], context="RADIO-final global"
    )
    summary_descriptors = _normalize_rows(
        arrays["summary_descriptors"], context="RADIO-final summary"
    )
    if (
        global_descriptors.shape[0] != len(image_ids)
        or summary_descriptors.shape[0] != len(image_ids)
    ):
        raise ValueError("RADIO-final full-image context rows do not align with image ids")
    return {
        "image_ids": image_ids,
        "global_descriptors": global_descriptors,
        "summary_descriptors": summary_descriptors,
    }, metadata


def fulltrack_global_context_edge_scores(
    *,
    query_id: str,
    geometry: SupportObservationGeometryIndex,
    geometry_image_indices: np.ndarray,
    edges: FullTrackCandidateEdges,
    context: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Score each fixed all-observation edge with query/support image cosine.

    The returned columns correspond to ``PROFILE_NAMES``.  Full-image context
    is constant across points in one query image, but remains candidate/view
    specific because every edge owns a fixed physical track observation.
    """

    image_ids = np.asarray(context["image_ids"]).astype(str).reshape(-1)
    lookup = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
    query_index = lookup.get(str(query_id))
    if query_index is None:
        raise ValueError(f"query image lacks RADIO-final full-image context: {query_id}")
    image_rows = np.asarray(geometry_image_indices, dtype=np.int64)
    if image_rows.shape != np.asarray(geometry.track_ids).shape:
        raise ValueError("support geometry image indirection is invalid")
    support_ids = np.asarray(geometry.image_ids).astype(str)[
        image_rows[np.asarray(edges.geometry_rows, dtype=np.int64)]
    ]
    support_indices = np.asarray(
        [lookup.get(str(image_id), -1) for image_id in support_ids.tolist()],
        dtype=np.int64,
    )
    if np.any(support_indices < 0):
        missing = str(support_ids[np.flatnonzero(support_indices < 0)[0]])
        raise ValueError(
            f"support image lacks RADIO-final full-image context: {missing}"
        )
    scores: list[np.ndarray] = []
    for profile, field in zip(
        PROFILE_NAMES, ("global_descriptors", "summary_descriptors")
    ):
        descriptors = np.asarray(context[field], dtype=np.float32)
        if descriptors.shape[0] != len(image_ids):
            raise ValueError(f"{profile} descriptors do not align with image ids")
        values = descriptors[support_indices] @ descriptors[int(query_index)]
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"{profile} edge cosine is non-finite")
        scores.append(values.astype(np.float32, copy=False))
    return np.stack(scores, axis=1).astype(np.float32, copy=False)


def _feature_arrays(summary: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    statistics = getattr(summary, "statistics")
    profile_count = len(PROFILE_NAMES)
    values: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    names: list[str] = []
    usable_counts = np.asarray(getattr(summary, "usable_counts"), dtype=np.int64)
    if usable_counts.ndim != 3 or usable_counts.shape[2] != profile_count:
        raise ValueError("full-track global context usable counts are invalid")
    for profile_index, profile_name in enumerate(PROFILE_NAMES):
        profile_valid = usable_counts[..., profile_index] > 0
        for statistic in FULL_TRACK_VIEW_STATISTIC_NAMES:
            field = np.asarray(statistics[statistic], dtype=np.float32)[
                ..., profile_index
            ]
            values.append(field)
            valid.append(profile_valid)
            names.append(f"{profile_name}__{_STATISTIC_SUFFIXES[statistic]}")
    feature_values = np.stack(values, axis=2).astype(np.float32, copy=False)
    feature_valid = np.stack(valid, axis=2).astype(bool, copy=False)
    if np.any(~np.isfinite(feature_values[feature_valid])) or np.any(
        np.isfinite(feature_values[~feature_valid])
    ):
        raise RuntimeError("full-track global context feature missingness is invalid")
    return feature_values, feature_valid, np.asarray(names, dtype=np.str_)


def _load_current_raw_per_view_source(
    *,
    source_path: Path,
    support_geometry_index: Path,
    geometry: SupportObservationGeometryIndex,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    FullTrackCandidateEdges,
    np.ndarray,
    dict[str, Any],
]:
    """Load the current raw CSR handoff without re-enumerating support views.

    The current main line has already frozen the all-observation candidate
    edge order in a raw per-view artifact.  Rebuilding that order from a track
    lookup would make a global-context probe depend on a second enumeration
    implementation.  This loader instead validates and reuses the exact CSR
    edges, which is the only admissible path for a current full-track input.
    """

    arrays, metadata = _load_frozen_fulltrack_per_view_artifact(Path(source_path))
    if (
        metadata.get("per_view_edge_feature_semantics")
        != FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS
    ):
        raise ValueError("global-context source must use raw-NCC CSR support edges")
    if str(metadata.get("support_geometry_index_sha256", "")) != file_sha256_short(
        Path(support_geometry_index)
    ):
        raise ValueError("raw per-view source and support geometry index differ")

    # ``_load_artifact`` verifies the protocol header.  Read the one additional
    # diagnostic field required by summary artifacts and retain no raw visual
    # score in this global-context path.
    with np.load(Path(source_path), allow_pickle=False) as payload:
        if "source_maplet_support_view_counts" not in payload.files:
            raise ValueError("raw per-view source lacks maplet support-count lineage")
        maplet_counts = np.asarray(
            payload["source_maplet_support_view_counts"], dtype=np.int64
        ).copy()

    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    counts = np.asarray(arrays["candidate_support_observation_counts"], dtype=np.int64)
    offsets = np.asarray(arrays["edge_candidate_offsets"], dtype=np.int64).reshape(-1)
    geometry_rows = np.asarray(arrays["edge_geometry_rows"], dtype=np.int64).reshape(-1)
    if (
        maplet_counts.shape != counts.shape
        or np.any(maplet_counts < 0)
        or np.any(maplet_counts > counts)
        or offsets.shape != (tracks.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(geometry_rows)
        or not np.array_equal(np.diff(offsets), counts.reshape(-1))
        or np.any(geometry_rows < 0)
        or np.any(geometry_rows >= len(geometry.track_ids))
    ):
        raise ValueError("raw per-view source CSR/maplet counts are invalid")
    edge_candidates = np.repeat(
        np.arange(tracks.size, dtype=np.int64), np.diff(offsets)
    )
    if (
        edge_candidates.shape != geometry_rows.shape
        or not np.array_equal(
            tracks.reshape(-1)[edge_candidates],
            np.asarray(geometry.track_ids, dtype=np.int64)[geometry_rows],
        )
        or np.any((candidate > 0.0) & (counts <= 0))
    ):
        raise ValueError("raw per-view CSR edges do not match frozen candidate tracks")
    source_s0_path = Path(str(metadata.get("source_frozen_appearance_artifact", "")))
    if not source_s0_path.is_absolute():
        source_s0_path = Path.cwd() / source_s0_path
    source_s0_hash = str(metadata.get("source_frozen_appearance_artifact_sha256", ""))
    if (
        not source_s0_hash
        or not source_s0_path.is_file()
        or file_sha256_short(source_s0_path) != source_s0_hash
    ):
        raise ValueError("raw per-view source frozen S0 lineage is stale")
    return (
        {
            "verification_query_ids": np.asarray(
                arrays["verification_query_ids"]
            ).astype(str),
            "split_names": np.asarray(arrays["split_names"]).astype(str),
            "verification_source_row_indices": np.asarray(
                arrays["verification_source_row_indices"], dtype=np.int64
            ),
            "verification_xy": np.asarray(arrays["verification_xy"], dtype=np.float32),
            "candidate_track_ids": tracks,
            "candidate_probabilities": candidate,
            "null_probabilities": np.asarray(arrays["null_probabilities"], dtype=np.float32),
        },
        metadata,
        FullTrackCandidateEdges(
            candidate_shape=tuple(tracks.shape),
            edge_candidate_indices=edge_candidates,
            geometry_rows=geometry_rows,
            candidate_observation_counts=counts,
        ),
        maplet_counts,
        {
            "source_kind": "raw_fulltrack_per_view_csr_v1",
            "source_fulltrack_per_view_artifact": str(Path(source_path)),
            "source_fulltrack_per_view_artifact_sha256": file_sha256_short(
                Path(source_path)
            ),
            "source_edge_candidate_offsets_sha256": _array_sha256_short(offsets),
            "source_edge_geometry_rows_sha256": _array_sha256_short(geometry_rows),
            "source_edge_feature_semantics": metadata.get(
                "per_view_edge_feature_semantics"
            ),
            "source_frozen_appearance_artifact": str(source_s0_path),
            "source_frozen_appearance_artifact_sha256": source_s0_hash,
        },
    )


def build_frozen_fulltrack_global_context(
    *,
    source_appearance_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Export one query's immutable full-track full-image-context summary."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite full-track global-context outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("full-track global context requires real SfM observation xy")
    source_path = Path(source_appearance_artifact)
    with np.load(source_path, allow_pickle=False) as source_payload:
        source_header = _metadata(source_payload, context="global-context source")
    if source_header.get("format") == "frozen_fulltrack_candidate_per_view_appearance_v1":
        (
            source,
            source_metadata,
            edges,
            source_maplet_counts,
            source_lineage,
        ) = _load_current_raw_per_view_source(
            source_path=source_path,
            support_geometry_index=Path(support_geometry_index),
            geometry=geometry,
        )
        source_context_hashes = source_metadata.get("context_cache_sha256")
        if (
            not isinstance(source_context_hashes, Mapping)
            or str(source_context_hashes.get("radio_final", ""))
            != file_sha256_short(Path(radio_final_context_cache))
        ):
            raise ValueError(
                "raw per-view source and RADIO-final global-context cache differ"
            )
    else:
        source, source_metadata = load_frozen_source_appearance(source_path)
        _source_cache_lineage(
            source_metadata,
            cache_name="radio_final_context_cache",
            cache_path=Path(radio_final_context_cache),
        )
        lookup = build_track_observation_lookup(geometry)
        edges = build_full_track_candidate_edges(
            candidate_track_ids=source["candidate_track_ids"],
            candidate_probabilities=source["candidate_probabilities"],
            lookup=lookup,
        )
        source_maplet_counts = np.sum(
            np.asarray(source["candidate_view_weights"], dtype=np.float32) > 0.0,
            axis=2,
            dtype=np.int64,
        )
        source_lineage = {
            "source_kind": "legacy_s0_maplet_weights_v1",
            "source_fulltrack_per_view_artifact": None,
            "source_fulltrack_per_view_artifact_sha256": None,
            "source_edge_candidate_offsets_sha256": None,
            "source_edge_geometry_rows_sha256": None,
            "source_edge_feature_semantics": None,
            "source_frozen_appearance_artifact": str(source_path),
            "source_frozen_appearance_artifact_sha256": file_sha256_short(source_path),
        }
    context, context_metadata = load_radio_final_full_image_context(
        Path(radio_final_context_cache)
    )
    geometry_image_indices = _geometry_image_indices(geometry)
    edge_scores = fulltrack_global_context_edge_scores(
        query_id=str(source["verification_query_ids"][0]),
        geometry=geometry,
        geometry_image_indices=geometry_image_indices,
        edges=edges,
        context=context,
    )
    full_summary = aggregate_full_track_view_scores(
        edges=edges,
        scores=edge_scores,
        usable=np.ones(edge_scores.shape, dtype=bool),
    )
    feature_values, feature_valid, feature_names = _feature_arrays(full_summary)
    if np.any(
        (np.asarray(source["candidate_probabilities"], dtype=np.float32) > 0.0)
        & (edges.candidate_observation_counts <= 0)
    ):
        raise RuntimeError("positive frozen candidate lost all support observations")
    query_id = str(source["verification_query_ids"][0])
    strict_contract = {
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
        "raw_summary_not_calibrated_likelihood": True,
        "soft_global_context_factor": True,
        "global_context_hard_retrieval_or_candidate_reselection": False,
        "source_fulltrack_csr_edges_preserved": (
            source_lineage["source_kind"] == "raw_fulltrack_per_view_csr_v1"
        ),
    }
    metadata: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
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
        "source_strict_frozen_appearance_contract": source_metadata.get(
            "strict_frozen_appearance_contract"
        ),
        "support_geometry_index": str(Path(support_geometry_index)),
        "support_geometry_index_sha256": file_sha256_short(Path(support_geometry_index)),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "context_cache": str(Path(radio_final_context_cache)),
        "context_cache_sha256": file_sha256_short(Path(radio_final_context_cache)),
        "context_cache_format": context_metadata.get("format"),
        "context_cache_pca_fit_scope": context_metadata.get("pca_fit_scope"),
        "context_cache_source_manifest_sha256": context_metadata.get(
            "source_image_manifest_sha256"
        ),
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_enumerate_all_track_observations_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "support_view_marginalization": "uncalibrated_uniform_all_observation_summary_v1",
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": [
            {
                "name": name,
                "source": "radio_final_full_image",
                "descriptor": descriptor,
                "feature_dim": int(np.asarray(context[field]).shape[1]),
            }
            for name, descriptor, field in zip(
                PROFILE_NAMES,
                ("global", "summary"),
                ("global_descriptors", "summary_descriptors"),
            )
        ],
        "summary_statistics": [
            _STATISTIC_SUFFIXES[name] for name in FULL_TRACK_VIEW_STATISTIC_NAMES
        ],
        "appearance_config": {
            "score": "l2_normalized_radio_final_full_image_cosine_v1",
            "per_view": True,
            "candidate_specific": True,
            "all_track_observations": True,
            "calibration_or_fusion_fitted": False,
            "pose_scoring_performed": False,
        },
        "strict_fulltrack_appearance_contract": strict_contract,
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "fulltrack_support_module_sha256": file_sha256_short(
                Path(__file__).parents[2]
                / "vfm"
                / "localization"
                / "full_track_support_view_probe.py"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        verification_query_ids=source["verification_query_ids"],
        split_names=source["split_names"],
        verification_source_row_indices=source["verification_source_row_indices"],
        verification_xy=source["verification_xy"],
        candidate_track_ids=source["candidate_track_ids"],
        candidate_probabilities=source["candidate_probabilities"],
        null_probabilities=source["null_probabilities"],
        candidate_support_observation_counts=edges.candidate_observation_counts,
        source_maplet_support_view_counts=source_maplet_counts,
        candidate_summary_features=feature_values,
        candidate_summary_feature_valid=feature_valid,
        feature_names=feature_names,
        profile_names=np.asarray(PROFILE_NAMES, dtype=np.str_),
        candidate_profile_usable_counts=np.asarray(
            full_summary.usable_counts, dtype=np.int64
        ),
        candidate_profile_usable_fractions=np.asarray(
            full_summary.usable_fractions, dtype=np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "build_frozen_fulltrack_global_context",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "edge_count": int(edges.edge_count),
        "candidate_entries_expanded_beyond_maplet": int(
            np.count_nonzero(
                edges.candidate_observation_counts > source_maplet_counts
            )
        ),
        "protocol": {
            "fixed_global_topl": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "all_real_sfm_track_observations": True,
            "soft_global_context_factor": True,
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "diagnostic_only": True,
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_global_context(
        source_appearance_artifact=Path(args.source_appearance_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
