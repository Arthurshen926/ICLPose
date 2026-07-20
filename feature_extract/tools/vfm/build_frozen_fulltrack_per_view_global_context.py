"""Export a target-free full-image RADIO context value for every frozen CSR edge.

This S1 diagnostic starts from the current raw full-track per-view artifact.
For each already-fixed ``(query point, candidate track, real support view)``
edge it writes two full-image RADIO-final cosine values.  It deliberately
does not aggregate support views: the downstream per-view mixture is the only
component permitted to learn their marginalization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_frozen_fulltrack_global_context import (
    _geometry_image_indices,
    _load_current_raw_per_view_source,
    fulltrack_global_context_edge_scores,
    load_radio_final_full_image_context,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_APPEARANCE_FORMAT,
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)


ARTIFACT_VERSION = "frozen_fulltrack_candidate_per_view_global_context_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-per-view-artifact", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
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


def build_frozen_fulltrack_per_view_global_context(
    *,
    source_per_view_artifact: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Copy immutable raw CSR edges and attach per-view global context only."""

    source_path = Path(source_per_view_artifact)
    geometry_path = Path(support_geometry_index)
    context_path = Path(radio_final_context_cache)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite per-view global-context outputs")
    started = time.monotonic()
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("per-view global context requires real SfM observation xy")
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
        != file_sha256_short(context_path)
    ):
        raise ValueError("raw per-view source and RADIO-final context cache differ")
    context, context_metadata = load_radio_final_full_image_context(context_path)
    scores = fulltrack_global_context_edge_scores(
        query_id=str(source["verification_query_ids"][0]),
        geometry=geometry,
        geometry_image_indices=_geometry_image_indices(geometry),
        edges=edges,
        context=context,
    )
    if scores.shape != (edges.edge_count, len(FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES)):
        raise RuntimeError("per-view global-context edge scores have an invalid shape")
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("per-view global-context edge scores are non-finite")
    offsets = _edge_offsets(edges.candidate_observation_counts)
    if (
        offsets[-1] != edges.edge_count
        or not np.array_equal(
            np.repeat(
                np.arange(edges.candidate_shape[0] * edges.candidate_shape[1]),
                np.diff(offsets),
            ),
            edges.edge_candidate_indices,
        )
    ):
        raise RuntimeError("per-view global-context CSR edge order changed")
    candidate = np.asarray(source["candidate_probabilities"], dtype=np.float32)
    if np.any((candidate > 0.0) & (edges.candidate_observation_counts <= 0)):
        raise RuntimeError("positive frozen candidate lost all support observations")
    query_id = str(source["verification_query_ids"][0])
    metadata: dict[str, Any] = {
        "format": FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_APPEARANCE_FORMAT,
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
        "support_geometry_index": str(geometry_path),
        "support_geometry_index_sha256": file_sha256_short(geometry_path),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "context_cache_sha256": {"radio_final": file_sha256_short(context_path)},
        "context_cache_format": context_metadata.get("format"),
        "context_cache_pca_fit_scope": context_metadata.get("pca_fit_scope"),
        "support_view_source": "all_real_sfm_track_observations_v1",
        "support_view_selection": "none_preserve_source_csr_order_v1",
        "support_view_count_cap": None,
        "support_view_descriptor_averaging": False,
        "per_view_edges_retained": True,
        "per_view_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
        "candidate_set": "frozen_s0_global_top20_tracks",
        "fixed_candidate_top_k": 20,
        "profiles": [
            {
                "name": "radio_final_global",
                "source": "radio_final_full_image",
                "descriptor": "global",
            },
            {
                "name": "radio_final_summary",
                "source": "radio_final_full_image",
                "descriptor": "summary",
            },
        ],
        "appearance_config": {
            "score": "l2_normalized_radio_final_full_image_cosine_v1",
            "per_view": True,
            "candidate_specific": True,
            "support_view_marginalization": "not_aggregated_export_per_view_v1",
            "soft_global_context_factor": True,
            "global_context_hard_retrieval_or_candidate_reselection": False,
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
            "raw_summary_not_calibrated_likelihood": True,
            "source_fulltrack_csr_edges_preserved": True,
            "soft_global_context_factor": True,
            "global_context_hard_retrieval_or_candidate_reselection": False,
        },
        "implementation": {
            "builder_sha256": file_sha256_short(Path(__file__)),
            "global_context_builder_sha256": file_sha256_short(
                Path(__file__).with_name("build_frozen_fulltrack_global_context.py")
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
                FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES, dtype=np.str_
            ),
            edge_candidate_offsets=offsets,
            edge_geometry_rows=np.asarray(edges.geometry_rows, dtype=np.int64),
            edge_profile_scores=scores.astype(np.float16, copy=False),
            edge_profile_valid=np.ones(scores.shape, dtype=bool),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(output_path)
    summary = {
        "stage": "build_frozen_fulltrack_per_view_global_context",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "query_id": query_id,
        "split_name": str(source["split_names"][0]),
        "row_count": int(len(source["verification_query_ids"])),
        "edge_count": int(edges.edge_count),
        "feature_count": int(scores.shape[1]),
        "protocol": {
            "fixed_global_topl": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "source_csr_edges_preserved": True,
            "all_real_sfm_track_observations": True,
            "soft_global_context_factor": True,
            "hard_image_retrieval_or_submap": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "diagnostic_only": True,
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_frozen_fulltrack_per_view_global_context(
        source_per_view_artifact=Path(args.source_per_view_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
