"""Freeze the target-free inputs for the multiscale context-attention probe.

The contract binds a pre-existing fixed top-L layout to real-image RADIO-final,
RADIO-intermediate, and ALIKE grids.  It deliberately contains no labels,
query pose, residuals, image retrieval, or candidate reselection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    CONTEXT_ATTENTION_CENTER_MASK_RADIUS,
    CONTEXT_ATTENTION_FAMILIES,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
    CONTEXT_ATTENTION_SCALES,
    build_fixed_candidate_context_runtime,
    load_context_attention_frozen_layout,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)


_PROPOSAL_OVERLAY_CANDIDATE_INPUT = "proposal_overlay_v1"
_MIXED_POINTS_CANDIDATE_INPUT = "mixed_verification_points_embedded_coarse_prior_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--alike_spatial_context_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _source_manifest(sources: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source, scale in zip(sources, CONTEXT_ATTENTION_SCALES):
        metadata = source.metadata
        output.append(
            {
                "name": scale.name,
                "path": str(source.path.resolve()),
                "sha256": file_sha256_short(source.path),
                "grid_size": int(scale.grid_size),
                "window_size": int(scale.window_size),
                "descriptor_dim": int(source.descriptor_dim),
                "format": metadata.get("format"),
                "source_image_manifest_sha256": metadata.get("source_image_manifest_sha256"),
                "radio_checkpoint_sha256": metadata.get("radio_checkpoint_sha256"),
                "alike_checkpoint_sha256": metadata.get("alike_checkpoint_sha256"),
                "pca_fit_scope": metadata.get("pca_fit_scope"),
                "intermediate_index": metadata.get("intermediate_index"),
            }
        )
    return output


def build_multiscale_context_attention_probe_contract(
    *,
    frozen_layout_features: Path,
    support_geometry_index: Path,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, Any]:
    """Write a validated, immutable source manifest for direct token probing."""

    output = Path(output)
    summary_json = Path(summary_json)
    if (output.exists() or summary_json.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite context-attention probe contract")
    layout, layout_metadata = load_context_attention_frozen_layout(
        Path(frozen_layout_features)
    )
    if bool(layout_metadata.get("whole_image_summary_or_global_used", True)):
        raise ValueError("context-attention contract must begin with local frozen candidates")
    expected_checkpoint = str(layout_metadata.get("radio_checkpoint_sha256", ""))
    sources = load_context_attention_sources(
        radio_final_context_cache=Path(radio_final_context_cache),
        radio_intermediate_context_cache=Path(radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(alike_spatial_context_cache),
        expected_radio_checkpoint=expected_checkpoint,
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        Path(support_geometry_index)
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("context-attention support geometry must use real SfM observation xy")
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.asarray(layout["query_ids"]).astype(str),
        query_xy=np.asarray(layout["xy"], dtype=np.float32),
        candidate_track_ids=np.asarray(layout["candidate_track_ids"], dtype=np.int64),
        candidate_support_image_ids=np.asarray(layout["candidate_support_image_ids"]).astype(str),
        candidate_view_valid=np.asarray(layout["candidate_view_valid"], dtype=bool),
        cache_image_ids=sources[0].image_ids,
        support_geometry=geometry,
    )
    proposals_sha256 = str(layout_metadata.get("proposals_sha256", ""))
    if not proposals_sha256:
        raise ValueError("frozen layout lacks proposal lineage")
    proposal_provenance = str(layout_metadata.get("proposal_provenance", ""))
    if proposal_provenance == "mixed_verification_points_full_global_faiss_top_l_v1":
        candidate_input_kind = _MIXED_POINTS_CANDIDATE_INPUT
        candidate_input_path = str(layout_metadata.get("verification_points", ""))
        candidate_input_sha256 = str(layout_metadata.get("verification_points_sha256", ""))
        if not candidate_input_path or not candidate_input_sha256:
            raise ValueError("mixed frozen layout lacks verification-point lineage")
        if candidate_input_sha256 != proposals_sha256:
            raise ValueError("mixed frozen layout candidate lineage is inconsistent")
    else:
        candidate_input_kind = _PROPOSAL_OVERLAY_CANDIDATE_INPUT
        candidate_input_path = ""
        candidate_input_sha256 = proposals_sha256
    source_manifest = str(sources[0].metadata.get("source_image_manifest_sha256", ""))
    if not source_manifest:
        raise ValueError("context-attention sources lack an image manifest")
    runtime_digest = hashlib.sha256()
    for value in (
        runtime.query_image_indices,
        runtime.support_image_indices,
        runtime.support_xy,
        runtime.view_valid,
    ):
        runtime_digest.update(np.ascontiguousarray(value).view(np.uint8))
    payload: dict[str, Any] = {
        "format": CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "whole_image_summary_or_global_used": False,
        "render": False,
        "candidate_set": "frozen_global_top20_tracks",
        "candidate_support_views": "frozen_layout_support_views_only",
        "query_support_alignment": "frozen_query_xy_to_fixed_sfm_support_observation_xy_v1",
        "context_only_center_mask": {
            "shape": "3x3",
            "radius": int(CONTEXT_ATTENTION_CENTER_MASK_RADIUS),
            "reason": "exclude_anchor_descriptor_from_candidate_specific_context_control",
        },
        "families": list(CONTEXT_ATTENTION_FAMILIES),
        "source_scales": _source_manifest(sources),
        "source_image_manifest_sha256": source_manifest,
        "frozen_layout_features": str(Path(frozen_layout_features).resolve()),
        "frozen_layout_features_sha256": file_sha256_short(frozen_layout_features),
        "full_frozen_source_rows_sha256": _array_sha256_short(
            np.asarray(layout["source_row_indices"], dtype=np.int64)
        ),
        "frozen_candidate_tracks_sha256": _array_sha256_short(
            np.asarray(layout["candidate_track_ids"], dtype=np.int64)
        ),
        "frozen_support_view_mask_sha256": _array_sha256_short(
            np.asarray(layout["candidate_view_valid"], dtype=bool)
        ),
        # ``proposals_sha256`` remains for compatibility with earlier probe
        # contracts.  The explicit candidate-input fields prevent a mixed
        # verification-point layout from being silently interpreted as an old
        # detector-proposal table during fitting.
        "proposals_sha256": proposals_sha256,
        "candidate_input_kind": candidate_input_kind,
        "candidate_input_lineage_path": candidate_input_path,
        "candidate_input_lineage_sha256": candidate_input_sha256,
        "support_geometry_index": str(Path(support_geometry_index).resolve()),
        "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
        "support_geometry_coordinate_source": geometry_metadata.get("coordinate_source"),
        "radio_checkpoint_sha256": sources[0].metadata.get("radio_checkpoint_sha256"),
        "query_row_count": int(len(layout["source_row_indices"])),
        "candidate_top_k": int(np.asarray(layout["candidate_track_ids"]).shape[1]),
        "support_view_count": int(np.asarray(layout["candidate_view_valid"]).shape[2]),
        "valid_candidate_view_count": int(np.sum(runtime.view_valid)),
        "runtime_indices_sha256": runtime_digest.hexdigest()[:16],
        "is_complete_frozen_layout": True,
        "diagnostic_max_queries": 0,
        "diagnostic_max_rows": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "freeze_multiscale_context_attention_probe_contract",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "format": CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
        "row_count": int(payload["query_row_count"]),
        "candidate_top_k": int(payload["candidate_top_k"]),
        "support_view_count": int(payload["support_view_count"]),
        "valid_candidate_view_count": int(payload["valid_candidate_view_count"]),
        "scales": [
            {"name": scale.name, "grid_size": scale.grid_size, "window_size": scale.window_size}
            for scale in CONTEXT_ATTENTION_SCALES
        ],
        "protocol": {
            "image_retrieval_or_submap_used": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "context_only_masks_anchor": True,
            "view_features_averaged_before_inference": False,
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_multiscale_context_attention_probe_contract(
        frozen_layout_features=Path(args.frozen_layout_features),
        support_geometry_index=Path(args.support_geometry_index),
        radio_final_context_cache=Path(args.radio_final_context_cache),
        radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
        alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
