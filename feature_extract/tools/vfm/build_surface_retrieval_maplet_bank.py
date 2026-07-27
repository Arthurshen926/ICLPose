"""Strip a legacy construction artifact into an anchor-free retrieval bank.

The legacy bank is accepted only as an offline migration input. The output
contains no anchor, observation, image/view identity, or per-view descriptor
arrays and is the only maplet artifact accepted by the V4 refinement entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_maplets", required=True)
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument("--maximum_components", type=int, default=4)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_maplets)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite output")
    source = VfmSurfaceMapletBank.load_npz(Path(args.legacy_maplets))
    maximum_components = int(args.maximum_components)
    if maximum_components <= 0:
        raise ValueError("maximum_components must be positive")
    components: list[np.ndarray] = []
    component_weights: list[np.ndarray] = []
    offsets = [0]
    for row in range(len(source)):
        begin, end = int(source.view_offsets[row]), int(source.view_offsets[row + 1])
        observations = source.view_descriptors[begin:end]
        observation_weights = np.clip(
            source.view_quality_scores[begin:end], 1e-4, None
        )
        if observations.shape[0] == 0:
            observations = source.descriptors[row : row + 1]
            observation_weights = np.ones((1,), dtype=np.float32)
        component_count = min(maximum_components, int(observations.shape[0]))
        chosen = [int(np.argmax(observation_weights))]
        while len(chosen) < component_count:
            similarity = observations @ observations[np.asarray(chosen)].T
            distance = 1.0 - np.max(similarity, axis=1)
            distance[np.asarray(chosen)] = -1.0
            chosen.append(int(np.argmax(distance)))
        centers = observations[np.asarray(chosen)].copy()
        assignments = np.zeros((observations.shape[0],), dtype=np.int64)
        for _iteration in range(8):
            assignments = np.argmax(observations @ centers.T, axis=1)
            updated = centers.copy()
            for cluster in range(component_count):
                members = np.flatnonzero(assignments == cluster)
                if members.size == 0:
                    continue
                updated[cluster] = np.average(
                    observations[members],
                    axis=0,
                    weights=observation_weights[members],
                )
            updated /= np.maximum(
                np.linalg.norm(updated, axis=1, keepdims=True), 1e-8
            )
            centers = updated
        weights = np.asarray(
            [
                np.sum(observation_weights[assignments == cluster])
                for cluster in range(component_count)
            ],
            dtype=np.float32,
        )
        order = np.argsort(-weights, kind="mergesort")
        components.append(centers[order])
        component_weights.append(weights[order])
        offsets.append(offsets[-1] + component_count)
    descriptors = np.concatenate(components, axis=0)
    descriptor_weights = np.concatenate(component_weights, axis=0)
    feature_space = "surface_maplet_radio_final"
    if str(args.metric_mapper_checkpoint):
        metric = load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device="cpu"
        )
        descriptors = metric.project_points(descriptors)
        feature_space = "surface_metric_radio_final"
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=source.maplet_ids,
        centers=source.centers,
        normals=source.normals,
        extents=source.extents,
        descriptor_offsets=np.asarray(offsets, dtype=np.int64),
        descriptors=descriptors,
        descriptor_weights=descriptor_weights,
        quality_scores=source.quality_scores,
        descriptor_uncertainties=source.descriptor_variances,
        metadata={
            "artifact_type": "anchor_free_surface_retrieval_maplets",
            "vfm_layer": "radio_final",
            "maplet_count": len(source),
            "representation": "compact_radio_final_mixture_per_metric_region",
            "maximum_components": maximum_components,
            "feature_space": feature_space,
            "migration_source_used_offline_only": True,
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_stable_anchor_identity": False,
            "uses_point_correspondences": False,
        },
    )
    bank.save_npz(output)
    summary = {
        "stage": "build_anchor_free_surface_retrieval_maplets",
        "output_maplets": str(output),
        "maplet_count": len(bank),
        "feature_dim": int(bank.descriptors.shape[1]),
        "descriptor_component_count": int(bank.descriptors.shape[0]),
        "maximum_components": maximum_components,
        "removed_fields": [
            "anchor_ids",
            "anchor_offsets",
            "support_element_ids",
            "support_offsets",
            "view_image_ids",
            "view_descriptors",
            "view_token_xy",
            "view_grid_sizes",
            "view_quality_scores",
        ],
        "production_contract": dict(bank.metadata or {}),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
