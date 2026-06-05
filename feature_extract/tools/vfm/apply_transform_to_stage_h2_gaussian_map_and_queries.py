"""Apply a feature compression transform to Stage H2 Gaussian anchors and queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import _dir_bytes, _write_compressed_query_manifest
from feature_extract.vfm.feature_compression import FeatureCompressionTransform
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap
from feature_extract.vfm.tokens import TokenBankManifest


def _project_anchor_map(anchor_map: SemiDenseAnchorMap, transform: FeatureCompressionTransform) -> SemiDenseAnchorMap:
    features = transform.apply_rows(anchor_map.features)
    metadata = dict(anchor_map.metadata or {})
    metadata.update(
        {
            "stage": "stage_h2_feature_transform_projected_anchor_map",
            "source_feature_dim": int(anchor_map.feature_dim),
            "output_feature_dim": int(transform.output_dim),
            "transform_method": str(transform.method),
        }
    )
    return SemiDenseAnchorMap(
        anchor_ids=anchor_map.anchor_ids,
        xyz=anchor_map.xyz,
        features=np.asarray(features, dtype=np.float32),
        source_types=anchor_map.source_types,
        source_track_ids=anchor_map.source_track_ids,
        source_gaussian_indices=anchor_map.source_gaussian_indices,
        support_counts=anchor_map.support_counts,
        mean_distances=anchor_map.mean_distances,
        feature_variances=anchor_map.feature_variances,
        observation_counts=anchor_map.observation_counts,
        visibility_counts=anchor_map.visibility_counts,
        quality_scores=anchor_map.quality_scores,
        opacity=anchor_map.opacity,
        scale=anchor_map.scale,
        observation_image_ids=anchor_map.observation_image_ids,
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply a transform to Stage H2 Gaussian anchors and query tokens")
    parser.add_argument("--input_anchor_npz", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--transform_npz", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--output_layer_name", default="")
    parser.add_argument("--output_anchor_npz", required=True)
    parser.add_argument("--output_query_dir", required=True)
    parser.add_argument("--output_query_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_tokens", type=int, default=65536)
    args = parser.parse_args(argv)

    transform = FeatureCompressionTransform.from_npz(Path(args.transform_npz))
    anchor_map = SemiDenseAnchorMap.load_npz(Path(args.input_anchor_npz))
    projected = _project_anchor_map(anchor_map, transform)
    projected.save_npz(Path(args.output_anchor_npz))

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    output_layer_name = args.output_layer_name or args.layer_name
    compressed_manifest, query_count = _write_compressed_query_manifest(
        manifest,
        transform,
        layer_name=args.layer_name,
        output_query_dir=Path(args.output_query_dir),
        output_layer_name=output_layer_name,
        device=args.device,
        batch_tokens=int(args.batch_tokens),
    )
    compressed_manifest.to_json(Path(args.output_query_manifest))
    summary = {
        "stage": "stage_h2_apply_feature_transform_to_gaussian_map_and_queries",
        "input_anchor_count": int(len(anchor_map)),
        "input_feature_dim": int(anchor_map.feature_dim),
        "output_anchor_count": int(len(projected)),
        "output_feature_dim": int(projected.feature_dim),
        "query_record_count": int(query_count),
        "transform": {
            "path": args.transform_npz,
            "method": str(transform.method),
            "input_dim": int(transform.input_dim),
            "output_dim": int(transform.output_dim),
        },
        "storage_bytes": {
            "anchor_map": _dir_bytes(Path(args.output_anchor_npz)),
            "query_tokens": _dir_bytes(Path(args.output_query_dir)),
        },
        "outputs": {
            "anchor_map": args.output_anchor_npz,
            "query_manifest": args.output_query_manifest,
            "query_dir": args.output_query_dir,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
