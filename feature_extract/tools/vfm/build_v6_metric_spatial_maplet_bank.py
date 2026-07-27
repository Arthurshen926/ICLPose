"""Package a lineaged V6 metric atlas as a maplet-internal surface bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    package_surface_feature_atlas,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric_atlas", required=True)
    parser.add_argument("--query_metric_encoder", required=True)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--spatial_stride", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output_maplets)
    summary_path = Path(args.summary_json)
    if (
        output_path.exists() or summary_path.exists()
    ) and not bool(args.force):
        raise FileExistsError("refusing to overwrite metric spatial bank")
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.metric_atlas))
    _model, query_metadata = load_v6_metric_encoder(
        Path(args.query_metric_encoder), device="cpu"
    )
    atlas_metadata = dict(atlas.metadata or {})
    map_encoder_sha256 = str(
        atlas_metadata.get("metric_encoder_sha256", "")
    )
    compatible_map_sha256 = str(
        query_metadata.get("compatible_map_encoder_sha256", "")
    )
    if not map_encoder_sha256:
        raise ValueError("metric atlas has no encoder lineage")
    if compatible_map_sha256 != map_encoder_sha256:
        raise ValueError(
            "query metric encoder is not trained for this map encoder"
        )
    lineage_keys = (
        "geometry_source_sha256",
        "clean_geometry_source_sha256",
        "clean_source_index_sha256",
    )
    missing_lineage = [
        key for key in lineage_keys if not str(atlas_metadata.get(key, ""))
    ]
    if missing_lineage or not bool(
        atlas_metadata.get("contributor_geometry_lineage_verified", False)
    ):
        raise ValueError(
            "metric atlas lacks verified clean-2DGS lineage: "
            f"{missing_lineage}"
        )
    query_encoder_sha256 = _sha256(Path(args.query_metric_encoder))
    level = str(atlas_metadata.get("metric_feature_level", ""))
    if level not in {"fine", "middle", "coarse"}:
        raise ValueError("metric atlas has no valid feature level")
    bank = package_surface_feature_atlas(
        atlas,
        spatial_stride=int(args.spatial_stride),
        feature_space=f"v6_metric_{level}",
        feature_space_sha256=query_encoder_sha256,
        representation="exact_canonical_v6_metric_surface_texture",
        contributor_lineage_verified=True,
        metadata={
            "metric_feature_level": level,
            "metric_feature_stride": int(
                atlas_metadata.get("metric_feature_stride", 0)
            ),
            "map_metric_encoder_sha256": map_encoder_sha256,
            "query_metric_encoder_sha256": query_encoder_sha256,
            **{
                key: str(atlas_metadata[key])
                for key in lineage_keys
            },
        },
    )
    bank.save_npz(output_path)
    report = {
        "stage": "v6_metric_spatial_maplet_bank",
        "metric_atlas": str(args.metric_atlas),
        "query_metric_encoder": str(args.query_metric_encoder),
        "output_maplets": str(output_path),
        "maplet_count": len(bank),
        "component_count": int(bank.descriptors.shape[0]),
        "feature_dim": int(bank.descriptors.shape[1]),
        "spatial_stride": int(args.spatial_stride),
        "metric_feature_level": level,
        "map_metric_encoder_sha256": map_encoder_sha256,
        "query_metric_encoder_sha256": query_encoder_sha256,
        "contributor_geometry_lineage_verified": True,
        "production_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_pairwise_image_matching": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
