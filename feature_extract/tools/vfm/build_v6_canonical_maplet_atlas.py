"""Build observation-independent canonical V6 maplet atlas geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_2dgs_surface_feature_field import (
    _clean_source_indices,
)
from feature_extract.vfm.gaussian_vfm_field import (
    load_gaussian_vfm_source_from_ply,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    canonical_maplet_geometry,
    empty_atlas_bank_from_geometry,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--clean_gaussian_ply", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--maximum_chart_depth_ratio", type=float, default=1.5)
    parser.add_argument("--maximum_maplets", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _subset_maplets(
    bank: VfmSurfaceMapletBank, maximum: int
) -> VfmSurfaceMapletBank:
    if maximum <= 0 or maximum >= len(bank):
        return bank
    count = int(maximum)
    support_end = int(bank.support_offsets[count])
    anchor_end = int(bank.anchor_offsets[count])
    view_end = int(bank.view_offsets[count])
    return VfmSurfaceMapletBank(
        maplet_ids=bank.maplet_ids[:count],
        centers=bank.centers[:count],
        normals=bank.normals[:count],
        tangent_frames=bank.tangent_frames[:count],
        extents=bank.extents[:count],
        descriptors=bank.descriptors[:count],
        quality_scores=bank.quality_scores[:count],
        descriptor_variances=bank.descriptor_variances[:count],
        anchor_offsets=bank.anchor_offsets[: count + 1],
        anchor_ids=bank.anchor_ids[:anchor_end],
        support_offsets=bank.support_offsets[: count + 1],
        support_element_ids=bank.support_element_ids[:support_end],
        view_offsets=bank.view_offsets[: count + 1],
        view_image_ids=bank.view_image_ids[:view_end],
        view_token_xy=bank.view_token_xy[:view_end],
        view_grid_sizes=bank.view_grid_sizes[:view_end],
        view_descriptors=bank.view_descriptors[:view_end],
        view_quality_scores=bank.view_quality_scores[:view_end],
        metadata=bank.metadata,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_index_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_atlas)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite V6 atlas outputs")
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply))
    clean = _clean_source_indices(Path(args.clean_gaussian_ply))
    maplets = _subset_maplets(
        VfmSurfaceMapletBank.load_npz(Path(args.maplets)),
        int(args.maximum_maplets),
    )
    geometry_source_sha256 = _file_sha256(Path(args.gaussian_ply))
    clean_source_sha256 = _file_sha256(Path(args.clean_gaussian_ply))
    clean_source_index_sha256 = _source_index_sha256(clean)
    maplet_source_sha256 = _file_sha256(Path(args.maplets))
    xyz, primitive_ids, valid, audit = canonical_maplet_geometry(
        source,
        maplets,
        resolution=int(args.resolution),
        maximum_chart_depth_ratio=float(args.maximum_chart_depth_ratio),
        allowed_source_indices=clean,
    )
    atlas = empty_atlas_bank_from_geometry(
        maplets,
        xyz,
        primitive_ids,
        valid,
        feature_dim=1,
        metadata={
            "artifact_type": "v6_canonical_maplet_atlas_geometry",
            "geometry_source": audit["geometry_source"],
            "geometry_source_sha256": geometry_source_sha256,
            "clean_geometry_source_sha256": clean_source_sha256,
            "clean_source_index_sha256": clean_source_index_sha256,
            "maplet_source_sha256": maplet_source_sha256,
            "observation_dependent_geometry": False,
            "feature_state": "unbaked",
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_point_correspondence_pnp": False,
        },
    )
    atlas.save_npz(output)
    report = {
        "stage": "v6_g0_canonical_maplet_atlas_geometry",
        **audit,
        "clean_primitive_count": int(clean.size),
        "geometry_source_sha256": geometry_source_sha256,
        "clean_geometry_source_sha256": clean_source_sha256,
        "clean_source_index_sha256": clean_source_index_sha256,
        "maplet_source_sha256": maplet_source_sha256,
        "output_atlas": str(output),
        "gate": {
            "observation_independent_geometry": True,
            "uses_raster_contributor_ids_for_feature_baking": True,
            "kdtree_production_assignment": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
