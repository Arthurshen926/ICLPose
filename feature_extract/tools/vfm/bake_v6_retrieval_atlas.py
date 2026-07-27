"""Bake an exact clean-2DGS RADIO-final spatial retrieval texture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_raw_final,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
    package_surface_feature_atlas,
)
from feature_extract.vfm.localization_v6.atlas_baking import (
    bake_feature_atlas,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    PrimitiveContributorBuffer,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    load_surface_spatial_projection,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--surface_mapper_checkpoint", default="")
    parser.add_argument("--surface_spatial_projection_checkpoint", default="")
    parser.add_argument("--contributor_dirs", nargs="+", required=True)
    parser.add_argument(
        "--trajectory_ids",
        nargs="*",
        default=[],
        help="Optional explicit mapping-trajectory subset.",
    )
    parser.add_argument("--output_atlas", required=True)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--minimum_support", type=int, default=2)
    parser.add_argument(
        "--appearance_modes", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument(
        "--spatial_stride",
        type=int,
        default=2,
        help="Subsample canonical atlas cells when packaging retrieval maplets.",
    )
    parser.add_argument(
        "--feature_transform",
        choices=(
            "raw_radio_pca",
            "surface_maplet_mapper",
            "surface_spatial_projection",
        ),
        default="raw_radio_pca",
        help=(
            "raw_radio_pca preserves within-maplet token variation; the old "
            "maplet-identity mapper is retained only as an ablation."
        ),
    )
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--projection_samples", type=int, default=8192)
    parser.add_argument("--projection_seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage(
    geometry: MapletFeatureAtlasBank,
    rows: list[dict[str, object]],
) -> tuple[dict[str, str], bool]:
    keys = (
        "geometry_source_sha256",
        "clean_geometry_source_sha256",
        "clean_source_index_sha256",
    )
    values = {
        key: {str(row.get(key, "")) for row in rows} for key in keys
    }
    if any(len(unique) != 1 for unique in values.values()):
        raise ValueError("contributor caches mix geometry source lineages")
    contributor = {key: next(iter(values[key])) for key in keys}
    geometry_metadata = dict(geometry.metadata or {})
    verified = all(
        bool(contributor[key])
        and contributor[key] == str(geometry_metadata.get(key, ""))
        for key in keys
    )
    declared = any(bool(value) for value in contributor.values()) or any(
        bool(geometry_metadata.get(key, "")) for key in keys
    )
    if declared and not verified:
        raise ValueError(
            "contributor cache and canonical geometry lineages differ"
        )
    return contributor, verified


def _fit_radio_projection(
    cache_paths: Sequence[Path],
    *,
    output_dim: int,
    maximum_samples: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Fit an origin-preserving RADIO subspace without maplet-ID collapse."""

    if not cache_paths:
        raise ValueError("no contributor caches were found")
    rng = np.random.default_rng(int(seed))
    per_view = max(
        1, int(np.ceil(int(maximum_samples) / len(cache_paths)))
    )
    samples = []
    input_dim = None
    for path in cache_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        raw = _load_raw_final(
            Path(str(metadata["token_path"])), "radio_final"
        )
        flat = np.asarray(raw, dtype=np.float32).reshape(raw.shape[0], -1).T
        flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
        input_dim = int(flat.shape[1])
        count = min(per_view, flat.shape[0])
        rows = rng.choice(flat.shape[0], size=count, replace=False)
        samples.append(flat[rows])
    matrix = np.concatenate(samples, axis=0)
    if matrix.shape[0] > int(maximum_samples):
        rows = rng.choice(
            matrix.shape[0], size=int(maximum_samples), replace=False
        )
        matrix = matrix[rows]
    dimension = min(int(output_dim), int(matrix.shape[1]), int(matrix.shape[0]))
    if dimension < 2 or input_dim is None:
        raise ValueError("insufficient RADIO samples for spatial projection")
    torch.manual_seed(int(seed))
    tensor = torch.as_tensor(
        matrix, dtype=torch.float32, device=str(device)
    )
    _u, _s, right = torch.pca_lowrank(
        tensor, q=dimension, center=False, niter=4
    )
    return right[:, :dimension].T.detach().cpu().numpy().astype(np.float32)


def _project_radio(
    raw: np.ndarray, projection: np.ndarray, *, device: str
) -> np.ndarray:
    tensor = torch.as_tensor(
        np.asarray(raw, dtype=np.float32),
        dtype=torch.float32,
        device=str(device),
    )
    tensor = tensor / torch.clamp(
        torch.linalg.norm(tensor, dim=0, keepdim=True), min=1e-8
    )
    matrix = torch.as_tensor(
        projection, dtype=torch.float32, device=str(device)
    )
    output = torch.einsum("oc,chw->ohw", matrix, tensor)
    output = output / torch.clamp(
        torch.linalg.norm(output, dim=0, keepdim=True), min=1e-8
    )
    return output.detach().cpu().numpy().astype(np.float32)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _package_retrieval_maplets(
    atlas: MapletFeatureAtlasBank,
    *,
    spatial_stride: int,
    feature_transform: str,
    feature_transform_sha256: str,
    query_projection: np.ndarray | None,
    contributor_lineage_verified: bool,
) -> SurfaceRetrievalMapletBank:
    return package_surface_feature_atlas(
        atlas,
        spatial_stride=int(spatial_stride),
        feature_space=str(feature_transform),
        feature_space_sha256=str(feature_transform_sha256),
        representation="exact_canonical_radio_final_spatial_texture",
        query_projection=query_projection,
        contributor_lineage_verified=bool(
            contributor_lineage_verified
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = (
        Path(args.output_atlas),
        Path(args.output_maplets),
        Path(args.summary_json),
    )
    if any(path.exists() for path in outputs) and not bool(args.force):
        raise FileExistsError("refusing to overwrite retrieval atlas outputs")
    geometry = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    requested_trajectories = {
        str(value) for value in args.trajectory_ids
    }
    cache_paths = sorted(
        {
            path.resolve()
            for directory in args.contributor_dirs
            for path in Path(directory).glob("*.npz")
        }
    )
    if requested_trajectories:
        filtered_paths = []
        for path in cache_paths:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
            if str(metadata["trajectory_id"]) in requested_trajectories:
                filtered_paths.append(path)
        cache_paths = filtered_paths
    feature_transform = str(args.feature_transform)
    mapper = None
    mapper_metadata: dict[str, object] = {}
    spatial_projection_metadata: dict[str, object] = {}
    query_projection = None
    if feature_transform == "surface_maplet_mapper":
        if not str(args.surface_mapper_checkpoint):
            raise ValueError(
                "surface_maplet_mapper transform requires its checkpoint"
            )
        mapper, loaded_metadata = load_surface_maplet_mapper(
            Path(args.surface_mapper_checkpoint), device=str(args.device)
        )
        mapper_metadata = dict(loaded_metadata)
        feature_transform_sha256 = _file_sha256(
            Path(args.surface_mapper_checkpoint)
        )
    elif feature_transform == "surface_spatial_projection":
        if not str(args.surface_spatial_projection_checkpoint):
            raise ValueError(
                "surface_spatial_projection transform requires its checkpoint"
            )
        projection_model, loaded_metadata = load_surface_spatial_projection(
            Path(args.surface_spatial_projection_checkpoint),
            device="cpu",
        )
        spatial_projection_metadata = dict(loaded_metadata)
        query_projection = (
            projection_model.projection.detach().cpu().numpy().astype(
                np.float32
            )
        )
        feature_transform_sha256 = _file_sha256(
            Path(args.surface_spatial_projection_checkpoint)
        )
        geometry_metadata = dict(geometry.metadata or {})
        for key in (
            "geometry_source_sha256",
            "clean_geometry_source_sha256",
            "clean_source_index_sha256",
        ):
            expected = str(geometry_metadata.get(key, ""))
            observed = str(spatial_projection_metadata.get(key, ""))
            if expected and observed != expected:
                raise ValueError(
                    f"surface spatial projection {key} differs from geometry"
                )
    else:
        query_projection = _fit_radio_projection(
            cache_paths,
            output_dim=int(args.projection_dim),
            maximum_samples=int(args.projection_samples),
            seed=int(args.projection_seed),
            device=str(args.device),
        )
        feature_transform_sha256 = _array_sha256(query_projection)
    views = []
    buffers = []
    metadata_rows = []
    trajectories = []
    occlusion_policies = []
    for index, path in enumerate(cache_paths):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            trajectory_id = str(metadata["trajectory_id"])
            raw = _load_raw_final(
                Path(str(metadata["token_path"])), "radio_final"
            )
            mapped = (
                mapper.project(raw).measurement_context
                if mapper is not None
                else _project_radio(
                    raw, query_projection, device=str(args.device)
                )
            )
            camera = ColmapCamera(
                camera_id=0,
                model_id=int(data["camera_model_id"]),
                width=int(data["camera_width"]),
                height=int(data["camera_height"]),
                params=tuple(
                    np.asarray(data["camera_params"], dtype=np.float64)
                ),
            )
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
            topk_ids = np.asarray(data["topk_ids"], dtype=np.int64)
            topk_weights = np.asarray(data["topk_weights"], dtype=np.float32)
            depth = np.asarray(data["dominant_depth"], dtype=np.float32)
        views.append(
            GaussianVFMFeatureView(
                image_id=str(metadata["image_id"]),
                feature_map=np.asarray(mapped, dtype=np.float32),
                pose_w2c=pose,
                camera=camera,
            )
        )
        buffers.append(
            PrimitiveContributorBuffer(
                dominant_ids=topk_ids[..., 0],
                dominant_weights=topk_weights[..., 0],
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                primitive_depth=depth,
                metadata=metadata,
            )
        )
        metadata_rows.append(metadata)
        trajectories.append(trajectory_id)
        occlusion_policies.append(
            str(metadata.get("occlusion_primitive_policy", ""))
        )
        print(
            f"[{index + 1}/{len(cache_paths)}] {metadata['image_id']}",
            flush=True,
        )
    if not views:
        raise ValueError("no contributor caches were found")
    if len(set(occlusion_policies)) != 1:
        raise ValueError("contributor caches mix occlusion policies")
    contributor_lineage, lineage_verified = _lineage(
        geometry, metadata_rows
    )
    atlas, atlas_report = bake_feature_atlas(
        geometry,
        views,
        buffers,
        minimum_support=int(args.minimum_support),
        appearance_modes=int(args.appearance_modes),
        trajectory_ids=trajectories,
        metadata={
            "vfm_layer": "radio_final",
            "metric_feature_level": "retrieval",
            "metric_feature_stride": 16,
            "query_feature_transform": feature_transform,
            "query_feature_transform_sha256": feature_transform_sha256,
            "surface_mapper_best_epoch": mapper_metadata.get(
                "best_epoch", -1
            ),
            "surface_spatial_projection_best_step": (
                spatial_projection_metadata.get("best_step", -1)
            ),
            "contributor_occlusion_primitive_policy": (
                occlusion_policies[0]
            ),
            "contributor_geometry_lineage_verified": bool(
                lineage_verified
            ),
            **{
                f"contributor_{key}": value
                for key, value in contributor_lineage.items()
            },
        },
    )
    atlas.save_npz(outputs[0])
    bank = _package_retrieval_maplets(
        atlas,
        spatial_stride=int(args.spatial_stride),
        feature_transform=feature_transform,
        feature_transform_sha256=feature_transform_sha256,
        query_projection=query_projection,
        contributor_lineage_verified=lineage_verified,
    )
    bank.save_npz(outputs[1])
    report = {
        **atlas_report,
        "stage": "v6_exact_canonical_radio_final_retrieval_atlas",
        "output_atlas": str(outputs[0]),
        "output_maplets": str(outputs[1]),
        "retrieval_component_count": int(bank.descriptors.shape[0]),
        "spatial_stride": int(args.spatial_stride),
        "query_feature_transform": feature_transform,
        "query_feature_transform_sha256": feature_transform_sha256,
        "projection_input_dim": (
            int(query_projection.shape[1])
            if query_projection is not None
            else None
        ),
        "projection_output_dim": (
            int(query_projection.shape[0])
            if query_projection is not None
            else None
        ),
        "contributor_geometry_lineage_verified": bool(lineage_verified),
        "production_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_paths": False,
            "stores_mapping_image_ids": False,
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    }
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
