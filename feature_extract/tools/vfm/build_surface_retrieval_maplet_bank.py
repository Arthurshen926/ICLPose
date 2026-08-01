"""Strip a legacy construction artifact into an anchor-free retrieval bank.

The legacy bank is accepted only as an offline migration input. The output
contains no anchor, observation, image/view identity, or per-view descriptor
arrays and is the only maplet artifact accepted by the V4 refinement entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.surface_maplet_bank import (
    SurfaceMapletBuildConfig,
    _assign_observations_to_maplets,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import (
    Vfm2DgsAnchorMap,
    Vfm2DgsObservationBank,
)
from feature_extract.tools.vfm.train_surface_maplet_mapper import (
    _load_raw_feature_maps,
    _mapped_observation_descriptors,
    _validate_final_only_manifest,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_maplets", required=True)
    parser.add_argument(
        "--observation_bank",
        default="",
        help=(
            "Optional construction observation bank used to retain anonymous "
            "descriptor-component surface moments."
        ),
    )
    parser.add_argument(
        "--region_map",
        default="",
        help="Region map paired with --observation_bank.",
    )
    parser.add_argument("--metric_mapper_checkpoint", default="")
    parser.add_argument(
        "--surface_mapper_checkpoint",
        default="",
        help=(
            "RADIO-final maplet-identity mapper applied before anonymous "
            "per-maplet appearance clustering."
        ),
    )
    parser.add_argument(
        "--radio_final_manifest",
        default="",
        help=(
            "Raw RADIO-final manifest used to rebuild mapping descriptors as "
            "full map -> mapper -> context pooling. Required when the legacy "
            "maplets were produced by another mapper."
        ),
    )
    parser.add_argument("--radio_final_layer", default="radio_final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--source_already_surface_mapped",
        action="store_true",
        help=(
            "Declare that construction descriptors were produced by the "
            "specified surface mapper; validate lineage without reapplying it."
        ),
    )
    parser.add_argument("--maximum_components", type=int, default=4)
    parser.add_argument(
        "--spatial_grid_size",
        type=int,
        default=0,
        help=(
            "If positive, build a canonical spatial component grid per "
            "maplet from the observation bank instead of whole-maplet modes."
        ),
    )
    parser.add_argument(
        "--appearance_modes_per_cell",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--trajectory_ids",
        nargs="*",
        default=(),
        help=(
            "Optional mapping trajectories whose anonymous view features "
            "may enter the production prototypes."
        ),
    )
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _view_surface_moments(
    source: VfmSurfaceMapletBank,
    observations: Vfm2DgsObservationBank,
    region_map: Vfm2DgsAnchorMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recreate the anonymous per-view surface moments discarded by V4.

    Surface-maplet construction aggregated all assigned RADIO regions from one
    image into one descriptor.  Repeating that same assignment and weighting
    yields a 3D mean/covariance for the aggregate without retaining an image
    identifier or observation list in the production artifact.
    """

    raw_config = dict((source.metadata or {}).get("build_config", {}))
    if not raw_config:
        raise ValueError("construction maplets do not declare build_config")
    if raw_config.get("allowed_primitive_classes") is not None:
        raw_config["allowed_primitive_classes"] = tuple(
            int(value)
            for value in raw_config["allowed_primitive_classes"]
        )
    assignments = _assign_observations_to_maplets(
        region_map,
        observations,
        SurfaceMapletBuildConfig(**raw_config),
    )
    region_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(region_map.anchor_ids.tolist())
    }
    output_centers = np.full(
        (len(source.view_image_ids), 3), np.nan, dtype=np.float64
    )
    output_covariances = np.full(
        (len(source.view_image_ids), 3, 3), np.nan, dtype=np.float64
    )
    for maplet_row, maplet_id in enumerate(source.maplet_ids.tolist()):
        region_row = region_row_by_id.get(int(maplet_id))
        if region_row is None:
            raise ValueError("surface maplet is absent from its region map")
        observation_rows = np.flatnonzero(assignments == int(region_row))
        per_image: dict[str, list[int]] = {}
        for observation_row in observation_rows.tolist():
            per_image.setdefault(
                str(observations.image_ids[observation_row]), []
            ).append(int(observation_row))
        begin, end = (
            int(source.view_offsets[maplet_row]),
            int(source.view_offsets[maplet_row + 1]),
        )
        for view_row in range(begin, end):
            image_id = str(source.view_image_ids[view_row])
            rows = np.asarray(per_image.get(image_id, []), dtype=np.int64)
            if rows.size == 0:
                raise ValueError(
                    "cannot recover construction surface moment for "
                    f"{image_id}, maplet {maplet_id}"
                )
            weights = np.maximum(
                observations.descriptor_weights[rows].astype(np.float64),
                1e-6,
            )
            weights /= np.sum(weights)
            centers = np.asarray(
                observations.centers[rows], dtype=np.float64
            )
            center = np.sum(centers * weights[:, None], axis=0)
            residual = centers - center
            covariance = np.sum(
                weights[:, None, None]
                * (
                    np.asarray(
                        observations.covariances[rows], dtype=np.float64
                    )
                    + residual[:, :, None] * residual[:, None, :]
                ),
                axis=0,
            )
            output_centers[view_row] = center
            output_covariances[view_row] = covariance
    if (
        not np.all(np.isfinite(output_centers))
        or not np.all(np.isfinite(output_covariances))
    ):
        raise ValueError("failed to recover every descriptor surface moment")
    return output_centers, output_covariances, assignments


def _cluster_features(
    observations: np.ndarray,
    observation_weights: np.ndarray,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component_count = min(
        max(int(component_count), 1), int(observations.shape[0])
    )
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
    return centers, assignments, weights


def _full_map_mapped_view_descriptors(
    source: VfmSurfaceMapletBank,
    mapper: object,
    mapper_metadata: dict[str, object],
    manifest_path: Path,
    *,
    layer_name: str,
    selected_rows: np.ndarray,
    device: str,
) -> np.ndarray:
    """Rebuild map descriptors with the exact deployed operation order.

    A surface mapper is nonlinear, so ``mapper(pool(raw))`` is not
    interchangeable with ``pool(mapper(full_raw_map))``. The legacy artifact
    supplies only geometry, view membership, and token locations here; raw
    RADIO-final maps are projected before regional context pooling exactly as
    they are for an online query.
    """

    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise ValueError("full-map descriptor rebuild selected no views")
    if np.any(rows < 0) or np.any(rows >= len(source.view_image_ids)):
        raise ValueError("selected mapping-view row is out of range")
    manifest = TokenBankManifest.from_json(Path(manifest_path))
    manifest.validate(verify_checksums=False)
    _validate_final_only_manifest(manifest, str(layer_name))
    image_ids = np.asarray(source.view_image_ids, dtype=object)
    required_images = {str(image_ids[row]) for row in rows.tolist()}
    feature_maps = _load_raw_feature_maps(
        manifest, required_images, str(layer_name)
    )
    selected_mask = np.zeros((len(image_ids),), dtype=bool)
    selected_mask[rows] = True
    pool_sizes = tuple(
        int(value)
        for value in mapper_metadata.get("pool_sizes", (1, 3, 5, 9))
    )
    pool_weights = tuple(
        float(value)
        for value in mapper_metadata.get(
            "pool_weights", (0.4, 0.3, 0.2, 0.1)
        )
    )
    resolved_device = torch.device(
        str(device)
        if torch.cuda.is_available()
        or not str(device).startswith("cuda")
        else "cpu"
    )
    mapper.model.to(resolved_device)
    mapper.model.eval()
    with torch.no_grad():
        descriptors, rebuilt_rows = _mapped_observation_descriptors(
            mapper.model,
            feature_maps,
            image_ids,
            np.asarray(source.view_token_xy, dtype=np.float32),
            selected_mask,
            resolved_device,
            pool_sizes,
            pool_weights,
        )
    if not np.array_equal(rebuilt_rows, np.sort(rows)):
        raise RuntimeError(
            "full-map mapping descriptor rebuild changed observation rows"
        )
    output = np.full(
        (
            len(source.view_image_ids),
            int(mapper.model.config.output_dim),
        ),
        np.nan,
        dtype=np.float32,
    )
    output[rebuilt_rows] = (
        descriptors.detach().cpu().numpy().astype(np.float32)
    )
    return output


def _spatial_maplet_components(
    source: VfmSurfaceMapletBank,
    observations: Vfm2DgsObservationBank,
    region_map: Vfm2DgsAnchorMap,
    observation_assignments: np.ndarray,
    *,
    grid_size: int,
    appearance_modes_per_cell: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Compress mapping observations into an image-free feature texture."""

    region_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(region_map.anchor_ids.tolist())
    }
    descriptor_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    center_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    for maplet_row, maplet_id in enumerate(source.maplet_ids.tolist()):
        region_row = region_row_by_id[int(maplet_id)]
        rows = np.flatnonzero(
            observation_assignments == int(region_row)
        )
        if rows.size == 0:
            raise ValueError(
                f"maplet {maplet_id} has no assigned spatial observations"
            )
        center = np.asarray(source.centers[maplet_row], dtype=np.float64)
        frame = np.asarray(
            source.tangent_frames[maplet_row], dtype=np.float64
        )
        extent = np.maximum(
            np.asarray(source.extents[maplet_row, :2], dtype=np.float64),
            1e-4,
        )
        local = (
            np.asarray(observations.centers[rows], dtype=np.float64) - center
        ) @ frame[:2].T
        normalized = 0.5 * (local / extent[None] + 1.0)
        cell_xy = np.floor(normalized * int(grid_size)).astype(np.int64)
        cell_xy = np.clip(cell_xy, 0, int(grid_size) - 1)
        cell_id = cell_xy[:, 1] * int(grid_size) + cell_xy[:, 0]
        maplet_descriptors = []
        maplet_weights = []
        maplet_centers = []
        maplet_covariances = []
        for spatial_cell in np.unique(cell_id).tolist():
            local_rows = np.flatnonzero(cell_id == int(spatial_cell))
            observation_rows = rows[local_rows]
            features = np.asarray(
                observations.features[observation_rows], dtype=np.float32
            )
            weights = np.maximum(
                observations.descriptor_weights[observation_rows],
                1e-6,
            ).astype(np.float64)
            modes, labels, mode_weights = _cluster_features(
                features,
                weights,
                int(appearance_modes_per_cell),
            )
            for mode in range(modes.shape[0]):
                members = np.flatnonzero(labels == mode)
                if members.size == 0:
                    continue
                selected_rows = observation_rows[members]
                selected_weights = weights[members]
                selected_weights /= max(
                    float(np.sum(selected_weights)), 1e-12
                )
                selected_centers = np.asarray(
                    observations.centers[selected_rows], dtype=np.float64
                )
                spatial_center = np.sum(
                    selected_centers * selected_weights[:, None], axis=0
                )
                residual = selected_centers - spatial_center
                spatial_covariance = np.sum(
                    selected_weights[:, None, None]
                    * (
                        np.asarray(
                            observations.covariances[selected_rows],
                            dtype=np.float64,
                        )
                        + residual[:, :, None] * residual[:, None, :]
                    ),
                    axis=0,
                )
                maplet_descriptors.append(modes[mode])
                maplet_weights.append(mode_weights[mode])
                maplet_centers.append(spatial_center)
                maplet_covariances.append(spatial_covariance)
        descriptor_rows.append(
            np.asarray(maplet_descriptors, dtype=np.float32)
        )
        weight_rows.append(np.asarray(maplet_weights, dtype=np.float32))
        center_rows.append(np.asarray(maplet_centers, dtype=np.float32))
        covariance_rows.append(
            np.asarray(maplet_covariances, dtype=np.float32)
        )
    return descriptor_rows, weight_rows, center_rows, covariance_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_maplets)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite output")
    source = VfmSurfaceMapletBank.load_npz(Path(args.legacy_maplets))
    requested_trajectories = {
        str(value) for value in args.trajectory_ids
    }
    source_view_image_ids = np.asarray(
        source.view_image_ids, dtype=object
    )
    selected_mapping_view_rows = np.asarray(
        [
            row
            for row, image_id in enumerate(
                source_view_image_ids.tolist()
            )
            if (
                not requested_trajectories
                or str(image_id).split("/", 1)[0]
                in requested_trajectories
            )
        ],
        dtype=np.int64,
    )
    if selected_mapping_view_rows.size == 0:
        raise ValueError("requested mapping trajectories contain no views")
    if str(args.metric_mapper_checkpoint) and str(
        args.surface_mapper_checkpoint
    ):
        raise ValueError("choose only one descriptor mapper")
    identity_view_descriptors = np.asarray(
        source.view_descriptors, dtype=np.float32
    )
    identity_fallback_descriptors = np.asarray(
        source.descriptors, dtype=np.float32
    )
    surface_mapper_sha256 = ""
    radio_final_manifest_sha256 = ""
    descriptor_construction = "legacy_precomputed_view_descriptors"
    if str(args.surface_mapper_checkpoint):
        mapper_path = Path(args.surface_mapper_checkpoint)
        mapper, mapper_metadata = load_surface_maplet_mapper(
            mapper_path, device=str(args.device)
        )
        strict_holdout = {
            str(value)
            for value in mapper_metadata.get(
                "strict_holdout_trajectory_ids", []
            )
        }
        if requested_trajectories & strict_holdout:
            raise ValueError(
                "mapping trajectories overlap mapper strict holdout"
            )
        if bool(args.source_already_surface_mapped):
            if str(args.radio_final_manifest):
                raise ValueError(
                    "source_already_surface_mapped and "
                    "radio_final_manifest are mutually exclusive"
                )
            if (
                identity_view_descriptors.shape[1]
                != int(mapper.model.config.output_dim)
            ):
                raise ValueError(
                    "declared mapped descriptors differ from mapper output"
                )
            descriptor_space = dict(
                (source.metadata or {}).get("descriptor_space", {})
            )
            declared_mapper = str(
                descriptor_space.get(
                    "surface_maplet_mapper_checkpoint", ""
                )
            )
            if (
                not bool(
                    descriptor_space.get("full_map_mapper_applied", False)
                )
                or descriptor_space.get("mapper_type") != "surface_maplet"
                or not declared_mapper
            ):
                raise ValueError(
                    "source does not declare a pre-applied surface mapper"
                )
            declared_mapper_path = Path(declared_mapper)
            if not declared_mapper_path.exists():
                raise FileNotFoundError(
                    "cannot verify the source descriptor mapper: "
                    f"{declared_mapper_path}"
                )
            if hashlib.sha256(
                declared_mapper_path.read_bytes()
            ).hexdigest() != hashlib.sha256(
                mapper_path.read_bytes()
            ).hexdigest():
                raise ValueError(
                    "source descriptors were produced by a different "
                    "surface mapper"
                )
            descriptor_construction = (
                "verified_existing_full_map_mapper_then_context_pooling"
            )
        elif str(args.radio_final_manifest):
            manifest_path = Path(args.radio_final_manifest)
            identity_view_descriptors = (
                _full_map_mapped_view_descriptors(
                    source,
                    mapper,
                    dict(mapper_metadata),
                    manifest_path,
                    layer_name=str(args.radio_final_layer),
                    selected_rows=selected_mapping_view_rows,
                    device=str(args.device),
                )
            )
            identity_fallback_descriptors = np.zeros(
                (
                    len(source),
                    int(mapper.model.config.output_dim),
                ),
                dtype=np.float32,
            )
            radio_final_manifest_sha256 = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            descriptor_construction = (
                "full_radio_map_then_mapper_then_context_pooling"
            )
        else:
            if (
                identity_view_descriptors.shape[1]
                != int(mapper.model.config.input_dim)
            ):
                raise ValueError(
                    "legacy view descriptors do not inhabit the mapper input "
                    "space; provide --radio_final_manifest to rebuild them "
                    "from full RADIO-final maps"
                )
            parameter_device = next(mapper.model.parameters()).device
            with torch.no_grad():
                identity_view_descriptors = (
                    mapper.model(
                        torch.from_numpy(identity_view_descriptors)[
                            :, :, None, None
                        ].to(parameter_device)
                    )[:, :, 0, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                identity_fallback_descriptors = (
                    mapper.model(
                        torch.from_numpy(identity_fallback_descriptors)[
                            :, :, None, None
                        ].to(parameter_device)
                    )[:, :, 0, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            descriptor_construction = (
                "legacy_pooled_descriptor_mapper_baseline"
            )
        surface_mapper_sha256 = hashlib.sha256(
            mapper_path.read_bytes()
        ).hexdigest()
    elif str(args.radio_final_manifest):
        raise ValueError(
            "radio_final_manifest requires surface_mapper_checkpoint"
        )
    if bool(str(args.observation_bank)) != bool(str(args.region_map)):
        raise ValueError(
            "observation_bank and region_map must be provided together"
        )
    observation_bank = None
    region_map = None
    observation_assignments = None
    if str(args.observation_bank):
        observation_bank = Vfm2DgsObservationBank.load_npz(
            Path(args.observation_bank)
        )
        region_map = Vfm2DgsAnchorMap.load_npz(Path(args.region_map))
        (
            view_surface_centers,
            view_surface_covariances,
            observation_assignments,
        ) = (
            _view_surface_moments(
                source,
                observation_bank,
                region_map,
            )
        )
    else:
        view_surface_centers = None
        view_surface_covariances = None
    maximum_components = int(args.maximum_components)
    if maximum_components <= 0:
        raise ValueError("maximum_components must be positive")
    spatial_grid_size = int(args.spatial_grid_size)
    if spatial_grid_size < 0:
        raise ValueError("spatial_grid_size must be non-negative")
    if spatial_grid_size > 0:
        if args.trajectory_ids:
            raise ValueError(
                "trajectory filtering is currently defined for compact "
                "retrieval-region prototypes, not legacy spatial grids"
            )
        if (
            observation_bank is None
            or region_map is None
            or observation_assignments is None
        ):
            raise ValueError(
                "spatial grid requires observation_bank and region_map"
            )
        (
            components,
            component_weights,
            component_centers,
            component_covariances,
        ) = _spatial_maplet_components(
            source,
            observation_bank,
            region_map,
            observation_assignments,
            grid_size=spatial_grid_size,
            appearance_modes_per_cell=int(
                args.appearance_modes_per_cell
            ),
        )
    else:
        components = []
        component_weights = []
        component_centers = []
        component_covariances = []
    offsets = [0]
    retained_source_rows = []
    mapping_trajectories = (
        requested_trajectories
        if requested_trajectories
        else {
            str(value).split("/", 1)[0]
            for value in source_view_image_ids.tolist()
        }
    )
    for row in range(len(source)) if spatial_grid_size == 0 else []:
        begin, end = int(source.view_offsets[row]), int(source.view_offsets[row + 1])
        view_rows = np.arange(begin, end, dtype=np.int64)
        if requested_trajectories:
            view_rows = view_rows[
                np.asarray(
                    [
                        str(source.view_image_ids[index]).split("/", 1)[0]
                        in requested_trajectories
                        for index in view_rows.tolist()
                    ],
                    dtype=bool,
                )
            ]
        if view_rows.size == 0 and requested_trajectories:
            continue
        retained_source_rows.append(int(row))
        observations = identity_view_descriptors[view_rows]
        observation_weights = np.clip(
            source.view_quality_scores[view_rows], 1e-4, None
        )
        if observations.shape[0] == 0:
            raise RuntimeError(
                "retained retrieval region has no mapping observation"
            )
        if not np.all(np.isfinite(observations)):
            raise RuntimeError(
                "retained retrieval region contains an unreconstructed "
                "mapping descriptor"
            )
        centers, assignments, weights = _cluster_features(
            observations, observation_weights, maximum_components
        )
        component_count = centers.shape[0]
        spatial_centers = []
        spatial_covariances = []
        for cluster in range(component_count):
            members = np.flatnonzero(assignments == cluster)
            if members.size == 0:
                members = np.asarray(
                    [int(np.argmax(observations @ centers[cluster]))],
                    dtype=np.int64,
                )
            if view_surface_centers is None:
                spatial_center = np.asarray(
                    source.centers[row], dtype=np.float64
                )
                radius = max(float(np.max(source.extents[row])), 1e-3)
                spatial_covariance = np.eye(3, dtype=np.float64) * radius**2
            else:
                local_centers = view_surface_centers[view_rows][members]
                local_covariances = view_surface_covariances[view_rows][members]
                local_weights = observation_weights[members].astype(
                    np.float64
                )
                local_weights /= max(float(np.sum(local_weights)), 1e-12)
                spatial_center = np.sum(
                    local_centers * local_weights[:, None], axis=0
                )
                residual = local_centers - spatial_center
                spatial_covariance = np.sum(
                    local_weights[:, None, None]
                    * (
                        local_covariances
                        + residual[:, :, None] * residual[:, None, :]
                    ),
                    axis=0,
                )
            spatial_centers.append(spatial_center)
            spatial_covariances.append(spatial_covariance)
        order = np.argsort(-weights, kind="mergesort")
        components.append(centers[order])
        component_weights.append(weights[order])
        component_centers.append(
            np.asarray(spatial_centers, dtype=np.float32)[order]
        )
        component_covariances.append(
            np.asarray(spatial_covariances, dtype=np.float32)[order]
        )
        offsets.append(offsets[-1] + component_count)
    if spatial_grid_size > 0:
        retained_source_rows = list(range(len(source)))
        offsets = [0]
        for value in components:
            offsets.append(offsets[-1] + int(value.shape[0]))
    descriptors = np.concatenate(components, axis=0)
    descriptor_weights = np.concatenate(component_weights, axis=0)
    descriptor_centers = np.concatenate(component_centers, axis=0)
    descriptor_covariances = np.concatenate(
        component_covariances, axis=0
    )
    feature_space = "surface_maplet_radio_final"
    if str(args.metric_mapper_checkpoint):
        metric = load_surface_metric_feature_mapper(
            Path(args.metric_mapper_checkpoint), device="cpu"
        )
        descriptors = metric.project_points(descriptors)
        feature_space = "surface_metric_radio_final"
    elif str(args.surface_mapper_checkpoint):
        feature_space = "surface_maplet_mapper_radio_final"
    retained = np.asarray(retained_source_rows, dtype=np.int64)
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=source.maplet_ids[retained],
        centers=source.centers[retained],
        normals=source.normals[retained],
        extents=source.extents[retained],
        tangent_frames=source.tangent_frames[retained],
        descriptor_offsets=np.asarray(offsets, dtype=np.int64),
        descriptors=descriptors,
        descriptor_weights=descriptor_weights,
        descriptor_centers=descriptor_centers,
        descriptor_covariances=descriptor_covariances,
        quality_scores=source.quality_scores[retained],
        descriptor_uncertainties=source.descriptor_variances[retained],
        metadata={
            "artifact_type": "anchor_free_surface_retrieval_maplets",
            "vfm_layer": "radio_final",
            "maplet_count": int(retained.size),
            "representation": (
                "canonical_spatial_radio_final_mixture_per_maplet"
                if spatial_grid_size > 0
                else "compact_radio_final_mixture_per_metric_region"
            ),
            "descriptor_surface_moments": bool(
                view_surface_centers is not None
            ),
            "spatial_grid_size": spatial_grid_size,
            "appearance_modes_per_cell": (
                int(args.appearance_modes_per_cell)
                if spatial_grid_size > 0
                else 0
            ),
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
            "has_canonical_tangent_frames": True,
            "surface_mapper_sha256": surface_mapper_sha256,
            "mapping_trajectory_ids": sorted(mapping_trajectories),
            "descriptor_construction": descriptor_construction,
            "radio_final_manifest_sha256": (
                radio_final_manifest_sha256
            ),
            "radio_final_layer": str(args.radio_final_layer),
        },
    )
    bank.save_npz(output)
    summary = {
        "stage": "build_anchor_free_surface_retrieval_maplets",
        "output_maplets": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "maplet_count": len(bank),
        "feature_dim": int(bank.descriptors.shape[1]),
        "descriptor_component_count": int(bank.descriptors.shape[0]),
        "maximum_components": maximum_components,
        "descriptor_surface_moments": bool(
            view_surface_centers is not None
        ),
        "spatial_grid_size": spatial_grid_size,
        "appearance_modes_per_cell": (
            int(args.appearance_modes_per_cell)
            if spatial_grid_size > 0
            else 0
        ),
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
        "retained_geometry_fields": [
            "centers",
            "normals",
            "tangent_frames",
            "extents",
        ],
        "surface_mapper_sha256": surface_mapper_sha256,
        "descriptor_construction": descriptor_construction,
        "radio_final_manifest_sha256": radio_final_manifest_sha256,
        "radio_final_layer": str(args.radio_final_layer),
        "mapping_trajectory_ids": sorted(mapping_trajectories),
        "production_contract": dict(bank.metadata or {}),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
