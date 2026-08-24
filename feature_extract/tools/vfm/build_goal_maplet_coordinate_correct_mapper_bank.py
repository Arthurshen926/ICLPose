"""Build radial-coordinate-correct parent identities for mapper supervision.

The existing clean 2DGS contributor cache is rendered on an ideal-pinhole
grid.  RADIO-final tokens, however, were extracted from the raw
``SIMPLE_RADIAL`` image.  This builder uses the same inverse-warp contract as
canonical fusion, aggregates exact contributor mass into physical parent
maplets, and emits a compact multi-view ``VfmSurfaceMapletBank``.  Query/test
routes are rejected before either contributor labels or RADIO tensors are
opened.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
    load_contributors_in_radio_coordinates,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    VfmSurfaceMapletBank,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest


SCHEMA = "goal_maplet_coordinate_correct_mapper_supervision_bank_v1"


def _trajectory(image_id: str) -> str:
    value = str(image_id)
    if "/" not in value:
        raise ValueError(f"image ID lacks a trajectory: {value}")
    return value.split("/", 1)[0]


def _contributor_filename_identity(path: Path) -> tuple[str, str]:
    """Recover route/image identity before opening a pose-bearing archive."""

    name = Path(path).name
    if not name.endswith(".npz"):
        raise ValueError(f"contributor is not an NPZ: {path}")
    fields = name[:-4].split("__", 1)
    if (
        len(fields) != 2
        or not fields[0]
        or not fields[1]
        or "/" in fields[0]
        or "\\" in fields[0]
    ):
        raise ValueError(f"contributor filename lacks route identity: {path}")
    return fields[0], f"{fields[0]}/{fields[1]}"


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("integer tuple cannot be empty")
    return values


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("float tuple cannot be empty")
    return values


def _primitive_to_parent_lookup(physical: GoalMapletPhysicalMap) -> np.ndarray:
    """Return primitive-ID -> unique physical-parent-row, failing on overlap."""

    primitive_count = int(physical.primitive_ids.size)
    membership_count = np.bincount(
        np.asarray(physical.membership_primitive_rows, dtype=np.int64),
        minlength=primitive_count,
    )
    if not np.all(membership_count == 1):
        raise ValueError("mapper supervision requires a parent partition of primitives")
    row_to_parent = np.full((primitive_count,), -1, dtype=np.int32)
    for parent_row in range(int(physical.maplet_ids.size)):
        begin = int(physical.membership_offsets[parent_row])
        end = int(physical.membership_offsets[parent_row + 1])
        row_to_parent[physical.membership_primitive_rows[begin:end]] = parent_row
    maximum_id = int(np.max(physical.primitive_ids, initial=-1))
    result = np.full((maximum_id + 1,), -1, dtype=np.int32)
    result[np.asarray(physical.primitive_ids, dtype=np.int64)] = row_to_parent
    return result


def _parent_observations(
    ids: np.ndarray,
    weights: np.ndarray,
    primitive_to_parent: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    minimum_parent_mass_tokens: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate raw-grid contributor samples into parent mass and centroids."""

    primitive_ids = np.asarray(ids, dtype=np.int64)
    contribution = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if primitive_ids.ndim != 3 or contribution.shape != primitive_ids.shape:
        raise ValueError("contributor arrays must have shape (H,W,K)")
    height, width, topk = primitive_ids.shape
    if height % int(token_height) or width % int(token_width):
        raise ValueError("contributor grid must divide exactly into the RADIO grid")
    samples_per_token = (height // int(token_height)) * (
        width // int(token_width)
    )
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.int64),
        np.arange(width, dtype=np.int64),
        indexing="ij",
    )
    token_x = xx * int(token_width) // width
    token_y = yy * int(token_height) // height
    flat_ids = primitive_ids.reshape(-1)
    flat_weight = contribution.reshape(-1) / float(samples_per_token)
    x = np.broadcast_to(token_x[..., None], (height, width, topk)).reshape(-1)
    y = np.broadcast_to(token_y[..., None], (height, width, topk)).reshape(-1)
    in_range = (flat_ids >= 0) & (flat_ids < primitive_to_parent.size)
    parent = np.full(flat_ids.shape, -1, dtype=np.int32)
    parent[in_range] = primitive_to_parent[flat_ids[in_range]]
    valid = in_range & (parent >= 0) & (flat_weight > 0.0)
    parent_count = int(np.max(primitive_to_parent, initial=-1)) + 1
    mass = np.bincount(
        parent[valid], weights=flat_weight[valid], minlength=parent_count,
    )
    sum_x = np.bincount(
        parent[valid], weights=flat_weight[valid] * x[valid], minlength=parent_count,
    )
    sum_y = np.bincount(
        parent[valid], weights=flat_weight[valid] * y[valid], minlength=parent_count,
    )
    selected = np.flatnonzero(mass >= float(minimum_parent_mass_tokens))
    xy = np.stack(
        [sum_x[selected] / mass[selected], sum_y[selected] / mass[selected]], axis=1,
    ).astype(np.float32)
    # Quality affects only relative weighting.  Saturation prevents a single
    # close-up parent from dominating all cross-view identities.
    quality = np.clip(
        np.sqrt(mass[selected] / 4.0), 0.05, 1.0,
    ).astype(np.float32)
    return selected.astype(np.int64), mass[selected].astype(np.float32), xy, quality


def _select_capped_views(
    observations: Sequence[dict[str, object]],
    *,
    training_trajectories: set[str],
    maximum_views_per_parent_per_trajectory: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        grouped[(int(row["parent_row"]), str(row["trajectory_id"]))].append(row)
    selected: list[dict[str, object]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda row: (-float(row["mass_tokens"]), str(row["image_id"])),
        )
        selected.extend(ranked[: int(maximum_views_per_parent_per_trajectory)])
    training_count: dict[int, int] = defaultdict(int)
    for row in selected:
        if str(row["trajectory_id"]) in training_trajectories:
            training_count[int(row["parent_row"])] += 1
    retained = np.asarray(
        sorted(parent for parent, count in training_count.items() if count >= 2),
        dtype=np.int64,
    )
    retained_set = set(retained.tolist())
    selected = [row for row in selected if int(row["parent_row"]) in retained_set]
    selected.sort(key=lambda row: (int(row["parent_row"]), str(row["image_id"])))
    return selected, retained


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--output_surface_maplets", required=True)
    parser.add_argument("--output_map_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--training_trajectories", nargs="+", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", required=True)
    parser.add_argument(
        "--strict_holdout_trajectories",
        nargs="+",
        default=("seq3", "seq5", "seq12", "seq13", "seq14"),
    )
    parser.add_argument("--minimum_parent_mass_tokens", type=float, default=0.5)
    parser.add_argument(
        "--maximum_views_per_parent_per_trajectory", type=int, default=8,
    )
    parser.add_argument("--pool_sizes", default="1,3,5,9")
    parser.add_argument("--pool_weights", default="0.4,0.3,0.2,0.1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    outputs = tuple(Path(value) for value in (
        args.output_surface_maplets, args.output_map_manifest, args.summary_json,
    ))
    if not bool(args.force) and any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite coordinate-correct mapper inputs")
    if float(args.minimum_parent_mass_tokens) <= 0.0:
        raise ValueError("minimum_parent_mass_tokens must be positive")
    if int(args.maximum_views_per_parent_per_trajectory) <= 0:
        raise ValueError("view cap must be positive")
    training = {str(value) for value in args.training_trajectories}
    validation = {str(value) for value in args.validation_trajectories}
    holdout = {str(value) for value in args.strict_holdout_trajectories}
    if not training or not validation or training & validation:
        raise ValueError("training and validation trajectories must be nonempty/disjoint")
    selected_routes = training | validation
    if selected_routes & holdout:
        raise ValueError("strict holdout route appears in mapper supervision")

    physical_path = Path(args.physical_map).resolve()
    source_manifest_path = Path(args.radio_final_manifest).resolve()
    contributors_root = Path(args.contributors).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    primitive_to_parent = _primitive_to_parent_lookup(physical)
    source_manifest = TokenBankManifest.from_json(source_manifest_path)
    source_manifest.validate(verify_checksums=False)
    record_by_id = {record.image_id: record for record in source_manifest.records}
    filtered_records = tuple(
        record for record in source_manifest.records
        if _trajectory(record.image_id) in selected_routes
    )
    if not filtered_records:
        raise ValueError("map-route RADIO manifest is empty")
    filtered_ids = {record.image_id for record in filtered_records}
    if len(filtered_ids) != len(filtered_records):
        raise ValueError("filtered RADIO manifest contains duplicate image IDs")
    for record in filtered_records:
        matches = [layer for layer in record.layers if layer.name == "radio_final"]
        if len(matches) != 1 or str(matches[0].layer).lower() != "final":
            raise ValueError("mapper supervision requires one RADIO-final layer")
    if any(_trajectory(value) in holdout for value in filtered_ids):
        raise ValueError("filtered RADIO manifest contains a strict holdout")

    output_manifest = Path(args.output_map_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    TokenBankManifest(filtered_records).to_json(output_manifest)
    filtered_manifest_sha256 = file_sha256(output_manifest)

    observations: list[dict[str, object]] = []
    contributor_bindings: list[dict[str, str]] = []
    coordinate_audits: list[dict[str, object]] = []
    seen_images: set[str] = set()
    expected_grid: tuple[int, int] | None = None
    expected_channels: int | None = None
    contributor_paths = sorted(contributors_root.glob("*.npz"))
    contributor_routes_before_filter = {
        _contributor_filename_identity(path)[0] for path in contributor_paths
    }
    if not selected_routes.issubset(contributor_routes_before_filter):
        raise ValueError("map-route contributor allowlist is incomplete")
    selected_contributor_paths = [
        path for path in contributor_paths
        if _contributor_filename_identity(path)[0] in selected_routes
    ]
    for path in selected_contributor_paths:
        filename_trajectory, filename_image_id = _contributor_filename_identity(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata.get("image_id", ""))
        trajectory = str(metadata.get("trajectory_id", ""))
        if trajectory != filename_trajectory or image_id != filename_image_id:
            raise ValueError("contributor filename and embedded route identity differ")
        if image_id in seen_images or image_id not in filtered_ids:
            raise ValueError("contributor and filtered RADIO inventories differ")
        if _trajectory(image_id) != trajectory or trajectory in holdout:
            raise ValueError("contributor route lineage differs")
        if not bool(metadata.get("uses_declared_clean_2dgs_for_occlusion", False)):
            raise ValueError("mapper supervision requires declared-clean 2DGS contributors")
        record = record_by_id[image_id]
        if Path(str(metadata.get("token_path", ""))).resolve() != record.token_path.resolve():
            raise ValueError("contributor and manifest RADIO token paths differ")
        if expected_grid is None:
            with np.load(record.token_path, allow_pickle=False) as token_data:
                raw_shape = np.asarray(token_data["radio_final"]).shape
            if len(raw_shape) == 4 and int(raw_shape[0]) == 1:
                raw_shape = raw_shape[1:]
            if len(raw_shape) != 3:
                raise ValueError("RADIO final tensor must have shape (C,H,W)")
            channels, token_height, token_width = (
                int(value) for value in raw_shape
            )
        else:
            token_height, token_width = expected_grid
            channels = int(expected_channels)
        if expected_grid is None:
            expected_grid = (token_height, token_width)
            expected_channels = channels
        labels, coordinate_audit = load_contributors_in_radio_coordinates(path)
        parent_rows, mass, xy, quality = _parent_observations(
            labels.topk_primitive_ids,
            labels.topk_weights,
            primitive_to_parent,
            token_height=token_height,
            token_width=token_width,
            minimum_parent_mass_tokens=float(args.minimum_parent_mass_tokens),
        )
        for parent_row, parent_mass, point, score in zip(
            parent_rows.tolist(), mass.tolist(), xy, quality.tolist(),
        ):
            observations.append({
                "parent_row": int(parent_row),
                "image_id": image_id,
                "trajectory_id": trajectory,
                "mass_tokens": float(parent_mass),
                "token_xy": np.asarray(point, dtype=np.float32),
                "quality": float(score),
                "token_path": str(record.token_path.resolve()),
            })
        coordinate_audits.append(coordinate_audit)
        contributor_bindings.append({
            "image_id": image_id,
            "file_sha256": file_sha256(path),
        })
        seen_images.add(image_id)
    missing = sorted(filtered_ids - seen_images)
    if missing:
        raise ValueError(f"map-route RADIO image lacks a contributor cache: {missing[0]}")
    if expected_grid is None or expected_channels is None:
        raise ValueError("no coordinate-correct mapper observations")

    selected, retained_parent_rows = _select_capped_views(
        observations,
        training_trajectories=training,
        maximum_views_per_parent_per_trajectory=int(
            args.maximum_views_per_parent_per_trajectory
        ),
    )
    if retained_parent_rows.size < 2 or not selected:
        raise ValueError("too few cross-view physical parent identities")

    pool_sizes = _parse_int_tuple(args.pool_sizes)
    pool_weights = _parse_float_tuple(args.pool_weights)
    config = RadioFinalRegionConfig(pool_sizes=pool_sizes, pool_weights=pool_weights)
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_image[str(row["image_id"])].append(row)
    for image_id in sorted(by_image):
        rows = by_image[image_id]
        with np.load(Path(str(rows[0]["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        if raw.ndim == 4 and int(raw.shape[0]) == 1:
            raw = raw[0]
        if tuple(raw.shape) != (
            int(expected_channels), int(expected_grid[0]), int(expected_grid[1])
        ):
            raise ValueError("mapper supervision RADIO shapes differ")
        descriptors = encode_radio_final_regions(
            raw,
            np.stack([np.asarray(row["token_xy"], dtype=np.float32) for row in rows]),
            config,
        )
        for row, descriptor in zip(rows, descriptors):
            row["descriptor"] = descriptor.astype(np.float32, copy=False)

    per_parent: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        per_parent[int(row["parent_row"])].append(row)
    view_offsets = [0]
    view_image_ids: list[str] = []
    view_xy: list[np.ndarray] = []
    view_descriptors: list[np.ndarray] = []
    view_quality: list[float] = []
    maplet_descriptors: list[np.ndarray] = []
    maplet_quality: list[float] = []
    descriptor_variance: list[float] = []
    for parent_row in retained_parent_rows.tolist():
        rows = sorted(per_parent[int(parent_row)], key=lambda row: str(row["image_id"]))
        descriptor = np.stack([
            np.asarray(row["descriptor"], dtype=np.float32) for row in rows
        ])
        quality = np.asarray([float(row["quality"]) for row in rows], dtype=np.float32)
        mean = np.average(descriptor, axis=0, weights=quality)
        resultant = float(np.linalg.norm(mean))
        maplet_descriptors.append(mean.astype(np.float32))
        descriptor_variance.append(float(np.clip(1.0 - resultant, 0.0, 1.0)))
        maplet_quality.append(float(np.mean(quality) * max(resultant, 1e-6)))
        for row in rows:
            view_image_ids.append(str(row["image_id"]))
            view_xy.append(np.asarray(row["token_xy"], dtype=np.float32))
            view_descriptors.append(np.asarray(row["descriptor"], dtype=np.float32))
            view_quality.append(float(row["quality"]))
        view_offsets.append(len(view_image_ids))

    support_offsets = [0]
    support_ids: list[int] = []
    for parent_row in retained_parent_rows.tolist():
        begin = int(physical.membership_offsets[parent_row])
        end = int(physical.membership_offsets[parent_row + 1])
        primitive_rows = physical.membership_primitive_rows[begin:end]
        support_ids.extend(physical.primitive_ids[primitive_rows].astype(int).tolist())
        support_offsets.append(len(support_ids))

    contributor_inventory_sha256 = canonical_json_sha256({
        "coordinate_contract": COORDINATE_CONTRACT,
        "contributors": contributor_bindings,
    })
    supervision_lineage = {
        "coordinate_correct": True,
        "coordinate_contract": COORDINATE_CONTRACT,
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": file_sha256(physical_path),
        "source_radio_final_manifest_file_sha256": file_sha256(source_manifest_path),
        "radio_final_manifest_file_sha256": filtered_manifest_sha256,
        "contributor_inventory_sha256": contributor_inventory_sha256,
        "training_trajectory_ids": sorted(training),
        "validation_trajectory_ids": sorted(validation),
        "strict_holdout_trajectory_ids": sorted(holdout),
        "strict_holdout_present": False,
        "route_allowlist_applied_before_opening_contributor_archives": True,
        "source_contributor_count_before_route_allowlist": len(contributor_paths),
        "selected_contributor_count": len(selected_contributor_paths),
    }
    bank = VfmSurfaceMapletBank(
        maplet_ids=physical.maplet_ids[retained_parent_rows],
        centers=physical.maplet_centers[retained_parent_rows],
        normals=physical.maplet_normals[retained_parent_rows],
        tangent_frames=physical.maplet_frames[retained_parent_rows],
        extents=physical.maplet_extents[retained_parent_rows],
        descriptors=np.stack(maplet_descriptors),
        quality_scores=np.asarray(maplet_quality, dtype=np.float32),
        descriptor_variances=np.asarray(descriptor_variance, dtype=np.float32),
        anchor_offsets=np.zeros((retained_parent_rows.size + 1,), dtype=np.int64),
        anchor_ids=np.empty((0,), dtype=np.int64),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(support_ids, dtype=np.int64),
        view_offsets=np.asarray(view_offsets, dtype=np.int64),
        view_image_ids=tuple(view_image_ids),
        view_token_xy=np.stack(view_xy),
        view_grid_sizes=np.tile(
            np.asarray([[expected_grid[1], expected_grid[0]]], dtype=np.int32),
            (len(view_image_ids), 1),
        ),
        view_descriptors=np.stack(view_descriptors),
        view_quality_scores=np.asarray(view_quality, dtype=np.float32),
        metadata={
            "artifact_type": SCHEMA,
            "representation": "coordinate_correct_physical_parent_cross_view_identity",
            "vfm_layer": "radio_final",
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "identity_source": "goal_maplet_physical_parent_partition",
            "region_config": config.to_dict(),
            "minimum_parent_mass_tokens": float(args.minimum_parent_mass_tokens),
            "maximum_views_per_parent_per_trajectory": int(
                args.maximum_views_per_parent_per_trajectory
            ),
            "supervision_coordinate_lineage": supervision_lineage,
        },
    )
    output_bank = Path(args.output_surface_maplets)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    bank.save_npz(output_bank)

    route_view_counts = {
        route: sum(_trajectory(image_id) == route for image_id in view_image_ids)
        for route in sorted(selected_routes)
    }
    report = {
        "artifact_type": SCHEMA,
        "physical_map_sha256": physical.content_sha256,
        "surface_maplets": str(output_bank.resolve()),
        "surface_maplets_file_sha256": file_sha256(output_bank),
        "map_manifest": str(output_manifest.resolve()),
        "map_manifest_file_sha256": filtered_manifest_sha256,
        "map_image_count": len(filtered_ids),
        "parent_identity_count": int(retained_parent_rows.size),
        "view_observation_count": len(view_image_ids),
        "view_count_by_trajectory": route_view_counts,
        "training_trajectory_ids": sorted(training),
        "validation_trajectory_ids": sorted(validation),
        "strict_holdout_trajectory_ids": sorted(holdout),
        "strict_holdout_present": False,
        "route_allowlist_applied_before_opening_contributor_archives": True,
        "source_contributor_count_before_route_allowlist": len(contributor_paths),
        "selected_contributor_count": len(selected_contributor_paths),
        "coordinate_correct": True,
        "coordinate_contract": COORDINATE_CONTRACT,
        "contributor_inventory_sha256": contributor_inventory_sha256,
        "minimum_valid_raw_sample_fraction": float(min(
            float(value["valid_raw_sample_fraction"]) for value in coordinate_audits
        )),
        "maximum_inverse_roundtrip_residual_contributor_px": float(max(
            float(value["maximum_inverse_roundtrip_residual_contributor_px"])
            for value in coordinate_audits
        )),
        "maximum_pinhole_to_raw_displacement_contributor_px": float(max(
            float(value["maximum_pinhole_to_raw_displacement_contributor_px"])
            for value in coordinate_audits
        )),
        "supervision_coordinate_lineage": supervision_lineage,
        "uses_existing_3dgs": True,
        "trains_or_reconstructs_3dgs": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "promotion_eligible_as_mapper_supervision": True,
    }
    summary = Path(args.summary_json)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
