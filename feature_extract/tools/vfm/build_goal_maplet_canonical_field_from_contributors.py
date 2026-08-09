"""Build the canonical field directly from exact clean-2DGS contributors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalFieldFusionAccumulator,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.canonical_codec import CanonicalRadioCodec
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


TEACHER_NAMES = ("dino_v3_7b", "sam3", "siglip2-g")


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def _compress_teacher(value: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] % int(dimensions):
        raise ValueError("teacher width is not divisible by requested compact width")
    compact = array.reshape(array.shape[0], int(dimensions), -1).mean(axis=2)
    return _normalize_rows(compact)


def _pixel_layout(
    ids: np.ndarray,
    weights: np.ndarray,
    raw_shape: tuple[int, int],
    dense_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width, topk = ids.shape
    raw_height, raw_width = (int(raw_shape[0]), int(raw_shape[1]))
    pixel = np.arange(height * width, dtype=np.int64)
    pixel_x, pixel_y = pixel % width, pixel // width
    token_x = np.minimum((pixel_x + 0.5) * raw_width / width, raw_width - 1).astype(np.int64)
    token_y = np.minimum((pixel_y + 0.5) * raw_height / height, raw_height - 1).astype(np.int64)
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_pixel = np.repeat(pixel, topk)
    valid = (flat_ids >= 0) & (flat_ids < dense_lookup.size) & (flat_weights > 0.0)
    flat_rows = np.full(flat_ids.shape, -1, dtype=np.int64)
    flat_rows[valid] = dense_lookup[flat_ids[valid]]
    valid &= flat_rows >= 0
    rows = flat_rows[valid]
    mass = flat_weights[valid]
    contributing_pixel = flat_pixel[valid]
    if rows.size == 0:
        raise ValueError("contributor view has no clean-2DGS support")
    order = np.argsort(rows, kind="stable")
    rows, mass, contributing_pixel = rows[order], mass[order], contributing_pixel[order]
    first = np.r_[True, rows[1:] != rows[:-1]]
    starts = np.flatnonzero(first)
    return rows, mass, contributing_pixel, starts, token_x, token_y


def _teacher_token_fields(
    teacher_path: Path,
    *,
    raw_height: int,
    raw_width: int,
    dimensions: int,
) -> dict[str, np.ndarray]:
    with np.load(teacher_path, allow_pickle=False) as teacher:
        xy = np.asarray(teacher["token_xy"], dtype=np.float32).reshape(-1, 2)
        values = {
            "dino_v3_7b": _compress_teacher(teacher["dino_v3_7b"], int(dimensions)),
            "sam3": _compress_teacher(teacher["sam3"], int(dimensions)),
            "siglip2-g": _compress_teacher(teacher["siglip2-g"], int(dimensions)),
        }
        summary = _compress_teacher(
            np.asarray(teacher["siglip2-g_summary"], dtype=np.float32)[None], int(dimensions),
        )[0]
    values["siglip2-g"] = _normalize_rows(
        values["siglip2-g"] + summary[None],
    )
    grid_y, grid_x = np.mgrid[: int(raw_height), : int(raw_width)]
    grid_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    distance = np.sum((grid_xy[:, None] - xy[None]) ** 2, axis=2)
    nearest = np.argmin(distance, axis=1)
    return {
        name: value[nearest].reshape(int(raw_height), int(raw_width), int(dimensions))
        for name, value in values.items()
    }


def _aggregate_view_teacher(
    field: np.ndarray,
    contributing_pixel: np.ndarray,
    mass: np.ndarray,
    starts: np.ndarray,
    token_x: np.ndarray,
    token_y: np.ndarray,
) -> np.ndarray:
    pixel_teacher = field[token_y, token_x]
    weighted = mass[:, None] * pixel_teacher[contributing_pixel]
    view_mass = np.add.reduceat(mass, starts)
    return _normalize_rows(
        np.add.reduceat(weighted, starts, axis=0) / np.maximum(view_mass[:, None], 1e-8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument(
        "--feature_space",
        choices=("raw_radio_final", "canonical_codec", "retrieval_mapper"),
        default="retrieval_mapper",
    )
    parser.add_argument("--canonical_codec", default="")
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--exclude_trajectories", nargs="*", default=["seq11", "seq3", "seq5", "seq13"])
    parser.add_argument("--maximum_images", type=int, default=0)
    parser.add_argument("--teacher_cache", default="")
    parser.add_argument("--teacher_dimensions", type=int, default=32)
    parser.add_argument("--teacher_quality_floor", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_field), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite exact canonical field")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    codec = CanonicalRadioCodec.load_npz(Path(args.canonical_codec)) if args.canonical_codec else None
    if args.feature_space == "canonical_codec" and codec is None:
        raise ValueError("canonical_codec feature space requires --canonical_codec")
    dense_lookup = np.full((int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64)
    dense_lookup[physical.primitive_ids] = np.arange(physical.primitive_ids.size, dtype=np.int64)
    accumulator = None
    excluded = set(str(value) for value in args.exclude_trajectories)
    eligible: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if str(metadata["trajectory_id"]) in excluded:
            continue
        if int(args.maximum_images) > 0 and len(eligible) >= int(args.maximum_images):
            break
        if not bool(metadata.get("uses_declared_clean_2dgs_for_occlusion", False)):
            raise ValueError(f"contributor cache is not declared-clean 2DGS: {path}")
        eligible.append((path, metadata))
    if not eligible:
        raise ValueError("no eligible exact contributor observations")

    teacher_root = Path(args.teacher_cache) if args.teacher_cache else None
    teacher_sum: dict[str, np.ndarray] | None = None
    teacher_view_count: np.ndarray | None = None
    teacher_cache_hash = hashlib.sha256()
    if teacher_root is not None:
        dimensions = int(args.teacher_dimensions)
        if dimensions <= 0:
            raise ValueError("teacher_dimensions must be positive")
        teacher_sum = {
            name: np.zeros((physical.primitive_ids.size, dimensions), dtype=np.float32)
            for name in TEACHER_NAMES
        }
        teacher_view_count = np.zeros((physical.primitive_ids.size,), dtype=np.int32)
        for first_pass_index, (path, metadata) in enumerate(eligible, start=1):
            teacher_path = teacher_root / (str(metadata["image_id"]).replace("/", "__") + ".npz")
            if not teacher_path.exists():
                raise ValueError(f"missing offline teacher cache: {teacher_path}")
            teacher_cache_hash.update(teacher_path.name.encode("utf-8"))
            teacher_cache_hash.update(bytes.fromhex(file_sha256(teacher_path)))
            with np.load(path, allow_pickle=False) as data:
                ids = np.asarray(data["topk_ids"], dtype=np.int64)
                weights = np.asarray(data["topk_weights"], dtype=np.float32)
            with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
                raw_shape = tuple(np.asarray(data["radio_final"]).shape[1:])
            rows, mass, contributing_pixel, starts, token_x, token_y = _pixel_layout(
                ids, weights, raw_shape, dense_lookup,
            )
            unique_rows = rows[starts]
            fields = _teacher_token_fields(
                teacher_path,
                raw_height=int(raw_shape[0]),
                raw_width=int(raw_shape[1]),
                dimensions=dimensions,
            )
            for name in TEACHER_NAMES:
                teacher_sum[name][unique_rows] += _aggregate_view_teacher(
                    fields[name], contributing_pixel, mass, starts, token_x, token_y,
                )
            teacher_view_count[unique_rows] += 1
            print(json.dumps({
                "teacher_prototype_pass": int(first_pass_index),
                "image_id": str(metadata["image_id"]),
                "observed_primitive_count": int(unique_rows.size),
            }), flush=True)

    image_count = 0
    trajectories: set[str] = set()
    geometry_hashes: set[str] = set()
    clean_hashes: set[str] = set()
    quality_values: list[np.ndarray] = []
    teacher_similarity_values: dict[str, list[np.ndarray]] = {name: [] for name in TEACHER_NAMES}
    for path, metadata in eligible:
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"], dtype=np.int64)
            weights = np.asarray(data["topk_weights"], dtype=np.float32)
        trajectory = str(metadata["trajectory_id"])
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        if args.feature_space == "raw_radio_final":
            mapped = raw
        elif args.feature_space == "canonical_codec":
            mapped = codec.transform_map(raw)
        else:
            mapped = mapper.project(raw).measurement_context
        if accumulator is None:
            accumulator = CanonicalFieldFusionAccumulator(physical.primitive_ids.size, int(mapped.shape[0]))
        rows, mass, contributing_pixel, starts, token_x, token_y = _pixel_layout(
            ids, weights, (int(raw.shape[1]), int(raw.shape[2])), dense_lookup,
        )
        pixel_descriptor = mapped[:, token_y, token_x].T.astype(np.float32, copy=False)
        descriptor = pixel_descriptor[contributing_pixel]
        unique_rows = rows[starts]
        view_mass = np.add.reduceat(mass, starts)
        view_sum = np.add.reduceat(mass[:, None] * descriptor, starts, axis=0)
        view_descriptor = view_sum / np.maximum(view_mass[:, None], 1e-8)
        observation_quality = None
        if teacher_root is not None:
            teacher_path = teacher_root / (str(metadata["image_id"]).replace("/", "__") + ".npz")
            fields = _teacher_token_fields(
                teacher_path,
                raw_height=int(raw.shape[1]),
                raw_width=int(raw.shape[2]),
                dimensions=int(args.teacher_dimensions),
            )
            agreement = []
            multiple = teacher_view_count[unique_rows] > 1
            for name in TEACHER_NAMES:
                current = _aggregate_view_teacher(
                    fields[name], contributing_pixel, mass, starts, token_x, token_y,
                )
                other = teacher_sum[name][unique_rows] - current
                other = _normalize_rows(other)
                similarity = np.sum(current * other, axis=1)
                similarity[~multiple] = 1.0
                teacher_similarity_values[name].append(similarity[multiple].astype(np.float32))
                agreement.append(np.clip(0.5 * (similarity + 1.0), 0.0, 1.0))
            combined = 0.40 * agreement[0] + 0.30 * agreement[1] + 0.30 * agreement[2]
            floor = float(args.teacher_quality_floor)
            if not 0.0 < floor <= 1.0:
                raise ValueError("teacher_quality_floor must be in (0, 1]")
            observation_quality = floor + (1.0 - floor) * combined
            observation_quality[~multiple] = 1.0
            quality_values.append(observation_quality[multiple].astype(np.float32))
        accumulator.add_view(
            unique_rows, view_descriptor, view_mass,
            observation_quality=observation_quality,
        )
        image_count += 1
        trajectories.add(trajectory)
        geometry_hashes.add(str(metadata.get("geometry_source_sha256", "")))
        clean_hashes.add(str(metadata.get("clean_source_index_sha256", "")))
        print(json.dumps({
            "image_count": image_count,
            "image_id": str(metadata["image_id"]),
            "observed_primitive_count": int(unique_rows.size),
            "cumulative_observed_primitive_count": int(np.sum(accumulator.weight_sum > 0.0)),
        }), flush=True)
    if accumulator is None or image_count == 0:
        raise ValueError("no eligible exact contributor observations")
    field = accumulator.finalize(
        physical,
        metadata={
            "fusion": (
                "view_balanced_exact_clean_2dgs_topk_contributor_teacher_consistency_weighted"
                if teacher_root is not None
                else "view_balanced_exact_clean_2dgs_topk_contributor_to_radio_final_token"
            ),
            "canonical_feature_space": str(args.feature_space),
            "canonical_feature_dimension": int(accumulator.feature_sum.shape[1]),
            "canonical_codec_sha256": codec.content_sha256 if codec is not None else None,
            "retrieval_mapper_is_regenerable_readout": bool(args.feature_space != "retrieval_mapper"),
            "surface_mapper_file_sha256": file_sha256(Path(args.surface_mapper)),
            "mapping_image_count": int(image_count),
            "mapping_trajectory_ids": sorted(trajectories),
            "contributor_geometry_source_sha256": sorted(geometry_hashes),
            "contributor_clean_source_index_sha256": sorted(clean_hashes),
            "migration": False,
            "offline_teacher_map_quality": (
                {
                    "teacher_names": list(TEACHER_NAMES),
                    "teacher_dimensions_during_mapping": int(args.teacher_dimensions),
                    "teacher_quality_floor": float(args.teacher_quality_floor),
                    "teacher_cache_sha256": teacher_cache_hash.hexdigest(),
                    "teacher_embeddings_discarded": True,
                    "fusion_role_weights": {
                        "dino_v3_7b": 0.40, "sam3": 0.30, "siglip2-g": 0.30,
                    },
                }
                if teacher_root is not None else None
            ),
        },
    )
    field.save_npz(output)
    readout = readout_canonical_field(field, physical)
    report = {
        "stage": "build_goal_maplet_canonical_field_from_exact_contributors",
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "mapping_image_count": image_count,
        "mapping_trajectory_ids": sorted(trajectories),
        "canonical_primitive_count": int(field.primitive_rows.size),
        "feature_dim": int(field.feature_dim),
        "primitive_coverage_fraction": float(field.primitive_rows.size / physical.primitive_ids.size),
        "parent_nonempty_fraction": float(np.mean(readout.parent_coverage > 0.0)),
        "parent_coverage_median": float(np.median(readout.parent_coverage)),
        "child_nonempty_fraction": float(np.mean(readout.child_coverage > 0.0)),
        "child_coverage_median": float(np.median(readout.child_coverage)),
        "storage_contract": dict(field.metadata),
        "output_field": str(output),
        "teacher_quality": (
            {
                "count": int(sum(value.size for value in quality_values)),
                "mean": float(np.mean(np.concatenate(quality_values))) if quality_values else None,
                "p10": float(np.percentile(np.concatenate(quality_values), 10.0)) if quality_values else None,
                "median": float(np.median(np.concatenate(quality_values))) if quality_values else None,
                "teacher_similarity_mean": {
                    name: (
                        float(np.mean(np.concatenate(values))) if values else None
                    ) for name, values in teacher_similarity_values.items()
                },
            }
            if teacher_root is not None else None
        ),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
