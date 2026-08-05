"""Build the canonical field directly from exact clean-2DGS contributors."""

from __future__ import annotations

import argparse
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
    image_count = 0
    trajectories: set[str] = set()
    geometry_hashes: set[str] = set()
    clean_hashes: set[str] = set()
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            ids = np.asarray(data["topk_ids"], dtype=np.int64)
            weights = np.asarray(data["topk_weights"], dtype=np.float32)
        trajectory = str(metadata["trajectory_id"])
        if trajectory in excluded:
            continue
        if int(args.maximum_images) > 0 and image_count >= int(args.maximum_images):
            break
        if not bool(metadata.get("uses_declared_clean_2dgs_for_occlusion", False)):
            raise ValueError(f"contributor cache is not declared-clean 2DGS: {path}")
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
        height, width, topk = ids.shape
        pixel = np.arange(height * width, dtype=np.int64)
        pixel_x, pixel_y = pixel % width, pixel // width
        token_x = np.minimum((pixel_x + 0.5) * raw.shape[2] / width, raw.shape[2] - 1).astype(np.int64)
        token_y = np.minimum((pixel_y + 0.5) * raw.shape[1] / height, raw.shape[1] - 1).astype(np.int64)
        pixel_descriptor = mapped[:, token_y, token_x].T.astype(np.float32, copy=False)
        flat_ids = ids.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_pixel = np.repeat(pixel, topk)
        valid = (flat_ids >= 0) & (flat_ids < dense_lookup.size) & (flat_weights > 0.0)
        flat_rows = np.full(flat_ids.shape, -1, dtype=np.int64)
        flat_rows[valid] = dense_lookup[flat_ids[valid]]
        valid &= flat_rows >= 0
        rows = flat_rows[valid]
        mass = flat_weights[valid]
        descriptor = pixel_descriptor[flat_pixel[valid]]
        order = np.argsort(rows, kind="stable")
        rows, mass, descriptor = rows[order], mass[order], descriptor[order]
        first = np.r_[True, rows[1:] != rows[:-1]]
        starts = np.flatnonzero(first)
        unique_rows = rows[starts]
        view_mass = np.add.reduceat(mass, starts)
        view_sum = np.add.reduceat(mass[:, None] * descriptor, starts, axis=0)
        view_descriptor = view_sum / np.maximum(view_mass[:, None], 1e-8)
        accumulator.add_view(unique_rows, view_descriptor, view_mass)
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
            "fusion": "view_balanced_exact_clean_2dgs_topk_contributor_to_radio_final_token",
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
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
