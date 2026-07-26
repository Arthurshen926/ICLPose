"""Merge feature-aligned anchors with quality-ranked stable-anchor coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.surface_maplet_bank import (
    StableSurfaceAnchorMap,
    VfmSurfaceMapletBank,
    load_2dgs_primitive_quality,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_maplets", required=True)
    parser.add_argument("--base_anchors", required=True)
    parser.add_argument("--base_local_descriptor_bank", required=True)
    parser.add_argument("--feature_maplets", required=True)
    parser.add_argument("--feature_anchors", required=True)
    parser.add_argument("--feature_local_descriptor_bank", required=True)
    parser.add_argument("--clean_gaussian_ply", default="")
    parser.add_argument("--maximum_anchors_per_maplet", type=int, default=64)
    parser.add_argument(
        "--base_fill_target_per_maplet",
        type=int,
        default=64,
        help=(
            "Fill geometry-first anchors only until this target is reached; "
            "feature-aligned anchors above the target are retained."
        ),
    )
    parser.add_argument("--minimum_feature_anchor_separation_m", type=float, default=0.02)
    parser.add_argument("--output_maplets", required=True)
    parser.add_argument("--output_anchors", required=True)
    parser.add_argument("--output_local_descriptor_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def _descriptor_rows(
    bank: AnchorLocalDescriptorBank, bank_row: int
) -> np.ndarray:
    return np.arange(
        int(bank.descriptor_offsets[bank_row]),
        int(bank.descriptor_offsets[bank_row + 1]),
        dtype=np.int64,
    )


def _base_localization_quality(
    anchor_ids: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    bank: AnchorLocalDescriptorBank,
) -> dict[int, float]:
    anchor_row_by_id = anchors.row_by_id()
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(bank.anchor_ids.tolist())
    }
    valid_ids = [
        int(anchor_id)
        for anchor_id in np.asarray(anchor_ids, dtype=np.int64).tolist()
        if int(anchor_id) in anchor_row_by_id
        and int(anchor_id) in bank_row_by_id
    ]
    if not valid_ids:
        return {}
    prototypes = []
    consistency = []
    repeatability = []
    detector_quality = []
    geometry = []
    for anchor_id in valid_ids:
        rows = _descriptor_rows(bank, bank_row_by_id[anchor_id])
        values = bank.descriptors[rows]
        weights = np.maximum(bank.descriptor_quality[rows], 1e-8)
        prototype = np.sum(values * weights[:, None], axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        prototypes.append(prototype)
        consistency.append(float(np.median(values @ prototype)))
        repeatability.append(float(min(len(rows) / 4.0, 1.0)))
        detector_quality.append(float(np.median(weights)))
        geometry.append(
            float(anchors.quality_scores[anchor_row_by_id[anchor_id]])
        )
    prototype_matrix = np.stack(prototypes, axis=0)
    similarity = prototype_matrix @ prototype_matrix.T
    np.fill_diagonal(similarity, -np.inf)
    nearest_negative = (
        np.max(similarity, axis=1)
        if len(valid_ids) > 1
        else np.zeros((1,), dtype=np.float32)
    )
    distinctiveness = np.clip((1.0 - nearest_negative) / 0.30, 0.02, 1.0)

    def robust_unit(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        low, high = np.quantile(array, [0.1, 0.9])
        return np.clip((array - low) / max(float(high - low), 1e-12), 0.0, 1.0)

    geometry_unit = robust_unit(geometry)
    detector_unit = robust_unit(np.log1p(np.asarray(detector_quality) * 1000.0))
    consistency_unit = np.clip(
        (np.asarray(consistency) - 0.5) / 0.5, 0.0, 1.0
    )
    quality = (
        np.maximum(geometry_unit, 0.05)
        * np.maximum(detector_unit, 0.05)
        * np.maximum(consistency_unit, 0.05)
        * np.maximum(np.asarray(repeatability), 0.05)
        * distinctiveness
    )
    return {
        anchor_id: float(value)
        for anchor_id, value in zip(valid_ids, quality.tolist())
    }


def _append_anchor(
    *,
    source_row: int,
    source_anchors: StableSurfaceAnchorMap,
    source_bank: AnchorLocalDescriptorBank,
    source_bank_row: int,
    output: dict[str, list],
    quality_override: float | None,
) -> None:
    output["anchor_ids"].append(int(source_anchors.anchor_ids[source_row]))
    output["owner_ids"].append(int(source_anchors.owner_maplet_ids[source_row]))
    output["surface_ids"].append(int(source_anchors.surface_element_ids[source_row]))
    output["parent_ids"].append(int(source_anchors.parent_primitive_indices[source_row]))
    output["xyz"].append(source_anchors.xyz[source_row])
    output["normals"].append(source_anchors.normals[source_row])
    output["covariances"].append(source_anchors.tangent_covariances[source_row])
    output["radii"].append(float(source_anchors.support_radii[source_row]))
    output["qualities"].append(
        float(source_anchors.quality_scores[source_row])
        if quality_override is None
        else float(quality_override)
    )
    output["geometry"].append(float(source_anchors.geometry_confidence[source_row]))
    output["opacity"].append(float(source_anchors.opacity[source_row]))
    start = int(source_anchors.observation_offsets[source_row])
    end = int(source_anchors.observation_offsets[source_row + 1])
    output["observation_image_ids"].extend(
        source_anchors.observation_image_ids[start:end]
    )
    output["observation_xy"].extend(source_anchors.observation_xy[start:end])
    output["observation_depth"].extend(source_anchors.observation_depth[start:end].tolist())
    output["observation_weights"].extend(source_anchors.observation_weights[start:end].tolist())
    output["observation_offsets"].append(len(output["observation_image_ids"]))
    rows = _descriptor_rows(source_bank, int(source_bank_row))
    output["descriptors"].extend(source_bank.descriptors[rows])
    output["descriptor_image_ids"].extend(
        source_bank.support_image_ids[int(rows[0]) : int(rows[-1]) + 1]
    )
    output["descriptor_quality"].extend(
        source_bank.descriptor_quality[rows].tolist()
    )
    output["descriptor_offsets"].append(len(output["descriptors"]))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    base_maplets = VfmSurfaceMapletBank.load_npz(Path(args.base_maplets))
    base_anchors = StableSurfaceAnchorMap.load_npz(Path(args.base_anchors))
    base_bank = AnchorLocalDescriptorBank.load_npz(Path(args.base_local_descriptor_bank))
    feature_maplets = VfmSurfaceMapletBank.load_npz(Path(args.feature_maplets))
    feature_anchors = StableSurfaceAnchorMap.load_npz(Path(args.feature_anchors))
    feature_bank = AnchorLocalDescriptorBank.load_npz(
        Path(args.feature_local_descriptor_bank)
    )
    base_anchor_rows = base_anchors.row_by_id()
    feature_anchor_rows = feature_anchors.row_by_id()
    base_bank_rows = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(base_bank.anchor_ids.tolist())
    }
    feature_bank_rows = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(feature_bank.anchor_ids.tolist())
    }
    feature_maplet_rows = {
        int(maplet_id): int(row)
        for row, maplet_id in enumerate(feature_maplets.maplet_ids.tolist())
    }
    clean_keep = np.ones((len(base_anchors),), dtype=bool)
    clean_metadata = None
    if str(args.clean_gaussian_ply):
        quality = load_2dgs_primitive_quality(Path(args.clean_gaussian_ply))
        parent = base_anchors.parent_primitive_indices
        clean_keep = (
            (parent >= 0)
            & (parent < len(quality))
            & (
                quality.geometry_confidence[
                    np.clip(parent, 0, len(quality) - 1)
                ]
                > 0.0
            )
        )
        clean_metadata = dict(quality.metadata or {})
    output: dict[str, list] = {
        "anchor_ids": [],
        "owner_ids": [],
        "surface_ids": [],
        "parent_ids": [],
        "xyz": [],
        "normals": [],
        "covariances": [],
        "radii": [],
        "qualities": [],
        "geometry": [],
        "opacity": [],
        "observation_offsets": [0],
        "observation_image_ids": [],
        "observation_xy": [],
        "observation_depth": [],
        "observation_weights": [],
        "descriptor_offsets": [0],
        "descriptors": [],
        "descriptor_image_ids": [],
        "descriptor_quality": [],
    }
    flattened_ids: list[int] = []
    anchor_offsets = [0]
    feature_count = 0
    base_fill_count = 0
    for base_maplet_row, maplet_id_value in enumerate(
        base_maplets.maplet_ids.tolist()
    ):
        maplet_id = int(maplet_id_value)
        selected_ids: list[int] = []
        selected_xyz: list[np.ndarray] = []
        feature_row = feature_maplet_rows.get(maplet_id)
        if feature_row is not None:
            start = int(feature_maplets.anchor_offsets[feature_row])
            end = int(feature_maplets.anchor_offsets[feature_row + 1])
            for anchor_id_value in feature_maplets.anchor_ids[start:end].tolist():
                anchor_id = int(anchor_id_value)
                source_row = feature_anchor_rows[anchor_id]
                if int(feature_anchors.owner_maplet_ids[source_row]) != maplet_id:
                    continue
                _append_anchor(
                    source_row=source_row,
                    source_anchors=feature_anchors,
                    source_bank=feature_bank,
                    source_bank_row=feature_bank_rows[anchor_id],
                    output=output,
                    quality_override=None,
                )
                selected_ids.append(anchor_id)
                selected_xyz.append(feature_anchors.xyz[source_row])
                feature_count += 1
        start = int(base_maplets.anchor_offsets[base_maplet_row])
        end = int(base_maplets.anchor_offsets[base_maplet_row + 1])
        base_ids = base_maplets.anchor_ids[start:end]
        quality_by_id = _base_localization_quality(
            base_ids, base_anchors, base_bank
        )
        ranked_base = sorted(
            quality_by_id,
            key=lambda anchor_id: (-quality_by_id[anchor_id], anchor_id),
        )
        fill_target = min(
            int(args.maximum_anchors_per_maplet),
            max(
                len(selected_ids),
                int(args.base_fill_target_per_maplet),
            ),
        )
        for anchor_id in ranked_base:
            if len(selected_ids) >= fill_target:
                break
            source_row = base_anchor_rows[anchor_id]
            if int(base_anchors.owner_maplet_ids[source_row]) != maplet_id:
                continue
            if not clean_keep[source_row]:
                continue
            if selected_xyz:
                distance = np.linalg.norm(
                    np.stack(selected_xyz, axis=0)
                    - base_anchors.xyz[source_row],
                    axis=1,
                )
                if float(np.min(distance)) < float(
                    args.minimum_feature_anchor_separation_m
                ):
                    continue
            _append_anchor(
                source_row=source_row,
                source_anchors=base_anchors,
                source_bank=base_bank,
                source_bank_row=base_bank_rows[anchor_id],
                output=output,
                quality_override=float(quality_by_id[anchor_id]) * 0.25,
            )
            selected_ids.append(anchor_id)
            selected_xyz.append(base_anchors.xyz[source_row])
            base_fill_count += 1
        flattened_ids.extend(selected_ids)
        anchor_offsets.append(len(flattened_ids))
    stable = StableSurfaceAnchorMap(
        anchor_ids=np.asarray(output["anchor_ids"], dtype=np.int64),
        owner_maplet_ids=np.asarray(output["owner_ids"], dtype=np.int64),
        surface_element_ids=np.asarray(output["surface_ids"], dtype=np.int64),
        parent_primitive_indices=np.asarray(output["parent_ids"], dtype=np.int64),
        xyz=np.asarray(output["xyz"], dtype=np.float64).reshape(-1, 3),
        normals=np.asarray(output["normals"], dtype=np.float32).reshape(-1, 3),
        tangent_covariances=np.asarray(output["covariances"], dtype=np.float32).reshape(-1, 3, 3),
        support_radii=np.asarray(output["radii"], dtype=np.float32),
        quality_scores=np.asarray(output["qualities"], dtype=np.float32),
        geometry_confidence=np.asarray(output["geometry"], dtype=np.float32),
        opacity=np.asarray(output["opacity"], dtype=np.float32),
        observation_offsets=np.asarray(output["observation_offsets"], dtype=np.int64),
        observation_image_ids=tuple(output["observation_image_ids"]),
        observation_xy=np.asarray(output["observation_xy"], dtype=np.float32).reshape(-1, 2),
        observation_depth=np.asarray(output["observation_depth"], dtype=np.float32),
        observation_weights=np.asarray(output["observation_weights"], dtype=np.float32),
        metadata={
            "representation": "hybrid_feature_first_2dgs_surface_anchor",
            "uses_mapping_rgb_at_inference": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    descriptor_dim = int(base_bank.feature_dim)
    bank = AnchorLocalDescriptorBank(
        anchor_ids=np.asarray(output["anchor_ids"], dtype=np.int64),
        descriptor_offsets=np.asarray(output["descriptor_offsets"], dtype=np.int64),
        descriptors=np.asarray(output["descriptors"], dtype=np.float32).reshape(-1, descriptor_dim),
        support_image_ids=tuple(output["descriptor_image_ids"]),
        descriptor_quality=np.asarray(output["descriptor_quality"], dtype=np.float32),
        metadata={
            "representation": "hybrid_feature_first_surface_anchor_descriptors",
            "uses_mapping_rgb_at_inference": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    )
    maplets = VfmSurfaceMapletBank(
        maplet_ids=base_maplets.maplet_ids,
        centers=base_maplets.centers,
        normals=base_maplets.normals,
        tangent_frames=base_maplets.tangent_frames,
        extents=base_maplets.extents,
        descriptors=base_maplets.descriptors,
        quality_scores=base_maplets.quality_scores,
        descriptor_variances=base_maplets.descriptor_variances,
        anchor_offsets=np.asarray(anchor_offsets, dtype=np.int64),
        anchor_ids=np.asarray(flattened_ids, dtype=np.int64),
        support_offsets=base_maplets.support_offsets,
        support_element_ids=base_maplets.support_element_ids,
        view_offsets=base_maplets.view_offsets,
        view_image_ids=base_maplets.view_image_ids,
        view_token_xy=base_maplets.view_token_xy,
        view_grid_sizes=base_maplets.view_grid_sizes,
        view_descriptors=base_maplets.view_descriptors,
        view_quality_scores=base_maplets.view_quality_scores,
        metadata={
            **dict(base_maplets.metadata or {}),
            "anchor_representation": "hybrid_feature_first_2dgs_surface_anchor",
        },
    )
    maplets.save_npz(Path(args.output_maplets))
    stable.save_npz(Path(args.output_anchors))
    bank.save_npz(Path(args.output_local_descriptor_bank))
    counts = np.diff(maplets.anchor_offsets)
    summary = {
        "stage": "build_hybrid_feature_surface_map",
        "maplet_count": int(len(maplets)),
        "anchor_count": int(len(stable)),
        "feature_aligned_anchor_count": int(feature_count),
        "quality_ranked_base_fill_anchor_count": int(base_fill_count),
        "base_fill_target_per_maplet": int(
            args.base_fill_target_per_maplet
        ),
        "anchors_per_maplet": {
            "min": int(np.min(counts)),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "max": int(np.max(counts)),
        },
        "clean_prior": clean_metadata,
        "production_contract": {
            "stores_mapping_rgb": False,
            "uses_mapping_rgb_at_inference": False,
            "uses_pairwise_image_matching": False,
            "uses_loftr": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_radio_intermediate": False,
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
