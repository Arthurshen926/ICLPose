"""Build the anchor-free RADIO-final feature field on a clean 2DGS prior.

Mapping observations are reduced to one feature distribution per retained
surfel.  The output stores no RGB, image path, observation descriptor list, or
stable anchor identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from plyfile import PlyData
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.vfm_2dgs_mapping import (
    Vfm2DgsObservationBank,
    _surface_tangent_axes_and_scales,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--clean_gaussian_ply", required=True)
    parser.add_argument("--observation_bank", required=True)
    parser.add_argument("--maplets", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--minimum_support_weight", type=float, default=1e-5)
    parser.add_argument("--minimum_support_count", type=int, default=2)
    parser.add_argument("--minimum_coherence", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _clean_source_indices(path: Path) -> np.ndarray:
    vertex = PlyData.read(Path(path), mmap=True).elements[0]
    names = set(vertex.data.dtype.names or ())
    if "source_index" not in names:
        raise ValueError("clean 2DGS PLY must retain source_index")
    indices = np.asarray(vertex["source_index"], dtype=np.int64)
    if np.any(indices < 0) or np.unique(indices).size != indices.size:
        raise ValueError("clean source_index must be unique and non-negative")
    if "primitive_class" in names:
        indices = indices[np.asarray(vertex["primitive_class"], dtype=np.int32) == 0]
    return np.sort(indices)


def _aggregate_features(
    bank: Vfm2DgsObservationBank,
    clean_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each token footprint to its dominant clean surfel, then fuse.

    Scattering a token descriptor to all 64 footprint elements makes adjacent
    surfels nearly identical and destroys the spatial gradient needed by pose
    alignment.  A token is therefore a soft surface measurement but contributes
    its descriptor only to the clean surfel with maximum responsibility.
    """

    offsets = np.asarray(bank.support_offsets, dtype=np.int64)
    link_ids = np.asarray(bank.element_ids, dtype=np.int64)
    link_weights = np.asarray(bank.element_weights, dtype=np.float32)
    lookup_size = int(max(np.max(clean_indices, initial=-1), np.max(link_ids, initial=-1))) + 1
    clean_row = np.full((lookup_size,), -1, dtype=np.int64)
    clean_row[clean_indices] = np.arange(clean_indices.size, dtype=np.int64)
    assigned_clean_rows: list[int] = []
    assigned_observation_rows: list[int] = []
    assigned_responsibility: list[float] = []
    for observation_row in range(len(bank)):
        start = int(offsets[observation_row])
        end = int(offsets[observation_row + 1])
        ids = link_ids[start:end]
        valid = (ids >= 0) & (ids < lookup_size)
        local_rows = np.full(ids.shape, -1, dtype=np.int64)
        local_rows[valid] = clean_row[ids[valid]]
        valid_rows = np.flatnonzero(local_rows >= 0)
        if valid_rows.size == 0:
            continue
        selected = int(
            valid_rows[
                np.argmax(link_weights[start:end][valid_rows])
            ]
        )
        assigned_clean_rows.append(int(local_rows[selected]))
        assigned_observation_rows.append(int(observation_row))
        assigned_responsibility.append(float(link_weights[start + selected]))

    quality = np.asarray(bank.quality_scores, dtype=np.float32)
    descriptor_weight = np.asarray(
        getattr(bank, "descriptor_weights", np.ones((len(bank),), dtype=np.float32)),
        dtype=np.float32,
    )
    observation_rows = np.asarray(assigned_observation_rows, dtype=np.int64)
    rows = np.asarray(assigned_clean_rows, dtype=np.int64)
    weights = np.asarray(assigned_responsibility, dtype=np.float32)
    weights *= np.maximum(quality[observation_rows], 0.0)
    weights *= np.maximum(descriptor_weight[observation_rows], 0.0)
    matrix = coo_matrix(
        (weights, (rows, observation_rows)),
        shape=(clean_indices.size, len(bank)),
        dtype=np.float32,
    ).tocsr()
    feature_sum = matrix @ np.asarray(bank.features, dtype=np.float32)
    support_weight = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    support_count = np.asarray(matrix.getnnz(axis=1), dtype=np.int32)
    mean = feature_sum / np.maximum(support_weight[:, None], 1e-8)
    coherence = np.linalg.norm(mean, axis=1).astype(np.float32)
    mean /= np.maximum(coherence[:, None], 1e-8)
    return mean, coherence, support_weight, support_count


def _maplet_owners(
    source_indices: np.ndarray,
    centers: np.ndarray,
    maplets: VfmSurfaceMapletBank,
) -> np.ndarray:
    """Assign a region label for rendering, without creating point identity."""

    owner = np.full((source_indices.size,), -1, dtype=np.int64)
    row_by_source = {int(value): row for row, value in enumerate(source_indices.tolist())}
    best_distance = np.full((source_indices.size,), np.inf, dtype=np.float32)
    for maplet_row, maplet_id in enumerate(maplets.maplet_ids.tolist()):
        start = int(maplets.support_offsets[maplet_row])
        end = int(maplets.support_offsets[maplet_row + 1])
        for source_id in maplets.support_element_ids[start:end].tolist():
            row = row_by_source.get(int(source_id))
            if row is None:
                continue
            distance = float(np.linalg.norm(centers[row] - maplets.centers[maplet_row]))
            if distance < float(best_distance[row]):
                best_distance[row] = distance
                owner[row] = int(maplet_id)
    missing = np.flatnonzero(owner < 0)
    if missing.size:
        nearest = cKDTree(np.asarray(maplets.centers, dtype=np.float64)).query(
            centers[missing], k=1
        )[1]
        owner[missing] = maplets.maplet_ids[np.asarray(nearest, dtype=np.int64)]
    return owner


def build_surface_feature_field(
    *,
    gaussian_ply: Path,
    clean_gaussian_ply: Path,
    observation_bank: Path,
    maplet_path: Path,
    minimum_support_weight: float,
    minimum_support_count: int,
    minimum_coherence: float,
) -> tuple[SurfaceFeatureField, dict[str, object]]:
    clean_indices = _clean_source_indices(clean_gaussian_ply)
    source = load_gaussian_vfm_source_from_ply(gaussian_ply)
    if np.max(clean_indices, initial=-1) >= source.xyz.shape[0]:
        raise ValueError("clean source_index exceeds full 2DGS primitive count")
    bank = Vfm2DgsObservationBank.load_npz(observation_bank)
    maplets = VfmSurfaceMapletBank.load_npz(maplet_path)
    features, coherence, support_weight, support_count = _aggregate_features(
        bank, clean_indices
    )
    retained = (
        (support_weight >= float(minimum_support_weight))
        & (support_count >= int(minimum_support_count))
        & (coherence >= float(minimum_coherence))
    )
    rows = clean_indices[retained]
    normals = np.asarray(source.normal, dtype=np.float32)[rows]
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
        source, rows, normals
    )
    centers = np.asarray(source.xyz, dtype=np.float64)[rows]
    owner = _maplet_owners(rows, centers, maplets)
    retained_weight = support_weight[retained]
    weight_scale = float(np.quantile(retained_weight, 0.90)) if rows.size else 1.0
    confidence = (
        np.clip(retained_weight / max(weight_scale, 1e-8), 0.0, 1.0)
        * coherence[retained]
        * np.asarray(source.opacity, dtype=np.float32)[rows]
    )
    metadata = {
        "artifact_type": "radio_final_2dgs_surface_feature_field",
        "vfm_layer": "radio_final",
        "representation": "one_distribution_per_clean_2dgs_surfel",
        "observation_assignment": "dominant_clean_surfel_responsibility",
        "feature_dim": int(features.shape[1]),
        "clean_primitive_count": int(clean_indices.size),
        "field_surfel_count": int(rows.size),
        "minimum_support_weight": float(minimum_support_weight),
        "minimum_support_count": int(minimum_support_count),
        "minimum_coherence": float(minimum_coherence),
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "uses_mapping_rgb_at_inference": False,
        "uses_pairwise_image_matching": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_stable_anchor_identity": False,
    }
    field = SurfaceFeatureField(
        source_indices=rows,
        centers=centers,
        normals=normals,
        tangent1=tangent1,
        tangent2=tangent2,
        scale1=scale1,
        scale2=scale2,
        opacity=np.asarray(source.opacity, dtype=np.float32)[rows],
        features=features[retained],
        uncertainty=np.clip(1.0 - coherence[retained], 0.0, 1.0),
        confidence=confidence,
        support_weight=retained_weight,
        support_count=support_count[retained],
        owner_maplet_ids=owner,
        metadata=metadata,
    )
    summary = {
        "stage": "build_anchor_free_2dgs_surface_feature_field",
        "field_surfel_count": len(field),
        "clean_primitive_count": int(clean_indices.size),
        "observed_clean_primitive_count": int(np.sum(support_count > 0)),
        "feature_dim": field.feature_dim,
        "coherence": {
            "median": float(np.median(coherence[retained])),
            "p10": float(np.quantile(coherence[retained], 0.10)),
            "p90": float(np.quantile(coherence[retained], 0.90)),
        },
        "contract": metadata,
    }
    return field, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_field)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite surface feature field outputs")
    field, summary = build_surface_feature_field(
        gaussian_ply=Path(args.gaussian_ply),
        clean_gaussian_ply=Path(args.clean_gaussian_ply),
        observation_bank=Path(args.observation_bank),
        maplet_path=Path(args.maplets),
        minimum_support_weight=float(args.minimum_support_weight),
        minimum_support_count=int(args.minimum_support_count),
        minimum_coherence=float(args.minimum_coherence),
    )
    field.save_npz(output)
    summary["output_field"] = str(output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
