"""Compact anchor-free RADIO-final maplets used only for region retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from feature_extract.vfm.localization_v6.maplet_atlas import (
        MapletFeatureAtlasBank,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class SurfaceRetrievalMapletBank:
    """A bounded RADIO-final mixture and metric region per maplet.

    Maplets are regional retrieval units, not stable point identities. The
    mixture has no component/view identity. The artifact deliberately has no
    per-view descriptors, image identifiers,
    anchor identifiers, observations, or point correspondences.
    """

    maplet_ids: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    extents: np.ndarray
    descriptor_offsets: np.ndarray
    descriptors: np.ndarray
    descriptor_weights: np.ndarray
    quality_scores: np.ndarray
    descriptor_uncertainties: np.ndarray
    descriptor_centers: np.ndarray | None = None
    descriptor_covariances: np.ndarray | None = None
    query_projection: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        count = int(ids.size)
        if len(np.unique(ids)) != count:
            raise ValueError("maplet_ids must be unique")
        object.__setattr__(self, "maplet_ids", ids)
        for name in ("centers", "normals", "extents"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _normalize_rows(self.normals))
        offsets = np.asarray(self.descriptor_offsets, dtype=np.int64).reshape(-1)
        descriptors = _normalize_rows(self.descriptors)
        weights = np.asarray(self.descriptor_weights, dtype=np.float32).reshape(-1)
        if (
            offsets.shape != (count + 1,)
            or offsets[0] != 0
            or offsets[-1] != descriptors.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("descriptor_offsets must define at least one component per maplet")
        if weights.shape != (descriptors.shape[0],) or np.any(weights < 0.0):
            raise ValueError("descriptor_weights must be non-negative per component")
        normalized_weights = weights.copy()
        for row in range(count):
            begin, end = int(offsets[row]), int(offsets[row + 1])
            total = float(np.sum(normalized_weights[begin:end]))
            if total <= 0.0:
                raise ValueError("each maplet mixture must have positive total weight")
            normalized_weights[begin:end] /= total
        object.__setattr__(self, "descriptor_offsets", offsets)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "descriptor_weights", normalized_weights)
        component_maplet_rows = np.repeat(
            np.arange(count, dtype=np.int64), np.diff(offsets)
        )
        if self.descriptor_centers is None:
            descriptor_centers = np.asarray(
                self.centers[component_maplet_rows], dtype=np.float32
            )
        else:
            descriptor_centers = np.asarray(
                self.descriptor_centers, dtype=np.float32
            )
        if descriptor_centers.shape != (descriptors.shape[0], 3):
            raise ValueError(
                "descriptor_centers must have shape (component_count,3)"
            )
        if self.descriptor_covariances is None:
            radius = np.max(
                np.asarray(self.extents[component_maplet_rows], dtype=np.float32),
                axis=1,
            )
            descriptor_covariances = (
                np.eye(3, dtype=np.float32)[None]
                * np.maximum(radius, 1e-3)[:, None, None] ** 2
            )
        else:
            descriptor_covariances = np.asarray(
                self.descriptor_covariances, dtype=np.float32
            )
        if descriptor_covariances.shape != (descriptors.shape[0], 3, 3):
            raise ValueError(
                "descriptor_covariances must have shape "
                "(component_count,3,3)"
            )
        descriptor_covariances = 0.5 * (
            descriptor_covariances
            + np.swapaxes(descriptor_covariances, 1, 2)
        )
        if (
            not np.all(np.isfinite(descriptor_centers))
            or not np.all(np.isfinite(descriptor_covariances))
            or np.min(
                np.linalg.eigvalsh(
                    descriptor_covariances.astype(np.float64)
                )
            )
            < -1e-5
        ):
            raise ValueError("invalid descriptor-conditioned surface moments")
        object.__setattr__(self, "descriptor_centers", descriptor_centers)
        object.__setattr__(
            self, "descriptor_covariances", descriptor_covariances
        )
        if self.query_projection is not None:
            query_projection = np.asarray(
                self.query_projection, dtype=np.float32
            )
            if (
                query_projection.ndim != 2
                or query_projection.shape[0] != descriptors.shape[1]
                or not np.all(np.isfinite(query_projection))
            ):
                raise ValueError(
                    "query_projection must have shape "
                    "(descriptor_dim,input_dim)"
                )
            object.__setattr__(self, "query_projection", query_projection)
        for name in ("quality_scores", "descriptor_uncertainties"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata or {})
        for key in (
            "stores_mapping_rgb",
            "stores_mapping_image_paths",
            "stores_mapping_image_ids",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_stable_anchor_identity",
            "uses_point_correspondences",
        ):
            if bool(metadata.get(key, False)):
                raise ValueError(f"retrieval maplet bank violates contract: {key}")
        if metadata.get("vfm_layer", "radio_final") != "radio_final":
            raise ValueError("retrieval maplets must use RADIO-final features")
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        return int(self.maplet_ids.size)

    def subset_maplets(
        self, retained_maplet_ids: np.ndarray
    ) -> "SurfaceRetrievalMapletBank":
        """Return an order-preserving bank restricted to declared identities."""

        retained_ids = set(
            np.asarray(retained_maplet_ids, dtype=np.int64).reshape(-1).tolist()
        )
        rows = np.asarray(
            [
                row
                for row, maplet_id in enumerate(self.maplet_ids.tolist())
                if int(maplet_id) in retained_ids
            ],
            dtype=np.int64,
        )
        if rows.size == 0:
            raise ValueError("maplet subset is empty")
        component_rows = []
        offsets = [0]
        for row in rows.tolist():
            begin, end = (
                int(self.descriptor_offsets[row]),
                int(self.descriptor_offsets[row + 1]),
            )
            component_rows.append(np.arange(begin, end, dtype=np.int64))
            offsets.append(offsets[-1] + end - begin)
        components = np.concatenate(component_rows)
        metadata = dict(self.metadata or {})
        metadata["subset_of_maplet_count"] = len(self)
        return SurfaceRetrievalMapletBank(
            maplet_ids=self.maplet_ids[rows],
            centers=self.centers[rows],
            normals=self.normals[rows],
            extents=self.extents[rows],
            descriptor_offsets=np.asarray(offsets, dtype=np.int64),
            descriptors=self.descriptors[components],
            descriptor_weights=self.descriptor_weights[components],
            descriptor_centers=self.descriptor_centers[components],
            descriptor_covariances=self.descriptor_covariances[components],
            query_projection=self.query_projection,
            quality_scores=self.quality_scores[rows],
            descriptor_uncertainties=self.descriptor_uncertainties[rows],
            metadata=metadata,
        )

    def project_query_feature_map(self, feature_map: np.ndarray) -> np.ndarray:
        if self.query_projection is None:
            raise ValueError("retrieval bank has no embedded query projection")
        feature = np.asarray(feature_map, dtype=np.float32)
        if (
            feature.ndim != 3
            or feature.shape[0] != self.query_projection.shape[1]
        ):
            raise ValueError("query feature map differs from embedded projection")
        projected = np.einsum(
            "oc,chw->ohw",
            self.query_projection,
            feature,
            optimize=True,
        )
        norm = np.linalg.norm(projected, axis=0, keepdims=True)
        return projected / np.maximum(norm, 1e-8)

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            maplet_ids=self.maplet_ids,
            centers=self.centers,
            normals=self.normals,
            extents=self.extents,
            descriptor_offsets=self.descriptor_offsets,
            descriptors=self.descriptors,
            descriptor_weights=self.descriptor_weights,
            descriptor_centers=self.descriptor_centers,
            descriptor_covariances=self.descriptor_covariances,
            quality_scores=self.quality_scores,
            descriptor_uncertainties=self.descriptor_uncertainties,
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )
        if self.query_projection is not None:
            payload["query_projection"] = self.query_projection
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load_npz(cls, path: Path) -> "SurfaceRetrievalMapletBank":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {
                "maplet_ids",
                "centers",
                "normals",
                "extents",
                "descriptor_offsets",
                "descriptors",
                "descriptor_weights",
                "quality_scores",
                "descriptor_uncertainties",
                "metadata_json",
            }
            spatial_fields = {
                "descriptor_centers",
                "descriptor_covariances",
            }
            present_spatial = spatial_fields & set(data.files)
            if present_spatial and present_spatial != spatial_fields:
                raise ValueError(
                    "descriptor surface moments must be stored together"
                )
            optional_fields = spatial_fields | {"query_projection"}
            extra = set(data.files) - expected - optional_fields
            missing = expected - set(data.files)
            if extra or missing:
                raise ValueError(
                    f"non-canonical retrieval maplet artifact; extra={sorted(extra)}, "
                    f"missing={sorted(missing)}"
                )
            return cls(
                maplet_ids=data["maplet_ids"],
                centers=data["centers"],
                normals=data["normals"],
                extents=data["extents"],
                descriptor_offsets=data["descriptor_offsets"],
                descriptors=data["descriptors"],
                descriptor_weights=data["descriptor_weights"],
                descriptor_centers=(
                    data["descriptor_centers"]
                    if "descriptor_centers" in data
                    else None
                ),
                descriptor_covariances=(
                    data["descriptor_covariances"]
                    if "descriptor_covariances" in data
                    else None
                ),
                query_projection=(
                    data["query_projection"]
                    if "query_projection" in data
                    else None
                ),
                quality_scores=data["quality_scores"],
                descriptor_uncertainties=data["descriptor_uncertainties"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def package_surface_feature_atlas(
    atlas: "MapletFeatureAtlasBank",
    *,
    spatial_stride: int,
    feature_space: str,
    feature_space_sha256: str,
    representation: str,
    query_projection: np.ndarray | None = None,
    contributor_lineage_verified: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> SurfaceRetrievalMapletBank:
    """Package real canonical atlas cells as bounded surface modes.

    This is shared by RADIO-only retrieval textures and the RGB/RADIO metric
    surface bank.  Missing maplets are omitted instead of receiving synthetic
    zero descriptors.
    """

    stride = int(spatial_stride)
    if stride < 1:
        raise ValueError("spatial_stride must be positive")
    descriptor_rows = []
    weight_rows = []
    center_rows = []
    covariance_rows = []
    offsets = [0]
    retained_maplet_rows = []
    quality_rows = []
    uncertainty_rows = []
    for maplet_row in range(len(atlas)):
        valid = np.asarray(atlas.valid_mask[maplet_row], dtype=bool)
        sample_mask = np.zeros_like(valid)
        sample_mask[::stride, ::stride] = True
        rows_y, rows_x = np.nonzero(valid & sample_mask)
        if rows_y.size == 0:
            rows_y, rows_x = np.nonzero(valid)
        if rows_y.size == 0:
            continue
        retained_maplet_rows.append(maplet_row)
        local_descriptors = []
        local_weights = []
        local_centers = []
        local_covariances = []
        frame = np.asarray(atlas.frames[maplet_row], dtype=np.float64)
        half_u = (
            float(atlas.extents[maplet_row, 0])
            * stride
            / max(atlas.width, 1)
        )
        half_v = (
            float(atlas.extents[maplet_row, 1])
            * stride
            / max(atlas.height, 1)
        )
        covariance = (
            np.outer(frame[0], frame[0]) * half_u**2 / 3.0
            + np.outer(frame[1], frame[1]) * half_v**2 / 3.0
            + np.outer(frame[2], frame[2]) * 0.005**2
        ).astype(np.float32)
        for y, x in zip(rows_y.tolist(), rows_x.tolist()):
            if atlas.mode_features is None:
                modes = atlas.features[maplet_row, :, y, x][None]
                mode_weights = np.ones((1,), dtype=np.float32)
            else:
                keep = atlas.mode_valid_mask[maplet_row, :, y, x]
                modes = atlas.mode_features[maplet_row, keep, :, y, x]
                mode_weights = atlas.mode_weights[maplet_row, keep, y, x]
            support_weight = max(
                float(atlas.support_count[maplet_row, y, x]), 1.0
            )
            for mode, mode_weight in zip(modes, mode_weights):
                local_descriptors.append(mode)
                local_weights.append(
                    support_weight * max(float(mode_weight), 1e-6)
                )
                local_centers.append(atlas.xyz[maplet_row, y, x])
                local_covariances.append(covariance)
        descriptor_rows.append(
            np.asarray(local_descriptors, dtype=np.float32)
        )
        weight_rows.append(np.asarray(local_weights, dtype=np.float32))
        center_rows.append(np.asarray(local_centers, dtype=np.float32))
        covariance_rows.append(
            np.asarray(local_covariances, dtype=np.float32)
        )
        offsets.append(offsets[-1] + len(local_descriptors))
        mean_support = float(
            np.mean(atlas.support_count[maplet_row][valid])
        )
        quality_rows.append(mean_support / (mean_support + 2.0))
        uncertainty_rows.append(
            float(np.median(atlas.variance[maplet_row][valid]))
        )
    if not retained_maplet_rows:
        raise ValueError("surface feature atlas has no observed maplets")
    retained = np.asarray(retained_maplet_rows, dtype=np.int64)
    descriptors = np.concatenate(descriptor_rows, axis=0)
    descriptors /= np.maximum(
        np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8
    )
    weights = np.concatenate(weight_rows)
    weights /= max(float(np.mean(weights)), 1e-8)
    contract = {
        "artifact_type": "anchor_free_surface_retrieval_maplets",
        "representation": str(representation),
        "feature_space": str(feature_space),
        "vfm_layer": "radio_final",
        "query_feature_transform": str(feature_space),
        "query_feature_transform_sha256": str(feature_space_sha256),
        "spatial_stride": stride,
        "appearance_mode_count": atlas.appearance_mode_count,
        "contributor_geometry_lineage_verified": bool(
            contributor_lineage_verified
        ),
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_stable_anchor_identity": False,
        "uses_point_correspondences": False,
    }
    contract.update(dict(metadata or {}))
    return SurfaceRetrievalMapletBank(
        maplet_ids=atlas.maplet_ids[retained],
        centers=atlas.centers[retained],
        normals=atlas.frames[retained, 2],
        extents=atlas.extents[retained],
        descriptor_offsets=np.asarray(offsets, dtype=np.int64),
        descriptors=descriptors,
        descriptor_weights=weights,
        descriptor_centers=np.concatenate(center_rows, axis=0),
        descriptor_covariances=np.concatenate(covariance_rows, axis=0),
        query_projection=query_projection,
        quality_scores=np.clip(
            np.asarray(quality_rows, dtype=np.float32), 1e-3, 1.0
        ),
        descriptor_uncertainties=np.clip(
            np.asarray(uncertainty_rows, dtype=np.float32), 0.0, 1.0
        ),
        metadata=contract,
    )


def retrieve_surface_maplets(
    query_descriptors: np.ndarray,
    bank: SurfaceRetrievalMapletBank,
    *,
    maximum_maplets: int = 12,
    candidates_per_region: int = 8,
    temperature: float = 0.08,
    null_logit: float = 0.0,
    mixture_temperature: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Accumulate soft regional RADIO evidence without point matching."""

    query = _normalize_rows(query_descriptors)
    if query.shape[1] != bank.descriptors.shape[1]:
        raise ValueError("query and maplet feature dimensions differ")
    if len(bank) == 0 or query.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {
            "retrieved_maplet_count": 0,
            "mean_null_probability": 1.0,
        }
    logits = np.empty((query.shape[0], len(bank)), dtype=np.float32)
    for row in range(len(bank)):
        begin = int(bank.descriptor_offsets[row])
        end = int(bank.descriptor_offsets[row + 1])
        component_scores = query @ bank.descriptors[begin:end].T
        maximum = np.max(component_scores, axis=1)
        weights = bank.descriptor_weights[begin:end]
        if float(mixture_temperature) <= 0.0:
            logits[:, row] = maximum
        else:
            logits[:, row] = maximum + float(mixture_temperature) * np.log(
                np.sum(
                    weights[None]
                    * np.exp(
                        (component_scores - maximum[:, None])
                        / float(mixture_temperature)
                    ),
                    axis=1,
                )
                + 1e-12
            )
    logits += 0.15 * np.log(np.clip(bank.quality_scores[None], 1e-4, 1.0))
    logits -= 0.10 * np.clip(bank.descriptor_uncertainties[None], 0.0, 10.0)
    keep = min(int(candidates_per_region), len(bank))
    columns = np.argpartition(-logits, kth=keep - 1, axis=1)[:, :keep]
    candidate_logits = np.take_along_axis(logits, columns, axis=1) / max(
        float(temperature), 1e-4
    )
    row_max = np.maximum(
        np.max(candidate_logits, axis=1, keepdims=True), float(null_logit)
    )
    numerator = np.exp(candidate_logits - row_max)
    null_numerator = np.exp(float(null_logit) - row_max[:, 0])
    denominator = np.sum(numerator, axis=1) + null_numerator
    probabilities = numerator / denominator[:, None]
    null_probabilities = null_numerator / denominator
    evidence = np.zeros((len(bank),), dtype=np.float64)
    np.add.at(evidence, columns.reshape(-1), probabilities.reshape(-1))
    ranked_rows = np.argsort(-evidence, kind="mergesort")
    ranked_rows = ranked_rows[evidence[ranked_rows] > 0.0][: int(maximum_maplets)]
    return bank.maplet_ids[ranked_rows], {
        "retrieved_maplet_count": int(ranked_rows.size),
        "mean_null_probability": float(np.mean(null_probabilities)),
        "regional_evidence_sum": float(np.sum(evidence[ranked_rows])),
        "representation": "compact_radio_final_mixture_per_maplet",
    }
