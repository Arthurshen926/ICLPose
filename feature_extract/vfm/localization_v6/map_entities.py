"""Decoupled retrieval-region and metric-chart entities for V6.

Retrieval regions carry only bounded RADIO-final mixtures and coarse geometry.
Metric charts carry canonical 2DGS geometry and metric features.  Their
relationship is represented explicitly by :class:`RegionChartIndex`; no image
or observation identity is part of any runtime map entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)


REGION_CHART_INDEX_ARTIFACT = "v6_region_chart_index"
REGION_CHART_INDEX_VERSION = 1


@dataclass(frozen=True)
class RetrievalRegionBank:
    """RADIO-final retrieval regions and their declared representation scope.

    This semantic wrapper deliberately does not expose descriptor-conditioned
    surface locations as metric correspondences.  Those historical fields may
    exist in the compact feature bank for calibration compatibility, but only
    identity mixtures and region geometry belong to this entity.  Wrapping a
    legacy single-scale maplet bank does not magically make it a context-rich
    super-region bank; artifacts must declare that distinction.
    """

    feature_bank: SurfaceRetrievalMapletBank

    def __post_init__(self) -> None:
        metadata = dict(self.feature_bank.metadata or {})
        if metadata.get("vfm_layer", "radio_final") != "radio_final":
            raise ValueError("retrieval regions must use RADIO-final")
        if not bool(metadata.get("has_canonical_tangent_frames", False)):
            raise ValueError(
                "retrieval regions require stored canonical tangent frames"
            )

    @property
    def region_ids(self) -> np.ndarray:
        return self.feature_bank.maplet_ids

    @property
    def centers(self) -> np.ndarray:
        return self.feature_bank.centers

    @property
    def frames(self) -> np.ndarray:
        return self.feature_bank.tangent_frames

    @property
    def extents(self) -> np.ndarray:
        return self.feature_bank.extents

    def __len__(self) -> int:
        return len(self.feature_bank)


@dataclass(frozen=True)
class MetricSurfaceChartBank:
    """Canonical, locally metric 2DGS surface charts."""

    atlas: MapletFeatureAtlasBank
    minimum_valid_fraction: float = 0.01

    def __post_init__(self) -> None:
        threshold = float(self.minimum_valid_fraction)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_valid_fraction must lie in [0,1]")
        metadata = dict(self.atlas.metadata or {})
        if metadata.get("artifact_type") not in (
            None,
            "v6_canonical_maplet_feature_atlas",
        ):
            raise ValueError("metric charts require a V6 canonical atlas")

    @property
    def valid_fraction(self) -> np.ndarray:
        return np.mean(self.atlas.valid_mask, axis=(1, 2)).astype(np.float32)

    @property
    def usable_rows(self) -> np.ndarray:
        return np.flatnonzero(
            self.valid_fraction >= float(self.minimum_valid_fraction)
        ).astype(np.int64)

    @property
    def chart_ids(self) -> np.ndarray:
        return self.atlas.maplet_ids[self.usable_rows]

    @property
    def physical_texel_size_m(self) -> np.ndarray:
        """Nominal tangent-plane texel dimensions for every chart."""

        return np.stack(
            [
                2.0 * self.atlas.extents[:, 0] / max(self.atlas.width, 1),
                2.0 * self.atlas.extents[:, 1] / max(self.atlas.height, 1),
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.usable_rows.size)


@dataclass(frozen=True)
class RegionChartIndex:
    """Strict many-to-many retrieval-region ↔ metric-chart adjacency."""

    region_ids: np.ndarray
    region_offsets: np.ndarray
    region_chart_ids: np.ndarray
    chart_ids: np.ndarray
    chart_offsets: np.ndarray
    chart_region_ids: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        region_ids = np.asarray(self.region_ids, dtype=np.int64).reshape(-1)
        chart_ids = np.asarray(self.chart_ids, dtype=np.int64).reshape(-1)
        region_offsets = np.asarray(
            self.region_offsets, dtype=np.int64
        ).reshape(-1)
        chart_offsets = np.asarray(
            self.chart_offsets, dtype=np.int64
        ).reshape(-1)
        region_chart_ids = np.asarray(
            self.region_chart_ids, dtype=np.int64
        ).reshape(-1)
        chart_region_ids = np.asarray(
            self.chart_region_ids, dtype=np.int64
        ).reshape(-1)
        if (
            np.unique(region_ids).size != region_ids.size
            or np.unique(chart_ids).size != chart_ids.size
        ):
            raise ValueError("region_ids and chart_ids must be unique")
        self._validate_csr(
            region_offsets,
            region_ids.size,
            region_chart_ids.size,
            "region",
        )
        self._validate_csr(
            chart_offsets,
            chart_ids.size,
            chart_region_ids.size,
            "chart",
        )
        if not np.all(np.isin(region_chart_ids, chart_ids)):
            raise ValueError("region adjacency contains an unknown chart")
        if not np.all(np.isin(chart_region_ids, region_ids)):
            raise ValueError("chart adjacency contains an unknown region")
        forward = {
            (int(region_ids[row]), int(chart))
            for row in range(region_ids.size)
            for chart in region_chart_ids[
                region_offsets[row] : region_offsets[row + 1]
            ].tolist()
        }
        inverse = {
            (int(region), int(chart_ids[row]))
            for row in range(chart_ids.size)
            for region in chart_region_ids[
                chart_offsets[row] : chart_offsets[row + 1]
            ].tolist()
        }
        if forward != inverse:
            raise ValueError("forward and inverse region-chart edges differ")
        metadata = dict(self.metadata or {})
        if metadata.get(
            "artifact_type", REGION_CHART_INDEX_ARTIFACT
        ) != REGION_CHART_INDEX_ARTIFACT:
            raise ValueError("not a V6 region-chart index")
        version = int(
            metadata.get(
                "artifact_version", REGION_CHART_INDEX_VERSION
            )
        )
        if version != REGION_CHART_INDEX_VERSION:
            raise ValueError(
                f"unsupported region-chart index version: {version}"
            )
        for key in (
            "stores_mapping_rgb",
            "stores_mapping_image_ids",
            "stores_mapping_image_paths",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_stable_anchor_identity",
            "uses_point_correspondences",
        ):
            if bool(metadata.get(key, False)):
                raise ValueError(f"region-chart index violates contract: {key}")
        metadata.update(
            artifact_type=REGION_CHART_INDEX_ARTIFACT,
            artifact_version=REGION_CHART_INDEX_VERSION,
            stores_mapping_rgb=False,
            stores_mapping_image_ids=False,
            stores_mapping_image_paths=False,
            uses_sfm_points=False,
            uses_sfm_tracks=False,
            uses_stable_anchor_identity=False,
            uses_point_correspondences=False,
        )
        object.__setattr__(self, "region_ids", region_ids)
        object.__setattr__(self, "region_offsets", region_offsets)
        object.__setattr__(self, "region_chart_ids", region_chart_ids)
        object.__setattr__(self, "chart_ids", chart_ids)
        object.__setattr__(self, "chart_offsets", chart_offsets)
        object.__setattr__(self, "chart_region_ids", chart_region_ids)
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _validate_csr(
        offsets: np.ndarray, owner_count: int, edge_count: int, name: str
    ) -> None:
        if (
            offsets.shape != (owner_count + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != edge_count
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError(f"{name}_offsets are not canonical CSR offsets")

    @property
    def edge_count(self) -> int:
        return int(self.region_chart_ids.size)

    def charts_for_regions(
        self, region_ids: np.ndarray, *, unique: bool = True
    ) -> np.ndarray:
        rows = {int(value): row for row, value in enumerate(self.region_ids)}
        result = []
        for region_id in np.asarray(region_ids, dtype=np.int64).reshape(-1):
            row = rows.get(int(region_id))
            if row is None:
                continue
            result.extend(
                self.region_chart_ids[
                    self.region_offsets[row] : self.region_offsets[row + 1]
                ].tolist()
            )
        values = np.asarray(result, dtype=np.int64)
        if not unique or values.size == 0:
            return values
        _, first = np.unique(values, return_index=True)
        return values[np.sort(first)]

    def regions_for_charts(
        self, chart_ids: np.ndarray, *, unique: bool = True
    ) -> np.ndarray:
        rows = {int(value): row for row, value in enumerate(self.chart_ids)}
        result = []
        for chart_id in np.asarray(chart_ids, dtype=np.int64).reshape(-1):
            row = rows.get(int(chart_id))
            if row is None:
                continue
            result.extend(
                self.chart_region_ids[
                    self.chart_offsets[row] : self.chart_offsets[row + 1]
                ].tolist()
            )
        values = np.asarray(result, dtype=np.int64)
        if not unique or values.size == 0:
            return values
        _, first = np.unique(values, return_index=True)
        return values[np.sort(first)]

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            region_ids=self.region_ids,
            region_offsets=self.region_offsets,
            region_chart_ids=self.region_chart_ids,
            chart_ids=self.chart_ids,
            chart_offsets=self.chart_offsets,
            chart_region_ids=self.chart_region_ids,
            metadata_json=np.asarray(
                json.dumps(dict(self.metadata or {}), sort_keys=True)
            ),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "RegionChartIndex":
        expected = {
            "region_ids",
            "region_offsets",
            "region_chart_ids",
            "chart_ids",
            "chart_offsets",
            "chart_region_ids",
            "metadata_json",
        }
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != expected:
                raise ValueError(
                    "non-canonical region-chart index fields: "
                    f"{sorted(data.files)}"
                )
            return cls(
                region_ids=data["region_ids"],
                region_offsets=data["region_offsets"],
                region_chart_ids=data["region_chart_ids"],
                chart_ids=data["chart_ids"],
                chart_offsets=data["chart_offsets"],
                chart_region_ids=data["chart_region_ids"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def build_region_chart_index(
    regions: RetrievalRegionBank,
    charts: MetricSurfaceChartBank,
    *,
    maximum_charts_per_region: int = 12,
    region_extent_multiplier: float = 1.5,
    minimum_normal_cosine: float = 0.65,
    metadata: Mapping[str, object] | None = None,
) -> RegionChartIndex:
    """Associate overlapping retrieval regions with nearby metric charts.

    The current St Mary's artifacts originate from one common partition, so
    every usable chart has a same-ID edge.  Additional geometry-overlap edges
    make the representation genuinely many-to-many and let future larger,
    overlapping retrieval regions reuse the same metric charts unchanged.
    """

    if int(maximum_charts_per_region) <= 0:
        raise ValueError("maximum_charts_per_region must be positive")
    usable = charts.usable_rows
    atlas = charts.atlas
    chart_ids = atlas.maplet_ids[usable]
    chart_centers = atlas.centers[usable]
    chart_normals = atlas.frames[usable, 2]
    region_edges: list[np.ndarray] = []
    for row, region_id in enumerate(regions.region_ids.tolist()):
        local = (
            chart_centers - regions.centers[row][None]
        ) @ regions.frames[row].T
        normal_cosine = np.abs(chart_normals @ regions.frames[row, 2])
        scale = np.maximum(
            regions.extents[row, :2] * float(region_extent_multiplier),
            1e-3,
        )
        tangent_distance = np.linalg.norm(local[:, :2] / scale[None], axis=1)
        normal_distance = np.abs(local[:, 2]) / max(
            float(regions.extents[row, 2]) * 3.0, 0.05
        )
        compatible = (
            (np.abs(local[:, 0]) <= scale[0])
            & (np.abs(local[:, 1]) <= scale[1])
            & (normal_distance <= 1.0)
            & (normal_cosine >= float(minimum_normal_cosine))
        )
        score = tangent_distance + 0.25 * normal_distance + (
            1.0 - normal_cosine
        )
        candidates = np.flatnonzero(compatible)
        same = np.flatnonzero(chart_ids == int(region_id))
        if same.size:
            candidates = np.unique(np.r_[same, candidates])
            score[same] = -1.0
        order = candidates[
            np.argsort(score[candidates], kind="mergesort")[
                : int(maximum_charts_per_region)
            ]
        ]
        region_edges.append(chart_ids[order].astype(np.int64))
    region_offsets = np.r_[
        0, np.cumsum([values.size for values in region_edges])
    ].astype(np.int64)
    region_chart_ids = (
        np.concatenate(region_edges)
        if region_edges
        else np.zeros((0,), dtype=np.int64)
    )
    inverse: dict[int, list[int]] = {
        int(chart_id): [] for chart_id in chart_ids.tolist()
    }
    for region_id, edges in zip(regions.region_ids.tolist(), region_edges):
        for chart_id in edges.tolist():
            inverse[int(chart_id)].append(int(region_id))
    chart_edges = [
        np.asarray(inverse[int(chart_id)], dtype=np.int64)
        for chart_id in chart_ids.tolist()
    ]
    chart_offsets = np.r_[
        0, np.cumsum([values.size for values in chart_edges])
    ].astype(np.int64)
    chart_region_ids = (
        np.concatenate(chart_edges)
        if chart_edges
        else np.zeros((0,), dtype=np.int64)
    )
    contract = dict(metadata or {})
    contract.update(
        association_method="canonical_frame_geometry_overlap",
        association_expands_geometry_only=True,
        context_descriptor_retrained=False,
        retrieval_region_representation=str(
            (regions.feature_bank.metadata or {}).get(
                "retrieval_region_representation",
                "legacy_single_scale_maplet_proxy",
            )
        ),
        maximum_charts_per_region=int(maximum_charts_per_region),
        region_extent_multiplier=float(region_extent_multiplier),
        minimum_normal_cosine=float(minimum_normal_cosine),
    )
    return RegionChartIndex(
        region_ids=regions.region_ids,
        region_offsets=region_offsets,
        region_chart_ids=region_chart_ids,
        chart_ids=chart_ids,
        chart_offsets=chart_offsets,
        chart_region_ids=chart_region_ids,
        metadata=contract,
    )


def compose_metric_region_atlas(
    regions: RetrievalRegionBank,
    charts: MetricSurfaceChartBank,
    index: RegionChartIndex,
    region_id: int,
    *,
    resolution: int = 48,
) -> MapletFeatureAtlasBank:
    """Rasterize associated, near-coplanar charts into one coarse region atlas.

    This is the map-side representation needed by frame alignment: multiple
    disconnected metric charts preserve their relative canonical layout,
    yielding a larger candidate footprint without introducing stable point
    identities.  Whether that footprint is better conditioned is an explicit
    M1 diagnostic, not an assumption.  The child chart features, bounded
    appearance modes and 2DGS geometry remain the only stored evidence.
    """

    size = int(resolution)
    if size < 8:
        raise ValueError("composite region resolution must be at least eight")
    region_rows = np.flatnonzero(regions.region_ids == int(region_id))
    if region_rows.size != 1:
        raise ValueError(f"unknown or duplicate retrieval region: {region_id}")
    region_row = int(region_rows[0])
    chart_ids = index.charts_for_regions(np.asarray([region_id]))
    chart_row_by_id = {
        int(value): row
        for row, value in enumerate(charts.atlas.maplet_ids.tolist())
    }
    chart_rows = np.asarray(
        [
            chart_row_by_id[int(chart_id)]
            for chart_id in chart_ids.tolist()
            if int(chart_id) in chart_row_by_id
        ],
        dtype=np.int64,
    )
    if chart_rows.size == 0:
        raise ValueError("retrieval region has no usable metric charts")
    atlas = charts.atlas
    xyz_parts = []
    feature_parts = []
    support_parts = []
    mode_feature_parts = []
    mode_weight_parts = []
    mode_direction_parts = []
    mode_covariance_parts = []
    mode_variance_parts = []
    mode_valid_parts = []
    for row in chart_rows.tolist():
        valid = (
            np.asarray(atlas.valid_mask[row], dtype=bool)
            & (np.asarray(atlas.support_count[row]) > 0)
            & (np.linalg.norm(atlas.features[row], axis=0) > 0.5)
        )
        if not np.any(valid):
            continue
        xyz_parts.append(atlas.xyz[row][valid])
        feature_parts.append(
            atlas.features[row].transpose(1, 2, 0)[valid]
        )
        support_parts.append(
            np.maximum(
                atlas.support_count[row][valid].astype(np.float32), 1.0
            )
        )
        if atlas.mode_features is not None:
            mode_feature_parts.append(
                atlas.mode_features[row].transpose(2, 3, 0, 1)[valid]
            )
            mode_weight_parts.append(
                atlas.mode_weights[row].transpose(1, 2, 0)[valid]
            )
            mode_direction_parts.append(
                atlas.mode_view_directions[row].transpose(1, 2, 0, 3)[
                    valid
                ]
            )
            mode_covariance_parts.append(
                atlas.mode_view_covariance[row].transpose(
                    1, 2, 0, 3, 4
                )[valid]
            )
            mode_variance_parts.append(
                atlas.mode_variance[row].transpose(1, 2, 0)[valid]
            )
            mode_valid_parts.append(
                atlas.mode_valid_mask[row].transpose(1, 2, 0)[valid]
            )
    if not xyz_parts:
        raise ValueError("associated metric charts contain no baked features")
    points = np.concatenate(xyz_parts, axis=0).astype(np.float64)
    features = np.concatenate(feature_parts, axis=0).astype(np.float64)
    support = np.concatenate(support_parts, axis=0).astype(np.float64)
    if atlas.mode_features is not None:
        mode_features = np.concatenate(mode_feature_parts, axis=0).astype(
            np.float64
        )
        mode_weights = np.concatenate(mode_weight_parts, axis=0).astype(
            np.float64
        )
        mode_directions = np.concatenate(
            mode_direction_parts, axis=0
        ).astype(np.float64)
        mode_covariances = np.concatenate(
            mode_covariance_parts, axis=0
        ).astype(np.float64)
        mode_variances = np.concatenate(
            mode_variance_parts, axis=0
        ).astype(np.float64)
        mode_valid = np.concatenate(mode_valid_parts, axis=0)
        mode_count = int(mode_features.shape[1])
        accumulated_mode_feature = np.zeros(
            (size * size, mode_count, atlas.feature_dim),
            dtype=np.float64,
        )
        accumulated_mode_direction = np.zeros(
            (size * size, mode_count, 3), dtype=np.float64
        )
        accumulated_mode_second = np.zeros(
            (size * size, mode_count, 3, 3), dtype=np.float64
        )
        accumulated_mode_variance = np.zeros(
            (size * size, mode_count), dtype=np.float64
        )
        accumulated_mode_weight = np.zeros(
            (size * size, mode_count), dtype=np.float64
        )
    else:
        mode_count = 0
    anchor_center = regions.centers[region_row].astype(np.float64)
    anchor_frame = regions.frames[region_row].astype(np.float64)
    local = (points - anchor_center[None]) @ anchor_frame.T
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    midpoint = 0.5 * (lower + upper)
    extent = np.maximum(0.5 * (upper - lower), 1e-3)
    composite_center = anchor_center + midpoint @ anchor_frame
    centered = local - midpoint[None]
    grid_x = (
        (centered[:, 0] / extent[0] + 1.0) * 0.5 * (size - 1)
    )
    grid_y = (
        (centered[:, 1] / extent[1] + 1.0) * 0.5 * (size - 1)
    )
    x0 = np.clip(np.floor(grid_x).astype(np.int64), 0, size - 1)
    y0 = np.clip(np.floor(grid_y).astype(np.int64), 0, size - 1)
    x1 = np.minimum(x0 + 1, size - 1)
    y1 = np.minimum(y0 + 1, size - 1)
    dx = grid_x - x0
    dy = grid_y - y0
    accumulated_feature = np.zeros(
        (size * size, atlas.feature_dim), dtype=np.float64
    )
    accumulated_xyz = np.zeros((size * size, 3), dtype=np.float64)
    accumulated_weight = np.zeros((size * size,), dtype=np.float64)
    accumulated_count = np.zeros((size * size,), dtype=np.float64)
    for px, py, interpolation in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        weight = interpolation * np.sqrt(support)
        flat = py * size + px
        np.add.at(
            accumulated_feature,
            flat,
            weight[:, None] * features,
        )
        np.add.at(accumulated_xyz, flat, weight[:, None] * points)
        np.add.at(accumulated_weight, flat, weight)
        np.add.at(accumulated_count, flat, interpolation > 1e-6)
        if atlas.mode_features is not None:
            mode_evidence = (
                weight[:, None]
                * mode_weights
                * mode_valid.astype(np.float64)
            )
            np.add.at(
                accumulated_mode_feature,
                flat,
                mode_evidence[:, :, None] * mode_features,
            )
            np.add.at(
                accumulated_mode_direction,
                flat,
                mode_evidence[:, :, None] * mode_directions,
            )
            direction_second = (
                mode_covariances
                + mode_directions[:, :, :, None]
                * mode_directions[:, :, None, :]
            )
            np.add.at(
                accumulated_mode_second,
                flat,
                mode_evidence[:, :, None, None] * direction_second,
            )
            np.add.at(
                accumulated_mode_variance,
                flat,
                mode_evidence * mode_variances,
            )
            np.add.at(
                accumulated_mode_weight,
                flat,
                mode_evidence,
            )
    valid = accumulated_weight > 1e-6
    accumulated_feature[valid] /= accumulated_weight[valid, None]
    accumulated_feature[valid] /= np.maximum(
        np.linalg.norm(
            accumulated_feature[valid], axis=1, keepdims=True
        ),
        1e-8,
    )
    accumulated_xyz[valid] /= accumulated_weight[valid, None]
    feature_grid = accumulated_feature.reshape(
        size, size, atlas.feature_dim
    ).transpose(2, 0, 1)
    xyz_grid = accumulated_xyz.reshape(size, size, 3)
    valid_grid = valid.reshape(size, size)
    primitive_ids = np.full((size, size), -1, dtype=np.int64)
    primitive_ids[valid_grid] = np.arange(
        int(np.sum(valid_grid)), dtype=np.int64
    )
    mode_payload = {}
    if atlas.mode_features is not None:
        composite_mode_valid = accumulated_mode_weight > 1e-8
        denominator = np.maximum(
            accumulated_mode_weight[:, :, None], 1e-8
        )
        accumulated_mode_feature /= denominator
        accumulated_mode_feature /= np.maximum(
            np.linalg.norm(
                accumulated_mode_feature, axis=2, keepdims=True
            ),
            1e-8,
        )
        raw_direction = accumulated_mode_direction / denominator
        accumulated_mode_direction = raw_direction / np.maximum(
            np.linalg.norm(raw_direction, axis=2, keepdims=True),
            1e-8,
        )
        accumulated_mode_second /= np.maximum(
            accumulated_mode_weight[:, :, None, None], 1e-8
        )
        accumulated_mode_covariance = (
            accumulated_mode_second
            - raw_direction[:, :, :, None]
            * raw_direction[:, :, None, :]
        )
        accumulated_mode_covariance = 0.5 * (
            accumulated_mode_covariance
            + accumulated_mode_covariance.transpose(0, 1, 3, 2)
        )
        for axis in range(3):
            accumulated_mode_covariance[:, :, axis, axis] = np.maximum(
                accumulated_mode_covariance[:, :, axis, axis], 0.0
            )
        accumulated_mode_variance /= np.maximum(
            accumulated_mode_weight, 1e-8
        )
        normalized_mode_weight = accumulated_mode_weight / np.maximum(
            np.sum(accumulated_mode_weight, axis=1, keepdims=True),
            1e-8,
        )
        mode_payload = {
            "mode_features": accumulated_mode_feature.reshape(
                size, size, mode_count, atlas.feature_dim
            )
            .transpose(2, 3, 0, 1)[None]
            .astype(np.float32),
            "mode_weights": normalized_mode_weight.reshape(
                size, size, mode_count
            )
            .transpose(2, 0, 1)[None]
            .astype(np.float32),
            "mode_view_directions": accumulated_mode_direction.reshape(
                size, size, mode_count, 3
            )
            .transpose(2, 0, 1, 3)[None]
            .astype(np.float32),
            "mode_view_covariance": accumulated_mode_covariance.reshape(
                size, size, mode_count, 3, 3
            )
            .transpose(2, 0, 1, 3, 4)[None]
            .astype(np.float32),
            "mode_variance": accumulated_mode_variance.reshape(
                size, size, mode_count
            )
            .transpose(2, 0, 1)[None]
            .astype(np.float32),
            "mode_valid_mask": composite_mode_valid.reshape(
                size, size, mode_count
            ).transpose(2, 0, 1)[None],
        }
    metadata = dict(atlas.metadata or {})
    metadata.update(
        artifact_type="v6_composite_metric_retrieval_region_atlas",
        representation=(
            "disconnected_metric_charts_in_retrieval_region_canonical_frame"
        ),
        retrieval_region_id=int(region_id),
        associated_chart_ids=chart_ids.tolist(),
        associated_chart_count=int(chart_rows.size),
        composite_resolution=size,
        observation_dependent_geometry=False,
        stores_mapping_rgb=False,
        stores_mapping_image_ids=False,
        stores_mapping_image_paths=False,
        uses_sfm_points=False,
        uses_sfm_tracks=False,
        uses_stable_anchor_identity=False,
        uses_point_correspondence_pnp=False,
    )
    return MapletFeatureAtlasBank(
        maplet_ids=np.asarray([region_id], dtype=np.int64),
        centers=np.asarray([composite_center], dtype=np.float32),
        frames=np.asarray([anchor_frame], dtype=np.float32),
        extents=np.asarray([extent], dtype=np.float32),
        xyz=xyz_grid[None].astype(np.float32),
        primitive_ids=primitive_ids[None],
        features=feature_grid[None].astype(np.float32),
        variance=np.zeros((1, size, size), dtype=np.float32),
        support_count=np.rint(accumulated_count)
        .reshape(1, size, size)
        .astype(np.int32),
        valid_mask=valid_grid[None],
        metadata=metadata,
        **mode_payload,
    )


def merge_metric_atlases(
    atlases: list[MapletFeatureAtlasBank] | tuple[MapletFeatureAtlasBank, ...],
) -> MapletFeatureAtlasBank:
    """Merge same-resolution single-chart atlases for grouped pose solving."""

    values = tuple(atlases)
    if not values:
        raise ValueError("cannot merge an empty metric-atlas list")
    mode_presence = [atlas.mode_features is not None for atlas in values]
    if any(
        atlas.height != values[0].height
        or atlas.width != values[0].width
        or atlas.feature_dim != values[0].feature_dim
        or len(atlas) != 1
        for atlas in values
    ) or any(mode_presence) != all(mode_presence):
        raise ValueError(
            "merged metric atlases must share resolution and mode schema"
        )
    if all(mode_presence) and any(
        atlas.appearance_mode_count != values[0].appearance_mode_count
        for atlas in values
    ):
        raise ValueError("merged metric atlases have different mode counts")
    ids = np.concatenate([atlas.maplet_ids for atlas in values])
    if np.unique(ids).size != ids.size:
        raise ValueError("merged metric chart IDs must be unique")
    metadata = dict(values[0].metadata or {})
    metadata.update(
        artifact_type="v6_merged_composite_metric_region_atlas",
        merged_chart_count=len(values),
        stores_mapping_rgb=False,
        stores_mapping_image_ids=False,
        stores_mapping_image_paths=False,
        uses_sfm_points=False,
        uses_sfm_tracks=False,
        uses_stable_anchor_identity=False,
        uses_point_correspondence_pnp=False,
    )
    mode_payload = {}
    if all(mode_presence):
        mode_payload = {
            name: np.concatenate(
                [np.asarray(getattr(atlas, name)) for atlas in values]
            )
            for name in (
                "mode_features",
                "mode_weights",
                "mode_view_directions",
                "mode_view_covariance",
                "mode_variance",
                "mode_valid_mask",
            )
        }
    return MapletFeatureAtlasBank(
        maplet_ids=ids,
        centers=np.concatenate([atlas.centers for atlas in values]),
        frames=np.concatenate([atlas.frames for atlas in values]),
        extents=np.concatenate([atlas.extents for atlas in values]),
        xyz=np.concatenate([atlas.xyz for atlas in values]),
        primitive_ids=np.concatenate(
            [atlas.primitive_ids for atlas in values]
        ),
        features=np.concatenate([atlas.features for atlas in values]),
        variance=np.concatenate([atlas.variance for atlas in values]),
        support_count=np.concatenate(
            [atlas.support_count for atlas in values]
        ),
        valid_mask=np.concatenate([atlas.valid_mask for atlas in values]),
        metadata=metadata,
        **mode_payload,
    )
