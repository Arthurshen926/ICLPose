"""VFM patch-token to 2DGS surface-region anchor mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.gaussian_raw_landmarks import _project_xyz_to_grid, vfm_token_saliency
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFeatureView,
    GaussianVFMSource,
    _intrinsic_matrix,
    _quaternion_rotation_matrices,
)
from feature_extract.vfm.query_to_3d_matching import normalize_rows


@dataclass(frozen=True)
class Vfm2DgsMappingConfig:
    token_top_fraction: float = 0.05
    min_token_saliency: float = 0.0
    saliency_mode: str = "norm"
    token_selection_mode: str = "top"
    token_grid_rows: int = 4
    token_grid_cols: int = 4
    max_tokens_per_cell: int = 0
    min_surface_token_contribution: float = 0.0
    max_surface_tokens: int = 0
    surface_token_saliency_power: float = 0.0
    footprint_radius_px: float = 1.25
    footprint_sample_grid: int = 1
    footprint_sample_extent_px: float = 0.0
    max_projected_disk_radius_px: float = 2.0
    depth_epsilon: float = 0.05
    opacity_threshold: float = 0.05
    view_angle_power: float = 0.0
    bidirectional_lambda: float = 0.5
    element_coverage_power: float = 1.0
    depth_purity_sigma: float = 0.25
    normal_purity_power: float = 1.0
    min_purity: float = 0.0
    max_effective_support_elements: float = 0.0
    support_mode: str = "projection_depth"
    token_anchor_competition: str = "none"
    token_anchor_neighborhood_hops: int = 1
    token_anchor_max_support_elements: int = 0
    min_component_concentration: float = 0.6
    min_full_component_concentration: float = 0.0
    max_full_component_count: int = 0
    max_full_support_elements: int = 0
    weak_observation_mode: str = "drop"
    weak_min_full_component_concentration: float = 0.0
    weak_max_full_support_elements: int = 0
    weak_quality_scale: float = 0.25
    weak_descriptor_weight: float = 0.0
    min_responsibility: float = 0.01
    max_elements_per_token: int = 64
    l2_normalize_features: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.token_top_fraction) <= 1.0:
            raise ValueError("token_top_fraction must be in (0, 1]")
        if float(self.min_token_saliency) < 0.0:
            raise ValueError("min_token_saliency must be non-negative")
        if self.saliency_mode not in {"norm", "local_contrast"}:
            raise ValueError("saliency_mode must be 'norm' or 'local_contrast'")
        if self.token_selection_mode not in {"top", "grid_top", "all", "surface"}:
            raise ValueError("token_selection_mode must be 'top', 'grid_top', 'all', or 'surface'")
        if int(self.token_grid_rows) <= 0 or int(self.token_grid_cols) <= 0:
            raise ValueError("token_grid_rows and token_grid_cols must be positive")
        if int(self.max_tokens_per_cell) < 0:
            raise ValueError("max_tokens_per_cell must be non-negative")
        if float(self.min_surface_token_contribution) < 0.0:
            raise ValueError("min_surface_token_contribution must be non-negative")
        if int(self.max_surface_tokens) < 0:
            raise ValueError("max_surface_tokens must be non-negative")
        if float(self.surface_token_saliency_power) < 0.0:
            raise ValueError("surface_token_saliency_power must be non-negative")
        if float(self.footprint_radius_px) <= 0.0:
            raise ValueError("footprint_radius_px must be positive")
        if int(self.footprint_sample_grid) <= 0:
            raise ValueError("footprint_sample_grid must be positive")
        if float(self.footprint_sample_extent_px) < 0.0:
            raise ValueError("footprint_sample_extent_px must be non-negative")
        if float(self.max_projected_disk_radius_px) < 0.0:
            raise ValueError("max_projected_disk_radius_px must be non-negative")
        if float(self.depth_epsilon) < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if not 0.0 <= float(self.opacity_threshold) <= 1.0:
            raise ValueError("opacity_threshold must be in [0, 1]")
        if float(self.view_angle_power) < 0.0:
            raise ValueError("view_angle_power must be non-negative")
        if not 0.0 <= float(self.bidirectional_lambda) <= 1.0:
            raise ValueError("bidirectional_lambda must be in [0, 1]")
        if float(self.element_coverage_power) < 0.0:
            raise ValueError("element_coverage_power must be non-negative")
        if float(self.depth_purity_sigma) <= 0.0:
            raise ValueError("depth_purity_sigma must be positive")
        if float(self.normal_purity_power) < 0.0:
            raise ValueError("normal_purity_power must be non-negative")
        if not 0.0 <= float(self.min_purity) <= 1.0:
            raise ValueError("min_purity must be in [0, 1]")
        if float(self.max_effective_support_elements) < 0.0:
            raise ValueError("max_effective_support_elements must be non-negative")
        if self.support_mode not in {"projection_depth", "surface_component"}:
            raise ValueError("support_mode must be 'projection_depth' or 'surface_component'")
        if self.token_anchor_competition not in {"none", "winner_component", "winner_element", "winner_neighborhood"}:
            raise ValueError("token_anchor_competition must be 'none', 'winner_component', 'winner_element', or 'winner_neighborhood'")
        if int(self.token_anchor_neighborhood_hops) < 0:
            raise ValueError("token_anchor_neighborhood_hops must be non-negative")
        if int(self.token_anchor_max_support_elements) < 0:
            raise ValueError("token_anchor_max_support_elements must be non-negative")
        if not 0.0 <= float(self.min_component_concentration) <= 1.0:
            raise ValueError("min_component_concentration must be in [0, 1]")
        if not 0.0 <= float(self.min_full_component_concentration) <= 1.0:
            raise ValueError("min_full_component_concentration must be in [0, 1]")
        if int(self.max_full_component_count) < 0:
            raise ValueError("max_full_component_count must be non-negative")
        if int(self.max_full_support_elements) < 0:
            raise ValueError("max_full_support_elements must be non-negative")
        if self.weak_observation_mode not in {"drop", "keep"}:
            raise ValueError("weak_observation_mode must be 'drop' or 'keep'")
        if not 0.0 <= float(self.weak_min_full_component_concentration) <= 1.0:
            raise ValueError("weak_min_full_component_concentration must be in [0, 1]")
        if int(self.weak_max_full_support_elements) < 0:
            raise ValueError("weak_max_full_support_elements must be non-negative")
        if float(self.weak_quality_scale) < 0.0:
            raise ValueError("weak_quality_scale must be non-negative")
        if float(self.weak_descriptor_weight) < 0.0:
            raise ValueError("weak_descriptor_weight must be non-negative")
        if float(self.min_responsibility) < 0.0:
            raise ValueError("min_responsibility must be non-negative")
        if int(self.max_elements_per_token) <= 0:
            raise ValueError("max_elements_per_token must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_top_fraction": float(self.token_top_fraction),
            "min_token_saliency": float(self.min_token_saliency),
            "saliency_mode": str(self.saliency_mode),
            "token_selection_mode": str(self.token_selection_mode),
            "token_grid_rows": int(self.token_grid_rows),
            "token_grid_cols": int(self.token_grid_cols),
            "max_tokens_per_cell": int(self.max_tokens_per_cell),
            "min_surface_token_contribution": float(self.min_surface_token_contribution),
            "max_surface_tokens": int(self.max_surface_tokens),
            "surface_token_saliency_power": float(self.surface_token_saliency_power),
            "footprint_radius_px": float(self.footprint_radius_px),
            "footprint_sample_grid": int(self.footprint_sample_grid),
            "footprint_sample_extent_px": float(self.footprint_sample_extent_px),
            "max_projected_disk_radius_px": float(self.max_projected_disk_radius_px),
            "depth_epsilon": float(self.depth_epsilon),
            "opacity_threshold": float(self.opacity_threshold),
            "view_angle_power": float(self.view_angle_power),
            "bidirectional_lambda": float(self.bidirectional_lambda),
            "element_coverage_power": float(self.element_coverage_power),
            "depth_purity_sigma": float(self.depth_purity_sigma),
            "normal_purity_power": float(self.normal_purity_power),
            "min_purity": float(self.min_purity),
            "max_effective_support_elements": float(self.max_effective_support_elements),
            "support_mode": str(self.support_mode),
            "token_anchor_competition": str(self.token_anchor_competition),
            "token_anchor_neighborhood_hops": int(self.token_anchor_neighborhood_hops),
            "token_anchor_max_support_elements": int(self.token_anchor_max_support_elements),
            "min_component_concentration": float(self.min_component_concentration),
            "min_full_component_concentration": float(self.min_full_component_concentration),
            "max_full_component_count": int(self.max_full_component_count),
            "max_full_support_elements": int(self.max_full_support_elements),
            "weak_observation_mode": str(self.weak_observation_mode),
            "weak_min_full_component_concentration": float(self.weak_min_full_component_concentration),
            "weak_max_full_support_elements": int(self.weak_max_full_support_elements),
            "weak_quality_scale": float(self.weak_quality_scale),
            "weak_descriptor_weight": float(self.weak_descriptor_weight),
            "min_responsibility": float(self.min_responsibility),
            "max_elements_per_token": int(self.max_elements_per_token),
            "l2_normalize_features": bool(self.l2_normalize_features),
        }


@dataclass(frozen=True)
class Vfm2DgsAnchorFusionConfig:
    fusion_mode: str = "greedy"
    source_merge_policy: str = "allow"
    cross_source_min_feature_cosine: float = 0.8
    min_surface_iou: float = 0.25
    min_dilated_surface_iou: float = 0.0
    support_iou_dilation_hops: int = 0
    min_parent_surface_iou: float = 0.0
    min_normal_cosine: float = 0.5
    max_center_distance: float = 0.5
    min_observations: int = 2
    min_descriptor_observations: int = 0
    support_core_min_observations: int = 0
    support_core_min_fraction: float = 0.0
    l2_normalize_features: bool = True
    max_feature_prototypes: int = 4
    prototype_min_cosine: float = 0.8
    view_bin_count: int = 4
    view_bin_feature_mode: str = "mean"
    feature_fusion_mode: str = "mean"
    feature_consensus_weight_power: float = 1.0
    min_feature_consensus_cosine: float = -1.0
    robust_feature_trim_fraction: float = 0.0
    surface_first_max_seeds_per_observation: int = 1
    surface_first_min_seed_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.fusion_mode not in {"greedy", "graph", "surface_first"}:
            raise ValueError("fusion_mode must be 'greedy', 'graph', or 'surface_first'")
        if self.source_merge_policy not in {"allow", "same_source", "feature_agree"}:
            raise ValueError("source_merge_policy must be 'allow', 'same_source', or 'feature_agree'")
        if not -1.0 <= float(self.cross_source_min_feature_cosine) <= 1.0:
            raise ValueError("cross_source_min_feature_cosine must be in [-1, 1]")
        if not 0.0 <= float(self.min_surface_iou) <= 1.0:
            raise ValueError("min_surface_iou must be in [0, 1]")
        if not 0.0 <= float(self.min_dilated_surface_iou) <= 1.0:
            raise ValueError("min_dilated_surface_iou must be in [0, 1]")
        if int(self.support_iou_dilation_hops) < 0:
            raise ValueError("support_iou_dilation_hops must be non-negative")
        if not 0.0 <= float(self.min_parent_surface_iou) <= 1.0:
            raise ValueError("min_parent_surface_iou must be in [0, 1]")
        if not -1.0 <= float(self.min_normal_cosine) <= 1.0:
            raise ValueError("min_normal_cosine must be in [-1, 1]")
        if float(self.max_center_distance) <= 0.0:
            raise ValueError("max_center_distance must be positive")
        if int(self.min_observations) <= 0:
            raise ValueError("min_observations must be positive")
        if int(self.min_descriptor_observations) < 0:
            raise ValueError("min_descriptor_observations must be non-negative")
        if int(self.support_core_min_observations) < 0:
            raise ValueError("support_core_min_observations must be non-negative")
        if not 0.0 <= float(self.support_core_min_fraction) <= 1.0:
            raise ValueError("support_core_min_fraction must be in [0, 1]")
        if int(self.max_feature_prototypes) <= 0:
            raise ValueError("max_feature_prototypes must be positive")
        if not -1.0 <= float(self.prototype_min_cosine) <= 1.0:
            raise ValueError("prototype_min_cosine must be in [-1, 1]")
        if int(self.view_bin_count) <= 0:
            raise ValueError("view_bin_count must be positive")
        if self.view_bin_feature_mode not in {"mean", "medoid", "consensus_weighted_mean"}:
            raise ValueError("view_bin_feature_mode must be 'mean', 'medoid', or 'consensus_weighted_mean'")
        if self.feature_fusion_mode not in {"mean", "consensus_weighted_mean"}:
            raise ValueError("feature_fusion_mode must be 'mean' or 'consensus_weighted_mean'")
        if float(self.feature_consensus_weight_power) < 0.0:
            raise ValueError("feature_consensus_weight_power must be non-negative")
        if not -1.0 <= float(self.min_feature_consensus_cosine) <= 1.0:
            raise ValueError("min_feature_consensus_cosine must be in [-1, 1]")
        if not 0.0 <= float(self.robust_feature_trim_fraction) < 1.0:
            raise ValueError("robust_feature_trim_fraction must be in [0, 1)")
        if int(self.surface_first_max_seeds_per_observation) < 0:
            raise ValueError("surface_first_max_seeds_per_observation must be non-negative")
        if float(self.surface_first_min_seed_weight) < 0.0:
            raise ValueError("surface_first_min_seed_weight must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "fusion_mode": str(self.fusion_mode),
            "source_merge_policy": str(self.source_merge_policy),
            "cross_source_min_feature_cosine": float(self.cross_source_min_feature_cosine),
            "min_surface_iou": float(self.min_surface_iou),
            "min_dilated_surface_iou": float(self.min_dilated_surface_iou),
            "support_iou_dilation_hops": int(self.support_iou_dilation_hops),
            "min_parent_surface_iou": float(self.min_parent_surface_iou),
            "min_normal_cosine": float(self.min_normal_cosine),
            "max_center_distance": float(self.max_center_distance),
            "min_observations": int(self.min_observations),
            "min_descriptor_observations": int(self.min_descriptor_observations),
            "support_core_min_observations": int(self.support_core_min_observations),
            "support_core_min_fraction": float(self.support_core_min_fraction),
            "l2_normalize_features": bool(self.l2_normalize_features),
            "max_feature_prototypes": int(self.max_feature_prototypes),
            "prototype_min_cosine": float(self.prototype_min_cosine),
            "view_bin_count": int(self.view_bin_count),
            "view_bin_feature_mode": str(self.view_bin_feature_mode),
            "feature_fusion_mode": str(self.feature_fusion_mode),
            "feature_consensus_weight_power": float(self.feature_consensus_weight_power),
            "min_feature_consensus_cosine": float(self.min_feature_consensus_cosine),
            "robust_feature_trim_fraction": float(self.robust_feature_trim_fraction),
            "surface_first_max_seeds_per_observation": int(self.surface_first_max_seeds_per_observation),
            "surface_first_min_seed_weight": float(self.surface_first_min_seed_weight),
        }


@dataclass(frozen=True)
class SurfaceElementMap:
    element_ids: np.ndarray
    parent_gaussian_indices: np.ndarray
    centers: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    normals: np.ndarray
    scale1: np.ndarray
    scale2: np.ndarray
    opacity: np.ndarray
    area: np.ndarray
    adjacency: tuple[np.ndarray, ...]
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        element_ids = np.asarray(self.element_ids, dtype=np.int64).reshape(-1)
        count = int(element_ids.shape[0])
        centers = np.asarray(self.centers, dtype=np.float64)
        tangent1 = np.asarray(self.tangent1, dtype=np.float32)
        tangent2 = np.asarray(self.tangent2, dtype=np.float32)
        normals = np.asarray(self.normals, dtype=np.float32)
        if centers.shape != (count, 3):
            raise ValueError("centers must have shape (N, 3)")
        if tangent1.shape != (count, 3):
            raise ValueError("tangent1 must have shape (N, 3)")
        if tangent2.shape != (count, 3):
            raise ValueError("tangent2 must have shape (N, 3)")
        if normals.shape != (count, 3):
            raise ValueError("normals must have shape (N, 3)")
        for name in ("parent_gaussian_indices",):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name in ("scale1", "scale2", "opacity", "area"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        tangent1, tangent2, normals = _normalize_surface_basis(tangent1, tangent2, normals)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "tangent1", tangent1)
        object.__setattr__(self, "tangent2", tangent2)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "adjacency", tuple(np.asarray(row, dtype=np.int64) for row in self.adjacency))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(self.element_ids.shape[0])

    @property
    def row_by_element_id(self) -> dict[int, int]:
        return {int(element_id): int(row) for row, element_id in enumerate(self.element_ids.tolist())}

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        adjacency_offsets = [0]
        adjacency_values = []
        for neighbors in self.adjacency:
            adjacency_values.extend(np.asarray(neighbors, dtype=np.int64).reshape(-1).tolist())
            adjacency_offsets.append(len(adjacency_values))
        np.savez_compressed(
            path,
            element_ids=self.element_ids.astype(np.int64),
            parent_gaussian_indices=self.parent_gaussian_indices.astype(np.int64),
            centers=self.centers.astype(np.float32),
            tangent1=self.tangent1.astype(np.float32),
            tangent2=self.tangent2.astype(np.float32),
            normals=self.normals.astype(np.float32),
            scale1=self.scale1.astype(np.float32),
            scale2=self.scale2.astype(np.float32),
            opacity=self.opacity.astype(np.float32),
            area=self.area.astype(np.float32),
            adjacency_offsets=np.asarray(adjacency_offsets, dtype=np.int64),
            adjacency_values=np.asarray(adjacency_values, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "SurfaceElementMap":
        with np.load(Path(path), allow_pickle=True) as data:
            offsets = np.asarray(data["adjacency_offsets"], dtype=np.int64)
            values = np.asarray(data["adjacency_values"], dtype=np.int64)
            adjacency = tuple(values[int(offsets[idx]) : int(offsets[idx + 1])] for idx in range(offsets.size - 1))
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item())) if "metadata_json" in data else {}
            return cls(
                element_ids=np.asarray(data["element_ids"], dtype=np.int64),
                parent_gaussian_indices=np.asarray(data["parent_gaussian_indices"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                tangent1=np.asarray(data["tangent1"], dtype=np.float32),
                tangent2=np.asarray(data["tangent2"], dtype=np.float32),
                normals=np.asarray(data["normals"], dtype=np.float32),
                scale1=np.asarray(data["scale1"], dtype=np.float32),
                scale2=np.asarray(data["scale2"], dtype=np.float32),
                opacity=np.asarray(data["opacity"], dtype=np.float32),
                area=np.asarray(data["area"], dtype=np.float32),
                adjacency=adjacency,
                metadata=metadata,
            )


@dataclass(frozen=True)
class TokenSurfaceObservation:
    image_id: str
    token_index: int
    token_xy: np.ndarray
    feature: np.ndarray
    element_ids: np.ndarray
    element_weights: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    covariance: np.ndarray
    purity_score: float
    component_concentration: float
    quality_score: float
    view_direction: np.ndarray | None = None
    purity_components: Mapping[str, float] | None = None
    source_id: str = ""
    observation_strength: str = "strong"
    descriptor_weight: float | None = None

    def __post_init__(self) -> None:
        element_ids = np.asarray(self.element_ids, dtype=np.int64).reshape(-1)
        element_weights = np.asarray(self.element_weights, dtype=np.float32).reshape(-1)
        if element_ids.shape != element_weights.shape:
            raise ValueError("element_ids and element_weights must have matching shape")
        total = float(np.sum(element_weights))
        if total > 0.0:
            element_weights = element_weights / total
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "token_index", int(self.token_index))
        object.__setattr__(self, "token_xy", np.asarray(self.token_xy, dtype=np.float32).reshape(2))
        object.__setattr__(self, "feature", np.asarray(self.feature, dtype=np.float32).reshape(-1))
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "element_weights", element_weights)
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64).reshape(3))
        object.__setattr__(self, "normal", _normalize_vectors(np.asarray(self.normal, dtype=np.float32).reshape(1, 3))[0])
        object.__setattr__(self, "covariance", np.asarray(self.covariance, dtype=np.float32).reshape(3, 3))
        object.__setattr__(self, "purity_score", float(self.purity_score))
        object.__setattr__(self, "component_concentration", float(self.component_concentration))
        object.__setattr__(self, "quality_score", float(self.quality_score))
        strength = str(self.observation_strength)
        if strength not in {"strong", "weak"}:
            raise ValueError("observation_strength must be 'strong' or 'weak'")
        descriptor_weight = (
            float(self.quality_score)
            if self.descriptor_weight is None and strength == "strong"
            else 0.0 if self.descriptor_weight is None else float(self.descriptor_weight)
        )
        if descriptor_weight < 0.0:
            raise ValueError("descriptor_weight must be non-negative")
        object.__setattr__(self, "observation_strength", strength)
        object.__setattr__(self, "descriptor_weight", descriptor_weight)
        view_direction = (
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            if self.view_direction is None
            else np.asarray(self.view_direction, dtype=np.float32).reshape(1, 3)[0]
        )
        norm = float(np.linalg.norm(view_direction))
        if norm > 1e-8:
            view_direction = view_direction / norm
        object.__setattr__(self, "view_direction", view_direction.astype(np.float32, copy=False))
        object.__setattr__(
            self,
            "purity_components",
            {str(key): float(value) for key, value in dict(self.purity_components or {}).items()},
        )


@dataclass(frozen=True)
class Vfm2DgsContributionBuffer:
    image_id: str
    renderer: str
    token_indices: np.ndarray
    token_xy: np.ndarray
    support_offsets: np.ndarray
    element_ids: np.ndarray
    element_weights: np.ndarray
    purity_scores: np.ndarray
    component_concentrations: np.ndarray
    quality_scores: np.ndarray
    top_alpha: np.ndarray
    alpha_entropy: np.ndarray
    metadata: Mapping[str, object] | None = None
    observation_strengths: tuple[str, ...] | None = None
    descriptor_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        token_indices = np.asarray(self.token_indices, dtype=np.int64).reshape(-1)
        count = int(token_indices.shape[0])
        token_xy = np.asarray(self.token_xy, dtype=np.float32)
        if token_xy.shape != (count, 2):
            raise ValueError("token_xy must have shape (N, 2)")
        support_offsets = np.asarray(self.support_offsets, dtype=np.int64).reshape(-1)
        if support_offsets.shape != (count + 1,):
            raise ValueError("support_offsets must have shape (N + 1,)")
        element_ids = np.asarray(self.element_ids, dtype=np.int64).reshape(-1)
        element_weights = np.asarray(self.element_weights, dtype=np.float32).reshape(-1)
        if element_ids.shape != element_weights.shape:
            raise ValueError("element_ids and element_weights must match")
        for name in ("purity_scores", "component_concentrations", "quality_scores", "top_alpha", "alpha_entropy"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        if self.observation_strengths is None:
            observation_strengths = tuple("strong" for _ in range(count))
        else:
            observation_strengths = tuple(str(item) for item in self.observation_strengths)
            if len(observation_strengths) != count:
                raise ValueError("observation_strengths must have length N")
            if any(item not in {"strong", "weak"} for item in observation_strengths):
                raise ValueError("observation_strengths entries must be 'strong' or 'weak'")
        if self.descriptor_weights is None:
            descriptor_weights = np.asarray(self.quality_scores, dtype=np.float32).reshape(-1)
        else:
            descriptor_weights = np.asarray(self.descriptor_weights, dtype=np.float32).reshape(-1)
            if descriptor_weights.shape != (count,):
                raise ValueError("descriptor_weights must have shape (N,)")
        descriptor_weights = np.maximum(descriptor_weights, 0.0).astype(np.float32, copy=False)
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "renderer", str(self.renderer))
        object.__setattr__(self, "token_indices", token_indices)
        object.__setattr__(self, "token_xy", token_xy)
        object.__setattr__(self, "support_offsets", support_offsets)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "element_weights", element_weights)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "observation_strengths", observation_strengths)
        object.__setattr__(self, "descriptor_weights", descriptor_weights)

    def __len__(self) -> int:
        return int(self.token_indices.shape[0])

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_id=np.asarray(self.image_id),
            renderer=np.asarray(self.renderer),
            token_indices=self.token_indices.astype(np.int64),
            token_xy=self.token_xy.astype(np.float32),
            support_offsets=self.support_offsets.astype(np.int64),
            element_ids=self.element_ids.astype(np.int64),
            element_weights=self.element_weights.astype(np.float32),
            purity_scores=self.purity_scores.astype(np.float32),
            component_concentrations=self.component_concentrations.astype(np.float32),
            quality_scores=self.quality_scores.astype(np.float32),
            top_alpha=self.top_alpha.astype(np.float32),
            alpha_entropy=self.alpha_entropy.astype(np.float32),
            observation_strengths_json=np.asarray(json.dumps(list(self.observation_strengths))),
            descriptor_weights=self.descriptor_weights.astype(np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Vfm2DgsContributionBuffer":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item())) if "metadata_json" in data else {}
            return cls(
                image_id=str(np.asarray(data["image_id"]).item()),
                renderer=str(np.asarray(data["renderer"]).item()),
                token_indices=np.asarray(data["token_indices"], dtype=np.int64),
                token_xy=np.asarray(data["token_xy"], dtype=np.float32),
                support_offsets=np.asarray(data["support_offsets"], dtype=np.int64),
                element_ids=np.asarray(data["element_ids"], dtype=np.int64),
                element_weights=np.asarray(data["element_weights"], dtype=np.float32),
                purity_scores=np.asarray(data["purity_scores"], dtype=np.float32),
                component_concentrations=np.asarray(data["component_concentrations"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                top_alpha=np.asarray(data["top_alpha"], dtype=np.float32),
                alpha_entropy=np.asarray(data["alpha_entropy"], dtype=np.float32),
                observation_strengths=(
                    tuple(json.loads(str(np.asarray(data["observation_strengths_json"]).item())))
                    if "observation_strengths_json" in data
                    else None
                ),
                descriptor_weights=(
                    np.asarray(data["descriptor_weights"], dtype=np.float32)
                    if "descriptor_weights" in data
                    else None
                ),
                metadata=metadata,
            )


@dataclass(frozen=True)
class Vfm2DgsObservationBank:
    image_ids: tuple[str, ...]
    token_indices: np.ndarray
    token_xy: np.ndarray
    features: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    covariances: np.ndarray
    support_offsets: np.ndarray
    element_ids: np.ndarray
    element_weights: np.ndarray
    purity_scores: np.ndarray
    component_concentrations: np.ndarray
    quality_scores: np.ndarray
    view_directions: np.ndarray
    metadata: Mapping[str, object] | None = None
    source_ids: tuple[str, ...] | None = None
    observation_strengths: tuple[str, ...] | None = None
    descriptor_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        image_ids = tuple(str(item) for item in self.image_ids)
        count = len(image_ids)
        if self.source_ids is None:
            source_ids = tuple("" for _ in range(count))
        else:
            source_ids = tuple(str(item) for item in self.source_ids)
            if len(source_ids) != count:
                raise ValueError("source_ids must have length N")
        if self.observation_strengths is None:
            observation_strengths = tuple("strong" for _ in range(count))
        else:
            observation_strengths = tuple(str(item) for item in self.observation_strengths)
            if len(observation_strengths) != count:
                raise ValueError("observation_strengths must have length N")
            if any(item not in {"strong", "weak"} for item in observation_strengths):
                raise ValueError("observation_strengths entries must be 'strong' or 'weak'")
        token_indices = np.asarray(self.token_indices, dtype=np.int64).reshape(-1)
        token_xy = np.asarray(self.token_xy, dtype=np.float32)
        features = np.asarray(self.features, dtype=np.float32)
        centers = np.asarray(self.centers, dtype=np.float64)
        normals = np.asarray(self.normals, dtype=np.float32)
        covariances = np.asarray(self.covariances, dtype=np.float32)
        view_directions = np.asarray(self.view_directions, dtype=np.float32)
        if token_indices.shape != (count,):
            raise ValueError("token_indices must match image_ids")
        if token_xy.shape != (count, 2):
            raise ValueError("token_xy must have shape (N, 2)")
        if features.ndim != 2 or features.shape[0] != count:
            raise ValueError("features must have shape (N, C)")
        if centers.shape != (count, 3):
            raise ValueError("centers must have shape (N, 3)")
        if normals.shape != (count, 3):
            raise ValueError("normals must have shape (N, 3)")
        if covariances.shape != (count, 3, 3):
            raise ValueError("covariances must have shape (N, 3, 3)")
        if view_directions.shape != (count, 3):
            raise ValueError("view_directions must have shape (N, 3)")
        support_offsets = np.asarray(self.support_offsets, dtype=np.int64).reshape(-1)
        if support_offsets.shape != (count + 1,):
            raise ValueError("support_offsets must have shape (N + 1,)")
        element_ids = np.asarray(self.element_ids, dtype=np.int64).reshape(-1)
        element_weights = np.asarray(self.element_weights, dtype=np.float32).reshape(-1)
        if element_ids.shape != element_weights.shape:
            raise ValueError("element_ids and element_weights must match")
        for name in ("purity_scores", "component_concentrations", "quality_scores"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        if self.descriptor_weights is None:
            descriptor_weights = np.asarray(self.quality_scores, dtype=np.float32).reshape(-1)
        else:
            descriptor_weights = np.asarray(self.descriptor_weights, dtype=np.float32).reshape(-1)
            if descriptor_weights.shape != (count,):
                raise ValueError("descriptor_weights must have shape (N,)")
        descriptor_weights = np.maximum(descriptor_weights, 0.0).astype(np.float32, copy=False)
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "token_indices", token_indices)
        object.__setattr__(self, "token_xy", token_xy)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "normals", _normalize_vectors(normals))
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(self, "support_offsets", support_offsets)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "element_weights", element_weights)
        object.__setattr__(self, "view_directions", _normalize_vectors(view_directions))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "observation_strengths", observation_strengths)
        object.__setattr__(self, "descriptor_weights", descriptor_weights)

    def __len__(self) -> int:
        return len(self.image_ids)

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[TokenSurfaceObservation],
        metadata: Mapping[str, object] | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> "Vfm2DgsObservationBank":
        if source_ids is not None and len(source_ids) != len(observations):
            raise ValueError("source_ids must match observations")
        support_offsets = [0]
        element_ids: list[int] = []
        element_weights: list[float] = []
        for obs in observations:
            element_ids.extend(obs.element_ids.astype(np.int64).tolist())
            element_weights.extend(obs.element_weights.astype(np.float32).tolist())
            support_offsets.append(len(element_ids))
        feature_dim = int(observations[0].feature.shape[0]) if observations else 0
        return cls(
            image_ids=tuple(str(obs.image_id) for obs in observations),
            token_indices=np.asarray([int(obs.token_index) for obs in observations], dtype=np.int64),
            token_xy=(
                np.stack([obs.token_xy for obs in observations], axis=0).astype(np.float32)
                if observations
                else np.zeros((0, 2), dtype=np.float32)
            ),
            features=(
                np.stack([obs.feature for obs in observations], axis=0).astype(np.float32)
                if observations
                else np.zeros((0, feature_dim), dtype=np.float32)
            ),
            centers=(
                np.stack([obs.center for obs in observations], axis=0).astype(np.float64)
                if observations
                else np.zeros((0, 3), dtype=np.float64)
            ),
            normals=(
                np.stack([obs.normal for obs in observations], axis=0).astype(np.float32)
                if observations
                else np.zeros((0, 3), dtype=np.float32)
            ),
            covariances=(
                np.stack([obs.covariance for obs in observations], axis=0).astype(np.float32)
                if observations
                else np.zeros((0, 3, 3), dtype=np.float32)
            ),
            support_offsets=np.asarray(support_offsets, dtype=np.int64),
            element_ids=np.asarray(element_ids, dtype=np.int64),
            element_weights=np.asarray(element_weights, dtype=np.float32),
            purity_scores=np.asarray([float(obs.purity_score) for obs in observations], dtype=np.float32),
            component_concentrations=np.asarray(
                [float(obs.component_concentration) for obs in observations],
                dtype=np.float32,
            ),
            quality_scores=np.asarray([float(obs.quality_score) for obs in observations], dtype=np.float32),
            view_directions=(
                np.stack([obs.view_direction for obs in observations], axis=0).astype(np.float32)
                if observations
                else np.zeros((0, 3), dtype=np.float32)
            ),
            metadata=metadata or {},
            source_ids=(
                tuple(str(item) for item in source_ids)
                if source_ids is not None
                else tuple(str(obs.source_id) for obs in observations)
            ),
            observation_strengths=tuple(str(obs.observation_strength) for obs in observations),
            descriptor_weights=np.asarray([float(obs.descriptor_weight) for obs in observations], dtype=np.float32),
        )

    def to_observations(self) -> list[TokenSurfaceObservation]:
        observations = []
        for row in range(len(self)):
            start = int(self.support_offsets[row])
            end = int(self.support_offsets[row + 1])
            purity = float(self.purity_scores[row])
            observations.append(
                TokenSurfaceObservation(
                    image_id=self.image_ids[row],
                    token_index=int(self.token_indices[row]),
                    token_xy=self.token_xy[row],
                    feature=self.features[row],
                    element_ids=self.element_ids[start:end],
                    element_weights=self.element_weights[start:end],
                    center=self.centers[row],
                    normal=self.normals[row],
                    covariance=self.covariances[row],
                    purity_score=purity,
                    component_concentration=float(self.component_concentrations[row]),
                    quality_score=float(self.quality_scores[row]),
                    view_direction=self.view_directions[row],
                    purity_components={"total": purity},
                    source_id=self.source_ids[row],
                    observation_strength=self.observation_strengths[row],
                    descriptor_weight=float(self.descriptor_weights[row]),
                )
            )
        return observations

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_ids_json=np.asarray(json.dumps(list(self.image_ids))),
            token_indices=self.token_indices.astype(np.int64),
            token_xy=self.token_xy.astype(np.float32),
            features=self.features.astype(np.float32),
            centers=self.centers.astype(np.float32),
            normals=self.normals.astype(np.float32),
            covariances=self.covariances.astype(np.float32),
            support_offsets=self.support_offsets.astype(np.int64),
            element_ids=self.element_ids.astype(np.int64),
            element_weights=self.element_weights.astype(np.float32),
            purity_scores=self.purity_scores.astype(np.float32),
            component_concentrations=self.component_concentrations.astype(np.float32),
            quality_scores=self.quality_scores.astype(np.float32),
            view_directions=self.view_directions.astype(np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
            source_ids_json=np.asarray(json.dumps(list(self.source_ids))),
            observation_strengths_json=np.asarray(json.dumps(list(self.observation_strengths))),
            descriptor_weights=self.descriptor_weights.astype(np.float32),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Vfm2DgsObservationBank":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item())) if "metadata_json" in data else {}
            return cls(
                image_ids=tuple(json.loads(str(np.asarray(data["image_ids_json"]).item()))),
                token_indices=np.asarray(data["token_indices"], dtype=np.int64),
                token_xy=np.asarray(data["token_xy"], dtype=np.float32),
                features=np.asarray(data["features"], dtype=np.float32),
                centers=np.asarray(data["centers"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float32),
                covariances=np.asarray(data["covariances"], dtype=np.float32),
                support_offsets=np.asarray(data["support_offsets"], dtype=np.int64),
                element_ids=np.asarray(data["element_ids"], dtype=np.int64),
                element_weights=np.asarray(data["element_weights"], dtype=np.float32),
                purity_scores=np.asarray(data["purity_scores"], dtype=np.float32),
                component_concentrations=np.asarray(data["component_concentrations"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                view_directions=np.asarray(data["view_directions"], dtype=np.float32),
                metadata=metadata,
                source_ids=(
                    tuple(json.loads(str(np.asarray(data["source_ids_json"]).item())))
                    if "source_ids_json" in data
                    else None
                ),
                observation_strengths=(
                    tuple(json.loads(str(np.asarray(data["observation_strengths_json"]).item())))
                    if "observation_strengths_json" in data
                    else None
                ),
                descriptor_weights=(
                    np.asarray(data["descriptor_weights"], dtype=np.float32)
                    if "descriptor_weights" in data
                    else None
                ),
            )


def _observation_bank_row_support(
    bank: Vfm2DgsObservationBank,
    row: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = int(bank.support_offsets[row])
    end = int(bank.support_offsets[row + 1])
    return bank.element_ids[start:end], bank.element_weights[start:end]


def _observation_bank_dedupe_key(
    bank: Vfm2DgsObservationBank,
    row: int,
    weight_precision: int = 6,
) -> tuple[str, int, tuple[int, ...], tuple[float, ...]]:
    element_ids, element_weights = _observation_bank_row_support(bank, row)
    order = np.argsort(element_ids, kind="mergesort")
    sorted_ids = tuple(int(item) for item in element_ids[order].tolist())
    sorted_weights = tuple(
        float(item)
        for item in np.round(np.asarray(element_weights[order], dtype=np.float32), int(weight_precision)).tolist()
    )
    return str(bank.image_ids[row]), int(bank.token_indices[row]), sorted_ids, sorted_weights


def _observation_bank_row_priority(bank: Vfm2DgsObservationBank, row: int) -> tuple[float, float, float, int]:
    support_count = int(bank.support_offsets[row + 1] - bank.support_offsets[row])
    return (
        float(bank.quality_scores[row]),
        float(bank.purity_scores[row]),
        float(bank.component_concentrations[row]),
        support_count,
    )


def merge_vfm_2dgs_observation_banks(
    banks: Sequence[Vfm2DgsObservationBank],
    source_names: Sequence[str] | None = None,
    deduplicate: bool = True,
) -> Vfm2DgsObservationBank:
    """Merge observation banks while preserving source labels and suppressing exact duplicates."""
    if source_names is not None and len(source_names) != len(banks):
        raise ValueError("source_names must match banks")
    resolved_source_names = []
    for idx, bank in enumerate(banks):
        if source_names is not None:
            resolved_source_names.append(str(source_names[idx]))
            continue
        name = str(dict(bank.metadata or {}).get("name", "")).strip()
        resolved_source_names.append(name or f"source_{idx}")

    selected_rows: list[tuple[Vfm2DgsObservationBank, int, str]] = []
    row_by_key: dict[tuple[str, int, tuple[int, ...], tuple[float, ...]], int] = {}
    duplicate_count = 0
    for bank, source_name in zip(banks, resolved_source_names):
        for row in range(len(bank)):
            key = _observation_bank_dedupe_key(bank, row)
            if bool(deduplicate) and key in row_by_key:
                duplicate_count += 1
                existing_index = row_by_key[key]
                existing_bank, existing_row, _existing_source_name = selected_rows[existing_index]
                if _observation_bank_row_priority(bank, row) > _observation_bank_row_priority(existing_bank, existing_row):
                    selected_rows[existing_index] = (bank, row, source_name)
                continue
            row_by_key[key] = len(selected_rows)
            selected_rows.append((bank, row, source_name))

    support_offsets = [0]
    element_ids: list[int] = []
    element_weights: list[float] = []
    for bank, row, _source_name in selected_rows:
        ids, weights = _observation_bank_row_support(bank, row)
        element_ids.extend(np.asarray(ids, dtype=np.int64).tolist())
        element_weights.extend(np.asarray(weights, dtype=np.float32).tolist())
        support_offsets.append(len(element_ids))

    if selected_rows:
        image_ids = tuple(str(bank.image_ids[row]) for bank, row, _source_name in selected_rows)
        token_indices = np.asarray([int(bank.token_indices[row]) for bank, row, _source_name in selected_rows], dtype=np.int64)
        token_xy = np.stack([bank.token_xy[row] for bank, row, _source_name in selected_rows], axis=0).astype(np.float32)
        features = np.stack([bank.features[row] for bank, row, _source_name in selected_rows], axis=0).astype(np.float32)
        centers = np.stack([bank.centers[row] for bank, row, _source_name in selected_rows], axis=0).astype(np.float64)
        normals = np.stack([bank.normals[row] for bank, row, _source_name in selected_rows], axis=0).astype(np.float32)
        covariances = np.stack([bank.covariances[row] for bank, row, _source_name in selected_rows], axis=0).astype(np.float32)
        purity_scores = np.asarray([float(bank.purity_scores[row]) for bank, row, _source_name in selected_rows], dtype=np.float32)
        component_concentrations = np.asarray(
            [float(bank.component_concentrations[row]) for bank, row, _source_name in selected_rows],
            dtype=np.float32,
        )
        quality_scores = np.asarray([float(bank.quality_scores[row]) for bank, row, _source_name in selected_rows], dtype=np.float32)
        observation_strengths = tuple(str(bank.observation_strengths[row]) for bank, row, _source_name in selected_rows)
        descriptor_weights = np.asarray(
            [float(bank.descriptor_weights[row]) for bank, row, _source_name in selected_rows],
            dtype=np.float32,
        )
        view_directions = np.stack(
            [bank.view_directions[row] for bank, row, _source_name in selected_rows],
            axis=0,
        ).astype(np.float32)
        source_ids = tuple(str(source_name) for _bank, _row, source_name in selected_rows)
    else:
        feature_dim = 0
        for bank in banks:
            if bank.features.ndim == 2 and bank.features.shape[1] > 0:
                feature_dim = int(bank.features.shape[1])
                break
        image_ids = tuple()
        token_indices = np.zeros((0,), dtype=np.int64)
        token_xy = np.zeros((0, 2), dtype=np.float32)
        features = np.zeros((0, feature_dim), dtype=np.float32)
        centers = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float32)
        covariances = np.zeros((0, 3, 3), dtype=np.float32)
        purity_scores = np.zeros((0,), dtype=np.float32)
        component_concentrations = np.zeros((0,), dtype=np.float32)
        quality_scores = np.zeros((0,), dtype=np.float32)
        observation_strengths = tuple()
        descriptor_weights = np.zeros((0,), dtype=np.float32)
        view_directions = np.zeros((0, 3), dtype=np.float32)
        source_ids = tuple()

    source_counts = {str(name): int(len(bank)) for name, bank in zip(resolved_source_names, banks)}
    selected_source_counts = {
        str(name): int(sum(1 for _bank, _row, source_name in selected_rows if source_name == name))
        for name in resolved_source_names
    }
    return Vfm2DgsObservationBank(
        image_ids=image_ids,
        token_indices=token_indices,
        token_xy=token_xy,
        features=features,
        centers=centers,
        normals=normals,
        covariances=covariances,
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        element_ids=np.asarray(element_ids, dtype=np.int64),
        element_weights=np.asarray(element_weights, dtype=np.float32),
        purity_scores=purity_scores,
        component_concentrations=component_concentrations,
        quality_scores=quality_scores,
        view_directions=view_directions,
        metadata={
            "stage": "vfm_2dgs_merged_observation_layer",
            "source_names": list(resolved_source_names),
            "source_observation_counts": source_counts,
            "selected_source_observation_counts": selected_source_counts,
            "duplicate_observation_count": int(duplicate_count),
            "deduplicate": bool(deduplicate),
            "dedupe_key": "image_id,token_index,rounded_surface_support",
        },
        source_ids=source_ids,
        observation_strengths=observation_strengths,
        descriptor_weights=descriptor_weights,
    )


@dataclass(frozen=True)
class Vfm2DgsAnchorMap:
    anchor_ids: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    covariances: np.ndarray
    features: np.ndarray
    feature_variances: np.ndarray
    quality_scores: np.ndarray
    purity_scores: np.ndarray
    observation_counts: np.ndarray
    surface_support_counts: np.ndarray
    support_offsets: np.ndarray
    support_element_ids: np.ndarray
    support_weights: np.ndarray
    observed_view_ids: tuple[tuple[str, ...], ...]
    feature_prototypes: np.ndarray | None = None
    feature_prototype_counts: np.ndarray | None = None
    view_bin_features: np.ndarray | None = None
    view_bin_counts: np.ndarray | None = None
    support_parent_gaussian_indices: np.ndarray | None = None
    covisibility_offsets: np.ndarray | None = None
    covisibility_anchor_ids: np.ndarray | None = None
    covisibility_scores: np.ndarray | None = None
    distinctiveness_scores: np.ndarray | None = None
    stability_scores: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        count = int(anchor_ids.shape[0])
        centers = np.asarray(self.centers, dtype=np.float64)
        normals = np.asarray(self.normals, dtype=np.float32)
        covariances = np.asarray(self.covariances, dtype=np.float32)
        features = np.asarray(self.features, dtype=np.float32)
        if centers.shape != (count, 3):
            raise ValueError("centers must have shape (N, 3)")
        if normals.shape != (count, 3):
            raise ValueError("normals must have shape (N, 3)")
        if covariances.shape != (count, 3, 3):
            raise ValueError("covariances must have shape (N, 3, 3)")
        if features.ndim != 2 or features.shape[0] != count:
            raise ValueError("features must have shape (N, C)")
        if self.feature_prototypes is None:
            feature_prototypes = features[:, None, :].astype(np.float32, copy=False)
        else:
            feature_prototypes = np.asarray(self.feature_prototypes, dtype=np.float32)
        if feature_prototypes.ndim != 3 or feature_prototypes.shape[0] != count or feature_prototypes.shape[2] != features.shape[1]:
            raise ValueError("feature_prototypes must have shape (N, P, C)")
        if self.feature_prototype_counts is None:
            feature_prototype_counts = np.ones((count,), dtype=np.int64)
        else:
            feature_prototype_counts = np.asarray(self.feature_prototype_counts, dtype=np.int64).reshape(-1)
        if feature_prototype_counts.shape != (count,):
            raise ValueError("feature_prototype_counts must have shape (N,)")
        if self.view_bin_features is None:
            view_bin_features = features[:, None, :].astype(np.float32, copy=False)
        else:
            view_bin_features = np.asarray(self.view_bin_features, dtype=np.float32)
        if view_bin_features.ndim != 3 or view_bin_features.shape[0] != count or view_bin_features.shape[2] != features.shape[1]:
            raise ValueError("view_bin_features must have shape (N, B, C)")
        if self.view_bin_counts is None:
            view_bin_counts = np.ones((count, view_bin_features.shape[1]), dtype=np.int64)
        else:
            view_bin_counts = np.asarray(self.view_bin_counts, dtype=np.int64)
        if view_bin_counts.shape != view_bin_features.shape[:2]:
            raise ValueError("view_bin_counts must have shape (N, B)")
        for name in ("feature_variances", "quality_scores", "purity_scores"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name, default_value in (("distinctiveness_scores", 0.0), ("stability_scores", 1.0)):
            raw_value = getattr(self, name)
            if raw_value is None:
                value = np.full((count,), float(default_value), dtype=np.float32)
            else:
                value = np.asarray(raw_value, dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        for name in ("observation_counts", "surface_support_counts"):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, value)
        support_offsets = np.asarray(self.support_offsets, dtype=np.int64).reshape(-1)
        if support_offsets.shape != (count + 1,):
            raise ValueError("support_offsets must have shape (N + 1,)")
        support_element_ids = np.asarray(self.support_element_ids, dtype=np.int64).reshape(-1)
        support_weights = np.asarray(self.support_weights, dtype=np.float32).reshape(-1)
        if support_element_ids.shape != support_weights.shape:
            raise ValueError("support_element_ids and support_weights must match")
        if self.support_parent_gaussian_indices is None:
            support_parent_gaussian_indices = support_element_ids.copy()
        else:
            support_parent_gaussian_indices = np.asarray(self.support_parent_gaussian_indices, dtype=np.int64).reshape(-1)
        if support_parent_gaussian_indices.shape != support_element_ids.shape:
            raise ValueError("support_parent_gaussian_indices must match support_element_ids")
        observed_view_ids = tuple(tuple(str(item) for item in row) for row in self.observed_view_ids)
        if len(observed_view_ids) != count:
            raise ValueError("observed_view_ids must have length N")
        if self.covisibility_offsets is None:
            covisibility_offsets = np.zeros((count + 1,), dtype=np.int64)
        else:
            covisibility_offsets = np.asarray(self.covisibility_offsets, dtype=np.int64).reshape(-1)
        if covisibility_offsets.shape != (count + 1,):
            raise ValueError("covisibility_offsets must have shape (N + 1,)")
        if self.covisibility_anchor_ids is None:
            covisibility_anchor_ids = np.zeros((0,), dtype=np.int64)
        else:
            covisibility_anchor_ids = np.asarray(self.covisibility_anchor_ids, dtype=np.int64).reshape(-1)
        if self.covisibility_scores is None:
            covisibility_scores = np.zeros((covisibility_anchor_ids.shape[0],), dtype=np.float32)
        else:
            covisibility_scores = np.asarray(self.covisibility_scores, dtype=np.float32).reshape(-1)
        if covisibility_scores.shape != covisibility_anchor_ids.shape:
            raise ValueError("covisibility_scores must match covisibility_anchor_ids")
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "normals", _normalize_vectors(normals))
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_prototypes", feature_prototypes)
        object.__setattr__(self, "feature_prototype_counts", feature_prototype_counts)
        object.__setattr__(self, "view_bin_features", view_bin_features)
        object.__setattr__(self, "view_bin_counts", view_bin_counts)
        object.__setattr__(self, "support_offsets", support_offsets)
        object.__setattr__(self, "support_element_ids", support_element_ids)
        object.__setattr__(self, "support_parent_gaussian_indices", support_parent_gaussian_indices)
        object.__setattr__(self, "support_weights", support_weights)
        object.__setattr__(self, "observed_view_ids", observed_view_ids)
        object.__setattr__(self, "covisibility_offsets", covisibility_offsets)
        object.__setattr__(self, "covisibility_anchor_ids", covisibility_anchor_ids)
        object.__setattr__(self, "covisibility_scores", covisibility_scores)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(self.anchor_ids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchor_ids=self.anchor_ids,
            centers=self.centers.astype(np.float32),
            normals=self.normals.astype(np.float32),
            covariances=self.covariances.astype(np.float32),
            features=self.features.astype(np.float32),
            feature_prototypes=self.feature_prototypes.astype(np.float32),
            feature_prototype_counts=self.feature_prototype_counts.astype(np.int64),
            view_bin_features=self.view_bin_features.astype(np.float32),
            view_bin_counts=self.view_bin_counts.astype(np.int64),
            feature_variances=self.feature_variances.astype(np.float32),
            quality_scores=self.quality_scores.astype(np.float32),
            purity_scores=self.purity_scores.astype(np.float32),
            observation_counts=self.observation_counts.astype(np.int64),
            surface_support_counts=self.surface_support_counts.astype(np.int64),
            support_offsets=self.support_offsets.astype(np.int64),
            support_element_ids=self.support_element_ids.astype(np.int64),
            support_parent_gaussian_indices=self.support_parent_gaussian_indices.astype(np.int64),
            support_weights=self.support_weights.astype(np.float32),
            covisibility_offsets=self.covisibility_offsets.astype(np.int64),
            covisibility_anchor_ids=self.covisibility_anchor_ids.astype(np.int64),
            covisibility_scores=self.covisibility_scores.astype(np.float32),
            distinctiveness_scores=self.distinctiveness_scores.astype(np.float32),
            stability_scores=self.stability_scores.astype(np.float32),
            observed_view_ids_json=np.asarray(json.dumps([list(row) for row in self.observed_view_ids])),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "Vfm2DgsAnchorMap":
        with np.load(Path(path), allow_pickle=True) as data:
            observed_view_ids = tuple(
                tuple(str(item) for item in row)
                for row in json.loads(str(np.asarray(data["observed_view_ids_json"]).item()))
            )
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item())) if "metadata_json" in data else {}
            return cls(
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                normals=np.asarray(data["normals"], dtype=np.float32),
                covariances=np.asarray(data["covariances"], dtype=np.float32),
                features=np.asarray(data["features"], dtype=np.float32),
                feature_prototypes=(
                    np.asarray(data["feature_prototypes"], dtype=np.float32)
                    if "feature_prototypes" in data
                    else None
                ),
                feature_prototype_counts=(
                    np.asarray(data["feature_prototype_counts"], dtype=np.int64)
                    if "feature_prototype_counts" in data
                    else None
                ),
                view_bin_features=(
                    np.asarray(data["view_bin_features"], dtype=np.float32)
                    if "view_bin_features" in data
                    else None
                ),
                view_bin_counts=(
                    np.asarray(data["view_bin_counts"], dtype=np.int64)
                    if "view_bin_counts" in data
                    else None
                ),
                feature_variances=np.asarray(data["feature_variances"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                purity_scores=np.asarray(data["purity_scores"], dtype=np.float32),
                observation_counts=np.asarray(data["observation_counts"], dtype=np.int64),
                surface_support_counts=np.asarray(data["surface_support_counts"], dtype=np.int64),
                support_offsets=np.asarray(data["support_offsets"], dtype=np.int64),
                support_element_ids=np.asarray(data["support_element_ids"], dtype=np.int64),
                support_parent_gaussian_indices=(
                    np.asarray(data["support_parent_gaussian_indices"], dtype=np.int64)
                    if "support_parent_gaussian_indices" in data
                    else None
                ),
                support_weights=np.asarray(data["support_weights"], dtype=np.float32),
                covisibility_offsets=(
                    np.asarray(data["covisibility_offsets"], dtype=np.int64)
                    if "covisibility_offsets" in data
                    else None
                ),
                covisibility_anchor_ids=(
                    np.asarray(data["covisibility_anchor_ids"], dtype=np.int64)
                    if "covisibility_anchor_ids" in data
                    else None
                ),
                covisibility_scores=(
                    np.asarray(data["covisibility_scores"], dtype=np.float32)
                    if "covisibility_scores" in data
                    else None
                ),
                observed_view_ids=observed_view_ids,
                distinctiveness_scores=(
                    np.asarray(data["distinctiveness_scores"], dtype=np.float32)
                    if "distinctiveness_scores" in data
                    else None
                ),
                stability_scores=(
                    np.asarray(data["stability_scores"], dtype=np.float32)
                    if "stability_scores" in data
                    else None
                ),
                metadata=metadata,
            )


@dataclass(frozen=True)
class Vfm2DgsDescriptorIndex:
    anchor_ids: np.ndarray
    prototype_ids: np.ndarray
    centers: np.ndarray
    descriptors: np.ndarray
    quality_scores: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        anchor_ids = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        prototype_ids = np.asarray(self.prototype_ids, dtype=np.int64).reshape(-1)
        centers = np.asarray(self.centers, dtype=np.float64)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        quality_scores = np.asarray(self.quality_scores, dtype=np.float32).reshape(-1)
        count = int(anchor_ids.shape[0])
        if prototype_ids.shape != (count,):
            raise ValueError("prototype_ids must match anchor_ids")
        if centers.shape != (count, 3):
            raise ValueError("centers must have shape (N, 3)")
        if descriptors.ndim != 2 or descriptors.shape[0] != count:
            raise ValueError("descriptors must have shape (N, C)")
        if quality_scores.shape != (count,):
            raise ValueError("quality_scores must match anchor_ids")
        descriptors, _valid = normalize_rows(descriptors)
        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "prototype_ids", prototype_ids)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "quality_scores", quality_scores)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __len__(self) -> int:
        return int(self.anchor_ids.shape[0])

    def search(self, query_descriptors: np.ndarray, top_k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = np.asarray(query_descriptors, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("query_descriptors must have shape (Q, C)")
        if queries.shape[1] != self.descriptors.shape[1]:
            raise ValueError("query descriptor dim must match index descriptor dim")
        if len(self) == 0:
            shape = (queries.shape[0], 0)
            return (
                np.zeros(shape, dtype=np.int64),
                np.zeros(shape, dtype=np.int64),
                np.zeros(shape, dtype=np.float32),
            )
        queries, _valid = normalize_rows(queries)
        scores = queries @ self.descriptors.T
        keep = min(max(int(top_k), 1), int(scores.shape[1]))
        partition = np.argpartition(-scores, kth=keep - 1, axis=1)[:, :keep]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1, kind="mergesort")
        cols = np.take_along_axis(partition, order, axis=1)
        top_scores = np.take_along_axis(scores, cols, axis=1).astype(np.float32, copy=False)
        return self.anchor_ids[cols], self.prototype_ids[cols], top_scores

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            anchor_ids=self.anchor_ids.astype(np.int64),
            prototype_ids=self.prototype_ids.astype(np.int64),
            centers=self.centers.astype(np.float32),
            descriptors=self.descriptors.astype(np.float32),
            quality_scores=self.quality_scores.astype(np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    def save_faiss(self, path: Path) -> None:
        try:
            import faiss
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("faiss is required to export a descriptor ANN index") from exc
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(int(self.descriptors.shape[1]))
        index.add(np.ascontiguousarray(self.descriptors.astype(np.float32, copy=False)))
        faiss.write_index(index, str(path))

    @classmethod
    def load_npz(cls, path: Path) -> "Vfm2DgsDescriptorIndex":
        with np.load(Path(path), allow_pickle=True) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item())) if "metadata_json" in data else {}
            return cls(
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64),
                prototype_ids=np.asarray(data["prototype_ids"], dtype=np.int64),
                centers=np.asarray(data["centers"], dtype=np.float64),
                descriptors=np.asarray(data["descriptors"], dtype=np.float32),
                quality_scores=np.asarray(data["quality_scores"], dtype=np.float32),
                metadata=metadata,
            )


def _normalize_vectors(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, float(eps))


def _normalize_surface_basis(
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    normals: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = _normalize_vectors(np.asarray(normals, dtype=np.float32), eps=eps)
    first = np.asarray(tangent1, dtype=np.float32)
    first = first - normal * np.sum(first * normal, axis=1, keepdims=True)
    first_norm = np.linalg.norm(first, axis=1)
    bad_first = first_norm <= float(eps)
    if np.any(bad_first):
        fallback_first, _fallback_second = _fallback_tangent_basis(normal[bad_first])
        first[bad_first] = fallback_first
        first_norm = np.linalg.norm(first, axis=1)
    first = first / np.maximum(first_norm[:, None], float(eps))

    second = np.asarray(tangent2, dtype=np.float32)
    second = second - normal * np.sum(second * normal, axis=1, keepdims=True)
    second = second - first * np.sum(second * first, axis=1, keepdims=True)
    second_norm = np.linalg.norm(second, axis=1)
    bad_second = second_norm <= float(eps)
    if np.any(bad_second):
        second[bad_second] = np.cross(normal[bad_second], first[bad_second])
        second_norm = np.linalg.norm(second, axis=1)
    second = second / np.maximum(second_norm[:, None], float(eps))

    handedness = np.sum(np.cross(first, second) * normal, axis=1)
    second[handedness < 0.0] *= -1.0
    return first.astype(np.float32, copy=False), second.astype(np.float32, copy=False), normal.astype(np.float32, copy=False)


def _signed_safe_denominator(values, eps: float = 1e-12):
    import torch

    value = torch.as_tensor(values)
    sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(torch.abs(value) < float(eps), sign * float(eps), value)


def build_surface_elements_from_2dgs_source(
    source: GaussianVFMSource,
    min_opacity: float = 0.0,
    max_scale: float | None = None,
    adjacency_radius: float = 0.05,
    adjacency_element_radius_cap: float = 0.0,
    normal_cosine_threshold: float = 0.8,
    virtual_cell_max_scale: float = 0.0,
    virtual_cell_grid_cap: int = 8,
) -> SurfaceElementMap:
    keep = np.asarray(source.opacity, dtype=np.float32) >= float(min_opacity)
    if max_scale is not None:
        keep &= np.asarray(source.scale, dtype=np.float32) <= float(max_scale)
    rows = np.flatnonzero(keep)
    centers = np.asarray(source.xyz, dtype=np.float64)[rows]
    scale_xyz = np.asarray(source.scale_xyz, dtype=np.float32)[rows]
    normals = source.normal[rows] if source.normal is not None else np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (rows.size, 1))
    normals = _normalize_vectors(normals)
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(source, rows, normals)
    (
        element_ids,
        parent_gaussian_indices,
        centers,
        tangent1,
        tangent2,
        normals,
        scale1,
        scale2,
        opacity,
        area,
        virtual_split_count,
    ) = _maybe_split_virtual_surface_cells(
        parent_ids=np.asarray(source.gaussian_indices, dtype=np.int64)[rows],
        centers=centers,
        tangent1=tangent1,
        tangent2=tangent2,
        normals=normals,
        scale1=scale1,
        scale2=scale2,
        opacity=np.asarray(source.opacity, dtype=np.float32)[rows],
        virtual_cell_max_scale=float(virtual_cell_max_scale),
        virtual_cell_grid_cap=int(virtual_cell_grid_cap),
    )
    adjacency = _build_adjacency(
        centers,
        normals,
        float(adjacency_radius),
        float(normal_cosine_threshold),
        element_radius=np.maximum(scale1, scale2),
        element_radius_cap=float(adjacency_element_radius_cap),
    )
    return SurfaceElementMap(
        element_ids=element_ids,
        parent_gaussian_indices=parent_gaussian_indices,
        centers=centers,
        tangent1=tangent1,
        tangent2=tangent2,
        normals=normals,
        scale1=scale1.astype(np.float32, copy=False),
        scale2=scale2.astype(np.float32, copy=False),
        opacity=opacity,
        area=area,
        adjacency=adjacency,
        metadata={
            "source_gaussian_count": int(source.xyz.shape[0]),
            "element_count": int(rows.size),
            "surface_element_count": int(element_ids.size),
            "virtual_split_count": int(virtual_split_count),
            "min_opacity": float(min_opacity),
            "max_scale": None if max_scale is None else float(max_scale),
            "adjacency_radius": float(adjacency_radius),
            "adjacency_element_radius_cap": float(adjacency_element_radius_cap),
            "normal_cosine_threshold": float(normal_cosine_threshold),
            "virtual_cell_max_scale": float(virtual_cell_max_scale),
            "virtual_cell_grid_cap": int(virtual_cell_grid_cap),
        },
    )


def _surface_tangent_axes_and_scales(
    source: GaussianVFMSource,
    rows: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale_xyz = np.asarray(source.scale_xyz, dtype=np.float32)[rows]
    if source.rotation is not None:
        matrices = _quaternion_rotation_matrices(np.asarray(source.rotation, dtype=np.float32)[rows])
        order = np.argsort(-scale_xyz, axis=1)
        row_ids = np.arange(rows.size, dtype=np.int64)
        tangent1 = matrices[row_ids, :, order[:, 0]]
        tangent2 = matrices[row_ids, :, order[:, 1]]
        scale1 = scale_xyz[row_ids, order[:, 0]]
        scale2 = scale_xyz[row_ids, order[:, 1]]
        tangent1, tangent2, _normals = _normalize_surface_basis(tangent1, tangent2, normals)
        return tangent1, tangent2, scale1, scale2
    tangent1, tangent2 = _fallback_tangent_basis(normals)
    scale_sorted = np.sort(scale_xyz, axis=1)
    scale1 = scale_sorted[:, -1]
    scale2 = scale_sorted[:, -2] if scale_sorted.shape[1] >= 2 else scale_sorted[:, -1]
    return tangent1, tangent2, scale1, scale2


def _fallback_tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize_vectors(np.asarray(normals, dtype=np.float32))
    up = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (normal.shape[0], 1))
    parallel = np.abs(np.sum(normal * up, axis=1)) > 0.9
    up[parallel] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tangent1 = np.cross(up, normal)
    tangent1 = _normalize_vectors(tangent1)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = _normalize_vectors(tangent2)
    return tangent1, tangent2


def _rotation_matrices_to_quaternions_wxyz(matrices: np.ndarray) -> np.ndarray:
    mats = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    quats = np.zeros((mats.shape[0], 4), dtype=np.float32)
    for idx, matrix in enumerate(mats):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                qw = (matrix[2, 1] - matrix[1, 2]) / scale
                qx = 0.25 * scale
                qy = (matrix[0, 1] + matrix[1, 0]) / scale
                qz = (matrix[0, 2] + matrix[2, 0]) / scale
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                qw = (matrix[0, 2] - matrix[2, 0]) / scale
                qx = (matrix[0, 1] + matrix[1, 0]) / scale
                qy = 0.25 * scale
                qz = (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                qw = (matrix[1, 0] - matrix[0, 1]) / scale
                qx = (matrix[0, 2] + matrix[2, 0]) / scale
                qy = (matrix[1, 2] + matrix[2, 1]) / scale
                qz = 0.25 * scale
        quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
        quats[idx] = quat.astype(np.float32)
    return quats


def _surface_element_quaternions_and_scales(elements: SurfaceElementMap) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.stack([elements.tangent1, elements.tangent2, elements.normals], axis=2).astype(np.float64)
    quats = _rotation_matrices_to_quaternions_wxyz(matrices)
    scales = np.stack(
        [
            np.maximum(elements.scale1, 1e-4),
            np.maximum(elements.scale2, 1e-4),
            np.full((len(elements),), 1e-4, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return quats, scales


def _maybe_split_virtual_surface_cells(
    parent_ids: np.ndarray,
    centers: np.ndarray,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    normals: np.ndarray,
    scale1: np.ndarray,
    scale2: np.ndarray,
    opacity: np.ndarray,
    virtual_cell_max_scale: float,
    virtual_cell_grid_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if float(virtual_cell_max_scale) <= 0.0:
        return (
            np.asarray(parent_ids, dtype=np.int64),
            np.asarray(parent_ids, dtype=np.int64),
            np.asarray(centers, dtype=np.float64),
            np.asarray(tangent1, dtype=np.float32),
            np.asarray(tangent2, dtype=np.float32),
            np.asarray(normals, dtype=np.float32),
            np.asarray(scale1, dtype=np.float32),
            np.asarray(scale2, dtype=np.float32),
            np.asarray(opacity, dtype=np.float32),
            (np.pi * np.asarray(scale1, dtype=np.float32) * np.asarray(scale2, dtype=np.float32)).astype(np.float32),
            0,
        )
    cap = max(int(virtual_cell_grid_cap), 1)
    output_ids: list[int] = []
    output_parents: list[int] = []
    output_centers: list[np.ndarray] = []
    output_tangent1: list[np.ndarray] = []
    output_tangent2: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_scale1: list[float] = []
    output_scale2: list[float] = []
    output_opacity: list[float] = []
    virtual_index = 0
    split_count = 0
    for idx, parent_id in enumerate(np.asarray(parent_ids, dtype=np.int64).tolist()):
        n1 = min(max(1, int(np.ceil(float(scale1[idx]) / float(virtual_cell_max_scale)))), cap)
        n2 = min(max(1, int(np.ceil(float(scale2[idx]) / float(virtual_cell_max_scale)))), cap)
        cell_count = int(n1 * n2)
        if cell_count <= 1:
            output_ids.append(int(parent_id))
        else:
            split_count += 1
        cell_scale1 = float(scale1[idx]) / float(n1)
        cell_scale2 = float(scale2[idx]) / float(n2)
        for a_idx in range(n1):
            a = -float(scale1[idx]) + (float(a_idx) + 0.5) * 2.0 * cell_scale1
            for b_idx in range(n2):
                b = -float(scale2[idx]) + (float(b_idx) + 0.5) * 2.0 * cell_scale2
                if cell_count > 1:
                    output_ids.append(-(virtual_index + 1))
                    virtual_index += 1
                output_parents.append(int(parent_id))
                output_centers.append(
                    np.asarray(centers[idx], dtype=np.float64)
                    + np.asarray(tangent1[idx], dtype=np.float64) * a
                    + np.asarray(tangent2[idx], dtype=np.float64) * b
                )
                output_tangent1.append(np.asarray(tangent1[idx], dtype=np.float32))
                output_tangent2.append(np.asarray(tangent2[idx], dtype=np.float32))
                output_normals.append(np.asarray(normals[idx], dtype=np.float32))
                output_scale1.append(cell_scale1)
                output_scale2.append(cell_scale2)
                output_opacity.append(float(opacity[idx]))
    scale1_arr = np.asarray(output_scale1, dtype=np.float32)
    scale2_arr = np.asarray(output_scale2, dtype=np.float32)
    return (
        np.asarray(output_ids, dtype=np.int64),
        np.asarray(output_parents, dtype=np.int64),
        np.stack(output_centers, axis=0).astype(np.float64),
        np.stack(output_tangent1, axis=0).astype(np.float32),
        np.stack(output_tangent2, axis=0).astype(np.float32),
        np.stack(output_normals, axis=0).astype(np.float32),
        scale1_arr,
        scale2_arr,
        np.asarray(output_opacity, dtype=np.float32),
        (np.pi * scale1_arr * scale2_arr).astype(np.float32),
        split_count,
    )


def _build_adjacency(
    centers: np.ndarray,
    normals: np.ndarray,
    radius: float,
    normal_cosine_threshold: float,
    element_radius: np.ndarray | None = None,
    element_radius_cap: float = 0.0,
) -> tuple[np.ndarray, ...]:
    if centers.shape[0] == 0:
        return ()
    element_radius_arr = (
        np.zeros((centers.shape[0],), dtype=np.float64)
        if element_radius is None
        else np.asarray(element_radius, dtype=np.float64).reshape(-1)
    )
    if element_radius_arr.shape != (centers.shape[0],):
        raise ValueError("element_radius must have shape (N,)")
    if float(element_radius_cap) < 0.0:
        raise ValueError("element_radius_cap must be non-negative")
    effective_radius = np.maximum(element_radius_arr, 0.0)
    if float(element_radius_cap) > 0.0:
        effective_radius = np.minimum(effective_radius, float(element_radius_cap))
    maximum_radius = float(np.max(effective_radius, initial=0.0))
    search_radii = float(radius) + effective_radius + maximum_radius
    if float(np.max(search_radii, initial=0.0)) <= 0.0:
        return tuple(np.zeros((0,), dtype=np.int64) for _ in range(centers.shape[0]))
    tree = cKDTree(np.asarray(centers, dtype=np.float64))
    # Per-element radii avoid one unusually large splat turning the complete
    # reconstruction into a near-all-pairs query.  The explicit cap affects
    # topology only; rendering retains the original 2DGS disk extent.
    neighbors = tree.query_ball_point(
        np.asarray(centers, dtype=np.float64),
        r=search_radii,
    )
    adjacency = []
    for row, candidates in enumerate(neighbors):
        current = []
        for col in candidates:
            if int(col) == int(row):
                continue
            if float(np.dot(normals[row], normals[int(col)])) < float(normal_cosine_threshold):
                continue
            center_distance = float(np.linalg.norm(centers[row] - centers[int(col)]))
            support_distance = float(radius) + float(effective_radius[row]) + float(effective_radius[int(col)])
            if center_distance > support_distance:
                continue
            current.append(int(col))
        adjacency.append(np.asarray(sorted(current), dtype=np.int64))
    return tuple(adjacency)


def _selected_token_indices(feature_map: np.ndarray, cfg: Vfm2DgsMappingConfig) -> np.ndarray:
    saliency_2d = vfm_token_saliency(feature_map, mode=cfg.saliency_mode)
    saliency = saliency_2d.reshape(-1)
    if saliency.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if cfg.token_selection_mode == "surface":
        selected = np.flatnonzero(saliency >= float(cfg.min_token_saliency))
        return np.asarray(selected, dtype=np.int64)
    if cfg.token_selection_mode == "all":
        selected = np.flatnonzero(saliency >= float(cfg.min_token_saliency))
        return np.asarray(selected, dtype=np.int64)
    keep = max(1, int(np.floor(float(saliency.size) * float(cfg.token_top_fraction))))
    keep = min(keep, int(saliency.size))
    if cfg.token_selection_mode == "grid_top":
        height, width = saliency_2d.shape
        rows = int(cfg.token_grid_rows)
        cols = int(cfg.token_grid_cols)
        per_cell = int(cfg.max_tokens_per_cell)
        if per_cell <= 0:
            per_cell = max(1, int(np.ceil(float(keep) / float(rows * cols))))
        selected_cells = []
        y_edges = np.linspace(0, height, rows + 1, dtype=np.int64)
        x_edges = np.linspace(0, width, cols + 1, dtype=np.int64)
        for row in range(rows):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            for col in range(cols):
                x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
                if y1 <= y0 or x1 <= x0:
                    continue
                local = saliency_2d[y0:y1, x0:x1].reshape(-1)
                if local.size == 0:
                    continue
                local_order = np.argsort(-local, kind="mergesort")
                taken = 0
                for local_index in local_order.tolist():
                    value = float(local[int(local_index)])
                    if value < float(cfg.min_token_saliency):
                        continue
                    yy = y0 + int(local_index) // max(x1 - x0, 1)
                    xx = x0 + int(local_index) % max(x1 - x0, 1)
                    selected_cells.append(yy * int(width) + xx)
                    taken += 1
                    if taken >= per_cell:
                        break
        if not selected_cells:
            return np.zeros((0,), dtype=np.int64)
        unique = np.asarray(sorted(set(int(item) for item in selected_cells)), dtype=np.int64)
        if unique.size > keep and int(cfg.max_tokens_per_cell) <= 0:
            order = np.argsort(-saliency[unique], kind="mergesort")[:keep]
            unique = unique[order]
        return np.asarray(unique, dtype=np.int64)
    order = np.argsort(-saliency, kind="mergesort")
    selected = order[:keep]
    selected = selected[saliency[selected] >= float(cfg.min_token_saliency)]
    return np.asarray(selected, dtype=np.int64)


def surface_supported_token_indices_from_hits(
    feature_map: np.ndarray,
    pixel_ids: np.ndarray,
    weights_per_hit: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
) -> np.ndarray:
    feature_map_arr = np.asarray(feature_map, dtype=np.float32)
    if feature_map_arr.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    _channels, height, width = feature_map_arr.shape
    token_count = int(height * width)
    pixel_ids_arr = np.asarray(pixel_ids, dtype=np.int64).reshape(-1)
    weights_arr = np.asarray(weights_per_hit, dtype=np.float64).reshape(-1)
    if pixel_ids_arr.shape != weights_arr.shape:
        raise ValueError("pixel_ids and weights_per_hit must have matching shape")
    valid = (pixel_ids_arr >= 0) & (pixel_ids_arr < token_count) & np.isfinite(weights_arr)
    valid &= weights_arr > 0.0
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)
    totals = np.bincount(pixel_ids_arr[valid], weights=weights_arr[valid], minlength=token_count)
    selected = np.flatnonzero(totals >= float(cfg.min_surface_token_contribution))
    if selected.size == 0:
        return np.zeros((0,), dtype=np.int64)
    saliency = vfm_token_saliency(feature_map_arr, mode=cfg.saliency_mode).reshape(-1)
    if float(cfg.min_token_saliency) > 0.0:
        selected = selected[saliency[selected] >= float(cfg.min_token_saliency)]
    if selected.size == 0:
        return np.zeros((0,), dtype=np.int64)
    ranking_scores = totals[selected].astype(np.float64, copy=True)
    if float(cfg.surface_token_saliency_power) > 0.0:
        saliency_term = np.maximum(saliency[selected].astype(np.float64), 1e-12)
        ranking_scores *= np.power(saliency_term, float(cfg.surface_token_saliency_power))
    order = np.lexsort((-saliency[selected], -ranking_scores))
    selected = selected[order]
    if int(cfg.max_surface_tokens) > 0:
        selected = selected[: int(cfg.max_surface_tokens)]
    return np.asarray(selected, dtype=np.int64)


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (-rotation.T @ translation).astype(np.float64, copy=False)


def _view_direction_to_camera(pose_w2c: np.ndarray, surface_center: np.ndarray) -> np.ndarray:
    camera_center = _camera_center_from_w2c(pose_w2c)
    direction = camera_center - np.asarray(surface_center, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (direction / norm).astype(np.float32, copy=False)


def _compute_view_weights(elements: SurfaceElementMap, pose_w2c: np.ndarray, power: float) -> np.ndarray:
    if float(power) <= 0.0:
        return np.ones((len(elements),), dtype=np.float32)
    center = _camera_center_from_w2c(pose_w2c)
    directions = center[None, :] - elements.centers
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    cosines = np.abs(np.sum(elements.normals.astype(np.float64) * directions, axis=1))
    return np.power(np.clip(cosines, 0.0, 1.0), float(power)).astype(np.float32, copy=False)


def _camera_focal_grid(camera, width: int, height: int) -> float:
    params = tuple(float(value) for value in camera.params)
    model_id = int(camera.model_id)
    if model_id in {0, 2, 3} and len(params) >= 1:
        fx = fy = params[0]
    elif model_id in {1, 4} and len(params) >= 2:
        fx, fy = params[:2]
    elif len(params) >= 1:
        fx = fy = params[0]
    else:
        fx = fy = max(float(camera.width), float(camera.height))
    scale_x = float(width) / max(float(camera.width), 1.0)
    scale_y = float(height) / max(float(camera.height), 1.0)
    return float(0.5 * (abs(fx) * scale_x + abs(fy) * scale_y))


def estimate_virtual_cell_max_scale_for_token_projection(
    source: GaussianVFMSource,
    views: Sequence[GaussianVFMFeatureView],
    target_projected_radius_px: float = 1.0,
    depth_quantile: float = 0.5,
    max_samples: int = 0,
    min_opacity: float = 0.0,
) -> float:
    """Estimate a world-space virtual-cell scale from projected VFM token size.

    A 2DGS disk with world tangent radius `s` projects to roughly
    `s * focal / depth` token-grid pixels. This helper inverts that relation
    over visible reference views, so virtual cells are sized for VFM token
    footprints rather than a scene-global world threshold.
    """
    if float(target_projected_radius_px) <= 0.0:
        raise ValueError("target_projected_radius_px must be positive")
    if not 0.0 <= float(depth_quantile) <= 1.0:
        raise ValueError("depth_quantile must be in [0, 1]")
    xyz = np.asarray(source.xyz, dtype=np.float64)
    if xyz.size == 0 or not views:
        return 0.0
    rows = np.arange(xyz.shape[0], dtype=np.int64)
    if float(min_opacity) > 0.0:
        rows = rows[np.asarray(source.opacity, dtype=np.float32)[rows] >= float(min_opacity)]
    if int(max_samples) > 0 and rows.size > int(max_samples):
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(rows, size=int(max_samples), replace=False))
    if rows.size == 0:
        return 0.0
    thresholds: list[np.ndarray] = []
    for view in views:
        feature_map = np.asarray(view.feature_map)
        if feature_map.ndim != 3:
            continue
        _channels, height, width = feature_map.shape
        uv, depth = _project_xyz_to_grid(xyz[rows], view.pose_w2c, view.camera, int(width), int(height))
        valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
        valid &= depth > 1e-8
        valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width))
        valid &= (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
        if not np.any(valid):
            continue
        focal_grid = _camera_focal_grid(view.camera, int(width), int(height))
        local = float(target_projected_radius_px) * depth[valid] / max(float(focal_grid), 1e-8)
        thresholds.append(local.astype(np.float64, copy=False))
    if not thresholds:
        return 0.0
    values = np.concatenate(thresholds, axis=0)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, float(depth_quantile)))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
    mean = float(np.sum(value_arr * weight_arr))
    return float(np.sum(np.square(value_arr - mean) * weight_arr))


def _compute_purity_components(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
    depth: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
    component_concentration: float,
) -> dict[str, float]:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    depth_var = _weighted_variance(np.asarray(depth, dtype=np.float64)[rows], weights)
    depth_score = float(np.exp(-depth_var / max(float(cfg.depth_purity_sigma) ** 2, 1e-12)))
    normal = np.sum(elements.normals[rows].astype(np.float64) * weights[:, None], axis=0)
    normal_score = float(np.clip(np.linalg.norm(normal), 0.0, 1.0))
    if float(cfg.normal_purity_power) > 0.0:
        normal_score = float(np.power(normal_score, float(cfg.normal_purity_power)))
    opacity_score = float(np.clip(np.sum(elements.opacity[rows].astype(np.float64) * weights), 0.0, 1.0))
    component_score = float(np.clip(component_concentration, 0.0, 1.0))
    effective_support = float(1.0 / max(float(np.sum(np.square(weights))), 1e-12))
    if float(cfg.max_effective_support_elements) > 0.0:
        capacity_score = float(
            np.clip(float(cfg.max_effective_support_elements) / max(effective_support, 1e-12), 0.0, 1.0)
        )
    else:
        capacity_score = 1.0
    total = float(np.clip(depth_score * normal_score * opacity_score * component_score * capacity_score, 0.0, 1.0))
    return {
        "depth": depth_score,
        "normal": normal_score,
        "opacity": opacity_score,
        "component": component_score,
        "capacity": capacity_score,
        "effective_support": effective_support,
        "total": total,
    }


def _component_summary(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    row_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if row_arr.size == 0 or weight_arr.size == 0:
        return {
            "component_count": 0,
            "dominant_concentration": 1.0,
            "dominant_rows": row_arr,
            "dominant_weights": weight_arr.astype(np.float32, copy=False),
        }
    if row_arr.shape != weight_arr.shape:
        raise ValueError("rows and weights must have matching shape")
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
    row_set = {int(row) for row in row_arr.tolist()}
    weight_by_row = {int(row): float(weight) for row, weight in zip(row_arr.tolist(), weight_arr.tolist())}
    components = []
    visited = set()
    for start in row_arr.tolist():
        start = int(start)
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in elements.adjacency[current].tolist():
                neighbor = int(neighbor)
                if neighbor in row_set and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    if not components:
        return {
            "component_count": 0,
            "dominant_concentration": 1.0,
            "dominant_rows": row_arr,
            "dominant_weights": weight_arr.astype(np.float32, copy=False),
        }
    component_scores = [sum(weight_by_row[row] for row in component) for component in components]
    best_index = int(np.argmax(component_scores))
    best_rows = np.asarray(components[best_index], dtype=np.int64)
    best_weights = np.asarray([weight_by_row[int(row)] for row in best_rows.tolist()], dtype=np.float64)
    best_weights = best_weights / max(float(np.sum(best_weights)), 1e-12)
    order = np.argsort(best_rows, kind="mergesort")
    return {
        "component_count": int(len(components)),
        "dominant_concentration": float(component_scores[best_index] / max(float(np.sum(weight_arr)), 1e-12)),
        "dominant_rows": best_rows[order],
        "dominant_weights": best_weights[order].astype(np.float32, copy=False),
    }


def _full_support_allowed(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
) -> tuple[bool, float, int, str]:
    decision, concentration, component_count, reason = _full_support_decision(elements, rows, weights, cfg)
    return decision != "reject", concentration, component_count, reason


def _full_support_decision(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
) -> tuple[str, float, int, str]:
    row_count = int(np.asarray(rows, dtype=np.int64).reshape(-1).size)
    needs_components = (
        float(cfg.min_full_component_concentration) > 0.0
        or int(cfg.max_full_component_count) > 0
        or (
            str(cfg.weak_observation_mode) == "keep"
            and float(cfg.weak_min_full_component_concentration) > 0.0
        )
    )
    if needs_components:
        summary = _component_summary(elements, rows, weights)
        concentration = float(summary["dominant_concentration"])
        component_count = int(summary["component_count"])
    else:
        concentration = 1.0
        component_count = 0

    reason = ""
    if int(cfg.max_full_support_elements) > 0 and row_count > int(cfg.max_full_support_elements):
        reason = "full_support_elements"
    elif concentration < float(cfg.min_full_component_concentration):
        reason = "full_component_concentration"
    elif int(cfg.max_full_component_count) > 0 and component_count > int(cfg.max_full_component_count):
        reason = "full_component_count"
    else:
        return "strong", concentration, component_count, ""

    if str(cfg.weak_observation_mode) != "keep":
        return "reject", concentration, component_count, reason
    weak_capacity_ok = (
        int(cfg.weak_max_full_support_elements) <= 0
        or row_count <= int(cfg.weak_max_full_support_elements)
    )
    weak_component_ok = concentration >= float(cfg.weak_min_full_component_concentration)
    if weak_capacity_ok and weak_component_ok:
        return "weak", concentration, component_count, reason
    return "reject", concentration, component_count, reason


def _footprint_sample_offsets(cfg: Vfm2DgsMappingConfig, radius: float) -> tuple[np.ndarray, np.ndarray]:
    grid = int(cfg.footprint_sample_grid)
    if grid <= 1:
        return np.zeros((1, 2), dtype=np.float64), np.ones((1,), dtype=np.float64)
    extent = float(cfg.footprint_sample_extent_px)
    if extent <= 0.0:
        extent = float(radius)
    coords = np.linspace(-extent, extent, grid, dtype=np.float64)
    offsets = np.asarray([(float(x), float(y)) for y in coords for x in coords], dtype=np.float64)
    sigma_sq = max((0.6 * max(float(radius), 1e-6)) ** 2, 1e-8)
    weights = np.exp(-0.5 * np.sum(np.square(offsets), axis=1) / sigma_sq)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    return offsets, weights


def _accumulate_token_surface_weights(
    tree: cKDTree,
    valid_rows: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    projected_disk_radius: np.ndarray,
    elements: SurfaceElementMap,
    view_weights: np.ndarray,
    token_xy: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = float(cfg.footprint_radius_px)
    sigma_sq = max((radius * 0.5) ** 2, 1e-6)
    sample_offsets, sample_weights = _footprint_sample_offsets(cfg, radius)
    contribution_by_row: dict[int, float] = {}
    for offset, sample_weight in zip(sample_offsets, sample_weights):
        sample_xy = token_xy + offset
        local = tree.query_ball_point(sample_xy, r=radius + float(cfg.max_projected_disk_radius_px))
        if not local:
            continue
        rows = valid_rows[np.asarray(local, dtype=np.int64)]
        surface_radius = projected_disk_radius[rows]
        offsets = uv[rows] - sample_xy[None, :]
        dist_sq = np.sum(np.square(offsets), axis=1)
        overlaps = np.sqrt(dist_sq) <= (radius + surface_radius)
        rows = rows[overlaps]
        surface_radius = surface_radius[overlaps]
        dist_sq = dist_sq[overlaps]
        if rows.size == 0:
            continue
        effective_sigma_sq = np.square(np.maximum(radius * 0.5, 1e-6) + 0.5 * surface_radius)
        raw = (
            float(sample_weight)
            * elements.opacity[rows].astype(np.float64)
            * np.exp(-0.5 * dist_sq / np.maximum(effective_sigma_sq, sigma_sq))
            * view_weights[rows].astype(np.float64)
        )
        for row, value in zip(rows.tolist(), raw.tolist()):
            row_int = int(row)
            contribution_by_row[row_int] = float(contribution_by_row.get(row_int, 0.0)) + float(value)
    if not contribution_by_row:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    rows = np.asarray(sorted(contribution_by_row), dtype=np.int64)
    raw_weights = np.asarray([contribution_by_row[int(row)] for row in rows.tolist()], dtype=np.float64)
    surface_radius = projected_disk_radius[rows]
    nearest_depth = float(np.min(depth[rows]))
    depth_keep = depth[rows] <= nearest_depth + float(cfg.depth_epsilon)
    return rows[depth_keep], surface_radius[depth_keep], raw_weights[depth_keep]


def _composite_sorted_packed_hits(
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized front-to-back alpha compositing for pixel-sorted hits.

    ``packed`` columns are pixel id, depth, element row and alpha.  The
    implementation preserves the legacy 1e-4 transmittance early-stop rule
    without a Python loop over potentially millions of raster hits.
    """

    values = np.asarray(packed, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("packed raster hits must have four columns")
    if values.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    pixels = values[:, 0].astype(np.int64)
    rows = values[:, 2].astype(np.int64)
    alpha = np.clip(values[:, 3], 0.0, 0.999)
    starts = np.r_[True, pixels[1:] != pixels[:-1]]
    start_indices = np.flatnonzero(starts)
    counts = np.diff(np.r_[start_indices, values.shape[0]])
    log_survival = np.log1p(-alpha)
    prefix = np.cumsum(log_survival)
    group_base = np.r_[0.0, prefix[start_indices[1:] - 1]]
    base = np.repeat(group_base, counts)
    log_transmittance = prefix - log_survival - base
    transmittance = np.exp(np.clip(log_transmittance, -745.0, 0.0))
    weights = transmittance * alpha
    transmittance_after = np.exp(np.clip(prefix - base, -745.0, 0.0))
    group_ids = np.cumsum(starts) - 1
    stop = transmittance_after <= 1e-4
    first_stop = np.full((start_indices.size,), values.shape[0], dtype=np.int64)
    indices = np.arange(values.shape[0], dtype=np.int64)
    np.minimum.at(first_stop, group_ids[stop], indices[stop])
    active = indices <= first_stop[group_ids]
    keep = active & (weights > 1e-12)
    return pixels[keep], rows[keep], weights[keep].astype(np.float32)


def _render_surface_element_pixel_contributions_2dgs(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    width: int,
    height: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from gsplat import rasterization_2dgs
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range_2dgs
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gsplat and torch are required for renderer contribution buffers") from exc
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat renderer contribution buffers")
    if len(elements) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    quats, scales = _surface_element_quaternions_and_scales(elements)
    means = torch.as_tensor(elements.centers, dtype=torch.float32, device=torch_device)
    quats_t = torch.as_tensor(quats, dtype=torch.float32, device=torch_device)
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=torch_device)
    opacities = torch.as_tensor(elements.opacity, dtype=torch.float32, device=torch_device).reshape(-1).clamp(0.0, 1.0)
    colors = torch.zeros((len(elements), 1), dtype=torch.float32, device=torch_device)
    pose = torch.as_tensor(np.asarray(view.pose_w2c, dtype=np.float32).reshape(4, 4), dtype=torch.float32, device=torch_device)[None]
    k_matrix = torch.as_tensor(
        _intrinsic_matrix(view.camera, int(width), int(height)),
        dtype=torch.float32,
        device=torch_device,
    )[None]
    _rendered, _alphas, _normals, _surf_normals, _distort, _median, meta = rasterization_2dgs(
        means=means,
        quats=quats_t,
        scales=scales_t,
        opacities=opacities,
        colors=colors,
        viewmats=pose,
        Ks=k_matrix,
        width=int(width),
        height=int(height),
        packed=False,
        render_mode="RGB",
    )
    transmittances = torch.ones((1, int(height), int(width)), dtype=torch.float32, device=torch_device)
    gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
        0,
        1_000_000_000,
        transmittances,
        meta["means2d"],
        meta["ray_transforms"],
        meta["opacities"],
        int(width),
        int(height),
        int(meta["tile_size"]),
        meta["isect_offsets"],
        meta["flatten_ids"],
    )
    if gs_ids.numel() == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.asarray(meta["depths"][0].detach().cpu().numpy(), dtype=np.float32),
        )
    pixel_ids_x = (pixel_ids % int(width)).to(torch.float32) + 0.5
    pixel_ids_y = (pixel_ids // int(width)).to(torch.float32) + 0.5
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1)
    means2d = meta["means2d"][camera_ids, gs_ids]
    deltas = pixel_coords - means2d
    transforms = meta["ray_transforms"][camera_ids, gs_ids]
    h_u = -transforms[..., 0, :3] + transforms[..., 2, :3] * pixel_ids_x[..., None]
    h_v = -transforms[..., 1, :3] + transforms[..., 2, :3] * pixel_ids_y[..., None]
    tmp = torch.cross(h_u, h_v, dim=-1)
    denom = _signed_safe_denominator(tmp[..., 2], eps=1e-12)
    us = tmp[..., 0] / denom
    vs = tmp[..., 1] / denom
    sigmas_3d = us * us + vs * vs
    sigmas_2d = 2.0 * (deltas[..., 0] * deltas[..., 0] + deltas[..., 1] * deltas[..., 1])
    sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)
    alphas = torch.clamp(meta["opacities"][camera_ids, gs_ids] * torch.exp(-sigmas), max=0.999)
    depths = meta["depths"][camera_ids, gs_ids]
    packed = torch.stack(
        [
            pixel_ids.to(torch.float64),
            depths.to(torch.float64),
            gs_ids.to(torch.float64),
            alphas.to(torch.float64),
        ],
        dim=1,
    ).detach().cpu().numpy()
    order = np.lexsort((packed[:, 2], packed[:, 1], packed[:, 0]))
    packed = packed[order]
    out_pixels, out_rows, out_weights = _composite_sorted_packed_hits(packed)
    return (
        out_pixels,
        out_rows,
        out_weights,
        np.asarray(meta["depths"][0].detach().cpu().numpy(), dtype=np.float32),
    )


def _empty_contribution_buffer(image_id: str, renderer: str, cfg: Vfm2DgsMappingConfig) -> Vfm2DgsContributionBuffer:
    return Vfm2DgsContributionBuffer(
        image_id=image_id,
        renderer=renderer,
        token_indices=np.zeros((0,), dtype=np.int64),
        token_xy=np.zeros((0, 2), dtype=np.float32),
        support_offsets=np.zeros((1,), dtype=np.int64),
        element_ids=np.zeros((0,), dtype=np.int64),
        element_weights=np.zeros((0,), dtype=np.float32),
        purity_scores=np.zeros((0,), dtype=np.float32),
        component_concentrations=np.zeros((0,), dtype=np.float32),
        quality_scores=np.zeros((0,), dtype=np.float32),
        top_alpha=np.zeros((0,), dtype=np.float32),
        alpha_entropy=np.zeros((0,), dtype=np.float32),
        metadata={
            "mapping_config": cfg.to_dict(),
            "rejection_stats": {
                "full_component_concentration": 0,
                "full_component_count": 0,
                "full_support_elements": 0,
                "weak_observations": 0,
            },
        },
    )


def compute_renderer_token_surface_contribution_buffer(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    config: Vfm2DgsMappingConfig | None = None,
    device: str = "cuda",
    renderer: str = "gsplat_2dgs",
) -> Vfm2DgsContributionBuffer:
    cfg = config or Vfm2DgsMappingConfig()
    if renderer != "gsplat_2dgs":
        return compute_token_surface_contribution_buffer(elements, view, cfg)
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    _channels, height, width = feature_map.shape
    if len(elements) == 0:
        return _empty_contribution_buffer(view.image_id, "gsplat_2dgs", cfg)
    pixel_ids, rows_per_hit, weights_per_hit, depth = _render_surface_element_pixel_contributions_2dgs(
        elements,
        view,
        width=int(width),
        height=int(height),
        device=str(device),
    )
    if pixel_ids.size == 0:
        return _empty_contribution_buffer(view.image_id, "gsplat_2dgs", cfg)
    element_totals = np.bincount(
        rows_per_hit.astype(np.int64),
        weights=weights_per_hit.astype(np.float64),
        minlength=len(elements),
    )
    pixel_x = (pixel_ids % int(width)).astype(np.float64)
    pixel_y = (pixel_ids // int(width)).astype(np.float64)
    if cfg.token_selection_mode == "surface":
        token_indices = surface_supported_token_indices_from_hits(
            feature_map,
            pixel_ids,
            weights_per_hit,
            cfg,
        )
    else:
        token_indices = _selected_token_indices(feature_map, cfg)
    radius = float(cfg.footprint_radius_px)
    sigma_sq = max((radius * 0.5) ** 2, 1e-6)
    output_token_indices: list[int] = []
    output_token_xy: list[np.ndarray] = []
    support_offsets = [0]
    output_element_ids: list[int] = []
    output_element_weights: list[float] = []
    purity_scores: list[float] = []
    concentrations: list[float] = []
    quality_scores: list[float] = []
    top_alpha: list[float] = []
    alpha_entropy: list[float] = []
    observation_strengths: list[str] = []
    descriptor_weights: list[float] = []
    rejection_stats = {
        "full_component_concentration": 0,
        "full_component_count": 0,
        "full_support_elements": 0,
        "weak_observations": 0,
    }
    for token_index in token_indices.tolist():
        y = int(token_index) // int(width)
        x = int(token_index) % int(width)
        token_xy = np.asarray([float(x), float(y)], dtype=np.float64)
        dist_sq = np.square(pixel_x - token_xy[0]) + np.square(pixel_y - token_xy[1])
        hit_mask = dist_sq <= radius * radius
        if not np.any(hit_mask):
            continue
        local_rows = rows_per_hit[hit_mask].astype(np.int64)
        local_weights = (
            weights_per_hit[hit_mask].astype(np.float64)
            * np.exp(-0.5 * dist_sq[hit_mask] / sigma_sq)
        )
        positive = local_weights > 1e-12
        if not np.any(positive):
            continue
        local_rows = local_rows[positive]
        local_weights = local_weights[positive]
        contribution_by_row: dict[int, float] = {}
        for row, value in zip(local_rows.tolist(), local_weights.tolist()):
            row_int = int(row)
            contribution_by_row[row_int] = float(contribution_by_row.get(row_int, 0.0)) + float(value)
        rows = np.asarray(sorted(contribution_by_row), dtype=np.int64)
        raw_weights = np.asarray([contribution_by_row[int(row)] for row in rows.tolist()], dtype=np.float64)
        observation_strength, full_concentration, _full_component_count, rejection_reason = _full_support_decision(
            elements,
            rows,
            raw_weights,
            cfg,
        )
        if observation_strength == "reject":
            rejection_stats[rejection_reason] = int(rejection_stats.get(rejection_reason, 0)) + 1
            continue
        if observation_strength == "weak":
            rejection_stats["weak_observations"] = int(rejection_stats.get("weak_observations", 0)) + 1
        order = np.argsort(-raw_weights, kind="mergesort")[: int(cfg.max_elements_per_token)]
        rows = rows[order]
        raw_weights = raw_weights[order]
        token_term = raw_weights / max(float(np.sum(raw_weights)), 1e-12)
        element_term = raw_weights / np.maximum(element_totals[rows], 1e-12)
        element_term = np.clip(element_term, 0.0, 1.0)
        lambda_value = float(cfg.bidirectional_lambda)
        responsibilities = np.power(token_term, lambda_value) * np.power(
            np.maximum(element_term, 1e-12),
            1.0 - lambda_value,
        )
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        keep = responsibilities >= float(cfg.min_responsibility)
        if not np.any(keep):
            continue
        rows = rows[keep]
        responsibilities = responsibilities[keep]
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        if cfg.support_mode == "surface_component":
            component_rows, component_weights, concentration = _dominant_component(elements, rows, responsibilities)
        else:
            component_rows = rows
            component_weights = responsibilities.astype(np.float32, copy=False)
            concentration = 1.0
        component_rows, component_weights, competition_concentration = _apply_token_anchor_competition(
            elements,
            component_rows,
            component_weights,
            cfg,
        )
        if competition_concentration is not None:
            concentration = min(float(concentration), float(competition_concentration))
        if float(cfg.min_full_component_concentration) > 0.0 or int(cfg.max_full_component_count) > 0:
            concentration = min(float(concentration), float(full_concentration))
        component_threshold = (
            float(cfg.min_component_concentration)
            if observation_strength == "strong"
            else float(cfg.weak_min_full_component_concentration)
        )
        if concentration < component_threshold:
            continue
        purity_components = _compute_purity_components(
            elements,
            component_rows,
            component_weights,
            depth,
            cfg,
            concentration,
        )
        purity = float(purity_components["total"])
        if purity < float(cfg.min_purity):
            continue
        weights = np.asarray(component_weights, dtype=np.float32).reshape(-1)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        output_token_indices.append(int(token_index))
        output_token_xy.append(token_xy.astype(np.float32))
        output_element_ids.extend(elements.element_ids[component_rows].astype(np.int64).tolist())
        output_element_weights.extend(weights.astype(np.float32).tolist())
        support_offsets.append(len(output_element_ids))
        purity_scores.append(float(purity))
        concentrations.append(float(concentration))
        quality = float(purity * min(1.0, float(np.sum(raw_weights)) / max(float(len(raw_weights)), 1.0)))
        if observation_strength == "weak":
            quality *= float(cfg.weak_quality_scale)
        quality_scores.append(float(quality))
        observation_strengths.append(str(observation_strength))
        descriptor_weights.append(
            float(quality if observation_strength == "strong" else quality * float(cfg.weak_descriptor_weight))
        )
        top_alpha.append(float(np.max(weights)) if weights.size else 0.0)
        if weights.size <= 1:
            alpha_entropy.append(0.0)
        else:
            entropy = -float(np.sum(weights.astype(np.float64) * np.log(np.maximum(weights.astype(np.float64), 1e-12))))
            alpha_entropy.append(float(entropy / max(float(np.log(weights.size)), 1e-12)))
    token_xy_arr = (
        np.stack(output_token_xy, axis=0).astype(np.float32, copy=False)
        if output_token_xy
        else np.zeros((0, 2), dtype=np.float32)
    )
    return Vfm2DgsContributionBuffer(
        image_id=view.image_id,
        renderer="gsplat_2dgs",
        token_indices=np.asarray(output_token_indices, dtype=np.int64),
        token_xy=token_xy_arr,
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        element_ids=np.asarray(output_element_ids, dtype=np.int64),
        element_weights=np.asarray(output_element_weights, dtype=np.float32),
        purity_scores=np.asarray(purity_scores, dtype=np.float32),
        component_concentrations=np.asarray(concentrations, dtype=np.float32),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        top_alpha=np.asarray(top_alpha, dtype=np.float32),
        alpha_entropy=np.asarray(alpha_entropy, dtype=np.float32),
        observation_strengths=tuple(observation_strengths),
        descriptor_weights=np.asarray(descriptor_weights, dtype=np.float32),
        metadata={
            "mapping_config": cfg.to_dict(),
            "renderer": "gsplat_2dgs",
            "rejection_stats": rejection_stats,
        },
    )


def compute_token_surface_contribution_buffer(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    config: Vfm2DgsMappingConfig | None = None,
) -> Vfm2DgsContributionBuffer:
    cfg = config or Vfm2DgsMappingConfig()
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    _channels, height, width = feature_map.shape
    if len(elements) == 0:
        return _empty_contribution_buffer(view.image_id, "projection_depth_soft", cfg)
    uv, depth = _project_xyz_to_grid(elements.centers, view.pose_w2c, view.camera, width, height)
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
    valid &= depth > 1e-8
    valid &= (uv[:, 0] >= -float(cfg.footprint_radius_px)) & (uv[:, 0] < float(width) + float(cfg.footprint_radius_px))
    valid &= (uv[:, 1] >= -float(cfg.footprint_radius_px)) & (uv[:, 1] < float(height) + float(cfg.footprint_radius_px))
    valid &= elements.opacity >= float(cfg.opacity_threshold)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        return _empty_contribution_buffer(view.image_id, "projection_depth_soft", cfg)
    tree = cKDTree(uv[valid_rows].astype(np.float64, copy=False))
    focal_grid = _camera_focal_grid(view.camera, width, height)
    projected_disk_radius = (
        np.maximum(elements.scale1, elements.scale2).astype(np.float64)
        * float(focal_grid)
        / np.maximum(np.abs(depth), 1e-8)
    )
    projected_disk_radius = np.clip(projected_disk_radius, 0.0, float(cfg.max_projected_disk_radius_px))
    token_indices = _selected_token_indices(feature_map, cfg)
    view_weights = _compute_view_weights(elements, view.pose_w2c, float(cfg.view_angle_power))
    radius = float(cfg.footprint_radius_px)
    output_token_indices: list[int] = []
    output_token_xy: list[np.ndarray] = []
    support_offsets = [0]
    output_element_ids: list[int] = []
    output_element_weights: list[float] = []
    purity_scores: list[float] = []
    concentrations: list[float] = []
    quality_scores: list[float] = []
    top_alpha: list[float] = []
    alpha_entropy: list[float] = []
    observation_strengths: list[str] = []
    descriptor_weights: list[float] = []
    rejection_stats = {
        "full_component_concentration": 0,
        "full_component_count": 0,
        "full_support_elements": 0,
        "weak_observations": 0,
    }
    for token_index in token_indices.tolist():
        y = int(token_index) // int(width)
        x = int(token_index) % int(width)
        token_xy = np.asarray([float(x), float(y)], dtype=np.float64)
        rows, surface_radius, raw_weights = _accumulate_token_surface_weights(
            tree=tree,
            valid_rows=valid_rows,
            uv=uv,
            depth=depth,
            projected_disk_radius=projected_disk_radius,
            elements=elements,
            view_weights=view_weights,
            token_xy=token_xy,
            cfg=cfg,
        )
        positive = raw_weights > 1e-12
        if not np.any(positive):
            continue
        rows = rows[positive]
        surface_radius = surface_radius[positive]
        raw_weights = raw_weights[positive]
        observation_strength, full_concentration, _full_component_count, rejection_reason = _full_support_decision(
            elements,
            rows,
            raw_weights,
            cfg,
        )
        if observation_strength == "reject":
            rejection_stats[rejection_reason] = int(rejection_stats.get(rejection_reason, 0)) + 1
            continue
        if observation_strength == "weak":
            rejection_stats["weak_observations"] = int(rejection_stats.get("weak_observations", 0)) + 1
        order = np.argsort(-raw_weights, kind="mergesort")[: int(cfg.max_elements_per_token)]
        rows = rows[order]
        surface_radius = surface_radius[order]
        raw_weights = raw_weights[order]
        token_term = raw_weights / max(float(np.sum(raw_weights)), 1e-12)
        covered_fraction = np.minimum(
            1.0,
            np.square(float(radius) / np.maximum(surface_radius, max(float(radius), 1e-6))),
        )
        if float(cfg.element_coverage_power) > 0.0:
            covered_fraction = np.power(covered_fraction, float(cfg.element_coverage_power))
        lambda_value = float(cfg.bidirectional_lambda)
        responsibilities = np.power(token_term, lambda_value) * np.power(
            np.maximum(covered_fraction, 1e-12),
            1.0 - lambda_value,
        )
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        keep = responsibilities >= float(cfg.min_responsibility)
        if not np.any(keep):
            continue
        rows = rows[keep]
        responsibilities = responsibilities[keep]
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        if cfg.support_mode == "surface_component":
            component_rows, component_weights, concentration = _dominant_component(elements, rows, responsibilities)
        else:
            component_rows = rows
            component_weights = responsibilities.astype(np.float32, copy=False)
            concentration = 1.0
        component_rows, component_weights, competition_concentration = _apply_token_anchor_competition(
            elements,
            component_rows,
            component_weights,
            cfg,
        )
        if competition_concentration is not None:
            concentration = min(float(concentration), float(competition_concentration))
        if float(cfg.min_full_component_concentration) > 0.0 or int(cfg.max_full_component_count) > 0:
            concentration = min(float(concentration), float(full_concentration))
        component_threshold = (
            float(cfg.min_component_concentration)
            if observation_strength == "strong"
            else float(cfg.weak_min_full_component_concentration)
        )
        if concentration < component_threshold:
            continue
        purity_components = _compute_purity_components(
            elements,
            component_rows,
            component_weights,
            depth,
            cfg,
            concentration,
        )
        purity = float(purity_components["total"])
        if purity < float(cfg.min_purity):
            continue
        weights = np.asarray(component_weights, dtype=np.float32).reshape(-1)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        output_token_indices.append(int(token_index))
        output_token_xy.append(token_xy.astype(np.float32))
        output_element_ids.extend(elements.element_ids[component_rows].astype(np.int64).tolist())
        output_element_weights.extend(weights.astype(np.float32).tolist())
        support_offsets.append(len(output_element_ids))
        purity_scores.append(float(purity))
        concentrations.append(float(concentration))
        quality = float(purity * min(1.0, float(np.sum(raw_weights)) / max(float(len(raw_weights)), 1.0)))
        if observation_strength == "weak":
            quality *= float(cfg.weak_quality_scale)
        quality_scores.append(float(quality))
        observation_strengths.append(str(observation_strength))
        descriptor_weights.append(
            float(quality if observation_strength == "strong" else quality * float(cfg.weak_descriptor_weight))
        )
        top_alpha.append(float(np.max(weights)) if weights.size else 0.0)
        if weights.size <= 1:
            alpha_entropy.append(0.0)
        else:
            entropy = -float(np.sum(weights.astype(np.float64) * np.log(np.maximum(weights.astype(np.float64), 1e-12))))
            alpha_entropy.append(float(entropy / max(float(np.log(weights.size)), 1e-12)))
    token_xy_arr = (
        np.stack(output_token_xy, axis=0).astype(np.float32, copy=False)
        if output_token_xy
        else np.zeros((0, 2), dtype=np.float32)
    )
    return Vfm2DgsContributionBuffer(
        image_id=view.image_id,
        renderer="projection_depth_soft",
        token_indices=np.asarray(output_token_indices, dtype=np.int64),
        token_xy=token_xy_arr,
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        element_ids=np.asarray(output_element_ids, dtype=np.int64),
        element_weights=np.asarray(output_element_weights, dtype=np.float32),
        purity_scores=np.asarray(purity_scores, dtype=np.float32),
        component_concentrations=np.asarray(concentrations, dtype=np.float32),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        top_alpha=np.asarray(top_alpha, dtype=np.float32),
        alpha_entropy=np.asarray(alpha_entropy, dtype=np.float32),
        observation_strengths=tuple(observation_strengths),
        descriptor_weights=np.asarray(descriptor_weights, dtype=np.float32),
        metadata={
            "mapping_config": cfg.to_dict(),
            "rejection_stats": rejection_stats,
        },
    )


def token_surface_observations_from_contribution_buffer(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    contribution_buffer: Vfm2DgsContributionBuffer,
    config: Vfm2DgsMappingConfig | None = None,
) -> list[TokenSurfaceObservation]:
    cfg = config or Vfm2DgsMappingConfig()
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    channels, height, width = feature_map.shape
    row_by_id = elements.row_by_element_id
    observations = []
    for row in range(len(contribution_buffer)):
        token_index = int(contribution_buffer.token_indices[row])
        y = token_index // int(width)
        x = token_index % int(width)
        if y < 0 or y >= int(height) or x < 0 or x >= int(width):
            continue
        start = int(contribution_buffer.support_offsets[row])
        end = int(contribution_buffer.support_offsets[row + 1])
        element_ids = contribution_buffer.element_ids[start:end]
        element_weights = contribution_buffer.element_weights[start:end]
        if element_ids.size == 0:
            continue
        rows = np.asarray([row_by_id[int(element_id)] for element_id in element_ids.tolist()], dtype=np.int64)
        weights = np.asarray(element_weights, dtype=np.float32).reshape(-1)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        feature = feature_map[:, y, x].astype(np.float32, copy=False)
        if cfg.l2_normalize_features:
            feature, _valid_norm = normalize_rows(feature.reshape(1, channels))
            feature = feature.reshape(-1)
        center, normal, covariance = _support_geometry(elements, rows, weights)
        view_direction = _view_direction_to_camera(view.pose_w2c, center)
        purity = float(contribution_buffer.purity_scores[row])
        concentration = float(contribution_buffer.component_concentrations[row])
        observations.append(
            TokenSurfaceObservation(
                image_id=view.image_id,
                token_index=token_index,
                token_xy=contribution_buffer.token_xy[row].astype(np.float32, copy=False),
                feature=feature,
                element_ids=element_ids,
                element_weights=weights,
                center=center,
                normal=normal,
                covariance=covariance,
                purity_score=purity,
                component_concentration=concentration,
                quality_score=float(contribution_buffer.quality_scores[row]),
                view_direction=view_direction,
                purity_components={"total": purity},
                observation_strength=contribution_buffer.observation_strengths[row],
                descriptor_weight=float(contribution_buffer.descriptor_weights[row]),
            )
        )
    return observations


def compute_token_surface_observations(
    elements: SurfaceElementMap,
    view: GaussianVFMFeatureView,
    config: Vfm2DgsMappingConfig | None = None,
) -> list[TokenSurfaceObservation]:
    cfg = config or Vfm2DgsMappingConfig()
    if len(elements) == 0:
        return []
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    channels, height, width = feature_map.shape
    uv, depth = _project_xyz_to_grid(elements.centers, view.pose_w2c, view.camera, width, height)
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
    valid &= depth > 1e-8
    valid &= (uv[:, 0] >= -float(cfg.footprint_radius_px)) & (uv[:, 0] < float(width) + float(cfg.footprint_radius_px))
    valid &= (uv[:, 1] >= -float(cfg.footprint_radius_px)) & (uv[:, 1] < float(height) + float(cfg.footprint_radius_px))
    valid &= elements.opacity >= float(cfg.opacity_threshold)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        return []
    tree = cKDTree(uv[valid_rows].astype(np.float64, copy=False))
    focal_grid = _camera_focal_grid(view.camera, width, height)
    projected_disk_radius = (
        np.maximum(elements.scale1, elements.scale2).astype(np.float64)
        * float(focal_grid)
        / np.maximum(np.abs(depth), 1e-8)
    )
    projected_disk_radius = np.clip(projected_disk_radius, 0.0, float(cfg.max_projected_disk_radius_px))
    token_indices = _selected_token_indices(feature_map, cfg)
    view_weights = _compute_view_weights(elements, view.pose_w2c, float(cfg.view_angle_power))
    observations = []
    radius = float(cfg.footprint_radius_px)
    sigma_sq = max((radius * 0.5) ** 2, 1e-6)
    for token_index in token_indices.tolist():
        y = int(token_index) // int(width)
        x = int(token_index) % int(width)
        token_xy = np.asarray([float(x), float(y)], dtype=np.float64)
        rows, surface_radius, raw_weights = _accumulate_token_surface_weights(
            tree=tree,
            valid_rows=valid_rows,
            uv=uv,
            depth=depth,
            projected_disk_radius=projected_disk_radius,
            elements=elements,
            view_weights=view_weights,
            token_xy=token_xy,
            cfg=cfg,
        )
        positive = raw_weights > 1e-12
        if not np.any(positive):
            continue
        rows = rows[positive]
        surface_radius = surface_radius[positive]
        raw_weights = raw_weights[positive]
        observation_strength, full_concentration, _full_component_count, _rejection_reason = _full_support_decision(
            elements,
            rows,
            raw_weights,
            cfg,
        )
        if observation_strength == "reject":
            continue
        order = np.argsort(-raw_weights, kind="mergesort")[: int(cfg.max_elements_per_token)]
        rows = rows[order]
        surface_radius = surface_radius[order]
        raw_weights = raw_weights[order]
        token_term = raw_weights / max(float(np.sum(raw_weights)), 1e-12)
        covered_fraction = np.minimum(
            1.0,
            np.square(float(radius) / np.maximum(surface_radius, max(float(radius), 1e-6))),
        )
        if float(cfg.element_coverage_power) > 0.0:
            covered_fraction = np.power(covered_fraction, float(cfg.element_coverage_power))
        lambda_value = float(cfg.bidirectional_lambda)
        responsibilities = np.power(token_term, lambda_value) * np.power(
            np.maximum(covered_fraction, 1e-12),
            1.0 - lambda_value,
        )
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        keep = responsibilities >= float(cfg.min_responsibility)
        if not np.any(keep):
            continue
        rows = rows[keep]
        responsibilities = responsibilities[keep]
        responsibilities = responsibilities / max(float(np.sum(responsibilities)), 1e-12)
        if cfg.support_mode == "surface_component":
            component_rows, component_weights, concentration = _dominant_component(elements, rows, responsibilities)
        else:
            component_rows = rows
            component_weights = responsibilities.astype(np.float32, copy=False)
            concentration = 1.0
        component_rows, component_weights, competition_concentration = _apply_token_anchor_competition(
            elements,
            component_rows,
            component_weights,
            cfg,
        )
        if competition_concentration is not None:
            concentration = min(float(concentration), float(competition_concentration))
        if float(cfg.min_full_component_concentration) > 0.0 or int(cfg.max_full_component_count) > 0:
            concentration = min(float(concentration), float(full_concentration))
        component_threshold = (
            float(cfg.min_component_concentration)
            if observation_strength == "strong"
            else float(cfg.weak_min_full_component_concentration)
        )
        if concentration < component_threshold:
            continue
        purity_components = _compute_purity_components(
            elements,
            component_rows,
            component_weights,
            depth,
            cfg,
            concentration,
        )
        purity = float(purity_components["total"])
        if purity < float(cfg.min_purity):
            continue
        feature = feature_map[:, y, x].astype(np.float32, copy=False)
        if cfg.l2_normalize_features:
            feature, _valid_norm = normalize_rows(feature.reshape(1, channels))
            feature = feature.reshape(-1)
        center, normal, covariance = _support_geometry(elements, component_rows, component_weights)
        view_direction = _view_direction_to_camera(view.pose_w2c, center)
        element_ids = elements.element_ids[component_rows]
        quality = float(purity * min(1.0, float(np.sum(raw_weights)) / max(float(len(raw_weights)), 1.0)))
        if observation_strength == "weak":
            quality *= float(cfg.weak_quality_scale)
        observations.append(
            TokenSurfaceObservation(
                image_id=view.image_id,
                token_index=int(token_index),
                token_xy=token_xy.astype(np.float32),
                feature=feature,
                element_ids=element_ids,
                element_weights=component_weights.astype(np.float32, copy=False),
                center=center,
                normal=normal,
                covariance=covariance,
                purity_score=purity,
                component_concentration=float(concentration),
                quality_score=quality,
                view_direction=view_direction,
                purity_components=purity_components,
                observation_strength=str(observation_strength),
                descriptor_weight=(
                    float(quality)
                    if observation_strength == "strong"
                    else float(quality) * float(cfg.weak_descriptor_weight)
                ),
            )
        )
    return observations


def _dominant_component(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    summary = _component_summary(elements, rows, weights)
    return (
        np.asarray(summary["dominant_rows"], dtype=np.int64),
        np.asarray(summary["dominant_weights"], dtype=np.float32),
        float(summary["dominant_concentration"]),
    )


def _apply_token_anchor_competition(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
    cfg: Vfm2DgsMappingConfig,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    mode = str(cfg.token_anchor_competition)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    if rows.size == 0 or mode == "none":
        return rows, weights.astype(np.float32, copy=False), None
    if mode == "winner_element":
        winner = int(np.argmax(weights))
        return rows[[winner]], np.ones((1,), dtype=np.float32), None
    if mode == "winner_neighborhood":
        winner_row = int(rows[int(np.argmax(weights))])
        allowed = {winner_row}
        frontier = {winner_row}
        for _hop in range(int(cfg.token_anchor_neighborhood_hops)):
            next_frontier: set[int] = set()
            for row in frontier:
                for neighbor in elements.adjacency[int(row)].tolist():
                    neighbor = int(neighbor)
                    if neighbor not in allowed:
                        allowed.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        keep = np.asarray([int(row) in allowed for row in rows.tolist()], dtype=bool)
        if not np.any(keep):
            winner = int(np.argmax(weights))
            return rows[[winner]], np.ones((1,), dtype=np.float32), None
        local_rows = rows[keep]
        local_weights = weights[keep]
        max_support = int(cfg.token_anchor_max_support_elements)
        if max_support > 0 and local_rows.size > max_support:
            winner_local = np.flatnonzero(local_rows == winner_row)
            winner_idx = int(winner_local[0]) if winner_local.size else int(np.argmax(local_weights))
            order = np.argsort(-local_weights, kind="mergesort")
            selected: list[int] = [winner_idx]
            for idx in order.tolist():
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= max_support:
                    break
            selected_arr = np.asarray(selected, dtype=np.int64)
            local_rows = local_rows[selected_arr]
            local_weights = local_weights[selected_arr]
        local_weights = local_weights / max(float(np.sum(local_weights)), 1e-12)
        order = np.argsort(local_rows, kind="mergesort")
        return local_rows[order], local_weights[order].astype(np.float32, copy=False), None
    if mode == "winner_component":
        component_rows, component_weights, concentration = _dominant_component(elements, rows, weights)
        return component_rows, component_weights.astype(np.float32, copy=False), float(concentration)
    raise ValueError(f"unsupported token_anchor_competition: {mode}")


def _support_geometry(
    elements: SurfaceElementMap,
    rows: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    centers = elements.centers[rows]
    center = np.sum(centers * weights[:, None], axis=0)
    normal = np.sum(elements.normals[rows].astype(np.float64) * weights[:, None], axis=0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    deltas = centers - center[None, :]
    covariance = np.einsum("n,ni,nj->ij", weights, deltas, deltas)
    scale_var = np.mean(np.square(elements.scale1[rows]) + np.square(elements.scale2[rows])) * 0.5
    covariance += np.eye(3, dtype=np.float64) * float(scale_var) * 0.01
    return center.astype(np.float64), normal.astype(np.float32), covariance.astype(np.float32)


def _weighted_iou(
    first_ids: np.ndarray,
    first_weights: np.ndarray,
    second_ids: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    first_ids_arr = np.asarray(first_ids, dtype=np.int64).reshape(-1)
    second_ids_arr = np.asarray(second_ids, dtype=np.int64).reshape(-1)
    first_weights_arr = np.asarray(first_weights, dtype=np.float32).reshape(-1)
    second_weights_arr = np.asarray(second_weights, dtype=np.float32).reshape(-1)
    if first_ids_arr.shape != first_weights_arr.shape or second_ids_arr.shape != second_weights_arr.shape:
        raise ValueError("weighted-IoU IDs and weights must have matching shapes")
    first_is_unique_sorted = first_ids_arr.size < 2 or bool(np.all(np.diff(first_ids_arr) > 0))
    second_is_unique_sorted = second_ids_arr.size < 2 or bool(np.all(np.diff(second_ids_arr) > 0))
    if first_is_unique_sorted and second_is_unique_sorted:
        # Observation, dilated, and parent supports are emitted as sorted
        # unique IDs.  The histogram-intersection identity below is exactly
        # equivalent to the dictionary implementation, but performs the
        # production hot path in NumPy instead of allocating two Python
        # dictionaries and their union for every candidate edge.
        _shared_ids, first_rows, second_rows = np.intersect1d(
            first_ids_arr,
            second_ids_arr,
            assume_unique=True,
            return_indices=True,
        )
        intersection = float(
            np.minimum(
                first_weights_arr[first_rows],
                second_weights_arr[second_rows],
            ).sum(dtype=np.float64)
        )
        union = (
            float(first_weights_arr.sum(dtype=np.float64))
            + float(second_weights_arr.sum(dtype=np.float64))
            - intersection
        )
        return float(intersection / max(union, 1e-12))
    first = {
        int(idx): float(weight)
        for idx, weight in zip(first_ids_arr.tolist(), first_weights_arr.tolist())
    }
    second = {
        int(idx): float(weight)
        for idx, weight in zip(second_ids_arr.tolist(), second_weights_arr.tolist())
    }
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    numerator = sum(min(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    denominator = sum(max(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    return float(numerator / max(denominator, 1e-12))


def _aggregate_weighted_ids(ids: Sequence[int], weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, float] = {}
    for idx, weight in zip(ids, weights):
        key = int(idx)
        merged[key] = float(merged.get(key, 0.0)) + float(weight)
    if not merged:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    output_ids = np.asarray(sorted(merged), dtype=np.int64)
    output_weights = np.asarray([merged[int(idx)] for idx in output_ids.tolist()], dtype=np.float32)
    output_weights = output_weights / max(float(np.sum(output_weights)), 1e-12)
    return output_ids, output_weights


def _dilated_support(
    elements: SurfaceElementMap,
    element_ids: np.ndarray,
    element_weights: np.ndarray,
    hops: int,
    row_by_id: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if int(hops) <= 0:
        return np.asarray(element_ids, dtype=np.int64), np.asarray(element_weights, dtype=np.float32)
    # Building this 2DGS-wide lookup is O(number of surface elements).  Callers
    # that process a bank of observations must construct it once and reuse it;
    # rebuilding it for every token makes full-scene graph fusion effectively
    # O(number of observations * number of surface elements).
    element_row_by_id = elements.row_by_element_id if row_by_id is None else row_by_id
    output_ids: list[int] = []
    output_weights: list[float] = []
    for element_id, weight in zip(element_ids.tolist(), element_weights.tolist()):
        if int(element_id) not in element_row_by_id:
            continue
        frontier = {int(element_row_by_id[int(element_id)])}
        visited = set(frontier)
        for _hop in range(int(hops)):
            next_frontier = set()
            for row in frontier:
                for neighbor in elements.adjacency[int(row)].tolist():
                    neighbor = int(neighbor)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        for row in visited:
            output_ids.append(int(elements.element_ids[int(row)]))
            output_weights.append(float(weight))
    return _aggregate_weighted_ids(output_ids, output_weights)


def _parent_support(
    elements: SurfaceElementMap,
    element_ids: np.ndarray,
    element_weights: np.ndarray,
    row_by_id: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    element_row_by_id = elements.row_by_element_id if row_by_id is None else row_by_id
    parent_ids: list[int] = []
    parent_weights: list[float] = []
    for element_id, weight in zip(element_ids.tolist(), element_weights.tolist()):
        if int(element_id) not in element_row_by_id:
            continue
        row = element_row_by_id[int(element_id)]
        parent_ids.append(int(elements.parent_gaussian_indices[int(row)]))
        parent_weights.append(float(weight))
    return _aggregate_weighted_ids(parent_ids, parent_weights)


def _support_association_scores(
    elements: SurfaceElementMap,
    first_ids: np.ndarray,
    first_weights: np.ndarray,
    second_ids: np.ndarray,
    second_weights: np.ndarray,
    cfg: Vfm2DgsAnchorFusionConfig,
    dilated_cache: dict[tuple[tuple[int, ...], tuple[float, ...], int], tuple[np.ndarray, np.ndarray]] | None = None,
    parent_cache: dict[tuple[tuple[int, ...], tuple[float, ...]], tuple[np.ndarray, np.ndarray]] | None = None,
    row_by_id: Mapping[int, int] | None = None,
) -> dict[str, float]:
    exact = _weighted_iou(first_ids, first_weights, second_ids, second_weights)
    dilated = 0.0
    if int(cfg.support_iou_dilation_hops) > 0 and float(cfg.min_dilated_surface_iou) > 0.0:
        first_d_ids, first_d_weights = _cached_dilated_support(
            elements,
            first_ids,
            first_weights,
            int(cfg.support_iou_dilation_hops),
            dilated_cache,
            row_by_id=row_by_id,
        )
        second_d_ids, second_d_weights = _cached_dilated_support(
            elements,
            second_ids,
            second_weights,
            int(cfg.support_iou_dilation_hops),
            dilated_cache,
            row_by_id=row_by_id,
        )
        dilated = _weighted_iou(first_d_ids, first_d_weights, second_d_ids, second_d_weights)
    parent = 0.0
    if float(cfg.min_parent_surface_iou) > 0.0:
        first_p_ids, first_p_weights = _cached_parent_support(
            elements,
            first_ids,
            first_weights,
            parent_cache,
            row_by_id=row_by_id,
        )
        second_p_ids, second_p_weights = _cached_parent_support(
            elements,
            second_ids,
            second_weights,
            parent_cache,
            row_by_id=row_by_id,
        )
        parent = _weighted_iou(first_p_ids, first_p_weights, second_p_ids, second_p_weights)
    return {"exact": exact, "dilated": dilated, "parent": parent}


def _support_cache_key(element_ids: np.ndarray, element_weights: np.ndarray) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ids = tuple(int(item) for item in np.asarray(element_ids, dtype=np.int64).reshape(-1).tolist())
    weights = tuple(float(item) for item in np.round(np.asarray(element_weights, dtype=np.float32).reshape(-1), 6).tolist())
    return ids, weights


def _cached_dilated_support(
    elements: SurfaceElementMap,
    element_ids: np.ndarray,
    element_weights: np.ndarray,
    hops: int,
    cache: dict[tuple[tuple[int, ...], tuple[float, ...], int], tuple[np.ndarray, np.ndarray]] | None,
    row_by_id: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    key_base = _support_cache_key(element_ids, element_weights)
    key = (key_base[0], key_base[1], int(hops))
    if cache is not None and key in cache:
        return cache[key]
    value = _dilated_support(
        elements,
        element_ids,
        element_weights,
        int(hops),
        row_by_id=row_by_id,
    )
    if cache is not None:
        cache[key] = value
    return value


def _cached_parent_support(
    elements: SurfaceElementMap,
    element_ids: np.ndarray,
    element_weights: np.ndarray,
    cache: dict[tuple[tuple[int, ...], tuple[float, ...]], tuple[np.ndarray, np.ndarray]] | None,
    row_by_id: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    key = _support_cache_key(element_ids, element_weights)
    if cache is not None and key in cache:
        return cache[key]
    value = _parent_support(elements, element_ids, element_weights, row_by_id=row_by_id)
    if cache is not None:
        cache[key] = value
    return value


def fuse_token_surface_observations(
    elements: SurfaceElementMap,
    observations: Sequence[TokenSurfaceObservation],
    config: Vfm2DgsAnchorFusionConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    cfg = config or Vfm2DgsAnchorFusionConfig()
    if cfg.fusion_mode == "surface_first":
        return _fuse_token_surface_observations_surface_first(elements, observations, cfg, metadata)
    if cfg.fusion_mode == "graph":
        return _fuse_token_surface_observations_graph(elements, observations, cfg, metadata)
    states: list[dict[str, object]] = []
    dilated_cache: dict[tuple[tuple[int, ...], tuple[float, ...], int], tuple[np.ndarray, np.ndarray]] = {}
    parent_cache: dict[tuple[tuple[int, ...], tuple[float, ...]], tuple[np.ndarray, np.ndarray]] = {}
    row_by_id = elements.row_by_element_id
    for obs in observations:
        best_index = -1
        best_score = 0.0
        for idx, state in enumerate(states):
            state_observations = state.get("observations", [])
            state_features = (
                np.stack([item.feature for item in state_observations], axis=0)
                if isinstance(state_observations, list) and state_observations
                else None
            )
            if not _source_merge_allowed(
                obs.source_id,
                state.get("source_ids", set()),
                cfg,
                first_feature=obs.feature,
                second_features=state_features,
            ):
                continue
            scores = _support_association_scores(
                elements,
                obs.element_ids,
                obs.element_weights,
                np.asarray(state["element_ids"], dtype=np.int64),
                np.asarray(state["element_weights"], dtype=np.float32),
                cfg,
                dilated_cache=dilated_cache,
                parent_cache=parent_cache,
                row_by_id=row_by_id,
            )
            normal_cos = float(np.dot(obs.normal, np.asarray(state["normal"], dtype=np.float32)))
            center_dist = float(np.linalg.norm(obs.center - np.asarray(state["center"], dtype=np.float64)))
            support_match = (
                scores["exact"] >= float(cfg.min_surface_iou)
                or (
                    float(cfg.min_dilated_surface_iou) > 0.0
                    and scores["dilated"] >= float(cfg.min_dilated_surface_iou)
                )
                or (
                    float(cfg.min_parent_surface_iou) > 0.0
                    and scores["parent"] >= float(cfg.min_parent_surface_iou)
                )
            )
            score = max(scores.values())
            if support_match and score > best_score and normal_cos >= float(cfg.min_normal_cosine) and center_dist <= float(cfg.max_center_distance):
                best_score = score
                best_index = idx
        if best_index < 0:
            states.append(_new_anchor_state(obs))
        else:
            _merge_anchor_state(states[best_index], obs)
    rows = [
        _finalize_anchor_state(elements, state, cfg, row_by_id=row_by_id)
        for state in states
        if _anchor_state_observation_count_ok(state, cfg)
    ]
    if not rows:
        return _empty_anchor_map(0, metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})})
    anchor_ids = np.arange(len(rows), dtype=np.int64)
    centers = np.stack([row["center"] for row in rows], axis=0)
    normals = np.stack([row["normal"] for row in rows], axis=0)
    covariances = np.stack([row["covariance"] for row in rows], axis=0)
    features = np.stack([row["feature"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features)
    feature_prototypes = np.stack([row["feature_prototypes"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        flat = feature_prototypes.reshape(-1, feature_prototypes.shape[-1])
        flat, _valid = normalize_rows(flat)
        feature_prototypes = flat.reshape(feature_prototypes.shape)
    feature_prototype_counts = np.asarray([row["feature_prototype_count"] for row in rows], dtype=np.int64)
    view_bin_features = np.stack([row["view_bin_features"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        flat_bins = view_bin_features.reshape(-1, view_bin_features.shape[-1])
        flat_bins, _valid = normalize_rows(flat_bins)
        view_bin_features = flat_bins.reshape(view_bin_features.shape)
    view_bin_counts = np.stack([row["view_bin_counts"] for row in rows], axis=0).astype(np.int64)
    feature_variances = np.asarray([row["feature_variance"] for row in rows], dtype=np.float32)
    stability_scores = np.exp(-feature_variances / 0.1).astype(np.float32, copy=False)
    distinctiveness_scores = _compute_feature_distinctiveness(features)
    base_quality_scores = np.asarray([row["quality"] for row in rows], dtype=np.float32)
    quality_scores = (
        base_quality_scores
        * np.clip(stability_scores, 0.0, 1.0)
        * (0.5 + 0.5 * np.clip(distinctiveness_scores, 0.0, 1.0))
    ).astype(np.float32, copy=False)
    support_offsets = [0]
    support_element_ids = []
    support_parent_gaussian_indices = []
    support_weights = []
    observed_view_ids = []
    row_by_id = elements.row_by_element_id
    for row in rows:
        support_element_ids.extend(row["element_ids"].tolist())
        support_parent_gaussian_indices.extend(
            int(elements.parent_gaussian_indices[row_by_id[int(element_id)]])
            for element_id in row["element_ids"].tolist()
            if int(element_id) in row_by_id
        )
        support_weights.extend(row["element_weights"].tolist())
        support_offsets.append(len(support_element_ids))
        observed_view_ids.append(tuple(row["view_ids"]))
    return Vfm2DgsAnchorMap(
        anchor_ids=anchor_ids,
        centers=centers,
        normals=normals,
        covariances=covariances,
        features=features,
        feature_prototypes=feature_prototypes,
        feature_prototype_counts=feature_prototype_counts,
        view_bin_features=view_bin_features,
        view_bin_counts=view_bin_counts,
        feature_variances=feature_variances,
        quality_scores=quality_scores,
        purity_scores=np.asarray([row["purity"] for row in rows], dtype=np.float32),
        observation_counts=np.asarray([row["observation_count"] for row in rows], dtype=np.int64),
        surface_support_counts=np.asarray([len(row["element_ids"]) for row in rows], dtype=np.int64),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(support_element_ids, dtype=np.int64),
        support_parent_gaussian_indices=np.asarray(support_parent_gaussian_indices, dtype=np.int64),
        support_weights=np.asarray(support_weights, dtype=np.float32),
        observed_view_ids=tuple(observed_view_ids),
        distinctiveness_scores=distinctiveness_scores,
        stability_scores=stability_scores,
        metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})},
    )


def _surface_first_seed_element_ids(
    obs: TokenSurfaceObservation,
    cfg: Vfm2DgsAnchorFusionConfig,
) -> np.ndarray:
    element_ids = np.asarray(obs.element_ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(obs.element_weights, dtype=np.float32).reshape(-1)
    if element_ids.size == 0 or weights.size == 0:
        return np.zeros((0,), dtype=np.int64)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    keep = weights >= float(cfg.surface_first_min_seed_weight)
    if not np.any(keep):
        keep[int(np.argmax(weights))] = True
    candidate_ids = element_ids[keep]
    candidate_weights = weights[keep]
    order = np.argsort(-candidate_weights, kind="mergesort")
    if int(cfg.surface_first_max_seeds_per_observation) > 0:
        order = order[: int(cfg.surface_first_max_seeds_per_observation)]
    return np.asarray(candidate_ids[order], dtype=np.int64)


def _fuse_token_surface_observations_surface_first(
    elements: SurfaceElementMap,
    observations: Sequence[TokenSurfaceObservation],
    cfg: Vfm2DgsAnchorFusionConfig,
    metadata: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    """Fuse observations by surface-element seed before descriptor aggregation.

    Each token observation votes for a small number of surface-element seeds,
    usually only its dominant weighted support element. This keeps the anchor
    proposal tied to the reconstructed 2DGS surface instead of letting broad
    token supports merge unrelated surface regions first.
    """
    if not observations:
        return _empty_anchor_map(0, metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})})
    buckets: dict[int, list[TokenSurfaceObservation]] = {}
    for obs in observations:
        for seed_id in _surface_first_seed_element_ids(obs, cfg).tolist():
            buckets.setdefault(int(seed_id), []).append(obs)
    states: list[dict[str, object]] = []
    for seed_id in sorted(buckets):
        bucket = buckets[int(seed_id)]
        if not bucket:
            continue
        state = _anchor_state_from_observations(bucket)
        state["surface_first_seed_element_id"] = int(seed_id)
        states.append(state)
    row_by_id = elements.row_by_element_id
    rows = [
        _finalize_anchor_state(
            elements,
            state,
            cfg,
            row_by_id=row_by_id,
        )
        for state in states
        if _anchor_state_observation_count_ok(state, cfg)
    ]
    if not rows:
        feature_dim = int(observations[0].feature.shape[0]) if observations else 0
        return _empty_anchor_map(feature_dim, metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})})
    return _anchor_map_from_finalized_rows(
        elements,
        rows,
        cfg,
        {
            "surface_first": {
                "seed_bucket_count": int(len(buckets)),
                "max_seeds_per_observation": int(cfg.surface_first_max_seeds_per_observation),
                "min_seed_weight": float(cfg.surface_first_min_seed_weight),
            },
            **dict(metadata or {}),
        },
    )


def _support_match(
    scores: Mapping[str, float],
    cfg: Vfm2DgsAnchorFusionConfig,
) -> bool:
    return (
        float(scores.get("exact", 0.0)) >= float(cfg.min_surface_iou)
        or (
            float(cfg.min_dilated_surface_iou) > 0.0
            and float(scores.get("dilated", 0.0)) >= float(cfg.min_dilated_surface_iou)
        )
        or (
            float(cfg.min_parent_surface_iou) > 0.0
            and float(scores.get("parent", 0.0)) >= float(cfg.min_parent_surface_iou)
        )
    )


def _source_merge_allowed(
    first_source: str,
    second_sources: str | Sequence[str],
    cfg: Vfm2DgsAnchorFusionConfig,
    first_feature: np.ndarray | None = None,
    second_features: np.ndarray | Sequence[np.ndarray] | None = None,
) -> bool:
    if str(cfg.source_merge_policy) == "allow":
        return True
    first = str(first_source)
    if isinstance(second_sources, str):
        seconds = {str(second_sources)}
    else:
        seconds = {str(item) for item in second_sources}
    if not first or not seconds or "" in seconds:
        return True
    if first in seconds:
        return True
    if str(cfg.source_merge_policy) == "same_source":
        return False
    if str(cfg.source_merge_policy) != "feature_agree":
        return False
    if first_feature is None or second_features is None:
        return False
    first_vec = np.asarray(first_feature, dtype=np.float32).reshape(-1)
    first_norm = float(np.linalg.norm(first_vec))
    if first_norm <= 1e-8:
        return False
    first_vec = first_vec / first_norm
    if isinstance(second_features, np.ndarray) and second_features.ndim == 1:
        candidates = second_features.reshape(1, -1)
    else:
        candidates = np.asarray(second_features, dtype=np.float32)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
    if candidates.ndim != 2 or candidates.shape[1] != first_vec.shape[0] or candidates.shape[0] == 0:
        return False
    norms = np.linalg.norm(candidates, axis=1)
    valid = norms > 1e-8
    if not np.any(valid):
        return False
    normalized = candidates[valid] / norms[valid, None]
    best_cosine = float(np.max(normalized @ first_vec.reshape(-1, 1)))
    return best_cosine >= float(cfg.cross_source_min_feature_cosine)


def _fuse_token_surface_observations_graph(
    elements: SurfaceElementMap,
    observations: Sequence[TokenSurfaceObservation],
    cfg: Vfm2DgsAnchorFusionConfig,
    metadata: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    count = len(observations)
    if count == 0:
        return _empty_anchor_map(0, metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})})
    parent = np.arange(count, dtype=np.int64)

    def find(row: int) -> int:
        current = int(row)
        while int(parent[current]) != current:
            parent[current] = parent[int(parent[current])]
            current = int(parent[current])
        return current

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            # Canonicalize the component representative so graph output is
            # independent of candidate traversal order.
            low = min(root_first, root_second)
            high = max(root_first, root_second)
            parent[high] = low

    exact_enabled = float(cfg.min_surface_iou) > 0.0
    dilated_enabled = (
        int(cfg.support_iou_dilation_hops) > 0
        and float(cfg.min_dilated_surface_iou) > 0.0
    )
    parent_enabled = float(cfg.min_parent_surface_iou) > 0.0
    # A zero exact-IoU threshold means every spatially compatible pair is a
    # candidate under the historical contract.  Otherwise, a pair can only
    # pass if at least one enabled support representation has a shared ID.
    unrestricted_support = not exact_enabled
    support_postings: dict[tuple[str, int], list[int]] = {}
    spatial_postings: dict[tuple[int, int, int], list[int]] = {}
    dilated_supports: list[tuple[np.ndarray, np.ndarray] | None] = []
    parent_supports: list[tuple[np.ndarray, np.ndarray] | None] = []
    row_by_id = elements.row_by_element_id
    cell_size = max(float(cfg.max_center_distance), 1e-8)
    candidate_pair_count = 0
    evaluated_pair_count = 0
    merged_pair_count = 0
    connected_pair_skip_count = 0
    for second_idx, second in enumerate(observations):
        second_dilated = (
            _dilated_support(
                elements,
                second.element_ids,
                second.element_weights,
                int(cfg.support_iou_dilation_hops),
                row_by_id=row_by_id,
            )
            if dilated_enabled
            else None
        )
        second_parent = (
            _parent_support(
                elements,
                second.element_ids,
                second.element_weights,
                row_by_id=row_by_id,
            )
            if parent_enabled
            else None
        )
        support_keys: list[tuple[str, int]] = []
        if exact_enabled:
            support_keys.extend(
                ("exact", int(element_id))
                for element_id in np.asarray(second.element_ids, dtype=np.int64).tolist()
            )
        if second_dilated is not None:
            support_keys.extend(
                ("dilated", int(element_id))
                for element_id in second_dilated[0].tolist()
            )
        if second_parent is not None:
            support_keys.extend(
                ("parent", int(parent_id))
                for parent_id in second_parent[0].tolist()
            )

        center_cell_array = np.floor(
            np.asarray(second.center, dtype=np.float64) / cell_size
        ).astype(np.int64)
        center_cell = tuple(int(value) for value in center_cell_array.tolist())
        spatial_candidates: set[int] = set()
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    spatial_candidates.update(
                        spatial_postings.get(
                            (
                                center_cell[0] + dx,
                                center_cell[1] + dy,
                                center_cell[2] + dz,
                            ),
                            (),
                        )
                    )
        if unrestricted_support:
            candidates = spatial_candidates
        else:
            candidates: set[int] = set()
            for key in support_keys:
                candidates.update(support_postings.get(key, ()))
            candidates.intersection_update(spatial_candidates)
        candidate_pair_count += len(candidates)
        # Connected components do not depend on edge traversal order.  Integer
        # set iteration is deterministic here, while sorting every dense
        # per-observation candidate set is a substantial full-scene cost.
        for first_idx in candidates:
            if find(int(first_idx)) == find(int(second_idx)):
                connected_pair_skip_count += 1
                continue
            first = observations[int(first_idx)]
            if not _source_merge_allowed(
                first.source_id,
                second.source_id,
                cfg,
                first_feature=first.feature,
                second_features=second.feature,
            ):
                continue
            normal_cos = float(np.dot(first.normal, second.normal))
            if normal_cos < float(cfg.min_normal_cosine):
                continue
            center_dist = float(np.linalg.norm(first.center - second.center))
            if center_dist > float(cfg.max_center_distance):
                continue
            evaluated_pair_count += 1
            scores = {
                "exact": _weighted_iou(
                    first.element_ids,
                    first.element_weights,
                    second.element_ids,
                    second.element_weights,
                ),
                "dilated": 0.0,
                "parent": 0.0,
            }
            if second_dilated is not None:
                first_dilated = dilated_supports[int(first_idx)]
                if first_dilated is None:
                    raise RuntimeError("dilated support index is inconsistent")
                scores["dilated"] = _weighted_iou(
                    first_dilated[0],
                    first_dilated[1],
                    second_dilated[0],
                    second_dilated[1],
                )
            if second_parent is not None:
                first_parent = parent_supports[int(first_idx)]
                if first_parent is None:
                    raise RuntimeError("parent support index is inconsistent")
                scores["parent"] = _weighted_iou(
                    first_parent[0],
                    first_parent[1],
                    second_parent[0],
                    second_parent[1],
                )
            if _support_match(scores, cfg):
                union(first_idx, second_idx)
                merged_pair_count += 1
        dilated_supports.append(second_dilated)
        parent_supports.append(second_parent)
        for key in set(support_keys):
            support_postings.setdefault(key, []).append(int(second_idx))
        spatial_postings.setdefault(center_cell, []).append(int(second_idx))
        if count >= 10_000 and (second_idx + 1) % 10_000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "vfm_2dgs_sparse_graph_fusion",
                        "completed_observations": int(second_idx + 1),
                        "total_observations": int(count),
                        "candidate_pair_count": int(candidate_pair_count),
                        "evaluated_pair_count": int(evaluated_pair_count),
                        "merged_pair_count": int(merged_pair_count),
                        "connected_pair_skip_count": int(
                            connected_pair_skip_count
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    clusters: dict[int, list[TokenSurfaceObservation]] = {}
    for idx, obs in enumerate(observations):
        clusters.setdefault(find(idx), []).append(obs)
    states: list[dict[str, object]] = []
    for cluster in clusters.values():
        states.append(_anchor_state_from_observations(cluster))
    rows = [
        _finalize_anchor_state(elements, state, cfg, row_by_id=row_by_id)
        for state in states
        if _anchor_state_observation_count_ok(state, cfg)
    ]
    if not rows:
        feature_dim = int(observations[0].feature.shape[0]) if observations else 0
        return _empty_anchor_map(feature_dim, metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})})
    return _anchor_map_from_finalized_rows(
        elements,
        rows,
        cfg,
        {
            "sparse_graph_fusion": {
                "observation_count": int(count),
                "all_pair_count": int(count * (count - 1) // 2),
                "candidate_pair_count": int(candidate_pair_count),
                "evaluated_pair_count": int(evaluated_pair_count),
                "merged_pair_count": int(merged_pair_count),
                "connected_pair_skip_count": int(connected_pair_skip_count),
                "spatial_cell_size": float(cell_size),
                "support_inverted_index": True,
            },
            **dict(metadata or {}),
        },
    )


def _anchor_map_from_finalized_rows(
    elements: SurfaceElementMap,
    rows: Sequence[Mapping[str, object]],
    cfg: Vfm2DgsAnchorFusionConfig,
    metadata: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    anchor_ids = np.arange(len(rows), dtype=np.int64)
    centers = np.stack([row["center"] for row in rows], axis=0)
    normals = np.stack([row["normal"] for row in rows], axis=0)
    covariances = np.stack([row["covariance"] for row in rows], axis=0)
    features = np.stack([row["feature"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features)
    feature_prototypes = np.stack([row["feature_prototypes"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        flat = feature_prototypes.reshape(-1, feature_prototypes.shape[-1])
        flat, _valid = normalize_rows(flat)
        feature_prototypes = flat.reshape(feature_prototypes.shape)
    feature_prototype_counts = np.asarray([row["feature_prototype_count"] for row in rows], dtype=np.int64)
    view_bin_features = np.stack([row["view_bin_features"] for row in rows], axis=0).astype(np.float32)
    if cfg.l2_normalize_features:
        flat_bins = view_bin_features.reshape(-1, view_bin_features.shape[-1])
        flat_bins, _valid = normalize_rows(flat_bins)
        view_bin_features = flat_bins.reshape(view_bin_features.shape)
    view_bin_counts = np.stack([row["view_bin_counts"] for row in rows], axis=0).astype(np.int64)
    feature_variances = np.asarray([row["feature_variance"] for row in rows], dtype=np.float32)
    stability_scores = np.exp(-feature_variances / 0.1).astype(np.float32, copy=False)
    distinctiveness_scores = _compute_feature_distinctiveness(features)
    base_quality_scores = np.asarray([row["quality"] for row in rows], dtype=np.float32)
    quality_scores = (
        base_quality_scores
        * np.clip(stability_scores, 0.0, 1.0)
        * (0.5 + 0.5 * np.clip(distinctiveness_scores, 0.0, 1.0))
    ).astype(np.float32, copy=False)
    support_offsets = [0]
    support_element_ids = []
    support_parent_gaussian_indices = []
    support_weights = []
    observed_view_ids = []
    row_by_id = elements.row_by_element_id
    for row in rows:
        support_element_ids.extend(row["element_ids"].tolist())
        support_parent_gaussian_indices.extend(
            int(elements.parent_gaussian_indices[row_by_id[int(element_id)]])
            for element_id in row["element_ids"].tolist()
            if int(element_id) in row_by_id
        )
        support_weights.extend(row["element_weights"].tolist())
        support_offsets.append(len(support_element_ids))
        observed_view_ids.append(tuple(row["view_ids"]))
    return Vfm2DgsAnchorMap(
        anchor_ids=anchor_ids,
        centers=centers,
        normals=normals,
        covariances=covariances,
        features=features,
        feature_prototypes=feature_prototypes,
        feature_prototype_counts=feature_prototype_counts,
        view_bin_features=view_bin_features,
        view_bin_counts=view_bin_counts,
        feature_variances=feature_variances,
        quality_scores=quality_scores,
        purity_scores=np.asarray([row["purity"] for row in rows], dtype=np.float32),
        observation_counts=np.asarray([row["observation_count"] for row in rows], dtype=np.int64),
        surface_support_counts=np.asarray([len(row["element_ids"]) for row in rows], dtype=np.int64),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(support_element_ids, dtype=np.int64),
        support_parent_gaussian_indices=np.asarray(support_parent_gaussian_indices, dtype=np.int64),
        support_weights=np.asarray(support_weights, dtype=np.float32),
        observed_view_ids=tuple(observed_view_ids),
        distinctiveness_scores=distinctiveness_scores,
        stability_scores=stability_scores,
        metadata={"fusion_config": cfg.to_dict(), **dict(metadata or {})},
    )


def _compute_feature_distinctiveness(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if feats.shape[0] == 1:
        return np.ones((1,), dtype=np.float32)
    normalized, _valid = normalize_rows(feats)
    # Keep the historical exact all-anchor nearest-neighbour definition, but
    # do not materialize the complete N x N similarity matrix.  Full-scene
    # maps can contain tens of thousands of anchors, for which the old
    # allocation could consume several gigabytes after an otherwise
    # successful fusion.  Blocking only changes peak memory, not the result.
    row_count = int(normalized.shape[0])
    target_similarity_values = 16 * 1024 * 1024  # 64 MiB at float32.
    block_size = max(1, min(row_count, target_similarity_values // row_count))
    nearest = np.full((row_count,), -np.inf, dtype=np.float32)
    for start in range(0, row_count, block_size):
        stop = min(start + block_size, row_count)
        similarity = normalized[start:stop] @ normalized.T
        local_rows = np.arange(stop - start, dtype=np.int64)
        similarity[local_rows, np.arange(start, stop, dtype=np.int64)] = -np.inf
        nearest[start:stop] = np.max(similarity, axis=1)
    return np.clip(1.0 - nearest, 0.0, 1.0).astype(np.float32, copy=False)


def spatial_nms_anchor_map(
    anchor_map: Vfm2DgsAnchorMap,
    radius: float,
    max_anchors: int = 0,
) -> Vfm2DgsAnchorMap:
    """Keep high-quality anchors while suppressing nearby lower-quality ones."""
    if len(anchor_map) == 0:
        return anchor_map
    if float(radius) <= 0.0 and int(max_anchors) <= 0:
        return anchor_map
    quality = np.asarray(anchor_map.quality_scores, dtype=np.float64)
    order = np.lexsort((anchor_map.anchor_ids.astype(np.int64), -quality))
    selected_rows: list[int] = []
    radius_value = float(radius)
    for row in order.tolist():
        row = int(row)
        if radius_value > 0.0 and selected_rows:
            distances = np.linalg.norm(anchor_map.centers[np.asarray(selected_rows, dtype=np.int64)] - anchor_map.centers[row], axis=1)
            if bool(np.any(distances <= radius_value)):
                continue
        selected_rows.append(row)
        if int(max_anchors) > 0 and len(selected_rows) >= int(max_anchors):
            break
    return _subset_anchor_map(
        anchor_map,
        np.asarray(selected_rows, dtype=np.int64),
        metadata_updates={
            "spatial_nms": {
                "enabled": True,
                "radius": float(radius),
                "max_anchors": int(max_anchors),
                "input_anchor_count": int(len(anchor_map)),
                "output_anchor_count": int(len(selected_rows)),
            }
        },
    )


def build_anchor_covisibility_graph(
    anchor_map: Vfm2DgsAnchorMap,
    min_score: float = 0.1,
    max_neighbors: int = 16,
) -> Vfm2DgsAnchorMap:
    """Attach a symmetric view-overlap graph to a VFM-2DGS anchor map."""
    count = len(anchor_map)
    if count == 0:
        return anchor_map
    view_sets = [set(row) for row in anchor_map.observed_view_ids]
    offsets = [0]
    neighbor_ids: list[int] = []
    scores: list[float] = []
    for row in range(count):
        row_neighbors: list[tuple[float, int]] = []
        first = view_sets[row]
        for col in range(count):
            if row == col:
                continue
            second = view_sets[col]
            union = first | second
            if not union:
                continue
            shared = first & second
            score = float(len(shared) / max(len(union), 1))
            if score >= float(min_score):
                row_neighbors.append((score, int(anchor_map.anchor_ids[col])))
        row_neighbors.sort(key=lambda item: (-item[0], item[1]))
        if int(max_neighbors) > 0:
            row_neighbors = row_neighbors[: int(max_neighbors)]
        neighbor_ids.extend(anchor_id for _score, anchor_id in row_neighbors)
        scores.extend(score for score, _anchor_id in row_neighbors)
        offsets.append(len(neighbor_ids))
    return _copy_anchor_map(
        anchor_map,
        covisibility_offsets=np.asarray(offsets, dtype=np.int64),
        covisibility_anchor_ids=np.asarray(neighbor_ids, dtype=np.int64),
        covisibility_scores=np.asarray(scores, dtype=np.float32),
        metadata_updates={
            "covisibility_graph": {
                "enabled": True,
                "min_score": float(min_score),
                "max_neighbors": int(max_neighbors),
                "edge_count": int(len(neighbor_ids)),
            }
        },
    )


def build_anchor_descriptor_index(
    anchor_map: Vfm2DgsAnchorMap,
    include_prototypes: bool = True,
) -> Vfm2DgsDescriptorIndex:
    anchor_ids: list[int] = []
    prototype_ids: list[int] = []
    centers: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    quality_scores: list[float] = []
    if bool(include_prototypes):
        for row, anchor_id in enumerate(anchor_map.anchor_ids.tolist()):
            count = int(anchor_map.feature_prototype_counts[row])
            count = min(max(count, 1), int(anchor_map.feature_prototypes.shape[1]))
            for prototype_id in range(count):
                anchor_ids.append(int(anchor_id))
                prototype_ids.append(int(prototype_id))
                centers.append(anchor_map.centers[row])
                descriptors.append(anchor_map.feature_prototypes[row, prototype_id])
                quality_scores.append(float(anchor_map.quality_scores[row]))
    else:
        for row, anchor_id in enumerate(anchor_map.anchor_ids.tolist()):
            anchor_ids.append(int(anchor_id))
            prototype_ids.append(0)
            centers.append(anchor_map.centers[row])
            descriptors.append(anchor_map.features[row])
            quality_scores.append(float(anchor_map.quality_scores[row]))
    feature_dim = int(anchor_map.feature_dim)
    return Vfm2DgsDescriptorIndex(
        anchor_ids=np.asarray(anchor_ids, dtype=np.int64),
        prototype_ids=np.asarray(prototype_ids, dtype=np.int64),
        centers=(
            np.stack(centers, axis=0).astype(np.float64, copy=False)
            if centers
            else np.zeros((0, 3), dtype=np.float64)
        ),
        descriptors=(
            np.stack(descriptors, axis=0).astype(np.float32, copy=False)
            if descriptors
            else np.zeros((0, feature_dim), dtype=np.float32)
        ),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        metadata={
            "stage": "vfm_2dgs_descriptor_index",
            "include_prototypes": bool(include_prototypes),
            "source_anchor_count": int(len(anchor_map)),
            "descriptor_count": int(len(anchor_ids)),
            "feature_dim": int(feature_dim),
        },
    )


def _subset_anchor_map(
    anchor_map: Vfm2DgsAnchorMap,
    rows: np.ndarray,
    metadata_updates: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    support_offsets = [0]
    support_element_ids: list[int] = []
    support_parent_gaussian_indices: list[int] = []
    support_weights: list[float] = []
    for row in rows.tolist():
        start = int(anchor_map.support_offsets[int(row)])
        end = int(anchor_map.support_offsets[int(row) + 1])
        support_element_ids.extend(anchor_map.support_element_ids[start:end].astype(np.int64).tolist())
        support_parent_gaussian_indices.extend(
            anchor_map.support_parent_gaussian_indices[start:end].astype(np.int64).tolist()
        )
        support_weights.extend(anchor_map.support_weights[start:end].astype(np.float32).tolist())
        support_offsets.append(len(support_element_ids))
    metadata = dict(anchor_map.metadata or {})
    metadata.update(dict(metadata_updates or {}))
    return Vfm2DgsAnchorMap(
        anchor_ids=anchor_map.anchor_ids[rows],
        centers=anchor_map.centers[rows],
        normals=anchor_map.normals[rows],
        covariances=anchor_map.covariances[rows],
        features=anchor_map.features[rows],
        feature_prototypes=anchor_map.feature_prototypes[rows],
        feature_prototype_counts=anchor_map.feature_prototype_counts[rows],
        view_bin_features=anchor_map.view_bin_features[rows],
        view_bin_counts=anchor_map.view_bin_counts[rows],
        feature_variances=anchor_map.feature_variances[rows],
        quality_scores=anchor_map.quality_scores[rows],
        purity_scores=anchor_map.purity_scores[rows],
        observation_counts=anchor_map.observation_counts[rows],
        surface_support_counts=anchor_map.surface_support_counts[rows],
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        support_element_ids=np.asarray(support_element_ids, dtype=np.int64),
        support_parent_gaussian_indices=np.asarray(support_parent_gaussian_indices, dtype=np.int64),
        support_weights=np.asarray(support_weights, dtype=np.float32),
        observed_view_ids=tuple(anchor_map.observed_view_ids[int(row)] for row in rows.tolist()),
        distinctiveness_scores=anchor_map.distinctiveness_scores[rows],
        stability_scores=anchor_map.stability_scores[rows],
        metadata=metadata,
    )


def _copy_anchor_map(
    anchor_map: Vfm2DgsAnchorMap,
    covisibility_offsets: np.ndarray | None = None,
    covisibility_anchor_ids: np.ndarray | None = None,
    covisibility_scores: np.ndarray | None = None,
    metadata_updates: Mapping[str, object] | None = None,
) -> Vfm2DgsAnchorMap:
    metadata = dict(anchor_map.metadata or {})
    metadata.update(dict(metadata_updates or {}))
    return Vfm2DgsAnchorMap(
        anchor_ids=anchor_map.anchor_ids,
        centers=anchor_map.centers,
        normals=anchor_map.normals,
        covariances=anchor_map.covariances,
        features=anchor_map.features,
        feature_prototypes=anchor_map.feature_prototypes,
        feature_prototype_counts=anchor_map.feature_prototype_counts,
        view_bin_features=anchor_map.view_bin_features,
        view_bin_counts=anchor_map.view_bin_counts,
        feature_variances=anchor_map.feature_variances,
        quality_scores=anchor_map.quality_scores,
        purity_scores=anchor_map.purity_scores,
        observation_counts=anchor_map.observation_counts,
        surface_support_counts=anchor_map.surface_support_counts,
        support_offsets=anchor_map.support_offsets,
        support_element_ids=anchor_map.support_element_ids,
        support_parent_gaussian_indices=anchor_map.support_parent_gaussian_indices,
        support_weights=anchor_map.support_weights,
        observed_view_ids=anchor_map.observed_view_ids,
        covisibility_offsets=covisibility_offsets if covisibility_offsets is not None else anchor_map.covisibility_offsets,
        covisibility_anchor_ids=(
            covisibility_anchor_ids if covisibility_anchor_ids is not None else anchor_map.covisibility_anchor_ids
        ),
        covisibility_scores=covisibility_scores if covisibility_scores is not None else anchor_map.covisibility_scores,
        distinctiveness_scores=anchor_map.distinctiveness_scores,
        stability_scores=anchor_map.stability_scores,
        metadata=metadata,
    )


def _cluster_feature_prototypes(
    features: np.ndarray,
    weights: np.ndarray,
    max_prototypes: int,
    min_cosine: float,
) -> tuple[np.ndarray, int]:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    proto_count = max(int(max_prototypes), 1)
    if feats.shape[0] == 0:
        return np.zeros((proto_count, feats.shape[1] if feats.ndim == 2 else 0), dtype=np.float32), 0
    normalized, _valid = normalize_rows(feats)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
    order = np.argsort(-weight_arr, kind="mergesort")
    proto_vectors: list[np.ndarray] = []
    weighted_sums: list[np.ndarray] = []
    for idx in order.tolist():
        vector = normalized[int(idx)]
        best = -1
        best_sim = -np.inf
        for cluster_idx, proto in enumerate(proto_vectors):
            sim = float(np.dot(vector, proto))
            if sim > best_sim:
                best_sim = sim
                best = cluster_idx
        if best >= 0 and best_sim >= float(min_cosine):
            assigned = best
        elif len(proto_vectors) < proto_count:
            assigned = len(proto_vectors)
            proto_vectors.append(np.zeros_like(vector, dtype=np.float32))
            weighted_sums.append(
                np.zeros_like(vector, dtype=np.float64)
            )
        else:
            if best >= 0:
                assigned = best
            else:
                assigned = 0
        weighted_sums[assigned] += (
            normalized[int(idx)].astype(np.float64)
            * float(weight_arr[int(idx)])
        )
        proto = weighted_sums[assigned]
        proto = proto / max(float(np.linalg.norm(proto)), 1e-8)
        proto_vectors[assigned] = proto.astype(np.float32)
    prototypes = np.zeros((proto_count, feats.shape[1]), dtype=np.float32)
    for idx, proto in enumerate(proto_vectors[:proto_count]):
        prototypes[idx] = proto
    if proto_vectors:
        for idx in range(len(proto_vectors), proto_count):
            prototypes[idx] = proto_vectors[0]
    return prototypes, int(len(proto_vectors))


def _view_bin_features(
    features: np.ndarray,
    weights: np.ndarray,
    view_directions: np.ndarray,
    view_bin_count: int,
    feature_mode: str = "mean",
    consensus_weight_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(features, dtype=np.float32)
    bin_count = max(int(view_bin_count), 1)
    mode = str(feature_mode)
    if mode not in {"mean", "medoid", "consensus_weighted_mean"}:
        raise ValueError("feature_mode must be 'mean', 'medoid', or 'consensus_weighted_mean'")
    output = np.zeros((bin_count, feats.shape[1]), dtype=np.float32)
    counts = np.zeros((bin_count,), dtype=np.int64)
    if feats.shape[0] == 0:
        return output, counts
    normalized, _valid = normalize_rows(feats)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    dirs = np.asarray(view_directions, dtype=np.float32).reshape(-1, 3)
    angles = np.arctan2(dirs[:, 1].astype(np.float64), dirs[:, 0].astype(np.float64))
    angles = np.mod(angles, 2.0 * np.pi)
    bins = np.floor(angles / max(2.0 * np.pi / float(bin_count), 1e-12)).astype(np.int64)
    bins = np.clip(bins, 0, bin_count - 1)
    for bin_id in range(bin_count):
        rows = np.flatnonzero(bins == bin_id)
        counts[bin_id] = int(rows.size)
        if rows.size == 0:
            continue
        local_weights = weight_arr[rows]
        local_weights = local_weights / max(float(np.sum(local_weights)), 1e-12)
        if mode == "medoid":
            local_feats = normalized[rows]
            center = np.sum(local_feats * local_weights[:, None], axis=0)
            center = center / max(float(np.linalg.norm(center)), 1e-8)
            best = int(np.argmax(local_feats @ center.reshape(-1, 1)))
            vector = local_feats[best]
        elif mode == "consensus_weighted_mean":
            local_feats = normalized[rows]
            consensus = _feature_consensus_scores(local_feats)
            consensus_weights = np.power(np.clip(consensus, 0.0, 1.0), float(consensus_weight_power))
            local_weights = local_weights * consensus_weights
            local_weights = local_weights / max(float(np.sum(local_weights)), 1e-12)
            vector = np.sum(local_feats * local_weights[:, None], axis=0)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                vector = vector / norm
        else:
            vector = np.sum(normalized[rows] * local_weights[:, None], axis=0)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-8:
                vector = vector / norm
        output[bin_id] = vector.astype(np.float32, copy=False)
    fallback = np.sum(normalized * (weight_arr / max(float(np.sum(weight_arr)), 1e-12))[:, None], axis=0)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-8:
        fallback = fallback / fallback_norm
    for bin_id in range(bin_count):
        if counts[bin_id] == 0:
            output[bin_id] = fallback.astype(np.float32, copy=False)
    return output, counts


def _feature_consensus_scores(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    count = int(feats.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    normalized, _valid = normalize_rows(feats)
    similarities = normalized @ normalized.T
    scores = np.mean(np.clip(similarities, -1.0, 1.0), axis=1)
    return np.asarray(scores, dtype=np.float64)


def _consensus_filtered_feature_rows(features: np.ndarray, min_consensus_cosine: float) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    count = int(feats.shape[0])
    if count <= 1 or float(min_consensus_cosine) <= -1.0:
        return np.arange(count, dtype=np.int64)
    scores = _feature_consensus_scores(feats)
    keep = scores >= float(min_consensus_cosine)
    if not np.any(keep):
        return np.arange(count, dtype=np.int64)
    return np.flatnonzero(keep)


def _robust_feature_rows(features: np.ndarray, weights: np.ndarray, trim_fraction: float) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("features must have shape (N, C)")
    count = int(feats.shape[0])
    if count <= 1 or float(trim_fraction) <= 0.0:
        return np.arange(count, dtype=np.int64)
    trim_count = int(np.floor(float(count) * float(trim_fraction)))
    trim_count = min(max(trim_count, 0), count - 1)
    if trim_count <= 0:
        return np.arange(count, dtype=np.int64)
    normalized, _valid = normalize_rows(feats)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
    center = np.sum(normalized * weight_arr[:, None], axis=0)
    center = center / max(float(np.linalg.norm(center)), 1e-8)
    similarities = normalized @ center.reshape(-1, 1)
    order = np.argsort(similarities.reshape(-1), kind="mergesort")
    keep = np.ones((count,), dtype=bool)
    keep[order[:trim_count]] = False
    return np.flatnonzero(keep)


def _new_anchor_state(obs: TokenSurfaceObservation) -> dict[str, object]:
    return {
        "support": {int(idx): float(weight) * float(obs.quality_score) for idx, weight in zip(obs.element_ids.tolist(), obs.element_weights.tolist())},
        "observations": [obs],
        "source_ids": {str(obs.source_id)} if str(obs.source_id) else set(),
        "center": obs.center.copy(),
        "normal": obs.normal.copy(),
        "element_ids": obs.element_ids.copy(),
        "element_weights": obs.element_weights.copy(),
    }


def _anchor_state_from_observations(
    observations: Sequence[TokenSurfaceObservation],
) -> dict[str, object]:
    """Build the exact final aggregate state in one linear pass.

    Repeated ``_merge_anchor_state`` calls are appropriate for online greedy
    fusion, where the current center and support affect the next association.
    Graph and surface-first fusion already know their complete membership, so
    recomputing all preceding centers and normals at every append is an
    unnecessary O(k^2) operation for a component of size k.
    """
    if not observations:
        raise ValueError("anchor state requires at least one observation")
    support: dict[int, float] = {}
    source_ids: set[str] = set()
    quality_weights = np.empty((len(observations),), dtype=np.float64)
    centers = np.empty((len(observations), 3), dtype=np.float64)
    normals = np.empty((len(observations), 3), dtype=np.float64)
    for row, obs in enumerate(observations):
        for element_id, weight in zip(
            obs.element_ids.tolist(),
            obs.element_weights.tolist(),
        ):
            key = int(element_id)
            support[key] = (
                float(support.get(key, 0.0))
                + float(weight) * float(obs.quality_score)
            )
        if str(obs.source_id):
            source_ids.add(str(obs.source_id))
        quality_weights[row] = max(float(obs.quality_score), 1e-6)
        centers[row] = obs.center
        normals[row] = obs.normal
    support_total = max(sum(float(value) for value in support.values()), 1e-12)
    element_ids = np.asarray(sorted(support), dtype=np.int64)
    element_weights = np.asarray(
        [float(support[int(idx)]) / support_total for idx in element_ids.tolist()],
        dtype=np.float32,
    )
    quality_weights = quality_weights / max(
        float(np.sum(quality_weights)),
        1e-12,
    )
    center = np.sum(centers * quality_weights[:, None], axis=0)
    normal = np.sum(normals * quality_weights[:, None], axis=0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    return {
        "support": support,
        "observations": list(observations),
        "source_ids": source_ids,
        "center": center.astype(np.float64, copy=False),
        "normal": normal.astype(np.float32, copy=False),
        "element_ids": element_ids,
        "element_weights": element_weights,
    }


def _anchor_state_observation_count_ok(state: Mapping[str, object], cfg: Vfm2DgsAnchorFusionConfig) -> bool:
    observations = state.get("observations", [])
    if not isinstance(observations, list):
        return False
    if len(observations) < int(cfg.min_observations):
        return False
    if int(cfg.min_descriptor_observations) <= 0:
        return True
    descriptor_count = sum(1 for obs in observations if float(getattr(obs, "descriptor_weight", 0.0)) > 1e-12)
    return descriptor_count >= int(cfg.min_descriptor_observations)


def _merge_anchor_state(state: dict[str, object], obs: TokenSurfaceObservation) -> None:
    support = state["support"]
    assert isinstance(support, dict)
    for element_id, weight in zip(obs.element_ids.tolist(), obs.element_weights.tolist()):
        key = int(element_id)
        support[key] = float(support.get(key, 0.0)) + float(weight) * float(obs.quality_score)
    observations = state["observations"]
    assert isinstance(observations, list)
    observations.append(obs)
    source_ids = state.setdefault("source_ids", set())
    assert isinstance(source_ids, set)
    if str(obs.source_id):
        source_ids.add(str(obs.source_id))
    total = max(sum(float(value) for value in support.values()), 1e-12)
    element_ids = np.asarray(sorted(support), dtype=np.int64)
    element_weights = np.asarray([float(support[int(idx)]) / total for idx in element_ids.tolist()], dtype=np.float32)
    state["element_ids"] = element_ids
    state["element_weights"] = element_weights
    quality_weights = np.asarray([max(float(item.quality_score), 1e-6) for item in observations], dtype=np.float64)
    quality_weights = quality_weights / max(float(np.sum(quality_weights)), 1e-12)
    centers = np.stack([item.center for item in observations], axis=0)
    normals = np.stack([item.normal for item in observations], axis=0)
    state["center"] = np.sum(centers * quality_weights[:, None], axis=0)
    normal = np.sum(normals * quality_weights[:, None], axis=0)
    state["normal"] = (normal / max(float(np.linalg.norm(normal)), 1e-8)).astype(np.float32)


def _finalize_anchor_state(
    elements: SurfaceElementMap,
    state: dict[str, object],
    cfg: Vfm2DgsAnchorFusionConfig,
    row_by_id: Mapping[int, int] | None = None,
) -> dict[str, object]:
    observations = state["observations"]
    assert isinstance(observations, list)
    descriptor_observations = [
        obs for obs in observations if float(getattr(obs, "descriptor_weight", 0.0)) > 1e-12
    ]
    support_observations = descriptor_observations if descriptor_observations else observations
    support: dict[int, float] = {}
    support_counts: dict[int, int] = {}
    for obs in support_observations:
        support_weight = float(getattr(obs, "descriptor_weight", 0.0))
        if support_weight <= 1e-12:
            support_weight = float(obs.quality_score)
        seen_in_observation: set[int] = set()
        for element_id, weight in zip(obs.element_ids.tolist(), obs.element_weights.tolist()):
            key = int(element_id)
            support[key] = float(support.get(key, 0.0)) + float(weight) * support_weight
            if key not in seen_in_observation:
                support_counts[key] = int(support_counts.get(key, 0)) + 1
                seen_in_observation.add(key)
    if support:
        min_count = int(cfg.support_core_min_observations)
        min_fraction = float(cfg.support_core_min_fraction)
        denominator = max(int(len(support_observations)), 1)
        core_keys = [
            key
            for key in support
            if int(support_counts.get(key, 0)) >= min_count
            and float(support_counts.get(key, 0)) / float(denominator) >= min_fraction
        ]
        if core_keys:
            support = {key: float(support[key]) for key in core_keys}
    total_support = max(float(sum(support.values())), 1e-12)
    element_ids = np.asarray(sorted(support), dtype=np.int64)
    element_weights = np.asarray(
        [float(support[int(element_id)]) / total_support for element_id in element_ids.tolist()],
        dtype=np.float32,
    )
    element_row_by_id = (
        elements.row_by_element_id if row_by_id is None else row_by_id
    )
    rows = np.asarray(
        [
            element_row_by_id[int(element_id)]
            for element_id in element_ids.tolist()
        ],
        dtype=np.int64,
    )
    center, normal, covariance = _support_geometry(elements, rows, element_weights)
    support_obs_weights = np.asarray([max(float(obs.quality_score), 1e-6) for obs in support_observations], dtype=np.float64)
    support_obs_weights = support_obs_weights / max(float(np.sum(support_obs_weights)), 1e-12)
    obs_weights = np.asarray([max(float(obs.descriptor_weight), 0.0) for obs in observations], dtype=np.float64)
    if float(np.sum(obs_weights)) <= 1e-12:
        obs_weights = support_obs_weights.copy()
        feature_observations = support_observations
    else:
        feature_observations = observations
    obs_weights = obs_weights / max(float(np.sum(obs_weights)), 1e-12)
    features_all = np.stack([obs.feature for obs in feature_observations], axis=0).astype(np.float32)
    view_directions_all = np.stack([obs.view_direction for obs in feature_observations], axis=0).astype(np.float32)
    descriptor_rows = np.flatnonzero(obs_weights > 1e-12)
    if descriptor_rows.size == 0:
        descriptor_rows = np.arange(features_all.shape[0], dtype=np.int64)
    features = features_all[descriptor_rows]
    obs_weights = obs_weights[descriptor_rows]
    obs_weights = obs_weights / max(float(np.sum(obs_weights)), 1e-12)
    robust_rows = _robust_feature_rows(features, obs_weights, float(cfg.robust_feature_trim_fraction))
    robust_features = features[robust_rows]
    robust_weights = obs_weights[robust_rows]
    robust_weights = robust_weights / max(float(np.sum(robust_weights)), 1e-12)
    consensus_rows = _consensus_filtered_feature_rows(
        robust_features,
        float(cfg.min_feature_consensus_cosine),
    )
    robust_features = robust_features[consensus_rows]
    robust_weights = robust_weights[consensus_rows]
    robust_weights = robust_weights / max(float(np.sum(robust_weights)), 1e-12)
    feature_weights = robust_weights.copy()
    if str(cfg.feature_fusion_mode) == "consensus_weighted_mean":
        consensus_scores = _feature_consensus_scores(robust_features)
        consensus_weights = np.power(
            np.clip(consensus_scores, 0.0, 1.0),
            float(cfg.feature_consensus_weight_power),
        )
        feature_weights = feature_weights * consensus_weights
        feature_weights = feature_weights / max(float(np.sum(feature_weights)), 1e-12)
    feature = np.sum(robust_features * feature_weights[:, None], axis=0).astype(np.float32)
    feature_norm = max(float(np.linalg.norm(feature)), 1e-8)
    feature = feature / feature_norm
    view_directions = view_directions_all[descriptor_rows][robust_rows][consensus_rows]
    view_bin_features, view_bin_counts = _view_bin_features(
        robust_features,
        robust_weights,
        view_directions,
        view_bin_count=int(cfg.view_bin_count),
        feature_mode=str(cfg.view_bin_feature_mode),
        consensus_weight_power=float(cfg.feature_consensus_weight_power),
    )
    prototypes, prototype_count = _cluster_feature_prototypes(
        robust_features,
        robust_weights,
        max_prototypes=int(cfg.max_feature_prototypes),
        min_cosine=float(cfg.prototype_min_cosine),
    )
    similarities = robust_features @ feature.reshape(-1, 1)
    feature_variance = float(np.mean(np.maximum(1.0 - similarities.reshape(-1), 0.0)))
    return {
        "center": center,
        "normal": normal,
        "covariance": covariance,
        "feature": feature,
        "feature_prototypes": prototypes,
        "feature_prototype_count": int(prototype_count),
        "view_bin_features": view_bin_features,
        "view_bin_counts": view_bin_counts,
        "feature_variance": feature_variance,
        "quality": float(np.mean([obs.quality_score for obs in observations])) * float(np.log1p(len(observations))),
        "purity": float(np.mean([obs.purity_score for obs in observations])),
        "observation_count": int(len(observations)),
        "element_ids": element_ids,
        "element_weights": element_weights,
        "view_ids": sorted(set(str(obs.image_id) for obs in observations)),
    }


def _empty_anchor_map(feature_dim: int, metadata: Mapping[str, object] | None = None) -> Vfm2DgsAnchorMap:
    return Vfm2DgsAnchorMap(
        anchor_ids=np.zeros((0,), dtype=np.int64),
        centers=np.zeros((0, 3), dtype=np.float64),
        normals=np.zeros((0, 3), dtype=np.float32),
        covariances=np.zeros((0, 3, 3), dtype=np.float32),
        features=np.zeros((0, int(feature_dim)), dtype=np.float32),
        feature_prototypes=np.zeros((0, 1, int(feature_dim)), dtype=np.float32),
        feature_prototype_counts=np.zeros((0,), dtype=np.int64),
        view_bin_features=np.zeros((0, 1, int(feature_dim)), dtype=np.float32),
        view_bin_counts=np.zeros((0, 1), dtype=np.int64),
        feature_variances=np.zeros((0,), dtype=np.float32),
        quality_scores=np.zeros((0,), dtype=np.float32),
        purity_scores=np.zeros((0,), dtype=np.float32),
        observation_counts=np.zeros((0,), dtype=np.int64),
        surface_support_counts=np.zeros((0,), dtype=np.int64),
        support_offsets=np.zeros((1,), dtype=np.int64),
        support_element_ids=np.zeros((0,), dtype=np.int64),
        support_parent_gaussian_indices=np.zeros((0,), dtype=np.int64),
        support_weights=np.zeros((0,), dtype=np.float32),
        observed_view_ids=(),
        distinctiveness_scores=np.zeros((0,), dtype=np.float32),
        stability_scores=np.zeros((0,), dtype=np.float32),
        metadata=metadata or {},
    )


def anchor_map_summary(anchor_map: Vfm2DgsAnchorMap, observation_count: int, surface_element_count: int) -> dict[str, object]:
    def stats(values: np.ndarray) -> dict[str, float | int]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"count": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "count": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p90": float(np.percentile(arr, 90.0)),
            "max": float(np.max(arr)),
        }

    covisibility_degrees = np.diff(anchor_map.covisibility_offsets).astype(np.float32, copy=False)
    return {
        "surface_element_count": int(surface_element_count),
        "observation_count": int(observation_count),
        "anchor_count": int(len(anchor_map)),
        "feature_dim": int(anchor_map.feature_dim),
        "observation_count_stats": stats(anchor_map.observation_counts),
        "support_count_stats": stats(anchor_map.surface_support_counts),
        "quality_stats": stats(anchor_map.quality_scores),
        "purity_stats": stats(anchor_map.purity_scores),
        "distinctiveness_stats": stats(anchor_map.distinctiveness_scores),
        "stability_stats": stats(anchor_map.stability_scores),
        "feature_variance_stats": stats(anchor_map.feature_variances),
        "covisibility_edge_count": int(anchor_map.covisibility_anchor_ids.shape[0]),
        "covisibility_degree_stats": stats(covisibility_degrees),
        "covisibility_score_stats": stats(anchor_map.covisibility_scores),
    }


def _vfm_2dgs_anchor_source_gaussian_indices(anchor_map: Vfm2DgsAnchorMap) -> np.ndarray:
    source_gaussian_indices = np.full((len(anchor_map),), -1, dtype=np.int64)
    for row in range(len(anchor_map)):
        start = int(anchor_map.support_offsets[row])
        end = int(anchor_map.support_offsets[row + 1])
        parents = np.asarray(anchor_map.support_parent_gaussian_indices[start:end], dtype=np.int64)
        if parents.size:
            values, counts = np.unique(parents, return_counts=True)
            source_gaussian_indices[row] = int(values[int(np.argmax(counts))])
    return source_gaussian_indices


def vfm_2dgs_anchor_map_to_semidense(
    anchor_map: Vfm2DgsAnchorMap,
    descriptor_mode: str = "mean",
):
    """Represent VFM-2DGS anchors as a SemiDenseAnchorMap for patch-to-3D evaluators."""
    from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap

    mode = str(descriptor_mode).strip().lower()
    if mode not in {"mean", "prototypes", "view_bins"}:
        raise ValueError("descriptor_mode must be one of: mean, prototypes, view_bins")
    source_gaussian_by_anchor = _vfm_2dgs_anchor_source_gaussian_indices(anchor_map)
    source_anchor_count = int(len(anchor_map))
    anchor_rows: list[int] = []
    descriptor_ids: list[int] = []
    features: list[np.ndarray] = []
    if mode == "mean":
        for row in range(source_anchor_count):
            anchor_rows.append(row)
            descriptor_ids.append(0)
            features.append(anchor_map.features[row])
    elif mode == "prototypes":
        for row in range(source_anchor_count):
            count = min(
                max(int(anchor_map.feature_prototype_counts[row]), 1),
                int(anchor_map.feature_prototypes.shape[1]),
            )
            for prototype_id in range(count):
                anchor_rows.append(row)
                descriptor_ids.append(prototype_id)
                features.append(anchor_map.feature_prototypes[row, prototype_id])
    else:
        for row in range(source_anchor_count):
            valid_bins = np.flatnonzero(anchor_map.view_bin_counts[row] > 0)
            if valid_bins.size == 0:
                valid_bins = np.asarray([0], dtype=np.int64)
            for bin_id in valid_bins.tolist():
                anchor_rows.append(row)
                descriptor_ids.append(int(bin_id))
                features.append(anchor_map.view_bin_features[row, int(bin_id)])
    row_idx = np.asarray(anchor_rows, dtype=np.int64)
    descriptor_idx = np.asarray(descriptor_ids, dtype=np.int64)
    count = int(row_idx.size)
    expanded_features = (
        np.stack(features, axis=0).astype(np.float32, copy=False)
        if features
        else np.zeros((0, anchor_map.feature_dim), dtype=np.float32)
    )
    source_type = "vfm_2dgs" if mode == "mean" else f"vfm_2dgs_{mode[:-1] if mode.endswith('s') else mode}"
    if mode == "mean":
        expanded_anchor_ids = anchor_map.anchor_ids[row_idx].astype(np.int64, copy=False)
        source_track_ids = np.full((count,), -1, dtype=np.int64)
    else:
        expanded_anchor_ids = anchor_map.anchor_ids[row_idx].astype(np.int64, copy=False) * 1000 + descriptor_idx
        source_track_ids = -100_000_000 - anchor_map.anchor_ids[row_idx].astype(np.int64, copy=False)
    return SemiDenseAnchorMap(
        anchor_ids=expanded_anchor_ids.astype(np.int64, copy=False),
        xyz=anchor_map.centers[row_idx].astype(np.float64, copy=True),
        features=expanded_features,
        source_types=np.asarray([source_type] * count, dtype=str),
        source_track_ids=source_track_ids,
        source_gaussian_indices=source_gaussian_by_anchor[row_idx].astype(np.int64, copy=True),
        support_counts=anchor_map.surface_support_counts[row_idx].astype(np.int64, copy=True),
        mean_distances=np.zeros((count,), dtype=np.float32),
        feature_variances=anchor_map.feature_variances[row_idx].astype(np.float32, copy=True),
        observation_counts=anchor_map.observation_counts[row_idx].astype(np.int64, copy=True),
        visibility_counts=np.asarray([len(anchor_map.observed_view_ids[int(row)]) for row in row_idx], dtype=np.int64),
        quality_scores=anchor_map.quality_scores[row_idx].astype(np.float32, copy=True),
        opacity=np.ones((count,), dtype=np.float32),
        scale=np.sqrt(np.maximum(anchor_map.surface_support_counts[row_idx].astype(np.float32), 1.0)),
        observation_image_ids=tuple(anchor_map.observed_view_ids[int(row)] for row in row_idx),
        metadata={
            "stage": "vfm_2dgs_anchor_map_to_semidense",
            "descriptor_mode": mode,
            "source_stage": dict(anchor_map.metadata or {}).get("stage", "vfm_2dgs_anchor_mapping"),
            "source_anchor_count": source_anchor_count,
            "descriptor_count": count,
        },
    )
