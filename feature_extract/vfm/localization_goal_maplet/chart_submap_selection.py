"""Route-clean, overlap-aware planning for explicit surface-chart submaps.

The primary-comparison planner operates on a physically isolated source-only
MASt3R mapping run and known mapping camera poses.  A MoGe-3 path exists only as
a labelled system control because it would bias a DAV2-vs-MoGe comparison.  The
selector does not decode RGB, inspect query images/poses, or open held-view
geometry.  Its output is an offline chart-selection plan, not a runtime
localization index.

Two graphs are kept separate:

``coverage graph``
    Two views observe a mutually depth/normal-consistent piece of surface.

``alignment graph``
    A coverage edge also has a useful non-degenerate camera baseline.  The
    selected charts must be connected in this graph so that selecting several
    near-identical revisits cannot masquerade as a multi-view submap.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .lineage import arrays_sha256, canonical_json_sha256, file_sha256


SCHEMA = "goal_maplet_overlap_aware_chart_submap_plan_v2"
CARDINALITY_SCHEMA = "goal_maplet_overlap_aware_chart_submap_plan_v3"
SUPPORTED_SCHEMAS = (SCHEMA, CARDINALITY_SCHEMA)
DISJOINT_AUTHORITY_SCHEMA = "goal_maplet_disjoint_chart_upstream_authority_v2"
ALIGNMENT_SELECTION_CONTRACT = (
    "load_plan_with_expected_content_sha256_then_use_exact_ordered_names_per_operational_submap"
)
PAIRED_STRIDE2_DIAGNOSTIC_ALIGNMENT_ADAPTER_MODE = (
    "paired_stride2_source_geometry_control_only_v1"
)
PAIRED_STRIDE2_TOPOLOGY_CAVEAT = (
    "paired_DAV2_MoGe_MASt3R_comparison_topology_not_final_model_neutral_map"
)
PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS = (
    "source_ray_projected_exact_face_visibility_same_side_v1"
)
PROJECTIVE_FORMAL_EDGE_DEFINITION = (
    "AND of both direction formal-valid rows; no bidirectional pooling"
)
PROJECT_SUPPORT_SEMANTICS = "source_projection_pixel_center_v2"
PROJECTION_PRINCIPAL_POINT_CONVENTION = (
    "pixel_centers_cx=(W-1)/2_cy=(H-1)/2"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ChartSelectionConfig:
    sample_stride: int = 4
    absolute_depth_tolerance_m: float = 0.30
    relative_depth_tolerance: float = 0.025
    maximum_unsigned_normal_angle_deg: float = 35.0
    minimum_same_surface_side_fraction: float = 0.80
    minimum_symmetric_surface_overlap: float = 0.08
    minimum_alignment_baseline_m: float = 0.50
    maximum_alignment_baseline_m: float = 20.0
    minimum_baseline_to_depth_ratio: float = 0.025
    maximum_baseline_to_depth_ratio: float = 0.70
    minimum_median_triangulation_angle_deg: float = 1.5
    maximum_median_triangulation_angle_deg: float = 60.0
    maximum_camera_forward_angle_deg: float = 70.0
    minimum_submap_views: int = 3
    target_supported_view_fraction: float = 0.90
    target_per_view_surface_support: float = 0.25
    minimum_selected_charts_per_submap: int = 2
    maximum_selected_charts_per_submap: int = 16
    maximum_self_reprojection_p90_fraction_of_sample_stride: float = 0.50
    maximum_aggregate_median_reprojection_bias_px: float = 0.25

    def validated(self) -> "ChartSelectionConfig":
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if self.absolute_depth_tolerance_m <= 0:
            raise ValueError("absolute depth tolerance must be positive")
        if not 0 < self.relative_depth_tolerance < 1:
            raise ValueError("relative depth tolerance must be in (0, 1)")
        if not 0 < self.maximum_unsigned_normal_angle_deg <= 90:
            raise ValueError("normal angle must be in (0, 90]")
        if not 0 <= self.minimum_same_surface_side_fraction <= 1:
            raise ValueError("same-surface-side fraction must be in [0, 1]")
        if not 0 < self.minimum_symmetric_surface_overlap <= 1:
            raise ValueError("surface overlap threshold must be in (0, 1]")
        if not 0 <= self.minimum_alignment_baseline_m < self.maximum_alignment_baseline_m:
            raise ValueError("invalid alignment baseline interval")
        if not 0 < self.minimum_baseline_to_depth_ratio < self.maximum_baseline_to_depth_ratio:
            raise ValueError("invalid relative-baseline interval")
        if not 0 < self.minimum_median_triangulation_angle_deg < self.maximum_median_triangulation_angle_deg < 180:
            raise ValueError("invalid triangulation-angle interval")
        if not 0 < self.maximum_camera_forward_angle_deg < 180:
            raise ValueError("invalid camera-forward angle")
        if self.minimum_submap_views < 2:
            raise ValueError("a submap needs at least two views")
        if not 0 < self.target_supported_view_fraction <= 1:
            raise ValueError("target supported-view fraction must be in (0, 1]")
        if not 0 < self.target_per_view_surface_support <= 1:
            raise ValueError("per-view surface support must be in (0, 1]")
        if self.minimum_selected_charts_per_submap < 2:
            raise ValueError("minimum selected charts must be at least two")
        if (
            self.maximum_selected_charts_per_submap
            < self.minimum_selected_charts_per_submap
        ):
            raise ValueError("selected-chart cardinality interval is invalid")
        if not 0 < self.maximum_self_reprojection_p90_fraction_of_sample_stride <= 1:
            raise ValueError("self-reprojection stride fraction must be in (0, 1]")
        if not 0 < self.maximum_aggregate_median_reprojection_bias_px < 0.5:
            raise ValueError("aggregate reprojection-bias threshold must be in (0, 0.5)")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class MappingChartView:
    name: str
    route: str
    camera_to_world: np.ndarray
    focal_px: float
    points_camera: np.ndarray
    depth_camera: np.ndarray
    normals_camera: np.ndarray
    valid: np.ndarray
    initializer_file_sha256: str
    initializer_content_sha256: str
    geometry_source: str = "unknown"

    def validated(self) -> "MappingChartView":
        h, w = self.depth_camera.shape
        if self.points_camera.shape != (h, w, 3):
            raise ValueError(f"{self.name}: point/depth shape mismatch")
        if self.normals_camera.shape != (h, w, 3) or self.valid.shape != (h, w):
            raise ValueError(f"{self.name}: normal/valid shape mismatch")
        if self.camera_to_world.shape != (4, 4):
            raise ValueError(f"{self.name}: invalid camera pose")
        if not np.isfinite(self.camera_to_world).all() or self.focal_px <= 0:
            raise ValueError(f"{self.name}: invalid camera calibration")
        valid = np.asarray(self.valid, bool)
        valid &= np.isfinite(self.points_camera).all(2)
        valid &= np.isfinite(self.depth_camera) & (self.depth_camera > 0)
        valid &= np.isfinite(self.normals_camera).all(2)
        if int(valid.sum()) < 64:
            raise ValueError(f"{self.name}: insufficient valid surface support")
        if self.route != self.name.split("__", 1)[0]:
            raise ValueError(f"{self.name}: route/name mismatch")
        return self


@dataclass(frozen=True)
class ChartSubmapPlan:
    chart_names: np.ndarray
    camera_centers_world: np.ndarray
    camera_forward_world: np.ndarray
    valid_sample_counts: np.ndarray
    directional_frustum_fraction: np.ndarray
    directional_depth_support: np.ndarray
    directional_surface_support: np.ndarray
    symmetric_surface_overlap: np.ndarray
    camera_baseline_m: np.ndarray
    median_overlap_depth_m: np.ndarray
    baseline_to_depth_ratio: np.ndarray
    median_triangulation_angle_deg: np.ndarray
    camera_forward_angle_deg: np.ndarray
    same_surface_side_fraction: np.ndarray
    coverage_edges: np.ndarray
    alignment_edges: np.ndarray
    component_ids: np.ndarray
    selected_mask: np.ndarray
    selection_rank: np.ndarray
    metadata: dict[str, object]

    @property
    def chart_count(self) -> int:
        return int(len(self.chart_names))

    @property
    def selected_chart_names_in_order(self) -> tuple[str, ...]:
        rows = np.flatnonzero(self.selected_mask)
        rows = rows[np.argsort(self.selection_rank[rows])]
        return tuple(str(self.chart_names[row]) for row in rows)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "chart_names",
                "camera_centers_world",
                "camera_forward_world",
                "valid_sample_counts",
                "directional_frustum_fraction",
                "directional_depth_support",
                "directional_surface_support",
                "symmetric_surface_overlap",
                "camera_baseline_m",
                "median_overlap_depth_m",
                "baseline_to_depth_ratio",
                "median_triangulation_angle_deg",
                "camera_forward_angle_deg",
                "same_surface_side_fraction",
                "coverage_edges",
                "alignment_edges",
                "component_ids",
                "selected_mask",
                "selection_rank",
            )
        }

    def validated(self) -> "ChartSubmapPlan":
        arrays = self.arrays()
        count = self.chart_count
        if count < 1 or len(set(self.chart_names.tolist())) != count:
            raise ValueError("chart inventory is empty or non-unique")
        if arrays["camera_centers_world"].shape != (count, 3):
            raise ValueError("invalid camera centers")
        if arrays["camera_forward_world"].shape != (count, 3):
            raise ValueError("invalid camera forwards")
        matrix_names = (
            "directional_frustum_fraction",
            "directional_depth_support",
            "directional_surface_support",
            "symmetric_surface_overlap",
            "camera_baseline_m",
            "median_overlap_depth_m",
            "baseline_to_depth_ratio",
            "median_triangulation_angle_deg",
            "camera_forward_angle_deg",
            "same_surface_side_fraction",
            "coverage_edges",
            "alignment_edges",
        )
        if any(arrays[name].shape != (count, count) for name in matrix_names):
            raise ValueError("invalid pairwise matrix shape")
        if arrays["valid_sample_counts"].shape != (count,):
            raise ValueError("invalid sample counts")
        if arrays["component_ids"].shape != (count,):
            raise ValueError("invalid component ids")
        if arrays["selected_mask"].shape != (count,) or arrays["selection_rank"].shape != (count,):
            raise ValueError("invalid selection arrays")
        for name in (
            "camera_centers_world",
            "camera_forward_world",
            "directional_frustum_fraction",
            "directional_depth_support",
            "directional_surface_support",
            "symmetric_surface_overlap",
            "camera_baseline_m",
            "median_overlap_depth_m",
            "baseline_to_depth_ratio",
            "median_triangulation_angle_deg",
            "camera_forward_angle_deg",
            "same_surface_side_fraction",
        ):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"nonfinite {name}")
        for name in (
            "directional_frustum_fraction",
            "directional_depth_support",
            "directional_surface_support",
            "symmetric_surface_overlap",
            "same_surface_side_fraction",
        ):
            if np.any((arrays[name] < 0) | (arrays[name] > 1)):
                raise ValueError(f"{name} outside [0, 1]")
        if not np.array_equal(arrays["coverage_edges"], arrays["coverage_edges"].T):
            raise ValueError("coverage graph is not symmetric")
        if not np.array_equal(arrays["alignment_edges"], arrays["alignment_edges"].T):
            raise ValueError("alignment graph is not symmetric")
        if np.any(np.diag(arrays["coverage_edges"])) or np.any(np.diag(arrays["alignment_edges"])):
            raise ValueError("graphs must not contain self edges")
        if np.any(arrays["alignment_edges"] & ~arrays["coverage_edges"]):
            raise ValueError("alignment graph is not a subgraph of coverage graph")
        ranks = arrays["selection_rank"][arrays["selected_mask"]]
        if len(ranks) and not np.array_equal(np.sort(ranks), np.arange(len(ranks))):
            raise ValueError("selected ranks are not contiguous")
        if np.any(arrays["selection_rank"][~arrays["selected_mask"]] != -1):
            raise ValueError("unselected charts carry a selection rank")
        if self.metadata.get("artifact_type") not in SUPPORTED_SCHEMAS:
            raise ValueError("wrong chart submap plan schema")
        if self.metadata.get("uses_query_or_ground_truth") is not False:
            raise ValueError("route-clean plan must explicitly reject query/GT")
        if self.metadata.get("chart_count") != count:
            raise ValueError("metadata chart count differs from arrays")
        if self.metadata.get("selected_chart_count") != int(arrays["selected_mask"].sum()):
            raise ValueError("metadata selected count differs from arrays")
        if self.metadata.get("source_ordered_names_sha256") != canonical_json_sha256(
            [str(value) for value in self.chart_names.tolist()]
        ):
            raise ValueError("source ordered names differ from lineage")
        selected_names = list(self.selected_chart_names_in_order)
        if self.metadata.get("selected_chart_names_in_order") != selected_names:
            raise ValueError("selected ordered names differ from lineage")
        if self.metadata.get("selected_chart_names_in_order_sha256") != canonical_json_sha256(
            selected_names
        ):
            raise ValueError("selected ordered-name hash differs from lineage")
        if self.metadata.get("alignment_runner_contract") != ALIGNMENT_SELECTION_CONTRACT:
            raise ValueError("alignment runner contract is absent or unsupported")
        component_names: list[str] = []
        operational_count = 0
        components = self.metadata.get("components")
        if not isinstance(components, list):
            raise ValueError("component lineage is absent")
        for component in components:
            if not isinstance(component, dict):
                raise ValueError("invalid component lineage")
            operational = component.get("operational_coverage_pass") is True
            names = component.get("selected_chart_names")
            if not isinstance(names, list):
                raise ValueError("component selected names are absent")
            if operational:
                operational_count += 1
                component_names.extend(str(value) for value in names)
            elif names:
                raise ValueError("failed component exposes charts to alignment")
        if component_names != selected_names:
            raise ValueError("component and global selected order differ")
        if self.metadata.get("operational_submap_count") != operational_count:
            raise ValueError("metadata operational component count differs")
        if self.metadata.get("artifact_type") == CARDINALITY_SCHEMA:
            config = self.metadata.get("config")
            if not isinstance(config, dict):
                raise ValueError("cardinality-frozen plan lacks selection config")
            minimum = config.get("minimum_selected_charts_per_submap")
            maximum = config.get("maximum_selected_charts_per_submap")
            if (
                not isinstance(minimum, int)
                or not isinstance(maximum, int)
                or minimum < 2
                or maximum < minimum
            ):
                raise ValueError("cardinality-frozen plan interval is invalid")
            if (
                self.metadata.get("selection_cardinality_frozen_before_held_geometry")
                is not True
                or self.metadata.get("held_geometry_used_for_selection") is not False
            ):
                raise ValueError("cardinality selection is not frozen before held geometry")
            for component in components:
                if component.get("operational_coverage_pass") is True:
                    selected_count = component.get("selected_chart_count")
                    if (
                        not isinstance(selected_count, int)
                        or not minimum <= selected_count <= maximum
                    ):
                        raise ValueError("operational component violates cardinality interval")
        return self

    def save_npz(self, path: Path) -> dict[str, object]:
        arrays = self.validated().arrays()
        metadata = dict(self.metadata)
        metadata["arrays_sha256"] = arrays_sha256(arrays)
        metadata["content_sha256"] = canonical_json_sha256(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".temporary.npz")
        np.savez_compressed(
            temporary,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temporary.replace(path)
        return metadata

    @classmethod
    def load_npz(cls, path: Path) -> "ChartSubmapPlan":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {
                name: np.asarray(data[name])
                for name in (
                    "chart_names",
                    "camera_centers_world",
                    "camera_forward_world",
                    "valid_sample_counts",
                    "directional_frustum_fraction",
                    "directional_depth_support",
                    "directional_surface_support",
                    "symmetric_surface_overlap",
                    "camera_baseline_m",
                    "median_overlap_depth_m",
                    "baseline_to_depth_ratio",
                    "median_triangulation_angle_deg",
                    "camera_forward_angle_deg",
                    "same_surface_side_fraction",
                    "coverage_edges",
                    "alignment_edges",
                    "component_ids",
                    "selected_mask",
                    "selection_rank",
                )
            }
        if arrays_sha256(arrays) != metadata.get("arrays_sha256"):
            raise ValueError("chart submap plan arrays differ from lineage")
        content = dict(metadata)
        expected_content = content.pop("content_sha256", None)
        if expected_content != canonical_json_sha256(content):
            raise ValueError("chart submap plan metadata differs from lineage")
        return cls(metadata=metadata, **arrays).validated()


@dataclass(frozen=True)
class AlignmentSelection:
    """Exact, hash-bound chart inventory that an alignment runner may consume."""

    plan_path: str
    plan_file_sha256: str
    plan_content_sha256: str
    ordered_names: tuple[str, ...]
    operational_submaps: tuple[tuple[str, ...], ...]


def _pinned_alignment_plan(
    plan_path: Path,
    *,
    expected_plan_content_sha256: str,
) -> tuple[Path, ChartSubmapPlan]:
    if not _is_sha256(expected_plan_content_sha256):
        raise ValueError("alignment runner must provide a 64-character expected plan hash")
    plan_path = Path(plan_path)
    plan = ChartSubmapPlan.load_npz(plan_path)
    if plan.metadata.get("content_sha256") != expected_plan_content_sha256:
        raise ValueError("chart submap plan content hash differs from runner authority")
    return plan_path, plan


def _selection_from_pinned_plan(
    plan_path: Path,
    plan: ChartSubmapPlan,
    *,
    expected_plan_content_sha256: str,
) -> AlignmentSelection:
    lineage = plan.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("chart plan lacks upstream lineage")
    if lineage.get("disjoint_authority_schema") != DISJOINT_AUTHORITY_SCHEMA:
        raise ValueError("chart plan is not bound to the physically isolated v2 authority")
    if not _is_sha256(lineage.get("source_tree_sha256")):
        raise ValueError("chart plan does not bind the source camera/pointmap tree")
    submaps = tuple(
        tuple(str(value) for value in component["selected_chart_names"])
        for component in plan.metadata["components"]
        if component["operational_coverage_pass"] is True
    )
    if not submaps:
        raise ValueError("chart plan has no operational submap for alignment")
    if any(len(names) < 2 for names in submaps):
        raise ValueError("operational submap has fewer than two selected charts")
    ordered_names = tuple(value for names in submaps for value in names)
    if ordered_names != plan.selected_chart_names_in_order:
        raise ValueError("operational submaps differ from the exact ordered selection")
    return AlignmentSelection(
        plan_path=str(plan_path.resolve()),
        plan_file_sha256=file_sha256(plan_path),
        plan_content_sha256=expected_plan_content_sha256,
        ordered_names=ordered_names,
        operational_submaps=submaps,
    )


def paired_stride2_diagnostic_alignment_metadata(
    plan: ChartSubmapPlan,
) -> dict[str, object]:
    """Validate and expose the exact non-promotable stride-2 adapter contract.

    This contract deliberately does not turn the paired DAV2/MoGe/MASt3R
    topology into a model-neutral comparison inventory.  It only authorizes a
    source-geometry control alignment whose outputs remain ineligible for the
    full gate and exporters.
    """

    plan = plan.validated()
    required_exact: dict[str, object] = {
        "artifact_type": CARDINALITY_SCHEMA,
        "representation": "offline_source_physical_seam_filtered_chart_selection",
        "system_control_only": True,
        "comparison_inventory_eligible": False,
        "topology_stride": 2,
        "paired_stride2_densification_diagnostic": True,
        "source_seam_authority_formal_selector_handoff_eligible": True,
        "edge_formal_valid_explicitly_sealed": True,
        "edge_formal_valid_source": "authority_metadata_sealed_mask",
        "edge_formal_valid_definition": PROJECTIVE_FORMAL_EDGE_DEFINITION,
        "source_geometry_selection_eligible": True,
        "source_geometry_selection_scope": (
            "paired_stride2_diagnostic_selector_handoff_only"
        ),
        "topology_caveat": PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
        "comparison_domain_v3_role": "common_valid_and_lineage_parent_only",
        "source_seam_authority_production_eligible": False,
        "projective_correspondence_production_candidate": False,
        "final_model_neutral_map_topology_eligible": False,
        "promotion_eligible": False,
        "formal_selector_handoff_is_not_production_authority": True,
        "diagnostic_alignment_adapter_required": True,
        "full_gate_or_exporter_consumption_eligible": False,
        "full_gate_fail_closed_by_system_control_only": True,
        "model_neutral_alignment_loader_fail_closed": True,
        "legacy_closest_surface_diagnostic_only": False,
        "output_coverage_edges_all_projective_formal": True,
        "output_alignment_edges_all_projective_formal": True,
        "selection_geometry_source": [
            "source_only_MASt3R_reference_on_exact_stride2_topology"
        ],
        "selection_cardinality_frozen_before_held_geometry": True,
        "held_geometry_used_for_selection": False,
        "official_order_preserved_for_runner": True,
        "uses_mapping_rgb": False,
        "uses_query_or_ground_truth": False,
        "route_clean": True,
        "source_seam_authority_semantics_version": (
            PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS
        ),
    }
    for key, expected in required_exact.items():
        if plan.metadata.get(key) != expected:
            raise ValueError(
                f"paired stride-2 diagnostic plan {key} must equal {expected!r}"
            )
    if (
        plan.metadata.get("selected_chart_count") != plan.chart_count
        or not np.all(plan.selected_mask)
        or not np.array_equal(
            plan.selection_rank, np.arange(plan.chart_count, dtype=plan.selection_rank.dtype)
        )
    ):
        raise ValueError(
            "paired stride-2 diagnostic plan must contain only its exact selected inventory"
        )

    lineage = plan.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("paired stride-2 diagnostic plan lacks source lineage")
    lineage_required_exact: dict[str, object] = {
        "paired_stride2_densification_diagnostic": True,
        "projective_topology_stride": 2,
        "topology_caveat": PAIRED_STRIDE2_TOPOLOGY_CAVEAT,
        "source_seam_authority_formal_selector_handoff_eligible": True,
        "source_seam_authority_production_eligible": False,
        "source_seam_authority_semantics_version": (
            PROJECTIVE_FORMAL_AUTHORITY_SEMANTICS
        ),
        "query_or_ground_truth_consumed": False,
        "held_root_opened_by_selector": False,
        "source_rgb_numeric_fields_used_by_selector": False,
        "source_tree_bytes_replayed_by_selector": True,
    }
    for key, expected in lineage_required_exact.items():
        if lineage.get(key) != expected:
            raise ValueError(
                f"paired stride-2 diagnostic lineage {key} must equal {expected!r}"
            )
    lineage_hashes = (
        "paired_stride2_topology_arrays_sha256",
        "paired_stride2_topology_content_sha256",
        "paired_stride2_topology_file_sha256",
        "reference_safe_v3_domain_content_sha256",
        "reference_safe_v3_domain_file_sha256",
        "source_seam_authority_content_sha256",
        "source_seam_authority_file_sha256",
        "edge_formal_valid_sha256",
        "source_seam_m0_metrics_sha256",
    )
    if any(not _is_sha256(lineage.get(key)) for key in lineage_hashes):
        raise ValueError("paired stride-2 diagnostic lineage hashes are incomplete")

    propagated_keys = (
        "system_control_only",
        "comparison_inventory_eligible",
        "topology_stride",
        "paired_stride2_densification_diagnostic",
        "source_seam_authority_formal_selector_handoff_eligible",
        "edge_formal_valid_explicitly_sealed",
        "edge_formal_valid_source",
        "edge_formal_valid_definition",
        "source_geometry_selection_eligible",
        "source_geometry_selection_scope",
        "topology_caveat",
        "comparison_domain_v3_role",
        "source_seam_authority_production_eligible",
        "projective_correspondence_production_candidate",
        "final_model_neutral_map_topology_eligible",
        "promotion_eligible",
        "formal_selector_handoff_is_not_production_authority",
        "diagnostic_alignment_adapter_required",
        "full_gate_or_exporter_consumption_eligible",
        "full_gate_fail_closed_by_system_control_only",
        "model_neutral_alignment_loader_fail_closed",
        "legacy_closest_surface_diagnostic_only",
        "output_coverage_edges_all_projective_formal",
        "output_alignment_edges_all_projective_formal",
        "selection_geometry_source",
        "source_seam_authority_semantics_version",
    )
    return {
        "diagnostic_alignment_adapter_mode": (
            PAIRED_STRIDE2_DIAGNOSTIC_ALIGNMENT_ADAPTER_MODE
        ),
        "diagnostic_alignment_adapter_used": True,
        **{key: required_exact[key] for key in propagated_keys},
    }


def load_model_neutral_alignment_selection(
    plan_path: Path,
    *,
    expected_plan_content_sha256: str,
) -> AlignmentSelection:
    """Resolve an operational primary-comparison plan without resampling.

    The caller must pin the expected content hash in its own experiment
    authority.  A MoGe-only/system-control plan, a plan with no operational
    submap, a stale hash, or a pre-v2 upstream authority all fail closed.
    """

    plan_path, plan = _pinned_alignment_plan(
        plan_path,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )
    if plan.metadata.get("comparison_inventory_eligible") is not True:
        raise ValueError("chart plan is not eligible to define the primary comparison inventory")
    if plan.metadata.get("system_control_only") is not False:
        raise ValueError("system-control chart plan cannot drive primary alignment")
    return _selection_from_pinned_plan(
        plan_path,
        plan,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )


def load_paired_stride2_diagnostic_alignment_selection(
    plan_path: Path,
    *,
    expected_plan_content_sha256: str,
    diagnostic_alignment_adapter_opt_in: bool = False,
) -> AlignmentSelection:
    """Resolve the exact control-only stride-2 plan with explicit opt-in.

    Keeping this separate from :func:`load_model_neutral_alignment_selection`
    ensures ordinary comparison, full-gate and exporter consumers continue to
    reject the paired diagnostic topology.
    """

    if diagnostic_alignment_adapter_opt_in is not True:
        raise ValueError(
            "paired stride-2 diagnostic alignment requires an explicit adapter opt-in"
        )
    plan_path, plan = _pinned_alignment_plan(
        plan_path,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )
    paired_stride2_diagnostic_alignment_metadata(plan)
    return _selection_from_pinned_plan(
        plan_path,
        plan,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )


def moge_reference_control_alignment_metadata(
    plan: ChartSubmapPlan,
) -> dict[str, object]:
    """Validate the explicit non-promotable MoGe/reference alignment adapter."""

    plan = plan.validated()
    required = {
        "artifact_type": CARDINALITY_SCHEMA,
        "representation": "offline_source_moge_reference_physical_seam_filtered_chart_selection",
        "system_control_only": True,
        "comparison_inventory_eligible": False,
        "topology_stride": 2,
        "paired_stride2_densification_diagnostic": False,
        "moge_reference_only_topology": True,
        "dav2_geometry_consumed": False,
        "paired_initializer_geometry_encoded_upstream": False,
        "source_geometry_selection_scope": "moge_reference_only_stride2_selector_control",
        "diagnostic_alignment_adapter_required": True,
        "moge_reference_alignment_adapter_required": True,
        "promotion_eligible": False,
        "final_model_neutral_map_topology_eligible": False,
        "full_gate_or_exporter_consumption_eligible": False,
        "uses_query_or_ground_truth": False,
        "held_geometry_used_for_selection": False,
        "selection_cardinality_frozen_before_held_geometry": True,
    }
    for key, expected in required.items():
        if plan.metadata.get(key) != expected:
            raise ValueError(f"MoGe/reference control plan {key} must equal {expected!r}")
    if (
        plan.metadata.get("selected_chart_count") != plan.chart_count
        or not np.all(plan.selected_mask)
        or not np.array_equal(
            plan.selection_rank,
            np.arange(plan.chart_count, dtype=plan.selection_rank.dtype),
        )
    ):
        raise ValueError("MoGe/reference control must contain its exact selected inventory")
    lineage = plan.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("MoGe/reference control lacks lineage")
    for key, expected in {
        "comparison_inventory_eligible": False,
        "system_control_only": True,
        "moge_reference_only_topology": True,
        "dav2_geometry_consumed": False,
        "paired_initializer_geometry_encoded_upstream": False,
        "query_or_ground_truth_consumed": False,
        "held_root_opened_by_selector": False,
    }.items():
        if lineage.get(key) != expected:
            raise ValueError(f"MoGe/reference control lineage {key} differs")
    for key in (
        "source_seam_authority_file_sha256",
        "source_seam_authority_content_sha256",
        "source_seam_authority_arrays_sha256",
        "edge_formal_valid_sha256",
        "optimizer_domain_file_sha256",
        "source_exact21_plan_file_sha256",
        "source_exact21_plan_content_sha256",
    ):
        if not _is_sha256(lineage.get(key)):
            raise ValueError(f"MoGe/reference control lineage {key} is missing")
    propagated = (
        "system_control_only",
        "comparison_inventory_eligible",
        "topology_stride",
        "paired_stride2_densification_diagnostic",
        "moge_reference_only_topology",
        "dav2_geometry_consumed",
        "paired_initializer_geometry_encoded_upstream",
        "source_geometry_selection_scope",
        "topology_caveat",
        "promotion_eligible",
        "final_model_neutral_map_topology_eligible",
        "full_gate_or_exporter_consumption_eligible",
    )
    return {
        "diagnostic_alignment_adapter_mode": "moge_reference_only_source_geometry_control_v1",
        "diagnostic_alignment_adapter_used": True,
        **{key: plan.metadata[key] for key in propagated},
    }


def load_moge_reference_control_alignment_selection(
    plan_path: Path,
    *,
    expected_plan_content_sha256: str,
    diagnostic_alignment_adapter_opt_in: bool = False,
) -> AlignmentSelection:
    if diagnostic_alignment_adapter_opt_in is not True:
        raise ValueError("MoGe/reference alignment requires explicit adapter opt-in")
    plan_path, plan = _pinned_alignment_plan(
        plan_path, expected_plan_content_sha256=expected_plan_content_sha256
    )
    moge_reference_control_alignment_metadata(plan)
    return _selection_from_pinned_plan(
        plan_path,
        plan,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )


def _validate_content_hash(value: dict[str, object], label: str) -> None:
    expected = value.get("content_sha256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if expected != canonical_json_sha256(payload):
        raise ValueError(f"{label} content hash does not replay")


def _tree_sha256(root: Path) -> str:
    """Replay the authority's complete source-output tree seal.

    Source RGB files are hashed as opaque bytes only; they are never decoded or
    used as selector measurements.  The held tree is deliberately not opened.
    """

    rows: list[dict[str, object]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        )
    return canonical_json_sha256(rows)


def load_route_clean_moge3_views(
    cameras_path: Path,
    initializer_dir: Path,
    *,
    mapping_routes: Iterable[str],
) -> tuple[list[MappingChartView], dict[str, object]]:
    """Load and hash-check only the declared mapping-route initializers."""

    cameras_path = Path(cameras_path)
    initializer_dir = Path(initializer_dir)
    routes = sorted(set(str(route) for route in mapping_routes))
    if not routes or any(not route for route in routes):
        raise ValueError("mapping_routes must be a non-empty unique route set")
    cameras = json.loads(cameras_path.read_text())
    required_camera_fields = {"filepaths", "focals", "cams2world"}
    if not required_camera_fields.issubset(cameras):
        raise ValueError("camera inventory lacks filepaths/focals/cams2world")
    lengths = [len(cameras[name]) for name in required_camera_fields]
    if len(set(lengths)) != 1:
        raise ValueError("camera inventory fields differ in length")
    names = [Path(path).name for path in cameras["filepaths"]]
    if len(set(names)) != len(names):
        raise ValueError("camera inventory names are not unique")
    camera_rows = {name: row for row, name in enumerate(names)}

    manifest_path = initializer_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != "goal_maplet_moge3_chart_initializer_run_v2":
        raise ValueError("initializer manifest is not focal-correct MoGe-3 v2")
    _validate_content_hash(manifest, "initializer manifest")
    if manifest.get("uses_camera_pose") is not False:
        raise ValueError("initializer manifest consumed camera pose")
    if manifest.get("uses_query_or_ground_truth") is not False:
        raise ValueError("initializer manifest consumed query or ground truth")
    if sorted(manifest.get("allowed_routes", [])) != routes:
        raise ValueError("declared mapping routes differ from initializer authority")
    if manifest.get("cameras_file_sha256") != file_sha256(cameras_path):
        raise ValueError("camera inventory differs from initializer lineage")
    if manifest.get("camera_focal_canvas_width") != 512:
        raise ValueError("only the audited 512-pixel focal canvas is supported")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or manifest.get("chart_count") != len(manifest_rows):
        raise ValueError("initializer manifest row inventory is invalid")
    row_by_name = {str(row.get("name")): row for row in manifest_rows}
    if len(row_by_name) != len(manifest_rows):
        raise ValueError("initializer manifest contains duplicate chart names")

    views: list[MappingChartView] = []
    consumed_hashes: dict[str, str] = {}
    for name in sorted(row_by_name):
        route = name.split("__", 1)[0]
        if route not in routes:
            raise ValueError(f"initializer row {name} escapes mapping routes")
        if name not in camera_rows:
            raise ValueError(f"initializer row {name} has no camera")
        path = initializer_dir / f"{name}.npz"
        actual_hash = file_sha256(path)
        row = row_by_name[name]
        if actual_hash != row.get("file_sha256"):
            raise ValueError(f"initializer file hash differs for {name}")
        with np.load(path, allow_pickle=False) as data:
            points = np.asarray(data["points_camera"], np.float64)
            depth = np.asarray(data["depth_camera"], np.float64)
            normals = np.asarray(data["normal_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
            metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("artifact_type") != "goal_maplet_moge3_chart_initializer_v2":
            raise ValueError(f"{name}: initializer schema differs")
        _validate_content_hash(metadata, f"{name} initializer")
        if metadata.get("source_name") != name:
            raise ValueError(f"{name}: source binding differs")
        if metadata.get("content_sha256") != row.get("content_sha256"):
            raise ValueError(f"{name}: initializer content differs from manifest")
        if metadata.get("uses_camera_pose") is not False:
            raise ValueError(f"{name}: pose-free initializer contract differs")
        if metadata.get("uses_query_or_ground_truth") is not False:
            raise ValueError(f"{name}: query/GT-free initializer contract differs")
        camera = camera_rows[name]
        height, width = depth.shape
        focal_px = float(cameras["focals"][camera]) * width / 512.0
        view = MappingChartView(
            name=name,
            route=route,
            camera_to_world=np.asarray(cameras["cams2world"][camera], np.float64),
            focal_px=focal_px,
            points_camera=points,
            depth_camera=depth,
            normals_camera=normals,
            valid=valid,
            initializer_file_sha256=actual_hash,
            initializer_content_sha256=str(metadata["content_sha256"]),
            geometry_source="moge3_initializer_system_control",
        ).validated()
        views.append(view)
        consumed_hashes[name] = actual_hash
    if not views:
        raise ValueError("no mapping chart initializers were loaded")
    lineage = {
        "cameras_path": str(cameras_path),
        "cameras_file_sha256": file_sha256(cameras_path),
        "initializer_manifest_path": str(manifest_path),
        "initializer_manifest_file_sha256": file_sha256(manifest_path),
        "initializer_manifest_content_sha256": manifest["content_sha256"],
        "initializer_files_sha256": consumed_hashes,
        "mapping_routes": routes,
        "camera_rows_consumed": [view.name for view in views],
        "camera_fields_consumed": ["filepaths", "focals", "cams2world"],
        "image_rgb_consumed": False,
        "query_or_ground_truth_consumed": False,
        "comparison_inventory_eligible": False,
        "system_control_only": True,
        "selection_bias_warning": (
            "MoGe-3 geometry cannot define the primary DAV2-vs-MoGe3 comparison inventory"
        ),
    }
    return views, lineage


def _pointmap_camera_geometry(
    pointmap_path: Path,
    camera_to_world: np.ndarray,
    *,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(Path(pointmap_path).read_text())
    points_world = np.asarray(payload["points"], np.float64).reshape(288, 512, 3)
    confidence = np.asarray(payload["confs"], np.float64)
    if confidence.shape != (288, 512):
        raise ValueError(f"{pointmap_path}: invalid confidence grid")
    rotation = camera_to_world[:3, :3]
    center = camera_to_world[:3, 3]
    points_camera = (points_world - center) @ rotation
    depth = points_camera[..., 2]
    normals = np.zeros_like(points_camera)
    dx = points_camera[1:-1, 2:] - points_camera[1:-1, :-2]
    dy = points_camera[2:, 1:-1] - points_camera[:-2, 1:-1]
    normals[1:-1, 1:-1] = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=2)
    normals /= np.maximum(length[..., None], 1e-12)
    confident = np.isfinite(confidence) & (confidence > confidence_threshold)
    stencil = np.zeros_like(confident)
    stencil[1:-1, 1:-1] = (
        confident[1:-1, 1:-1]
        & confident[1:-1, :-2]
        & confident[1:-1, 2:]
        & confident[:-2, 1:-1]
        & confident[2:, 1:-1]
    )
    valid = (
        stencil
        & np.isfinite(points_camera).all(2)
        & np.isfinite(depth)
        & (depth > 0)
        & (length > 1e-8)
    )
    points_camera[~valid] = np.nan
    depth[~valid] = np.nan
    normals[~valid] = 0.0
    return points_camera, depth, normals, valid


def load_disjoint_mast3r_source_views(
    source_root: Path,
    disjoint_authority_path: Path,
    *,
    confidence_threshold: float = 0.25,
) -> tuple[list[MappingChartView], dict[str, object]]:
    """Load the model-neutral source arm of a sealed source/held authority.

    The held root is never opened here.  Its exact inventory and disjointness
    are trusted only after the authority's own canonical hash and explicit
    booleans replay.
    """

    source_root = Path(source_root).resolve()
    authority_path = Path(disjoint_authority_path)
    authority = json.loads(authority_path.read_text())
    if authority.get("artifact_type") != DISJOINT_AUTHORITY_SCHEMA:
        raise ValueError(
            "primary comparison selection requires the physically isolated v2 "
            "disjoint chart upstream authority"
        )
    _validate_content_hash(authority, "disjoint chart upstream authority")
    required_true = (
        "source_held_image_disjoint",
        "source_held_route_disjoint",
        "strict_disjoint_upstream",
        "physical_source_held_input_roots_disjoint",
    )
    if any(authority.get(name) is not True for name in required_true):
        raise ValueError("source/held upstream is not strictly disjoint")
    if authority.get("forbidden_routes_opened") is not False:
        raise ValueError("forbidden route was opened upstream")
    if authority.get("uses_query_or_ground_truth") is not False:
        raise ValueError("disjoint upstream consumed query or ground truth")
    source = authority.get("source")
    held = authority.get("held")
    if not isinstance(source, dict) or not isinstance(held, dict):
        raise ValueError("disjoint authority lacks source/held records")
    if Path(str(source.get("root"))).resolve() != source_root:
        raise ValueError("source root differs from disjoint authority")
    if Path(str(held.get("root"))).resolve() == source_root:
        raise ValueError("source and held output roots are not physically distinct")
    isolated_source = authority.get("isolated_source_input")
    isolated_held = authority.get("isolated_held_input")
    if not isinstance(isolated_source, dict) or not isinstance(isolated_held, dict):
        raise ValueError("v2 authority lacks physically isolated input records")
    if Path(str(isolated_source.get("root"))).resolve() == Path(
        str(isolated_held.get("root"))
    ).resolve():
        raise ValueError("source and held input roots are not physically distinct")
    bound_hashes = (
        "isolated_inputs_file_sha256",
        "isolated_inputs_content_sha256",
        "preexecution_contract_file_sha256",
        "preexecution_contract_content_sha256",
    )
    if any(not _is_sha256(authority.get(key)) for key in bound_hashes):
        raise ValueError("v2 authority does not bind isolated inputs and preexecution contract")
    source_names = [str(value) for value in source.get("ordered_names", [])]
    held_names = [str(value) for value in held.get("ordered_names", [])]
    source_routes = set(str(value) for value in source.get("routes", []))
    held_routes = set(str(value) for value in held.get("routes", []))
    forbidden_routes = set(str(value) for value in authority.get("forbidden_routes", []))
    if not source_names or len(set(source_names)) != len(source_names):
        raise ValueError("source chart inventory is empty or duplicated")
    if not held_names or len(set(held_names)) != len(held_names):
        raise ValueError("held chart inventory is empty or duplicated")
    if source.get("image_count") != len(source_names) or held.get("image_count") != len(held_names):
        raise ValueError("source/held image count differs from ordered inventory")
    if set(source_names) & set(held_names):
        raise ValueError("source and held names overlap")
    if source_routes & (held_routes | forbidden_routes):
        raise ValueError("source routes intersect held/forbidden routes")
    if {name.split("__", 1)[0] for name in source_names} != source_routes:
        raise ValueError("source names/routes differ")
    if {name.split("__", 1)[0] for name in held_names} != held_routes:
        raise ValueError("held names/routes differ")
    if isolated_source.get("ordered_names") != source_names:
        raise ValueError("source output inventory differs from isolated input authority")
    if isolated_held.get("ordered_names") != held_names:
        raise ValueError("held output inventory differs from isolated input authority")

    cameras_path = source_root / "cameras.json"
    if file_sha256(cameras_path) != source.get("cameras_file_sha256"):
        raise ValueError("source cameras differ from disjoint authority")
    cameras = json.loads(cameras_path.read_text())
    camera_names = [Path(path).name for path in cameras["filepaths"]]
    if camera_names != source_names:
        raise ValueError("source camera order differs from authority")
    if not (len(camera_names) == len(cameras["focals"]) == len(cameras["cams2world"])):
        raise ValueError("source camera arrays differ in length")

    pointmap_rows = []
    for name in sorted(source_names):
        path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        pointmap_rows.append({"name": name, "file_sha256": file_sha256(path)})
    if canonical_json_sha256(pointmap_rows) != source.get("pointmap_inventory_sha256"):
        raise ValueError("source pointmap inventory differs from disjoint authority")
    if _tree_sha256(source_root) != source.get("tree_sha256"):
        raise ValueError("source output tree differs from disjoint authority")
    pointmap_hash = {row["name"]: row["file_sha256"] for row in pointmap_rows}

    views: list[MappingChartView] = []
    for row, name in enumerate(source_names):
        pose = np.asarray(cameras["cams2world"][row], np.float64)
        pointmap_path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        points, depth, normals, valid = _pointmap_camera_geometry(
            pointmap_path,
            pose,
            confidence_threshold=confidence_threshold,
        )
        views.append(
            MappingChartView(
                name=name,
                route=name.split("__", 1)[0],
                camera_to_world=pose,
                focal_px=float(cameras["focals"][row]),
                points_camera=points,
                depth_camera=depth,
                normals_camera=normals,
                valid=valid,
                initializer_file_sha256=pointmap_hash[name],
                initializer_content_sha256=pointmap_hash[name],
                geometry_source="source_only_mast3r_mapping_pointmap",
            ).validated()
        )
    lineage = {
        "selection_source": "source_only_mast3r_mapping_visibility",
        "comparison_inventory_eligible": True,
        "system_control_only": False,
        "disjoint_authority_schema": authority["artifact_type"],
        "disjoint_authority_path": str(authority_path.resolve()),
        "disjoint_authority_file_sha256": file_sha256(authority_path),
        "disjoint_authority_content_sha256": authority["content_sha256"],
        "source_root": str(source_root),
        "source_routes": sorted(source_routes),
        "held_routes_bound_but_not_opened": sorted(held_routes),
        "forbidden_routes": sorted(forbidden_routes),
        "source_camera_file_sha256": source["cameras_file_sha256"],
        "source_pointmap_inventory_sha256": source["pointmap_inventory_sha256"],
        "source_tree_sha256": source["tree_sha256"],
        "source_ordered_names_sha256": canonical_json_sha256(source_names),
        "source_tree_bytes_replayed_by_selector": True,
        "source_rgb_numeric_fields_used_by_selector": False,
        "mast3r_upstream_used_mapping_rgb": True,
        "held_root_opened_by_selector": False,
        "query_or_ground_truth_consumed": False,
    }
    return views, lineage


def _project_support(
    source: MappingChartView,
    target: MappingChartView,
    config: ChartSelectionConfig,
) -> tuple[float, float, float, int, float, float, float]:
    stride = config.sample_stride
    source_valid = np.asarray(source.valid[::stride, ::stride], bool)
    source_points = source.points_camera[::stride, ::stride][source_valid]
    source_normals = source.normals_camera[::stride, ::stride][source_valid]
    source_pose = source.camera_to_world
    points_world = source_points @ source_pose[:3, :3].T + source_pose[:3, 3]
    normals_world = source_normals @ source_pose[:3, :3].T

    target_pose = target.camera_to_world
    target_points = (points_world - target_pose[:3, 3]) @ target_pose[:3, :3]
    z = target_points[:, 2]
    height, width = target.depth_camera.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        x = target.focal_px * target_points[:, 0] / z + (width - 1.0) / 2.0
        y = target.focal_px * target_points[:, 1] / z + (height - 1.0) / 2.0
    in_frame = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (z > 0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    sample_count = len(source_points)
    if sample_count == 0 or not in_frame.any():
        return 0.0, 0.0, 0.0, sample_count, 0.0, 0.0, 0.0
    rows = np.flatnonzero(in_frame)
    xi = np.clip(np.rint(x[rows]).astype(np.int64), 0, width - 1)
    yi = np.clip(np.rint(y[rows]).astype(np.int64), 0, height - 1)
    target_valid = target.valid[yi, xi]
    target_depth = target.depth_camera[yi, xi]
    valid_projection = target_valid & np.isfinite(target_depth) & (target_depth > 0)
    tolerance = np.maximum(
        config.absolute_depth_tolerance_m,
        config.relative_depth_tolerance * np.minimum(z[rows], target_depth),
    )
    depth_consistent = valid_projection & (np.abs(z[rows] - target_depth) <= tolerance)

    target_normals_world = target.normals_camera[yi, xi] @ target_pose[:3, :3].T
    source_norm = np.linalg.norm(normals_world[rows], axis=1)
    target_norm = np.linalg.norm(target_normals_world, axis=1)
    normal_valid = (source_norm > 0.5) & (target_norm > 0.5)
    dot = np.zeros(len(rows), np.float64)
    dot[normal_valid] = np.abs(
        np.sum(normals_world[rows][normal_valid] * target_normals_world[normal_valid], axis=1)
        / (source_norm[normal_valid] * target_norm[normal_valid])
    )
    minimum_dot = np.cos(np.deg2rad(config.maximum_unsigned_normal_angle_deg))
    surface_consistent = depth_consistent & normal_valid & (dot >= minimum_dot)
    consistent_rows = rows[surface_consistent]
    if len(consistent_rows):
        points = points_world[consistent_rows]
        normals = target_normals_world[surface_consistent]
        source_side = np.sum((source_pose[:3, 3] - points) * normals, axis=1)
        target_side = np.sum((target_pose[:3, 3] - points) * normals, axis=1)
        same_side_fraction = float(np.mean(source_side * target_side > 0))
        source_rays = points - source_pose[:3, 3]
        target_rays = points - target_pose[:3, 3]
        source_rays /= np.maximum(np.linalg.norm(source_rays, axis=1, keepdims=True), 1e-12)
        target_rays /= np.maximum(np.linalg.norm(target_rays, axis=1, keepdims=True), 1e-12)
        triangulation = np.degrees(
            np.arccos(np.clip(np.sum(source_rays * target_rays, axis=1), -1.0, 1.0))
        )
        median_triangulation = float(np.median(triangulation))
        median_depth = float(np.median(z[consistent_rows]))
    else:
        same_side_fraction = 0.0
        median_triangulation = 0.0
        median_depth = 0.0
    denominator = float(sample_count)
    return (
        float(in_frame.sum() / denominator),
        float(depth_consistent.sum() / denominator),
        float(surface_consistent.sum() / denominator),
        sample_count,
        median_depth,
        median_triangulation,
        same_side_fraction,
    )


def _self_reprojection_error_px(view: MappingChartView) -> dict[str, float | int]:
    """Audit the pointmap/camera pixel-center convention without another view.

    MASt3R pointmaps are sampled at pixel centers.  After a 2x area resize the
    principal point is therefore ``((W-1)/2, (H-1)/2)``.  A half-pixel phase
    error is large relative to the sparse chart triangles and must fail before
    an overlap graph is allowed to become an alignment authority.
    """

    height, width = view.depth_camera.shape
    yy, xx = np.mgrid[:height, :width]
    points = np.asarray(view.points_camera, np.float64)
    valid = np.asarray(view.valid, bool).copy()
    valid &= np.isfinite(points).all(2)
    valid &= points[..., 2] > 0
    if not valid.any():
        return {"valid_count": 0, "median_px": float("inf"), "p90_px": float("inf"), "max_px": float("inf")}
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_x = (
            view.focal_px * points[..., 0] / points[..., 2] + (width - 1.0) / 2.0
        )
        projected_y = (
            view.focal_px * points[..., 1] / points[..., 2] + (height - 1.0) / 2.0
        )
    residual_x = (projected_x - xx)[valid]
    residual_y = (projected_y - yy)[valid]
    error = np.hypot(residual_x, residual_y)
    error = error[np.isfinite(error)]
    if not len(error):
        return {"valid_count": 0, "median_px": float("inf"), "p90_px": float("inf"), "max_px": float("inf")}
    return {
        "valid_count": int(len(error)),
        "median_dx_px": float(np.median(residual_x)),
        "median_dy_px": float(np.median(residual_y)),
        "median_px": float(np.median(error)),
        "p90_px": float(np.quantile(error, 0.90)),
        "max_px": float(np.max(error)),
    }


def _connected_components(edges: np.ndarray) -> np.ndarray:
    count = len(edges)
    labels = np.full(count, -1, np.int32)
    component = 0
    for seed in range(count):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = component
        while stack:
            row = stack.pop()
            for neighbor in np.flatnonzero(edges[row]):
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    stack.append(int(neighbor))
        component += 1
    return labels


def _select_component(
    rows: np.ndarray,
    directional_support: np.ndarray,
    symmetric_overlap: np.ndarray,
    alignment_edges: np.ndarray,
    config: ChartSelectionConfig,
) -> tuple[list[int], np.ndarray, bool, str]:
    if len(rows) < config.minimum_selected_charts_per_submap:
        return (
            [],
            np.zeros(len(rows), np.float64),
            False,
            "component_below_minimum_selected_cardinality",
        )
    if len(rows) < config.minimum_submap_views:
        return [], np.zeros(len(rows), np.float64), False, "component_too_small"
    if not np.any(alignment_edges[np.ix_(rows, rows)]):
        return [], np.zeros(len(rows), np.float64), False, "no_non_degenerate_alignment_edge"
    selected: list[int] = []
    coverage = np.zeros(len(rows), np.float64)
    while len(selected) < min(config.maximum_selected_charts_per_submap, len(rows)):
        best: tuple[tuple[float, ...], int, np.ndarray] | None = None
        for local, candidate in enumerate(rows.tolist()):
            if candidate in selected:
                continue
            if not selected and not np.any(alignment_edges[candidate, rows]):
                # A coverage-only seed with no usable baseline would exhaust
                # the frontier immediately even when another part of the
                # component contains a valid alignment subgraph.
                continue
            if selected and not np.any(alignment_edges[candidate, selected]):
                continue
            proposal = np.maximum(coverage, directional_support[rows, candidate])
            # A selected chart explicitly carries its own surface even if no
            # second view covers every pixel.  Alignment connectivity is
            # enforced separately by the frontier constraint above.
            proposal[local] = 1.0
            supported_fraction = float(
                np.mean(proposal >= config.target_per_view_surface_support)
            )
            score = (
                supported_fraction,
                float(np.mean(proposal) - np.mean(coverage)),
                float(np.sum(symmetric_overlap[candidate, rows])),
                -float(candidate),
            )
            if best is None or score > best[0]:
                best = (score, candidate, proposal)
        if best is None:
            break
        _, candidate, coverage = best
        selected.append(candidate)
        supported_fraction = float(
            np.mean(coverage >= config.target_per_view_surface_support)
        )
        if (
            len(selected) >= config.minimum_selected_charts_per_submap
            and supported_fraction >= config.target_supported_view_fraction
        ):
            break
    supported_fraction = float(
        np.mean(coverage >= config.target_per_view_surface_support)
    )
    success = (
        len(selected) >= config.minimum_selected_charts_per_submap
        and supported_fraction >= config.target_supported_view_fraction
    )
    if not success and len(selected) >= config.maximum_selected_charts_per_submap:
        reason = "selection_budget_exhausted_before_coverage_target"
    elif not success:
        reason = "alignment_frontier_exhausted_before_cardinality_and_coverage_target"
    else:
        reason = "passed_operational_submap_coverage_rule"
    return selected, coverage, success, reason


def build_chart_submap_plan(
    views: list[MappingChartView],
    *,
    config: ChartSelectionConfig,
    lineage: dict[str, object],
) -> ChartSubmapPlan:
    config = config.validated()
    views = [view.validated() for view in views]
    if len({view.name for view in views}) != len(views):
        raise ValueError("mapping chart names are not unique")
    self_reprojection = {
        view.name: _self_reprojection_error_px(view)
        for view in views
    }
    self_reprojection_threshold_px = (
        config.maximum_self_reprojection_p90_fraction_of_sample_stride
        * config.sample_stride
    )
    aggregate_median_dx_px = float(np.median([
        row["median_dx_px"] for row in self_reprojection.values()
    ]))
    aggregate_median_dy_px = float(np.median([
        row["median_dy_px"] for row in self_reprojection.values()
    ]))
    self_reprojection_shape_pass = all(
        row["valid_count"] > 0
        and np.isfinite(row["p90_px"])
        and row["p90_px"] <= self_reprojection_threshold_px
        for row in self_reprojection.values()
    )
    self_reprojection_phase_pass = (
        abs(aggregate_median_dx_px)
        <= config.maximum_aggregate_median_reprojection_bias_px
        and abs(aggregate_median_dy_px)
        <= config.maximum_aggregate_median_reprojection_bias_px
    )
    self_reprojection_pass = (
        self_reprojection_shape_pass and self_reprojection_phase_pass
    )
    if not self_reprojection_pass:
        failures = {
            name: row["p90_px"]
            for name, row in self_reprojection.items()
            if row["valid_count"] <= 0
            or not np.isfinite(row["p90_px"])
            or row["p90_px"] > self_reprojection_threshold_px
        }
        raise ValueError(
            "mapping pointmap/camera self-reprojection failed the pixel-center contract: "
            f"{failures}"
        )
    count = len(views)
    frustum = np.zeros((count, count), np.float64)
    depth_support = np.zeros((count, count), np.float64)
    surface_support = np.zeros((count, count), np.float64)
    directional_overlap_depth = np.zeros((count, count), np.float64)
    directional_triangulation = np.zeros((count, count), np.float64)
    directional_same_side = np.zeros((count, count), np.float64)
    sample_counts = np.zeros(count, np.int64)
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            values = _project_support(views[source], views[target], config)
            (
                frustum[source, target],
                depth_support[source, target],
                surface_support[source, target],
                samples,
                directional_overlap_depth[source, target],
                directional_triangulation[source, target],
                directional_same_side[source, target],
            ) = values
            sample_counts[source] = samples
    np.fill_diagonal(frustum, 1.0)
    np.fill_diagonal(depth_support, 1.0)
    np.fill_diagonal(surface_support, 1.0)
    symmetric = np.minimum(surface_support, surface_support.T)
    np.fill_diagonal(symmetric, 0.0)

    centers = np.stack([view.camera_to_world[:3, 3] for view in views])
    forwards = np.stack([view.camera_to_world[:3, 2] for view in views])
    forwards /= np.maximum(np.linalg.norm(forwards, axis=1, keepdims=True), 1e-12)
    baseline = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    median_overlap_depth = np.where(
        (directional_overlap_depth > 0) & (directional_overlap_depth.T > 0),
        0.5 * (directional_overlap_depth + directional_overlap_depth.T),
        np.maximum(directional_overlap_depth, directional_overlap_depth.T),
    )
    median_triangulation = np.minimum(
        directional_triangulation, directional_triangulation.T,
    )
    same_side = np.minimum(directional_same_side, directional_same_side.T)
    baseline_to_depth = np.divide(
        baseline,
        median_overlap_depth,
        out=np.zeros_like(baseline),
        where=median_overlap_depth > 0,
    )
    forward_dot = np.clip(forwards @ forwards.T, -1.0, 1.0)
    forward_angle = np.degrees(np.arccos(forward_dot))
    coverage_edges = symmetric >= config.minimum_symmetric_surface_overlap
    coverage_edges &= same_side >= config.minimum_same_surface_side_fraction
    coverage_edges &= forward_angle <= config.maximum_camera_forward_angle_deg
    np.fill_diagonal(coverage_edges, False)
    alignment_edges = coverage_edges.copy()
    alignment_edges &= baseline >= config.minimum_alignment_baseline_m
    alignment_edges &= baseline <= config.maximum_alignment_baseline_m
    alignment_edges &= baseline_to_depth >= config.minimum_baseline_to_depth_ratio
    alignment_edges &= baseline_to_depth <= config.maximum_baseline_to_depth_ratio
    alignment_edges &= median_triangulation >= config.minimum_median_triangulation_angle_deg
    alignment_edges &= median_triangulation <= config.maximum_median_triangulation_angle_deg
    np.fill_diagonal(alignment_edges, False)
    component_ids = _connected_components(coverage_edges)

    selected_mask = np.zeros(count, bool)
    selection_rank = np.full(count, -1, np.int32)
    components: list[dict[str, object]] = []
    global_rank = 0
    for component_id in range(int(component_ids.max()) + 1):
        rows = np.flatnonzero(component_ids == component_id)
        selected, coverage, passed, reason = _select_component(
            rows,
            surface_support,
            symmetric,
            alignment_edges,
            config,
        )
        operational_selected = selected if passed else []
        for row in operational_selected:
            selected_mask[row] = True
            selection_rank[row] = global_rank
            global_rank += 1
        components.append(
            {
                "component_id": component_id,
                "source_chart_count": int(len(rows)),
                "source_chart_names": [views[row].name for row in rows],
                "selected_chart_count": int(len(operational_selected)),
                "selected_chart_names": [views[row].name for row in operational_selected],
                "attempted_selected_chart_count": int(len(selected)),
                "attempted_selected_chart_names": [views[row].name for row in selected],
                "supported_view_fraction": float(
                    np.mean(coverage >= config.target_per_view_surface_support)
                ) if len(rows) else 0.0,
                "mean_best_selected_surface_support": float(np.mean(coverage)) if len(rows) else 0.0,
                "route_inventory": sorted({views[row].route for row in rows}),
                "operational_coverage_pass": bool(passed),
                "decision_reason": reason,
            }
        )
    passed_components = [row for row in components if row["operational_coverage_pass"]]
    chart_names = [view.name for view in views]
    selected_rows = np.flatnonzero(selected_mask)
    selected_rows = selected_rows[np.argsort(selection_rank[selected_rows])]
    selected_names = [chart_names[row] for row in selected_rows]
    artifact_type = (
        CARDINALITY_SCHEMA
        if config.minimum_selected_charts_per_submap > 2
        else SCHEMA
    )
    metadata: dict[str, object] = {
        "artifact_type": artifact_type,
        "representation": "offline_source_chart_selection_and_overlap_graph",
        "runtime_candidate_unit": "not_applicable_offline_plan",
        "chart_count": count,
        "selected_chart_count": int(selected_mask.sum()),
        "source_ordered_names_sha256": canonical_json_sha256(chart_names),
        "selected_chart_names_in_order": selected_names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(selected_names),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "alignment_runner_must_supply_expected_plan_content_sha256": True,
        "alignment_runner_must_not_resample_by_route_or_count": True,
        "coverage_component_count": len(components),
        "operational_submap_count": len(passed_components),
        "mapping_routes": sorted({view.route for view in views}),
        "uses_mapping_camera_pose": True,
        "selection_geometry_source": sorted({view.geometry_source for view in views}),
        "comparison_inventory_eligible": bool(lineage.get("comparison_inventory_eligible", False)),
        "system_control_only": bool(lineage.get("system_control_only", True)),
        "uses_mapping_rgb": False,
        "uses_query_or_ground_truth": False,
        "route_clean": True,
        "project_support_semantics_version": PROJECT_SUPPORT_SEMANTICS,
        "projection_principal_point_convention": PROJECTION_PRINCIPAL_POINT_CONVENTION,
        "project_support_self_reprojection_floor_pass": self_reprojection_pass,
        "project_support_self_reprojection_shape_pass": self_reprojection_shape_pass,
        "project_support_self_reprojection_phase_pass": self_reprojection_phase_pass,
        "project_support_self_reprojection_p90_threshold_px": self_reprojection_threshold_px,
        "project_support_aggregate_median_dx_px": aggregate_median_dx_px,
        "project_support_aggregate_median_dy_px": aggregate_median_dy_px,
        "project_support_self_reprojection_metrics": self_reprojection,
        "selection_cardinality_frozen_before_held_geometry": True,
        "held_geometry_used_for_selection": False,
        "config": config.to_dict(),
        "overlap_definition": (
            "min of bidirectional fractions of valid source samples that project inside the "
            "target, agree in z within max(abs_tol,rel_tol*near_depth), agree in unsigned normal, "
            "lie on the same tangent-plane side, and satisfy the camera-forward-angle gate"
        ),
        "alignment_edge_definition": (
            "coverage edge AND absolute/relative mapping-camera baseline and median "
            "triangulation angle inside configured non-degenerate intervals"
        ),
        "operational_submap_coverage_rule": (
            "coverage-graph component has minimum_submap_views and at least "
            "minimum_selected_charts_per_submap source charts; selected charts are connected by "
            "alignment edges; selection cannot stop before the frozen minimum cardinality and "
            "target_supported_view_fraction of component views has at least "
            "target_per_view_surface_support from a selected chart"
        ),
        "semantic_facade_completeness_claimed": False,
        "held_view_surface_completeness_claimed": False,
        "components": components,
        "lineage": dict(lineage),
    }
    return ChartSubmapPlan(
        chart_names=np.asarray(chart_names),
        camera_centers_world=centers,
        camera_forward_world=forwards,
        valid_sample_counts=sample_counts,
        directional_frustum_fraction=frustum.astype(np.float32),
        directional_depth_support=depth_support.astype(np.float32),
        directional_surface_support=surface_support.astype(np.float32),
        symmetric_surface_overlap=symmetric.astype(np.float32),
        camera_baseline_m=baseline.astype(np.float32),
        median_overlap_depth_m=median_overlap_depth.astype(np.float32),
        baseline_to_depth_ratio=baseline_to_depth.astype(np.float32),
        median_triangulation_angle_deg=median_triangulation.astype(np.float32),
        camera_forward_angle_deg=forward_angle.astype(np.float32),
        same_surface_side_fraction=same_side.astype(np.float32),
        coverage_edges=coverage_edges,
        alignment_edges=alignment_edges,
        component_ids=component_ids,
        selected_mask=selected_mask,
        selection_rank=selection_rank,
        metadata=metadata,
    ).validated()


__all__ = [
    "CARDINALITY_SCHEMA",
    "ChartSelectionConfig",
    "ChartSubmapPlan",
    "MappingChartView",
    "build_chart_submap_plan",
    "load_disjoint_mast3r_source_views",
    "load_route_clean_moge3_views",
]
