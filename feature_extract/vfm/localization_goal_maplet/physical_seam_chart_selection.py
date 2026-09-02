"""Select a connected chart submap using source-only physical seams.

The coarse chart planner deliberately favours view coverage.  Its coverage
edges are therefore hypotheses that two views see the same surface, not proof
that the reference-safe chart surfaces have a usable material overlap.  This
module applies the frozen :class:`SourceSeamCorrespondenceAuthority` as a
second, source-only selection stage.

Only edges that are both reachable and geometrically valid in M0 are retained.
The result is a standard ``ChartSubmapPlan`` v3, so existing alignment runners
consume the exact ordered names without learning about this selector.  No arm
geometry and no held-view input is accepted by this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .chart_comparison_domain import (
    BASE_ARRAY_NAMES,
    topology_array_names,
    validate_exact_topology_arrays,
)
from .chart_comparison_reference_safe_domain import SCHEMA as DOMAIN_V3_SCHEMA
from .chart_submap_selection import (
    ALIGNMENT_SELECTION_CONTRACT,
    CARDINALITY_SCHEMA,
    PROJECT_SUPPORT_SEMANTICS,
    PROJECTION_PRINCIPAL_POINT_CONVENTION,
    ChartSubmapPlan,
)
from .lineage import arrays_sha256, canonical_json_sha256, file_sha256
from .projective_source_seam_authority import (
    AUTHORITY_SCHEMA as PROJECTIVE_AUTHORITY_SCHEMA,
    AUTHORITY_SEMANTICS_VERSION as PROJECTIVE_AUTHORITY_SEMANTICS_VERSION,
    EDGE_FORMAL_VALID_DEFINITION as PROJECTIVE_EDGE_FORMAL_VALID_DEFINITION,
    ProjectiveExactFaceSeamAuthority,
    ProjectiveSeamConfig,
)
from .source_seam_correspondence import (
    SourceSeamCorrespondenceAuthority,
    evaluate_source_seam_geometry,
)


AUDIT_SCHEMA = "goal_maplet_physical_seam_chart_selection_audit_v1"
PRODUCTION_AUTHORITY_SEMANTICS = (
    "source_ray_projected_exact_face_visibility_same_side_v1"
)
PRODUCTION_EDGE_FORMAL_VALID_DEFINITION = (
    "AND of both direction formal-valid rows; no bidirectional pooling"
)
LEGACY_AUTHORITY_SEMANTICS = (
    "closest_world_triangle_unsigned_normal_v1_diagnostic_only"
)
LEGACY_EDGE_FORMAL_VALID_DEFINITION = (
    "diagnostic replay of pooled M0 reachability AND geometry; not production"
)
PAIRED_STRIDE2_TOPOLOGY_CAVEAT = (
    "paired_DAV2_MoGe_MASt3R_comparison_topology_not_final_model_neutral_map"
)
PAIRED_STRIDE2_TOPOLOGY_SCHEMA = (
    "goal_maplet_paired_stride2_topology_diagnostic_v1"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class PhysicalSeamSelectionConfig:
    """Frozen source-only cardinality and coverage preferences."""

    minimum_selected_charts: int = 12
    maximum_selected_charts: int = 16
    target_per_view_surface_support: float | None = None
    diagnostic_allow_legacy_closest_surface: bool = False

    def validated(self) -> "PhysicalSeamSelectionConfig":
        if self.minimum_selected_charts < 2:
            raise ValueError("physical-seam submap needs at least two charts")
        if self.maximum_selected_charts < self.minimum_selected_charts:
            raise ValueError("physical-seam cardinality interval is invalid")
        if (
            self.target_per_view_surface_support is not None
            and not 0 < self.target_per_view_surface_support <= 1
        ):
            raise ValueError("physical-seam coverage threshold is invalid")
        if not isinstance(self.diagnostic_allow_legacy_closest_surface, bool):
            raise ValueError("legacy-authority diagnostic flag must be boolean")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_selected_charts": self.minimum_selected_charts,
            "maximum_selected_charts": self.maximum_selected_charts,
            "target_per_view_surface_support": (
                self.target_per_view_surface_support
            ),
            "diagnostic_allow_legacy_closest_surface": (
                self.diagnostic_allow_legacy_closest_surface
            ),
        }


@dataclass(frozen=True)
class PhysicalSeamSelectionResult:
    plan: ChartSubmapPlan | None
    audit: dict[str, object]


class PhysicalSeamSelectionFailure(ValueError):
    """Fail-closed selection error carrying a machine-readable graph audit."""

    def __init__(self, message: str, audit: Mapping[str, object]):
        super().__init__(message)
        self.audit = dict(audit)


def _connected_components(edges: np.ndarray) -> list[np.ndarray]:
    edges = np.asarray(edges, bool)
    if edges.ndim != 2 or edges.shape[0] != edges.shape[1]:
        raise ValueError("physical-seam graph must be square")
    seen = np.zeros(len(edges), bool)
    output: list[np.ndarray] = []
    for seed in range(len(edges)):
        if seen[seed]:
            continue
        stack = [seed]
        rows: list[int] = []
        seen[seed] = True
        while stack:
            row = stack.pop()
            rows.append(row)
            for neighbor in np.flatnonzero(edges[row]).tolist():
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        output.append(np.asarray(sorted(rows), np.int64))
    return output


def _is_connected(edges: np.ndarray, rows: Sequence[int]) -> bool:
    rows = np.asarray(rows, np.int64)
    if len(rows) == 0:
        return False
    if len(rows) == 1:
        return True
    local = np.asarray(edges, bool)[np.ix_(rows, rows)]
    return len(_connected_components(local)) == 1


def _graph_component_inventory(
    edges: np.ndarray, names: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    for index, component in enumerate(_connected_components(edges)):
        local = edges[np.ix_(component, component)]
        rows.append(
            {
                "component_id": index,
                "chart_count": int(len(component)),
                "chart_names_in_official_order": names[component].astype(str).tolist(),
                "edge_count": int(np.triu(local, 1).sum()),
                "isolated_singleton": bool(
                    len(component) == 1 and int(local.sum()) == 0
                ),
            }
        )
    return rows


def _edge_mask_graph_summary(
    names: np.ndarray,
    edge_chart_indices: np.ndarray,
    edge_mask: np.ndarray,
) -> dict[str, object]:
    """Replay a compact official-order graph summary from a sealed edge mask."""

    names = np.asarray(names).astype(str)
    edge_chart_indices = np.asarray(edge_chart_indices, np.int64)
    edge_mask = np.asarray(edge_mask, bool)
    if edge_chart_indices.shape != (len(edge_mask), 2):
        raise ValueError("graph-summary edge mask differs from edge inventory")
    graph = np.zeros((len(names), len(names)), bool)
    for passed, (first, second) in zip(edge_mask, edge_chart_indices):
        if passed:
            graph[first, second] = graph[second, first] = True
    components = _connected_components(graph)
    largest = max(components, key=lambda rows: (len(rows), -int(rows[0])))
    return {
        "component_count": int(len(components)),
        "edge_count": int(edge_mask.sum()),
        "isolated_chart_names": names[graph.sum(axis=1) == 0].tolist(),
        "largest_component_chart_count": int(len(largest)),
        "largest_component_chart_names": names[largest].tolist(),
    }


def _validate_upstream_project_support_contract(
    plan: ChartSubmapPlan,
) -> dict[str, object]:
    metadata = plan.metadata
    if metadata.get("project_support_semantics_version") != PROJECT_SUPPORT_SEMANTICS:
        raise ValueError("production selector requires corrected project-support semantics")
    if metadata.get(
        "projection_principal_point_convention"
    ) != PROJECTION_PRINCIPAL_POINT_CONVENTION:
        raise ValueError("production selector rejects legacy width/2 projection phase")
    if (
        metadata.get("project_support_self_reprojection_shape_pass") is not True
        or metadata.get("project_support_self_reprojection_phase_pass") is not True
        or metadata.get("project_support_self_reprojection_floor_pass") is not True
    ):
        raise ValueError("production selector requires sealed shape and phase floors")
    config = metadata.get("config")
    metrics = metadata.get("project_support_self_reprojection_metrics")
    if not isinstance(config, dict) or not isinstance(metrics, dict):
        raise ValueError("production plan lacks self-reprojection audit values")
    try:
        stride = float(config["sample_stride"])
        relative_limit = float(
            config["maximum_self_reprojection_p90_fraction_of_sample_stride"]
        )
        phase_limit = float(
            config["maximum_aggregate_median_reprojection_bias_px"]
        )
        recorded_limit = float(
            metadata["project_support_self_reprojection_p90_threshold_px"]
        )
        aggregate_dx = float(metadata["project_support_aggregate_median_dx_px"])
        aggregate_dy = float(metadata["project_support_aggregate_median_dy_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("production plan self-reprojection contract is incomplete") from error
    numeric = np.asarray(
        (stride, relative_limit, phase_limit, recorded_limit, aggregate_dx, aggregate_dy),
        np.float64,
    )
    if not np.isfinite(numeric).all() or stride <= 0 or relative_limit <= 0 or phase_limit <= 0:
        raise ValueError("production plan self-reprojection contract is invalid")
    if not np.isclose(recorded_limit, stride * relative_limit, atol=1e-12, rtol=0.0):
        raise ValueError("production plan self-reprojection threshold does not replay")
    if abs(aggregate_dx) > phase_limit or abs(aggregate_dy) > phase_limit:
        raise ValueError("production plan aggregate projection phase exceeds its floor")
    for name in plan.chart_names.astype(str).tolist():
        row = metrics.get(name)
        if not isinstance(row, dict):
            raise ValueError("production plan lacks a chart self-reprojection row")
        try:
            valid_count = int(row["valid_count"])
            p90 = float(row["p90_px"])
            median_dx = float(row["median_dx_px"])
            median_dy = float(row["median_dy_px"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("production chart self-reprojection row is incomplete") from error
        if (
            valid_count <= 0
            or not np.isfinite([p90, median_dx, median_dy]).all()
            or p90 > recorded_limit + 1e-12
        ):
            raise ValueError("production chart self-reprojection row fails replay")
    return {
        "project_support_semantics_version": PROJECT_SUPPORT_SEMANTICS,
        "projection_principal_point_convention": (
            PROJECTION_PRINCIPAL_POINT_CONVENTION
        ),
        "self_reprojection_p90_threshold_px": recorded_limit,
        "aggregate_median_dx_px": aggregate_dx,
        "aggregate_median_dy_px": aggregate_dy,
        "shape_pass": True,
        "phase_pass": True,
    }


def _coverage(
    directional_support: np.ndarray,
    selected: Sequence[int],
) -> np.ndarray:
    selected = np.asarray(selected, np.int64)
    if len(selected) == 0:
        return np.zeros(len(directional_support), np.float64)
    coverage = np.max(directional_support[:, selected], axis=1).astype(np.float64)
    coverage[selected] = 1.0
    return coverage


def _selection_metrics(
    selected: Sequence[int],
    *,
    directional_support: np.ndarray,
    edges: np.ndarray,
    edge_support: np.ndarray,
    coverage_threshold: float,
) -> dict[str, object]:
    rows = np.asarray(sorted(selected), np.int64)
    coverage = _coverage(directional_support, rows)
    local_edges = edges[np.ix_(rows, rows)]
    local_support = edge_support[np.ix_(rows, rows)]
    degrees = local_edges.sum(axis=1).astype(np.int64)
    return {
        "selected_count": int(len(rows)),
        "supported_candidate_fraction": float(
            np.mean(coverage >= coverage_threshold)
        ),
        "mean_best_directional_surface_support": float(np.mean(coverage)),
        "minimum_induced_degree": int(degrees.min()) if len(degrees) else 0,
        "induced_edge_count": int(np.triu(local_edges, 1).sum()),
        "induced_edge_support_sum": float(np.triu(local_support, 1).sum()),
        "connected": _is_connected(edges, rows),
    }


def _metric_score(metrics: Mapping[str, object]) -> tuple[float, ...]:
    """Coverage, then physical support, then connectivity, as frozen."""

    return (
        float(metrics["supported_candidate_fraction"]),
        float(metrics["mean_best_directional_surface_support"]),
        float(metrics["induced_edge_support_sum"]),
        float(metrics["induced_edge_count"]),
        float(metrics["minimum_induced_degree"]),
        float(metrics["selected_count"]),
    )


def _greedy_connected_subset(
    component: np.ndarray,
    *,
    target_count: int,
    directional_support: np.ndarray,
    edges: np.ndarray,
    edge_support: np.ndarray,
    valid_sample_counts: np.ndarray,
    coverage_threshold: float,
) -> tuple[np.ndarray, list[int], dict[str, object]]:
    """Choose a deterministic connected subset in official candidate order."""

    component = np.asarray(sorted(component.tolist()), np.int64)
    if not 2 <= target_count <= len(component):
        raise ValueError("invalid physical-seam greedy target")
    candidate_edges = [
        (int(first), int(second))
        for first in component.tolist()
        for second in component.tolist()
        if first < second and edges[first, second]
    ]
    if not candidate_edges:
        raise ValueError("physical-seam component has no usable edge")

    best_seed: tuple[tuple[float, ...], tuple[int, int]] | None = None
    for first, second in candidate_edges:
        metrics = _selection_metrics(
            (first, second),
            directional_support=directional_support,
            edges=edges,
            edge_support=edge_support,
            coverage_threshold=coverage_threshold,
        )
        degree_sum = float(edges[first, component].sum() + edges[second, component].sum())
        score = (
            *_metric_score(metrics),
            degree_sum,
            float(valid_sample_counts[first] + valid_sample_counts[second]),
            -float(first),
            -float(second),
        )
        if best_seed is None or score > best_seed[0]:
            best_seed = (score, (first, second))
    assert best_seed is not None
    selected = list(best_seed[1])
    addition_order = list(selected)

    while len(selected) < target_count:
        frontier = [
            row
            for row in component.tolist()
            if row not in selected and np.any(edges[row, selected])
        ]
        if not frontier:
            raise AssertionError("connected component exhausted its frontier")
        best_addition: tuple[tuple[float, ...], int] | None = None
        for candidate in frontier:
            proposal = selected + [candidate]
            metrics = _selection_metrics(
                proposal,
                directional_support=directional_support,
                edges=edges,
                edge_support=edge_support,
                coverage_threshold=coverage_threshold,
            )
            support_to_selected = float(edge_support[candidate, selected].sum())
            edge_count_to_selected = int(edges[candidate, selected].sum())
            score = (
                float(metrics["supported_candidate_fraction"]),
                float(metrics["mean_best_directional_surface_support"]),
                support_to_selected,
                float(edge_count_to_selected),
                float(metrics["induced_edge_support_sum"]),
                float(metrics["induced_edge_count"]),
                float(edges[candidate, component].sum()),
                float(valid_sample_counts[candidate]),
                -float(candidate),
            )
            if best_addition is None or score > best_addition[0]:
                best_addition = (score, candidate)
        assert best_addition is not None
        selected.append(best_addition[1])
        addition_order.append(best_addition[1])

    official_rows = np.asarray(sorted(selected), np.int64)
    metrics = _selection_metrics(
        official_rows,
        directional_support=directional_support,
        edges=edges,
        edge_support=edge_support,
        coverage_threshold=coverage_threshold,
    )
    if not metrics["connected"]:
        raise AssertionError("physical-seam greedy result is disconnected")
    return official_rows, addition_order, metrics


def _physical_graph_from_sealed_mask(
    candidate_names: np.ndarray,
    authority_edges: np.ndarray,
    source_supported_fraction_by_direction: np.ndarray,
    direction_formal_valid: np.ndarray,
    edge_formal_valid: np.ndarray,
    direction_support_valid: np.ndarray | None = None,
    direction_geometry_valid: np.ndarray | None = None,
    m0_per_edge: Sequence[Mapping[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    candidate_names = np.asarray(candidate_names).astype(str)
    authority_edges = np.asarray(authority_edges, np.int64)
    edge_count = len(authority_edges)
    if authority_edges.shape != (edge_count, 2):
        raise ValueError("seam authority edge inventory is invalid")
    if (
        np.any(authority_edges < 0)
        or np.any(authority_edges >= len(candidate_names))
        or np.any(authority_edges[:, 0] >= authority_edges[:, 1])
        or len({tuple(map(int, edge)) for edge in authority_edges}) != edge_count
    ):
        raise ValueError("seam authority edges must be unique ordered chart pairs")
    source_supported_fraction_by_direction = np.asarray(
        source_supported_fraction_by_direction, np.float64
    )
    direction_formal_valid = np.asarray(direction_formal_valid, bool)
    edge_formal_valid = np.asarray(edge_formal_valid, bool)
    if source_supported_fraction_by_direction.shape != (edge_count, 2):
        raise ValueError("seam authority directional support differs from edges")
    if (
        not np.isfinite(source_supported_fraction_by_direction).all()
        or np.any(source_supported_fraction_by_direction < 0)
        or np.any(source_supported_fraction_by_direction > 1)
    ):
        raise ValueError("seam authority directional support is invalid")
    if direction_formal_valid.shape != (edge_count, 2):
        raise ValueError("seam authority directional formal mask differs from edges")
    if edge_formal_valid.shape != (edge_count,):
        raise ValueError("sealed formal edge mask differs from seam authority")
    if not np.array_equal(edge_formal_valid, direction_formal_valid.all(axis=1)):
        raise ValueError(
            "sealed formal edge mask is not the AND of two independent directions"
        )
    if (direction_support_valid is None) != (direction_geometry_valid is None):
        raise ValueError(
            "directional support and geometry decisions must be supplied together"
        )
    if direction_support_valid is not None:
        direction_support_valid = np.asarray(direction_support_valid, bool)
        direction_geometry_valid = np.asarray(direction_geometry_valid, bool)
        if (
            direction_support_valid.shape != (edge_count, 2)
            or direction_geometry_valid.shape != (edge_count, 2)
        ):
            raise ValueError("directional support/geometry masks differ from edges")
        if not np.array_equal(
            direction_formal_valid,
            direction_support_valid & direction_geometry_valid,
        ):
            raise ValueError(
                "directional formal mask is not independent support AND geometry"
            )
    if m0_per_edge is not None and len(m0_per_edge) != edge_count:
        raise ValueError("M0 edge metrics differ from seam authority")
    graph = np.zeros((len(candidate_names), len(candidate_names)), bool)
    support = np.zeros((len(candidate_names), len(candidate_names)), np.float64)
    rows: list[dict[str, object]] = []
    for edge_index, edge in enumerate(authority_edges):
        first, second = map(int, edge)
        first_name = str(candidate_names[first])
        second_name = str(candidate_names[second])
        directional = source_supported_fraction_by_direction[edge_index]
        metric = m0_per_edge[edge_index] if m0_per_edge is not None else None
        if metric is not None:
            if metric.get("first") != first_name or metric.get("second") != second_name:
                raise ValueError("M0 edge order/names differ from seam authority")
            replayed_directional = np.asarray(
                metric.get("supported_fraction_by_direction"), np.float64
            )
            if not np.allclose(
                replayed_directional, directional, atol=1e-12, rtol=0.0
            ):
                raise ValueError("M0 and authority directional support differ")
            reachable: bool | None = metric.get("m0_reachability_pass") is True
            geometry: bool | None = metric.get("geometry_pass") is True
        else:
            reachable = None
            geometry = None
        passed = bool(edge_formal_valid[edge_index])
        if metric is not None and passed and not (reachable and geometry):
            raise ValueError(
                "sealed formal edge passes while replayed M0 reachability/geometry fails"
            )
        graph[first, second] = graph[second, first] = passed
        if passed:
            support[first, second] = support[second, first] = float(
                directional.min()
            )
        rows.append(
            {
                "edge_index": edge_index,
                "first": first_name,
                "second": second_name,
                "m0_reachability_pass": reachable,
                "m0_geometry_pass": geometry,
                "direction_formal_valid": direction_formal_valid[
                    edge_index
                ].astype(bool).tolist(),
                "direction_support_valid": (
                    direction_support_valid[edge_index].astype(bool).tolist()
                    if direction_support_valid is not None
                    else None
                ),
                "direction_geometry_valid": (
                    direction_geometry_valid[edge_index].astype(bool).tolist()
                    if direction_geometry_valid is not None
                    else None
                ),
                "sealed_edge_formal_valid": passed,
                "physical_edge_pass": passed,
                "rejection_reasons": [
                    reason
                    for reason, rejected in (
                        (
                            "first_to_second_support_gate_failed",
                            direction_support_valid is not None
                            and not bool(direction_support_valid[edge_index, 0]),
                        ),
                        (
                            "first_to_second_geometry_gate_failed",
                            direction_geometry_valid is not None
                            and not bool(direction_geometry_valid[edge_index, 0]),
                        ),
                        (
                            "second_to_first_support_gate_failed",
                            direction_support_valid is not None
                            and not bool(direction_support_valid[edge_index, 1]),
                        ),
                        (
                            "second_to_first_geometry_gate_failed",
                            direction_geometry_valid is not None
                            and not bool(direction_geometry_valid[edge_index, 1]),
                        ),
                        (
                            "first_to_second_legacy_formal_gate_failed",
                            direction_support_valid is None
                            and not bool(direction_formal_valid[edge_index, 0]),
                        ),
                        (
                            "second_to_first_legacy_formal_gate_failed",
                            direction_support_valid is None
                            and not bool(direction_formal_valid[edge_index, 1]),
                        ),
                        (
                            "legacy_m0_reachability_failed",
                            metric is not None and not bool(reachable),
                        ),
                        (
                            "legacy_m0_geometry_failed",
                            metric is not None and not bool(geometry),
                        ),
                    )
                    if rejected
                ],
                "minimum_directional_supported_fraction": float(
                    directional.min()
                ),
            }
        )
    return graph, support, rows


def select_physical_seam_chart_submap_plan(
    upstream_plan: ChartSubmapPlan,
    *,
    domain_chart_names: np.ndarray,
    authority_edge_chart_indices: np.ndarray,
    source_supported_fraction_by_direction: np.ndarray,
    direction_formal_valid: np.ndarray,
    edge_formal_valid: np.ndarray,
    edge_formal_valid_definition: str,
    authority_semantics_version: str,
    production_authority: bool,
    final_model_neutral_map_topology_eligible: bool,
    formal_selector_handoff_eligible: bool | None = None,
    topology_stride: int = 4,
    paired_stride2_densification_diagnostic: bool = False,
    topology_caveat: str | None = None,
    direction_support_valid: np.ndarray | None = None,
    direction_geometry_valid: np.ndarray | None = None,
    m0_per_edge: Sequence[Mapping[str, object]] | None = None,
    config: PhysicalSeamSelectionConfig = PhysicalSeamSelectionConfig(),
    lineage: Mapping[str, object] | None = None,
) -> PhysicalSeamSelectionResult:
    """Filter a coarse plan into one physically connected source-only submap."""

    config = config.validated()
    upstream_plan = upstream_plan.validated()
    formal_handoff = (
        production_authority
        if formal_selector_handoff_eligible is None
        else formal_selector_handoff_eligible
    )
    if topology_stride not in (2, 4):
        raise ValueError("physical-seam selector supports only stride 2 or stride 4")
    if production_authority and not formal_handoff:
        raise ValueError("production authority cannot disable its formal handoff")
    if topology_stride == 2:
        if (
            not formal_handoff
            or production_authority
            or not paired_stride2_densification_diagnostic
            or final_model_neutral_map_topology_eligible
            or topology_caveat != PAIRED_STRIDE2_TOPOLOGY_CAVEAT
        ):
            raise ValueError("stride-2 formal handoff contract is invalid")
    elif paired_stride2_densification_diagnostic:
        raise ValueError("stride-4 selection cannot claim stride-2 densification")
    candidate_names = np.asarray(domain_chart_names).astype(str)
    official_names = list(upstream_plan.selected_chart_names_in_order)
    if candidate_names.tolist() != official_names:
        raise ValueError(
            "reference-safe domain must equal the coarse plan official selection order"
        )
    plan_rows_by_name = {
        str(name): row for row, name in enumerate(upstream_plan.chart_names.astype(str))
    }
    if any(name not in plan_rows_by_name for name in candidate_names.tolist()):
        raise ValueError("physical-seam candidate is absent from the coarse plan")
    plan_rows = np.asarray(
        [plan_rows_by_name[name] for name in candidate_names.tolist()], np.int64
    )
    coarse_coverage = upstream_plan.coverage_edges[np.ix_(plan_rows, plan_rows)]
    coarse_alignment = upstream_plan.alignment_edges[np.ix_(plan_rows, plan_rows)]
    directional_support = upstream_plan.directional_surface_support[
        np.ix_(plan_rows, plan_rows)
    ].astype(np.float64)
    valid_sample_counts = upstream_plan.valid_sample_counts[plan_rows]

    if formal_handoff:
        if direction_support_valid is None or direction_geometry_valid is None:
            raise ValueError(
                "production selector requires independent directional support "
                "and geometry masks"
            )
        if authority_semantics_version != PRODUCTION_AUTHORITY_SEMANTICS:
            raise ValueError("production physical-seam authority semantics differ")
        if edge_formal_valid_definition != PRODUCTION_EDGE_FORMAL_VALID_DEFINITION:
            raise ValueError(
                "production seam authority does not seal independent directional gates"
            )
        project_support_audit = _validate_upstream_project_support_contract(
            upstream_plan
        )
    elif authority_semantics_version != LEGACY_AUTHORITY_SEMANTICS:
        raise ValueError("unsupported diagnostic physical-seam authority semantics")
    else:
        project_support_audit = {
            "diagnostic_legacy_plan_not_validated_for_production": True
        }
    physical_edges, edge_support, edge_rows = _physical_graph_from_sealed_mask(
        candidate_names,
        authority_edge_chart_indices,
        source_supported_fraction_by_direction,
        direction_formal_valid,
        edge_formal_valid,
        direction_support_valid,
        direction_geometry_valid,
        m0_per_edge,
    )
    if np.any(physical_edges & ~coarse_coverage):
        raise ValueError("physical-seam graph expands the coarse coverage graph")
    # A physical seam is necessary but not sufficient for the original
    # multi-view alignment: retain its pre-frozen baseline constraints too.
    usable_edges = physical_edges & coarse_alignment
    np.fill_diagonal(usable_edges, False)
    for row, edge in zip(edge_rows, authority_edge_chart_indices):
        first, second = map(int, edge)
        alignment_pass = bool(coarse_alignment[first, second])
        usable_pass = bool(usable_edges[first, second])
        row["upstream_alignment_edge_pass"] = alignment_pass
        row["usable_physical_alignment_edge_pass"] = usable_pass
        if row["physical_edge_pass"] and not alignment_pass:
            row["rejection_reasons"].append("upstream_alignment_edge_failed")
    physically_isolated = np.flatnonzero(physical_edges.sum(axis=1) == 0)
    physical_active = np.flatnonzero(physical_edges.sum(axis=1) > 0)
    isolated = np.flatnonzero(usable_edges.sum(axis=1) == 0)
    active = np.flatnonzero(usable_edges.sum(axis=1) > 0)
    active_components = [
        active[component]
        for component in _connected_components(usable_edges[np.ix_(active, active)])
    ]
    physical_component_rows = _graph_component_inventory(
        physical_edges, candidate_names
    )
    usable_component_rows = _graph_component_inventory(
        usable_edges, candidate_names
    )

    def fail_graph(reason: str, message: str) -> None:
        rejection_counts = {
            value: sum(value in row["rejection_reasons"] for row in edge_rows)
            for value in sorted(
                {
                    value
                    for row in edge_rows
                    for value in row["rejection_reasons"]
                }
            )
        }
        audit = {
            "artifact_type": AUDIT_SCHEMA,
            "formal_decision": "KILL",
            "failure_reason": reason,
            "failure_message": message,
            "uses_mapping_source_geometry": True,
            "uses_query_or_ground_truth": False,
            "held_geometry_consumed": False,
            "aligned_arm_geometry_consumed": False,
            "paired_initializer_geometry_encoded_by_v3_upstream": True,
            "source_seam_authority_semantics_version": (
                authority_semantics_version
            ),
            "source_seam_authority_production_eligible": production_authority,
            "source_seam_authority_formal_selector_handoff_eligible": (
                formal_handoff
            ),
            "topology_stride": topology_stride,
            "paired_stride2_densification_diagnostic": (
                paired_stride2_densification_diagnostic
            ),
            "topology_caveat": topology_caveat,
            "source_geometry_selection_eligible": False,
            "final_model_neutral_map_topology_eligible": (
                final_model_neutral_map_topology_eligible
            ),
            "promotion_eligible": False,
            "edge_formal_valid_definition": edge_formal_valid_definition,
            "config": config.to_dict(),
            "required_minimum_selected_charts": (
                config.minimum_selected_charts
            ),
            "maximum_selected_charts": config.maximum_selected_charts,
            "candidate_count": int(len(candidate_names)),
            "candidate_official_ordered_names": candidate_names.tolist(),
            "coarse_coverage_edge_count": int(np.triu(coarse_coverage, 1).sum()),
            "m0_physical_edge_count": int(np.triu(physical_edges, 1).sum()),
            "m0_physical_alignment_edge_count": int(
                np.triu(usable_edges, 1).sum()
            ),
            "physically_isolated_candidate_names": candidate_names[
                physically_isolated
            ].tolist(),
            "isolated_candidate_names": candidate_names[isolated].tolist(),
            "physical_graph_component_inventory": physical_component_rows,
            "usable_graph_component_inventory": usable_component_rows,
            "maximum_physical_component_chart_count": max(
                (row["chart_count"] for row in physical_component_rows),
                default=0,
            ),
            "maximum_usable_component_chart_count": max(
                (row["chart_count"] for row in usable_component_rows),
                default=0,
            ),
            "rejected_edge_count_by_reason": rejection_counts,
            "m0_edge_inventory": edge_rows,
            "source_formal_edge_inventory": edge_rows,
            "upstream_project_support_contract_audit": project_support_audit,
            "lineage": dict(lineage or {}),
            "output_plan_created": False,
        }
        raise PhysicalSeamSelectionFailure(message, audit)

    if len(active) < config.minimum_selected_charts:
        fail_graph(
            "fewer_than_minimum_nonisolated_charts",
            "physical-seam graph has fewer than the required non-isolated charts",
        )
    eligible = [
        component
        for component in active_components
        if len(component) >= config.minimum_selected_charts
    ]
    if not eligible:
        fail_graph(
            "no_connected_component_meets_minimum_cardinality",
            "physical-seam graph has no connected component meeting minimum cardinality",
        )

    upstream_config = upstream_plan.metadata.get("config")
    if not isinstance(upstream_config, dict):
        raise ValueError("coarse plan lacks its frozen coverage configuration")
    coverage_threshold = config.target_per_view_surface_support
    if coverage_threshold is None:
        coverage_threshold = float(
            upstream_config.get("target_per_view_surface_support", 0.25)
        )
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coarse plan coverage threshold is invalid")

    proposals: list[tuple[np.ndarray, list[int], dict[str, object]]] = []
    for component in eligible:
        target_count = min(config.maximum_selected_charts, len(component))
        proposal = _greedy_connected_subset(
            component,
            target_count=target_count,
            directional_support=directional_support,
            edges=usable_edges,
            edge_support=edge_support,
            valid_sample_counts=valid_sample_counts,
            coverage_threshold=coverage_threshold,
        )
        proposals.append(proposal)
    best_proposal: tuple[
        tuple[float, ...], tuple[np.ndarray, list[int], dict[str, object]]
    ] | None = None
    for proposal in proposals:
        selected, _, metrics = proposal
        score = (*_metric_score(metrics), -float(selected[0]))
        if best_proposal is None or score > best_proposal[0]:
            best_proposal = (score, proposal)
    assert best_proposal is not None
    selected, addition_order, selected_metrics = best_proposal[1]
    if not config.minimum_selected_charts <= len(selected) <= config.maximum_selected_charts:
        raise AssertionError("physical-seam selection violates its cardinality interval")
    if not _is_connected(usable_edges, selected):
        raise AssertionError("physical-seam selection is not alignment-connected")

    output_plan_rows = plan_rows[selected]
    output_names = candidate_names[selected]
    pairwise_names = (
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
    )
    pairwise = {
        name: np.asarray(getattr(upstream_plan, name))[
            np.ix_(output_plan_rows, output_plan_rows)
        ]
        for name in pairwise_names
    }
    output_physical_edges = physical_edges[np.ix_(selected, selected)]
    output_usable_edges = usable_edges[np.ix_(selected, selected)]
    if np.any(output_usable_edges & ~output_physical_edges):
        raise AssertionError("output alignment edge is not projective-formal")
    output_count = len(selected)
    selected_names = output_names.astype(str).tolist()
    dropped_names = [
        name for row, name in enumerate(candidate_names.tolist()) if row not in set(selected.tolist())
    ]
    component_rows = [
        {
            "component_id": index,
            "candidate_count": int(len(component)),
            "candidate_names_in_official_order": candidate_names[component].tolist(),
            "eligible": bool(len(component) >= config.minimum_selected_charts),
        }
        for index, component in enumerate(active_components)
    ]
    output_config = dict(upstream_config)
    output_config["minimum_selected_charts_per_submap"] = (
        config.minimum_selected_charts
    )
    output_config["maximum_selected_charts_per_submap"] = (
        config.maximum_selected_charts
    )
    output_lineage = dict(upstream_plan.metadata.get("lineage", {}))
    output_lineage.update(dict(lineage or {}))
    output_comparison_inventory_eligible = bool(
        topology_stride != 2
        and formal_handoff
        and upstream_plan.metadata.get("comparison_inventory_eligible", False)
    )
    output_system_control_only = bool(
        topology_stride == 2
        or (not formal_handoff)
        or upstream_plan.metadata.get("system_control_only", True)
    )
    # Do not leave stale upstream promotion semantics in the nested lineage.
    # Some downstream authorities intentionally replay lineage without first
    # consulting the plan's top-level fields.  Preserve the inputs under an
    # explicit upstream name and make the effective output semantics agree at
    # both levels.
    output_lineage["upstream_comparison_inventory_eligible"] = (
        output_lineage.get("comparison_inventory_eligible")
    )
    output_lineage["upstream_system_control_only"] = output_lineage.get(
        "system_control_only"
    )
    output_lineage["comparison_inventory_eligible"] = (
        output_comparison_inventory_eligible
    )
    output_lineage["system_control_only"] = output_system_control_only
    metadata: dict[str, object] = {
        "artifact_type": CARDINALITY_SCHEMA,
        "representation": "offline_source_physical_seam_filtered_chart_selection",
        "runtime_candidate_unit": "not_applicable_offline_plan",
        "chart_count": output_count,
        "selected_chart_count": output_count,
        "source_ordered_names_sha256": canonical_json_sha256(selected_names),
        "selected_chart_names_in_order": selected_names,
        "selected_chart_names_in_order_sha256": canonical_json_sha256(
            selected_names
        ),
        "alignment_runner_contract": ALIGNMENT_SELECTION_CONTRACT,
        "alignment_runner_must_supply_expected_plan_content_sha256": True,
        "alignment_runner_must_not_resample_by_route_or_count": True,
        "coverage_component_count": 1,
        "operational_submap_count": 1,
        "mapping_routes": sorted({name.split("__", 1)[0] for name in selected_names}),
        "uses_mapping_camera_pose": bool(
            upstream_plan.metadata.get("uses_mapping_camera_pose", True)
        ),
        "selection_geometry_source": [
            "source_only_MASt3R_reference_on_exact_stride"
            f"{topology_stride}_topology"
        ],
        "comparison_inventory_eligible": output_comparison_inventory_eligible,
        "system_control_only": output_system_control_only,
        "uses_mapping_rgb": False,
        "uses_query_or_ground_truth": False,
        "route_clean": True,
        "selection_cardinality_frozen_before_held_geometry": True,
        "held_geometry_used_for_selection": False,
        "config": output_config,
        "physical_seam_selection_config": config.to_dict(),
        "physical_seam_selection_config_sha256": canonical_json_sha256(
            config.to_dict()
        ),
        "source_seam_authority_semantics_version": authority_semantics_version,
        "edge_formal_valid_definition": edge_formal_valid_definition,
        "source_seam_authority_production_eligible": production_authority,
        "projective_correspondence_production_candidate": production_authority,
        "source_seam_authority_formal_selector_handoff_eligible": formal_handoff,
        "edge_formal_valid_explicitly_sealed": formal_handoff,
        "edge_formal_valid_source": (
            "authority_metadata_sealed_mask"
            if formal_handoff
            else "diagnostic_replay_of_legacy_M0_reachability_AND_geometry"
        ),
        "source_geometry_selection_eligible": formal_handoff,
        "source_geometry_selection_scope": (
            "paired_stride2_diagnostic_selector_handoff_only"
            if topology_stride == 2
            else (
                "projective_stride4_production_candidate"
                if formal_handoff
                else "legacy_diagnostic_only"
            )
        ),
        "topology_stride": topology_stride,
        "paired_stride2_densification_diagnostic": (
            paired_stride2_densification_diagnostic
        ),
        "topology_caveat": topology_caveat,
        "comparison_domain_v3_role": (
            "common_valid_and_lineage_parent_only"
            if topology_stride == 2
            else "common_valid_and_exact_stride4_topology_parent"
        ),
        "output_coverage_edges_all_projective_formal": True,
        "output_alignment_edges_all_projective_formal": True,
        "final_model_neutral_map_topology_eligible": (
            final_model_neutral_map_topology_eligible
        ),
        "promotion_eligible": bool(
            formal_handoff and final_model_neutral_map_topology_eligible
        ),
        "formal_selector_handoff_is_not_production_authority": bool(
            formal_handoff and not production_authority
        ),
        "diagnostic_alignment_adapter_required": topology_stride == 2,
        "full_gate_or_exporter_consumption_eligible": bool(
            formal_handoff
            and production_authority
            and final_model_neutral_map_topology_eligible
        ),
        "full_gate_fail_closed_by_system_control_only": topology_stride == 2,
        "model_neutral_alignment_loader_fail_closed": topology_stride == 2,
        "legacy_closest_surface_diagnostic_only": not formal_handoff,
        "upstream_project_support_semantics_version": upstream_plan.metadata.get(
            "project_support_semantics_version"
        ),
        "upstream_projection_principal_point_convention": upstream_plan.metadata.get(
            "projection_principal_point_convention"
        ),
        "upstream_project_support_self_reprojection_floor_pass": (
            upstream_plan.metadata.get("project_support_self_reprojection_floor_pass")
        ),
        "upstream_project_support_contract_audit": project_support_audit,
        "candidate_official_ordered_names": candidate_names.tolist(),
        "candidate_official_ordered_names_sha256": canonical_json_sha256(
            candidate_names.tolist()
        ),
        "dropped_candidate_names": dropped_names,
        "isolated_candidate_names": candidate_names[isolated].tolist(),
        "physically_isolated_candidate_names": candidate_names[
            physically_isolated
        ].tolist(),
        "greedy_addition_order_diagnostic_only": candidate_names[
            np.asarray(addition_order, np.int64)
        ].tolist(),
        "official_order_preserved_for_runner": True,
        "coverage_definition": (
            "coarse coverage edge AND M0 source-seam reachability AND M0 "
            "source-seam geometry pass"
        ),
        "overlap_definition": (
            "frozen source-only bidirectional projective exact-face material "
            "correspondence with support, point-to-plane, and normal gates"
        ),
        "alignment_edge_definition": (
            "physical-seam coverage edge AND the upstream plan's frozen "
            "non-degenerate alignment edge"
        ),
        "operational_submap_coverage_rule": (
            "delete nodes isolated in the physical-plus-alignment graph; choose "
            "one connected component and at most the frozen maximum charts by "
            "directional coverage, seam support, and induced connectivity"
        ),
        "semantic_facade_completeness_claimed": False,
        "held_view_surface_completeness_claimed": False,
        "components": [
            {
                "component_id": 0,
                "source_chart_count": output_count,
                "source_chart_names": selected_names,
                "selected_chart_count": output_count,
                "selected_chart_names": selected_names,
                "attempted_selected_chart_count": output_count,
                "attempted_selected_chart_names": selected_names,
                "supported_view_fraction": selected_metrics[
                    "supported_candidate_fraction"
                ],
                "mean_best_selected_surface_support": selected_metrics[
                    "mean_best_directional_surface_support"
                ],
                "route_inventory": sorted(
                    {name.split("__", 1)[0] for name in selected_names}
                ),
                "operational_coverage_pass": True,
                "decision_reason": "passed_source_physical_seam_connected_subgraph_rule",
            }
        ],
        "lineage": output_lineage,
    }
    plan = ChartSubmapPlan(
        chart_names=output_names,
        camera_centers_world=upstream_plan.camera_centers_world[output_plan_rows],
        camera_forward_world=upstream_plan.camera_forward_world[output_plan_rows],
        valid_sample_counts=upstream_plan.valid_sample_counts[output_plan_rows],
        coverage_edges=output_physical_edges,
        alignment_edges=output_usable_edges,
        component_ids=np.zeros(output_count, np.int32),
        selected_mask=np.ones(output_count, bool),
        selection_rank=np.arange(output_count, dtype=np.int32),
        metadata=metadata,
        **pairwise,
    ).validated()

    audit: dict[str, object] = {
        "artifact_type": AUDIT_SCHEMA,
        "uses_mapping_source_geometry": True,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "aligned_arm_geometry_consumed": False,
        "paired_initializer_geometry_encoded_by_v3_upstream": True,
        "paired_initializer_role": (
            "common-valid-and-reference-safe-topology-only; no aligned M1/M2 vertices"
        ),
        "selection_inputs": (
            "coarse_plan_plus_reference_safe_v3_common_valid_parent_plus_"
            f"source_seam_M0_exact_stride{topology_stride}_topology_only"
        ),
        "config": config.to_dict(),
        "source_seam_authority_semantics_version": authority_semantics_version,
        "edge_formal_valid_definition": edge_formal_valid_definition,
        "source_seam_authority_production_eligible": production_authority,
        "projective_correspondence_production_candidate": production_authority,
        "source_seam_authority_formal_selector_handoff_eligible": formal_handoff,
        "edge_formal_valid_explicitly_sealed": formal_handoff,
        "edge_formal_valid_source": (
            "authority_metadata_sealed_mask"
            if formal_handoff
            else "diagnostic_replay_of_legacy_M0_reachability_AND_geometry"
        ),
        "source_geometry_selection_eligible": formal_handoff,
        "source_geometry_selection_scope": (
            "paired_stride2_diagnostic_selector_handoff_only"
            if topology_stride == 2
            else (
                "projective_stride4_production_candidate"
                if formal_handoff
                else "legacy_diagnostic_only"
            )
        ),
        "topology_stride": topology_stride,
        "paired_stride2_densification_diagnostic": (
            paired_stride2_densification_diagnostic
        ),
        "topology_caveat": topology_caveat,
        "comparison_domain_v3_role": (
            "common_valid_and_lineage_parent_only"
            if topology_stride == 2
            else "common_valid_and_exact_stride4_topology_parent"
        ),
        "final_model_neutral_map_topology_eligible": (
            final_model_neutral_map_topology_eligible
        ),
        "promotion_eligible": bool(
            formal_handoff and final_model_neutral_map_topology_eligible
        ),
        "formal_selector_handoff_is_not_production_authority": bool(
            formal_handoff and not production_authority
        ),
        "legacy_closest_surface_diagnostic_only": not formal_handoff,
        "upstream_project_support_semantics_version": upstream_plan.metadata.get(
            "project_support_semantics_version"
        ),
        "upstream_projection_principal_point_convention": upstream_plan.metadata.get(
            "projection_principal_point_convention"
        ),
        "upstream_project_support_self_reprojection_floor_pass": (
            upstream_plan.metadata.get("project_support_self_reprojection_floor_pass")
        ),
        "upstream_project_support_contract_audit": project_support_audit,
        "coverage_threshold": coverage_threshold,
        "candidate_count": int(len(candidate_names)),
        "candidate_official_ordered_names": candidate_names.tolist(),
        "coarse_coverage_edge_count": int(np.triu(coarse_coverage, 1).sum()),
        "m0_physical_edge_count": int(np.triu(physical_edges, 1).sum()),
        "m0_physical_alignment_edge_count": int(np.triu(usable_edges, 1).sum()),
        "m0_physical_graph_connected_before_isolate_removal": bool(
            len(candidate_names) > 0
            and len(_connected_components(physical_edges)) == 1
        ),
        "m0_physical_graph_connected_after_isolate_removal": bool(
            len(physical_active) > 0
            and len(
                _connected_components(
                    physical_edges[np.ix_(physical_active, physical_active)]
                )
            )
            == 1
        ),
        "usable_graph_connected_after_isolate_removal": bool(
            len(active) > 0
            and len(_connected_components(usable_edges[np.ix_(active, active)]))
            == 1
        ),
        "physical_graph_component_inventory": physical_component_rows,
        "usable_graph_component_inventory": usable_component_rows,
        "rejected_edge_count_by_reason": {
            reason: sum(reason in row["rejection_reasons"] for row in edge_rows)
            for reason in sorted(
                {
                    reason
                    for row in edge_rows
                    for reason in row["rejection_reasons"]
                }
            )
        },
        "isolated_candidate_names": candidate_names[isolated].tolist(),
        "physically_isolated_candidate_names": candidate_names[
            physically_isolated
        ].tolist(),
        "active_component_inventory": component_rows,
        "selected_chart_count": output_count,
        "selected_chart_names_in_official_order": selected_names,
        "dropped_candidate_names": dropped_names,
        "greedy_addition_order_diagnostic_only": candidate_names[
            np.asarray(addition_order, np.int64)
        ].tolist(),
        "selected_subgraph_metrics": selected_metrics,
        "m0_edge_inventory": edge_rows,
        "source_formal_edge_inventory": edge_rows,
        "standard_chart_submap_plan_v3": True,
        "existing_alignment_runner_structurally_compatible": True,
        "existing_alignment_runner_authority_eligible": bool(
            formal_handoff and topology_stride != 2
        ),
        "diagnostic_alignment_adapter_required": topology_stride == 2,
        "full_gate_or_exporter_consumption_eligible": bool(
            formal_handoff
            and production_authority
            and final_model_neutral_map_topology_eligible
        ),
        "full_gate_fail_closed_by_system_control_only": topology_stride == 2,
        "model_neutral_alignment_loader_fail_closed": topology_stride == 2,
        "full_gate_fail_closed_contract_audit": {
            "comparison_inventory_eligible_required": True,
            "comparison_inventory_eligible_observed": metadata[
                "comparison_inventory_eligible"
            ],
            "system_control_only_required": False,
            "system_control_only_observed": metadata["system_control_only"],
            "eligible": bool(
                metadata["comparison_inventory_eligible"] is True
                and metadata["system_control_only"] is False
            ),
        },
        "output_coverage_edges_all_projective_formal": True,
        "output_alignment_edges_all_projective_formal": True,
        "downstream_domains_must_be_rebuilt_for_selected_inventory": True,
    }
    return PhysicalSeamSelectionResult(plan=plan, audit=audit)


def _load_reference_safe_domain(
    path: Path,
    *,
    expected_content_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(path)
    names = tuple(BASE_ARRAY_NAMES) + topology_array_names()
    with np.load(path, allow_pickle=False) as data:
        if any(name not in data.files for name in names):
            raise ValueError("reference-safe v3 domain array inventory is incomplete")
        arrays = {name: np.asarray(data[name]) for name in names}
        metadata = json.loads(str(data["metadata_json"].item()))
    content = dict(metadata)
    claimed = content.pop("content_sha256", None)
    if claimed != canonical_json_sha256(content):
        raise ValueError("reference-safe v3 domain metadata hash differs")
    if claimed != expected_content_sha256:
        raise ValueError("reference-safe v3 domain differs from experiment pin")
    if metadata.get("arrays_sha256") != arrays_sha256(arrays):
        raise ValueError("reference-safe v3 domain arrays differ from lineage")
    if metadata.get("artifact_type") != DOMAIN_V3_SCHEMA:
        raise ValueError("physical-seam selector requires a reference-safe v3 domain")
    if (
        metadata.get("uses_query_or_ground_truth") is not False
        or metadata.get("source_reference_edge_safe") is not True
        or metadata.get("full_submap_gate_primary_stride") != 4
    ):
        raise ValueError("reference-safe v3 domain is not source-only physical topology")
    base = {name: arrays[name] for name in BASE_ARRAY_NAMES}
    topology = {name: arrays[name] for name in topology_array_names()}
    validate_exact_topology_arrays(
        base,
        topology,
        expected_sha256=metadata.get("exact_topology_arrays_sha256"),
    )
    return (
        np.asarray(base["chart_names"]).astype(str),
        np.asarray(base["valid"], bool),
        metadata,
    )


def _peek_artifact_type(path: Path) -> str:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    return str(metadata.get("artifact_type", ""))


def _projective_self_reprojection_audit(
    authority: ProjectiveExactFaceSeamAuthority,
) -> dict[str, object]:
    config = ProjectiveSeamConfig(**authority.metadata["config"]).validated()
    rows: list[dict[str, object]] = []
    for chart, name in enumerate(authority.chart_names.astype(str).tolist()):
        lo, hi = map(int, authority.chart_vertex_offsets[chart : chart + 2])
        error = np.asarray(authority.self_reprojection_error_px[lo:hi], np.float64)
        if not len(error) or not np.isfinite(error).all():
            raise ValueError("projective authority self-reprojection row is invalid")
        p90 = float(np.quantile(error, 0.90))
        passed = p90 <= config.maximum_selected_chart_self_reprojection_p90_px
        rows.append(
            {
                "name": name,
                "vertex_count": int(len(error)),
                "p50_px": float(np.quantile(error, 0.50)),
                "p90_px": p90,
                "maximum_px": float(error.max()),
                "p90_pass": passed,
            }
        )
    if not all(row["p90_pass"] for row in rows):
        failed = [row["name"] for row in rows if not row["p90_pass"]]
        raise ValueError(
            f"projective authority selected-chart self-reprojection failed: {failed}"
        )
    return {
        "maximum_selected_chart_self_reprojection_p90_px": (
            config.maximum_selected_chart_self_reprojection_p90_px
        ),
        "all_selected_charts_pass": True,
        "per_chart": rows,
    }


def _validate_paired_stride2_topology_parent(
    authority: ProjectiveExactFaceSeamAuthority,
    authority_path: Path,
    *,
    domain_common_valid: np.ndarray,
    expected_domain_content_sha256: str,
    expected_plan_content_sha256: str,
) -> dict[str, object]:
    """Replay the content-pinned stride-2 topology embedded by an authority."""

    metadata = authority.metadata
    filename = metadata.get("paired_stride2_topology_filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError("stride-2 authority lacks a stable topology filename")
    topology_path = Path(authority_path).parent / filename
    if not topology_path.is_file():
        raise ValueError("stride-2 topology sibling is absent")
    expected_file = metadata.get("paired_stride2_topology_file_sha256")
    expected_content = metadata.get("paired_stride2_topology_content_sha256")
    expected_arrays = metadata.get("paired_stride2_topology_arrays_sha256")
    if any(
        not _is_sha256(value)
        for value in (expected_file, expected_content, expected_arrays)
    ):
        raise ValueError("stride-2 topology hashes are incomplete")
    observed_file = file_sha256(topology_path)
    if observed_file != expected_file:
        raise ValueError("stride-2 topology file differs from authority")
    with np.load(topology_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("stride-2 topology lacks metadata")
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
        topology_metadata = json.loads(str(data["metadata_json"].item()))
    content_payload = dict(topology_metadata)
    claimed_content = content_payload.pop("content_sha256", None)
    if (
        claimed_content != canonical_json_sha256(content_payload)
        or claimed_content != expected_content
    ):
        raise ValueError("stride-2 topology content hash differs")
    observed_arrays = arrays_sha256(arrays)
    if (
        topology_metadata.get("arrays_sha256") != observed_arrays
        or observed_arrays != expected_arrays
    ):
        raise ValueError("stride-2 topology arrays hash differs")
    if (
        topology_metadata.get("artifact_type")
        != PAIRED_STRIDE2_TOPOLOGY_SCHEMA
        or topology_metadata.get("comparison_domain_v3_content_sha256")
        != expected_domain_content_sha256
        or topology_metadata.get("frozen_submap_plan_content_sha256")
        != expected_plan_content_sha256
        or topology_metadata.get("topology_caveat")
        != PAIRED_STRIDE2_TOPOLOGY_CAVEAT
    ):
        raise ValueError("stride-2 topology semantic lineage differs")
    required = {
        "chart_names",
        "valid",
        "sampled_vertex_offsets_stride2",
        "sampled_vertex_pixel_indices_stride2",
        "face_offsets_stride2",
        "faces_stride2",
    }
    if not required.issubset(arrays):
        raise ValueError("stride-2 topology array inventory is incomplete")
    comparisons = (
        (arrays["chart_names"].astype(str), authority.chart_names.astype(str)),
        (np.asarray(arrays["valid"], bool), domain_common_valid),
        (
            np.asarray(arrays["sampled_vertex_offsets_stride2"], np.int64),
            np.asarray(authority.chart_vertex_offsets, np.int64),
        ),
        (
            np.asarray(arrays["sampled_vertex_pixel_indices_stride2"], np.int64),
            np.asarray(authority.sampled_vertex_pixel_indices, np.int64),
        ),
        (
            np.asarray(arrays["face_offsets_stride2"], np.int64),
            np.asarray(authority.chart_face_offsets, np.int64),
        ),
        (
            np.asarray(arrays["faces_stride2"], np.int64),
            np.asarray(authority.faces, np.int64),
        ),
    )
    if any(not np.array_equal(observed, expected) for observed, expected in comparisons):
        raise ValueError("stride-2 authority does not embed its sealed topology")
    return {
        "topology_path": str(topology_path.resolve()),
        "topology_filename": filename,
        "topology_file_sha256": observed_file,
        "topology_content_sha256": claimed_content,
        "topology_arrays_sha256": observed_arrays,
        "embedded_topology_bit_equal": True,
        "common_valid_parent_bit_equal": True,
    }


def build_physical_seam_chart_submap_plan(
    upstream_plan_path: Path,
    *,
    expected_upstream_plan_content_sha256: str,
    reference_safe_domain_v3_path: Path,
    expected_reference_safe_domain_content_sha256: str,
    source_seam_authority_path: Path,
    expected_source_seam_authority_content_sha256: str,
    config: PhysicalSeamSelectionConfig = PhysicalSeamSelectionConfig(),
) -> PhysicalSeamSelectionResult:
    """Replay three pinned source-only inputs and produce the filtered plan."""

    pins = (
        expected_upstream_plan_content_sha256,
        expected_reference_safe_domain_content_sha256,
        expected_source_seam_authority_content_sha256,
    )
    if any(not _is_sha256(value) for value in pins):
        raise ValueError("physical-seam selector requires three explicit SHA-256 pins")
    upstream_plan_path = Path(upstream_plan_path)
    reference_safe_domain_v3_path = Path(reference_safe_domain_v3_path)
    source_seam_authority_path = Path(source_seam_authority_path)

    upstream_plan = ChartSubmapPlan.load_npz(upstream_plan_path)
    if upstream_plan.metadata.get("content_sha256") != (
        expected_upstream_plan_content_sha256
    ):
        raise ValueError("coarse plan differs from experiment pin")
    if (
        upstream_plan.metadata.get("uses_query_or_ground_truth") is not False
        or upstream_plan.metadata.get("held_geometry_used_for_selection") is not False
    ):
        raise ValueError("coarse plan is not source-only")
    domain_names, domain_common_valid, domain_metadata = _load_reference_safe_domain(
        reference_safe_domain_v3_path,
        expected_content_sha256=expected_reference_safe_domain_content_sha256,
    )
    if domain_metadata.get("frozen_submap_plan_content_sha256") != (
        expected_upstream_plan_content_sha256
    ):
        raise ValueError("reference-safe domain and coarse plan lineage differ")

    artifact_type = _peek_artifact_type(source_seam_authority_path)
    projective_self_reprojection: dict[str, object] | None = None
    paired_stride2_topology_audit: dict[str, object] | None = None
    if artifact_type == PROJECTIVE_AUTHORITY_SCHEMA:
        authority = ProjectiveExactFaceSeamAuthority.load_npz(
            source_seam_authority_path
        )
        authority_metadata = authority.metadata
        if authority_metadata.get("content_sha256") != (
            expected_source_seam_authority_content_sha256
        ):
            raise ValueError("projective seam authority differs from experiment pin")
        if authority_metadata.get("nominal_grid_uv_not_used_for_face_containment") is not True:
            raise ValueError("projective seam authority used nominal UV for face containment")
        if (
            authority_metadata.get("legacy_plan_phase_diagnostic_only") is not False
            or authority_metadata.get("plan_projection_contract_corrected") is not True
        ):
            raise ValueError(
                "projective seam authority is not sealed on the corrected plan phase"
            )
        if (
            PROJECTIVE_AUTHORITY_SEMANTICS_VERSION
            != PRODUCTION_AUTHORITY_SEMANTICS
            or PROJECTIVE_EDGE_FORMAL_VALID_DEFINITION
            != PRODUCTION_EDGE_FORMAL_VALID_DEFINITION
        ):
            raise ValueError("selector/projective-authority semantic constants differ")
        projective_self_reprojection = _projective_self_reprojection_audit(
            authority
        )
        authority_config = ProjectiveSeamConfig(
            **authority_metadata["config"]
        ).validated()
        topology_stride = authority_config.topology_stride
        topology_caveat = authority_metadata.get("topology_caveat")
        if topology_stride == 4:
            if (
                authority_metadata.get(
                    "projective_correspondence_production_candidate"
                )
                is not True
                or authority_metadata.get("full_submap_gate_primary_stride") != 4
                or authority_metadata.get("paired_stride2_densification_diagnostic")
                is True
            ):
                raise ValueError("stride-4 projective authority is not production-ready")
            production_authority = True
            formal_selector_handoff_eligible = True
            paired_stride2_densification_diagnostic = False
        else:
            paired_topology_content = authority_metadata.get(
                "paired_stride2_topology_content_sha256"
            )
            paired_topology_file = authority_metadata.get(
                "paired_stride2_topology_file_sha256"
            )
            if (
                authority_metadata.get("formal_selector_handoff_eligible") is not True
                or authority_metadata.get("paired_stride2_densification_diagnostic")
                is not True
                or authority_metadata.get(
                    "projective_correspondence_production_candidate"
                )
                is not False
                or authority_metadata.get("full_submap_gate_primary_stride") != 2
                or authority_metadata.get("paired_comparison_topology") is not True
                or authority_metadata.get("model_neutral_topology_claimed") is not False
                or authority_metadata.get(
                    "final_model_neutral_map_topology_eligible"
                )
                is not False
                or topology_caveat != PAIRED_STRIDE2_TOPOLOGY_CAVEAT
                or not _is_sha256(paired_topology_content)
                or not _is_sha256(paired_topology_file)
            ):
                raise ValueError("stride-2 projective formal-handoff contract differs")
            production_authority = False
            formal_selector_handoff_eligible = True
            paired_stride2_densification_diagnostic = True
            paired_stride2_topology_audit = _validate_paired_stride2_topology_parent(
                authority,
                source_seam_authority_path,
                domain_common_valid=domain_common_valid,
                expected_domain_content_sha256=(
                    expected_reference_safe_domain_content_sha256
                ),
                expected_plan_content_sha256=(
                    expected_upstream_plan_content_sha256
                ),
            )
        final_model_neutral_map_topology_eligible = bool(
            authority_metadata.get("final_model_neutral_map_topology_eligible")
            is True
        )
        semantics = PROJECTIVE_AUTHORITY_SEMANTICS_VERSION
        edge_chart_indices = authority.edge_chart_indices
        directional_support = authority.source_supported_fraction_by_direction
        direction_formal_valid = authority.direction_formal_valid
        direction_support_valid = authority.direction_support_valid
        direction_geometry_valid = authority.direction_geometry_valid
        edge_formal_valid = authority.edge_formal_valid
        edge_formal_valid_definition = PROJECTIVE_EDGE_FORMAL_VALID_DEFINITION
        graph_summary = {
            "any_correspondence": _edge_mask_graph_summary(
                authority.chart_names,
                authority.edge_chart_indices,
                (authority.source_count_by_direction > 0).any(axis=1),
            ),
            "bidirectional_correspondence": _edge_mask_graph_summary(
                authority.chart_names,
                authority.edge_chart_indices,
                (authority.source_count_by_direction > 0).all(axis=1),
            ),
            "support_valid_both_directions": _edge_mask_graph_summary(
                authority.chart_names,
                authority.edge_chart_indices,
                authority.direction_support_valid.all(axis=1),
            ),
            "geometry_valid_both_directions": _edge_mask_graph_summary(
                authority.chart_names,
                authority.edge_chart_indices,
                authority.direction_geometry_valid.all(axis=1),
            ),
            "formal_valid_both_directions": _edge_mask_graph_summary(
                authority.chart_names,
                authority.edge_chart_indices,
                authority.edge_formal_valid,
            ),
        }
        if topology_stride == 2:
            claimed_graph_summary = authority_metadata.get("stride2_graph_summary")
            if not isinstance(claimed_graph_summary, dict) or any(
                claimed_graph_summary.get(name) != value
                for name, value in graph_summary.items()
            ):
                raise ValueError("stride-2 authority graph summary does not replay")
            if (
                authority_metadata.get(
                    "stride2_formal_component_target_minimum_chart_count"
                )
                != 16
                or authority_metadata.get("stride2_formal_component_target_pass")
                is not True
                or graph_summary["formal_valid_both_directions"][
                    "largest_component_chart_count"
                ]
                < 16
            ):
                raise ValueError("stride-2 authority does not meet its formal handoff floor")
        sealed_mask_hash = authority_metadata.get("edge_formal_valid_sha256")
        m0_per_edge = None
        m0_hash = arrays_sha256(
            {
                "source_supported_fraction_by_direction": directional_support,
                "direction_support_valid": authority.direction_support_valid,
                "direction_geometry_valid": authority.direction_geometry_valid,
                "direction_formal_valid": direction_formal_valid,
                "edge_formal_valid": edge_formal_valid,
            }
        )
        m0_summary: dict[str, object] = {
            "frozen_edge_count": int(len(edge_formal_valid)),
            "direction_support_valid_count": int(direction_support_valid.sum()),
            "direction_geometry_valid_count": int(direction_geometry_valid.sum()),
            "direction_formal_valid_count": int(direction_formal_valid.sum()),
            "edge_formal_valid_count": int(edge_formal_valid.sum()),
            "edge_formal_invalid_count": int(len(edge_formal_valid) - edge_formal_valid.sum()),
            "no_bidirectional_pooling": True,
            "topology_stride": topology_stride,
            "paired_stride2_densification_diagnostic": (
                paired_stride2_densification_diagnostic
            ),
            "graph_summary": graph_summary,
            "paired_stride2_topology_replay": paired_stride2_topology_audit,
            "selected_chart_self_reprojection": projective_self_reprojection,
        }
    elif artifact_type == "goal_maplet_source_seam_correspondence_authority_v1":
        authority = SourceSeamCorrespondenceAuthority.load_npz(
            source_seam_authority_path
        )
        authority_metadata = authority.metadata
        if authority_metadata.get("content_sha256") != (
            expected_source_seam_authority_content_sha256
        ):
            raise ValueError("source-seam authority differs from experiment pin")
        m0 = evaluate_source_seam_geometry(
            authority, {}
        )["m0_source_reference"]
        semantics = authority_metadata.get("authority_semantics_version")
        is_current_legacy = (
            semantics in (None, LEGACY_AUTHORITY_SEMANTICS)
            and authority_metadata.get("association_source")
            == "isolated_source_MASt3R_reference_on_v3_exact_stride4_topology"
            and authority_metadata.get("continuous_surface_correspondence") is True
            and authority_metadata.get("target_material_location")
            == "frozen_global_face_id_plus_barycentric_weights"
        )
        if not is_current_legacy:
            raise ValueError("source-seam authority semantics are absent or unsupported")
        if not config.validated().diagnostic_allow_legacy_closest_surface:
            raise ValueError(
                "closest-world-triangle seam authority is diagnostic-only; "
                "explicitly enable the legacy dry-run flag"
            )
        semantics = LEGACY_AUTHORITY_SEMANTICS
        edge_formal_valid = np.asarray(
            [
                row["m0_reachability_pass"] is True
                and row["geometry_pass"] is True
                for row in m0["per_edge"]
            ],
            bool,
        )
        direction_formal_valid = np.repeat(
            edge_formal_valid[:, None], 2, axis=1
        )
        direction_support_valid = None
        direction_geometry_valid = None
        edge_formal_valid_definition = LEGACY_EDGE_FORMAL_VALID_DEFINITION
        sealed_mask_hash = canonical_json_sha256(
            edge_formal_valid.astype(bool).tolist()
        )
        production_authority = False
        formal_selector_handoff_eligible = False
        topology_stride = 4
        paired_stride2_densification_diagnostic = False
        topology_caveat = None
        final_model_neutral_map_topology_eligible = False
        edge_chart_indices = authority.edge_chart_indices
        directional_support = authority.source_supported_fraction_by_direction
        m0_per_edge = m0["per_edge"]
        m0_hash = canonical_json_sha256(m0)
        m0_summary = {
            "frozen_edge_count": m0["frozen_edge_count"],
            "m0_reachable_edge_count": m0["m0_reachable_edge_count"],
            "reachable_geometry_pass_count": m0[
                "reachable_geometry_pass_count"
            ],
        }
    else:
        raise ValueError("unsupported source-seam authority schema")

    if authority.chart_names.astype(str).tolist() != domain_names.tolist():
        raise ValueError("source-seam authority and v3 domain chart order differ")
    if artifact_type == PROJECTIVE_AUTHORITY_SCHEMA and not np.array_equal(
        np.asarray(authority.common_valid, bool), domain_common_valid
    ):
        raise ValueError(
            "projective authority common-valid mask differs from its v3 parent"
        )
    if authority_metadata.get("comparison_domain_v3_content_sha256") != (
        expected_reference_safe_domain_content_sha256
    ):
        raise ValueError("source-seam authority and v3 domain lineage differ")
    if (
        artifact_type == PROJECTIVE_AUTHORITY_SCHEMA
        and authority_metadata.get("comparison_domain_v3_file_sha256")
        != file_sha256(reference_safe_domain_v3_path)
    ):
        raise ValueError("projective authority and v3 parent file lineage differ")
    if authority_metadata.get("frozen_submap_plan_content_sha256") != (
        expected_upstream_plan_content_sha256
    ):
        raise ValueError("source-seam authority and coarse plan lineage differ")
    if (
        authority_metadata.get("uses_query_or_ground_truth") is not False
        or authority_metadata.get("held_geometry_consumed") is not False
    ):
        raise ValueError("source-seam authority is not source-only M0 geometry")
    if (
        artifact_type == PROJECTIVE_AUTHORITY_SCHEMA
        and authority_metadata.get("aligned_arm_geometry_consumed") is not False
    ):
        raise ValueError("projective authority consumed aligned-arm geometry")
    if (
        artifact_type != PROJECTIVE_AUTHORITY_SCHEMA
        and authority_metadata.get("aligned_arm_geometry_consumed") is True
    ):
        raise ValueError("legacy seam authority consumed aligned-arm geometry")
    lineage = {
        "upstream_coarse_plan_file_sha256": file_sha256(upstream_plan_path),
        "upstream_coarse_plan_content_sha256": (
            expected_upstream_plan_content_sha256
        ),
        "reference_safe_v3_domain_file_sha256": file_sha256(
            reference_safe_domain_v3_path
        ),
        "reference_safe_v3_domain_content_sha256": (
            expected_reference_safe_domain_content_sha256
        ),
        "source_seam_authority_file_sha256": file_sha256(
            source_seam_authority_path
        ),
        "source_seam_authority_content_sha256": (
            expected_source_seam_authority_content_sha256
        ),
        "source_seam_m0_metrics_sha256": m0_hash,
        "source_seam_authority_semantics_version": semantics,
        "edge_formal_valid_sha256": sealed_mask_hash,
        "source_seam_authority_production_eligible": production_authority,
        "source_seam_authority_formal_selector_handoff_eligible": (
            formal_selector_handoff_eligible
        ),
        "projective_topology_stride": topology_stride,
        "paired_stride2_densification_diagnostic": (
            paired_stride2_densification_diagnostic
        ),
        "paired_stride2_topology_content_sha256": authority_metadata.get(
            "paired_stride2_topology_content_sha256"
        ),
        "paired_stride2_topology_file_sha256": authority_metadata.get(
            "paired_stride2_topology_file_sha256"
        ),
        "paired_stride2_topology_arrays_sha256": authority_metadata.get(
            "paired_stride2_topology_arrays_sha256"
        ),
        "paired_stride2_topology_filename": authority_metadata.get(
            "paired_stride2_topology_filename"
        ),
        "topology_caveat": topology_caveat,
        "source_tree_sha256": authority_metadata.get("source_tree_sha256"),
        "disjoint_authority_schema": upstream_plan.metadata.get("lineage", {}).get(
            "disjoint_authority_schema"
        ),
    }
    try:
        result = select_physical_seam_chart_submap_plan(
            upstream_plan,
            domain_chart_names=domain_names,
            authority_edge_chart_indices=edge_chart_indices,
            source_supported_fraction_by_direction=directional_support,
            direction_formal_valid=direction_formal_valid,
            direction_support_valid=direction_support_valid,
            direction_geometry_valid=direction_geometry_valid,
            m0_per_edge=m0_per_edge,
            edge_formal_valid=edge_formal_valid,
            edge_formal_valid_definition=edge_formal_valid_definition,
            authority_semantics_version=str(semantics),
            production_authority=production_authority,
            formal_selector_handoff_eligible=(
                formal_selector_handoff_eligible
            ),
            topology_stride=topology_stride,
            paired_stride2_densification_diagnostic=(
                paired_stride2_densification_diagnostic
            ),
            topology_caveat=(
                str(topology_caveat) if topology_caveat is not None else None
            ),
            final_model_neutral_map_topology_eligible=(
                final_model_neutral_map_topology_eligible
            ),
            config=config,
            lineage=lineage,
        )
        audit = dict(result.audit)
        output_plan = result.plan
    except PhysicalSeamSelectionFailure as error:
        audit = dict(error.audit)
        output_plan = None
    audit.update(lineage)
    audit["upstream_plan_selected_chart_names_sha256"] = upstream_plan.metadata.get(
        "selected_chart_names_in_order_sha256"
    )
    audit["source_seam_m0_metrics"] = m0_summary
    if paired_stride2_topology_audit is not None:
        audit["paired_stride2_topology_replay"] = paired_stride2_topology_audit
    if projective_self_reprojection is not None:
        audit["projective_authority_semantics"] = {
            "principal_point_semantics": authority_metadata.get(
                "principal_point_semantics"
            ),
            "target_face_screen_domain": authority_metadata.get(
                "target_face_screen_domain"
            ),
            "target_3d_interpolation": authority_metadata.get(
                "target_3d_interpolation"
            ),
            "visibility_rule": authority_metadata.get("visibility_rule"),
            "nominal_grid_uv_not_used_for_face_containment": True,
            "topology_stride": topology_stride,
            "comparison_domain_v3_role": (
                "common_valid_and_lineage_parent_only"
                if topology_stride == 2
                else "common_valid_and_exact_stride4_topology_parent"
            ),
            "projective_correspondence_production_candidate": (
                authority_metadata.get(
                    "projective_correspondence_production_candidate"
                )
            ),
            "formal_selector_handoff_eligible": (
                formal_selector_handoff_eligible
            ),
            "paired_stride2_densification_diagnostic": (
                paired_stride2_densification_diagnostic
            ),
            "final_model_neutral_map_topology_eligible": (
                final_model_neutral_map_topology_eligible
            ),
            "topology_caveat": authority_metadata.get("topology_caveat"),
        }
    return PhysicalSeamSelectionResult(plan=output_plan, audit=audit)


__all__ = [
    "AUDIT_SCHEMA",
    "LEGACY_AUTHORITY_SEMANTICS",
    "LEGACY_EDGE_FORMAL_VALID_DEFINITION",
    "PAIRED_STRIDE2_TOPOLOGY_CAVEAT",
    "PhysicalSeamSelectionConfig",
    "PhysicalSeamSelectionFailure",
    "PhysicalSeamSelectionResult",
    "PRODUCTION_AUTHORITY_SEMANTICS",
    "PRODUCTION_EDGE_FORMAL_VALID_DEFINITION",
    "build_physical_seam_chart_submap_plan",
    "select_physical_seam_chart_submap_plan",
]
