"""Build strict, source-bounded inputs for the full-submap geometry gate.

The builder has an intentional two-phase information barrier:

1. replay only the isolated mapping-source authority, sealed comparison
   topology, frozen plan, and source-only initializer/reference geometry;
2. freeze the AABB and only then open the physically isolated held root.

The held MASt3R point maps are a source-disjoint mapping diagnostic, not
sensor depth ground truth.  No Cambridge query image, query pose, or GT is an
input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .chart_comparison_domain import source_tree_sha256
from .chart_submap_selection import ChartSubmapPlan
from .full_submap_chart_geometry_gate import (
    BOUNDARY_SEMANTICS,
    FullSubmapGeometryGateConfig,
    SOURCE_SURFACE_SCHEMA,
    SourceOnlyBoundedSurfaceBaseline,
    StrictHeldRayInventory,
    _ComparisonDomain,
    bounded_submap_content_sha256,
)
from .lineage import canonical_json_sha256, file_sha256


SOURCE_BOUND_SCHEMA = "goal_maplet_source_only_submap_bound_derivation_v1"
DENSE_V4_FROZEN_CONFIG = {
    "topology_stride": 4,
    "bound_margin_m": 1.0,
    "held_confidence_threshold": 0.25,
    "temporal_block_size": 3,
    "output_height": 144,
    "output_width": 256,
    "source_canvas_height": 288,
    "source_canvas_width": 512,
    "normal_minimum_edge_m": 0.50,
    "normal_relative_edge": 0.05,
}


@dataclass(frozen=True)
class StrictGateInputBuildConfig:
    topology_stride: int = 4
    bound_margin_m: float = 1.0
    held_confidence_threshold: float = 0.25
    temporal_block_size: int = 3
    output_height: int = 144
    output_width: int = 256
    source_canvas_height: int = 288
    source_canvas_width: int = 512
    normal_minimum_edge_m: float = 0.50
    normal_relative_edge: float = 0.05
    enforce_dense_v4_frozen_contract: bool = True

    def frozen_payload(self) -> dict[str, object]:
        return {
            key: getattr(self, key)
            for key in DENSE_V4_FROZEN_CONFIG
        }

    def validated(self) -> "StrictGateInputBuildConfig":
        if self.topology_stride not in (4, 8):
            raise ValueError("strict gate supports only sealed stride 4 or 8")
        if self.bound_margin_m <= 0:
            raise ValueError("source-only bound margin must be positive")
        if self.held_confidence_threshold < 0:
            raise ValueError("held confidence threshold must be nonnegative")
        if self.temporal_block_size < 1:
            raise ValueError("temporal block size must be positive")
        if (
            self.source_canvas_height != 2 * self.output_height
            or self.source_canvas_width != 2 * self.output_width
        ):
            raise ValueError("strict MASt3R inventory requires exact 2x area downsampling")
        if self.normal_minimum_edge_m <= 0 or self.normal_relative_edge <= 0:
            raise ValueError("normal continuity thresholds must be positive")
        if self.enforce_dense_v4_frozen_contract and self.frozen_payload() != (
            DENSE_V4_FROZEN_CONFIG
        ):
            raise ValueError(
                "dense-v4 held gate parameters are pre-frozen; refusing post-hoc change"
            )
        return self


@dataclass(frozen=True)
class _FrozenSourcePhase:
    authority: dict[str, object]
    comparison_domain: _ComparisonDomain
    plan: ChartSubmapPlan
    source_cameras: dict[str, object]
    selected_names: tuple[str, ...]
    source_surface: SourceOnlyBoundedSurfaceBaseline
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    bounds_hash: str
    source_pointmap_inventory_sha256: str
    dav2_manifest_file_sha256: str
    dav2_manifest_content_sha256: str
    moge_manifest_file_sha256: str
    moge_manifest_content_sha256: str


def _sealed_json(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    claimed = payload.get("content_sha256")
    replay = dict(payload)
    replay.pop("content_sha256", None)
    if claimed != canonical_json_sha256(replay):
        raise ValueError(f"{label} content hash does not replay")
    return payload


def _pointmap_path(root: Path, name: str) -> Path:
    return Path(root) / "pointmaps" / f"{Path(name).stem}.json"


def _pointmap_inventory_sha256(root: Path, ordered_names: list[str]) -> str:
    rows = []
    for name in ordered_names:
        path = _pointmap_path(root, name)
        if not path.is_file():
            raise ValueError(f"missing certified point map: {name}")
        rows.append({"name": name, "file_sha256": file_sha256(path)})
    return canonical_json_sha256(rows)


def _camera_inventory(
    path: Path,
    expected_names: list[str],
) -> tuple[dict[str, object], dict[str, int]]:
    cameras = json.loads(Path(path).read_text())
    names = [Path(value).name for value in cameras.get("filepaths", [])]
    if names != expected_names or len(names) != len(set(names)):
        raise ValueError("camera inventory is not exact authority order")
    c2w = np.asarray(cameras.get("cams2world"), np.float64)
    focal = np.asarray(cameras.get("focals"), np.float64)
    if c2w.shape != (len(names), 4, 4) or focal.shape != (len(names),):
        raise ValueError("camera arrays have the wrong shape")
    if not np.isfinite(c2w).all() or not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError("camera inventory contains invalid calibration")
    return cameras, {name: row for row, name in enumerate(names)}


def _area_downsample_2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    height, width = value.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("area downsampling requires an even source grid")
    trailing = value.shape[2:]
    reshaped = value.reshape(height // 2, 2, width // 2, 2, *trailing)
    return reshaped.mean(axis=(1, 3))


def _load_pointmap_world(
    path: Path,
    *,
    source_height: int,
    source_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    points = np.asarray(payload.get("points"), np.float64)
    confidence = np.asarray(payload.get("confs"), np.float64)
    if points.shape == (source_height * source_width, 3):
        points = points.reshape(source_height, source_width, 3)
    if points.shape != (source_height, source_width, 3):
        raise ValueError(f"point map has the wrong shape: {path.name}")
    if confidence.shape != (source_height, source_width):
        raise ValueError(f"point confidence has the wrong shape: {path.name}")
    return _area_downsample_2(points), _area_downsample_2(confidence)


def _manifest_rows(
    root: Path,
    *,
    expected_schema: str,
    expected_names: list[str],
    label: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    manifest_path = Path(root) / "manifest.json"
    manifest = _sealed_json(manifest_path, f"{label} manifest")
    if manifest.get("artifact_type") != expected_schema:
        raise ValueError(f"wrong {label} manifest schema")
    if manifest.get("uses_query_or_ground_truth") is not False:
        raise ValueError(f"{label} manifest consumed query/GT")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} manifest lacks rows")
    by_name: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} manifest row is invalid")
        name = str(row.get("name"))
        if name in by_name:
            raise ValueError(f"duplicate {label} manifest row")
        by_name[name] = row
    if list(by_name) != expected_names:
        raise ValueError(f"{label} manifest is not exact source authority order")
    for name, row in by_name.items():
        path = Path(root) / f"{name}.npz"
        if not path.is_file() or row.get("file_sha256") != file_sha256(path):
            raise ValueError(f"{label} initializer bytes differ for {name}")
    return manifest, by_name


def _initializer_points(
    *,
    root: Path,
    name: str,
    arm: str,
    camera_to_world: np.ndarray,
    expected_content_sha256: object,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    path = Path(root) / f"{name}.npz"
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        valid = np.asarray(data["valid"], bool)
        if arm == "DAV2":
            points = np.asarray(data["points_world"], np.float64)
        elif arm == "MoGe3":
            camera_points = np.asarray(data["points_camera"], np.float64)
            points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        else:
            raise ValueError("unsupported initializer arm")
    claimed = metadata.get("content_sha256")
    replay = dict(metadata)
    replay.pop("content_sha256", None)
    if claimed != canonical_json_sha256(replay) or claimed != expected_content_sha256:
        raise ValueError(f"{arm} initializer content differs for {name}")
    expected_schema = (
        "goal_maplet_dav2_chart_initializer_v1"
        if arm == "DAV2"
        else "goal_maplet_moge3_chart_initializer_v2"
    )
    if (
        metadata.get("artifact_type") != expected_schema
        or metadata.get("source_name") != name
        or metadata.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError(f"{arm} initializer lineage differs for {name}")
    if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
        raise ValueError(f"{arm} initializer geometry has the wrong shape")
    if points.shape[:2] != (expected_height, expected_width):
        raise ValueError(f"{arm} initializer is outside the frozen pixel domain")
    return np.where(valid[..., None], points, np.nan)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle = vertices[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    length = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-10):
        raise ValueError("source reference surface contains a collapsed common face")
    normal = np.zeros_like(vertices, np.float64)
    for corner in range(3):
        np.add.at(normal, faces[:, corner], cross)
    normal_length = np.linalg.norm(normal, axis=1)
    if np.any(normal_length <= 1e-10):
        raise ValueError("source reference surface contains an unoriented vertex")
    return normal / normal_length[:, None]


def _freeze_source_phase(
    *,
    authority_path: Path,
    comparison_domain_path: Path,
    frozen_submap_plan_path: Path,
    dav2_initializers: Path,
    moge3_initializers: Path,
    config: StrictGateInputBuildConfig,
) -> _FrozenSourcePhase:
    authority_path = Path(authority_path)
    comparison_domain_path = Path(comparison_domain_path)
    frozen_submap_plan_path = Path(frozen_submap_plan_path)
    authority = _sealed_json(authority_path, "disjoint upstream authority")
    if authority.get("artifact_type") != "goal_maplet_disjoint_chart_upstream_authority_v2":
        raise ValueError("strict gate inputs require physically isolated v2 authority")
    required_true = (
        "source_held_image_disjoint",
        "source_held_route_disjoint",
        "physical_source_held_input_roots_disjoint",
        "strict_disjoint_upstream",
    )
    if any(authority.get(key) is not True for key in required_true):
        raise ValueError("authority does not certify strict source/held isolation")
    if (
        authority.get("uses_query_or_ground_truth") is not False
        or authority.get("forbidden_routes_opened") is not False
    ):
        raise ValueError("authority consumed forbidden query/GT inputs")
    source = authority.get("source")
    held = authority.get("held")
    if not isinstance(source, dict) or not isinstance(held, dict):
        raise ValueError("authority lacks source/held records")
    source_names = [str(name) for name in source.get("ordered_names", [])]
    if len(source_names) < 2 or len(source_names) != len(set(source_names)):
        raise ValueError("authority source inventory is invalid")
    source_root = Path(str(source.get("root"))).resolve()
    if source_tree_sha256(source_root) != source.get("tree_sha256"):
        raise ValueError("isolated source tree differs from authority")
    source_cameras_path = source_root / "cameras.json"
    if file_sha256(source_cameras_path) != source.get("cameras_file_sha256"):
        raise ValueError("source cameras differ from authority")
    source_cameras, camera_rows = _camera_inventory(source_cameras_path, source_names)
    source_pointmap_hash = _pointmap_inventory_sha256(source_root, source_names)
    if source_pointmap_hash != source.get("pointmap_inventory_sha256"):
        raise ValueError("source point-map inventory differs from authority")

    domain = _ComparisonDomain.load_npz(comparison_domain_path)
    plan = ChartSubmapPlan.load_npz(frozen_submap_plan_path)
    selected_names = tuple(str(name) for name in plan.selected_chart_names_in_order)
    if domain.chart_names.astype(str).tolist() != list(selected_names):
        raise ValueError("sealed domain is not exact frozen selection order")
    authority_content = str(authority["content_sha256"])
    plan_content = str(plan.metadata.get("content_sha256"))
    source_names_hash = canonical_json_sha256(source_names)
    expected_domain = {
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority_content,
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": plan_content,
        "mapping_source_ordered_names_sha256": source_names_hash,
        "source_tree_sha256": source.get("tree_sha256"),
    }
    for key, expected in expected_domain.items():
        if domain.metadata.get(key) != expected:
            raise ValueError(f"sealed comparison domain {key} differs")
    plan_lineage = plan.metadata.get("lineage")
    if (
        plan.metadata.get("comparison_inventory_eligible") is not True
        or plan.metadata.get("system_control_only") is not False
        or not isinstance(plan_lineage, dict)
        or plan_lineage.get("query_or_ground_truth_consumed") is not False
        or plan_lineage.get("held_root_opened_by_selector") is not False
        or plan_lineage.get("disjoint_authority_content_sha256") != authority_content
        or plan_lineage.get("source_ordered_names_sha256") != source_names_hash
    ):
        raise ValueError("frozen plan is not strict source-only comparison authority")

    dav2_manifest, dav2_rows = _manifest_rows(
        dav2_initializers,
        expected_schema="goal_maplet_dav2_chart_initializer_run_v1",
        expected_names=source_names,
        label="DAV2",
    )
    moge_manifest, moge_rows = _manifest_rows(
        moge3_initializers,
        expected_schema="goal_maplet_moge3_chart_initializer_run_v2",
        expected_names=source_names,
        label="MoGe3",
    )
    if (
        dav2_manifest.get("disjoint_upstream_authority_content_sha256")
        != authority_content
        or dav2_manifest.get("source_only_mast3r_tree_sha256")
        != source.get("tree_sha256")
        or moge_manifest.get("cameras_file_sha256") != file_sha256(source_cameras_path)
        or moge_manifest.get("uses_camera_pose") is not False
    ):
        raise ValueError("initializer manifests do not bind the isolated source authority")
    initializer_domain_bindings = {
        "dav2_initializer_manifest_file_sha256": file_sha256(
            Path(dav2_initializers) / "manifest.json"
        ),
        "dav2_initializer_manifest_content_sha256": dav2_manifest[
            "content_sha256"
        ],
        "moge_initializer_manifest_file_sha256": file_sha256(
            Path(moge3_initializers) / "manifest.json"
        ),
        "moge_initializer_manifest_content_sha256": moge_manifest[
            "content_sha256"
        ],
    }
    for key, expected in initializer_domain_bindings.items():
        if domain.metadata.get(key) != expected:
            raise ValueError(f"sealed comparison domain {key} differs from initializer")
    domain_dav2_files = domain.metadata.get("dav2_initializer_file_sha256")
    domain_moge_files = domain.metadata.get("moge_initializer_file_sha256")
    if not isinstance(domain_dav2_files, dict) or not isinstance(domain_moge_files, dict):
        raise ValueError("sealed comparison domain lacks selected initializer file inventory")
    for name in selected_names:
        if domain_dav2_files.get(name) != file_sha256(
            Path(dav2_initializers) / f"{name}.npz"
        ):
            raise ValueError(f"sealed comparison domain DAV2 file differs for {name}")
        if domain_moge_files.get(name) != file_sha256(
            Path(moge3_initializers) / f"{name}.npz"
        ):
            raise ValueError(f"sealed comparison domain MoGe3 file differs for {name}")

    stride = config.topology_stride
    vertex_offsets = np.asarray(
        domain.exact_topology[f"sampled_vertex_offsets_stride{stride}"], np.int64
    )
    pixel_indices = np.asarray(
        domain.exact_topology[f"sampled_vertex_pixel_indices_stride{stride}"], np.int64
    )
    face_offsets = np.asarray(
        domain.exact_topology[f"face_offsets_stride{stride}"], np.int64
    )
    faces = np.asarray(domain.exact_topology[f"faces_stride{stride}"], np.int64)
    source_vertices: list[np.ndarray] = []
    bound_components: list[np.ndarray] = []
    component_counts: dict[str, int] = {"source_MASt3R": 0, "DAV2": 0, "MoGe3": 0}
    for chart, name in enumerate(selected_names):
        row = camera_rows[name]
        camera_to_world = np.asarray(source_cameras["cams2world"][row], np.float64)
        reference_world, _ = _load_pointmap_world(
            _pointmap_path(source_root, name),
            source_height=config.source_canvas_height,
            source_width=config.source_canvas_width,
        )
        dav2_world = _initializer_points(
            root=dav2_initializers,
            name=name,
            arm="DAV2",
            camera_to_world=camera_to_world,
            expected_content_sha256=dav2_rows[name].get("content_sha256"),
            expected_height=config.output_height,
            expected_width=config.output_width,
        )
        moge_world = _initializer_points(
            root=moge3_initializers,
            name=name,
            arm="MoGe3",
            camera_to_world=camera_to_world,
            expected_content_sha256=moge_rows[name].get("content_sha256"),
            expected_height=config.output_height,
            expected_width=config.output_width,
        )
        lo, hi = map(int, vertex_offsets[chart : chart + 2])
        indices = pixel_indices[lo:hi]
        reference_selected = reference_world.reshape(-1, 3)[indices]
        dav2_selected = dav2_world.reshape(-1, 3)[indices]
        moge_selected = moge_world.reshape(-1, 3)[indices]
        for label, selected in (
            ("source_MASt3R", reference_selected),
            ("DAV2", dav2_selected),
            ("MoGe3", moge_selected),
        ):
            if not np.isfinite(selected).all():
                raise ValueError(f"{label} is nonfinite on the sealed common topology")
            bound_components.append(selected)
            component_counts[label] += len(selected)
        source_vertices.append(reference_selected)
    vertices = np.concatenate(source_vertices)
    bounds_points = np.concatenate(bound_components)
    bounds_min = bounds_points.min(axis=0) - config.bound_margin_m
    bounds_max = bounds_points.max(axis=0) + config.bound_margin_m
    bounds_hash = bounded_submap_content_sha256(bounds_min, bounds_max)
    normals = _vertex_normals(vertices, faces)
    baseline_metadata: dict[str, object] = {
        "artifact_type": SOURCE_SURFACE_SCHEMA,
        "representation": "source_only_MASt3R_reference_surface_on_sealed_common_topology",
        "comparison_role": "equal_budget_main_table_source_reference_surface_control",
        "comparison_budget": "exact_frozen_source_ordered_pool",
        "is_2DGS": False,
        "claim_supported": "chart_atlas_vs_source_reference_surface_control_only",
        "claim_not_supported": "chart_atlas_vs_2DGS",
        "uses_query_pose_or_ground_truth": False,
        "held_mapping_images_consumed": False,
        "outside_frozen_source_mapping_images_consumed": False,
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority_content,
        "mapping_source_ordered_names_sha256": source_names_hash,
        "source_pointmap_inventory_sha256": source_pointmap_hash,
        "comparison_domain_file_sha256": file_sha256(comparison_domain_path),
        "comparison_domain_content_sha256": domain.metadata["content_sha256"],
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "frozen_submap_plan_content_sha256": plan_content,
        "bounded_submap_content_sha256": bounds_hash,
        "bounded_submap_min_world": bounds_min.tolist(),
        "bounded_submap_max_world": bounds_max.tolist(),
        "coordinate_cameras_file_sha256": authority["posed_colmap_cameras_file_sha256"],
        "coordinate_images_file_sha256": authority["posed_colmap_images_file_sha256"],
        "paired_common_face_inventory": True,
        "stride": stride,
        "source_bounds_schema": SOURCE_BOUND_SCHEMA,
        "source_bound_margin_m": config.bound_margin_m,
        "source_bound_component_vertex_counts": component_counts,
        "source_bound_uses_exact_face_referenced_vertices_only": True,
        "source_bound_uses_held_geometry": False,
        "dav2_initializer_manifest_file_sha256": file_sha256(
            Path(dav2_initializers) / "manifest.json"
        ),
        "dav2_initializer_manifest_content_sha256": dav2_manifest["content_sha256"],
        "moge3_initializer_manifest_file_sha256": file_sha256(
            Path(moge3_initializers) / "manifest.json"
        ),
        "moge3_initializer_manifest_content_sha256": moge_manifest["content_sha256"],
    }
    source_surface = SourceOnlyBoundedSurfaceBaseline(
        chart_names=np.asarray(selected_names),
        chart_vertex_offsets=vertex_offsets,
        vertices_world=vertices,
        normals_world=normals,
        chart_face_offsets=face_offsets,
        faces=faces,
        metadata=baseline_metadata,
    ).validated()
    return _FrozenSourcePhase(
        authority=authority,
        comparison_domain=domain,
        plan=plan,
        source_cameras=source_cameras,
        selected_names=selected_names,
        source_surface=source_surface,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        bounds_hash=bounds_hash,
        source_pointmap_inventory_sha256=source_pointmap_hash,
        dav2_manifest_file_sha256=baseline_metadata[
            "dav2_initializer_manifest_file_sha256"
        ],
        dav2_manifest_content_sha256=str(dav2_manifest["content_sha256"]),
        moge_manifest_file_sha256=baseline_metadata[
            "moge3_initializer_manifest_file_sha256"
        ],
        moge_manifest_content_sha256=str(moge_manifest["content_sha256"]),
    )


def _held_geometry(
    *,
    pointmap_path: Path,
    camera_to_world: np.ndarray,
    focal_canvas: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    config: StrictGateInputBuildConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    world_grid, confidence = _load_pointmap_world(
        pointmap_path,
        source_height=config.source_canvas_height,
        source_width=config.source_canvas_width,
    )
    depth = ((world_grid - camera_to_world[:3, 3]) @ camera_to_world[:3, :3])[
        ..., 2
    ]
    height, width = config.output_height, config.output_width
    focal = float(focal_canvas) * width / config.source_canvas_width
    principal = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0], np.float64)
    yy, xx = np.mgrid[:height, :width]
    direction_camera = np.stack(
        ((xx - principal[0]) / focal, (yy - principal[1]) / focal, np.ones_like(xx)),
        axis=2,
    )
    points_camera = direction_camera * depth[..., None]
    reconstructed_world = (
        points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    )
    base_valid = (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(confidence)
        & (confidence > config.held_confidence_threshold)
        & np.isfinite(reconstructed_world).all(2)
    )
    normal = np.zeros((height, width, 3), np.float64)
    dx = reconstructed_world[1:-1, 2:] - reconstructed_world[1:-1, :-2]
    dy = reconstructed_world[2:, 1:-1] - reconstructed_world[:-2, 1:-1]
    cross = np.cross(dx, dy)
    cross_length = np.linalg.norm(cross, axis=2)
    normal[1:-1, 1:-1] = cross / np.maximum(cross_length[..., None], 1e-15)
    stencil = np.zeros((height, width), bool)
    center = base_valid[1:-1, 1:-1]
    stencil_inner = (
        center
        & base_valid[1:-1, :-2]
        & base_valid[1:-1, 2:]
        & base_valid[:-2, 1:-1]
        & base_valid[2:, 1:-1]
        & np.isfinite(cross_length)
        & (cross_length > 1e-10)
    )
    center_world = reconstructed_world[1:-1, 1:-1]
    threshold = np.maximum(
        config.normal_minimum_edge_m,
        config.normal_relative_edge * depth[1:-1, 1:-1],
    )
    for neighbour in (
        reconstructed_world[1:-1, :-2],
        reconstructed_world[1:-1, 2:],
        reconstructed_world[:-2, 1:-1],
        reconstructed_world[2:, 1:-1],
    ):
        stencil_inner &= np.linalg.norm(neighbour - center_world, axis=2) <= threshold
    stencil[1:-1, 1:-1] = stencil_inner
    in_bounds = np.all(reconstructed_world >= bounds_min, axis=2) & np.all(
        reconstructed_world <= bounds_max, axis=2
    )
    valid = base_valid & stencil & in_bounds
    if not valid.any():
        raise ValueError(f"held view has no valid ray inside source-only bounds: {pointmap_path.name}")
    boundary_config = FullSubmapGeometryGateConfig(bootstrap_resamples=100)
    # Confidence-mask holes are a property of the mapping diagnostic, not a
    # physical surface boundary.  Keep only depth/normal discontinuities for
    # which both adjacent pixels have geometric support.
    boundary = np.zeros_like(base_valid)
    for first, second in (
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :]),
    ):
        both = base_valid[first] & base_valid[second]
        delta = np.zeros_like(both, np.float64)
        delta[both] = np.abs(depth[first][both] - depth[second][both])
        limit = np.maximum(
            boundary_config.boundary_depth_absolute_m,
            boundary_config.boundary_depth_relative
            * np.minimum(depth[first], depth[second]),
        )
        normal_first = normal[first]
        normal_second = normal[second]
        normal_pair = (
            both
            & (np.linalg.norm(normal_first, axis=2) > 0.5)
            & (np.linalg.norm(normal_second, axis=2) > 0.5)
        )
        dot = np.abs(np.sum(normal_first * normal_second, axis=2))
        edge = both & (delta > limit)
        edge |= normal_pair & (
            dot < np.cos(np.deg2rad(boundary_config.boundary_normal_angle_deg))
        )
        boundary[first] |= edge
        boundary[second] |= edge
    output_depth = np.where(valid, depth, 0.0)
    output_normal = np.where(valid[..., None], normal, 0.0)
    return output_depth, output_normal, boundary, int(valid.sum())


def build_strict_full_submap_gate_inputs(
    *,
    authority_path: Path,
    comparison_domain_path: Path,
    frozen_submap_plan_path: Path,
    dav2_initializers: Path,
    moge3_initializers: Path,
    config: StrictGateInputBuildConfig = StrictGateInputBuildConfig(),
) -> tuple[StrictHeldRayInventory, SourceOnlyBoundedSurfaceBaseline, dict[str, object]]:
    """Build held rays and equal-budget source surface without query/GT access."""

    config = config.validated()
    # Information barrier: this call does not resolve or open authority.held.root.
    frozen = _freeze_source_phase(
        authority_path=authority_path,
        comparison_domain_path=comparison_domain_path,
        frozen_submap_plan_path=frozen_submap_plan_path,
        dav2_initializers=dav2_initializers,
        moge3_initializers=moge3_initializers,
        config=config,
    )
    bounds_min = np.asarray(frozen.bounds_min, np.float64).copy()
    bounds_max = np.asarray(frozen.bounds_max, np.float64).copy()
    bounds_min.setflags(write=False)
    bounds_max.setflags(write=False)

    # Held phase starts only after the source-only bounds are immutable.
    authority = frozen.authority
    held = authority["held"]
    held_names = [str(name) for name in held.get("ordered_names", [])]
    held_root = Path(str(held.get("root"))).resolve()
    held_cameras_path = held_root / "cameras.json"
    if file_sha256(held_cameras_path) != held.get("cameras_file_sha256"):
        raise ValueError("held cameras differ from isolated authority")
    held_cameras, held_rows = _camera_inventory(held_cameras_path, held_names)
    held_pointmap_hash = _pointmap_inventory_sha256(held_root, held_names)
    if held_pointmap_hash != held.get("pointmap_inventory_sha256"):
        raise ValueError("held point-map inventory differs from authority")
    camera_to_world = np.asarray(held_cameras["cams2world"], np.float64)
    focal_canvas = np.asarray(held_cameras["focals"], np.float64)
    depths = []
    normals = []
    boundaries = []
    valid_masks = []
    valid_counts = []
    for name in held_names:
        row = held_rows[name]
        depth, normal, boundary, count = _held_geometry(
            pointmap_path=_pointmap_path(held_root, name),
            camera_to_world=camera_to_world[row],
            focal_canvas=float(focal_canvas[row]),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            config=config,
        )
        valid = depth > 0
        depths.append(depth)
        normals.append(normal)
        boundaries.append(boundary)
        valid_masks.append(valid)
        valid_counts.append(count)
    if len(held_names) // config.temporal_block_size < 2:
        raise ValueError("held inventory cannot form two temporal bootstrap blocks")
    height, width = config.output_height, config.output_width
    focal_xy = np.repeat(
        (focal_canvas * width / config.source_canvas_width)[:, None], 2, axis=1
    )
    principal_xy = np.repeat(
        np.asarray([[(width - 1) / 2.0, (height - 1) / 2.0]]),
        len(held_names),
        axis=0,
    )
    source_names = [str(name) for name in authority["source"]["ordered_names"]]
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_strict_held_ray_inventory_v1",
        "uses_query_or_ground_truth": False,
        "dense_row_major_pixel_centers": True,
        "reference_semantics": "source_disjoint_held_mapping_geometry_not_sensor_depth_gt",
        "boundary_semantics": BOUNDARY_SEMANTICS,
        "disjoint_upstream_authority_file_sha256": file_sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
        "held_pointmap_inventory_sha256": held_pointmap_hash,
        "held_cameras_file_sha256": file_sha256(held_cameras_path),
        "mapping_source_ordered_names_sha256": canonical_json_sha256(source_names),
        "source_pointmap_inventory_sha256": frozen.source_pointmap_inventory_sha256,
        "comparison_domain_content_sha256": frozen.comparison_domain.metadata[
            "content_sha256"
        ],
        "comparison_domain_file_sha256": file_sha256(comparison_domain_path),
        "frozen_submap_plan_content_sha256": frozen.plan.metadata["content_sha256"],
        "frozen_submap_plan_file_sha256": file_sha256(frozen_submap_plan_path),
        "bounded_submap_content_sha256": frozen.bounds_hash,
        "bounded_submap_min_world": bounds_min.tolist(),
        "bounded_submap_max_world": bounds_max.tolist(),
        "coordinate_cameras_file_sha256": authority[
            "posed_colmap_cameras_file_sha256"
        ],
        "coordinate_images_file_sha256": authority["posed_colmap_images_file_sha256"],
        "held_view_names_sha256": canonical_json_sha256(held_names),
        "held_inventory_exactly_authority_order": True,
        "held_geometry_opened_after_source_bounds_frozen": True,
        "held_pointmap_inventory_verified_after_source_bounds_frozen": True,
        "bound_uses_held_geometry": False,
        "builder_config_frozen_before_held": True,
        "pre_frozen_builder_config": config.frozen_payload(),
        "pre_frozen_builder_config_content_sha256": canonical_json_sha256(
            config.frozen_payload()
        ),
        "dense_v4_frozen_contract_enforced": config.enforce_dense_v4_frozen_contract,
        "source_bound_derivation_schema": SOURCE_BOUND_SCHEMA,
        "source_bound_derivation": (
            "exact sealed face-referenced source MASt3R, DAV2, and world-transformed "
            "MoGe3 vertices plus fixed metric margin; held bytes unopened"
        ),
        "source_bound_margin_m": config.bound_margin_m,
        "held_confidence_threshold": config.held_confidence_threshold,
        "normal_stencil": (
            "central_difference_four-neighbour_confident_continuous_surface"
        ),
        "normal_minimum_edge_m": config.normal_minimum_edge_m,
        "normal_relative_edge": config.normal_relative_edge,
        "principal_point_semantics": "((output_width-1)/2,(output_height-1)/2)",
        "focal_rescaling": "focal_output=focal_512*output_width/512",
        "temporal_block_size": config.temporal_block_size,
        "per_view_valid_ray_counts": valid_counts,
        "per_view_geometry_boundary_fractions": [
            float(np.mean(boundary)) for boundary in boundaries
        ],
        "confidence_mask_edges_excluded_from_boundary": True,
        "dav2_initializer_manifest_file_sha256": frozen.dav2_manifest_file_sha256,
        "dav2_initializer_manifest_content_sha256": frozen.dav2_manifest_content_sha256,
        "moge3_initializer_manifest_file_sha256": frozen.moge_manifest_file_sha256,
        "moge3_initializer_manifest_content_sha256": frozen.moge_manifest_content_sha256,
    }
    rays = StrictHeldRayInventory(
        view_names=np.asarray(held_names),
        camera_to_world=camera_to_world,
        focal_xy=focal_xy,
        principal_xy=principal_xy,
        reference_depth_m=np.stack(depths),
        reference_normal_world=np.stack(normals),
        reference_valid=np.stack(valid_masks),
        reference_boundary=np.stack(boundaries),
        block_ids=np.asarray(
            [f"temporal_block_{row // config.temporal_block_size:03d}" for row in range(len(held_names))]
        ),
        metadata=metadata,
    ).validated()
    audit = {
        "artifact_type": "goal_maplet_strict_full_submap_gate_input_build_audit_v1",
        "source_phase_completed_before_held_phase": True,
        "builder_config_frozen_before_held": True,
        "pre_frozen_builder_config": config.frozen_payload(),
        "pre_frozen_builder_config_content_sha256": canonical_json_sha256(
            config.frozen_payload()
        ),
        "dense_v4_frozen_contract_enforced": config.enforce_dense_v4_frozen_contract,
        "bound_uses_held_geometry": False,
        "bounded_submap_content_sha256": frozen.bounds_hash,
        "bounded_submap_min_world": bounds_min.tolist(),
        "bounded_submap_max_world": bounds_max.tolist(),
        "selected_chart_names": list(frozen.selected_names),
        "held_view_names": held_names,
        "held_valid_ray_counts": valid_counts,
        "held_valid_ray_count_total": int(sum(valid_counts)),
        "M0_role": frozen.source_surface.metadata["comparison_role"],
        "M0_is_2DGS": False,
        "full_train_2DGS_comparison_role": "unequal_budget_diagnostic_only",
        "scientific_claim": "chart_atlas_vs_source_reference_surface_control",
        "scientific_claim_not_yet_available": "chart_atlas_vs_source_only_2DGS",
        "uses_query_or_ground_truth": False,
    }
    audit["content_sha256"] = canonical_json_sha256(audit)
    return rays, frozen.source_surface, audit


def classify_2dgs_comparison_budget(
    metadata: Mapping[str, object],
    *,
    authority_source_names: list[str],
) -> str:
    """Fail-closed role classifier for an external 2DGS diagnostic artifact."""

    source_hash = canonical_json_sha256([str(name) for name in authority_source_names])
    exact = (
        metadata.get("mapping_source_ordered_names_sha256") == source_hash
        and metadata.get("held_mapping_images_consumed") is False
        and metadata.get("outside_frozen_source_mapping_images_consumed") is False
        and metadata.get("source_2dgs_training_inventory_exact_authority_source") is True
    )
    return (
        "equal_budget_main_table_source_only_2DGS"
        if exact
        else "unequal_budget_diagnostic_only"
    )


def bounded_submap_authority_from_artifacts(
    rays: StrictHeldRayInventory,
    surface: SourceOnlyBoundedSurfaceBaseline,
) -> dict[str, object]:
    """Materialize the source-only AABB as a replayable standalone authority."""

    rays = rays.validated()
    surface = surface.validated()
    shared = (
        "disjoint_upstream_authority_file_sha256",
        "disjoint_upstream_authority_content_sha256",
        "mapping_source_ordered_names_sha256",
        "comparison_domain_file_sha256",
        "comparison_domain_content_sha256",
        "frozen_submap_plan_file_sha256",
        "frozen_submap_plan_content_sha256",
        "coordinate_cameras_file_sha256",
        "coordinate_images_file_sha256",
        "bounded_submap_content_sha256",
    )
    for key in shared:
        if rays.metadata.get(key) != surface.metadata.get(key):
            raise ValueError(f"held rays/source surface disagree on bound authority {key}")
    minimum = list(rays.metadata["bounded_submap_min_world"])
    maximum = list(rays.metadata["bounded_submap_max_world"])
    payload: dict[str, object] = {
        "artifact_type": "goal_maplet_axis_aligned_bounded_submap_v1",
        "minimum_world": minimum,
        "maximum_world": maximum,
        "bounded_submap_content_sha256": rays.metadata[
            "bounded_submap_content_sha256"
        ],
        "disjoint_upstream_authority_file_sha256": rays.metadata[
            "disjoint_upstream_authority_file_sha256"
        ],
        "disjoint_upstream_authority_content_sha256": rays.metadata[
            "disjoint_upstream_authority_content_sha256"
        ],
        "mapping_source_ordered_names_sha256": rays.metadata[
            "mapping_source_ordered_names_sha256"
        ],
        "comparison_domain_file_sha256": rays.metadata[
            "comparison_domain_file_sha256"
        ],
        "comparison_domain_content_sha256": rays.metadata[
            "comparison_domain_content_sha256"
        ],
        "frozen_submap_plan_file_sha256": rays.metadata[
            "frozen_submap_plan_file_sha256"
        ],
        "frozen_submap_plan_content_sha256": rays.metadata[
            "frozen_submap_plan_content_sha256"
        ],
        "coordinate_cameras_file_sha256": rays.metadata[
            "coordinate_cameras_file_sha256"
        ],
        "coordinate_images_file_sha256": rays.metadata[
            "coordinate_images_file_sha256"
        ],
        "source_bound_derivation_schema": SOURCE_BOUND_SCHEMA,
        "source_bound_margin_m": rays.metadata["source_bound_margin_m"],
        "pre_frozen_builder_config_content_sha256": rays.metadata[
            "pre_frozen_builder_config_content_sha256"
        ],
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "production_eligible": False,
    }
    if payload["bounded_submap_content_sha256"] != bounded_submap_content_sha256(
        minimum, maximum
    ):
        raise ValueError("standalone bounded-submap authority does not replay its AABB")
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


__all__ = [
    "SOURCE_BOUND_SCHEMA",
    "StrictGateInputBuildConfig",
    "build_strict_full_submap_gate_inputs",
    "bounded_submap_authority_from_artifacts",
    "classify_2dgs_comparison_budget",
]
