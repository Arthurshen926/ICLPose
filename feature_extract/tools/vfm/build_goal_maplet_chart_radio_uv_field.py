"""Build a coordinate-correct RADIO field on an explicit chart atlas.

This is an offline mapping-only builder.  It first emits a complete source-
view observation artifact, then either an explicitly diagnostic layout or a
layout constrained by a separately sealed canonical surface-family carrier,
and finally a view-balanced canonical field.  None of these artifacts claims
pose accuracy or a shared fused canonical mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_radio_uv_field import (
    IdealChartCamera,
    RawSimpleRadialCamera,
    attach_radio_to_chart_atlas,
    build_carrier_constrained_surface_family_layout,
    build_diagnostic_metric_surface_families,
    chart_name_to_image_id,
    fuse_canonical_chart_radio_field,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
    DISJOINT_AUTHORITY_SCHEMA,
    SCHEMA as CHART_SUBMAP_PLAN_SCHEMA,
    load_model_neutral_alignment_selection,
)
from feature_extract.vfm.localization_goal_maplet.chart_surface_families import (
    CanonicalSurfaceFamilyCarrier,
    SCHEMA as SURFACE_FAMILY_CARRIER_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import (
    ExplicitChartAtlas,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.tools.vfm.visualize_goal_maplet_chart_radio_uv_field import (
    visualize_chart_radio_uv_field,
)


def _load_radio_records(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("RADIO token manifest lacks records")
    result: dict[str, dict[str, object]] = {}
    for row in records:
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in result:
            raise ValueError("RADIO token manifest contains empty/duplicate image IDs")
        result[image_id] = dict(row)
    return result


def _load_chart_camera_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    names = [Path(value).name for value in payload["filepaths"]]
    c2w = np.asarray(payload["cams2world"], dtype=np.float64)
    if len(names) != len(set(names)) or c2w.shape != (len(names), 4, 4):
        raise ValueError("chart cameras.json inventory differs")
    if np.any(~np.isfinite(c2w)):
        raise ValueError("chart cameras.json contains nonfinite poses")
    focal = np.asarray(payload["focals"], dtype=np.float64).reshape(-1)
    if focal.shape != (len(names),) or np.any(~np.isfinite(focal)) or np.any(focal <= 0.0):
        raise ValueError("chart cameras.json focal inventory differs")
    return {
        name: {"pose_c2w": c2w[row].copy(), "focal": float(focal[row])}
        for row, name in enumerate(names)
    }


def _load_sealed_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    claimed = str(payload.pop("content_sha256", ""))
    if len(claimed) != 64 or canonical_json_sha256(payload) != claimed:
        raise ValueError(f"JSON authority content seal differs: {path}")
    payload["content_sha256"] = claimed
    return payload


def _strict_disjoint_surface_audit(
    *,
    authority_path: Path | None,
    submap_plan_path: Path | None,
    family_carrier_path: Path | None,
    expected_plan_content_sha256: str | None,
    expected_family_carrier_content_sha256: str | None,
    chart_names: list[str],
    atlas_content_sha256: str,
) -> dict[str, object]:
    supplied = (
        authority_path is not None,
        submap_plan_path is not None,
        family_carrier_path is not None,
    )
    if not any(supplied):
        result = {
            "artifact_type": "goal_maplet_chart_radio_uv_input_composition_audit_v1",
            "strict_disjoint_authority_verified": False,
            "chart_submap_plan_verified": False,
            "surface_family_carrier_verified": False,
            "strict_promotion_eligible": False,
            "reason": "v2_authority_submap_plan_and_family_carrier_not_supplied",
        }
        result["content_sha256"] = canonical_json_sha256(result)
        return result
    if not all(supplied):
        raise ValueError(
            "strict chart RADIO run requires v2 authority, chart submap plan, and "
            "canonical surface-family carrier"
        )
    if (
        expected_plan_content_sha256 is None
        or len(expected_plan_content_sha256) != 64
        or expected_family_carrier_content_sha256 is None
        or len(expected_family_carrier_content_sha256) != 64
    ):
        raise ValueError(
            "strict chart RADIO run requires externally pinned plan and family-carrier hashes"
        )
    assert authority_path is not None
    assert submap_plan_path is not None
    assert family_carrier_path is not None
    authority = _load_sealed_json(authority_path)
    if (
        authority.get("artifact_type") != DISJOINT_AUTHORITY_SCHEMA
        or authority.get("strict_disjoint_upstream") is not True
        or authority.get("source_held_image_disjoint") is not True
        or authority.get("source_held_route_disjoint") is not True
        or authority.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError("chart RADIO builder requires a strict query-free v2 upstream authority")
    source_names = set(authority.get("source", {}).get("ordered_names", []))
    held_names = set(authority.get("held", {}).get("ordered_names", []))
    if not set(chart_names).issubset(source_names) or set(chart_names) & held_names:
        raise ValueError("chart inventory is not source-only under the v2 authority")
    plan = ChartSubmapPlan.load_npz(submap_plan_path)
    if plan.metadata.get("artifact_type") != CHART_SUBMAP_PLAN_SCHEMA:
        raise ValueError("unknown overlap-aware chart submap plan schema")
    selection = load_model_neutral_alignment_selection(
        submap_plan_path,
        expected_plan_content_sha256=expected_plan_content_sha256,
    )
    if list(selection.ordered_names) != chart_names:
        raise ValueError("chart submap plan and explicit atlas chart order differ")
    plan_lineage = plan.metadata.get("lineage")
    if not isinstance(plan_lineage, dict):
        raise ValueError("chart submap plan lacks upstream lineage")
    if str(plan_lineage.get("disjoint_authority_content_sha256", "")) != str(
        authority["content_sha256"]
    ):
        raise ValueError("chart submap plan and v2 source authority differ")
    authority_source_tree = str(authority.get("source", {}).get("tree_sha256", ""))
    if not authority_source_tree or str(plan_lineage.get("source_tree_sha256", "")) != authority_source_tree:
        raise ValueError("chart submap plan and authority source-tree seals differ")
    carrier = CanonicalSurfaceFamilyCarrier.load_npz(family_carrier_path)
    if carrier.metadata.get("artifact_type") != SURFACE_FAMILY_CARRIER_SCHEMA:
        raise ValueError("unknown canonical surface-family carrier schema")
    if str(carrier.metadata.get("source_atlas_content_sha256", "")) != str(
        atlas_content_sha256
    ):
        raise ValueError("surface-family carrier and explicit atlas differ")
    if int(carrier.metadata.get("parameterization_count", -1)) != len(chart_names):
        raise ValueError("surface-family carrier parameterization inventory differs")
    if carrier.metadata.get("content_sha256") != expected_family_carrier_content_sha256:
        raise ValueError("surface-family carrier differs from runner authority")
    family_gate = bool(carrier.metadata.get("family_canonicalization_gate_pass", False))
    result = {
        "artifact_type": "goal_maplet_chart_radio_uv_input_composition_audit_v1",
        "strict_disjoint_authority_verified": True,
        "chart_submap_plan_verified": True,
        "surface_family_carrier_verified": True,
        "strict_promotion_eligible": family_gate,
        "reason": (
            "strict_composition_verified"
            if family_gate else "canonical_surface_family_gate_failed"
        ),
        "authority_file_sha256": file_sha256(authority_path),
        "authority_content_sha256": authority["content_sha256"],
        "authority_source_tree_sha256": authority_source_tree,
        "chart_submap_plan_schema": CHART_SUBMAP_PLAN_SCHEMA,
        "chart_submap_plan_file_sha256": file_sha256(submap_plan_path),
        "chart_submap_plan_content_sha256": selection.plan_content_sha256,
        "selected_chart_names_in_order_sha256": plan.metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "alignment_runner_contract": plan.metadata["alignment_runner_contract"],
        "surface_family_carrier_schema": SURFACE_FAMILY_CARRIER_SCHEMA,
        "surface_family_carrier_file_sha256": file_sha256(family_carrier_path),
        "surface_family_carrier_content_sha256": carrier.metadata["content_sha256"],
        "surface_family_canonicalization_gate_pass": family_gate,
        "atlas_content_sha256": str(atlas_content_sha256),
        "composition_contract": (
            "exact_ordered_plan_names_equal_atlas_names; carrier_parameterization_order_"
            "bound_indirectly_by_exact_atlas_content_sha256"
        ),
    }
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def _summary(
    *,
    source,
    layout,
    canonical,
    outputs: dict[str, Path],
    input_hashes: dict[str, str],
    strict_input_audit: dict[str, object],
    runtime_seconds: float,
    peak_rss_bytes: int,
) -> dict[str, object]:
    view_count = int(source.view_names.size)
    complete_tokens = int(source.token_codes.shape[0])
    expected_tokens = int(np.sum(np.prod(source.view_token_shapes, axis=1)))
    node_views = np.asarray(canonical.view_count, dtype=np.int64)
    same_node_cosine = []
    for node in np.flatnonzero(np.diff(canonical.prototype_offsets) >= 2).tolist():
        begin = int(canonical.prototype_offsets[node])
        same_node_cosine.append(
            float(np.dot(canonical.prototype_codes[begin], canonical.prototype_codes[begin + 1]))
        )
    observed = np.flatnonzero(node_views >= 1)
    rng = np.random.default_rng(20260830)
    random_cosine = []
    if observed.size >= 2:
        pair = rng.choice(observed, size=(max(1024, len(same_node_cosine)), 2), replace=True)
        pair = pair[pair[:, 0] != pair[:, 1]]
        random_cosine = np.sum(
            canonical.codes[pair[:, 0]] * canonical.codes[pair[:, 1]], axis=1,
        ).astype(np.float64).tolist()
    payload = {
        "artifact_type": "goal_maplet_chart_radio_uv_field_build_summary_v1",
        "claim_scope": "mapping_only_coordinate_and_fusion_smoke_not_pose_validation",
        "strict_input_audit": strict_input_audit,
        "runtime_seconds": float(runtime_seconds),
        "peak_process_rss_bytes": int(peak_rss_bytes),
        "source_view_role": "offline_observation_inventory",
        "canonical_role": "physical_surface_family_nodes_not_reference_image_ids",
        "view_count": view_count,
        "feature_dim": int(source.feature_dim),
        "complete_token_count": complete_tokens,
        "source_token_feature_storage_dtype": str(source.token_codes.dtype),
        "source_token_feature_uncompressed_bytes": int(source.token_codes.nbytes),
        "source_token_feature_fp16_lower_bound_bytes": int(source.token_codes.size * 2),
        "complete_token_inventory_verified": complete_tokens == expected_tokens,
        "coordinate_valid_vertex_fraction": float(np.mean(source.vertex_coordinate_valid)),
        "complete_token_chart_uv_valid_fraction": float(np.mean(source.token_chart_uv_valid)),
        "mean_vertex_ideal_to_raw_warp_displacement_px": float(
            source.metadata["mean_vertex_ideal_to_raw_warp_displacement_px"]
        ),
        "maximum_vertex_ideal_to_raw_warp_displacement_px": float(
            source.metadata["maximum_vertex_ideal_to_raw_warp_displacement_px"]
        ),
        "positive_observation_weight_vertex_fraction": float(np.mean(source.vertex_observation_weight > 0.0)),
        "mean_observation_weight": float(np.mean(source.vertex_observation_weight)),
        "family_count": int(layout.family_keys.size),
        "canonical_node_count": int(layout.node_points_world.shape[0]),
        "canonical_feature_storage_dtype": str(canonical.codes.dtype),
        "canonical_feature_dimension": int(canonical.codes.shape[1]),
        "canonical_feature_projection": "none",
        "canonical_feature_uncompressed_bytes": int(
            canonical.codes.nbytes + canonical.prototype_codes.nbytes
        ),
        "multi_view_node_count": int(np.sum(node_views >= 2)),
        "multi_view_node_fraction": float(np.mean(node_views >= 2)),
        "observed_node_fraction": float(np.mean(node_views >= 1)),
        "median_observed_node_confidence": float(
            np.median(canonical.confidence[node_views >= 1]) if np.any(node_views >= 1) else 0.0
        ),
        "median_multi_view_uncertainty": float(
            np.median(canonical.uncertainty[node_views >= 2]) if np.any(node_views >= 2) else 0.0
        ),
        "median_same_node_cross_view_prototype_cosine": float(
            np.median(same_node_cosine) if same_node_cosine else 0.0
        ),
        "median_random_observed_node_cosine_seed20260830": float(
            np.median(random_cosine) if random_cosine else 0.0
        ),
        "same_node_prototype_pair_count": len(same_node_cosine),
        "layout_diagnostic_only": bool(layout.metadata.get("diagnostic_only", False)),
        "layout_feature_interface_eligible": bool(
            layout.metadata.get("feature_interface_eligible", False)
        ),
        "layout_production_blocker": layout.metadata.get("production_blocker"),
        "canonical_production_eligible": bool(canonical.metadata.get("production_eligible", False)),
        "deployment_resource_gate": "BLOCK_full_1280d_smoke_requires_post_fusion_64_to_128d_or_sparse_multiresolution_gate",
        "uses_query_or_ground_truth": False,
        "uses_gaussian_or_2dgs": False,
        "input_file_sha256": input_hashes,
        "output_file_sha256": {name: file_sha256(path) for name, path in outputs.items()},
        "output_serialized_bytes": {name: int(path.stat().st_size) for name, path in outputs.items()},
        "source_field_content_sha256": source.content_sha256,
        "family_layout_content_sha256": layout.content_sha256,
        "canonical_field_content_sha256": canonical.content_sha256,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--chart_cameras_json", required=True)
    parser.add_argument("--source_charts", required=True)
    parser.add_argument("--chart_focal_canvas_width", type=int, required=True)
    parser.add_argument("--raw_camera_manifest", required=True)
    parser.add_argument("--radio_manifest", required=True)
    parser.add_argument("--allowed_routes", nargs="+", required=True)
    parser.add_argument("--output_source_field", required=True)
    parser.add_argument("--output_family_layout", required=True)
    parser.add_argument("--output_canonical_field", required=True)
    parser.add_argument("--output_visualization_png", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--disjoint_upstream_authority")
    parser.add_argument("--chart_submap_plan")
    parser.add_argument("--expected_chart_submap_plan_content_sha256")
    parser.add_argument("--canonical_family_carrier")
    parser.add_argument("--expected_canonical_family_carrier_content_sha256")
    parser.add_argument("--maximum_cross_chart_distance_m", type=float, default=0.5)
    parser.add_argument("--maximum_cross_chart_normal_degrees", type=float, default=60.0)
    parser.add_argument("--maximum_intra_family_edge_m", type=float, default=1.0)
    parser.add_argument("--maximum_intra_family_normal_degrees", type=float, default=45.0)
    parser.add_argument("--maximum_anonymous_prototypes", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    atlas_path = Path(args.atlas).resolve()
    chart_cameras_path = Path(args.chart_cameras_json).resolve()
    source_charts_path = Path(args.source_charts).resolve()
    raw_camera_manifest_path = Path(args.raw_camera_manifest).resolve()
    radio_manifest_path = Path(args.radio_manifest).resolve()
    outputs = {
        "source_field": Path(args.output_source_field).resolve(),
        "family_layout": Path(args.output_family_layout).resolve(),
        "canonical_field": Path(args.output_canonical_field).resolve(),
        "visualization": Path(args.output_visualization_png).resolve(),
    }
    summary_path = Path(args.summary_json).resolve()
    if not bool(args.force) and any(path.exists() for path in (*outputs.values(), summary_path)):
        raise FileExistsError("refusing to overwrite chart RADIO UV artifacts")
    atlas = ExplicitChartAtlas.load_npz(atlas_path)
    atlas_metadata_without_content = dict(atlas.metadata)
    declared_atlas_content = str(atlas_metadata_without_content.pop("content_sha256", ""))
    if canonical_json_sha256(atlas_metadata_without_content) != declared_atlas_content:
        raise ValueError("explicit chart atlas metadata content seal differs")
    atlas_hash = str(atlas.metadata.get("content_sha256", ""))
    if len(atlas_hash) != 64:
        raise ValueError("explicit chart atlas lacks a sealed content hash")
    names = atlas.chart_names.astype(str).tolist()
    family_carrier_path = (
        None if args.canonical_family_carrier is None
        else Path(args.canonical_family_carrier).resolve()
    )
    strict_input_audit = _strict_disjoint_surface_audit(
        authority_path=(
            None if args.disjoint_upstream_authority is None
            else Path(args.disjoint_upstream_authority).resolve()
        ),
        submap_plan_path=(
            None if args.chart_submap_plan is None
            else Path(args.chart_submap_plan).resolve()
        ),
        family_carrier_path=family_carrier_path,
        expected_plan_content_sha256=args.expected_chart_submap_plan_content_sha256,
        expected_family_carrier_content_sha256=(
            args.expected_canonical_family_carrier_content_sha256
        ),
        chart_names=names,
        atlas_content_sha256=atlas_hash,
    )
    allowed = {str(value) for value in args.allowed_routes}
    if not allowed:
        raise ValueError("chart RADIO builder requires a nonempty route allowlist")
    image_ids = {name: chart_name_to_image_id(name) for name in names}
    routes = {image_id.split("/", 1)[0] for image_id in image_ids.values()}
    # This check occurs before any RADIO archive is opened.
    if not routes.issubset(allowed):
        raise ValueError("chart atlas contains a route outside the mapping allowlist")
    camera_payload = json.loads(raw_camera_manifest_path.read_text())
    if not isinstance(camera_payload, dict):
        raise ValueError("raw mapping camera manifest must be an object")
    camera_manifest = camera_payload.get("cameras", camera_payload)
    if not isinstance(camera_manifest, dict):
        raise ValueError("raw mapping camera manifest lacks a camera inventory")
    if int(args.chart_focal_canvas_width) <= 0:
        raise ValueError("chart focal canvas width must be positive")
    with np.load(source_charts_path, allow_pickle=False) as charts_data:
        source_depth_shape = tuple(np.asarray(charts_data["depths"]).shape)
    if len(source_depth_shape) != 3 or source_depth_shape[0] != len(names):
        raise ValueError("source charts and explicit atlas inventories differ")
    chart_height, chart_width = map(int, source_depth_shape[1:])
    records = _load_radio_records(radio_manifest_path)
    chart_rows_all = _load_chart_camera_rows(chart_cameras_path)
    if not set(names).issubset(chart_rows_all):
        raise ValueError("chart cameras.json lacks selected source views")

    radio_by_view, raw_camera_by_view, chart_camera_by_view, poses = {}, {}, {}, {}
    token_hashes: dict[str, str] = {}
    for name in names:
        image_id = image_ids[name]
        if image_id not in camera_manifest or image_id not in records:
            raise ValueError("raw camera/RADIO manifest lacks a chart source view")
        record = records[image_id]
        if not str(record.get("split", "")).startswith("train_"):
            raise ValueError("chart RADIO source is not a mapping/train split record")
        layers = record.get("layers", [])
        matching = [row for row in layers if str(row.get("name", "")) == "radio_final"]
        if len(matching) != 1 or str(matching[0].get("layer", "")).lower() != "final":
            raise ValueError("chart source requires exactly one RADIO-final layer")
        token_path = Path(str(record["token_path"]))
        if not token_path.is_absolute():
            token_path = (Path.cwd() / token_path).resolve()
        actual_hash = file_sha256(token_path)
        if actual_hash != str(record.get("checksum", "")):
            raise ValueError("RADIO token file differs from its manifest checksum")
        with np.load(token_path, allow_pickle=False) as data:
            value = np.asarray(data["radio_final"])
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or int(value.shape[0]) != int(matching[0]["channels"]):
            raise ValueError("RADIO-final tensor differs from manifest layer shape")
        raw_camera = RawSimpleRadialCamera.from_manifest_row(camera_manifest[image_id])
        chart_row = chart_rows_all[name]
        chart_focal = float(chart_row["focal"]) * float(chart_width) / float(
            args.chart_focal_canvas_width
        )
        # Fail closed if the raw ideal-pinhole rebuild and MAtCha chart camera
        # do not describe the same normalized rays.
        if not np.isclose(
            raw_camera.focal / raw_camera.width,
            chart_focal / chart_width,
            atol=1e-7, rtol=1e-6,
        ):
            raise ValueError("raw and low-resolution chart camera fields of view differ")
        radio_by_view[name] = value
        raw_camera_by_view[name] = raw_camera
        chart_camera_by_view[name] = IdealChartCamera(
            width=chart_width, height=chart_height, focal=chart_focal,
            cx=chart_width / 2.0, cy=chart_height / 2.0,
        ).validated()
        poses[name] = np.asarray(chart_row["pose_c2w"], dtype=np.float64)
        token_hashes[image_id] = actual_hash

    input_hashes = {
        "atlas": file_sha256(atlas_path),
        "chart_cameras_json": file_sha256(chart_cameras_path),
        "source_charts": file_sha256(source_charts_path),
        "raw_camera_manifest": file_sha256(raw_camera_manifest_path),
        "radio_manifest": file_sha256(radio_manifest_path),
        "selected_radio_files_inventory": canonical_json_sha256(token_hashes),
    }
    if bool(strict_input_audit.get("strict_disjoint_authority_verified", False)):
        input_hashes["disjoint_upstream_authority"] = str(
            strict_input_audit["authority_file_sha256"]
        )
        input_hashes["chart_submap_plan"] = str(
            strict_input_audit["chart_submap_plan_file_sha256"]
        )
        input_hashes["canonical_surface_family_carrier"] = str(
            strict_input_audit["surface_family_carrier_file_sha256"]
        )
    source = attach_radio_to_chart_atlas(
        atlas,
        radio_by_view=radio_by_view,
        raw_camera_by_view=raw_camera_by_view,
        chart_camera_by_view=chart_camera_by_view,
        camera_pose_c2w_by_view=poses,
        atlas_content_sha256=atlas_hash,
        lineage_metadata={
            "input_file_sha256": input_hashes,
            "selected_token_files_sha256": token_hashes,
            "route_allowlist": sorted(allowed),
            "route_allowlist_applied_before_opening_radio_archives": True,
            "selected_radio_split_contract": "mapping_train_prefix_only",
            "query_split_records_opened": 0,
            "chart_grid_shape_hw": [chart_height, chart_width],
            "chart_focal_canvas_width": int(args.chart_focal_canvas_width),
            "chart_focal_rescaling_contract": "chart_focal=cameras_json_focal*chart_width/focal_canvas_width",
            "strict_input_audit": strict_input_audit,
        },
    )
    source.save_npz(outputs["source_field"])
    if family_carrier_path is None:
        layout = build_diagnostic_metric_surface_families(
            atlas,
            atlas_content_sha256=atlas_hash,
            maximum_cross_chart_distance_m=float(args.maximum_cross_chart_distance_m),
            maximum_cross_chart_normal_degrees=float(args.maximum_cross_chart_normal_degrees),
            maximum_intra_family_edge_m=float(args.maximum_intra_family_edge_m),
            maximum_intra_family_normal_degrees=float(args.maximum_intra_family_normal_degrees),
        )
    else:
        carrier = CanonicalSurfaceFamilyCarrier.load_npz(family_carrier_path)
        layout = build_carrier_constrained_surface_family_layout(
            atlas,
            carrier,
            atlas_content_sha256=atlas_hash,
            carrier_content_sha256=str(
                strict_input_audit["surface_family_carrier_content_sha256"]
            ),
            maximum_cross_chart_distance_m=float(args.maximum_cross_chart_distance_m),
            maximum_cross_chart_normal_degrees=float(args.maximum_cross_chart_normal_degrees),
        )
    layout.save_npz(outputs["family_layout"])
    canonical = fuse_canonical_chart_radio_field(
        source,
        layout,
        maximum_anonymous_prototypes=int(args.maximum_anonymous_prototypes),
    )
    canonical.save_npz(outputs["canonical_field"])
    visualize_chart_radio_uv_field(source, canonical, outputs["visualization"])
    runtime_seconds = time.perf_counter() - started
    # Linux reports ru_maxrss in KiB.  This builder is Linux-only in the
    # current research environment; record the unit conversion explicitly.
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    summary = _summary(
        source=source, layout=layout, canonical=canonical,
        outputs=outputs, input_hashes=input_hashes,
        strict_input_audit=strict_input_audit,
        runtime_seconds=runtime_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
