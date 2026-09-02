"""Freeze a common pixel/topology domain for paired DAV2/MoGe chart gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.run_goal_maplet_masked_chart_alignment_gate import (
    _load_reference_depth,
    _manifest_rows,
    _validated_disjoint_source_authority,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
    load_model_neutral_alignment_selection,
    load_moge_reference_control_alignment_selection,
    load_paired_stride2_diagnostic_alignment_selection,
    moge_reference_control_alignment_metadata,
    paired_stride2_diagnostic_alignment_metadata,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    _load_source_reference_dense,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dav2_initial_metric(
    charts: Path, cameras: dict, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(charts, allow_pickle=False) as data:
        aligned = np.asarray(data["pts"], np.float64)
        aligned_depth = np.asarray(data["depths"], np.float64)
        prior_depth = np.asarray(data["prior_depths"], np.float64)
        scale = float(data["scale_factor"])
    all_names = [Path(path).name for path in cameras["filepaths"]]
    lexical = sorted(all_names)
    if len(lexical) != len(aligned):
        raise ValueError("DAV2 chart count differs from cameras")
    chart_row = {name: row for row, name in enumerate(lexical)}
    camera_row = {name: row for row, name in enumerate(all_names)}
    points = []
    valid = []
    for name in names:
        row = chart_row[name]
        camera = camera_row[name]
        c2w = np.asarray(cameras["cams2world"][camera], np.float64)
        center = c2w[:3, 3] * scale
        ratio = prior_depth[row] / aligned_depth[row]
        initial = center + (aligned[row] - center) * ratio[..., None]
        initial = cv2.resize(initial / scale, (256, 144), interpolation=cv2.INTER_AREA)
        prior = cv2.resize(
            prior_depth[row] / scale, (256, 144), interpolation=cv2.INTER_AREA
        )
        good = np.isfinite(initial).all(2) & np.isfinite(prior) & (prior > 0)
        points.append(initial.astype(np.float32))
        valid.append(good)
    return np.stack(points), np.stack(valid)


def _dav2_initial_files(
    root: Path, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    valid = []
    for name in names:
        with np.load(root / f"{name}.npz", allow_pickle=False) as data:
            world = np.asarray(data["points_world"], np.float32)
            good = np.asarray(data["valid"], bool)
            metadata = json.loads(str(data["metadata_json"].item()))
        if (
            metadata.get("artifact_type") != "goal_maplet_dav2_chart_initializer_v1"
            or metadata.get("source_name") != name
        ):
            raise ValueError("wrong DAV2 initializer binding")
        claimed = metadata.pop("content_sha256", None)
        if claimed != _canonical(metadata):
            raise ValueError("DAV2 initializer content hash differs")
        points.append(world)
        valid.append(good & np.isfinite(world).all(2))
    return np.stack(points), np.stack(valid)


def _moge_initial_metric(
    root: Path, names: list[str], cameras: dict
) -> tuple[np.ndarray, np.ndarray]:
    camera_row = {
        Path(path).name: row for row, path in enumerate(cameras["filepaths"])
    }
    points = []
    valid = []
    for name in names:
        path = root / f"{name}.npz"
        with np.load(path, allow_pickle=False) as data:
            camera_points = np.asarray(data["points_camera"], np.float32)
            good = np.asarray(data["valid"], bool)
            metadata = json.loads(str(data["metadata_json"].item()))
        row = camera_row[name]
        source_path = Path(cameras["filepaths"][row])
        if (
            metadata.get("artifact_type") != "goal_maplet_moge3_chart_initializer_v2"
            or metadata.get("source_name") != name
            or metadata.get("camera_focal_canvas_width") != 512
            or float(metadata.get("camera_focal_canvas_px"))
            != float(cameras["focals"][row])
            or metadata.get("source_image_file_sha256") != _sha(source_path)
        ):
            raise ValueError("wrong MoGe initializer binding")
        claimed = metadata.pop("content_sha256", None)
        if claimed != _canonical(metadata):
            raise ValueError("MoGe initializer content hash differs")
        c2w = np.asarray(cameras["cams2world"][row], np.float64)
        world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        points.append(world.astype(np.float32))
        valid.append(good & np.isfinite(world).all(2))
    return np.stack(points), np.stack(valid)


def _conservative_face_domain(
    valid: np.ndarray,
    point_sets: tuple[np.ndarray, ...],
    camera_centers: np.ndarray,
    stride: int,
    minimum_edge_m: float = 0.5,
    relative_edge: float = 0.05,
) -> np.ndarray:
    height, width = valid.shape[-2:]
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    output = np.zeros((len(valid), len(ys) - 1, len(xs) - 1), bool)
    for chart in range(len(valid)):
        for oy, y in enumerate(ys[:-1]):
            y1 = int(ys[oy + 1])
            for ox, x in enumerate(xs[:-1]):
                x1 = int(xs[ox + 1])
                region = valid[chart, y : y1 + 1, x : x1 + 1]
                if not region.all():
                    continue
                safe = True
                for points in point_sets:
                    patch = points[chart, y : y1 + 1, x : x1 + 1]
                    horizontal = np.linalg.norm(patch[:, 1:] - patch[:, :-1], axis=2)
                    vertical = np.linalg.norm(patch[1:] - patch[:-1], axis=2)
                    range_m = np.median(
                        np.linalg.norm(patch - camera_centers[chart], axis=2)
                    )
                    threshold = max(minimum_edge_m, relative_edge * float(range_m))
                    if max(float(horizontal.max()), float(vertical.max())) > threshold:
                        safe = False
                        break
                output[chart, oy, ox] = safe
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--pointmaps_dir", type=Path, required=True)
    parser.add_argument("--dav2_charts", type=Path)
    parser.add_argument("--dav2_initializers", type=Path)
    parser.add_argument("--moge3_initializers", type=Path, required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path, required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--route", default="seq4")
    parser.add_argument(
        "--allow_paired_stride2_diagnostic_alignment_adapter",
        action="store_true",
        help=(
            "Explicitly allow the non-promotable paired-stride2 source-geometry "
            "control plan. This never authorizes full-gate/exporter use."
        ),
    )
    parser.add_argument(
        "--moge_reference_only_control",
        action="store_true",
        help=(
            "Build a non-promotable optimizer domain from MoGe3 and the "
            "source MASt3R reference only; DAV2 bytes and validity are not read."
        ),
    )
    parser.add_argument(
        "--allow_moge_reference_only_alignment_adapter",
        action="store_true",
        help="Bind the domain to a sealed MoGe/reference-only control plan.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite chart comparison domain")
    if args.allow_paired_stride2_diagnostic_alignment_adapter and args.allow_moge_reference_only_alignment_adapter:
        raise ValueError("comparison-domain adapters are mutually exclusive")
    if args.moge_reference_only_control:
        if args.dav2_charts is not None or args.dav2_initializers is not None:
            raise ValueError("MoGe/reference-only control forbids DAV2 inputs")
    elif (args.dav2_charts is None) == (args.dav2_initializers is None):
        raise ValueError("provide exactly one DAV2 chart source")
    cameras = json.loads(args.cameras.read_text())
    authority = _validated_disjoint_source_authority(
        args.disjoint_upstream_authority, args.cameras, args.pointmaps_dir
    )
    if (
        authority is None
        or authority.get("artifact_type")
        != "goal_maplet_disjoint_chart_upstream_authority_v2"
        or not authority.get("physical_source_held_input_roots_disjoint")
    ):
        raise ValueError("comparison domain requires physically isolated v2 authority")
    authority_claimed = authority["content_sha256"]
    plan = ChartSubmapPlan.load_npz(args.frozen_submap_plan)
    diagnostic_adapter_metadata = None
    if args.allow_moge_reference_only_alignment_adapter:
        selection = load_moge_reference_control_alignment_selection(
            args.frozen_submap_plan,
            expected_plan_content_sha256=args.expected_plan_content_sha256,
            diagnostic_alignment_adapter_opt_in=True,
        )
        diagnostic_adapter_metadata = moge_reference_control_alignment_metadata(plan)
    elif args.allow_paired_stride2_diagnostic_alignment_adapter:
        selection = load_paired_stride2_diagnostic_alignment_selection(
            args.frozen_submap_plan,
            expected_plan_content_sha256=args.expected_plan_content_sha256,
            diagnostic_alignment_adapter_opt_in=True,
        )
        diagnostic_adapter_metadata = paired_stride2_diagnostic_alignment_metadata(
            plan
        )
    else:
        selection = load_model_neutral_alignment_selection(
            args.frozen_submap_plan,
            expected_plan_content_sha256=args.expected_plan_content_sha256,
        )
    plan_lineage = plan.metadata.get("lineage")
    if not isinstance(plan_lineage, dict) or plan_lineage.get(
        "disjoint_authority_content_sha256"
    ) != authority_claimed:
        raise ValueError("frozen submap plan and disjoint authority differ")
    if plan_lineage.get("source_tree_sha256") != authority["source"].get("tree_sha256"):
        raise ValueError("frozen submap plan and authority source tree differ")
    if len(selection.operational_submaps) != 1:
        raise ValueError("comparison domain requires exactly one operational submap")
    names = list(selection.ordered_names)
    if len(names) < 2:
        raise ValueError("comparison route has fewer than two charts")
    camera_names = [Path(path).name for path in cameras["filepaths"]]
    if len(camera_names) != len(set(camera_names)) or not set(names).issubset(camera_names):
        raise ValueError("frozen submap inventory is absent or ambiguous in source cameras")
    moge_manifest = json.loads((args.moge3_initializers / "manifest.json").read_text())
    moge_claimed = moge_manifest.pop("content_sha256", None)
    if moge_claimed != _canonical(moge_manifest):
        raise ValueError("MoGe initializer manifest content hash differs")
    moge_manifest["content_sha256"] = moge_claimed
    if (
        moge_manifest.get("artifact_type")
        != "goal_maplet_moge3_chart_initializer_run_v2"
        or moge_manifest.get("cameras_file_sha256") != _sha(args.cameras)
        or moge_manifest.get("uses_camera_pose") is not False
        or moge_manifest.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError("MoGe initializer manifest is not bound to source cameras")
    if sorted(moge_manifest.get("allowed_routes", [])) != sorted(
        authority["source"].get("routes", [])
    ):
        raise ValueError("MoGe initializer routes differ from strict authority")
    moge_rows = _manifest_rows(moge_manifest, "MoGe initializer")
    for name in names:
        if (
            name not in moge_rows
            or moge_rows[name]["file_sha256"]
            != _sha(args.moge3_initializers / f"{name}.npz")
        ):
            raise ValueError("MoGe initializer file differs from run manifest")
    if args.dav2_initializers is not None:
        dav2_manifest = json.loads((args.dav2_initializers / "manifest.json").read_text())
        dav2_claimed = dav2_manifest.pop("content_sha256", None)
        if dav2_claimed != _canonical(dav2_manifest):
            raise ValueError("DAV2 initializer manifest content hash differs")
        dav2_manifest["content_sha256"] = dav2_claimed
        if dav2_manifest.get("disjoint_upstream_authority_content_sha256") != authority_claimed:
            raise ValueError("DAV2 initializer authority differs")
        if (
            dav2_manifest.get("artifact_type")
            != "goal_maplet_dav2_chart_initializer_run_v1"
            or Path(dav2_manifest.get("source_only_mast3r_root", "")).resolve()
            != Path(authority["source"]["root"]).resolve()
            or Path(dav2_manifest.get("isolated_source_posed_colmap_root", "")).resolve()
            != Path(authority["isolated_source_input"]["root"]).resolve()
        ):
            raise ValueError("DAV2 initializer manifest source lineage differs")
        if dav2_manifest.get("source_only_mast3r_tree_sha256") != authority[
            "source"
        ]["tree_sha256"]:
            raise ValueError("DAV2 initializer source tree differs from authority")
        dav2_rows = _manifest_rows(dav2_manifest, "DAV2 initializer")
        for name in names:
            if (
                name not in dav2_rows
                or dav2_rows[name]["file_sha256"]
                != _sha(args.dav2_initializers / f"{name}.npz")
            ):
                raise ValueError("DAV2 initializer file differs from run manifest")
    camera_row = {
        Path(path).name: row for row, path in enumerate(cameras["filepaths"])
    }
    camera_centers = np.asarray(
        [np.asarray(cameras["cams2world"][camera_row[name]])[:3, 3] for name in names],
        np.float64,
    )
    if args.moge_reference_only_control:
        dav2_points = dav2_valid = None
    else:
        dav2_points, dav2_valid = (
            _dav2_initial_files(args.dav2_initializers, names)
            if args.dav2_initializers is not None
            else _dav2_initial_metric(args.dav2_charts, cameras, names)
        )
    moge_points, moge_valid = _moge_initial_metric(
        args.moge3_initializers, names, cameras
    )
    reference_valid = []
    reference_points = []
    for name in names:
        c2w = np.asarray(cameras["cams2world"][camera_row[name]], np.float64)
        _, good = _load_reference_depth(
            args.pointmaps_dir / f"{Path(name).stem}.json", c2w
        )
        reference_valid.append(good)
        points, _ = _load_source_reference_dense(
            args.pointmaps_dir / f"{Path(name).stem}.json",
            output_height=144,
            output_width=256,
        )
        reference_points.append(points)
    reference_valid = np.stack(reference_valid)
    reference_points = np.stack(reference_points)
    common = moge_valid & reference_valid
    if not args.moge_reference_only_control:
        common &= dav2_valid
    edge_geometries = (
        (moge_points, reference_points)
        if args.moge_reference_only_control
        else (dav2_points, moge_points)
    )
    arrays = {
        "chart_names": np.asarray(names),
        "valid": common,
        "face_valid_stride4": _conservative_face_domain(
            common, edge_geometries, camera_centers, 4
        ),
        "face_valid_stride8": _conservative_face_domain(
            common, edge_geometries, camera_centers, 8
        ),
    }
    metadata = {
        "artifact_type": "goal_maplet_chart_comparison_domain_v1",
        "route": args.route,
        "chart_count": len(names),
        "cameras_file_sha256": _sha(args.cameras),
        "pointmap_inventory": {
            name: _sha(args.pointmaps_dir / f"{Path(name).stem}.json")
            for name in names
        },
        "dav2_charts_file_sha256": _sha(args.dav2_charts) if args.dav2_charts else None,
        "dav2_initializer_file_sha256": (
            {name: _sha(args.dav2_initializers / f"{name}.npz") for name in names}
            if args.dav2_initializers
            else None
        ),
        "moge_initializer_file_sha256": {
            name: _sha(args.moge3_initializers / f"{name}.npz") for name in names
        },
        "disjoint_upstream_authority_file_sha256": _sha(
            args.disjoint_upstream_authority
        ),
        "disjoint_upstream_authority_content_sha256": authority_claimed,
        "source_tree_sha256": authority["source"]["tree_sha256"],
        "frozen_submap_plan_file_sha256": _sha(args.frozen_submap_plan),
        "frozen_submap_plan_content_sha256": selection.plan_content_sha256,
        "selected_chart_names_in_order_sha256": plan.metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "alignment_runner_contract": plan.metadata["alignment_runner_contract"],
        "dav2_initializer_manifest_file_sha256": (
            _sha(args.dav2_initializers / "manifest.json")
            if args.dav2_initializers else None
        ),
        "dav2_initializer_manifest_content_sha256": (
            dav2_claimed if args.dav2_initializers else None
        ),
        "moge_initializer_manifest_file_sha256": _sha(
            args.moge3_initializers / "manifest.json"
        ),
        "moge_initializer_manifest_content_sha256": moge_claimed,
        "validity": (
            "focal_correct_MoGe3_valid AND source_only_MASt3R_reference_valid"
            if args.moge_reference_only_control
            else "DAV2_valid AND focal_correct_MoGe3_valid AND source_only_MASt3R_reference_valid"
        ),
        "topology": (
            "all_pixels_common_valid_and_all_unit_edges_safe_in_MoGe3_and_source_reference"
            if args.moge_reference_only_control
            else "all_pixels_in_sample_quad_common_valid_and_all_unit_edges_safe_in_both_initializers"
        ),
        "moge_reference_only_control": bool(args.moge_reference_only_control),
        "dav2_geometry_consumed": not args.moge_reference_only_control,
        "valid_fraction_per_chart": common.reshape(len(common), -1).mean(1).tolist(),
        "arrays_sha256": arrays_sha256(arrays),
        "uses_query_or_ground_truth": False,
    }
    if diagnostic_adapter_metadata is not None:
        metadata.update(diagnostic_adapter_metadata)
        metadata.update(
            {
                "diagnostic_only": True,
                "production_eligible": False,
                "diagnostic_alignment_source_plan_artifact_type": plan.metadata[
                    "artifact_type"
                ],
                "diagnostic_alignment_source_plan_representation": plan.metadata[
                    "representation"
                ],
                "optimizer_pixel_domain_only": True,
                "paired_stride2_topology_embedded": False,
                "face_valid_stride4_stride8_role": (
                    "legacy_optimizer_domain_diagnostics_only_not_the_sealed_"
                    "paired_stride2_topology"
                ),
                "full_gate_exact_topology_sealer_eligible": False,
            }
        )
    metadata["content_sha256"] = _canonical(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
