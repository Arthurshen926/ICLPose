"""Run a bounded, route-clean DAV2/MoGe-3 chart-alignment comparison.

This is a geometry gate, not a production mapper.  It deliberately selects a
small frozen chart inventory, uses mapping cameras only, and never opens query
poses or Cambridge ground-truth files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
    load_model_neutral_alignment_selection,
    load_moge_reference_control_alignment_selection,
    load_paired_stride2_diagnostic_alignment_selection,
    moge_reference_control_alignment_metadata,
    paired_stride2_diagnostic_alignment_metadata,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_sha256(root: Path) -> str:
    rows = []
    for path in sorted(value for value in Path(root).rglob("*") if value.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )
    return _canonical_sha256(rows)


def _alignment_code_inventory(matcha_repo: Path) -> tuple[list[dict], str]:
    source_root = Path(matcha_repo) / "matcha"
    rows = []
    for path in sorted(source_root.rglob("*.py")):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(matcha_repo)),
                "size": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise ValueError("MAtCha alignment source inventory is empty")
    return rows, _canonical_sha256(rows)


def _matcha_internal_lexical_permutations(
    official_names: list[str], matcha_img_paths: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return official->lexical input order and lexical->official restore order.

    MAtCha currently sorts cameras by image path inside
    ``create_gs_cameras_from_pointmap`` without sorting the PointMap tensors.
    All numerical inputs must therefore be put in that same lexical order,
    then restored to the frozen plan order before an artifact is written.
    """
    if len(set(official_names)) != len(official_names):
        raise ValueError("frozen chart inventory contains duplicate names")
    sort_keys = official_names if matcha_img_paths is None else matcha_img_paths
    if len(sort_keys) != len(official_names):
        raise ValueError("chart names and MAtCha image paths differ in cardinality")
    if len(set(sort_keys)) != len(sort_keys):
        raise ValueError("MAtCha image path inventory contains duplicates")
    internal = np.asarray(
        sorted(range(len(official_names)), key=lambda row: sort_keys[row]),
        dtype=np.int64,
    )
    restore = np.argsort(internal, kind="stable")
    if not np.array_equal(internal[restore], np.arange(len(official_names))):
        raise AssertionError("chart-order permutation is not invertible")
    return internal, restore


def _validated_disjoint_source_authority(
    path: Path | None, cameras_path: Path, pointmaps_dir: Path
) -> dict | None:
    if path is None:
        return None
    authority = json.loads(path.read_text())
    claimed = authority.pop("content_sha256", None)
    if claimed != _canonical_sha256(authority):
        raise ValueError("disjoint upstream authority content hash differs")
    authority["content_sha256"] = claimed
    if (
        authority.get("artifact_type")
        not in {
            "goal_maplet_disjoint_chart_upstream_authority_v1",
            "goal_maplet_disjoint_chart_upstream_authority_v2",
        }
        or not authority.get("strict_disjoint_upstream")
        or not authority.get("source_held_image_disjoint")
        or not authority.get("source_held_route_disjoint")
    ):
        raise ValueError("upstream authority does not certify source/held separation")
    source = authority["source"]
    if Path(source["root"]).resolve() != cameras_path.resolve().parent:
        raise ValueError("cameras are not from the certified source-only run")
    if cameras_path.resolve() != (Path(source["root"]) / "cameras.json").resolve():
        raise ValueError("source cameras path differs from authority")
    if pointmaps_dir.resolve() != (Path(source["root"]) / "pointmaps").resolve():
        raise ValueError("source point-map directory differs from authority")
    if _sha256(cameras_path) != source["cameras_file_sha256"]:
        raise ValueError("source cameras bytes differ from authority")
    rows = [
        {
            "name": name,
            "file_sha256": _sha256(pointmaps_dir / f"{Path(name).stem}.json"),
        }
        for name in sorted(source["ordered_names"])
    ]
    if _canonical_sha256(rows) != source["pointmap_inventory_sha256"]:
        raise ValueError("source point-map bytes differ from authority")
    if _tree_sha256(Path(source["root"])) != source.get("tree_sha256"):
        raise ValueError("complete source tree differs from authority")
    return authority


def _nearest_valid_fill(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise ValueError("chart initializer has no valid pixels")
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = np.asarray(points, np.float32).copy()
    filled[~valid] = filled[nearest[0][~valid], nearest[1][~valid]]
    if not np.isfinite(filled).all():
        raise ValueError("nearest-valid carrier fill remained nonfinite")
    return filled


def _manifest_rows(manifest: dict, label: str) -> dict[str, dict]:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or manifest.get("chart_count") != len(rows):
        raise ValueError(f"{label} manifest row inventory differs")
    result = {str(row.get("name", "")): row for row in rows if isinstance(row, dict)}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"{label} manifest contains empty/duplicate chart names")
    return result


def _load_reference_depth(pointmap_path: Path, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(pointmap_path.read_text())
    points = np.asarray(payload["points"], np.float64).reshape(288, 512, 3)
    confidence = np.asarray(payload["confs"], np.float64)
    camera_depth = ((points - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]
    depth = cv2.resize(camera_depth, (256, 144), interpolation=cv2.INTER_AREA)
    conf = cv2.resize(confidence, (256, 144), interpolation=cv2.INTER_AREA)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(conf) & (conf > 0.25)
    return depth.astype(np.float32), valid


def _selected_names(cameras: dict, route: str, count: int) -> list[str]:
    names = sorted(
        Path(path).name
        for path in cameras["filepaths"]
        if Path(path).name.split("__", 1)[0] == route
    )
    if len(names) < count:
        raise ValueError(f"route {route} has only {len(names)} chart cameras")
    # Evenly cover the selected route inventory, rather than using a temporal
    # prefix whose charts can be nearly duplicate.
    rows = np.linspace(0, len(names) - 1, count).round().astype(int)
    selected = [names[int(row)] for row in rows]
    if len(set(selected)) != count:
        raise RuntimeError("chart selection produced duplicates")
    return selected


def _strict_plan_selection(
    plan_path: Path,
    *,
    expected_plan_content_sha256: str,
    authority: dict,
    cameras: dict,
    allow_paired_stride2_diagnostic_alignment_adapter: bool = False,
    allow_moge_reference_only_alignment_adapter: bool = False,
) -> tuple[list[str], ChartSubmapPlan]:
    """Resolve an externally hash-pinned, source-only operational submap."""

    plan = ChartSubmapPlan.load_npz(plan_path)
    if allow_paired_stride2_diagnostic_alignment_adapter and allow_moge_reference_only_alignment_adapter:
        raise ValueError("alignment adapters are mutually exclusive")
    if allow_moge_reference_only_alignment_adapter:
        selection = load_moge_reference_control_alignment_selection(
            plan_path,
            expected_plan_content_sha256=expected_plan_content_sha256,
            diagnostic_alignment_adapter_opt_in=True,
        )
    elif allow_paired_stride2_diagnostic_alignment_adapter:
        selection = load_paired_stride2_diagnostic_alignment_selection(
            plan_path,
            expected_plan_content_sha256=expected_plan_content_sha256,
            diagnostic_alignment_adapter_opt_in=True,
        )
    else:
        selection = load_model_neutral_alignment_selection(
            plan_path,
            expected_plan_content_sha256=expected_plan_content_sha256,
        )
    if len(selection.operational_submaps) != 1:
        raise ValueError("alignment runner requires exactly one operational frozen submap")
    lineage = plan.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("frozen submap plan lacks source-only lineage")
    if lineage.get("disjoint_authority_content_sha256") != authority.get("content_sha256"):
        raise ValueError("frozen submap plan and disjoint authority differ")
    if lineage.get("source_tree_sha256") != authority.get("source", {}).get("tree_sha256"):
        raise ValueError("frozen submap plan and authority source tree differ")
    names = list(selection.ordered_names)
    camera_names = [Path(path).name for path in cameras["filepaths"]]
    if len(camera_names) != len(set(camera_names)) or not set(names).issubset(camera_names):
        raise ValueError("frozen submap inventory is absent or ambiguous in source cameras")
    return names, plan


def _load_strict_comparison_domain(
    path: Path,
    *,
    expected_content_sha256: str,
    selected_names: list[str],
    authority_path: Path,
    authority: dict,
    plan_path: Path,
    plan: ChartSubmapPlan,
    cameras_path: Path,
    pointmaps_dir: Path,
    initializer: str,
    initializer_path: Path,
    allow_paired_stride2_diagnostic_alignment_adapter: bool = False,
    allow_moge_reference_only_alignment_adapter: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load a common domain and replay all mutable upstream bindings.

    A self-consistent NPZ is insufficient authority: an old or alternate
    common mask could otherwise change the paired sample.  Both plan/domain
    content hashes must therefore be supplied by the runner invocation, and
    the domain must bind the exact current source and initializer bytes.
    """

    if len(str(expected_content_sha256)) != 64:
        raise ValueError("alignment runner must pin a comparison-domain content hash")
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "chart_names", "valid", "face_valid_stride4", "face_valid_stride8",
            )
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    claimed = metadata.pop("content_sha256", None)
    if claimed != _canonical_sha256(metadata):
        raise ValueError("comparison domain content hash differs")
    metadata["content_sha256"] = claimed
    if claimed != expected_content_sha256:
        raise ValueError("comparison domain differs from runner authority")
    if metadata.get("artifact_type") != "goal_maplet_chart_comparison_domain_v1":
        raise ValueError("wrong chart comparison domain schema")
    if metadata.get("arrays_sha256") != arrays_sha256(arrays):
        raise ValueError("comparison domain arrays differ from metadata")
    if arrays["chart_names"].astype(str).tolist() != selected_names:
        raise ValueError("comparison domain and frozen submap inventory differ")
    required_equal = {
        "disjoint_upstream_authority_file_sha256": _sha256(authority_path),
        "disjoint_upstream_authority_content_sha256": authority["content_sha256"],
        "source_tree_sha256": authority["source"]["tree_sha256"],
        "frozen_submap_plan_file_sha256": _sha256(plan_path),
        "frozen_submap_plan_content_sha256": plan.metadata["content_sha256"],
        "selected_chart_names_in_order_sha256": plan.metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "alignment_runner_contract": plan.metadata["alignment_runner_contract"],
        "cameras_file_sha256": _sha256(cameras_path),
    }
    for key, expected in required_equal.items():
        if metadata.get(key) != expected:
            raise ValueError(f"comparison domain {key} differs from current strict inputs")
    if allow_paired_stride2_diagnostic_alignment_adapter and allow_moge_reference_only_alignment_adapter:
        raise ValueError("comparison-domain adapters are mutually exclusive")
    if allow_moge_reference_only_alignment_adapter:
        expected_adapter = moge_reference_control_alignment_metadata(plan)
        for key, expected in expected_adapter.items():
            if metadata.get(key) != expected:
                raise ValueError(f"comparison domain MoGe/reference adapter {key} differs")
        for key, expected in {
            "diagnostic_only": True,
            "production_eligible": False,
            "optimizer_pixel_domain_only": True,
            "full_gate_exact_topology_sealer_eligible": False,
            "moge_reference_only_control": True,
            "dav2_geometry_consumed": False,
        }.items():
            if metadata.get(key) != expected:
                raise ValueError(f"comparison domain MoGe/reference control {key} differs")
    elif allow_paired_stride2_diagnostic_alignment_adapter:
        expected_adapter = paired_stride2_diagnostic_alignment_metadata(plan)
        for key, expected in expected_adapter.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"comparison domain diagnostic adapter {key} differs from plan"
                )
        diagnostic_domain_required = {
            "diagnostic_only": True,
            "production_eligible": False,
            "optimizer_pixel_domain_only": True,
            "paired_stride2_topology_embedded": False,
            "full_gate_exact_topology_sealer_eligible": False,
        }
        for key, expected in diagnostic_domain_required.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"comparison domain diagnostic control {key} must equal {expected!r}"
                )
    elif metadata.get("diagnostic_alignment_adapter_used") is True:
        raise ValueError(
            "paired stride-2 diagnostic comparison domain requires explicit adapter opt-in"
        )
    expected_pointmaps = {
        name: _sha256(pointmaps_dir / f"{Path(name).stem}.json")
        for name in selected_names
    }
    if metadata.get("pointmap_inventory") != expected_pointmaps:
        raise ValueError("comparison domain point-map inventory differs from current source")
    if initializer == "dav2":
        recorded = metadata.get("dav2_initializer_file_sha256")
        current = {
            name: _sha256(initializer_path / f"{name}.npz")
            for name in selected_names
        }
    elif initializer == "moge3":
        recorded = metadata.get("moge_initializer_file_sha256")
        current = {
            name: _sha256(initializer_path / f"{name}.npz")
            for name in selected_names
        }
    else:
        raise ValueError("unknown strict initializer arm")
    if recorded != current:
        raise ValueError("comparison domain initializer inventory differs from current arm")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializer", choices=("dav2", "moge3"), required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--pointmaps_dir", type=Path, required=True)
    parser.add_argument("--dav2_charts", type=Path)
    parser.add_argument("--dav2_initializers", type=Path)
    parser.add_argument("--moge3_initializers", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--disjoint_upstream_authority", type=Path)
    parser.add_argument(
        "--comparison_domain",
        type=Path,
        help="Frozen common DAV2/MoGe/reference pixel domain for paired gates.",
    )
    parser.add_argument("--expected_comparison_domain_content_sha256")
    parser.add_argument("--frozen_submap_plan", type=Path)
    parser.add_argument("--expected_plan_content_sha256")
    parser.add_argument("--expected_alignment_code_inventory_sha256")
    parser.add_argument(
        "--allow_paired_stride2_diagnostic_alignment_adapter",
        action="store_true",
        help=(
            "Explicitly run the non-promotable paired-stride2 source-geometry "
            "alignment control. This never authorizes full-gate/exporter use."
        ),
    )
    parser.add_argument(
        "--allow_moge_reference_only_alignment_adapter",
        action="store_true",
        help="Explicitly run the non-promotable MoGe3/reference-only alignment control.",
    )
    parser.add_argument("--route", default="seq4")
    parser.add_argument("--chart_count", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse chart-alignment gate output")
    if args.chart_count < 2 or args.iterations < 1:
        raise ValueError("need at least two charts and one iteration")
    if args.initializer == "dav2" and (args.dav2_charts is None) == (args.dav2_initializers is None):
        raise ValueError("DAV2 initializer requires exactly one of --dav2_charts/--dav2_initializers")
    if args.initializer == "moge3" and args.moge3_initializers is None:
        raise ValueError("MoGe-3 initializer requires --moge3_initializers")
    if args.allow_paired_stride2_diagnostic_alignment_adapter and args.allow_moge_reference_only_alignment_adapter:
        raise ValueError("alignment adapters are mutually exclusive")
    if (
        (args.allow_paired_stride2_diagnostic_alignment_adapter or args.allow_moge_reference_only_alignment_adapter)
        and args.comparison_domain is None
    ):
        raise ValueError(
            "paired stride-2 diagnostic adapter requires a pinned comparison domain"
        )
    if args.comparison_domain is not None and any(
        value is None
        for value in (
            args.expected_comparison_domain_content_sha256,
            args.frozen_submap_plan,
            args.expected_plan_content_sha256,
            args.expected_alignment_code_inventory_sha256,
        )
    ):
        raise ValueError(
            "paired alignment requires externally pinned comparison-domain and frozen-plan hashes"
        )
    if args.comparison_domain is not None and args.initializer == "dav2" and args.dav2_initializers is None:
        raise ValueError("strict paired DAV2 alignment requires sealed per-chart initializers")

    cameras = json.loads(args.cameras.read_text())
    upstream_authority = _validated_disjoint_source_authority(
        args.disjoint_upstream_authority, args.cameras, args.pointmaps_dir
    )
    if args.comparison_domain is not None and (
        upstream_authority is None
        or upstream_authority.get("artifact_type")
        != "goal_maplet_disjoint_chart_upstream_authority_v2"
        or not upstream_authority.get("physical_source_held_input_roots_disjoint")
    ):
        raise ValueError("paired comparison requires physically isolated v2 authority")
    all_names = [Path(path).name for path in cameras["filepaths"]]
    if len(set(all_names)) != len(all_names):
        raise ValueError("camera image names are not unique")
    camera_row = {name: row for row, name in enumerate(all_names)}
    frozen_plan = None
    if args.comparison_domain is not None:
        selected, frozen_plan = _strict_plan_selection(
            args.frozen_submap_plan,
            expected_plan_content_sha256=args.expected_plan_content_sha256,
            authority=upstream_authority,
            cameras=cameras,
            allow_paired_stride2_diagnostic_alignment_adapter=(
                args.allow_paired_stride2_diagnostic_alignment_adapter
            ),
            allow_moge_reference_only_alignment_adapter=(
                args.allow_moge_reference_only_alignment_adapter
            ),
        )
    else:
        selected = _selected_names(cameras, args.route, args.chart_count)
    c2w = np.asarray([cameras["cams2world"][camera_row[name]] for name in selected])
    focal_512 = np.asarray([cameras["focals"][camera_row[name]] for name in selected])
    filepaths = [Path(cameras["filepaths"][camera_row[name]]) for name in selected]

    dav2_manifest_rows = None
    if args.dav2_initializers is not None:
        manifest_path = args.dav2_initializers / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_claimed = manifest.pop("content_sha256", None)
        if manifest_claimed != _canonical_sha256(manifest):
            raise ValueError("DAV2 initializer manifest content hash differs")
        manifest["content_sha256"] = manifest_claimed
        if (
            manifest.get("artifact_type")
            != "goal_maplet_dav2_chart_initializer_run_v1"
            or upstream_authority is None
            or manifest.get("disjoint_upstream_authority_content_sha256")
            != upstream_authority["content_sha256"]
            or Path(manifest.get("source_only_mast3r_root", "")).resolve()
            != Path(upstream_authority["source"]["root"]).resolve()
            or Path(manifest.get("isolated_source_posed_colmap_root", "")).resolve()
            != Path(upstream_authority["isolated_source_input"]["root"]).resolve()
        ):
            raise ValueError("DAV2 initializer run is not bound to strict source authority")
        if manifest.get("source_only_mast3r_tree_sha256") != upstream_authority[
            "source"
        ]["tree_sha256"]:
            raise ValueError("DAV2 initializer source tree differs from strict authority")
        dav2_manifest_rows = _manifest_rows(manifest, "DAV2 initializer")
    moge_manifest_rows = None
    if args.moge3_initializers is not None:
        manifest_path = args.moge3_initializers / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_claimed = manifest.pop("content_sha256", None)
        if manifest_claimed != _canonical_sha256(manifest):
            raise ValueError("MoGe-3 initializer manifest content hash differs")
        manifest["content_sha256"] = manifest_claimed
        if (
            manifest.get("artifact_type")
            != "goal_maplet_moge3_chart_initializer_run_v2"
            or manifest.get("cameras_file_sha256") != _sha256(args.cameras)
            or manifest.get("uses_camera_pose") is not False
            or manifest.get("uses_query_or_ground_truth") is not False
        ):
            raise ValueError("MoGe-3 initializer run is not bound to source cameras")
        if upstream_authority is not None and sorted(
            manifest.get("allowed_routes", [])
        ) != sorted(upstream_authority["source"].get("routes", [])):
            raise ValueError("MoGe initializer route inventory differs from strict authority")
        moge_manifest_rows = _manifest_rows(manifest, "MoGe initializer")

    reference_depths = []
    reference_valid = []
    for name, pose in zip(selected, c2w):
        depth, valid = _load_reference_depth(args.pointmaps_dir / f"{Path(name).stem}.json", pose)
        reference_depths.append(depth)
        reference_valid.append(valid)
    reference_depths = np.stack(reference_depths)
    reference_valid = np.stack(reference_valid)

    if args.initializer == "moge3":
        initializer_points = []
        initializer_valid = []
        initializer_hashes = []
        for name in selected:
            path = args.moge3_initializers / f"{name}.npz"
            if (
                moge_manifest_rows is None
                or name not in moge_manifest_rows
                or moge_manifest_rows[name]["file_sha256"] != _sha256(path)
            ):
                raise ValueError("MoGe-3 initializer differs from run manifest")
            with np.load(path, allow_pickle=False) as data:
                points_camera = np.asarray(data["points_camera"], np.float32)
                valid = np.asarray(data["valid"], bool)
                metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("artifact_type") != "goal_maplet_moge3_chart_initializer_v2":
                raise ValueError("MoGe-3 initializer lacks the focal-correct v2 contract")
            metadata_claimed = metadata.pop("content_sha256", None)
            if metadata_claimed != _canonical_sha256(metadata):
                raise ValueError("MoGe-3 initializer content hash differs")
            metadata["content_sha256"] = metadata_claimed
            row = camera_row[name]
            source_path = Path(cameras["filepaths"][row])
            if (
                metadata.get("source_name") != name
                or metadata.get("camera_focal_canvas_width") != 512
                or float(metadata.get("camera_focal_canvas_px", float("nan")))
                != float(cameras["focals"][row])
                or metadata.get("source_image_file_sha256") != _sha256(source_path)
                or metadata.get("uses_camera_pose") is not False
                or metadata.get("uses_query_or_ground_truth") is not False
                or metadata_claimed != moge_manifest_rows[name].get("content_sha256")
            ):
                raise ValueError("MoGe-3 initializer camera binding differs")
            if points_camera.shape != (144, 256, 3) or valid.shape != (144, 256):
                raise ValueError("MoGe-3 initializer grid contract differs")
            points_camera = _nearest_valid_fill(points_camera, valid)
            world = points_camera @ np.asarray(cameras["cams2world"][row])[:3, :3].T
            world += np.asarray(cameras["cams2world"][row])[:3, 3]
            initializer_points.append(world)
            initializer_valid.append(valid)
            initializer_hashes.append(_sha256(path))
        initializer_points = np.stack(initializer_points)
        initializer_valid = np.stack(initializer_valid)
        source_scale_factor = None
    else:
        if args.dav2_initializers is not None:
            initializer_points = []
            initializer_valid = []
            initializer_hashes = []
            for name in selected:
                path = args.dav2_initializers / f"{name}.npz"
                if (
                    dav2_manifest_rows is None
                    or name not in dav2_manifest_rows
                    or dav2_manifest_rows[name]["file_sha256"] != _sha256(path)
                ):
                    raise ValueError("DAV2 initializer differs from run manifest")
                with np.load(path, allow_pickle=False) as data:
                    points_world = np.asarray(data["points_world"], np.float32)
                    valid = np.asarray(data["valid"], bool)
                    metadata = json.loads(str(data["metadata_json"].item()))
                if (
                    metadata.get("artifact_type") != "goal_maplet_dav2_chart_initializer_v1"
                    or metadata.get("source_name") != name
                    or metadata.get("disjoint_upstream_authority_content_sha256")
                    != upstream_authority["content_sha256"]
                ):
                    raise ValueError("DAV2 initializer binding differs")
                metadata_claimed = metadata.pop("content_sha256", None)
                if metadata_claimed != _canonical_sha256(metadata):
                    raise ValueError("DAV2 initializer content hash differs")
                metadata["content_sha256"] = metadata_claimed
                if (
                    metadata_claimed != dav2_manifest_rows[name].get("content_sha256")
                    or metadata.get("source_only_mast3r_root")
                    != str(Path(upstream_authority["source"]["root"]).resolve())
                    or metadata.get("uses_query_or_ground_truth") is not False
                    or points_world.shape != (144, 256, 3)
                    or valid.shape != (144, 256)
                ):
                    raise ValueError("DAV2 initializer semantic binding differs")
                initializer_points.append(_nearest_valid_fill(points_world, valid))
                initializer_valid.append(valid)
                initializer_hashes.append(_sha256(path))
            initializer_points = np.stack(initializer_points).astype(np.float32)
            initializer_valid = np.stack(initializer_valid)
            source_scale_factor = None
        else:
            with np.load(args.dav2_charts, allow_pickle=False) as data:
                aligned = np.asarray(data["pts"], np.float64)
                aligned_depth = np.asarray(data["depths"], np.float64)
                prior_depth = np.asarray(data["prior_depths"], np.float64)
                source_scale_factor = float(data["scale_factor"])
            lexical_names = sorted(all_names)
            if len(lexical_names) != len(aligned):
                raise ValueError("DAV2 charts and camera inventory differ")
            chart_row = {name: row for row, name in enumerate(lexical_names)}
            initializer_points = []
            initializer_valid = []
            for name in selected:
                row = chart_row[name]
                center_scaled = c2w[selected.index(name), :3, 3] * source_scale_factor
                ratio = prior_depth[row] / aligned_depth[row]
                initial_scaled = center_scaled + (aligned[row] - center_scaled) * ratio[..., None]
                initial_metric = initial_scaled / source_scale_factor
                initial_metric = cv2.resize(initial_metric, (256, 144), interpolation=cv2.INTER_AREA)
                valid = np.isfinite(initial_metric).all(2) & np.isfinite(
                    cv2.resize(prior_depth[row], (256, 144), interpolation=cv2.INTER_AREA)
                )
                initializer_points.append(_nearest_valid_fill(initial_metric, valid))
                initializer_valid.append(valid)
            initializer_points = np.stack(initializer_points).astype(np.float32)
            initializer_valid = np.stack(initializer_valid)
            initializer_hashes = [_sha256(args.dav2_charts)]

    initializer_points = np.asarray(initializer_points, dtype=np.float32)
    initializer_valid = np.asarray(initializer_valid, dtype=bool)
    valid = initializer_valid & reference_valid
    comparison_domain_content_sha256 = None
    comparison_arrays = None
    comparison_metadata = None
    if args.comparison_domain is not None:
        initializer_path = (
            args.dav2_initializers
            if args.initializer == "dav2" else args.moge3_initializers
        )
        comparison_arrays, comparison_metadata = _load_strict_comparison_domain(
            args.comparison_domain,
            expected_content_sha256=args.expected_comparison_domain_content_sha256,
            selected_names=selected,
            authority_path=args.disjoint_upstream_authority,
            authority=upstream_authority,
            plan_path=args.frozen_submap_plan,
            plan=frozen_plan,
            cameras_path=args.cameras,
            pointmaps_dir=args.pointmaps_dir,
            initializer=args.initializer,
            initializer_path=initializer_path,
            allow_paired_stride2_diagnostic_alignment_adapter=(
                args.allow_paired_stride2_diagnostic_alignment_adapter
            ),
            allow_moge_reference_only_alignment_adapter=(
                args.allow_moge_reference_only_alignment_adapter
            ),
        )
        comparison_valid = np.asarray(comparison_arrays["valid"], bool)
        if comparison_valid.shape != valid.shape:
            raise ValueError("comparison domain inventory/shape differs from selected charts")
        if np.any(comparison_valid & ~valid):
            raise ValueError("comparison domain includes invalid initializer/reference pixels")
        valid = comparison_valid
        comparison_domain_content_sha256 = comparison_metadata["content_sha256"]
    if np.any(valid.reshape(len(valid), -1).sum(1) < 1024):
        raise ValueError("a chart has insufficient joint initializer/reference support")
    rgb_small = []
    rgb_full = []
    for path in filepaths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_full.append(rgb)
        rgb_small.append(cv2.resize(rgb, (256, 144), interpolation=cv2.INTER_AREA))
    # All mutable lineage, inventory and common-domain checks happen before
    # importing the GPU alignment stack or creating an output directory.
    alignment_code_rows, alignment_code_hash = _alignment_code_inventory(
        args.matcha_repo
    )
    if (
        args.comparison_domain is not None
        and alignment_code_hash != args.expected_alignment_code_inventory_sha256
    ):
        raise ValueError("MAtCha alignment code differs from runner authority")
    depth_anything_repo = args.matcha_repo / "Depth-Anything-V2"
    # The vendored project intentionally uses a namespace package and has no
    # ``__init__.py``; ``dpt.py`` is the concrete module MAtCha imports.
    if not (depth_anything_repo / "depth_anything_v2" / "dpt.py").is_file():
        raise FileNotFoundError(
            "MAtCha Depth-Anything-V2 dependency is absent from the pinned repo"
        )
    # MAtCha imports ``depth_anything_v2`` as a top-level package.  Adding only
    # the MAtCha repository root works in an already-activated developer shell
    # but fails in a clean, explicitly selected environment.  Bind the vendored
    # dependency before importing any GPU module so strict CLI runs are
    # reproducible without ambient PYTHONPATH state.
    sys.path.insert(0, str(depth_anything_repo))
    sys.path.insert(0, str(args.matcha_repo))
    import torch

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda" or requested_device.index is None:
        raise ValueError("strict chart alignment requires an explicit cuda:<index> device")
    # MAtCha's GSCamera constructor still creates several tensors on the
    # process' current CUDA device.  Make that implicit device agree with the
    # explicit PointMap/reference device before any MAtCha camera is built.
    torch.cuda.set_device(requested_device)
    from matcha.dm_scene.cameras import (
        CamerasWrapper,
        create_gs_cameras_from_pointmap,
    )
    from matcha.dm_trainers.charts_alignment import align_charts_in_parallel
    from matcha.pointmap.base import PointMap

    matcha_img_paths = [str(path) for path in filepaths]
    internal_order, restore_official_order = _matcha_internal_lexical_permutations(
        selected, matcha_img_paths
    )
    internal_names = [selected[int(row)] for row in internal_order]
    internal_filepaths = [matcha_img_paths[int(row)] for row in internal_order]
    if internal_filepaths != sorted(matcha_img_paths):
        raise AssertionError("internal PointMap paths do not replay MAtCha lexical order")
    rgb_small_array = np.stack(rgb_small)[internal_order]
    rgb_full_array = np.stack(rgb_full)[internal_order]
    pointmap = PointMap(
        img_paths=internal_filepaths,
        images=rgb_small_array,
        original_images=rgb_full_array,
        focals=(focal_512[internal_order, None] * 0.5).astype(np.float32),
        poses=c2w[internal_order].astype(np.float32),
        points3d=initializer_points[internal_order],
        confidence=valid[internal_order].astype(np.float32),
        masks=valid[internal_order],
        device=args.device,
    )
    pointmap.move_everything_to_device(args.device)
    camera_wrapper = CamerasWrapper(
        create_gs_cameras_from_pointmap(
            pointmap,
            image_resolution=1,
            load_gt_images=True,
            max_img_size=1024,
            use_original_image_size=True,
            average_focal_distances=False,
            verbose=False,
        ),
        no_p3d_cameras=False,
    )
    actual_camera_names = [camera.image_name for camera in camera_wrapper.gs_cameras]
    if actual_camera_names != internal_names:
        raise ValueError("MAtCha camera order differs from synchronized PointMap order")
    scale_factor = float(5.0 / camera_wrapper.get_spatial_extent())
    reference = torch.as_tensor(
        reference_depths[internal_order] * scale_factor,
        dtype=torch.float32,
        device=args.device,
    )
    mask = torch.as_tensor(
        valid[internal_order], dtype=torch.bool, device=args.device
    )

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    output = align_charts_in_parallel(
        pointmap,
        reference_data=reference,
        masks=mask,
        rendering_size=1024,
        target_scale=5.0,
        n_iterations=args.iterations,
        use_gradient_loss=False,
        use_hessian_loss=False,
        use_normal_loss=True,
        use_curvature_loss=True,
        use_matching_loss=True,
        use_reprojection_loss=False,
        matching_thr_factor=0.05,
        matching_update_iters=None,
        use_confidence_in_matching_loss=False,
        weight_encodings_with_confidence=False,
        regularize_chart_encodings_norms=False,
        use_total_variation_on_depth_encodings=False,
        gradient_loss_weight=50.0,
        hessian_loss_weight=100.0,
        normal_loss_weight=4.0,
        curvature_loss_weight=1.0,
        matching_loss_weight=5.0,
        encodings_lr=0.01,
        mlp_lr=0.001,
        confidence_lr=0.001,
        lr_update_iters=[1000],
        lr_update_factor=0.1,
        verbose=False,
        return_training_losses=True,
        save_charts_data=False,
    )
    output_verts, output_depths, output_confs, training_losses = output
    elapsed = time.perf_counter() - started
    output_verts_np = output_verts.detach().cpu().numpy()[restore_official_order]
    output_depths_np = output_depths.detach().cpu().numpy()[restore_official_order]
    output_confs_np = output_confs.detach().cpu().numpy()[restore_official_order]
    initial_depths_np = np.stack([
        ((initializer_points[row] - c2w[row, :3, 3]) @ c2w[row, :3, :3])[..., 2]
        * scale_factor
        for row in range(len(selected))
    ]).astype(np.float32)
    arrays_path = args.output_dir / "charts_data.npz"
    chart_arrays = {
        "prior_depths": initial_depths_np,
        "depths": output_depths_np,
        "pts": output_verts_np,
        "confs": output_confs_np,
        "valid": valid,
        "chart_names": np.asarray(selected),
        "scale_factor": np.asarray(scale_factor),
        "training_losses": np.asarray(training_losses, np.float64),
    }
    if comparison_arrays is not None:
        chart_arrays["comparison_face_valid_stride4"] = comparison_arrays[
            "face_valid_stride4"
        ]
        chart_arrays["comparison_face_valid_stride8"] = comparison_arrays[
            "face_valid_stride8"
        ]
    np.savez_compressed(arrays_path, **chart_arrays)
    subset_cameras = {
        "filepaths": [str(path) for path in filepaths],
        "focals": focal_512.tolist(),
        "cams2world": c2w.tolist(),
    }
    cameras_path = args.output_dir / "cameras.json"
    cameras_path.write_text(json.dumps(subset_cameras, indent=2, sort_keys=True))
    manifest = {
        "artifact_type": "goal_maplet_masked_chart_alignment_gate_v1",
        "initializer": args.initializer,
        "route": args.route,
        "chart_names": selected,
        "chart_count": len(selected),
        "iterations": args.iterations,
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
        "route_clean": bool(upstream_authority),
        "strict_source_held_disjoint_upstream": bool(upstream_authority),
        "disjoint_upstream_authority_file_sha256": (
            _sha256(args.disjoint_upstream_authority)
            if args.disjoint_upstream_authority
            else None
        ),
        "disjoint_upstream_authority_content_sha256": (
            upstream_authority["content_sha256"] if upstream_authority else None
        ),
        "comparison_domain_file_sha256": (
            _sha256(args.comparison_domain) if args.comparison_domain else None
        ),
        "comparison_domain_content_sha256": comparison_domain_content_sha256,
        "paired_common_pixel_domain": bool(args.comparison_domain),
        "common_pixel_mask_sha256": (
            arrays_sha256({"valid": np.asarray(valid, bool)})
            if args.comparison_domain else None
        ),
        "common_face_mask_sha256": (
            {
                "stride4": arrays_sha256(
                    {"face_valid_stride4": comparison_arrays["face_valid_stride4"]}
                ),
                "stride8": arrays_sha256(
                    {"face_valid_stride8": comparison_arrays["face_valid_stride8"]}
                ),
            }
            if comparison_arrays is not None else None
        ),
        "common_face_inventory_consumed_by_alignment_optimizer": False,
        "common_face_inventory_deferred_to_explicit_atlas_export": bool(
            comparison_arrays is not None
        ),
        "frozen_submap_plan_file_sha256": (
            _sha256(args.frozen_submap_plan) if args.frozen_submap_plan else None
        ),
        "frozen_submap_plan_content_sha256": (
            frozen_plan.metadata["content_sha256"] if frozen_plan is not None else None
        ),
        "selected_chart_names_in_order_sha256": (
            frozen_plan.metadata["selected_chart_names_in_order_sha256"]
            if frozen_plan is not None else None
        ),
        "matcha_internal_chart_names": internal_names,
        "matcha_internal_lexical_order": internal_order.tolist(),
        "output_restored_to_frozen_plan_order": True,
        "selection_contract": (
            frozen_plan.metadata["alignment_runner_contract"]
            if frozen_plan is not None else "legacy_route_count_diagnostic"
        ),
        "alignment_code_inventory_sha256": alignment_code_hash,
        "alignment_code_inventory": alignment_code_rows,
        "alignment_runner_file_sha256": _sha256(Path(__file__)),
        "validity_contract": "initializer_valid AND MASt3R_confidence_gt_0.25; exact stencil erosion for normal/curvature; invalid matching source/target excluded",
        "valid_fraction_per_chart": valid.reshape(len(valid), -1).mean(1).tolist(),
        "initializer_file_sha256": initializer_hashes,
        "cameras_source_file_sha256": _sha256(args.cameras),
        "pointmap_file_sha256": {
            name: _sha256(args.pointmaps_dir / f"{Path(name).stem}.json") for name in selected
        },
        "charts_data_file_sha256": _sha256(arrays_path),
        "subset_cameras_file_sha256": _sha256(cameras_path),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "diagnostic_only": True,
        "production_eligible": False,
        "blockers": [
            *([] if upstream_authority else ["upstream_source_held_disjoint_not_proven"]),
            *([] if args.comparison_domain else ["DAV2_MoGe_support_domains_not_frozen_equal"]),
            "bounded_single_route_chart_gate",
            *(["alignment_iterations_below_MAtCha_default_1000"] if args.iterations < 1000 else []),
            "no_chart_family_canonicalization",
            "no_RADIO_UV_field",
        ],
    }
    if args.allow_paired_stride2_diagnostic_alignment_adapter:
        manifest.update(paired_stride2_diagnostic_alignment_metadata(frozen_plan))
        manifest.update(
            {
                "diagnostic_only": True,
                "production_eligible": False,
                "diagnostic_alignment_source_plan_artifact_type": (
                    frozen_plan.metadata["artifact_type"]
                ),
                "diagnostic_alignment_source_plan_representation": (
                    frozen_plan.metadata["representation"]
                ),
                "optimizer_pixel_domain_only": True,
                "paired_stride2_topology_embedded_in_comparison_domain": False,
                "full_gate_or_exporter_consumption_eligible": False,
                "full_gate_exact_topology_sealer_eligible": False,
                "blockers": [
                    *manifest["blockers"],
                    "paired_stride2_comparison_topology_not_model_neutral",
                    "diagnostic_alignment_adapter_nonpromotion_contract",
                ],
            }
        )
    elif args.allow_moge_reference_only_alignment_adapter:
        manifest.update(moge_reference_control_alignment_metadata(frozen_plan))
        manifest.update(
            {
                "diagnostic_only": True,
                "production_eligible": False,
                "diagnostic_alignment_source_plan_artifact_type": frozen_plan.metadata["artifact_type"],
                "diagnostic_alignment_source_plan_representation": frozen_plan.metadata["representation"],
                "optimizer_pixel_domain_only": True,
                "full_gate_or_exporter_consumption_eligible": False,
                "full_gate_exact_topology_sealer_eligible": False,
                "blockers": [
                    *manifest["blockers"],
                    "MoGe3_specific_topology_not_model_neutral",
                    "diagnostic_alignment_adapter_nonpromotion_contract",
                ],
            }
        )
    manifest["content_sha256"] = _canonical_sha256(manifest)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
