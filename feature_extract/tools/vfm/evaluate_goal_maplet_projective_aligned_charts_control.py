"""Evaluate control-only MAtCha charts on a sealed exact projective authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    _surface_vertex_normals,
    evaluate_projective_exact_face_seam_geometry,
)


def _arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must have NAME=ALIGNMENT_DIR form")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("arm must have NAME=ALIGNMENT_DIR form")
    return name, Path(path)


def _load_aligned_vertices(
    root: Path, authority: ProjectiveExactFaceSeamAuthority
) -> np.ndarray:
    manifest_path = root / "manifest.json"
    charts_path = root / "charts_data.npz"
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("content_sha256", None)
    if claimed != canonical_json_sha256(manifest):
        raise ValueError(f"{root}: alignment manifest content hash differs")
    manifest["content_sha256"] = claimed
    moge_reference = authority.metadata.get("moge_reference_only_topology") is True
    required = {
        "artifact_type": "goal_maplet_masked_chart_alignment_gate_v1",
        "chart_count": len(authority.chart_names),
        "iterations": 1000,
        "system_control_only": True,
        "diagnostic_only": True,
        "diagnostic_alignment_adapter_used": True,
        "paired_stride2_densification_diagnostic": not moge_reference,
        "comparison_inventory_eligible": False,
        "promotion_eligible": False,
        "final_model_neutral_map_topology_eligible": False,
        "full_gate_or_exporter_consumption_eligible": False,
        "uses_query_or_ground_truth": False,
    }
    if moge_reference:
        required.update(
            {
                "moge_reference_only_topology": True,
                "dav2_geometry_consumed": False,
                "paired_initializer_geometry_encoded_upstream": False,
                "diagnostic_alignment_adapter_mode": "moge_reference_only_source_geometry_control_v1",
            }
        )
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{root}: alignment manifest {key} differs")
    names = authority.chart_names.astype(str).tolist()
    if manifest.get("chart_names") != names:
        raise ValueError(f"{root}: alignment chart order differs")
    if manifest.get("charts_data_file_sha256") != file_sha256(charts_path):
        raise ValueError(f"{root}: aligned charts bytes differ from manifest")

    with np.load(charts_path, allow_pickle=False) as data:
        if not np.array_equal(data["chart_names"].astype(str), authority.chart_names.astype(str)):
            raise ValueError(f"{root}: charts_data order differs")
        dense = np.asarray(data["pts"], np.float64)
        scale = float(data["scale_factor"])
        if dense.shape != authority.reference_points_world_dense.shape:
            raise ValueError(f"{root}: aligned dense point shape differs")
        if not np.isfinite(dense).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{root}: aligned metric geometry is invalid")
        dense /= scale
    vertices = np.empty_like(authority.reference_vertices_world)
    height, width = authority.common_valid.shape[1:]
    for chart in range(len(authority.chart_names)):
        lo, hi = map(int, authority.chart_vertex_offsets[chart : chart + 2])
        pixel = authority.sampled_vertex_pixel_indices[lo:hi]
        vertices[lo:hi] = dense[chart].reshape(height * width, 3)[pixel]
    return vertices


def _distortion(
    authority: ProjectiveExactFaceSeamAuthority,
    aligned: np.ndarray,
) -> dict[str, object]:
    initial = np.asarray(authority.reference_vertices_world, np.float64)
    faces = np.asarray(authority.faces, np.int64)
    tri0 = initial[faces]
    tri1 = aligned[faces]
    cross0 = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
    cross1 = np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0])
    length0 = np.linalg.norm(cross0, axis=1)
    length1 = np.linalg.norm(cross1, axis=1)
    if np.any(length0 <= 1e-12) or np.any(~np.isfinite(length1)):
        raise ValueError("source distortion topology is invalid")
    raw_ratio = length1 / length0
    global_area_scale = float(np.median(raw_ratio))
    if not np.isfinite(global_area_scale) or global_area_scale <= 1e-12:
        raise ValueError("aligned control has no robust global scale")
    ratio = raw_ratio / global_area_scale
    normal0 = cross0 / length0[:, None]
    normal1 = cross1 / np.maximum(length1[:, None], 1e-15)
    face_flip = np.sum(normal0 * normal1, axis=1) < 0
    chart_ratio = []
    for chart in range(len(authority.chart_names)):
        lo, hi = map(int, authority.chart_face_offsets[chart : chart + 2])
        chart_ratio.append(
            float(np.sum(length1[lo:hi]) / np.sum(length0[lo:hi]))
            / global_area_scale
        )

    width = authority.common_valid.shape[2]
    pixel = authority.sampled_vertex_pixel_indices.astype(np.int64)
    uv = np.column_stack((pixel % width, pixel // width)).astype(np.float64)
    singular_ratios = []
    global_linear_scale = float(np.sqrt(global_area_scale))
    for face in faces:
        uv_edge = np.stack(
            (uv[face[1]] - uv[face[0]], uv[face[2]] - uv[face[0]]),
            axis=1,
        )
        if abs(float(np.linalg.det(uv_edge))) <= 1e-12:
            raise ValueError("source distortion UV face is singular")
        inverse = np.linalg.inv(uv_edge)
        edge0 = np.stack(
            (initial[face[1]] - initial[face[0]], initial[face[2]] - initial[face[0]]),
            axis=1,
        )
        edge1 = np.stack(
            (aligned[face[1]] - aligned[face[0]], aligned[face[2]] - aligned[face[0]]),
            axis=1,
        )
        singular0 = np.linalg.svd(edge0 @ inverse, compute_uv=False)
        singular1 = np.linalg.svd(edge1 @ inverse, compute_uv=False)
        if np.any(singular0 <= 1e-12):
            raise ValueError("source distortion initial Jacobian is singular")
        singular_ratios.extend(
            (singular1 / singular0 / global_linear_scale).tolist()
        )
    singular_ratios = np.asarray(singular_ratios, np.float64)
    vertex_normal0 = _surface_vertex_normals(initial, faces)
    vertex_normal1 = _surface_vertex_normals(aligned, faces)
    vertex_flip = np.sum(vertex_normal0 * vertex_normal1, axis=1) < 0
    metrics = {
        "estimated_global_linear_scale": global_linear_scale,
        "face_area_ratio_p05": float(np.quantile(ratio, 0.05)),
        "face_area_ratio_median": float(np.median(ratio)),
        "face_area_ratio_p95": float(np.quantile(ratio, 0.95)),
        "face_collapse_fraction": float(np.mean(ratio < 0.25)),
        "face_expansion_fraction": float(np.mean(ratio > 4.0)),
        "face_flip_fraction": float(np.mean(face_flip)),
        "vertex_normal_flip_fraction": float(np.mean(vertex_flip)),
        "chart_area_ratio_min": float(np.min(chart_ratio)),
        "chart_area_ratio_median": float(np.median(chart_ratio)),
        "chart_area_ratio_max": float(np.max(chart_ratio)),
        "chart_area_ratios_in_official_order": chart_ratio,
        "jacobian_singular_ratio_p05": float(np.quantile(singular_ratios, 0.05)),
        "jacobian_singular_ratio_median": float(np.median(singular_ratios)),
        "jacobian_singular_ratio_p95": float(np.quantile(singular_ratios, 0.95)),
    }
    failures = []
    checks = (
        (metrics["chart_area_ratio_min"] >= 0.80, "chart_area_ratio_min"),
        (metrics["chart_area_ratio_max"] <= 1.20, "chart_area_ratio_max"),
        (metrics["face_collapse_fraction"] <= 0.01, "face_collapse_fraction"),
        (metrics["face_expansion_fraction"] <= 0.01, "face_expansion_fraction"),
        (metrics["face_flip_fraction"] <= 0.001, "face_flip_fraction"),
        (metrics["vertex_normal_flip_fraction"] <= 0.001, "vertex_normal_flip_fraction"),
        (metrics["jacobian_singular_ratio_p05"] >= 0.50, "jacobian_singular_ratio_p05"),
        (metrics["jacobian_singular_ratio_p95"] <= 2.00, "jacobian_singular_ratio_p95"),
    )
    failures.extend(label for passed, label in checks if not passed)
    metrics["thresholds"] = {
        "chart_area_ratio": [0.80, 1.20],
        "maximum_face_collapse_fraction": 0.01,
        "maximum_face_expansion_fraction": 0.01,
        "maximum_face_flip_fraction": 0.001,
        "maximum_vertex_normal_flip_fraction": 0.001,
        "jacobian_singular_ratio": [0.50, 2.00],
    }
    metrics["failures"] = failures
    metrics["decision"] = "GO" if not failures else "KILL"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_arm, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite projective alignment report")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if (
        authority.metadata.get("system_control_only") is not True
        or authority.metadata.get("promotion_eligible") is not False
        or authority.metadata.get("final_model_neutral_map_topology_eligible")
        is not False
    ):
        raise ValueError("selected projective authority is not control-only")
    arms = {name: _load_aligned_vertices(root, authority) for name, root in args.arm}
    if len(arms) != len(args.arm):
        raise ValueError("aligned arm names are duplicated")
    report = evaluate_projective_exact_face_seam_geometry(authority, arms)
    distortion = {name: _distortion(authority, vertices) for name, vertices in arms.items()}
    for name, row in report["arms"].items():
        row["distortion"] = distortion[name]
        row["source_geometry_decision"] = (
            "GO"
            if row["composite_decision"] == "GO"
            and distortion[name]["decision"] == "GO"
            else "KILL"
        )
    report.update(
        {
            "authority_file_sha256": file_sha256(args.authority),
            "aligned_arm_manifest_file_sha256": {
                name: file_sha256(root / "manifest.json")
                for name, root in args.arm
            },
            "system_control_only": True,
            "promotion_eligible": False,
            "production_eligible": False,
        }
    )
    report.pop("content_sha256", None)
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_sha256": file_sha256(args.output),
                "content_sha256": report["content_sha256"],
                "m0_surface": report["m0_source_reference"]["formal_decision"],
                "arms": {
                    name: {
                        "surface": row["formal_decision"],
                        "material_weld": row["material_weld_decision"],
                        "composite": row["composite_decision"],
                        "distortion": row["distortion"]["decision"],
                        "source_geometry": row["source_geometry_decision"],
                        "surface_edges": row["edge_formal_valid_count"],
                        "material_edges": row["edge_material_weld_valid_count"],
                        "composite_edges": row["edge_composite_valid_count"],
                    }
                    for name, row in report["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
