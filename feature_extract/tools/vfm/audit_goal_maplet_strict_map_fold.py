"""Audit one strict fold from route-clean inputs through its physical map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _load(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def audit_strict_fold(fold_dir: Path) -> dict[str, object]:
    fold_dir = Path(fold_dir)
    paths = {
        "inputs": fold_dir / "strict_inputs" / "strict_mapping_inputs.json",
        "colmap": fold_dir / "posed_colmap" / "fold_colmap_dataset.json",
        "matcha": fold_dir / "strict_geometry" / "strict_matcha_run_manifest.json",
        "bootstrap_map": fold_dir / "bootstrap_map" / "region_summary.json",
        "bootstrap_surface": fold_dir / "bootstrap_surface" / "surface_map_summary.json",
        "surface_mapper": fold_dir / "surface_mapper.json",
        "final_map": fold_dir / "final_map" / "region_summary.json",
        "final_surface": fold_dir / "final_surface" / "surface_map_summary.json",
        "physical_map": fold_dir / "physical_map_audit.json",
    }
    reports = {name: _load(path) for name, path in paths.items()}
    inputs = reports["inputs"]
    mapping_count = int(inputs["mapping_image_count"])
    mapping_routes = set(inputs["mapping_trajectories"])
    held_routes = set(inputs["held_query_trajectories"])
    if bool(inputs["contains_held_query"]) or bool(inputs["contains_official_test"]):
        raise ValueError("strict mapping inputs contain forbidden images")

    colmap = reports["colmap"]
    matcha = reports["matcha"]
    if (
        int(colmap["mapping_image_count"]) != mapping_count
        or int(colmap["dense_supervision_image_count"]) != mapping_count
        or not bool(colmap["dense_supervision_uses_all_mapping_images"])
        or bool(colmap["selected_contains_held_query"])
        or int(matcha["mapping_image_count"]) != mapping_count
        or not bool(matcha["dense_supervision_uses_all_mapping_images"])
        or set(matcha["held_query_trajectories"]) != held_routes
    ):
        raise ValueError("strict MAtCha split contract mismatch")
    patch_contract = matcha.get("matcha_tracked_patch_contract", {})
    patch_path = Path(str(patch_contract.get("path", "")))
    patch_sha = str(patch_contract.get("sha256", ""))
    if (
        not patch_path.is_file()
        or file_sha256(patch_path) != patch_sha
        or str(patch_contract.get("applied_diff_sha256", "")) != patch_sha
        or str(matcha.get("matcha_tracked_patch_sha256", "")) != patch_sha
        or not bool(matcha.get("matcha_untracked_files_are_not_runtime_inputs", False))
    ):
        raise ValueError("strict MAtCha source patch contract is absent or changed")
    gaussian_outputs = matcha["stages"]["gaussians"]["outputs"]
    if len(gaussian_outputs) != 1:
        raise ValueError("strict MAtCha manifest has an ambiguous final PLY")
    gaussian_ply = Path(gaussian_outputs[0]["path"])
    if file_sha256(gaussian_ply) != str(gaussian_outputs[0]["sha256"]):
        raise ValueError("strict MAtCha PLY hash changed")

    for stage in ("bootstrap_map", "final_map"):
        report = reports[stage]
        if (
            int(report["view_count"]) != mapping_count
            or not bool(report["canonical_vfm_2dgs"])
            or not bool(report["auto_virtual_cell"]["disabled"])
            or not bool(report["inputs"]["require_camera_for_every_view"])
            or int(report["inputs"]["missing_camera_view_count"]) != 0
            or Path(report["inputs"]["gaussian_ply"]).resolve()
            != gaussian_ply.resolve()
        ):
            raise ValueError(f"{stage} is not bound to strict fold geometry")
    mapper = reports["surface_mapper"]
    split = mapper["split"]
    if (
        mapper["config"]["checkpoint_protocol"] != "fixed_epoch_no_selection"
        or mapper["best_validation"] is not None
        or set(split["training_trajectory_ids"]) != mapping_routes
        or not held_routes.issubset(set(split["strict_holdout_trajectory_ids"]))
    ):
        raise ValueError("strict fold mapper split or checkpoint protocol mismatch")
    final_map = reports["final_map"]
    final_surface = reports["final_surface"]
    if (
        not bool(final_map["inputs"]["full_map_mapper_applied"])
        or not bool(final_surface["inputs"]["full_map_mapper_applied"])
        or Path(final_map["inputs"]["surface_maplet_mapper_checkpoint"]).resolve()
        != (fold_dir / "surface_mapper.pt").resolve()
        or not bool(final_surface["inputs"]["require_camera_for_every_view"])
        or int(final_surface["inputs"]["missing_camera_view_count"]) != 0
        or not bool(
            reports["bootstrap_surface"]["inputs"]["require_camera_for_every_view"]
        )
        or int(
            reports["bootstrap_surface"]["inputs"]["missing_camera_view_count"]
        ) != 0
    ):
        raise ValueError("strict final map did not use the fold mapper")
    physical = reports["physical_map"]
    if (
        physical.get("clean_primitive_selection")
        != "all_declared_surface_elements"
        or Path(physical["inputs"]["surface_elements"]).resolve()
        != (fold_dir / "bootstrap_map" / "surface_elements.npz").resolve()
        or Path(physical["inputs"]["region_map"]).resolve()
        != (fold_dir / "final_map" / "region_map.npz").resolve()
    ):
        raise ValueError("physical hierarchy is not bound to strict surface geometry")
    return {
        "artifact_type": "goal_maplet_strict_map_fold_audit_v1",
        "fold_id": str(inputs["fold_id"]),
        "mapping_image_count": mapping_count,
        "mapping_trajectories": sorted(mapping_routes),
        "held_query_trajectories": sorted(held_routes),
        "chart_count": int(matcha["chart_count"]),
        "gaussian_iterations": int(matcha["gaussian_iterations"]),
        "matcha_commit": str(matcha["matcha_commit"]),
        "matcha_source_patch_sha256": patch_sha,
        "gaussian_ply": str(gaussian_ply),
        "gaussian_ply_sha256": file_sha256(gaussian_ply),
        "physical_map_ready": bool(physical["physical_map_ready"]),
        "route_clean_end_to_end": True,
        "hashes": {name: file_sha256(path) for name, path in paths.items()},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite strict fold audit")
    result = audit_strict_fold(Path(args.fold_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
