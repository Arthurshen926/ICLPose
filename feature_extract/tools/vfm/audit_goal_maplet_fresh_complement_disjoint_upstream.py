"""Seal a fresh complement-held, physically disjoint MASt3R authority.

This is deliberately separate from the original pose-only authority signer:
the source window is unchanged, while held cameras are a pre-geometry
complement of the earlier diagnostic inventory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from feature_extract.tools.vfm.audit_goal_maplet_disjoint_chart_upstream import (
    _canonical_sha256,
    _expected_names,
    _official_camera_rows,
    _sha256,
    _validate_run,
)


def _load_sealed_json(
    path: Path,
    *,
    expected_schema: str,
    expected_content_sha256: str | None = None,
) -> dict:
    value = json.loads(path.read_text())
    claimed = value.pop("content_sha256", None)
    if claimed != _canonical_sha256(value):
        raise ValueError(f"{expected_schema} content hash differs")
    value["content_sha256"] = claimed
    if value.get("artifact_type") != expected_schema:
        raise ValueError(f"wrong {expected_schema} schema")
    if expected_content_sha256 is not None and claimed != expected_content_sha256:
        raise ValueError(f"{expected_schema} differs from external authority")
    return value


def _values_after(command: list[str], flag: str) -> list[str]:
    if flag not in command:
        raise ValueError(f"frozen command lacks {flag}")
    offset = command.index(flag) + 1
    values = []
    while offset < len(command) and not command[offset].startswith("--"):
        values.append(command[offset])
        offset += 1
    return values


def _one_after(command: list[str], flag: str) -> str:
    values = _values_after(command, flag)
    if len(values) != 1:
        raise ValueError(f"frozen command has ambiguous {flag}")
    return values[0]


def _replay_isolated_role(role: dict, posed_colmap: Path) -> None:
    root = Path(role["root"]).resolve()
    names = role["ordered_names"]
    image_files = sorted(path.name for path in (root / "images").iterdir() if path.is_file())
    if image_files != sorted(names) or role.get("image_count") != len(names):
        raise ValueError("isolated role image inventory differs from manifest")
    rows = role.get("rows")
    if not isinstance(rows, list) or [row.get("name") for row in rows] != names:
        raise ValueError("isolated role row order differs from manifest")
    for row in rows:
        name = row["name"]
        if (
            row.get("source_image_file_sha256")
            != _sha256(posed_colmap / "images" / name)
            or row.get("isolated_image_file_sha256")
            != _sha256(root / "images" / name)
        ):
            raise ValueError("isolated role image bytes differ from manifest")
    for key, filename in (
        ("cameras_file_sha256", "cameras.bin"),
        ("images_file_sha256", "images.bin"),
        ("points3D_file_sha256", "points3D.bin"),
    ):
        if role.get(key) != _sha256(root / "sparse" / "0" / filename):
            raise ValueError("isolated role sparse bytes differ from manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed_colmap", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--held_root", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--isolated_inputs", type=Path, required=True)
    parser.add_argument("--preexecution_contract", type=Path, required=True)
    parser.add_argument("--complement_plan", type=Path, required=True)
    parser.add_argument("--expected_complement_plan_content_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fresh disjoint authority")

    complement = _load_sealed_json(
        args.complement_plan,
        expected_schema="goal_maplet_complement_held_preexecution_plan_v1",
        expected_content_sha256=args.expected_complement_plan_content_sha256,
    )
    required_complement_true = (
        "fresh_held_disjoint_from_superseded_diagnostic",
        "fresh_held_subset_of_pose_only_eligible_inventory",
        "selection_frozen_before_new_held_geometry",
    )
    required_complement_false = (
        "new_held_geometry_used_for_selection",
        "new_held_rgb_numeric_fields_used_for_selection",
        "new_held_pointmaps_opened",
        "held_camera_fields_materialized_by_this_planner",
        "source_selection_changed_by_fresh_held_choice",
        "new_rgb_or_pointmap_may_be_opened_before_contract_freeze",
        "uses_query_or_ground_truth",
    )
    if (
        any(complement.get(key) is not True for key in required_complement_true)
        or any(complement.get(key) is not False for key in required_complement_false)
    ):
        raise ValueError("fresh complement was not frozen before geometry")
    source_indices = [int(value) for value in complement["source_indices"]]
    held_indices = [int(value) for value in complement["fresh_held_indices"]]
    source_names = [str(value) for value in complement["source_ordered_names"]]
    held_names = [str(value) for value in complement["fresh_held_ordered_names"]]
    if set(held_indices) & set(complement["superseded_diagnostic_held_indices"]):
        raise ValueError("fresh complement reuses a diagnostic held camera")

    isolation_command = complement["physical_isolation_input_builder_command"]
    contract_command = complement["sfm_preexecution_contract_builder_command"]
    if (
        Path(_one_after(isolation_command, "--posed_colmap")).resolve()
        != args.posed_colmap.resolve()
        or [int(value) for value in _values_after(isolation_command, "--source_indices")]
        != source_indices
        or [int(value) for value in _values_after(isolation_command, "--held_indices")]
        != held_indices
        or Path(_one_after(contract_command, "--matcha_repo")).resolve()
        != args.matcha_repo.resolve()
        or Path(complement["planned_matcha_repo"]).resolve()
        != args.matcha_repo.resolve()
    ):
        raise ValueError("executed paths or indices differ from complement command seal")
    source_routes = set(_values_after(isolation_command, "--source_routes"))
    held_routes = set(_values_after(isolation_command, "--held_routes"))
    forbidden = set(_values_after(isolation_command, "--forbidden_routes"))
    if source_routes & held_routes or (source_routes | held_routes) & forbidden:
        raise ValueError("source/held/forbidden route sets overlap")

    isolated = _load_sealed_json(
        args.isolated_inputs,
        expected_schema="goal_maplet_isolated_chart_sfm_inputs_v1",
    )
    if (
        Path(_one_after(isolation_command, "--output_root")).resolve()
        != args.isolated_inputs.parent.resolve()
        or Path(isolated["original_posed_colmap_root"]).resolve()
        != args.posed_colmap.resolve()
        or isolated["original_cameras_file_sha256"]
        != _sha256(args.posed_colmap / "sparse" / "0" / "cameras.bin")
        or isolated["original_images_file_sha256"]
        != _sha256(args.posed_colmap / "sparse" / "0" / "images.bin")
        or isolated["source"]["indices_in_original_lexical_inventory"] != source_indices
        or isolated["held"]["indices_in_original_lexical_inventory"] != held_indices
        or isolated["source"]["ordered_names"] != source_names
        or isolated["held"]["ordered_names"] != held_names
        or set(isolated["source"]["routes"]) != source_routes
        or set(isolated["held"]["routes"]) != held_routes
        or set(isolated["forbidden_routes"]) != forbidden
        or isolated.get("physical_image_input_roots_disjoint") is not True
        or isolated.get("source_held_image_disjoint") is not True
        or isolated.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError("isolated inputs differ from complement seal")
    if _expected_names(args.posed_colmap / "images", source_indices) != source_names:
        raise ValueError("source global indices differ from posed-COLMAP lexical inventory")
    if _expected_names(args.posed_colmap / "images", held_indices) != held_names:
        raise ValueError("held global indices differ from posed-COLMAP lexical inventory")
    _replay_isolated_role(isolated["source"], args.posed_colmap)
    _replay_isolated_role(isolated["held"], args.posed_colmap)

    preexecution = _load_sealed_json(
        args.preexecution_contract,
        expected_schema="goal_maplet_chart_sfm_preexecution_contract_v2",
    )
    if (
        Path(_one_after(contract_command, "--isolated_inputs")).resolve()
        != args.isolated_inputs.resolve()
        or Path(_one_after(contract_command, "--output")).resolve()
        != args.preexecution_contract.resolve()
        or preexecution["isolated_inputs_file_sha256"] != _sha256(args.isolated_inputs)
        or preexecution["isolated_inputs_content_sha256"] != isolated["content_sha256"]
        or Path(preexecution["matcha_repo"]).resolve() != args.matcha_repo.resolve()
        or preexecution.get("all_isolated_images_explicitly_indexed") is not True
        or preexecution.get("source_output_absent_at_freeze") is not True
        or preexecution.get("held_output_absent_at_freeze") is not True
        or preexecution.get("uses_query_or_ground_truth") is not False
    ):
        raise ValueError("preexecution contract differs from complement/isolated seal")
    for role, expected_root, image_count in (
        ("source", args.source_root, len(source_names)),
        ("held", args.held_root, len(held_names)),
    ):
        command = preexecution[f"{role}_command"]
        if (
            Path(_one_after(command, "--scene_path")).resolve()
            != Path(isolated[role]["root"]).resolve()
            or Path(_one_after(command, "--output_dir")).resolve()
            != expected_root.resolve()
            or _values_after(command, "--image_idx")
            != [str(value) for value in range(image_count)]
        ):
            raise ValueError(f"preexecution {role} command was not fully replayed")

    official = _official_camera_rows(args.posed_colmap)
    source = _validate_run(args.source_root, source_names, source_routes, official)
    held = _validate_run(args.held_root, held_names, held_routes, official)
    source_files = {
        "mast3r/run_mast3r.py": args.matcha_repo / "mast3r" / "run_mast3r.py",
        "mast3r/mast3r/cloud_opt/sparse_ga.py": args.matcha_repo
        / "mast3r"
        / "mast3r"
        / "cloud_opt"
        / "sparse_ga.py",
        "metric_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "retrieval_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    }
    observed_source_hashes = {name: _sha256(path) for name, path in source_files.items()}
    if preexecution["source_file_sha256"] != observed_source_hashes:
        raise ValueError("MASt3R code/checkpoint bytes changed after preexecution freeze")

    report = {
        "artifact_type": "goal_maplet_disjoint_chart_upstream_authority_v2",
        "posed_colmap_root": str(args.posed_colmap.resolve()),
        "posed_colmap_cameras_file_sha256": _sha256(
            args.posed_colmap / "sparse" / "0" / "cameras.bin"
        ),
        "posed_colmap_images_file_sha256": _sha256(
            args.posed_colmap / "sparse" / "0" / "images.bin"
        ),
        "source_indices": source_indices,
        "held_indices": held_indices,
        "source": source,
        "held": held,
        "source_held_image_disjoint": True,
        "source_held_route_disjoint": True,
        "forbidden_routes": sorted(forbidden),
        "forbidden_routes_opened": False,
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
        "upstream_source_file_sha256": observed_source_hashes,
        "isolated_inputs_file_sha256": _sha256(args.isolated_inputs),
        "isolated_inputs_content_sha256": isolated["content_sha256"],
        "isolated_source_input": {
            key: isolated["source"][key]
            for key in (
                "root",
                "image_count",
                "ordered_names",
                "cameras_file_sha256",
                "images_file_sha256",
                "points3D_file_sha256",
            )
        },
        "isolated_held_input": {
            key: isolated["held"][key]
            for key in (
                "root",
                "image_count",
                "ordered_names",
                "cameras_file_sha256",
                "images_file_sha256",
                "points3D_file_sha256",
            )
        },
        "preexecution_contract_file_sha256": _sha256(args.preexecution_contract),
        "preexecution_contract_content_sha256": preexecution["content_sha256"],
        "complement_held_preexecution_plan_file_sha256": _sha256(
            args.complement_plan
        ),
        "complement_held_preexecution_plan_content_sha256": complement[
            "content_sha256"
        ],
        "intended_source_chart_plan_content_sha256": complement[
            "intended_source_chart_plan_content_sha256"
        ],
        "fresh_held_selection_frozen_before_geometry": True,
        "fresh_held_disjoint_from_superseded_diagnostic": True,
        "complete_source_held_output_inventory_replayed": True,
        "source_held_tree_bytes_replayed": True,
        "source_held_pose_and_intrinsics_replayed": True,
        "source_process_exit_code": 0,
        "held_process_exit_code": 0,
        "physical_source_held_input_roots_disjoint": True,
        "strict_disjoint_upstream": True,
        "production_eligible": False,
        "diagnostic_semantics": (
            "fresh-complement-source/held-disjoint_mapping_geometry_gate_not_sensor_depth_GT"
        ),
    }
    report["content_sha256"] = _canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "file_sha256": _sha256(args.output),
                "content_sha256": report["content_sha256"],
                "source_tree_sha256": source["tree_sha256"],
                "held_tree_sha256": held["tree_sha256"],
                "source_image_count": source["image_count"],
                "held_image_count": held["image_count"],
                "source_max_pose_replay_abs_error": source[
                    "max_pose_replay_abs_error"
                ],
                "held_max_pose_replay_abs_error": held[
                    "max_pose_replay_abs_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
