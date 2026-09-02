"""Use an existing HLoc pose only when the planar pose has low support.

This is a deliberately narrow hybrid: the frozen RADIO/SIFT planar pose stays
authoritative unless its already-defined inlier ratio is below the planar
pipeline's maximum-primary-ratio boundary.  HLoc is NetVLAD retrieval,
SuperPoint/SuperGlue matching and SfM-PnP; this tool does not run or consume
LoFTR.  It never opens query poses, query depth, or the HLoc pickle log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


PLANE_KEYS = (
    "names", "pose_w2c", "usable", "selected_branch", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
)


def _plane_name_to_hloc_name(name: str) -> str:
    parts = str(name).split("__")
    if len(parts) != 2 or not parts[0].startswith("seq") or not parts[1].endswith(".png.npz"):
        raise ValueError(f"unexpected planar query name: {name}")
    return f"{parts[0]}/{parts[1][:-4]}"


def _parse_hloc_results(path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 8:
            raise ValueError(f"invalid HLoc result row {line_number}")
        name = str(tokens[0])
        if name in poses:
            raise ValueError(f"duplicate HLoc result name: {name}")
        values = np.asarray([float(value) for value in tokens[1:]], np.float64)
        if not np.all(np.isfinite(values)) or float(np.linalg.norm(values[:4])) < 1e-12:
            raise ValueError(f"non-finite HLoc pose: {name}")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = qvec_to_rotmat(values[:4])
        pose[:3, 3] = values[4:]
        poses[name] = pose
    if not poses:
        raise ValueError("HLoc result file is empty")
    return poses


def _read_first_column(path: Path) -> list[str]:
    rows = []
    for line in path.read_text().splitlines():
        tokens = line.split()
        if tokens:
            rows.append(str(tokens[0]))
    return rows


def _load_plane(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
    if (
        metadata.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_radio_sift_agreement_midpoint_v1"
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
        or any(key not in complete for key in PLANE_KEYS)
    ):
        raise ValueError("planar pose inventory is not strict and frozen")
    count = len(complete["names"])
    if (
        complete["pose_w2c"].shape != (count, 4, 4)
        or len(set(complete["names"].astype(str).tolist())) != count
    ):
        raise ValueError("planar pose inventory shape or names differ")
    return complete, metadata


def _select_hloc_fallback(
    plane_usable: np.ndarray,
    plane_inlier_ratio: np.ndarray,
    maximum_plane_inlier_ratio: float,
) -> np.ndarray:
    usable = np.asarray(plane_usable, bool)
    ratios = np.asarray(plane_inlier_ratio, np.float64)
    if usable.shape != ratios.shape or np.any(~np.isfinite(ratios)):
        raise ValueError("planar usability and inlier-ratio arrays differ")
    return (~usable) | (ratios < float(maximum_plane_inlier_ratio))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_pose_inventory", type=Path, required=True)
    parser.add_argument("--hloc_results", type=Path, required=True)
    parser.add_argument("--hloc_sfm_model", type=Path, required=True)
    parser.add_argument("--hloc_query_pairs", type=Path, required=True)
    parser.add_argument("--hloc_query_intrinsics", type=Path, required=True)
    parser.add_argument("--hloc_global_features", type=Path, required=True)
    parser.add_argument("--hloc_local_features", type=Path, required=True)
    parser.add_argument("--hloc_query_matches", type=Path, required=True)
    parser.add_argument("--maximum_plane_inlier_ratio", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite HLoc low-support fusion")
    if not 0.0 < float(args.maximum_plane_inlier_ratio) <= 1.0:
        raise ValueError("maximum plane inlier ratio must lie in (0,1]")

    plane, plane_meta = _load_plane(args.plane_pose_inventory)
    hloc_by_name = _parse_hloc_results(args.hloc_results)
    plane_names = plane["names"].astype(str)
    query_names = [_plane_name_to_hloc_name(name) for name in plane_names.tolist()]
    if len(set(query_names)) != len(query_names) or set(query_names) != set(hloc_by_name):
        raise ValueError("HLoc results and planar query inventories differ")

    query_pair_names = _read_first_column(args.hloc_query_pairs)
    query_intrinsics_names = _read_first_column(args.hloc_query_intrinsics)
    if set(query_pair_names) != set(query_names) or set(query_intrinsics_names) != set(query_names):
        raise ValueError("HLoc pair/intrinsics query inventory differs")

    mapping_names_by_camera = read_colmap_image_camera_ids_binary(
        args.hloc_sfm_model / "images.bin"
    )
    mapping_names = set(mapping_names_by_camera)
    pair_rows = [line.split() for line in args.hloc_query_pairs.read_text().splitlines() if line.split()]
    if any(len(row) != 2 for row in pair_rows):
        raise ValueError("HLoc retrieval pair rows must have two columns")
    paired_mapping_names = {str(row[1]) for row in pair_rows}
    if not paired_mapping_names.issubset(mapping_names):
        raise ValueError("HLoc retrieval pairs reference images outside the SfM map")
    if mapping_names.intersection(query_names):
        raise ValueError("HLoc SfM mapping and query image names overlap")
    mapping_routes = sorted({name.split("/", 1)[0] for name in mapping_names})
    query_routes = sorted({name.split("/", 1)[0] for name in query_names})
    if set(mapping_routes).intersection(query_routes):
        raise ValueError("HLoc mapping and query routes overlap")

    fallback = _select_hloc_fallback(
        plane["usable"], plane["selected_inlier_ratio"],
        float(args.maximum_plane_inlier_ratio),
    )
    hloc_pose = np.stack([hloc_by_name[name] for name in query_names], axis=0)
    output_pose = np.asarray(plane["pose_w2c"], np.float64).copy()
    output_pose[fallback] = hloc_pose[fallback]
    output_usable = np.asarray(plane["usable"], bool).copy()
    output_usable[fallback] = True

    arrays = {
        "names": plane_names,
        "pose_w2c": output_pose,
        "usable": output_usable,
        "selected_branch": np.where(fallback, 101, 100).astype(np.int16),
        # Deliberately preserve the planar confidence.  HLoc confidence is not
        # calibrated onto the planar inlier-ratio scale; raw recall is valid,
        # while selective precision remains a planar-confidence diagnostic.
        "selected_inlier_ratio": np.asarray(plane["selected_inlier_ratio"], np.float64),
        "selected_candidate_correspondence_count": np.asarray(
            plane["selected_candidate_correspondence_count"], np.int64
        ),
        "selected_pnp_inlier_count": np.asarray(
            plane["selected_pnp_inlier_count"], np.int64
        ),
        "hloc_pose_w2c": hloc_pose,
        "hloc_fallback_selected": fallback,
    }
    artifact_paths = {
        "results": args.hloc_results,
        "pairs": args.hloc_query_pairs,
        "intrinsics": args.hloc_query_intrinsics,
        "global_features": args.hloc_global_features,
        "local_features": args.hloc_local_features,
        "query_matches": args.hloc_query_matches,
        "sfm_cameras": args.hloc_sfm_model / "cameras.bin",
        "sfm_images": args.hloc_sfm_model / "images.bin",
        "sfm_points3D": args.hloc_sfm_model / "points3D.bin",
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_hloc_low_support_fallback_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(plane_names)),
        "selection_rule": "planar_default_else_hloc_if_planar_unusable_or_inlier_ratio_below_fixed_boundary",
        "maximum_plane_inlier_ratio": float(args.maximum_plane_inlier_ratio),
        "fallback_selected_count": int(np.sum(fallback)),
        "source_branch_codes": {"100": "radio_sift_plane", "101": "hloc_low_support_fallback"},
        "hloc_method": "NetVLAD_retrieval_SuperPoint_SuperGlue_SfM_PnP_not_LoFTR",
        "plane_pose_inventory_file_sha256": file_sha256(args.plane_pose_inventory),
        "plane_pose_inventory_content_sha256": plane_meta.get("content_sha256"),
        "hloc_artifact_file_sha256": {
            name: file_sha256(path) for name, path in artifact_paths.items()
        },
        "hloc_mapping_image_count": int(len(mapping_names)),
        "hloc_query_image_count": int(len(query_names)),
        "hloc_mapping_routes": mapping_routes,
        "hloc_query_routes": query_routes,
        "hloc_mapping_query_name_overlap_count": 0,
        "hloc_mapping_query_route_overlap_count": 0,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "strict_runtime_phase_separation_eligible": True,
        "fallback_confidence_calibrated_to_plane_ratio": False,
        "configuration_role": "historical_cross_scene_hybrid_validation_fixed_existing_plane_boundary",
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
