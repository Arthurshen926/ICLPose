"""Select a sparse-occlusion PnP branch by frozen cross-island inlier support.

No pose label is read.  A probe pose may replace the connected-plane baseline
only when it creates strictly more PnP inliers that are independently supported
by at least two observed islands of one carrier on the same physical map plane,
without reducing total inliers or the query-image inlier hull.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.build_goal_maplet_direct_radio_plane_ranking import (
    _base_to_carrier_regions,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _pnp,
    _pose_diagnostics,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def _load_correspondences(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        keys = (
            "names", "correspondence_offsets", "world_points", "query_tokens",
            "provenance_region_plane_atlas_row", "camera_matrices", "radial_k1",
        )
        arrays = {key: np.asarray(data[key]) for key in keys}
    count = len(arrays["names"])
    if (
        metadata.get("artifact_type")
        != "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v1"
        or metadata.get("pose_or_ground_truth_opened") is not False
        or metadata.get("query_depth_or_scale_used") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or arrays["correspondence_offsets"].shape != (count + 1,)
        or arrays["camera_matrices"].shape != (count, 3, 3)
        or len(set(arrays["names"].astype(str).tolist())) != count
    ):
        raise ValueError("frozen PnP correspondence inventory differs")
    total = int(arrays["correspondence_offsets"][-1])
    if (
        arrays["world_points"].shape != (total, 3)
        or arrays["query_tokens"].shape != (total,)
        or arrays["provenance_region_plane_atlas_row"].shape != (total, 3)
    ):
        raise ValueError("frozen PnP correspondence arrays differ")
    return arrays, metadata


def _cross_island_support(
    inlier: np.ndarray,
    provenance: np.ndarray,
    base_to_carrier: np.ndarray,
    *,
    minimum_inliers_per_island: int = 2,
) -> dict[str, int]:
    """Count same-map-plane inliers independently supported by two islands."""

    selected = np.asarray(provenance, np.int64)[np.asarray(inlier, np.int64)]
    if selected.size == 0:
        return {
            "cross_island_plane_group_count": 0,
            "cross_island_inlier_count": 0,
            "cross_island_region_count": 0,
        }
    regions = selected[:, 0]
    if np.any(regions < 0) or np.any(regions >= len(base_to_carrier)):
        raise ValueError("PnP provenance region is outside the base-plane inventory")
    carriers = np.asarray(base_to_carrier, np.int64)[regions]
    supported_rows: set[int] = set()
    supported_regions: set[int] = set()
    group_count = 0
    for carrier in np.unique(carriers).tolist():
        carrier_rows = np.flatnonzero(carriers == int(carrier))
        carrier_regions = np.unique(regions[carrier_rows])
        if len(carrier_regions) < 2:
            continue
        for plane in np.unique(selected[carrier_rows, 1]).tolist():
            plane_rows = carrier_rows[selected[carrier_rows, 1] == int(plane)]
            counts = {
                int(region): int(np.sum(regions[plane_rows] == int(region)))
                for region in np.unique(regions[plane_rows]).tolist()
            }
            eligible = {
                region for region, count in counts.items()
                if count >= int(minimum_inliers_per_island)
            }
            if len(eligible) < 2:
                continue
            group_count += 1
            supported_regions.update(eligible)
            supported_rows.update(
                int(row) for row in plane_rows.tolist()
                if int(regions[int(row)]) in eligible
            )
    return {
        "cross_island_plane_group_count": int(group_count),
        "cross_island_inlier_count": int(len(supported_rows)),
        "cross_island_region_count": int(len(supported_regions)),
    }


def _branch_metrics(
    pose_arrays: dict[str, np.ndarray],
    correspondence_arrays: dict[str, np.ndarray],
    index: int,
    base_to_carrier: np.ndarray,
    token_grid: tuple[int, int],
) -> dict[str, object]:
    lo, hi = map(
        int, correspondence_arrays["correspondence_offsets"][index:index + 2],
    )
    world = np.asarray(correspondence_arrays["world_points"][lo:hi], np.float64)
    tokens = np.asarray(correspondence_arrays["query_tokens"][lo:hi], np.int64)
    provenance = np.asarray(
        correspondence_arrays["provenance_region_plane_atlas_row"][lo:hi], np.int64,
    )
    K = np.asarray(correspondence_arrays["camera_matrices"][index], np.float64)
    k1 = float(correspondence_arrays["radial_k1"][index])
    pose, inlier = _pnp(world, tokens, K, k1, token_grid=token_grid)
    stored_usable = bool(pose_arrays["usable"][index])
    if (pose is not None) != stored_usable:
        raise ValueError("replayed PnP usability differs from frozen pose")
    if pose is not None and not np.allclose(
        pose, pose_arrays["pose_w2c"][index], rtol=0.0, atol=1e-8,
    ):
        raise ValueError("replayed PnP pose differs from frozen pose")
    diagnostics = _pose_diagnostics(
        pose, inlier, world, tokens,
        [tuple(map(int, row)) for row in provenance.tolist()], K, k1,
        token_grid=token_grid,
    )
    return {
        "usable": pose is not None,
        "pnp_inlier_count": int(len(inlier)),
        "candidate_correspondence_count": int(len(world)),
        "inlier_query_hull_fraction": float(diagnostics["inlier_query_hull_fraction"]),
        **_cross_island_support(inlier, provenance, base_to_carrier),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline_correspondence_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--probe_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--probe_correspondence_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--base_query_plane_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--carrier_query_plane_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = {
        len(args.baseline_pose_inventory), len(args.baseline_correspondence_inventory),
        len(args.probe_pose_inventory), len(args.probe_correspondence_inventory),
        len(args.base_query_plane_dir), len(args.carrier_query_plane_dir),
    }
    if len(lengths) != 1:
        raise ValueError("cross-island selection shard counts differ")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite cross-island PnP selection")

    output_rows: list[dict[str, object]] = []
    source_hashes: dict[str, list[str]] = {
        "baseline_pose": [], "baseline_correspondence": [],
        "probe_pose": [], "probe_correspondence": [],
        "base_plane_manifest": [], "carrier_plane_manifest": [],
    }
    selected_arrays: dict[str, list[np.ndarray | int | float | bool | str]] = {
        key: [] for key in (
            "names", "pose_w2c", "usable", "selected_branch",
            "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
            "baseline_inlier_ratio", "carrier_inlier_ratio",
        )
    }
    for base_pose_path, base_corr_path, probe_pose_path, probe_corr_path, base_dir, carrier_dir in zip(
        args.baseline_pose_inventory, args.baseline_correspondence_inventory,
        args.probe_pose_inventory, args.probe_correspondence_inventory,
        args.base_query_plane_dir, args.carrier_query_plane_dir,
    ):
        base_pose, base_pose_meta = _load_frozen_poses(base_pose_path)
        probe_pose, probe_pose_meta = _load_frozen_poses(probe_pose_path)
        base_corr, base_corr_meta = _load_correspondences(base_corr_path)
        probe_corr, probe_corr_meta = _load_correspondences(probe_corr_path)
        names = base_pose["names"].astype(str)
        if not (
            np.array_equal(names, probe_pose["names"].astype(str))
            and np.array_equal(names, base_corr["names"].astype(str))
            and np.array_equal(names, probe_corr["names"].astype(str))
        ):
            raise ValueError("cross-island branch names differ")
        if (
            base_pose_meta.get("correspondence_report_file_sha256")
            != base_corr_meta.get("correspondence_report_file_sha256")
            or probe_pose_meta.get("correspondence_report_file_sha256")
            != probe_corr_meta.get("correspondence_report_file_sha256")
            or base_pose_meta.get("query_camera_only_inventory_file_sha256")
            != probe_pose_meta.get("query_camera_only_inventory_file_sha256")
            or base_pose_meta.get("source_observation_bank_file_sha256")
            != probe_pose_meta.get("source_observation_bank_file_sha256")
        ):
            raise ValueError("cross-island branch lineage differs")
        token_grid = tuple(map(int, base_pose_meta.get("token_grid", (36, 64))))
        if list(token_grid) != list(probe_pose_meta.get("token_grid", ())):
            raise ValueError("cross-island branch token grids differ")
        for index, name in enumerate(names.tolist()):
            base_planes, _ = QueryPlaneRegions.load_npz(base_dir / name)
            carrier_planes, carrier_meta = QueryPlaneRegions.load_npz(carrier_dir / name)
            diagnostics = carrier_meta.get("carrier_diagnostics", {})
            if (
                carrier_meta.get("sparse_occlusion_carrier") is not True
                or diagnostics.get("observed_support_bit_exact") is not True
                or diagnostics.get("hidden_pixel_count_added") != 0
            ):
                raise ValueError("cross-island carrier member differs")
            base_to_carrier = _base_to_carrier_regions(
                base_planes.labels, carrier_planes.labels,
            )
            baseline = _branch_metrics(
                base_pose, base_corr, index, base_to_carrier, token_grid,
            )
            probe = _branch_metrics(
                probe_pose, probe_corr, index, base_to_carrier, token_grid,
            )
            choose_probe = bool(probe["usable"]) and (
                not bool(baseline["usable"])
                or (
                    int(probe["cross_island_inlier_count"])
                    > int(baseline["cross_island_inlier_count"])
                    and int(probe["pnp_inlier_count"]) >= int(baseline["pnp_inlier_count"])
                    and float(probe["inlier_query_hull_fraction"])
                    >= float(baseline["inlier_query_hull_fraction"])
                )
            )
            source = probe_pose if choose_probe else base_pose
            metrics = probe if choose_probe else baseline
            selected_arrays["names"].append(name)
            selected_arrays["pose_w2c"].append(np.asarray(source["pose_w2c"][index], np.float64))
            selected_arrays["usable"].append(bool(source["usable"][index]))
            selected_arrays["selected_branch"].append(1 if choose_probe else 0)
            selected_arrays["selected_candidate_correspondence_count"].append(
                int(metrics["candidate_correspondence_count"])
            )
            selected_arrays["selected_pnp_inlier_count"].append(int(metrics["pnp_inlier_count"]))
            selected_arrays["baseline_inlier_ratio"].append(
                int(baseline["pnp_inlier_count"]) / max(int(baseline["candidate_correspondence_count"]), 1)
            )
            selected_arrays["carrier_inlier_ratio"].append(
                int(probe["pnp_inlier_count"]) / max(int(probe["candidate_correspondence_count"]), 1)
            )
            output_rows.append({
                "name": name, "selected_branch": 1 if choose_probe else 0,
                "baseline": baseline, "probe": probe,
            })
        for key, path in (
            ("baseline_pose", base_pose_path), ("baseline_correspondence", base_corr_path),
            ("probe_pose", probe_pose_path), ("probe_correspondence", probe_corr_path),
            ("base_plane_manifest", base_dir / "manifest.json"),
            ("carrier_plane_manifest", carrier_dir / "manifest.json"),
        ):
            source_hashes[key].append(file_sha256(path))

    arrays = {
        "names": np.asarray(selected_arrays["names"]),
        "pose_w2c": np.asarray(selected_arrays["pose_w2c"], np.float64),
        "usable": np.asarray(selected_arrays["usable"], bool),
        "selected_branch": np.asarray(selected_arrays["selected_branch"], np.int8),
        "selected_candidate_correspondence_count": np.asarray(
            selected_arrays["selected_candidate_correspondence_count"], np.int64,
        ),
        "selected_pnp_inlier_count": np.asarray(
            selected_arrays["selected_pnp_inlier_count"], np.int64,
        ),
        "baseline_inlier_ratio": np.asarray(selected_arrays["baseline_inlier_ratio"], np.float64),
        "carrier_inlier_ratio": np.asarray(selected_arrays["carrier_inlier_ratio"], np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_sparse_occlusion_pnp_cross_island_selection_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(arrays["names"])),
        "selected_carrier_count": int(np.sum(arrays["selected_branch"] == 1)),
        "selection_rule": (
            "strictly_more_same_map_plane_cross_island_inliers_and_"
            "nondecreasing_total_inliers_and_query_hull;_tie_baseline"
        ),
        "minimum_inliers_per_island": 2,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "baseline_connected_regions_always_available": True,
        "carrier_is_auxiliary_pose_hypothesis": True,
        "selection_designed_after_oldhospital_a6_mechanism_audit": True,
        "eligible_use": "historical_mechanism_control_only_then_freeze_for_new_unseen_window",
        "source_file_sha256": source_hashes,
        "selection_rows_sha256": canonical_json_sha256(output_rows),
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
