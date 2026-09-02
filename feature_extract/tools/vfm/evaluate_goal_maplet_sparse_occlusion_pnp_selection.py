"""Post-label evaluation of a frozen sparse-occlusion PnP branch selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    usable = [row for row in rows if bool(row["usable"])]
    translation = np.asarray([row["translation_error_m"] for row in usable], np.float64)
    rotation = np.asarray([row["rotation_error_deg"] for row in usable], np.float64)
    thresholds = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 10.0), (2.0, 45.0))
    return {
        "query_count": int(len(rows)),
        "usable_count": int(len(usable)),
        "selected_carrier_count": int(sum(int(row["selected_branch"]) == 1 for row in rows)),
        "median_translation_m": float(np.median(translation)) if len(translation) else None,
        "median_rotation_deg": float(np.median(rotation)) if len(rotation) else None,
        "threshold_hits": {
            f"{translation_limit:g}m_{rotation_limit:g}deg": int(np.sum(
                (translation <= translation_limit) & (rotation <= rotation_limit)
            ))
            for translation_limit, rotation_limit in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-occlusion selection evaluation")
    with np.load(args.selection, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        keys = (
            "names", "pose_w2c", "usable", "selected_branch",
            "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
            "baseline_inlier_ratio", "carrier_inlier_ratio",
        )
        arrays = {key: np.asarray(data[key]) for key in keys}
    count = len(arrays["names"])
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_sparse_occlusion_pnp_pareto_selection_v1",
            "goal_maplet_sparse_occlusion_pnp_cross_island_selection_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("baseline_connected_regions_always_available") is not True
        or metadata.get("carrier_is_auxiliary_pose_hypothesis") is not True
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or arrays["pose_w2c"].shape != (count, 4, 4)
        or len(set(arrays["names"].astype(str).tolist())) != count
    ):
        raise ValueError("sparse-occlusion PnP selection differs")
    rows: list[dict[str, object]] = []
    for index, name in enumerate(arrays["names"].astype(str).tolist()):
        usable = bool(arrays["usable"][index])
        row: dict[str, object] = {
            "name": name,
            "route": name.split("__", 1)[0],
            "usable": usable,
            "selected_branch": int(arrays["selected_branch"][index]),
        }
        if usable:
            pose = np.asarray(arrays["pose_w2c"][index], np.float64)
            with np.load(args.query_contributors / name, allow_pickle=False) as data:
                gt = np.asarray(data["pose_w2c"], np.float64)
            center = -pose[:3, :3].T @ pose[:3, 3]
            gt_center = -gt[:3, :3].T @ gt[:3, 3]
            row["translation_error_m"] = float(np.linalg.norm(center - gt_center))
            row["rotation_error_deg"] = float(
                Rotation.from_matrix(pose[:3, :3] @ gt[:3, :3].T).magnitude()
                * 180.0 / np.pi
            )
        rows.append(row)
    report = {
        "artifact_type": (
            "goal_maplet_sparse_occlusion_pnp_cross_island_selection_evaluation_v1"
            if metadata.get("artifact_type")
            == "goal_maplet_sparse_occlusion_pnp_cross_island_selection_v1"
            else "goal_maplet_sparse_occlusion_pnp_pareto_selection_evaluation_v1"
        ),
        "selection_artifact_type": metadata.get("artifact_type"),
        "selection_file_sha256": file_sha256(args.selection),
        "selection_content_sha256": metadata.get("content_sha256"),
        "selection_frozen_before_pose_labels_opened": True,
        "selection_role": "historical_mechanism_control_not_blind_promotion",
        "summary": _summary(rows),
        "route_summaries": {
            route: _summary([row for row in rows if row["route"] == route])
            for route in sorted(set(str(row["route"]) for row in rows))
        },
        "production_eligible": False,
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
