"""Select Top5/Top10/H3-balanced PnP candidates by unique-token inlier ratio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _load_h3(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    keys = (
        "names", "balanced_support_pose_w2c", "balanced_support_inlier_count",
        "balanced_support_inlier_ratio",
    )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        all_arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        arrays = {key: all_arrays[key] for key in keys}
    if (
        metadata.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_grouped_multihypothesis_v1"
        or metadata.get("query_pose_or_ground_truth_opened") is not False
        or arrays_sha256(all_arrays) != metadata.get("arrays_sha256")
        or arrays["balanced_support_pose_w2c"].shape != (len(arrays["names"]), 4, 4)
    ):
        raise ValueError("H3 balanced candidate inventory differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5_pose_inventory", type=Path, required=True)
    parser.add_argument("--top10_pose_inventory", type=Path, required=True)
    parser.add_argument("--h3_candidate_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite H3 selection")
    top5, meta5 = _load_frozen_poses(args.top5_pose_inventory)
    top10, meta10 = _load_frozen_poses(args.top10_pose_inventory)
    h3, meta3 = _load_h3(args.h3_candidate_inventory)
    names = top5["names"].astype(str)
    if not np.array_equal(names, top10["names"].astype(str)) or not np.array_equal(
        names, h3["names"].astype(str)
    ):
        raise ValueError("candidate names differ")
    ratio5 = top5["pnp_inlier_count"] / np.maximum(top5["candidate_correspondence_count"], 1)
    ratio10 = top10["pnp_inlier_count"] / np.maximum(top10["candidate_correspondence_count"], 1)
    ratio3 = np.asarray(h3["balanced_support_inlier_ratio"], np.float64)
    poses = []
    sources = []
    ratios = []
    usable = []
    for index in range(len(names)):
        candidates = []
        if bool(top5["usable"][index]):
            candidates.append((float(ratio5[index]), 5, top5["pose_w2c"][index]))
        if bool(top10["usable"][index]):
            candidates.append((float(ratio10[index]), 10, top10["pose_w2c"][index]))
        pose3 = np.asarray(h3["balanced_support_pose_w2c"][index], np.float64)
        if np.all(np.isfinite(pose3)):
            candidates.append((float(ratio3[index]), 30, pose3))
        if candidates:
            # Stable tie preference is Top5, then Top10, then H3.
            best = max(candidates, key=lambda value: (value[0], -value[1]))
            ratios.append(best[0]); sources.append(best[1]); poses.append(best[2]); usable.append(True)
        else:
            ratios.append(0.0); sources.append(0); poses.append(np.full((4, 4), np.nan)); usable.append(False)
    arrays = {
        "names": names,
        "pose_w2c": np.asarray(poses, np.float64),
        "usable": np.asarray(usable, bool),
        "selected_source": np.asarray(sources, np.int8),
        "selected_unique_token_inlier_ratio": np.asarray(ratios, np.float64),
        "top5_inlier_ratio": np.asarray(ratio5, np.float64),
        "top10_inlier_ratio": np.asarray(ratio10, np.float64),
        "h3_balanced_inlier_ratio": ratio3,
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_top5_top10_h3_balanced_selected_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": "maximum_unique_query_token_inlier_ratio_tie_top5_then_top10_then_h3_balanced",
        "query_pose_or_ground_truth_opened": False,
        "top5_pose_inventory_file_sha256": file_sha256(args.top5_pose_inventory),
        "top5_pose_inventory_content_sha256": meta5.get("content_sha256"),
        "top10_pose_inventory_file_sha256": file_sha256(args.top10_pose_inventory),
        "top10_pose_inventory_content_sha256": meta10.get("content_sha256"),
        "h3_candidate_inventory_file_sha256": file_sha256(args.h3_candidate_inventory),
        "h3_candidate_inventory_content_sha256": meta3.get("content_sha256"),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(args.output)
    print(json.dumps({**metadata, "selected_top5": int(np.sum(arrays["selected_source"] == 5)),
                      "selected_top10": int(np.sum(arrays["selected_source"] == 10)),
                      "selected_h3": int(np.sum(arrays["selected_source"] == 30)),
                      "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
