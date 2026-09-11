"""Freeze the exact dense-render subset implied by sparse geometry consensus.

The V5/V11 consensus can select V11 only when its sparse plane/scale objective
is already strictly better.  Dense full-map rendering is therefore unnecessary
for every other query.  This pose/label-free plan records that logical
short-circuit before either dense render is opened; it changes scheduling only,
not the declared selection rule. The optional missing-evidence policy also
renders both candidates when both sparse objectives are positive infinity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import (
    _load_plane_geometry,
    _sparse_allows_alternate,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


ARTIFACT_TYPE = "goal_maplet_sparse_first_dense_render_plan_v1"


def _dense_render_required(objective: np.ndarray, usable: np.ndarray, missing_sparse_policy: str = "retain_primary") -> np.ndarray:
    value = np.asarray(objective, np.float64)
    valid = np.asarray(usable, bool)
    if value.ndim != 2 or value.shape[1:] != (2,) or valid.shape != value.shape:
        raise ValueError("sparse-first candidate arrays differ")
    return valid[:, 0] & valid[:, 1] & _sparse_allows_alternate(value, missing_sparse_policy)


def _load_sparse_first_plan(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    expected = canonical_json_sha256({
        key: value for key, value in metadata.items() if key != "content_sha256"
    })
    count = len(arrays.get("names", []))
    required = arrays.get("dense_render_required", np.empty(0))
    if (
        metadata.get("artifact_type") != ARTIFACT_TYPE
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("selection_rule_changed") is not False
        or metadata.get("missing_sparse_policy", "retain_primary") not in {"retain_primary", "dense_when_both_missing"}
        or "names" not in arrays
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or metadata.get("content_sha256") != expected
        or int(metadata.get("query_count", -1)) != count
        or required.shape != (count,)
        or required.dtype != np.bool_
        or int(np.sum(required)) != int(metadata.get("dense_render_query_count", -1))
        or len(set(arrays["names"].astype(str).tolist())) != count
    ):
        raise ValueError("sparse-first render plan differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose", type=Path, required=True)
    parser.add_argument("--alternate_pose", type=Path, required=True)
    parser.add_argument("--plane_geometry", type=Path, required=True)
    parser.add_argument("--missing_sparse_policy", choices=("retain_primary", "dense_when_both_missing"), default="retain_primary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-first render plan")

    primary, primary_meta = _load_pose_candidate(args.primary_pose)
    alternate, alternate_meta = _load_pose_candidate(args.alternate_pose)
    plane, plane_meta = _load_plane_geometry(args.plane_geometry)
    names = primary["names"].astype(str)
    if not (
        np.array_equal(names, alternate["names"].astype(str))
        and np.array_equal(names, plane["names"].astype(str))
        and plane_meta.get("point_pose_file_sha256") == file_sha256(args.primary_pose)
        and plane_meta.get("surface_pose_file_sha256") == file_sha256(args.alternate_pose)
    ):
        raise ValueError("sparse-first pose/plane lineage differs")
    usable = np.stack([primary["usable"], alternate["usable"]], axis=1).astype(bool)
    objective = np.asarray(plane["moge3_plane_geometry_objective"], np.float64)
    required = _dense_render_required(objective, usable, args.missing_sparse_policy)
    arrays = {
        "names": primary["names"],
        "dense_render_required": required,
    }
    metadata: dict[str, object] = {
        "artifact_type": ARTIFACT_TYPE,
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "dense_render_query_count": int(np.sum(required)),
        "short_circuit_rule": (
            "render_both_candidates_iff_both_usable_and_V11_sparse_plane_scale_"
            "objective_is_strictly_better_or_declared_missing_sparse_override_applies"
        ),
        "missing_sparse_policy": args.missing_sparse_policy,
        "selection_rule_changed": False,
        "selection_rule_changed_semantics": "scheduling preserves the explicitly declared selection policy",
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "primary_pose_file_sha256": file_sha256(args.primary_pose),
        "primary_pose_content_sha256": primary_meta.get("content_sha256"),
        "alternate_pose_file_sha256": file_sha256(args.alternate_pose),
        "alternate_pose_content_sha256": alternate_meta.get("content_sha256"),
        "plane_geometry_file_sha256": file_sha256(args.plane_geometry),
        "plane_geometry_content_sha256": plane_meta.get("content_sha256"),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _load_sparse_first_plan(args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
