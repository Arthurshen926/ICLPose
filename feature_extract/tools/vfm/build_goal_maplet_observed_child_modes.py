"""Build query-disjoint physical-child RADIO appearance modes from mapping views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_view_conditioned_field import (
    _eligible_contributors,
    _observation,
)
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.multimodal_parent_retrieval import (
    AnonymousChildModeReadout,
)
from feature_extract.vfm.localization_goal_maplet.observed_child_modes import (
    MODE_SEMANTICS,
    ObservedChildModeArtifact,
    update_online_child_modes,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_renderer import dominant_child_owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--exclude_trajectories", nargs="*", default=[])
    parser.add_argument("--maximum_modes", type=int, default=4)
    parser.add_argument("--minimum_angular_residual", type=float, default=0.05)
    parser.add_argument("--artifact_root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output, summary = Path(args.output_npz), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite observed child mode artifact")
    if int(args.maximum_modes) <= 0:
        raise ValueError("maximum_modes must be positive")

    physical_path = Path(args.physical_map)
    canonical_path = Path(args.canonical_field)
    mapper_path = Path(args.surface_mapper)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    canonical = CanonicalSurfaceField.load_npz(canonical_path)
    if canonical.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    mapper_hash = file_sha256(mapper_path)
    if canonical.metadata.get("surface_mapper_file_sha256") not in (None, mapper_hash):
        raise ValueError("surface mapper differs from canonical field")
    mapper, _ = load_surface_maplet_mapper(mapper_path, device=str(args.device))
    eligible = _eligible_contributors(
        Path(args.contributors), set(str(value) for value in args.exclude_trajectories)
    )
    trajectories = sorted({str(metadata["trajectory_id"]) for _, metadata in eligible})
    if int(canonical.metadata.get("mapping_image_count", len(eligible))) != len(eligible):
        raise ValueError("mapping image count differs from canonical field")
    if sorted(canonical.metadata.get("mapping_trajectory_ids", trajectories)) != trajectories:
        raise ValueError("mapping trajectories differ from canonical field")

    dense_physical = np.full(int(np.max(physical.primitive_ids)) + 1, -1, dtype=np.int64)
    dense_physical[physical.primitive_ids] = np.arange(physical.primitive_ids.size)
    dense_field = np.full(physical.primitive_ids.size, -1, dtype=np.int64)
    dense_field[canonical.primitive_rows] = np.arange(canonical.primitive_rows.size)
    owner = dominant_child_owner(physical)
    child_count = int(physical.child_parent_rows.size)
    descriptors = np.zeros(
        (child_count, int(args.maximum_modes), canonical.feature_dim), dtype=np.float32
    )
    accumulated = np.zeros((child_count, int(args.maximum_modes)), dtype=np.float64)
    mode_counts = np.zeros((child_count,), dtype=np.int32)
    view_observation_count = np.zeros((child_count,), dtype=np.int32)
    contributor_hash = hashlib.sha256()
    coordinate_contracts: set[str] = set()
    total_child_view_observations = 0
    for index, (path, metadata) in enumerate(eligible, start=1):
        observation = _observation(
            path, metadata, physical, dense_physical, dense_field, mapper,
            Path(args.artifact_root).resolve(),
        )
        primitive_rows = canonical.primitive_rows[observation.field_indices]
        child_rows = owner[primitive_rows]
        keep = child_rows >= 0
        child_rows = child_rows[keep]
        value = observation.descriptors[keep]
        weight = observation.weights[keep].astype(np.float64)
        order = np.argsort(child_rows, kind="stable")
        ordered_child = child_rows[order]
        starts = np.r_[0, np.flatnonzero(ordered_child[1:] != ordered_child[:-1]) + 1]
        unique_child = ordered_child[starts]
        mass = np.add.reduceat(weight[order], starts)
        feature_sum = np.add.reduceat(weight[order, None] * value[order], starts, axis=0)
        child_descriptor = feature_sum / np.maximum(mass[:, None], 1e-8)
        child_descriptor /= np.maximum(
            np.linalg.norm(child_descriptor, axis=1, keepdims=True), 1e-8
        )
        update_online_child_modes(
            descriptors, accumulated, mode_counts, unique_child,
            child_descriptor, mass,
            minimum_angular_residual=float(args.minimum_angular_residual),
        )
        view_observation_count[unique_child] += 1
        total_child_view_observations += int(unique_child.size)
        coordinate_contracts.add(str(observation.coordinate_audit["coordinate_contract"]))
        contributor_hash.update(path.name.encode("utf8"))
        contributor_hash.update(bytes.fromhex(file_sha256(path)))
        if index == 1 or index % 100 == 0 or index == len(eligible):
            print(json.dumps({
                "view": index, "view_count": len(eligible),
                "image_id": metadata["image_id"],
                "child_observation_count": int(unique_child.size),
            }), flush=True)

    canonical_readout = readout_canonical_field(canonical, physical)
    observed_active = mode_counts > 0
    fallback = (~observed_active) & (canonical_readout.child_coverage > 0.0)
    descriptors[fallback, 0] = canonical_readout.child_descriptors[fallback]
    accumulated[fallback, 0] = 1.0
    mode_counts[fallback] = 1
    total = np.sum(accumulated, axis=1, keepdims=True)
    weights = np.divide(
        accumulated, np.maximum(total, 1e-12),
        out=np.zeros_like(accumulated), where=total > 0.0,
    ).astype(np.float32)
    artifact = ObservedChildModeArtifact(
        readout=AnonymousChildModeReadout(
            descriptors=descriptors,
            weights=weights,
            child_coverage=canonical_readout.child_coverage,
        ),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=canonical.content_sha256,
        surface_mapper_file_sha256=mapper_hash,
        metadata={
            "artifact_type": "goal_maplet_observed_child_mode_readout_v1",
            "mode_semantics": MODE_SEMANTICS,
            "mapping_image_count": len(eligible),
            "mapping_trajectory_ids": trajectories,
            "excluded_trajectory_ids": sorted(str(v) for v in args.exclude_trajectories),
            "maximum_modes": int(args.maximum_modes),
            "minimum_angular_residual": float(args.minimum_angular_residual),
            "contributor_set_sha256": contributor_hash.hexdigest(),
            "coordinate_contracts": sorted(coordinate_contracts),
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "uses_mapping_pose_for_surface_association": True,
            "stores_mapping_image_ids": False,
            "stores_mapping_rgb": False,
        },
    )
    artifact.save_npz(output)
    reopened = ObservedChildModeArtifact.load_npz(output)
    if reopened.content_sha256 != artifact.content_sha256:
        raise AssertionError("observed child mode replay differs")
    active = mode_counts > 0
    payload = {
        "artifact_type": "goal_maplet_observed_child_mode_build_audit_v1",
        "output_npz": str(output.resolve()),
        "output_file_sha256": file_sha256(output),
        "content_sha256": artifact.content_sha256,
        "mapping_image_count": len(eligible),
        "mapping_trajectory_ids": trajectories,
        "total_child_view_observation_count": total_child_view_observations,
        "active_child_count": int(np.sum(active)),
        "active_child_fraction": float(np.mean(active)),
        "observed_child_count": int(np.sum(observed_active)),
        "canonical_fallback_child_count": int(np.sum(fallback)),
        "mode_count_mean_active": float(np.mean(mode_counts[active])),
        "mode_count_histogram": {
            str(value): int(np.sum(mode_counts == value))
            for value in range(int(args.maximum_modes) + 1)
        },
        "view_observation_count_median_active": float(
            np.median(view_observation_count[active])
        ),
        "query_route_excluded": sorted(str(v) for v in args.exclude_trajectories),
        "promotion_eligible": False,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
