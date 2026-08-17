"""Audit mapping-pose coordinate round-trip, typed mass, and payload invariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import feature_extract.vfm.localization_goal_maplet.surface_renderer as surface_renderer_module

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    load_contributors_in_radio_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    dominant_child_owner,
    render_soft_child_surface_field,
)


def _camera_and_pose(path: Path) -> tuple[ColmapCamera, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            ColmapCamera(
                0,
                int(data["camera_model_id"]),
                int(data["camera_width"]),
                int(data["camera_height"]),
                tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()),
            ),
            np.asarray(data["pose_w2c"], dtype=np.float64),
        )


def _contributor_token_children(
    physical: GoalMapletPhysicalMap,
    contributor: Path,
    *,
    token_height: int,
    token_width: int,
    top_l: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    labels, coordinate_audit = load_contributors_in_radio_coordinates(contributor)
    ids = np.asarray(labels.topk_primitive_ids, dtype=np.int64)
    weights = np.asarray(labels.topk_weights, dtype=np.float64)
    height, width, source_topk = ids.shape
    if height % int(token_height) or width % int(token_width):
        raise ValueError("contributor grid is not an integer RADIO-token multiple")
    factor_y, factor_x = height // int(token_height), width // int(token_width)
    if factor_x != factor_y:
        raise ValueError("contributor/token grid scale must be isotropic")
    dense = np.full((int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64)
    dense[physical.primitive_ids] = np.arange(physical.primitive_ids.size)
    valid_id = (ids >= 0) & (ids < dense.size)
    primitive = np.full(ids.shape, -1, dtype=np.int64)
    primitive[valid_id] = dense[ids[valid_id]]
    owner = dominant_child_owner(physical)
    child = np.full(ids.shape, -1, dtype=np.int64)
    valid_primitive = primitive >= 0
    child[valid_primitive] = owner[primitive[valid_primitive]]
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    token = ((yy // factor_y) * int(token_width) + xx // factor_x)[..., None]
    token = np.broadcast_to(token, ids.shape).reshape(-1)
    child_flat, weight_flat = child.reshape(-1), weights.reshape(-1)
    valid = (child_flat >= 0) & (weight_flat > 0.0)
    normalization = float(factor_x * factor_y)
    key = token[valid] * int(physical.child_parent_rows.size) + child_flat[valid]
    order = np.argsort(key, kind="stable")
    ordered_key = key[order]
    starts = np.r_[0, np.flatnonzero(ordered_key[1:] != ordered_key[:-1]) + 1]
    unique_key = ordered_key[starts]
    unique_mass = np.add.reduceat(weight_flat[valid][order], starts) / normalization
    unique_token = unique_key // int(physical.child_parent_rows.size)
    unique_child = unique_key % int(physical.child_parent_rows.size)
    rank_order = np.lexsort((unique_child, -unique_mass, unique_token))
    ranked_token = unique_token[rank_order]
    group_start = np.r_[0, np.flatnonzero(ranked_token[1:] != ranked_token[:-1]) + 1]
    group_count = np.diff(np.r_[group_start, rank_order.size])
    within = np.arange(rank_order.size) - np.repeat(group_start, group_count)
    keep = within < int(top_l)
    chosen, chosen_rank = rank_order[keep], within[keep]
    count = int(token_height) * int(token_width)
    rows = np.full((count, int(top_l)), -1, dtype=np.int64)
    mass = np.zeros((count, int(top_l)), dtype=np.float64)
    rows[unique_token[chosen], chosen_rank] = unique_child[chosen]
    mass[unique_token[chosen], chosen_rank] = unique_mass[chosen]
    total = np.zeros((count,), dtype=np.float64)
    valid_weight = (ids >= 0) & (weights > 0.0)
    np.add.at(total, token[valid_weight.reshape(-1)], weights.reshape(-1)[valid_weight.reshape(-1)] / normalization)
    null = np.maximum(1.0 - np.sum(mass, axis=1), 0.0)
    norm = np.sum(mass, axis=1) + null
    mass /= np.maximum(norm[:, None], 1e-12)
    null /= np.maximum(norm, 1e-12)
    return rows, mass, null, {**coordinate_audit, "mean_total_alpha": float(np.mean(total))}


def _identity_overlap(
    left_rows: np.ndarray, left_mass: np.ndarray, left_null: np.ndarray,
    right_rows: np.ndarray, right_mass: np.ndarray, right_null: np.ndarray,
) -> np.ndarray:
    match = left_rows[:, :, None] == right_rows[:, None, :]
    match &= (left_rows[:, :, None] >= 0) & (right_rows[:, None, :] >= 0)
    common = np.sum(
        np.minimum(left_mass[:, :, None], right_mass[:, None, :]) * match,
        axis=(1, 2),
    )
    return np.clip(common + np.minimum(left_null, right_null), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--contributor", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--top_l", type=int, default=4)
    parser.add_argument("--coordinate_supersample_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite renderer contract audit")
    physical_path, field_path = Path(args.physical_map), Path(args.canonical_field)
    contributor_path, retrieval_path = Path(args.contributor), Path(args.retrieval)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
    camera, pose = _camera_and_pose(contributor_path)
    height = int(retrieval.metadata["token_height"])
    width = int(retrieval.metadata["token_width"])
    common = dict(
        width=width, height=height, top_l=int(args.top_l), device=str(args.device),
        coordinate_supersample_factor=int(args.coordinate_supersample_factor),
    )
    full = render_soft_child_surface_field(physical, field, pose, camera, **common)
    subset = render_soft_child_surface_field(
        physical, field, pose, camera,
        selected_child_rows=retrieval.scene_child_rows, **common,
    )
    identity_fields = (
        "child_rows", "child_weights", "child_tail_weight",
        "unassigned_geometry_weight", "background_weight", "null_weight", "total_alpha",
    )
    exact = {
        name: bool(np.array_equal(getattr(full, name), getattr(subset, name)))
        for name in identity_fields
    }
    retained = np.sum(full.child_weights, axis=2)
    conservation = (
        retained + full.child_tail_weight
        + full.unassigned_geometry_weight + full.background_weight
    )
    maximum_mass_error = float(np.max(np.abs(conservation - 1.0), initial=0.0))
    truth_rows, truth_mass, truth_null, coordinate_audit = _contributor_token_children(
        physical, contributor_path, token_height=height, token_width=width,
        top_l=int(args.top_l),
    )
    render_rows = full.child_rows.reshape(height * width, -1)
    render_mass = full.child_weights.reshape(height * width, -1).astype(np.float64)
    render_null = full.null_weight.reshape(-1).astype(np.float64)
    overlap = _identity_overlap(
        truth_rows, truth_mass, truth_null, render_rows, render_mass, render_null,
    )
    truth_observed = np.sum(truth_mass, axis=1)
    render_observed = np.sum(render_mass, axis=1)
    conditional_overlap = _identity_overlap(
        truth_rows,
        np.divide(truth_mass, np.maximum(truth_observed[:, None], 1e-12)),
        np.zeros_like(truth_null),
        render_rows,
        np.divide(render_mass, np.maximum(render_observed[:, None], 1e-12)),
        np.zeros_like(render_null),
    )
    report = {
        "artifact_type": "goal_maplet_soft_child_renderer_contract_audit_v2",
        "image_id": str(retrieval.image_id),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "retrieval_content_sha256": retrieval.content_sha256,
        "input_file_sha256": {
            "physical_map": file_sha256(physical_path),
            "canonical_field": file_sha256(field_path),
            "contributor": file_sha256(contributor_path),
            "retrieval": file_sha256(retrieval_path),
        },
        "source_file_sha256": {
            "audit": file_sha256(Path(__file__).resolve()),
            "surface_renderer": file_sha256(Path(surface_renderer_module.__file__).resolve()),
        },
        "coordinate_supersample_factor": int(args.coordinate_supersample_factor),
        "coordinate_audit": coordinate_audit,
        "payload_identity_exact_by_field": exact,
        "payload_identity_exact": bool(all(exact.values())),
        "maximum_typed_mass_conservation_error": maximum_mass_error,
        "mapping_pose_online_vs_contributor_mean_identity_overlap": float(np.mean(overlap)),
        "mapping_pose_online_vs_contributor_minimum_identity_overlap": float(np.min(overlap)),
        "mapping_pose_online_vs_contributor_mean_observed_conditional_overlap": float(
            np.mean(conditional_overlap[(truth_observed > 0.0) & (render_observed > 0.0)])
        ),
        "mean_rendered_total_alpha": float(np.mean(full.total_alpha)),
        "mean_child_tail_weight": float(np.mean(full.child_tail_weight)),
        "mean_unassigned_geometry_weight": float(np.mean(full.unassigned_geometry_weight)),
        "mean_background_weight": float(np.mean(full.background_weight)),
        "mean_canonical_field_missing_weight": float(
            np.mean(full.canonical_field_missing_weight)
        ),
        "mean_payload_excluded_weight_subset": float(np.mean(subset.payload_excluded_weight)),
        "maximum_alpha_overflow": float(full.maximum_alpha_overflow),
        "overflow_token_fraction": float(full.overflow_token_fraction),
        "renderer_internal_contract_passed": bool(
            all(exact.values()) and maximum_mass_error <= 2e-5
            and full.maximum_alpha_overflow <= 2e-5
        ),
        "promotion_gate_passed": False,
        "promotion_blockers": [
            "mapping contributor stores only top4 primitive alpha while online renderer retains full composited mass; exact same-semantic round-trip authority is absent",
        ],
        "claims": {
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
            "mapping_pose_is_read_only_from_contributor": True,
            "comparison_is_coordinate_correct_raw_simple_radial": True,
            "contributor_agreement_is_diagnostic_not_a_calibrated_threshold": True,
        },
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
