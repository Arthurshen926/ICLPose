"""Attribute child-evidence loss between retrieval, contributor truth, and render."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_soft_child_renderer_contract import (
    _camera_and_pose,
    _contributor_token_children,
    _identity_overlap,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_soft_child_surface_field,
)


def _product_overlap(
    left_rows: np.ndarray, left_mass: np.ndarray,
    right_rows: np.ndarray, right_mass: np.ndarray,
) -> np.ndarray:
    match = left_rows[:, :, None] == right_rows[:, None, :]
    match &= (left_rows[:, :, None] >= 0) & (right_rows[:, None, :] >= 0)
    return np.sum(left_mass[:, :, None] * right_mass[:, None, :] * match, axis=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--query_contributor", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--top_l", type=int, default=4)
    parser.add_argument("--coordinate_supersample_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite evidence ladder")
    physical_path, field_path = Path(args.physical_map), Path(args.canonical_field)
    contributor_path, retrieval_path = Path(args.query_contributor), Path(args.retrieval)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
    height, width = int(retrieval.metadata["token_height"]), int(retrieval.metadata["token_width"])
    camera, oracle_pose = _camera_and_pose(contributor_path)
    rendered = render_soft_child_surface_field(
        physical, field, oracle_pose, camera,
        width=width, height=height, top_l=int(args.top_l), device=str(args.device),
        coordinate_supersample_factor=int(args.coordinate_supersample_factor),
    )
    contributor_rows, contributor_mass, contributor_null, coordinate = _contributor_token_children(
        physical, contributor_path, token_height=height, token_width=width,
        top_l=int(args.top_l),
    )
    query_rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    query_mass = np.asarray(retrieval.token_child_probabilities, dtype=np.float64)
    render_rows = rendered.child_rows.reshape(height * width, -1)
    render_mass = rendered.child_weights.reshape(height * width, -1).astype(np.float64)
    render_null = rendered.null_weight.reshape(-1).astype(np.float64)
    contributor_observed = np.sum(contributor_mass, axis=1)
    render_observed = np.sum(render_mass, axis=1)
    valid_conditional = (contributor_observed > 0.0) & (render_observed > 0.0)
    conditional_agreement = _identity_overlap(
        contributor_rows,
        np.divide(contributor_mass, np.maximum(contributor_observed[:, None], 1e-12)),
        np.zeros_like(contributor_null),
        render_rows,
        np.divide(render_mass, np.maximum(render_observed[:, None], 1e-12)),
        np.zeros_like(render_null),
    )
    retrieval_vs_contributor = _product_overlap(
        query_rows, query_mass, contributor_rows, contributor_mass,
    )
    retrieval_vs_render = _product_overlap(
        query_rows, query_mass, render_rows, render_mass,
    )
    contributor_vs_render = _identity_overlap(
        contributor_rows, contributor_mass, contributor_null,
        render_rows, render_mass, render_null,
    )
    report = {
        "artifact_type": "goal_maplet_matched_child_evidence_ladder_v2",
        "image_id": retrieval.image_id,
        "diagnostic_uses_query_contributor_pose": True,
        "not_a_localization_metric": True,
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "retrieval_content_sha256": retrieval.content_sha256,
        "input_file_sha256": {
            "physical_map": file_sha256(physical_path),
            "canonical_field": file_sha256(field_path),
            "query_contributor": file_sha256(contributor_path),
            "retrieval": file_sha256(retrieval_path),
        },
        "coordinate_audit": coordinate,
        "mean_retrieval_retained_child_mass": float(np.mean(np.sum(query_mass, axis=1))),
        "mean_contributor_retained_child_mass": float(np.mean(contributor_observed)),
        "mean_render_retained_child_mass": float(np.mean(render_observed)),
        "mean_retrieval_times_contributor_child_overlap": float(np.mean(retrieval_vs_contributor)),
        "mean_retrieval_times_oracle_pose_render_child_overlap": float(np.mean(retrieval_vs_render)),
        "mean_contributor_vs_oracle_pose_render_identity_overlap": float(np.mean(contributor_vs_render)),
        "mean_contributor_vs_render_observed_conditional_overlap": float(
            np.mean(conditional_agreement[valid_conditional])
        ),
        "render_typed_mass": {
            "child_tail": float(np.mean(rendered.child_tail_weight)),
            "unassigned_geometry": float(np.mean(rendered.unassigned_geometry_weight)),
            "background": float(np.mean(rendered.background_weight)),
            "canonical_field_missing": float(
                np.mean(rendered.canonical_field_missing_weight)
            ),
            "payload_excluded": float(np.mean(rendered.payload_excluded_weight)),
            "maximum_alpha_overflow": float(rendered.maximum_alpha_overflow),
            "overflow_token_fraction": float(rendered.overflow_token_fraction),
        },
        "interpretation_contract": {
            "retrieval_vs_contributor": "retrieval child evidence loss before online rendering",
            "contributor_vs_render": "coordinate_renderer_and_top4_primitive_truncation_combined_control",
            "retrieval_vs_render": "end_to_end_matched_child_evidence_at_oracle_pose",
            "unknown_is_not_a_positive_matching_class": True,
        },
        "promotion_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
