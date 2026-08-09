"""Create a parameter-free phase operator bound to one feature-cross-fit fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)


def _policy(
    operator: str,
    *,
    physical_map_sha256: str,
    canonical_field_sha256: str,
    physical_instance_readout_sha256: str,
) -> dict[str, object]:
    common = {
        "physical_map_sha256": physical_map_sha256,
        "canonical_field_sha256": canonical_field_sha256,
        "physical_instance_readout_sha256": physical_instance_readout_sha256,
        "intercept": 0.0,
        "required_render_supersample_factor": 2,
        "render_protocol": "complete_clean_2dgs_fractional_supersample_pool_v2",
        "absolute_image_coordinates_used": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "feature_pipeline_crossfit_binding": True,
        "operator_parameters_refit": False,
        "operator_parameter_source": "analytic_parameter_free",
        "ranking_only": True,
        "null_or_abstention_included": False,
    }
    if operator == "directional":
        return {
            **common,
            "artifact_type": "goal_maplet_phase_readout_policy_v1",
            "component_names": ["horizontal_phase", "vertical_phase"],
            "standardizer_mean": [0.0, 0.0],
            "standardizer_scale": [1.0, 1.0],
            "coefficient": [0.5, 0.5],
            "selected_variant": "parameter_free_directional_axis_mean",
            "phase_definition": "equal_axis_vfm_directional_phase_v1",
            "historical_g19c_learned_coefficients_used": False,
        }
    if operator == "jacobian":
        return {
            **common,
            "artifact_type": "goal_maplet_phase_readout_policy_v2",
            "role": "phase_residual_only",
            "component_names": ["jacobian_phase_visible"],
            "standardizer_mean": [0.0],
            "standardizer_scale": [1.0],
            "coefficient": [1.0],
            "phase_definition": (
                "query_carrier_weighted_vfm_jacobian_frobenius_cosine_v1"
            ),
            "carrier_is_candidate_independent": True,
            "observability_is_separate_from_phase": True,
        }
    raise ValueError("unsupported phase operator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("directional", "jacobian"), required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite map-cross-fit phase policy")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    if field.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map differ")
    result = _policy(
        str(args.operator),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256=field.content_sha256,
        physical_instance_readout_sha256=file_sha256(
            Path(args.physical_instance_readout)
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
