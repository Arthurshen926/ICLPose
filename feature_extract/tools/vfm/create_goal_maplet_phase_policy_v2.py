"""Create the parameter-free G20 orientation-equivariant phase policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_policy", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase policy")
    base_path = Path(args.base_policy)
    base = json.loads(base_path.read_text())
    required = (
        "physical_map_sha256",
        "canonical_field_sha256",
        "physical_instance_readout_sha256",
    )
    if any(not base.get(key) for key in required):
        raise ValueError("base phase policy lacks map lineage")
    result = {
        "artifact_type": "goal_maplet_phase_readout_policy_v2",
        "role": "phase_residual_only",
        "component_names": ["jacobian_phase_visible"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [1.0],
        "intercept": 0.0,
        "required_render_supersample_factor": 2,
        "render_protocol": "complete_clean_2dgs_fractional_supersample_pool_v2",
        "phase_definition": "query_carrier_weighted_vfm_jacobian_frobenius_cosine_v1",
        "carrier_is_candidate_independent": True,
        "observability_is_separate_from_phase": True,
        "absolute_image_coordinates_used": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "base_policy_sha256": file_sha256(base_path),
        **{key: base[key] for key in required},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
