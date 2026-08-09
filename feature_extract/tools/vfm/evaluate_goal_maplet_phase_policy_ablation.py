"""Evaluate fixed phase symmetries from stored exact-render components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _score_policies(component: dict[str, float]) -> dict[str, float]:
    single = np.asarray([component["horizontal_phase"], component["vertical_phase"]], dtype=np.float64)
    diagonal = np.asarray([component["diagonal_down_phase"], component["diagonal_up_phase"]], dtype=np.float64)
    step2 = np.asarray([component["horizontal_phase_step2"], component["vertical_phase_step2"]], dtype=np.float64)
    result = {
        "isotropic_single_scale": float(np.mean(single)),
        "isotropic_single_plus_diagonal": float(np.mean(np.r_[single, diagonal])),
        "isotropic_all_offsets": float(np.mean(np.r_[single, diagonal, step2])),
    }
    if "phase_visible" in component:
        result["conditional_visible_isotropic"] = float(component["phase_visible"])
        # A neutral, parameter-free observability combination.  This equals
        # fixed-grid phase only when horizontal/vertical conditional scores
        # and visibility are identical, so it is reported as an ablation and
        # never silently substituted for the selected runtime policy.
        result["conditional_times_mean_observability"] = float(component["phase_visible"] * component["phase_observability"])
    return result


def _metrics(rows: list[dict[str, object]], policy: str) -> dict[str, object]:
    selected = [row["policies"][policy] for row in rows]
    translation = np.asarray([row["translation_m"] for row in selected], dtype=np.float64)
    rotation = np.asarray([row["rotation_deg"] for row in selected], dtype=np.float64)
    return {
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_0.5m_5deg": float(np.mean((translation <= 0.5) & (rotation <= 5.0))),
        "success_1m_10deg": float(np.mean((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_rate": float(np.mean((translation > 5.0) | (rotation > 30.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase-policy ablation")
    source = json.loads(Path(args.input_report).read_text())
    rows = []
    for source_row in source.get("rows", []):
        name = "actual_parent_actual_child"
        details = source_row.get("mode_details", {}).get(name, [])
        diagnostics = source_row.get("ranking_diagnostics", {}).get(name, {})
        components = diagnostics.get("surface_phase_components_preorder") or []
        original = diagnostics.get("surface_alignment_original_indices") or []
        candidates = []
        for rank, detail in enumerate(details):
            source_index = int(original[rank]) if rank < len(original) else -1
            if 0 <= source_index < len(components):
                candidates.append((detail, components[source_index]))
        if not candidates:
            raise ValueError(f"report lacks phase components: {source_row.get('image_id')}")
        policy_scores = [_score_policies(component) for _, component in candidates]
        names = list(policy_scores[0])
        row = {"image_id": str(source_row["image_id"]), "policies": {}}
        for policy in names:
            index = int(np.argmax([score[policy] for score in policy_scores]))
            detail, _ = candidates[index]
            row["policies"][policy] = {
                "candidate_index": index,
                "translation_m": float(detail["translation_m"]),
                "rotation_deg": float(detail["rotation_deg"]),
                "score": float(policy_scores[index][policy]),
            }
        rows.append(row)
    names = list(rows[0]["policies"]) if rows else []
    result = {
        "stage": "goal_maplet_phase_fixed_symmetry_ablation_g19_c",
        "input_report": str(Path(args.input_report)),
        "query_count": len(rows),
        "parameter_training_used": False,
        "promotion_allowed": False,
        "summary": {name: _metrics(rows, name) for name in names},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
