"""Evaluate raster-scale consensus without learning on the evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _pose_key
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _normalize(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    return (value - np.mean(value)) / max(float(np.std(value)), 1.0e-8)


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
        "selected_original_indices": [int(row["original_index"]) for row in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_report", required=True)
    parser.add_argument("--supersampled_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase scale consensus")
    base = json.loads(Path(args.base_report).read_text())
    sampled = json.loads(Path(args.supersampled_report).read_text())
    for key in ("physical_map_sha256", "canonical_field_sha256"):
        if base.get(key) != sampled.get(key):
            raise ValueError(f"phase reports differ: {key}")
    base_rows = {str(row["image_id"]): row for row in base.get("rows", [])}
    sampled_rows = {str(row["image_id"]): row for row in sampled.get("rows", [])}
    if set(base_rows) != set(sampled_rows):
        raise ValueError("phase reports contain different queries")
    rows = []
    for image_id in sorted(base_rows):
        base_details = base_rows[image_id]["mode_details"]["actual_parent_actual_child"]
        sampled_details = sampled_rows[image_id]["mode_details"]["actual_parent_actual_child"]
        base_by_pose = {_pose_key(row["pose_w2c"]): row for row in base_details if row.get("surface_alignment_score") is not None}
        sampled_by_pose = {_pose_key(row["pose_w2c"]): row for row in sampled_details if row.get("surface_alignment_score") is not None}
        if set(base_by_pose) != set(sampled_by_pose):
            raise ValueError(f"phase candidates differ: {image_id}")
        keys = list(base_by_pose)
        score_1x = np.asarray([base_by_pose[key]["surface_alignment_score"] for key in keys], dtype=np.float64)
        score_2x = np.asarray([sampled_by_pose[key]["surface_alignment_score"] for key in keys], dtype=np.float64)
        z1, z2 = _normalize(score_1x), _normalize(score_2x)
        policies = {
            "base_1x": score_1x,
            "supersample_2x": score_2x,
            "raw_mean": 0.5 * (score_1x + score_2x),
            "equal_scale_zmean": 0.5 * (z1 + z2),
            "robust_zmin": np.minimum(z1, z2),
            "zmean_disagreement_penalty_0.25": 0.5 * (z1 + z2) - 0.25 * np.abs(z1 - z2),
        }
        result = {"image_id": image_id, "policies": {}, "score_rank_correlation": float(np.corrcoef(z1, z2)[0, 1])}
        for name, score in policies.items():
            index = int(np.argmax(score))
            detail = base_by_pose[keys[index]]
            result["policies"][name] = {
                "original_index": int(index),
                "translation_m": float(detail["translation_m"]),
                "rotation_deg": float(detail["rotation_deg"]),
                "score": float(score[index]),
            }
        rows.append(result)
    policy_names = list(rows[0]["policies"]) if rows else []
    result = {
        "stage": "goal_maplet_phase_scale_consensus_g19_c",
        "selection_split_used_for_parameter_training": False,
        "promotion_allowed": False,
        "base_report_sha256": file_sha256(Path(args.base_report)),
        "supersampled_report_sha256": file_sha256(Path(args.supersampled_report)),
        "query_count": len(rows),
        "score_rank_correlation_median": float(np.median([row["score_rank_correlation"] for row in rows])),
        "summary": {name: _metrics(rows, name) for name in policy_names},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
