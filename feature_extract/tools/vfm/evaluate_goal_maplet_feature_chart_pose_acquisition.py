"""Evaluate child co-visibility plus canonical RADIO feature chart retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    K_VALUES,
    THRESHOLDS,
    _hit,
    _pose_errors,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.visibility_feature_atlas import (
    CanonicalFeatureVisibilityPoseAtlas,
    score_canonical_feature_visibility_pose_atlas,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    diverse_pose_rows,
    score_visibility_pose_atlas,
)
from feature_extract.vfm.tokens import TokenBankManifest, compute_file_sha256


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child_atlas", required=True)
    parser.add_argument("--feature_atlas", required=True)
    parser.add_argument("--retrieval_run", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--feature_weight", type=float, default=0.5)
    parser.add_argument("--maximum_modes", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite feature-chart evaluation")
    feature_weight = float(args.feature_weight)
    if not 0.0 <= feature_weight <= 1.0:
        raise ValueError("feature_weight must lie in [0,1]")
    child_atlas = ChildVisibilityPoseAtlas.load_npz(Path(args.child_atlas))
    feature_atlas = CanonicalFeatureVisibilityPoseAtlas.load_npz(Path(args.feature_atlas))
    if (
        feature_atlas.physical_map_sha256 != child_atlas.physical_map_sha256
        or not np.array_equal(feature_atlas.poses_w2c, child_atlas.poses_w2c)
    ):
        raise ValueError("child and feature visibility atlases differ")
    run_path = Path(args.retrieval_run)
    run = json.loads(run_path.read_text())
    records = run.get("rows")
    if not isinstance(records, list):
        raise ValueError("retrieval run has no rows")
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate()
    token_by_id = {record.image_id: record for record in manifest.records}
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device)
    )
    gt_values = parse_cambridge_pose_file(Path(args.query_pose_file))
    gt_by_id = {record.image_id: record.pose_w2c for record in gt_values}
    matrices = child_atlas.sparse_matrices()
    recall = {name: {str(k): 0 for k in K_VALUES} for name in THRESHOLDS}
    oracle = {name: 0 for name in THRESHOLDS}
    top1_translation, top1_rotation = [], []
    best_translation, best_rotation = [], []
    rows_out = []
    for source in records:
        image_id = str(source["image_id"])
        if image_id not in token_by_id or image_id not in gt_by_id:
            raise KeyError(f"missing query token/pose record {image_id}")
        retrieval = PureRadioPhysicalRetrieval.load_npz(Path(source["artifact"]))
        raw = _load_raw_final(token_by_id[image_id].token_path, "radio_final")
        query = np.asarray(mapper.project(raw).measurement_context, dtype=np.float32)
        set_score = score_visibility_pose_atlas(
            child_atlas, retrieval, layout_weight=0.0,
            selected_children_only=True, matrices=matrices,
        )[0]
        feature_score = score_canonical_feature_visibility_pose_atlas(
            feature_atlas, query
        )
        feature_unit = 0.5 * (feature_score + 1.0)
        combined = (1.0 - feature_weight) * set_score + feature_weight * feature_unit
        selected = diverse_pose_rows(
            child_atlas.poses_w2c, combined,
            maximum_modes=int(args.maximum_modes),
            translation_nms_m=0.5, rotation_nms_deg=5.0,
        )
        translation_all, rotation_all = _pose_errors(
            child_atlas.poses_w2c, gt_by_id[image_id]
        )
        translation, rotation = translation_all[selected], rotation_all[selected]
        for name, threshold in THRESHOLDS.items():
            oracle[name] += int(_hit(translation_all, rotation_all, threshold))
            for k in K_VALUES:
                recall[name][str(k)] += int(_hit(translation[:k], rotation[:k], threshold))
        top1_translation.append(float(translation[0]))
        top1_rotation.append(float(rotation[0]))
        scale = np.maximum(translation / 2.0, rotation / 45.0)
        best = int(np.argmin(scale))
        best_translation.append(float(translation[best]))
        best_rotation.append(float(rotation[best]))
        rows_out.append({
            "image_id": image_id,
            "mode_rows": selected.tolist(),
            "combined_scores": combined[selected].astype(float).tolist(),
            "child_set_scores": set_score[selected].astype(float).tolist(),
            "canonical_feature_scores": feature_score[selected].astype(float).tolist(),
            "translation_m": translation.astype(float).tolist(),
            "rotation_deg": rotation.astype(float).tolist(),
        })
    count = len(rows_out)
    metrics = {}
    for name in THRESHOLDS:
        values = {
            f"recall_at_{k}": float(recall[name][str(k)] / count) for k in K_VALUES
        }
        values["atlas_oracle_recall"] = float(oracle[name] / count)
        metrics[name] = values
    report = {
        "artifact_type": "goal_maplet_canonical_feature_chart_pose_acquisition_v1",
        "query_count": count,
        "method": "child_covisibility_plus_frozen_canonical_radio_surface_grid",
        "feature_weight": feature_weight,
        "maximum_modes": int(args.maximum_modes),
        "metrics": metrics,
        "error_summary": {
            "top1_median_translation_m": float(np.median(top1_translation)),
            "top1_median_rotation_deg": float(np.median(top1_rotation)),
            "best_pool_median_translation_m": float(np.median(best_translation)),
            "best_pool_median_rotation_deg": float(np.median(best_rotation)),
        },
        "child_atlas_content_sha256": child_atlas.content_sha256,
        "feature_atlas_content_sha256": feature_atlas.content_sha256,
        "retrieval_run_file_sha256": _sha(run_path),
        "query_manifest_file_sha256": _sha(Path(args.query_manifest)),
        "surface_mapper_file_sha256": compute_file_sha256(Path(args.surface_mapper)),
        "query_pose_file_sha256": _sha(Path(args.query_pose_file)),
        "mapper_metadata": mapper_metadata,
        "gt_access_scope": "evaluation_only_after_pose_free_child_and_feature_scores",
        "claims": {
            "uses_mapping_rgb": False,
            "uses_reference_image_retrieval": False,
            "uses_alike": False,
            "uses_pnp": False,
            "chart_centres_are_final_pose": False,
            "measures_acquisition_not_final_localization": True,
        },
        "rows": rows_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
