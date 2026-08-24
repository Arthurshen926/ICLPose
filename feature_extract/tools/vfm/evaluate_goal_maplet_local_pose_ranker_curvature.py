"""Audit whether a locally supervised full-token score is concave near GT.

This is a held-label diagnostic, not a production-time operation.  It reuses
the frozen 73-point complete central-difference subset embedded in the v2
local supervision design and fits the full six-dimensional Hessian of
``loss=-score`` independently for every query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_feature_artifact,
    _load_local_supervision_dataset,
    _score_queries,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    load_fulltoken_candidate_pose_ranker,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    LOCAL_POSE_SUPERVISION_SEMANTICS,
    complete_quadratic_candidate_indices,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    fit_complete_local_se3_quadratic,
)


SCHEMA = "goal_maplet_local_pose_ranker_complete_curvature_audit_v1"


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    positive = np.asarray([bool(row["positive_definite"]) for row in rows])
    anchor = np.asarray([bool(row["anchor_is_highest_complete_probe"]) for row in rows])
    minimum = np.asarray([float(row["minimum_hessian_eigenvalue"]) for row in rows])
    finite_bias = np.asarray([
        float(row["predicted_bias_joint_norm"])
        for row in rows if row["predicted_bias_joint_norm"] is not None
    ], dtype=np.float64)
    return {
        "query_count": len(rows),
        "positive_definite_rate": float(np.mean(positive)),
        "anchor_is_highest_complete_probe_rate": float(np.mean(anchor)),
        "median_minimum_hessian_eigenvalue": float(np.median(minimum)),
        "median_predicted_bias_joint_norm_when_positive": (
            None if not finite_bias.size else float(np.median(finite_bias))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_queries", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite local curvature audit")
    dataset_path = Path(args.dataset)
    arrays, metadata = _load_local_supervision_dataset(dataset_path)
    if metadata.get("supervision_semantics") != LOCAL_POSE_SUPERVISION_SEMANTICS:
        raise ValueError("local curvature audit requires the current v2 design")
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest), dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    device = torch.device(str(args.device))
    model, model_metadata = load_fulltoken_candidate_pose_ranker(
        Path(args.model), device=device,
    )
    if (
        not bool(model_metadata.get("local_supervision", False))
        or model_metadata.get("local_pose_supervision_semantics")
        != LOCAL_POSE_SUPERVISION_SEMANTICS
        or model_metadata.get("feature_file_sha256")
        != feature_manifest["feature_file_sha256"]
        or model_metadata.get("dataset_content_sha256") != metadata["content_sha256"]
    ):
        raise ValueError("local curvature model lineage differs")
    query_rows = np.arange(arrays["image_ids"].size, dtype=np.int64)
    scores = _score_queries(
        model, features, query_rows,
        batch_queries=int(args.batch_queries), device=device,
    )
    index = complete_quadratic_candidate_indices()
    rows = []
    for query, image_id in enumerate(arrays["image_ids"].tolist()):
        values = {name: float(scores[query, row]) for name, row in index.items()}
        fit = fit_complete_local_se3_quadratic(values)
        predicted = np.asarray(fit.predicted_bias_normalized, dtype=np.float64)
        rows.append({
            "image_id": str(image_id),
            "route": str(image_id).split("/", 1)[0],
            "positive_definite": bool(fit.positive_definite),
            "minimum_hessian_eigenvalue": float(fit.hessian_eigenvalues[0]),
            "maximum_hessian_eigenvalue": float(fit.hessian_eigenvalues[-1]),
            "hessian_condition_number": float(fit.hessian_condition_number),
            "loss_gradient_norm": float(np.linalg.norm(fit.loss_gradient)),
            "predicted_bias_joint_norm": (
                float(np.linalg.norm(predicted))
                if np.all(np.isfinite(predicted)) else None
            ),
            "anchor_is_highest_complete_probe": bool(
                values["center"] >= max(values.values()) - 1.0e-8
            ),
        })
    routes = sorted({str(row["route"]) for row in rows})
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "model_file_sha256": file_sha256(Path(args.model)),
        "model_content_sha256": model_metadata["model_content_sha256"],
        "coordinate_scales": {"translation_m": 1.0, "rotation_deg": 10.0},
        "complete_probe_count": len(index),
        "all": _summary(rows),
        "per_route": {
            route: _summary([row for row in rows if row["route"] == route])
            for route in routes
        },
        "rows": rows,
        "target_pose_used_only_for_diagnostic_supervision": True,
        "production_eligible": False,
        "claim": "local_score_curvature_diagnostic_not_localization_success",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all": report["all"], "per_route": report["per_route"]}, indent=2))


if __name__ == "__main__":
    main()
