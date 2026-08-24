"""Replay a full-token ranker and its unweighted retrieval-order union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _basin_recall_metrics,
    _load_feature_artifact,
    _score_queries,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import _metrics
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    continuous_seed_domain_recall_metrics,
    load_fulltoken_candidate_pose_ranker,
    round_robin_union_score_rows,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


SCHEMA = "goal_maplet_fulltoken_ranker_retrieval_union_evaluation_v1"


def _strategy_report(score, arrays, rows):
    return {
        "pose_point_metrics": _metrics(
            score,
            arrays["translation_m"][rows], arrays["rotation_deg"][rows],
            arrays["candidate_valid"][rows], arrays["image_ids"][rows],
        ),
        "distinct_pose_basin_recall": _basin_recall_metrics(score, arrays, rows),
        "continuous_domain_acquisition": continuous_seed_domain_recall_metrics(
            score,
            arrays["candidate_poses_w2c"][rows], arrays["candidate_valid"][rows],
            translation_half_extent_m=8.0, rotation_radius_deg=45.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_queries", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite ranker-union evaluation")
    dataset_path = Path(args.dataset)
    arrays, metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest), dataset_path=dataset_path,
        dataset_content_sha256=str(metadata["content_sha256"]),
    )
    device = torch.device(str(args.device))
    model, model_metadata = load_fulltoken_candidate_pose_ranker(
        Path(args.model), device=device,
    )
    if (
        model_metadata.get("feature_file_sha256") != feature_manifest["feature_file_sha256"]
        or model_metadata.get("dataset_content_sha256") != metadata["content_sha256"]
    ):
        raise ValueError("full-token ranker feature/dataset lineage differs")
    rows = np.arange(arrays["image_ids"].size, dtype=np.int64)
    appearance = _score_queries(
        model, features, rows, batch_queries=int(args.batch_queries), device=device,
    )
    retrieval = np.broadcast_to(
        -np.arange(appearance.shape[1], dtype=np.float32)[None], appearance.shape,
    ).copy()
    union = round_robin_union_score_rows(
        np.stack([appearance, retrieval], axis=1),
        arrays["candidate_poses_w2c"], arrays["candidate_valid"],
    )
    routes = sorted({str(value).split("/", 1)[0] for value in arrays["image_ids"]})
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": metadata["content_sha256"],
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "model_file_sha256": file_sha256(Path(args.model)),
        "model_content_sha256": model_metadata["model_content_sha256"],
        "union_semantics": "unweighted_round_robin_learned_fulltoken_then_retrieval_order_v1",
        "all": {
            name: _strategy_report(score, arrays, rows)
            for name, score in (
                ("retrieval_order", retrieval),
                ("learned_fulltoken", appearance),
                ("round_robin_union", union),
            )
        },
        "per_route": {
            route: {
                name: _strategy_report(score[mask], arrays, rows[mask])
                for name, score in (
                    ("retrieval_order", retrieval),
                    ("learned_fulltoken", appearance),
                    ("round_robin_union", union),
                )
            }
            for route in routes
            for mask in [np.asarray([
                str(value).split("/", 1)[0] == route for value in arrays["image_ids"]
            ], dtype=bool)]
        },
        "all_candidate_features_frozen_before_pose_metrics": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "multi_basin_acquisition_diagnostic_not_pose_search_success",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: report["all"][name]["continuous_domain_acquisition"]
        for name in report["all"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
