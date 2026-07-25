"""Audit frozen P1 visual edges with an OOF coherent-pose group probe.

This is a deliberately train-only diagnostic.  It consumes the target-free
candidate-edge feature shards exported by
``audit_p1_candidate_edge_representation_crossfit.py`` and joins the
train-only coherent-repeat targets only after those frozen visual forwards.

Unlike the earlier edge-wise diagonal probe, the single linear score here is
optimized through the same failure mode used by the hard-pose audit:

* competing wrong candidate identities are reduced by a normalized soft-min
  for every query point; and
* those strongest-wrong margins are averaged within one coherent wrong pose.

The result is never a checkpoint, runtime scorer, PnP input, validation/test
evaluation, or a promotion decision.  It only tells us whether the already
exported visual evidence has *any* held-query separability under the actual
coherent-repeat grouping before another visual architecture is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_p1_candidate_edge_representation_crossfit import (
    AUDIT_FORMAT as EDGE_REPRESENTATION_AUDIT_FORMAT,
    PROFILE_SOURCES,
    CandidateEdgeProbeQueryFeatures,
    _feature_filename,
    _load_training_feature,
    _pair_feature_differences,
    _probe_gate,
    _profile_summary,
    aggregate_hard_pose_group_gaps,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _write_json_atomically,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_edge_group_linear_probe import (
    CANDIDATE_EDGE_GROUP_LINEAR_PROBE_FORMAT,
    CandidateEdgeGroupLinearBatch,
    CandidateEdgeGroupLinearProbe,
    fit_candidate_edge_group_linear_probe,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)


AUDIT_FORMAT = "p1_candidate_edge_group_linear_crossfit_audit_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-audit",
        required=True,
        help="Completed s1572-style target-free frozen candidate-edge audit JSON.",
    )
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crossfit-fold-count", type=int, default=5)
    parser.add_argument("--minimum-points-per-pose", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--softmin-temperature", type=float, default=0.1)
    parser.add_argument("--train-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--minimum-eligible-query-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--catastrophic-gap-threshold", type=float, default=-0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_feature_audit_contract(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Load only a completed, strictly target-free upstream feature audit."""

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("candidate-edge group probe feature audit is unreadable") from error
    if not isinstance(payload, Mapping):
        raise ValueError("candidate-edge group probe feature audit is not an object")
    result = dict(payload)
    protocol = result.get("protocol")
    lineage = result.get("feature_lineage")
    if (
        result.get("format") != EDGE_REPRESENTATION_AUDIT_FORMAT
        or not isinstance(protocol, Mapping)
        or not isinstance(lineage, Mapping)
        or protocol.get("diagnostic_only") is not True
        or protocol.get("runtime_layout_target_free") is not True
        or protocol.get("raw_feature_artifacts_contain_targets") is not False
        or protocol.get("target_join_after_visual_inference") is not True
        or protocol.get("runtime_scorer_must_not_load_feature_artifacts") is not True
        or protocol.get("train_query_only") is not True
        or protocol.get("heldout_validation_or_test_not_run") is not True
        or protocol.get("pnp_or_pose_estimation_run") is not False
        or protocol.get("no_render") is not True
        or protocol.get("no_image_retrieval_or_submap") is not True
    ):
        raise ValueError("candidate-edge group probe feature audit contract is invalid")
    required_lineage = {
        "layout_sha256",
        "identity_checkpoint_sha256",
        "candidate_count",
        "support_view_count",
        "feature_sources",
        "source_lineage",
        "visual_content_controls",
    }
    if required_lineage - set(lineage):
        raise ValueError("candidate-edge group probe feature audit lineage is incomplete")
    query_ids = result.get("query_ids")
    if (
        not isinstance(query_ids, list)
        or not query_ids
        or len({str(query_id) for query_id in query_ids}) != len(query_ids)
        or int(result.get("query_count", 0)) != len(query_ids)
    ):
        raise ValueError("candidate-edge group probe feature audit query table is invalid")
    return result, dict(lineage)


def _group_linear_batch(
    *,
    queries: Mapping[str, CandidateEdgeProbeQueryFeatures],
    hard_targets: Mapping[str, HardRepeatQueryTargets],
    query_ids: Sequence[str],
    profile: str,
    minimum_points: int,
) -> tuple[CandidateEdgeGroupLinearBatch, dict[str, int]]:
    """Create one train-only soft-min grouping over frozen normal features."""

    if str(profile) not in PROFILE_SOURCES or int(minimum_points) < 2:
        raise ValueError("candidate-edge group probe batch configuration is invalid")
    features: list[np.ndarray] = []
    edge_to_point: list[int] = []
    point_to_pose: list[int] = []
    selected_pose_groups = 0
    selected_queries = 0
    for query_id in query_ids:
        if query_id not in queries or query_id not in hard_targets:
            raise ValueError("candidate-edge group probe query target join is incomplete")
        query = queries[query_id]
        targets = hard_targets[query_id]
        normal, normal_active = _pair_feature_differences(
            query=query,
            targets=targets,
            profile=profile,
            branch="normal",
        )
        if not bool(np.any(normal_active)):
            continue
        query_has_group = False
        pair_ids = np.asarray(targets.pair_ids, dtype=np.int64)
        source_ids = np.asarray(targets.source_point_ids, dtype=np.int64)
        for pair_id in np.unique(pair_ids).tolist():
            pair_rows = np.flatnonzero((pair_ids == int(pair_id)) & normal_active)
            active_source_ids = np.unique(source_ids[pair_rows])
            if len(active_source_ids) < int(minimum_points):
                continue
            # ``point_to_pose`` must contain dense pose IDs.  It is tempting
            # to use its current length, but that is a point count and would
            # leave empty pose segments as soon as a prior pose has >1 point.
            pose_index = selected_pose_groups
            for source_id in active_source_ids.tolist():
                source_rows = pair_rows[source_ids[pair_rows] == int(source_id)]
                if len(source_rows) == 0:
                    raise RuntimeError("candidate-edge group probe lost an active source point")
                point_index = len(point_to_pose)
                point_to_pose.append(pose_index)
                for row in source_rows.tolist():
                    features.append(np.asarray(normal[int(row)], dtype=np.float32))
                    edge_to_point.append(point_index)
            selected_pose_groups += 1
            query_has_group = True
        selected_queries += int(query_has_group)
    if not features or not point_to_pose or not edge_to_point:
        raise RuntimeError("candidate-edge group probe has no usable coherent pose group")
    values = np.stack(features, axis=0).astype(np.float32, copy=False)
    batch = CandidateEdgeGroupLinearBatch(
        edge_features=torch.from_numpy(values),
        edge_to_point=torch.tensor(edge_to_point, dtype=torch.long),
        point_to_pose=torch.tensor(point_to_pose, dtype=torch.long),
    )
    return batch, {
        "train_query_count_with_usable_pose": int(selected_queries),
        "train_pose_group_count": int(selected_pose_groups),
        "train_point_count": int(batch.point_count),
        "train_edge_count": int(batch.edge_count),
        "feature_dimension": int(batch.feature_dimension),
    }


def _probe_margin(probe: CandidateEdgeGroupLinearProbe, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(probe.weights) or not np.isfinite(values).all():
        raise ValueError("candidate-edge group probe score inputs are invalid")
    with torch.inference_mode():
        output = probe.score(torch.from_numpy(values)).cpu().numpy()
    margins = np.asarray(output, dtype=np.float64).reshape(-1)
    if margins.shape != (len(values),) or not np.isfinite(margins).all():
        raise RuntimeError("candidate-edge group probe score is invalid")
    return margins


def crossfit_group_profile(
    *,
    queries: Mapping[str, CandidateEdgeProbeQueryFeatures],
    hard_targets: Mapping[str, HardRepeatQueryTargets],
    profile: str,
    fold_count: int,
    minimum_points: int,
    device: torch.device | str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    softmin_temperature: float,
    train_margin: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Query-grouped OOF fit and exact hard-min held-query evaluation."""

    query_ids = tuple(sorted(queries))
    if (
        set(query_ids) != set(hard_targets)
        or int(fold_count) < 2
        or int(fold_count) > len(query_ids)
        or str(profile) not in PROFILE_SOURCES
    ):
        raise ValueError("candidate-edge group probe cross-fit contract is invalid")
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    for fold in range(int(fold_count)):
        train_ids = [query_id for index, query_id in enumerate(query_ids) if index % fold_count != fold]
        held_ids = [query_id for index, query_id in enumerate(query_ids) if index % fold_count == fold]
        batch, batch_stats = _group_linear_batch(
            queries=queries,
            hard_targets=hard_targets,
            query_ids=train_ids,
            profile=profile,
            minimum_points=int(minimum_points),
        )
        probe, fit_stats = fit_candidate_edge_group_linear_probe(
            batch=batch,
            device=device,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            softmin_temperature=float(softmin_temperature),
            margin=float(train_margin),
            seed=int(seed) + int(fold),
        )
        fits.append(
            {
                "fold": int(fold),
                "train_query_count": int(len(train_ids)),
                "held_query_count": int(len(held_ids)),
                "probe_format": CANDIDATE_EDGE_GROUP_LINEAR_PROBE_FORMAT,
                **batch_stats,
                **fit_stats,
            }
        )
        for query_id in held_ids:
            query = queries[query_id]
            targets = hard_targets[query_id]
            normal, normal_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="normal"
            )
            permuted, permuted_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="permuted"
            )
            position, position_active = _pair_feature_differences(
                query=query, targets=targets, profile=profile, branch="position"
            )
            common_active = normal_active & permuted_active & position_active
            normal_groups, permuted_groups, position_groups, active_points = (
                aggregate_hard_pose_group_gaps(
                    normal_margins=_probe_margin(probe, normal),
                    permuted_margins=_probe_margin(probe, permuted),
                    position_margins=_probe_margin(probe, position),
                    common_active=common_active,
                    targets=targets,
                    minimum_points=int(minimum_points),
                )
            )
            if len(normal_groups) == 0:
                rows.append(
                    {
                        "query_id": query_id,
                        "fold": int(fold),
                        "eligible": False,
                        "common_active_edge_count": int(np.sum(common_active)),
                        "pose_group_count": 0,
                        "active_point_count": 0,
                    }
                )
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "fold": int(fold),
                    "eligible": True,
                    "common_active_edge_count": int(np.sum(common_active)),
                    "pose_group_count": int(len(normal_groups)),
                    "active_point_count": int(active_points),
                    "normal_gap": float(np.mean(normal_groups)),
                    "permuted_gap": float(np.mean(permuted_groups)),
                    "position_gap": float(np.mean(position_groups)),
                }
            )
    rows.sort(key=lambda row: str(row["query_id"]))
    return rows, fits


def audit_p1_candidate_edge_group_linear_crossfit(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "feature_audit": Path(args.feature_audit),
        "layout": Path(args.rgb_spatial_layout),
        "targets": Path(args.training_targets),
        "hard_repeat": Path(args.hard_repeat_targets),
    }
    output_dir = Path(args.output_dir)
    values = (
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.softmin_temperature),
        float(args.train_margin),
        float(args.minimum_eligible_query_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_win_fraction),
        float(args.minimum_visual_gap_delta),
        float(args.catastrophic_gap_threshold),
    )
    if (
        any(not path.exists() for path in paths.values())
        or output_dir.exists() and not bool(args.force)
        or int(args.crossfit_fold_count) < 2
        or int(args.minimum_points_per_pose) < 2
        or int(args.epochs) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.softmin_temperature) <= 0.0
        or not 0.0 < float(args.minimum_eligible_query_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or not 0.0 < float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_visual_gap_delta) < 0.0
    ):
        raise ValueError("candidate-edge group probe arguments are invalid")
    upstream, feature_lineage = _load_feature_audit_contract(paths["feature_audit"])
    layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
    targets = load_candidate_pose_rgb_spatial_training_targets(paths["targets"])
    layout_sha256 = file_sha256_short(paths["layout"])
    targets_sha256 = file_sha256_short(paths["targets"])
    validate_training_layout_and_targets(
        layout=layout, targets=targets, layout_sha256=layout_sha256
    )
    if str(feature_lineage.get("layout_sha256", "")) != layout_sha256:
        raise ValueError("candidate-edge group probe feature layout lineage is stale")
    if int(feature_lineage.get("candidate_count", 0)) != int(layout.candidate_count):
        raise ValueError("candidate-edge group probe candidate-count lineage is stale")
    if int(feature_lineage.get("support_view_count", 0)) != int(layout.support_view_count):
        raise ValueError("candidate-edge group probe support-view lineage is stale")
    hard_repeat = load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["hard_repeat"])
    hard_targets = build_hard_repeat_query_targets(
        layout=layout,
        targets=targets,
        hard_repeat_targets=hard_repeat,
        layout_sha256=layout_sha256,
        targets_sha256=targets_sha256,
    )
    groups = build_train_query_groups(layout=layout, targets=targets)
    query_ids = tuple(sorted(groups))
    upstream_ids = tuple(sorted(str(value) for value in upstream["query_ids"]))
    if (
        query_ids != upstream_ids
        or set(query_ids) != set(hard_targets)
        or int(args.crossfit_fold_count) > len(query_ids)
        or any(int(group.point_count) != 32 for group in groups.values())
    ):
        raise ValueError("candidate-edge group probe frozen P1 query contract differs from feature audit")
    upstream_target_lineage = upstream.get("target_lineage")
    if (
        not isinstance(upstream_target_lineage, Mapping)
        or str(upstream_target_lineage.get("training_targets_sha256", "")) != targets_sha256
        or str(upstream_target_lineage.get("hard_repeat_targets_sha256", ""))
        != file_sha256_short(paths["hard_repeat"])
        or upstream_target_lineage.get("target_join_after_visual_inference") is not True
    ):
        raise ValueError("candidate-edge group probe upstream target lineage is stale")
    feature_dir = paths["feature_audit"].parent / "training_features"
    if not feature_dir.is_dir():
        raise ValueError("candidate-edge group probe feature audit lacks feature shards")
    queries = {
        query_id: _load_training_feature(
            path=feature_dir / _feature_filename(query_id),
            expected_query_id=query_id,
            expected_lineage=feature_lineage,
        )
        for query_id in query_ids
    }
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("candidate-edge group probe requested CUDA but it is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, object] = {}
    for profile in PROFILE_SOURCES:
        rows, fits = crossfit_group_profile(
            queries=queries,
            hard_targets=hard_targets,
            profile=profile,
            fold_count=int(args.crossfit_fold_count),
            minimum_points=int(args.minimum_points_per_pose),
            device=device,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            softmin_temperature=float(args.softmin_temperature),
            train_margin=float(args.train_margin),
            seed=int(args.seed),
        )
        summary = _profile_summary(
            rows=rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
        )
        profiles[profile] = {
            "sources": list(PROFILE_SOURCES[profile]),
            "fold_fits": fits,
            "oof_rows": rows,
            "summary": summary,
            "gate": _probe_gate(summary=summary, total_query_count=len(query_ids), args=args),
        }
    passed = [
        profile for profile, value in profiles.items() if bool(value["gate"]["passed"])
    ]
    result: dict[str, object] = {
        "format": AUDIT_FORMAT,
        "stage": "frozen_current_p1_candidate_edge_group_softmin_linear_query_grouped_crossfit",
        "output_dir": str(output_dir),
        "query_count": int(len(query_ids)),
        "query_ids": list(query_ids),
        "upstream_feature_audit": {
            "path": str(paths["feature_audit"]),
            "sha256": file_sha256_short(paths["feature_audit"]),
            "format": EDGE_REPRESENTATION_AUDIT_FORMAT,
        },
        "feature_lineage": feature_lineage,
        "target_lineage": {
            "training_targets_sha256": targets_sha256,
            "hard_repeat_targets_sha256": file_sha256_short(paths["hard_repeat"]),
            "target_join_after_visual_inference": True,
        },
        "probe_configuration": {
            "probe_format": CANDIDATE_EDGE_GROUP_LINEAR_PROBE_FORMAT,
            "crossfit_fold_count": int(args.crossfit_fold_count),
            "predeclared_profiles": {
                name: list(values) for name, values in PROFILE_SOURCES.items()
            },
            "single_linear_visual_score": True,
            "train_only_grouped_softmin_objective": (
                "per_point_normalized_softmin_over_coherent_wrong_candidates_then_pose_mean_v1"
            ),
            "held_query_metric": "exact_per_point_strongest_wrong_then_pose_mean_v1",
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "softmin_temperature": float(args.softmin_temperature),
            "train_margin": float(args.train_margin),
            "minimum_points_per_coherent_pose": int(args.minimum_points_per_pose),
            "selection_uses_train_folds_only": True,
        },
        "profiles": profiles,
        "passed_representation_profiles": passed,
        "protocol": {
            "diagnostic_only": True,
            "model_weights_updated": False,
            "runtime_layout_target_free": True,
            "raw_feature_artifacts_contain_targets": False,
            "target_join_after_visual_inference": True,
            "train_query_only": True,
            "heldout_validation_or_test_not_run": True,
            "pnp_or_pose_estimation_run": False,
            "runtime_scorer_must_not_load_feature_artifacts": True,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
        },
    }
    _write_json_atomically(output_dir / "audit.json", result)
    return {
        "output_dir": str(output_dir),
        "passed_representation_profiles": passed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_p1_candidate_edge_group_linear_crossfit(parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
