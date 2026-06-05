"""Run SuperPoint keypoint usefulness experiments A/B/C/D."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, estimate_pose_pnp_fixed, pnp_pose_error
from feature_extract.vfm.superpoint_keypoint_abcd import (
    best_action_label,
    finite_float,
    group_rows,
    is_useful_snap,
    load_jsonl,
    precision_at_fraction,
    query_split,
    roc_auc,
    select_heuristic_candidate,
    select_softmax_candidate,
    snap_candidate_rows,
    snap_improvement_summary,
    support_bias_bucket_summary,
    train_no_snap_softmax_selector,
    train_ridge_residual_model,
    train_snap_improvement_gate,
    write_json,
)


Policy = Callable[[Sequence[Mapping[str, object]]], Optional[Mapping[str, object]]]


def _camera_from_summary(summary: Mapping[str, object]) -> ColmapCamera:
    camera = summary["camera"]
    return ColmapCamera(
        camera_id=1,
        model_id=int(camera["model_id"]),
        width=int(camera["width"]),
        height=int(camera["height"]),
        params=tuple(float(value) for value in camera["params"]),
    )


def _query_pose_file_from_summary(summary: Mapping[str, object]) -> str:
    inputs = summary.get("inputs", {})
    if not isinstance(inputs, Mapping) or not inputs.get("query_pose_file"):
        raise ValueError("summary_json must contain inputs.query_pose_file")
    return str(inputs["query_pose_file"])


def _match_from_no_snap(row: Mapping[str, object], xy: np.ndarray | None = None) -> QueryTo3DMatch:
    base_xy = np.asarray(row["base_xy"] if xy is None else xy, dtype=np.float64).reshape(2)
    return QueryTo3DMatch(
        token_index=int(row.get("token_index", 0)),
        xy=base_xy,
        track_id=int(row.get("track_id", -1)),
        xyz=np.asarray(row["xyz"], dtype=np.float64).reshape(3),
        similarity=finite_float(row.get("match_similarity")),
        ratio=finite_float(row.get("match_ratio")),
        landmark_variance=finite_float(row.get("landmark_variance")),
        observation_count=None if row.get("observation_count") is None else int(row["observation_count"]),
        visibility_count=None if row.get("visibility_count") is None else int(row["visibility_count"]),
        landmark_reprojection_error=None
        if row.get("landmark_reprojection_error") is None
        else float(row["landmark_reprojection_error"]),
        landmark_quality=None if row.get("landmark_quality") is None else float(row["landmark_quality"]),
        landmark_ambiguity=None if row.get("landmark_ambiguity") is None else float(row["landmark_ambiguity"]),
        similarity_margin=None if row.get("similarity_margin") is None else float(row["similarity_margin"]),
    )


def _no_snap_row(group: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    for row in group:
        if row.get("action") == "no_snap":
            return row
    raise ValueError("candidate group has no no_snap row")


def _candidate_xy(row: Mapping[str, object], mode: str, residual_model=None) -> np.ndarray:
    if mode == "candidate":
        return np.asarray(row["candidate_xy"], dtype=np.float64).reshape(2)
    if mode == "support_residual":
        return np.asarray(row.get("residual_xy", row["candidate_xy"]), dtype=np.float64).reshape(2)
    if mode == "learned_residual":
        if residual_model is None:
            raise ValueError("residual_model is required for learned_residual")
        return np.asarray(row["candidate_xy"], dtype=np.float64).reshape(2) + residual_model.predict([row])[0]
    raise ValueError(f"unsupported xy mode: {mode}")


def evaluate_pose_policy(
    rows: Sequence[Mapping[str, object]],
    *,
    query_ids: set[str],
    camera: ColmapCamera,
    gt_by_query: Mapping[str, object],
    policy: Policy,
    xy_mode: str = "candidate",
    residual_model=None,
    pnp_method: str = "EPNP",
    refine_method: str = "LM",
) -> dict[str, object]:
    grouped = group_rows([row for row in rows if str(row.get("query_id")) in query_ids])
    by_query: dict[str, list[tuple[Sequence[Mapping[str, object]], Mapping[str, object]]]] = {}
    for key, group in grouped.items():
        by_query.setdefault(str(key[0]), []).append((group, _no_snap_row(group)))
    results = []
    applied = 0
    total = 0
    measurement_errors = []
    for query_id, items in sorted(by_query.items()):
        matches = []
        for group, no_snap in items:
            selected = policy(group)
            xy = None
            if selected is not None:
                xy = _candidate_xy(selected, xy_mode, residual_model=residual_model)
                applied += 1
                err_key = "candidate_error_px" if xy_mode == "candidate" else "residual_error_px"
                if xy_mode == "learned_residual":
                    gt_xy = selected.get("gt_xy")
                    if gt_xy is not None:
                        measurement_errors.append(float(np.linalg.norm(xy - np.asarray(gt_xy, dtype=np.float64).reshape(2))))
                elif selected.get(err_key) is not None:
                    measurement_errors.append(float(selected[err_key]))
            elif no_snap.get("center_error_px") is not None:
                measurement_errors.append(float(no_snap["center_error_px"]))
            matches.append(_match_from_no_snap(no_snap, xy=xy))
            total += 1
        pnp = estimate_pose_pnp_fixed(
            matches,
            camera,
            min_inliers=4,
            pnp_method=pnp_method,
            refine_method=refine_method,
        )
        gt = gt_by_query.get(query_id)
        if gt is None:
            raise ValueError(f"missing GT pose for query_id={query_id}")
        error = pnp_pose_error(pnp.pose_w2c, gt.pose_w2c)
        results.append(
            {
                "query_id": query_id,
                "pnp_solve": bool(pnp.success),
                "inlier_count": int(pnp.inlier_count),
                "translation_error_m": float(error.translation_m),
                "rotation_error_deg": float(error.rotation_deg),
                "success_10cm_5deg": bool(error.translation_m <= 0.10 and error.rotation_deg <= 5.0),
                "success_25cm_10deg": bool(error.translation_m <= 0.25 and error.rotation_deg <= 10.0),
                "success_50cm_10deg": bool(error.translation_m <= 0.50 and error.rotation_deg <= 10.0),
            }
        )
    translations = np.asarray([row["translation_error_m"] for row in results], dtype=np.float64)
    rotations = np.asarray([row["rotation_error_deg"] for row in results], dtype=np.float64)
    return {
        "query_count": int(len(results)),
        "pnp_solve_rate": float(np.mean([row["pnp_solve"] for row in results])) if results else 0.0,
        "success_10cm_5deg": float(np.mean([row["success_10cm_5deg"] for row in results])) if results else 0.0,
        "success_25cm_10deg": float(np.mean([row["success_25cm_10deg"] for row in results])) if results else 0.0,
        "success_50cm_10deg": float(np.mean([row["success_50cm_10deg"] for row in results])) if results else 0.0,
        "median_translation_error_m": float(np.median(translations)) if translations.size else None,
        "median_rotation_error_deg": float(np.median(rotations)) if rotations.size else None,
        "applied_ratio": float(applied / max(total, 1)),
        "mean_measurement_error_px": None if not measurement_errors else float(np.mean(measurement_errors)),
        "rows": results,
    }


def choose_gate_threshold(
    rows: Sequence[Mapping[str, object]],
    model,
    *,
    positive_px: float,
    margin_px: float,
) -> tuple[float, dict[str, float]]:
    selected = [row for group in group_rows(rows).values() if (row := select_heuristic_candidate(group)) is not None]
    if not selected:
        return 1.0, {"train_mean_measurement_error_px": float("nan")}
    probs = model.predict_proba(selected)
    thresholds = sorted(set([0.0, 1.0] + [float(np.quantile(probs, q)) for q in np.linspace(0.05, 0.95, 19)]))
    best_threshold = thresholds[0]
    best_error = float("inf")
    best_apply = 0.0
    for threshold in thresholds:
        errors = []
        applied = 0
        for row, prob in zip(selected, probs):
            if float(prob) >= float(threshold):
                errors.append(finite_float(row.get("candidate_error_px")))
                applied += 1
            else:
                errors.append(finite_float(row.get("center_error_px")))
        value = float(np.mean(errors))
        if value < best_error:
            best_error = value
            best_threshold = float(threshold)
            best_apply = float(applied / max(len(selected), 1))
    labels = np.asarray([1.0 if is_useful_snap(row, positive_px=positive_px, margin_px=margin_px) else 0.0 for row in selected])
    return best_threshold, {
        "train_mean_measurement_error_px": best_error,
        "train_applied_ratio": best_apply,
        "train_heuristic_auc": roc_auc(probs, labels),
        "train_heuristic_precision_top20": precision_at_fraction(probs, labels, 0.20),
    }


def oracle_gap_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    grouped = group_rows(rows)
    values = {
        "groups": 0,
        "heuristic_top1_snap8": 0,
        "top5_contains_snap8": 0,
        "any_candidate_snap8": 0,
        "best_candidate_improves": 0,
        "best_residual_improves": 0,
    }
    for group in grouped.values():
        candidates = [row for row in group if row.get("action") == "snap"]
        if not candidates:
            continue
        values["groups"] += 1
        heuristic = select_heuristic_candidate(group)
        if heuristic is not None and bool(heuristic.get("snap_correct_at_8px", False)):
            values["heuristic_top1_snap8"] += 1
        ordered = sorted(candidates, key=lambda row: int(row.get("candidate_rank", 1_000_000)))
        if any(bool(row.get("snap_correct_at_8px", False)) for row in ordered[:5]):
            values["top5_contains_snap8"] += 1
        if any(bool(row.get("snap_correct_at_8px", False)) for row in candidates):
            values["any_candidate_snap8"] += 1
        if any(finite_float(row.get("snap_improvement_px")) > 0.0 for row in candidates):
            values["best_candidate_improves"] += 1
        if any(finite_float(row.get("residual_improvement_px")) > 0.0 for row in candidates):
            values["best_residual_improves"] += 1
    denom = max(values["groups"], 1)
    return {key: (float(value / denom) if key != "groups" else float(value)) for key, value in values.items()}


def risk_curve(
    rows: Sequence[Mapping[str, object]],
    *,
    eval_query_ids: set[str],
    model,
    camera: ColmapCamera,
    gt_by_query: Mapping[str, object],
    fractions: Sequence[float],
    pnp_method: str,
) -> list[dict[str, object]]:
    selected = [
        row
        for group in group_rows([row for row in rows if str(row.get("query_id")) in eval_query_ids]).values()
        if (row := select_heuristic_candidate(group)) is not None
    ]
    if not selected:
        return []
    probs = model.predict_proba(selected)
    by_key = {group_key: prob for group_key, prob in zip([tuple([str(row.get("query_id")), int(row.get("match_index", -1))]) for row in selected], probs)}
    output = []
    for fraction in fractions:
        keep = max(1, int(round(float(fraction) * len(probs))))
        threshold = float(np.sort(probs)[-keep])

        def policy(group, threshold=threshold):
            candidate = select_heuristic_candidate(group)
            if candidate is None:
                return None
            key = (str(candidate.get("query_id")), int(candidate.get("match_index", -1)))
            return candidate if float(by_key.get(key, -1.0)) >= threshold else None

        metrics = evaluate_pose_policy(
            rows,
            query_ids=eval_query_ids,
            camera=camera,
            gt_by_query=gt_by_query,
            policy=policy,
            xy_mode="candidate",
            pnp_method=pnp_method,
        )
        output.append({"fraction": float(fraction), "threshold": threshold, **{k: v for k, v in metrics.items() if k != "rows"}})
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive_px", type=float, default=8.0)
    parser.add_argument("--margin_px", type=float, default=2.0)
    parser.add_argument("--gate_iterations", type=int, default=400)
    parser.add_argument("--softmax_iterations", type=int, default=120)
    parser.add_argument("--softmax_max_groups", type=int, default=20000)
    parser.add_argument("--residual_positive_px", type=float, default=12.0)
    parser.add_argument("--residual_max_px", type=float, default=16.0)
    parser.add_argument("--risk_fractions", default="0.2,0.4,0.6,0.8,1.0")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.candidate_jsonl)
    summary = json.loads(Path(args.summary_json).read_text())
    camera = _camera_from_summary(summary)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(_query_pose_file_from_summary(summary)))}
    query_ids = sorted({str(row.get("query_id")) for row in rows if row.get("query_id") is not None})
    train_ids, eval_ids = query_split(query_ids, train_fraction=float(args.train_fraction), seed=int(args.seed))
    train_rows = [row for row in rows if str(row.get("query_id")) in train_ids]
    eval_rows = [row for row in rows if str(row.get("query_id")) in eval_ids]
    pnp_method = str(summary.get("matching_config", {}).get("pnp_method", "EPNP"))

    gate, gate_train_metrics = train_snap_improvement_gate(
        train_rows,
        positive_px=float(args.positive_px),
        margin_px=float(args.margin_px),
        iterations=int(args.gate_iterations),
    )
    gate_threshold, gate_threshold_metrics = choose_gate_threshold(
        train_rows,
        gate,
        positive_px=float(args.positive_px),
        margin_px=float(args.margin_px),
    )
    eval_candidates = snap_candidate_rows(eval_rows)
    eval_labels = np.asarray([
        1.0 if is_useful_snap(row, positive_px=float(args.positive_px), margin_px=float(args.margin_px)) else 0.0
        for row in eval_candidates
    ])
    eval_probs = gate.predict_proba(eval_candidates)
    softmax = train_no_snap_softmax_selector(
        train_rows,
        positive_px=float(args.positive_px),
        margin_px=float(args.margin_px),
        iterations=int(args.softmax_iterations),
        max_groups=int(args.softmax_max_groups) if args.softmax_max_groups > 0 else None,
        seed=int(args.seed),
    )
    residual_model = train_ridge_residual_model(
        train_rows,
        positive_px=float(args.residual_positive_px),
        max_norm_px=float(args.residual_max_px),
    )

    def no_snap_policy(_group):
        return None

    def heuristic_policy(group):
        return select_heuristic_candidate(group)

    def gate_policy(group):
        candidate = select_heuristic_candidate(group)
        if candidate is None:
            return None
        prob = float(gate.predict_proba([candidate])[0])
        return candidate if prob >= gate_threshold else None

    def softmax_policy(group):
        return select_softmax_candidate(group, softmax)

    metrics = {
        "C2.5_no_snap": evaluate_pose_policy(
            rows, query_ids=eval_ids, camera=camera, gt_by_query=gt_by_query, policy=no_snap_policy, pnp_method=pnp_method
        ),
        "heuristic_LCKS": evaluate_pose_policy(
            rows, query_ids=eval_ids, camera=camera, gt_by_query=gt_by_query, policy=heuristic_policy, pnp_method=pnp_method
        ),
        "A_gate_heuristic_snap": evaluate_pose_policy(
            rows, query_ids=eval_ids, camera=camera, gt_by_query=gt_by_query, policy=gate_policy, pnp_method=pnp_method
        ),
        "B_no_snap_softmax": evaluate_pose_policy(
            rows, query_ids=eval_ids, camera=camera, gt_by_query=gt_by_query, policy=softmax_policy, pnp_method=pnp_method
        ),
        "C_heuristic_support_residual": evaluate_pose_policy(
            rows,
            query_ids=eval_ids,
            camera=camera,
            gt_by_query=gt_by_query,
            policy=heuristic_policy,
            xy_mode="support_residual",
            pnp_method=pnp_method,
        ),
        "C_heuristic_learned_residual": evaluate_pose_policy(
            rows,
            query_ids=eval_ids,
            camera=camera,
            gt_by_query=gt_by_query,
            policy=heuristic_policy,
            xy_mode="learned_residual",
            residual_model=residual_model,
            pnp_method=pnp_method,
        ),
        "C_softmax_support_residual": evaluate_pose_policy(
            rows,
            query_ids=eval_ids,
            camera=camera,
            gt_by_query=gt_by_query,
            policy=softmax_policy,
            xy_mode="support_residual",
            pnp_method=pnp_method,
        ),
        "C_softmax_learned_residual": evaluate_pose_policy(
            rows,
            query_ids=eval_ids,
            camera=camera,
            gt_by_query=gt_by_query,
            policy=softmax_policy,
            xy_mode="learned_residual",
            residual_model=residual_model,
            pnp_method=pnp_method,
        ),
    }
    for value in metrics.values():
        value.pop("rows", None)

    fractions = tuple(float(item.strip()) for item in str(args.risk_fractions).split(",") if item.strip())
    result = {
        "scene": args.scene,
        "candidate_jsonl": args.candidate_jsonl,
        "summary_json": args.summary_json,
        "query_split": {
            "train_query_count": int(len(train_ids)),
            "eval_query_count": int(len(eval_ids)),
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
        },
        "diagnostics": {
            "train_snap_improvement": snap_improvement_summary(train_rows),
            "eval_snap_improvement": snap_improvement_summary(eval_rows),
            "train_support_bias_buckets": support_bias_bucket_summary(train_rows),
            "eval_support_bias_buckets": support_bias_bucket_summary(eval_rows),
            "train_oracle_gap": oracle_gap_summary(train_rows),
            "eval_oracle_gap": oracle_gap_summary(eval_rows),
        },
        "experiment_A": {
            "gate_train": gate_train_metrics,
            "gate_threshold": float(gate_threshold),
            "gate_threshold_selection": gate_threshold_metrics,
            "gate_eval": {
                "positive_prior": float(np.mean(eval_labels)) if eval_labels.size else 0.0,
                "auc": roc_auc(eval_probs, eval_labels) if eval_labels.size else float("nan"),
                "precision_top20": precision_at_fraction(eval_probs, eval_labels, 0.20) if eval_labels.size else float("nan"),
                "precision_top40": precision_at_fraction(eval_probs, eval_labels, 0.40) if eval_labels.size else float("nan"),
            },
        },
        "experiment_B": {
            "softmax_no_snap_bias": float(softmax.no_snap_bias),
            "train_group_count": int(len(group_rows(train_rows))),
        },
        "experiment_C": {
            "residual_train_positive_px": float(args.residual_positive_px),
            "residual_max_px": float(args.residual_max_px),
        },
        "experiment_D": risk_curve(
            rows,
            eval_query_ids=eval_ids,
            model=gate,
            camera=camera,
            gt_by_query=gt_by_query,
            fractions=fractions,
            pnp_method=pnp_method,
        ),
        "pose_metrics_eval_split": metrics,
    }
    write_json(output_dir / "abcd_summary.json", result)
    write_markdown(output_dir / "abcd_summary.md", result)


def write_markdown(path: str | Path, result: Mapping[str, object]) -> None:
    lines = [
        "# SuperPoint Keypoint Experiments ABCD",
        "",
        f"Scene: `{result.get('scene')}`",
        "",
        "## Pose Metrics On Held-Out Queries",
        "",
        "| method | S@10 | S@25 | S@50 | median t | median r | applied | measurement err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["pose_metrics_eval_split"].items():
        lines.append(
            "| {name} | {s10:.3f} | {s25:.3f} | {s50:.3f} | {mt:.3f}m | {mr:.3f}deg | {applied:.3f} | {meas} |".format(
                name=name,
                s10=float(metrics["success_10cm_5deg"]),
                s25=float(metrics["success_25cm_10deg"]),
                s50=float(metrics["success_50cm_10deg"]),
                mt=float(metrics["median_translation_error_m"]),
                mr=float(metrics["median_rotation_error_deg"]),
                applied=float(metrics["applied_ratio"]),
                meas="-"
                if metrics.get("mean_measurement_error_px") is None
                else f"{float(metrics['mean_measurement_error_px']):.2f}px",
            )
        )
    lines.extend(
        [
            "",
            "## Experiment A Gate",
            "",
            "```json",
            json.dumps(result["experiment_A"], indent=2, sort_keys=True),
            "```",
            "",
            "## Experiment D Risk Curve",
            "",
            "| applied fraction | S@25 | median t | applied ratio | measurement err |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["experiment_D"]:
        lines.append(
            "| {fraction:.2f} | {s25:.3f} | {mt:.3f}m | {applied:.3f} | {meas} |".format(
                fraction=float(row["fraction"]),
                s25=float(row["success_25cm_10deg"]),
                mt=float(row["median_translation_error_m"]),
                applied=float(row["applied_ratio"]),
                meas="-"
                if row.get("mean_measurement_error_px") is None
                else f"{float(row['mean_measurement_error_px']):.2f}px",
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "```json",
            json.dumps(result["diagnostics"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
