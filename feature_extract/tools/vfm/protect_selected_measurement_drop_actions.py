"""Restore pose-critical matches from frozen measurement DROP actions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action_predictions_csv", required=True)
    parser.add_argument("--coarse_pose_context_csv", required=True)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protect_coarse_pose_inliers", action="store_true")
    parser.add_argument("--min_retained_matches", type=int, default=0)
    parser.add_argument("--min_grid_cells", type=int, default=0)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=576)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _bool(row: Mapping[str, object], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def _load_policy(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            key: np.asarray(payload[key])
            for key in (
                "query_ids",
                "query_xy",
                "selected_track_ids",
                "selected_pose_selection_scores",
            )
        }
    if metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected-policy artifact")
    count = len(arrays["query_ids"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("selected-policy arrays have inconsistent lengths")
    return arrays, metadata


def _context_by_policy_row(path: Path) -> dict[int, dict[str, str]]:
    rows, _fields = _read_csv(path)
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        policy_row = int(row["policy_row_index"])
        if policy_row in output:
            raise ValueError(f"duplicate coarse-pose policy row: {policy_row}")
        output[policy_row] = row
    return output


def _grid_cell(
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
) -> int:
    x = float(np.asarray(xy).reshape(2)[0])
    y = float(np.asarray(xy).reshape(2)[1])
    col = min(
        max(int(np.floor(x / max(float(image_width), 1.0) * grid_cols)), 0),
        grid_cols - 1,
    )
    row = min(
        max(int(np.floor(y / max(float(image_height), 1.0) * grid_rows)), 0),
        grid_rows - 1,
    )
    return int(row * grid_cols + col)


def protect_drop_actions(
    *,
    action_rows: Sequence[Mapping[str, object]],
    context_by_policy_row: Mapping[int, Mapping[str, object]],
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    selected_track_ids: np.ndarray,
    pose_selection_scores: np.ndarray,
    protect_coarse_pose_inliers: bool,
    min_retained_matches: int,
    min_grid_cells: int,
    grid_rows: int,
    grid_cols: int,
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if int(min_retained_matches) < 0:
        raise ValueError("min_retained_matches must be non-negative")
    if int(min_grid_cells) < 0:
        raise ValueError("min_grid_cells must be non-negative")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    if int(min_grid_cells) > int(grid_rows) * int(grid_cols):
        raise ValueError("min_grid_cells exceeds the grid capacity")
    rows = [dict(row) for row in action_rows]
    action_by_policy: dict[int, dict[str, object]] = {}
    query_set: set[str] = set()
    for row in rows:
        policy_row = int(str(row.get("policy_row_index", "-1")))
        if policy_row in action_by_policy:
            raise ValueError(f"duplicate action policy row: {policy_row}")
        action_by_policy[policy_row] = row
        query_set.add(str(row.get("query_id", "")))
    policy_query_ids = np.asarray(query_ids).astype(str)
    policy_xy = np.asarray(query_xy, dtype=np.float64)
    policy_tracks = np.asarray(selected_track_ids, dtype=np.int64)
    policy_scores = np.asarray(pose_selection_scores, dtype=np.float64)
    policy_rows_by_query: dict[str, list[int]] = {}
    for policy_row, query_id in enumerate(policy_query_ids.tolist()):
        if query_id in query_set:
            policy_rows_by_query.setdefault(query_id, []).append(policy_row)

    restored_reason: dict[int, list[str]] = {}
    query_reports: list[dict[str, object]] = []
    for query_id in sorted(query_set):
        policy_rows = policy_rows_by_query.get(query_id)
        if not policy_rows:
            raise ValueError(f"action query is absent from policy: {query_id}")
        dropped = {
            policy_row
            for policy_row in policy_rows
            if str(action_by_policy.get(policy_row, {}).get("action", "KEEP"))
            == "DROP"
        }
        original_drop_count = len(dropped)

        def restore(policy_row: int, reason: str) -> None:
            if policy_row not in dropped:
                return
            dropped.remove(policy_row)
            restored_reason.setdefault(policy_row, []).append(reason)

        if bool(protect_coarse_pose_inliers):
            for policy_row in sorted(dropped):
                context = context_by_policy_row.get(policy_row)
                if context is None:
                    raise ValueError(
                        f"coarse-pose context is missing policy row {policy_row}"
                    )
                if str(context.get("query_id", "")) != query_id:
                    raise ValueError("coarse-pose context query identity mismatch")
                if int(float(str(context.get("track_id", "-1")))) != int(
                    policy_tracks[policy_row]
                ):
                    raise ValueError("coarse-pose context track identity mismatch")
                if _bool(context, "coarse_pose_inlier"):
                    restore(policy_row, "coarse_pose_inlier")

        retained = [row for row in policy_rows if row not in dropped]
        occupied_cells = {
            _grid_cell(
                policy_xy[row],
                image_width=int(image_width),
                image_height=int(image_height),
                grid_rows=int(grid_rows),
                grid_cols=int(grid_cols),
            )
            for row in retained
        }

        def priority(policy_row: int) -> tuple[float, float, int]:
            action = action_by_policy.get(policy_row, {})
            geometry_probability = float(
                str(action.get("geometry_probability", "-inf"))
            )
            return (
                geometry_probability,
                float(policy_scores[policy_row]),
                -int(policy_row),
            )

        if len(occupied_cells) < int(min_grid_cells):
            best_by_cell: dict[int, int] = {}
            for policy_row in dropped:
                cell = _grid_cell(
                    policy_xy[policy_row],
                    image_width=int(image_width),
                    image_height=int(image_height),
                    grid_rows=int(grid_rows),
                    grid_cols=int(grid_cols),
                )
                if cell in occupied_cells:
                    continue
                existing = best_by_cell.get(cell)
                if existing is None or priority(policy_row) > priority(existing):
                    best_by_cell[cell] = policy_row
            for cell, policy_row in sorted(
                best_by_cell.items(),
                key=lambda item: priority(item[1]),
                reverse=True,
            ):
                if len(occupied_cells) >= int(min_grid_cells):
                    break
                restore(policy_row, "grid_coverage")
                occupied_cells.add(cell)

        retained_count = len(policy_rows) - len(dropped)
        needed = max(0, int(min_retained_matches) - retained_count)
        for policy_row in sorted(dropped, key=priority, reverse=True)[:needed]:
            restore(policy_row, "minimum_retained_matches")

        final_retained = len(policy_rows) - len(dropped)
        final_cells = {
            _grid_cell(
                policy_xy[row],
                image_width=int(image_width),
                image_height=int(image_height),
                grid_rows=int(grid_rows),
                grid_cols=int(grid_cols),
            )
            for row in policy_rows
            if row not in dropped
        }
        query_reports.append(
            {
                "query_id": query_id,
                "policy_match_count": len(policy_rows),
                "original_drop_count": original_drop_count,
                "restored_count": original_drop_count - len(dropped),
                "final_drop_count": len(dropped),
                "final_retained_count": final_retained,
                "final_grid_cell_count": len(final_cells),
            }
        )

    output_rows: list[dict[str, object]] = []
    for row in rows:
        policy_row = int(str(row["policy_row_index"]))
        reasons = restored_reason.get(policy_row, [])
        output_rows.append(
            {
                **row,
                "action": "KEEP" if reasons else str(row["action"]),
                "drop_protection_reason": "+".join(reasons),
            }
        )
    return output_rows, query_reports


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    action_path = Path(args.action_predictions_csv)
    context_path = Path(args.coarse_pose_context_csv)
    policy_path = Path(args.policy_artifact)
    action_rows, action_fields = _read_csv(action_path)
    context = _context_by_policy_row(context_path)
    arrays, policy_metadata = _load_policy(policy_path)
    output_rows, query_reports = protect_drop_actions(
        action_rows=action_rows,
        context_by_policy_row=context,
        query_ids=arrays["query_ids"],
        query_xy=arrays["query_xy"],
        selected_track_ids=arrays["selected_track_ids"],
        pose_selection_scores=arrays["selected_pose_selection_scores"],
        protect_coarse_pose_inliers=bool(args.protect_coarse_pose_inliers),
        min_retained_matches=int(args.min_retained_matches),
        min_grid_cells=int(args.min_grid_cells),
        grid_rows=int(args.grid_rows),
        grid_cols=int(args.grid_cols),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "action_predictions.csv"
    fieldnames = list(action_fields)
    if "drop_protection_reason" not in fieldnames:
        fieldnames.append("drop_protection_reason")
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    reports_path = output / "query_reports.json"
    reports_path.write_text(
        json.dumps(query_reports, indent=2, sort_keys=True) + "\n"
    )
    actions = [str(row["action"]) for row in output_rows]
    summary = {
        "stage": "selected_measurement_pose_safe_drop_protection",
        "protocol": {
            "gt_pose_used": False,
            "identity_reassignment": False,
            "coordinate_update_added": False,
            "only_drop_to_keep_restoration": True,
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "policy": {
            "protect_coarse_pose_inliers": bool(
                args.protect_coarse_pose_inliers
            ),
            "min_retained_matches": int(args.min_retained_matches),
            "min_grid_cells": int(args.min_grid_cells),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "image_width": int(args.image_width),
            "image_height": int(args.image_height),
        },
        "query_count": len(query_reports),
        "action_row_count": len(output_rows),
        "action_counts": {
            name: actions.count(name)
            for name in ("KEEP", "UPDATE_MEAN", "UPDATE_MODE", "DROP")
        },
        "restored_count": int(
            sum(int(report["restored_count"]) for report in query_reports)
        ),
        "inputs": {
            "action_predictions_csv": str(action_path),
            "action_predictions_sha256": file_sha256_short(action_path),
            "coarse_pose_context_csv": str(context_path),
            "coarse_pose_context_sha256": file_sha256_short(context_path),
            "policy_artifact": str(policy_path),
            "policy_artifact_sha256": file_sha256_short(policy_path),
            "policy_projected_landmark_bank_sha256": policy_metadata.get(
                "projected_landmark_bank_sha256"
            ),
            "policy_split_json_sha256": policy_metadata.get(
                "split_json_sha256"
            ),
        },
        "outputs": {
            "action_predictions": str(predictions_path),
            "action_predictions_sha256": file_sha256_short(predictions_path),
            "query_reports": str(reports_path),
            "query_reports_sha256": file_sha256_short(reports_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
