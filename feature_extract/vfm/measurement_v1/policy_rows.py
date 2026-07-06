from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_training import _read_csv


POLICY_FIELDNAMES = [
    "measurement_policy",
    "policy_target_x",
    "policy_target_y",
    "policy_target_source",
    "policy_loss_weight",
    "teacher_valid",
    "teacher_reason",
    "teacher_epe_px",
    "teacher_pred_x",
    "teacher_pred_y",
]


def _float(row: Mapping[str, object], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    text = str(row.get(key, "")).strip()
    if not text:
        return None
    return float(text)


def _bool_text(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _teacher_by_row_index(rows: Sequence[Mapping[str, str]]) -> dict[int, Mapping[str, str]]:
    out: dict[int, Mapping[str, str]] = {}
    for row in rows:
        text = str(row.get("row_index", "")).strip()
        if not text:
            raise ValueError("dense teacher rows must contain row_index")
        index = int(text)
        if index in out:
            raise ValueError(f"duplicate dense teacher row_index={index}")
        out[index] = row
    return out


def build_measurement_policy_rows(
    *,
    rows_csv: Path,
    dense_teacher_rows_csv: Path,
    output_rows_csv: Path,
    center_preserve_below_px: float = 0.5,
    teacher_valid_max_epe_px: float = 1.0,
    center_loss_weight: float = 1.0,
    teacher_loss_weight: float = 1.0,
    fallback_loss_weight: float = 0.25,
) -> dict[str, Any]:
    """Create opt-in measurement training policy rows from dense teacher diagnostics.

    The original ``query_gt_x/y`` columns are preserved as evaluation truth.
    New ``policy_target_x/y`` columns encode the optional training target:
    center-preserving rows learn zero offset, trusted dense-teacher rows learn
    the teacher correction, and invalid teacher rows fall back to SfM GT with a
    lower weight.
    """

    rows = _read_csv(Path(rows_csv))
    teacher_rows = _teacher_by_row_index(_read_csv(Path(dense_teacher_rows_csv)))
    if not rows:
        raise ValueError("rows_csv contains no rows")
    out_rows: list[dict[str, object]] = []
    policy_counts: dict[str, int] = {}
    teacher_valid_count = 0
    for row_index, row in enumerate(rows):
        teacher = teacher_rows.get(int(row_index))
        if teacher is None:
            raise ValueError(f"missing dense teacher row for row_index={row_index}")
        requested = _optional_float(row, "requested_residual_px")
        center_x = _float(row, "center_x")
        center_y = _float(row, "center_y")
        gt_x = _float(row, "query_gt_x")
        gt_y = _float(row, "query_gt_y")
        teacher_epe = _optional_float(teacher, "lk_epe_px")
        teacher_pred_x = _optional_float(teacher, "lk_pred_x")
        teacher_pred_y = _optional_float(teacher, "lk_pred_y")
        teacher_applied = _bool_text(teacher.get("lk_applied", "False"))
        teacher_valid = (
            bool(teacher_applied)
            and teacher_epe is not None
            and teacher_pred_x is not None
            and teacher_pred_y is not None
            and float(teacher_epe) <= float(teacher_valid_max_epe_px)
        )
        if requested is not None and float(requested) <= float(center_preserve_below_px):
            policy = "center_preserve"
            target_x = center_x
            target_y = center_y
            target_source = "center"
            loss_weight = float(center_loss_weight)
        elif teacher_valid:
            policy = "teacher_correction"
            target_x = float(teacher_pred_x)
            target_y = float(teacher_pred_y)
            target_source = "dense_teacher"
            loss_weight = float(teacher_loss_weight)
            teacher_valid_count += 1
        else:
            policy = "gt_fallback"
            target_x = gt_x
            target_y = gt_y
            target_source = "query_gt"
            loss_weight = float(fallback_loss_weight)
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
        out_rows.append(
            {
                **row,
                "measurement_policy": policy,
                "policy_target_x": float(target_x),
                "policy_target_y": float(target_y),
                "policy_target_source": target_source,
                "policy_loss_weight": float(loss_weight),
                "teacher_valid": bool(teacher_valid),
                "teacher_reason": str(teacher.get("lk_reason", "")),
                "teacher_epe_px": "" if teacher_epe is None else float(teacher_epe),
                "teacher_pred_x": "" if teacher_pred_x is None else float(teacher_pred_x),
                "teacher_pred_y": "" if teacher_pred_y is None else float(teacher_pred_y),
            }
        )
    fieldnames = list(rows[0].keys())
    for name in POLICY_FIELDNAMES:
        if name not in fieldnames:
            fieldnames.append(name)
    _write_csv(Path(output_rows_csv), out_rows, fieldnames=fieldnames)
    summary = {
        "stage": "measurement_v1_policy_rows",
        "rows_csv": str(rows_csv),
        "dense_teacher_rows_csv": str(dense_teacher_rows_csv),
        "output_rows_csv": str(output_rows_csv),
        "row_count": int(len(out_rows)),
        "policy_counts": dict(sorted(policy_counts.items())),
        "teacher_valid_count": int(teacher_valid_count),
        "config": {
            "center_preserve_below_px": float(center_preserve_below_px),
            "teacher_valid_max_epe_px": float(teacher_valid_max_epe_px),
            "center_loss_weight": float(center_loss_weight),
            "teacher_loss_weight": float(teacher_loss_weight),
            "fallback_loss_weight": float(fallback_loss_weight),
        },
    }
    Path(output_rows_csv).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
