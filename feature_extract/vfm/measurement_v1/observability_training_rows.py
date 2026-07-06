from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.vfm.measurement_v1.observability_gate import POSITIVE_OBSERVABILITY_CLASSES


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fieldnames(rows: Sequence[Mapping[str, object]], appended: Sequence[str]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(str(key))
    for key in appended:
        if key not in names:
            names.append(str(key))
    return names


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def export_observability_dustbin_rows(
    *,
    rows_csv: Path,
    output_csv: Path,
    positive_classes: Sequence[str] = tuple(sorted(POSITIVE_OBSERVABILITY_CLASSES)),
) -> dict[str, Any]:
    rows = _read_csv(Path(rows_csv))
    positives = {str(value) for value in positive_classes}
    out_rows: list[dict[str, object]] = []
    valid_rows = 0
    dustbin_rows = 0
    class_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("observability_class", "")).strip()
        class_counts[label] = class_counts.get(label, 0) + 1
        is_valid = label in positives
        valid_rows += int(is_valid)
        dustbin_rows += int(not is_valid)
        out = dict(row)
        out["target_is_dustbin"] = "False" if is_valid else "True"
        out["observability_positive"] = "1" if is_valid else "0"
        out["observability_dustbin_policy"] = "positive_classes_valid_else_dustbin"
        out_rows.append(out)
    output = Path(output_csv)
    appended = ["target_is_dustbin", "observability_positive", "observability_dustbin_policy"]
    _write_csv(output, out_rows, fieldnames=_fieldnames(out_rows, appended))
    summary = {
        "rows_csv": str(rows_csv),
        "output_csv": str(output),
        "row_count": int(len(out_rows)),
        "valid_rows": int(valid_rows),
        "dustbin_rows": int(dustbin_rows),
        "valid_fraction": float(valid_rows) / float(len(out_rows)) if out_rows else 0.0,
        "class_counts": class_counts,
        "positive_classes": sorted(positives),
    }
    summary_json = output.with_suffix(output.suffix + ".summary.json")
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    summary["summary_json"] = str(summary_json)
    return summary
