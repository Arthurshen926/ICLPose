"""Merge disjoint Goal-Maplet oracle-ladder shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite merged oracle report")
    reports = [json.loads(Path(path).read_text()) for path in args.inputs]
    for key in ("stage", "physical_map_sha256", "canonical_field_sha256", "validity_calibration_sha256"):
        values = [report.get(key) for report in reports]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"oracle shard mismatch for {key}: {values}")
    rows = [row for report in reports for row in report.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("oracle shards contain duplicate images")
    names = list(rows[0]["oracles"]) if rows else []
    summary = {}
    for name in names:
        values = [row["oracles"][name] for row in rows]
        translation = [float(value["translation_m"]) for value in values if value.get("translation_m") is not None]
        rotation = [float(value["rotation_deg"]) for value in values if value.get("rotation_deg") is not None]
        reprojection = [float(value["gt_reprojection_median_px"]) for value in values if value.get("gt_reprojection_median_px") is not None]
        summary[name] = {
            "success_fraction": float(np.mean([bool(value["success"]) for value in values])) if values else 0.0,
            "translation_median_m": float(np.median(translation)) if translation else None,
            "translation_p90_m": float(np.percentile(translation, 90.0)) if translation else None,
            "rotation_median_deg": float(np.median(rotation)) if rotation else None,
            "rotation_p90_deg": float(np.percentile(rotation, 90.0)) if rotation else None,
            "gt_reprojection_median_px": float(np.median(reprojection)) if reprojection else None,
            "within_0.25m_5deg": float(np.mean([
                value.get("translation_m") is not None and float(value["translation_m"]) <= 0.25
                and float(value["rotation_deg"]) <= 5.0 for value in values
            ])) if values else 0.0,
            "within_0.5m_10deg": float(np.mean([
                value.get("translation_m") is not None and float(value["translation_m"]) <= 0.5
                and float(value["rotation_deg"]) <= 10.0 for value in values
            ])) if values else 0.0,
        }
    result = {
        "stage": reports[0]["stage"],
        "physical_map_sha256": reports[0]["physical_map_sha256"],
        "canonical_field_sha256": reports[0]["canonical_field_sha256"],
        "validity_calibration_sha256": reports[0]["validity_calibration_sha256"],
        "query_count": len(rows),
        "shard_count": len(reports),
        "source_shards": [str(Path(path)) for path in args.inputs],
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
