"""Freeze an inlier-ratio rejection threshold on a development PnP report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


GRID = (0.0, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development_report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite PnP confidence config")
    report = json.loads(args.development_report.read_text())
    rows = [row for row in report["rows"] if row["usable"]]
    table = []
    for threshold in GRID:
        accepted = [
            row for row in rows
            if row["pnp_inlier_count"] / max(row["candidate_correspondence_count"], 1) >= threshold
        ]
        good = [row for row in accepted if row["translation_error_m"] <= 2 and row["rotation_error_deg"] <= 45]
        table.append({
            "threshold": threshold,
            "accepted_count": len(accepted),
            "accepted_good_count": len(good),
            "accepted_precision_2m45": len(good) / len(accepted) if accepted else 0.0,
        })
    eligible = [row for row in table if row["accepted_count"] > 0 and row["accepted_precision_2m45"] == 1.0]
    if not eligible:
        raise ValueError("no zero-false-acceptance threshold exists on development")
    winner = max(eligible, key=lambda row: (row["accepted_good_count"], -row["threshold"]))
    payload = {
        "artifact_type": "goal_maplet_direct_plane_pnp_confidence_config_v1",
        "development_report_file_sha256": file_sha256(args.development_report),
        "score": "pnp_inlier_count/candidate_correspondence_count",
        "selection": "maximize accepted 2m45-good count subject to zero accepted 2m45 failures; tie lower threshold",
        "threshold": winner["threshold"],
        "grid": list(GRID),
        "grid_results": table,
        "development_query_count": len(report["rows"]),
        "official_test_metrics_used_by_builder": False,
        "protocol_preregistered_before_official_test_pilot": False,
        "eligible_use": "future unseen official-test complement or new scene only",
        "production_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
