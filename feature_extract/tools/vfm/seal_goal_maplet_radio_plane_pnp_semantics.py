"""Correct and seal the ranking-semantics label of a completed PnP report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnp_report", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sealed PnP report")
    report = json.loads(args.pnp_report.read_text())
    ranking = json.loads(args.ranking.read_text())
    if (
        report.get("artifact_type") != "goal_maplet_radio_plane_2d3d_pnp_control_v2"
        or report.get("correspondence_report_file_sha256") != file_sha256(args.ranking)
        or ranking.get("artifact_type") != "goal_maplet_direct_radio_to_finite_plane_ranking_v1"
    ):
        raise ValueError("PnP report/ranking lineage differs")
    report["artifact_type"] = "goal_maplet_radio_plane_2d3d_pnp_control_v3_semantics_sealed"
    report["plane_retrieval"] = "direct query-region RADIO to finite-plane observation descriptors"
    report["upstream_pnp_report_file_sha256"] = file_sha256(args.pnp_report)
    report["semantics_only_reseal_no_numeric_recomputation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
