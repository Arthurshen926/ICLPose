"""Insert post-label GT planes into frozen rankings for bottleneck diagnosis only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postlabel_correspondence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_purity", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite oracle insertion control")
    source = json.loads(args.postlabel_correspondence.read_text())
    inserted = 0
    rows = []
    for query in source["rows"]:
        records = []
        for record in query["regions"]:
            value = dict(record)
            ranking = [int(row) for row in record["top10"]]
            gt = int(record.get("gt_plane", -1))
            if gt >= 0 and float(record.get("gt_purity", 0.0)) >= args.minimum_purity:
                ranking = [gt] + [row for row in ranking if row != gt]
                inserted += 1
            value["top10"] = ranking[:10]
            records.append(value)
        rows.append({"image": query["image"], "query_plane_count": len(records), "regions": records})
    report = {
        "artifact_type": "goal_maplet_postlabel_gt_plane_oracle_insertion_control_v1",
        "query_count": len(rows),
        "inserted_region_count": inserted,
        "minimum_gt_purity": float(args.minimum_purity),
        "uses_pose_or_ground_truth": True,
        "contains_postlabel_fields": True,
        "source_file_sha256": file_sha256(args.postlabel_correspondence),
        "production_eligible": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
