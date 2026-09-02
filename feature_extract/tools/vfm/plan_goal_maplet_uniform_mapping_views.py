"""Freeze a deterministic uniform mapping-view inventory from a RADIO manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output_ids", type=Path, required=True)
    parser.add_argument("--output_plan", type=Path, required=True)
    args = parser.parse_args()
    if args.output_ids.exists() or args.output_plan.exists():
        raise FileExistsError("refusing to overwrite mapping-view plan")
    manifest = json.loads(args.radio_manifest.read_text())
    names = sorted(
        str(row["image_id"]) for row in manifest["records"]
        if str(row["image_id"]).split("/", 1)[0] == str(args.route)
    )
    count = int(args.count)
    if count < 2 or count > len(names):
        raise ValueError("mapping-view count is outside inventory")
    indices = np.rint(np.linspace(0, len(names) - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("uniform mapping indices are duplicated")
    selected = [names[int(index)] for index in indices]
    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_ids.write_text("\n".join(selected) + "\n")
    payload = {
        "artifact_type": "goal_maplet_uniform_mapping_view_plan_v1",
        "route": str(args.route), "source_count": len(names), "selected_count": count,
        "selection": "rounded_linspace_over_sorted_route_inventory",
        "selected_indices": indices.tolist(), "selected_names_in_order": selected,
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "uses_query_or_ground_truth": False,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_count": count, "ids_file_sha256": file_sha256(args.output_ids),
                      "plan_file_sha256": file_sha256(args.output_plan),
                      "content_sha256": payload["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
