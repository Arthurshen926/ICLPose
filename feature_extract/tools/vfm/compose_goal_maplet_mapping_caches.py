"""Compose disjoint contributor/plane-observation caches using exact hard links."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def _files(directories: list[Path], pattern: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob(pattern)):
            if path.name in output:
                raise ValueError(f"cache member is duplicated: {path.name}")
            output[path.name] = path
    if not output:
        raise ValueError("cache inventory is empty")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributor_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--observation_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_contributors", type=Path, required=True)
    parser.add_argument("--output_observations", type=Path, required=True)
    parser.add_argument("--output_audit", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_contributors, args.output_observations, args.output_audit):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    contributors = _files(args.contributor_dirs, "*.npz")
    observations = _files(args.observation_dirs, "*.planes.npz")
    contributor_ids = set(contributors)
    observation_ids = {name[:-len(".planes.npz")] + ".npz" for name in observations}
    if contributor_ids != observation_ids:
        raise ValueError("contributor and plane-observation inventories differ")
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in args.observation_dirs]
    surface_hashes = {str(row["primitive_surface_table_file_sha256"]) for row in manifests}
    if len(surface_hashes) != 1 or any(bool(row.get("uses_query_or_gt", True)) for row in manifests):
        raise ValueError("observation manifests differ or consume query evidence")

    args.output_contributors.mkdir(parents=True)
    args.output_observations.mkdir(parents=True)
    for name, source in contributors.items():
        os.link(source, args.output_contributors / name)
    for name, source in observations.items():
        os.link(source, args.output_observations / name)
    rows = sorted(row for manifest in manifests for row in manifest["rows"])
    if len(rows) != len(observations) or len({str(row[0]) for row in rows}) != len(rows):
        raise ValueError("observation manifest rows are incomplete or duplicated")
    manifest = {
        "artifact_type": "goal_maplet_rendered_plane_observation_run_v1",
        "routes": sorted({route for source in manifests for route in source["routes"]}),
        "view_count": len(rows),
        "plane_count": sum(int(row[1]) for row in rows),
        "covered_pixel_count": sum(int(row[2]) for row in rows),
        "rows": rows,
        "primitive_surface_table_file_sha256": next(iter(surface_hashes)),
        "uses_parent_child_partition": False,
        "uses_query_or_gt": False,
        "composition_input_manifest_file_sha256_in_order": [
            file_sha256(directory / "manifest.json") for directory in args.observation_dirs
        ],
    }
    (args.output_observations / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "artifact_type": "goal_maplet_composed_mapping_cache_audit_v1",
        "view_count": len(contributors),
        "contributor_member_file_sha256": {
            name: file_sha256(path) for name, path in sorted(contributors.items())
        },
        "observation_member_file_sha256": {
            name: file_sha256(path) for name, path in sorted(observations.items())
        },
        "output_observation_manifest_file_sha256": file_sha256(
            args.output_observations / "manifest.json"
        ),
        "hard_link_composition": True,
        "uses_query_or_ground_truth": False,
    }
    audit["content_sha256"] = canonical_json_sha256(audit)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "view_count": len(contributors), "content_sha256": audit["content_sha256"],
        "output_audit_file_sha256": file_sha256(args.output_audit),
        "observation_manifest_file_sha256": audit["output_observation_manifest_file_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
