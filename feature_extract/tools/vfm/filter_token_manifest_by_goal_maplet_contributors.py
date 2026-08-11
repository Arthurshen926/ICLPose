"""Filter a RADIO token manifest to an audited Goal-Maplet contributor set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.tokens import TokenBankManifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--include_trajectories", nargs="*", default=None)
    parser.add_argument("--exclude_trajectories", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite filtered token manifest")
    selected_ids: set[str] = set()
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        selected_ids.add(str(metadata["image_id"]))
    include = None if args.include_trajectories is None else set(args.include_trajectories)
    exclude = set(args.exclude_trajectories or ())
    source = TokenBankManifest.from_json(Path(args.token_manifest))
    records = []
    for record in source.records:
        trajectory = str(record.image_id).replace("\\", "/").split("/", 1)[0]
        if record.image_id not in selected_ids:
            continue
        if include is not None and trajectory not in include:
            continue
        if trajectory in exclude:
            continue
        records.append(record)
    missing = selected_ids - {record.image_id for record in source.records}
    if missing:
        raise ValueError(f"token manifest misses {len(missing)} contributor images")
    result = TokenBankManifest(tuple(records))
    result.validate(verify_checksums=False)
    result.to_json(output)
    print(json.dumps({
        "stage": "filter_goal_maplet_contributor_token_manifest",
        "record_count": len(records),
        "include_trajectories": None if include is None else sorted(include),
        "exclude_trajectories": sorted(exclude),
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
