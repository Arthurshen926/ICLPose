"""Merge rendered geometry shards and create trajectory-held train/eval folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_manifests", nargs="+", required=True)
    parser.add_argument("--held_trajectories", nargs="+", required=True)
    parser.add_argument(
        "--combined_holdout_name",
        default="",
        help=(
            "Also emit one train/eval pair holding out the union of all "
            "--held_trajectories under this name."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = [Path(value) for value in args.geometry_manifests]
    payloads = [json.loads(path.read_text()) for path in paths]
    records = [record for payload in payloads for record in payload["records"]]
    image_ids = [str(record["image_id"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("geometry shards contain duplicate image ids")
    records.sort(key=lambda value: str(value["image_id"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage = [
        {"path": str(path), "sha256": _sha256(path)} for path in paths
    ]
    written = {}
    holdouts = [(str(held), {str(held)}) for held in args.held_trajectories]
    if args.combined_holdout_name:
        combined = set(str(value) for value in args.held_trajectories)
        if len(combined) < 2:
            raise ValueError("combined geometry holdout requires at least two trajectories")
        holdouts.append((str(args.combined_holdout_name), combined))
    for held, held_set in holdouts:
        eval_records = [
            record for record in records
            if str(record["image_id"]).replace("\\", "/").split("/", 1)[0] in held_set
        ]
        train_records = [record for record in records if record not in eval_records]
        if not eval_records or not train_records:
            raise ValueError(f"empty geometry fold for {held}")
        for role, values in (("train", train_records), ("eval", eval_records)):
            path = output_dir / f"{role}_{held}.json"
            if path.exists() and not args.force:
                raise FileExistsError(f"refusing to overwrite {path}")
            payload = {
                "stage": "goal_maplet_geometry_outer_trajectory_fold",
                "role": role,
                "held_trajectory": str(held),
                "held_trajectories": sorted(held_set),
                "outer_trajectory_disjoint": role == "train",
                "source_geometry_manifests": lineage,
                "records": values,
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            written[f"{role}_{held}"] = {"path": str(path), "record_count": len(values)}
    print(json.dumps({
        "stage": "split_goal_maplet_geometry_manifests",
        "total_record_count": len(records),
        "outputs": written,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
