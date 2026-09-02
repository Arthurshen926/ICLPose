"""Freeze nested route-balanced mapping-view inventories without query evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def _route_progressive_addition(
    inventory: list[str], selected: set[str]
) -> str:
    index = {name: i for i, name in enumerate(inventory)}
    selected_indices = [index[name] for name in selected if name in index]
    candidates = [name for name in inventory if name not in selected]
    if not candidates:
        raise ValueError("route inventory is exhausted")
    denominator = max(len(inventory) - 1, 1)
    if not selected_indices:
        target = 0.5 * denominator
        return min(candidates, key=lambda name: (abs(index[name] - target), index[name]))
    return min(
        candidates,
        key=lambda name: (
            -min(abs(index[name] - value) / denominator for value in selected_indices),
            index[name],
        ),
    )


def plan_nested_views(
    records: list[dict[str, object]],
    *,
    routes: tuple[str, ...],
    required_names: list[str],
    maximum_count: int,
) -> tuple[list[str], dict[str, list[str]]]:
    allowed = tuple(sorted(set(routes)))
    grouped = {
        route: sorted(
            str(row["image_id"])
            for row in records
            if str(row["image_id"]).split("/", 1)[0] == route
        )
        for route in allowed
    }
    if any(not values for values in grouped.values()):
        raise ValueError("one or more requested mapping routes are empty")
    all_names = {name for values in grouped.values() for name in values}
    if len(all_names) != sum(map(len, grouped.values())):
        raise ValueError("mapping RADIO inventory is duplicated")
    required = list(dict.fromkeys(required_names))
    if not required or not set(required).issubset(all_names):
        raise ValueError("required seed views are empty or outside mapping inventory")
    if maximum_count < len(required) or maximum_count > len(all_names):
        raise ValueError("maximum mapping-view count is outside inventory")

    selected = list(required)
    selected_set = set(selected)
    while len(selected) < maximum_count:
        counts = {
            route: sum(name.split("/", 1)[0] == route for name in selected)
            for route in allowed
        }
        available = [
            route for route in allowed
            if counts[route] < len(grouped[route])
        ]
        route = min(available, key=lambda value: (counts[value], value))
        addition = _route_progressive_addition(grouped[route], selected_set)
        selected.append(addition)
        selected_set.add(addition)
    return selected, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--required_ids", type=Path, required=True)
    parser.add_argument("--prefix_counts", nargs="+", type=int, required=True)
    parser.add_argument("--output_additions_after", type=int)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_plan", type=Path, required=True)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to overwrite route-balanced mapping plan")
    radio = json.loads(args.radio_manifest.read_text())
    required = [line.strip() for line in args.required_ids.read_text().splitlines() if line.strip()]
    counts = sorted(set(int(value) for value in args.prefix_counts))
    if not counts:
        raise ValueError("no prefix counts requested")
    selected, grouped = plan_nested_views(
        list(radio["records"]), routes=tuple(args.routes), required_names=required,
        maximum_count=counts[-1],
    )
    args.output_dir.mkdir(parents=True)
    prefixes = {}
    for count in counts:
        if count < len(required):
            raise ValueError("prefix count is smaller than required seed inventory")
        names = selected[:count]
        output = args.output_dir / f"mapping_image_ids_route_balanced_{count}.txt"
        output.write_text("\n".join(names) + "\n")
        prefixes[str(count)] = {
            "selected_names_in_order": names,
            "route_counts": {
                route: sum(name.split("/", 1)[0] == route for name in names)
                for route in sorted(grouped)
            },
            "ids_file": str(output),
            "ids_file_sha256": file_sha256(output),
        }
    additions = None
    if args.output_additions_after is not None:
        base = int(args.output_additions_after)
        if base < len(required) or base >= counts[-1]:
            raise ValueError("addition base count is outside the selected inventory")
        output = args.output_dir / f"mapping_image_ids_additions_after_{base}_to_{counts[-1]}.txt"
        names = selected[base:counts[-1]]
        output.write_text("\n".join(names) + "\n")
        additions = {
            "base_count": base, "maximum_count": counts[-1],
            "selected_names_in_order": names, "ids_file": str(output),
            "ids_file_sha256": file_sha256(output),
        }
    payload = {
        "artifact_type": "goal_maplet_nested_route_balanced_mapping_view_plan_v1",
        "selection": "required_seed_then_lowest_route_count_progressive_max_gap",
        "routes": sorted(grouped),
        "source_route_counts": {route: len(values) for route, values in grouped.items()},
        "required_seed_count": len(required),
        "required_ids_file_sha256": file_sha256(args.required_ids),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "prefixes": prefixes,
        "additions": additions,
        "uses_query_or_ground_truth": False,
        "test_routes_consumed": False,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "content_sha256": payload["content_sha256"],
        "plan_file_sha256": file_sha256(args.output_plan),
        "prefix_route_counts": {key: value["route_counts"] for key, value in prefixes.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
