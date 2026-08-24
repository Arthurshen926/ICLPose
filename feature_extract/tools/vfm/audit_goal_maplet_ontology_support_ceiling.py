"""Audit the physical ontology and trained mapper-prototype support ceiling.

The mapper is trained only on parent identities observed on fit/validation
routes.  A retrieval score cannot recover GT-visible mass whose parent is not
in that identity support.  This audit measures that ceiling independently of
retrieval ranking for calibration and query routes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
    PhysicalIncidence,
    load_contributors_in_radio_coordinates,
    token_primitive_visibility,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--surface_maplets", required=True)
    parser.add_argument("--canonical_field", default="")
    parser.add_argument("--include_trajectory", action="append", required=True)
    parser.add_argument("--token_height", type=int, default=36)
    parser.add_argument("--token_width", type=int, default=64)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _filename_identity(path: Path) -> tuple[str, str]:
    stem = path.name
    if not stem.endswith(".npz") or "__" not in stem:
        raise ValueError(f"contributor filename has no route identity: {path}")
    route, image = stem[:-4].split("__", 1)
    if not route or not image:
        raise ValueError(f"contributor filename has empty route identity: {path}")
    return route, f"{route}/{image}"


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty support audit")
    scalar_keys = (
        "parent_gt_visible_mass_supported_fraction_within_physical",
        "child_gt_visible_mass_supported_fraction_within_physical",
        "parent_absolute_visible_mass_support_ceiling",
        "child_absolute_visible_mass_support_ceiling",
        "raw_empty_token_fraction",
        "parent_zero_support_given_raw_nonempty_fraction",
        "child_zero_support_given_raw_nonempty_fraction",
        "parent_prototype_supported_nonempty_token_fraction",
        "child_prototype_supported_nonempty_token_fraction",
        "parent_prototype_supported_mass_mean_per_token",
        "child_prototype_supported_mass_mean_per_token",
    )
    output: dict[str, object] = {"query_count": len(rows)}
    for key in scalar_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = float(np.mean(values))
        output[f"{key}_minimum"] = float(np.min(values))
        output[f"{key}_p10"] = float(np.quantile(values, 0.10))
    for level in ("parent", "child"):
        supported = float(sum(float(row[f"{level}_supported_mass"]) for row in rows))
        physical = float(sum(float(row[f"{level}_physical_mass"]) for row in rows))
        raw = float(sum(float(row["raw_visible_mass"]) for row in rows))
        output[f"{level}_mass_weighted_supported_fraction_within_physical"] = _ratio(
            supported, physical
        )
        output[f"{level}_mass_weighted_absolute_support_ceiling"] = _ratio(
            supported, raw
        )
    if all("canonical_joint_parent_supported_mass" in row for row in rows):
        for level in ("parent", "child"):
            joint = float(
                sum(
                    float(row[f"canonical_joint_{level}_supported_mass"])
                    for row in rows
                )
            )
            physical = float(
                sum(float(row[f"{level}_physical_mass"]) for row in rows)
            )
            output[
                f"canonical_joint_{level}_mass_weighted_supported_fraction_within_physical"
            ] = _ratio(joint, physical)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite ontology support audit")
    requested_routes = sorted({str(value) for value in args.include_trajectory})
    if not requested_routes:
        raise ValueError("include_trajectory cannot be empty")
    contributor_root = Path(args.contributors)
    selected: list[tuple[Path, str, str]] = []
    source_route_counts: Counter[str] = Counter()
    for path in sorted(contributor_root.glob("*.npz")):
        route, image_id = _filename_identity(path)
        source_route_counts[route] += 1
        if route in requested_routes:
            selected.append((path, route, image_id))
    selected_counts = Counter(route for _, route, _ in selected)
    if any(selected_counts.get(route, 0) <= 0 for route in requested_routes):
        raise ValueError("one or more requested routes have no contributors")

    physical_path = Path(args.physical_map)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    incidence = PhysicalIncidence.from_physical_map(physical)
    bank_path = Path(args.surface_maplets)
    bank = VfmSurfaceMapletBank.load_npz(bank_path)
    bank_metadata = dict(bank.metadata or {})
    lineage = bank_metadata.get("supervision_coordinate_lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("surface-maplet bank coordinate lineage is missing")
    if (
        lineage.get("coordinate_correct") is not True
        or str(lineage.get("coordinate_contract", "")) != COORDINATE_CONTRACT
        or str(lineage.get("physical_map_sha256", "")) != physical.content_sha256
        or lineage.get("route_allowlist_applied_before_opening_contributor_archives")
        is not True
        or lineage.get("strict_holdout_present") is not False
    ):
        raise ValueError("surface-maplet bank is not strict coordinate/route bound")
    strict_holdout = sorted(
        str(value) for value in lineage.get("strict_holdout_trajectory_ids", [])
    )
    if not set(requested_routes) <= set(strict_holdout):
        raise ValueError("audited routes are not all mapper strict holdouts")
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    unknown = sorted(set(bank.maplet_ids.tolist()) - set(parent_row_by_id))
    if unknown:
        raise ValueError("surface-maplet bank contains unknown physical parent IDs")
    parent_support = np.zeros((physical.maplet_ids.size,), dtype=bool)
    parent_support[
        [parent_row_by_id[int(value)] for value in bank.maplet_ids.tolist()]
    ] = True
    child_support = parent_support[
        np.asarray(physical.child_parent_rows, dtype=np.int64)
    ]

    canonical_binding: dict[str, object] | None = None
    canonical_parent = None
    canonical_child = None
    if str(args.canonical_field):
        canonical_path = Path(args.canonical_field)
        field = CanonicalSurfaceField.load_npz(canonical_path)
        if field.physical_map_sha256 != physical.content_sha256:
            raise ValueError("canonical field physical-map lineage differs")
        if dict(field.metadata or {}).get("promotion_eligible") is not True:
            raise ValueError("canonical field is not promotion eligible")
        readout = readout_canonical_field(field, physical)
        canonical_parent = np.asarray(readout.parent_coverage) > 0.0
        canonical_child = np.asarray(readout.child_coverage) > 0.0
        canonical_binding = {
            "path": str(canonical_path.resolve()),
            "file_sha256": file_sha256(canonical_path),
            "content_sha256": field.content_sha256,
            "stored_primitive_count": int(field.primitive_rows.size),
            "covered_parent_count": int(np.sum(canonical_parent)),
            "covered_child_count": int(np.sum(canonical_child)),
            "promotion_eligible": True,
        }

    rows: list[dict[str, object]] = []
    coordinate_contracts: set[str] = set()
    for path, route, image_id in selected:
        labels, coordinate_audit = load_contributors_in_radio_coordinates(path)
        coordinate_contracts.add(str(coordinate_audit.get("coordinate_contract", "")))
        token_primitive, raw_mass_by_token, sampling_audit = token_primitive_visibility(
            labels,
            physical,
            token_height=int(args.token_height),
            token_width=int(args.token_width),
        )
        token_parent = (token_primitive @ incidence.primitive_to_parent).tocsr()
        token_child = (token_primitive @ incidence.primitive_to_child).tocsr()
        parent_all = np.asarray(token_parent.sum(axis=1)).reshape(-1)
        child_all = np.asarray(token_child.sum(axis=1)).reshape(-1)
        parent_supported = np.asarray(
            token_parent[:, parent_support].sum(axis=1)
        ).reshape(-1)
        child_supported = np.asarray(
            token_child[:, child_support].sum(axis=1)
        ).reshape(-1)
        raw_nonempty = raw_mass_by_token > 1e-12
        parent_nonempty = parent_supported > 1e-12
        child_nonempty = child_supported > 1e-12
        raw_mass = float(np.sum(raw_mass_by_token))
        parent_mass = float(np.sum(parent_all))
        child_mass = float(np.sum(child_all))
        row: dict[str, object] = {
            "image_id": image_id,
            "trajectory_id": route,
            "contributor": str(path.resolve()),
            "coordinate_valid_raw_sample_fraction": float(
                coordinate_audit["valid_raw_sample_fraction"]
            ),
            "recognized_physical_primitive_mass_fraction": float(
                sampling_audit["recognized_physical_primitive_mass_fraction"]
            ),
            "raw_visible_mass": raw_mass,
            "parent_physical_mass": parent_mass,
            "child_physical_mass": child_mass,
            "parent_supported_mass": float(np.sum(parent_supported)),
            "child_supported_mass": float(np.sum(child_supported)),
            "parent_gt_visible_mass_supported_fraction_within_physical": _ratio(
                float(np.sum(parent_supported)), parent_mass
            ),
            "child_gt_visible_mass_supported_fraction_within_physical": _ratio(
                float(np.sum(child_supported)), child_mass
            ),
            "parent_absolute_visible_mass_support_ceiling": _ratio(
                float(np.sum(parent_supported)), raw_mass
            ),
            "child_absolute_visible_mass_support_ceiling": _ratio(
                float(np.sum(child_supported)), raw_mass
            ),
            "raw_empty_token_fraction": float(np.mean(~raw_nonempty)),
            "parent_prototype_supported_nonempty_token_fraction": float(
                np.mean(parent_nonempty)
            ),
            "child_prototype_supported_nonempty_token_fraction": float(
                np.mean(child_nonempty)
            ),
            "parent_zero_support_given_raw_nonempty_fraction": float(
                np.mean(~parent_nonempty[raw_nonempty]) if np.any(raw_nonempty) else 1.0
            ),
            "child_zero_support_given_raw_nonempty_fraction": float(
                np.mean(~child_nonempty[raw_nonempty]) if np.any(raw_nonempty) else 1.0
            ),
            "parent_prototype_supported_mass_mean_per_token": float(
                np.mean(parent_supported)
            ),
            "child_prototype_supported_mass_mean_per_token": float(
                np.mean(child_supported)
            ),
        }
        if canonical_parent is not None and canonical_child is not None:
            joint_parent = parent_support & canonical_parent
            joint_child = child_support & canonical_child
            row["canonical_joint_parent_supported_mass"] = float(
                token_parent[:, joint_parent].sum()
            )
            row["canonical_joint_child_supported_mass"] = float(
                token_child[:, joint_child].sum()
            )
        rows.append(row)

    if coordinate_contracts != {COORDINATE_CONTRACT}:
        raise ValueError("query support audit coordinate contract differs")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    inventory = [
        {
            "image_id": image_id,
            "resolved_path": str(path.resolve()),
            "file_sha256": file_sha256(path),
        }
        for path, _, image_id in selected
    ]
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_ontology_prototype_support_ceiling_audit_v1",
        "physical_map": {
            "path": str(physical_path.resolve()),
            "file_sha256": file_sha256(physical_path),
            "content_sha256": physical.content_sha256,
            "physical_parent_ontology_count": int(physical.maplet_ids.size),
            "physical_child_ontology_count": int(physical.child_parent_rows.size),
        },
        "mapper_prototype_support": {
            "path": str(bank_path.resolve()),
            "file_sha256": file_sha256(bank_path),
            "prototype_parent_count": int(np.sum(parent_support)),
            "prototype_parent_fraction_of_ontology": float(np.mean(parent_support)),
            "prototype_supported_child_count": int(np.sum(child_support)),
            "training_trajectory_ids": sorted(
                str(value) for value in lineage.get("training_trajectory_ids", [])
            ),
            "validation_trajectory_ids": sorted(
                str(value) for value in lineage.get("validation_trajectory_ids", [])
            ),
            "strict_holdout_trajectory_ids": strict_holdout,
            "coordinate_correct": True,
        },
        "canonical_support": canonical_binding,
        "audited_trajectory_ids": requested_routes,
        "source_contributor_trajectory_counts": dict(sorted(source_route_counts.items())),
        "selected_contributor_trajectory_counts": dict(sorted(selected_counts.items())),
        "selected_contributor_count": len(selected),
        "selected_contributor_inventory_sha256": canonical_json_sha256(inventory),
        "coordinate_contract": COORDINATE_CONTRACT,
        "aggregate": _aggregate(rows),
        "trajectory": {
            route: _aggregate(values) for route, values in sorted(grouped.items())
        },
        "posterior_usability_semantics": {
            "effective_mass": (
                "GT-visible physical mass whose parent identity was present in mapper "
                "prototype supervision"
            ),
            "empty_token": "raw contributor top-k mass <= 1e-12",
            "zero_support_token": (
                "raw-nonempty token with no mass assigned to a prototype-supported "
                "parent or its children"
            ),
            "does_not_measure_predicted_spatial_posterior_quality": True,
        },
        "claim_scope": {
            "retrieval_ranking_used": False,
            "query_pose_used": False,
            "query_ground_truth_opened_only_for_posthoc_visibility_ceiling": True,
            "metric_is_sampled_map_relative_visible_2dgs_surface": True,
            "not_localization_success": True,
        },
        "rows": rows,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "content_sha256": report["content_sha256"],
                "selected_contributor_count": len(selected),
                "mapper_prototype_support": report["mapper_prototype_support"],
                "trajectory": report["trajectory"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
