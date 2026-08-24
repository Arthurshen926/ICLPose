"""Evaluate pose-free RADIO retrieval with coordinate-correct 2DGS visibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    child_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    PhysicalIncidence,
    evaluate_pure_retrieval_query,
)
from feature_extract.vfm.tokens import compute_file_sha256
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument(
        "--surface_maplets",
        default="",
        help=(
            "Optional mapper-supervision bank defining the trained parent-identity "
            "support ceiling. Strict reports should provide it."
        ),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _json_without_duplicates(path: Path) -> dict[str, object]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _contributor_inventory(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            expected = {
                "topk_ids",
                "topk_weights",
                "dominant_depth",
                "pose_w2c",
                "camera_model_id",
                "camera_width",
                "camera_height",
                "camera_params",
                "metadata_json",
            }
            if set(data.files) != expected:
                raise ValueError(f"contributor NPZ members differ: {path}")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            image_id = str(metadata.get("image_id", ""))
            if not image_id or image_id in result:
                raise ValueError("contributor image identities are empty or duplicated")
            result[image_id] = path
    if not result:
        raise ValueError("contributor inventory is empty")
    return result


def _load_contributor(
    path: Path,
) -> tuple[ContributorLabels, int, int, int, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        labels = ContributorLabels(
            np.asarray(data["topk_ids"], dtype=np.int64),
            np.asarray(data["topk_weights"], dtype=np.float32),
            np.asarray(data["pose_w2c"], dtype=np.float64),
        )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        return (
            labels,
            int(np.asarray(data["camera_model_id"]).item()),
            int(np.asarray(data["camera_width"]).item()),
            int(np.asarray(data["camera_height"]).item()),
            np.asarray(data["camera_params"], dtype=np.float64),
            metadata,
        )


def _mean(rows: list[dict[str, object]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        value: object = row
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


_TOKEN_RECALL_KS = (1, 5, 10, 20, 32, 64)


def _token_recall_curve(
    rows: list[dict[str, object]], level: str
) -> dict[str, float]:
    return {
        f"recall_at_{k}": _mean(
            rows, (f"token_{level}", f"conditional_recall_at_{k}")
        )
        for k in _TOKEN_RECALL_KS
    }


def _prototype_support_recall_curve(
    rows: list[dict[str, object]], level: str
) -> dict[str, float]:
    return {
        f"recall_at_{k}": _mean(
            rows,
            (
                "prototype_support",
                f"{level}_conditional_recall_within_prototype_support",
                f"recall_at_{k}",
            ),
        )
        for k in _TOKEN_RECALL_KS
    }


def _failure_categories(row: dict[str, object]) -> list[str]:
    failures: list[str] = []
    parent_ceiling = float(row["parent_representable_visible_mass_ceiling"])
    parent32 = float(row["token_parent"]["conditional_recall_at_32"])
    child64 = float(row["token_child"]["conditional_recall_at_64"])
    child_surface64 = float(
        row["surface_sets"]["child_returned_set"]["exact_visible_mass_recall"]
    )
    if float(row["coordinate_audit"]["valid_raw_sample_fraction"]) < 0.999:
        failures.append("R8_coordinate_or_render_truth_incomplete")
    if parent_ceiling < 0.80:
        failures.append("R0_canonical_physical_coverage_missing")
    if parent32 < 0.80:
        failures.append("R1_parent_identity_missing")
    if parent32 >= 0.80 and child64 < 0.80:
        failures.append("R2_parent_present_child_identity_missing")
    if child64 >= 0.80 and child_surface64 < min(0.80, 0.90 * parent_ceiling):
        failures.append("R3_returned_child_set_misses_positive_candidate_mass")
    if float(row["null_calibration"]["out_of_map_brier"]) > 0.10:
        failures.append("R5_in_map_false_null_probability_high")
    if not bool(
        row["retrieved_set_geometric_non_degeneracy_at_64"]["non_degenerate"]
    ):
        failures.append("R7_retrieved_set_geometrically_degenerate")
    return failures or ["baseline_passes_diagnostic_thresholds"]


def _sequence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"]).split("/", 1)[0]].append(row)
    output = {}
    for sequence, values in sorted(grouped.items()):
        output[sequence] = {
            "query_count": len(values),
            "parent_conditional_recall": _token_recall_curve(values, "parent"),
            "child_conditional_recall": _token_recall_curve(values, "child"),
            "parent_conditional_recall_at_32": _mean(
                values, ("token_parent", "conditional_recall_at_32")
            ),
            "child_conditional_recall_at_64": _mean(
                values, ("token_child", "conditional_recall_at_64")
            ),
            "child_surface_exact_recall_at_64": _mean(
                values,
                ("surface_sets", "child_top64", "exact_visible_mass_recall"),
            ),
            "child_returned_set_exact_recall": _mean(
                values,
                ("surface_sets", "child_returned_set", "exact_visible_mass_recall"),
            ),
            "child_returned_set_tolerant_recall_0.5m": _mean(
                values,
                (
                    "surface_sets",
                    "child_returned_set",
                    "tolerant_visible_mass_recall_0.5m",
                ),
            ),
            "child_surface_tolerant_recall_0.5m_at_64": _mean(
                values,
                (
                    "surface_sets",
                    "child_top64",
                    "tolerant_visible_mass_recall_0.5m",
                ),
            ),
            "parent_surface_tolerant_recall_0.5m_at_64": _mean(
                values,
                (
                    "surface_sets",
                    "parent_top64",
                    "tolerant_visible_mass_recall_0.5m",
                ),
            ),
            "retrieved_set_geometric_non_degeneracy_at_64": float(
                np.mean(
                    [
                        bool(
                            value["retrieved_set_geometric_non_degeneracy_at_64"]
                            ["non_degenerate"]
                        )
                        for value in values
                    ]
                )
            ),
        }
        if all("prototype_support" in value for value in values):
            output[sequence]["prototype_support"] = {
                "parent_conditional_recall_within_prototype_support": (
                    _prototype_support_recall_curve(values, "parent")
                ),
                "child_conditional_recall_within_prototype_support": (
                    _prototype_support_recall_curve(values, "child")
                ),
                "parent_gt_visible_mass_supported_fraction_within_physical": _mean(
                    values,
                    (
                        "prototype_support",
                        "parent_gt_visible_mass_supported_fraction_within_physical",
                    ),
                ),
                "child_gt_visible_mass_supported_fraction_within_physical": _mean(
                    values,
                    (
                        "prototype_support",
                        "child_gt_visible_mass_supported_fraction_within_physical",
                    ),
                ),
                "parent_conditional_recall_at_32_within_prototype_support": _mean(
                    values,
                    (
                        "prototype_support",
                        "parent_conditional_recall_within_prototype_support",
                        "recall_at_32",
                    ),
                ),
                "child_conditional_recall_at_64_within_prototype_support": _mean(
                    values,
                    (
                        "prototype_support",
                        "child_conditional_recall_within_prototype_support",
                        "recall_at_64",
                    ),
                ),
                "raw_empty_token_fraction": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "raw_empty_token_fraction",
                    ),
                ),
                "parent_zero_support_given_raw_nonempty_fraction": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "parent_zero_support_given_raw_nonempty_fraction",
                    ),
                ),
                "child_zero_support_given_raw_nonempty_fraction": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "child_zero_support_given_raw_nonempty_fraction",
                    ),
                ),
                "predicted_parent_prototype_supported_probability_mass_mean_per_token": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "predicted_parent_prototype_supported_probability_mass_mean_per_token",
                    ),
                ),
                "predicted_child_prototype_supported_probability_mass_mean_per_token": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "predicted_child_prototype_supported_probability_mass_mean_per_token",
                    ),
                ),
                "predicted_parent_zero_prototype_supported_mass_token_fraction": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "predicted_parent_zero_prototype_supported_mass_token_fraction",
                    ),
                ),
                "predicted_child_zero_prototype_supported_mass_token_fraction": _mean(
                    values,
                    (
                        "prototype_support",
                        "token_support_posterior_diagnostic",
                        "predicted_child_zero_prototype_supported_mass_token_fraction",
                    ),
                ),
            }
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite pure retrieval evaluation")
    summary_paths = [Path(value) for value in args.retrieval_summary]
    required_false = (
        "uses_query_pose",
        "uses_query_ground_truth",
        "uses_alike",
        "uses_pnp",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_mapping_rgb",
        "uses_image_retrieval",
    )
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    child_area = child_surface_area_m2(physical)
    incidence = PhysicalIncidence.from_physical_map(physical)
    prototype_parent_mask = None
    prototype_child_mask = None
    mapper_bank_binding = None
    if str(args.surface_maplets):
        surface_maplets_path = Path(args.surface_maplets)
        bank = VfmSurfaceMapletBank.load_npz(surface_maplets_path)
        lineage = dict(bank.metadata or {}).get("supervision_coordinate_lineage", {})
        if not isinstance(lineage, dict):
            raise ValueError("surface-maplet bank coordinate lineage is missing")
        if (
            lineage.get("coordinate_correct") is not True
            or str(lineage.get("physical_map_sha256", "")) != physical.content_sha256
            or lineage.get("route_allowlist_applied_before_opening_contributor_archives")
            is not True
            or lineage.get("strict_holdout_present") is not False
        ):
            raise ValueError("surface-maplet bank is not strict coordinate/route bound")
        parent_row_by_id = {
            int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
        }
        unknown = sorted(set(bank.maplet_ids.tolist()) - set(parent_row_by_id))
        if unknown:
            raise ValueError("surface-maplet bank contains unknown physical parents")
        prototype_parent_mask = np.zeros((physical.maplet_ids.size,), dtype=bool)
        prototype_parent_mask[
            [parent_row_by_id[int(value)] for value in bank.maplet_ids.tolist()]
        ] = True
        prototype_child_mask = prototype_parent_mask[
            np.asarray(physical.child_parent_rows, dtype=np.int64)
        ]
        mapper_bank_binding = {
            "path": str(surface_maplets_path.resolve()),
            "file_sha256": compute_file_sha256(surface_maplets_path),
            "physical_parent_ontology_count": int(physical.maplet_ids.size),
            "mapper_prototype_parent_count": int(np.sum(prototype_parent_mask)),
            "mapper_prototype_parent_fraction": float(np.mean(prototype_parent_mask)),
            "prototype_supported_child_count": int(np.sum(prototype_child_mask)),
            "strict_holdout_trajectory_ids": sorted(
                str(value)
                for value in lineage.get("strict_holdout_trajectory_ids", [])
            ),
        }
    summaries = [_json_without_duplicates(path) for path in summary_paths]
    records = []
    for summary in summaries:
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure retrieval run")
        for key in required_false:
            if summary.get(key) is not False:
                raise ValueError(f"retrieval summary requires {key}=false")
        if str(summary.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("retrieval summary physical map differs")
        records.extend(list(summary.get("rows", [])))
    image_ids = [str(record["image_id"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("retrieval summaries contain duplicate query identities")
    if mapper_bank_binding is not None:
        query_routes = {value.split("/", 1)[0] for value in image_ids}
        mapper_holdout = set(mapper_bank_binding["strict_holdout_trajectory_ids"])
        if not query_routes <= mapper_holdout:
            raise ValueError("retrieval query route is not a mapper strict holdout")
    records = sorted(records, key=lambda record: str(record["image_id"]))
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    if int(args.expected_queries) > 0 and int(args.max_queries) <= 0:
        if len(records) != int(args.expected_queries):
            raise ValueError("retrieval result count differs from expected query count")
    contributor_by_image = _contributor_inventory(Path(args.contributors))
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        retrieval_path = Path(str(record["artifact"]))
        if compute_file_sha256(retrieval_path) != str(record["artifact_sha256"]):
            raise ValueError("retrieval artifact file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.image_id != image_id or retrieval.content_sha256 != str(
            record["content_sha256"]
        ):
            raise ValueError("retrieval row identity/content differs")
        contributor_path = contributor_by_image.get(image_id)
        if contributor_path is None:
            raise ValueError(f"missing GT contributor for {image_id}")
        labels, model, width, height, params, contributor_metadata = (
            _load_contributor(contributor_path)
        )
        if str(contributor_metadata.get("image_id", "")) != image_id:
            raise ValueError("contributor image identity differs")
        row = evaluate_pure_retrieval_query(
            retrieval,
            labels,
            physical,
            camera_model_id=model,
            camera_width=width,
            camera_height=height,
            camera_params=params,
            incidence=incidence,
            # Token recall is cheap at all cutoffs.  Physical distance metrics
            # require a KD-tree for every parent/child cutoff; the full-run
            # decision and failure taxonomy only consume Top64, so compute it
            # once instead of rebuilding ten trees per query.
            surface_ks=(64,),
            precomputed_child_surface_area_m2=child_area,
            prototype_supported_parent_rows=prototype_parent_mask,
            prototype_supported_child_rows=prototype_child_mask,
        )
        row["failure_categories"] = _failure_categories(row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "count": len(records),
                    "image_id": image_id,
                    "failure_categories": row["failure_categories"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    failure_counts = Counter(
        category for row in rows for category in row["failure_categories"]
    )
    total_physical_surface_area_m2 = float(
        np.sum(
            np.pi
            * np.asarray(physical.primitive_scale1, dtype=np.float64)
            * np.asarray(physical.primitive_scale2, dtype=np.float64)
        )
    )
    aggregate = {
        "query_count": len(rows),
        "parent_conditional_recall": _token_recall_curve(rows, "parent"),
        "child_conditional_recall": _token_recall_curve(rows, "child"),
        "parent_representable_visible_mass_ceiling": _mean(
            rows, ("parent_representable_visible_mass_ceiling",)
        ),
        "child_representable_visible_mass_ceiling": _mean(
            rows, ("child_representable_visible_mass_ceiling",)
        ),
        "parent_conditional_recall_at_32": _mean(
            rows, ("token_parent", "conditional_recall_at_32")
        ),
        "parent_absolute_visible_mass_coverage_at_32": _mean(
            rows, ("token_parent", "absolute_visible_mass_coverage_at_32")
        ),
        "child_conditional_recall_at_64": _mean(
            rows, ("token_child", "conditional_recall_at_64")
        ),
        "child_absolute_visible_mass_coverage_at_64": _mean(
            rows, ("token_child", "absolute_visible_mass_coverage_at_64")
        ),
        "parent_surface_exact_recall_at_64": _mean(
            rows, ("surface_sets", "parent_top64", "exact_visible_mass_recall")
        ),
        "parent_surface_tolerant_recall_0.5m_at_64": _mean(
            rows,
            (
                "surface_sets",
                "parent_top64",
                "tolerant_visible_mass_recall_0.5m",
            ),
        ),
        "parent_surface_distance_p90_m_at_64": _mean(
            rows, ("surface_sets", "parent_top64", "weighted_distance_p90_m")
        ),
        "parent_predicted_surface_area_m2_at_64": _mean(
            rows, ("surface_sets", "parent_top64", "predicted_surface_area_m2")
        ),
        "parent_visible_surface_area_precision_at_64": _mean(
            rows,
            ("surface_sets", "parent_top64", "visible_primitive_area_precision"),
        ),
        "parent_surface_exact_recall_at_area20pct": _mean(
            rows, ("surface_sets", "parent_area20pct", "exact_visible_mass_recall")
        ),
        "parent_surface_tolerant_recall_0.5m_at_area20pct": _mean(
            rows,
            (
                "surface_sets",
                "parent_area20pct",
                "tolerant_visible_mass_recall_0.5m",
            ),
        ),
        "parent_surface_distance_p90_m_at_area20pct": _mean(
            rows, ("surface_sets", "parent_area20pct", "weighted_distance_p90_m")
        ),
        "parent_selected_count_at_area20pct": _mean(
            rows, ("surface_sets", "parent_area20pct", "selected_parent_count")
        ),
        "parent_charged_surface_area_m2_at_area20pct": _mean(
            rows, ("surface_sets", "parent_area20pct", "charged_surface_area_m2")
        ),
        "child_surface_exact_recall_at_64": _mean(
            rows, ("surface_sets", "child_top64", "exact_visible_mass_recall")
        ),
        "child_surface_tolerant_recall_0.5m_at_64": _mean(
            rows,
            (
                "surface_sets",
                "child_top64",
                "tolerant_visible_mass_recall_0.5m",
            ),
        ),
        "child_surface_distance_p90_m_at_64": _mean(
            rows, ("surface_sets", "child_top64", "weighted_distance_p90_m")
        ),
        "child_predicted_surface_area_m2_at_64": _mean(
            rows, ("surface_sets", "child_top64", "predicted_surface_area_m2")
        ),
        "child_visible_surface_area_precision_at_64": _mean(
            rows,
            ("surface_sets", "child_top64", "visible_primitive_area_precision"),
        ),
        "child_returned_set_exact_recall": _mean(
            rows, ("surface_sets", "child_returned_set", "exact_visible_mass_recall")
        ),
        "child_returned_set_tolerant_recall_0.5m": _mean(
            rows,
            (
                "surface_sets",
                "child_returned_set",
                "tolerant_visible_mass_recall_0.5m",
            ),
        ),
        "child_returned_set_distance_p90_m": _mean(
            rows, ("surface_sets", "child_returned_set", "weighted_distance_p90_m")
        ),
        "child_returned_set_predicted_surface_area_m2": _mean(
            rows, ("surface_sets", "child_returned_set", "predicted_surface_area_m2")
        ),
        "child_returned_set_visible_surface_area_precision": _mean(
            rows,
            (
                "surface_sets",
                "child_returned_set",
                "visible_primitive_area_precision",
            ),
        ),
        "child_returned_set_selected_child_count": _mean(
            rows, ("surface_sets", "child_returned_set", "selected_child_count")
        ),
        "child_returned_set_connected_component_count": _mean(
            rows,
            ("surface_sets", "child_returned_set", "connected_component_count"),
        ),
        "child_returned_set_mean_children_per_component": _mean(
            rows,
            ("surface_sets", "child_returned_set", "mean_children_per_component"),
        ),
        "hierarchical_child_surface_exact_recall_parent16_x4": _mean(
            rows,
            (
                "surface_sets",
                "hierarchical_child_parent16_x4",
                "exact_visible_mass_recall",
            ),
        ),
        "hierarchical_child_surface_tolerant_recall_0.5m_parent16_x4": _mean(
            rows,
            (
                "surface_sets",
                "hierarchical_child_parent16_x4",
                "tolerant_visible_mass_recall_0.5m",
            ),
        ),
        "hierarchical_child_surface_distance_p90_m_parent16_x4": _mean(
            rows,
            (
                "surface_sets",
                "hierarchical_child_parent16_x4",
                "weighted_distance_p90_m",
            ),
        ),
        "hierarchical_child_predicted_surface_area_m2_parent16_x4": _mean(
            rows,
            (
                "surface_sets",
                "hierarchical_child_parent16_x4",
                "predicted_surface_area_m2",
            ),
        ),
        "hierarchical_child_visible_surface_area_precision_parent16_x4": _mean(
            rows,
            (
                "surface_sets",
                "hierarchical_child_parent16_x4",
                "visible_primitive_area_precision",
            ),
        ),
        "retrieved_set_geometric_non_degeneracy_at_64": float(
            np.mean(
                [
                    bool(
                        row["retrieved_set_geometric_non_degeneracy_at_64"]
                        ["non_degenerate"]
                    )
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
        "out_of_map_brier": _mean(rows, ("null_calibration", "out_of_map_brier")),
        "out_of_map_ece": _mean(rows, ("null_calibration", "out_of_map_ece")),
    }
    if mapper_bank_binding is not None:
        aggregate["prototype_support"] = {
            **mapper_bank_binding,
            "parent_conditional_recall_within_prototype_support": (
                _prototype_support_recall_curve(rows, "parent")
            ),
            "child_conditional_recall_within_prototype_support": (
                _prototype_support_recall_curve(rows, "child")
            ),
            "parent_gt_visible_mass_supported_fraction_within_physical": _mean(
                rows,
                (
                    "prototype_support",
                    "parent_gt_visible_mass_supported_fraction_within_physical",
                ),
            ),
            "child_gt_visible_mass_supported_fraction_within_physical": _mean(
                rows,
                (
                    "prototype_support",
                    "child_gt_visible_mass_supported_fraction_within_physical",
                ),
            ),
            "parent_conditional_recall_at_32_within_prototype_support": _mean(
                rows,
                (
                    "prototype_support",
                    "parent_conditional_recall_within_prototype_support",
                    "recall_at_32",
                ),
            ),
            "child_conditional_recall_at_64_within_prototype_support": _mean(
                rows,
                (
                    "prototype_support",
                    "child_conditional_recall_within_prototype_support",
                    "recall_at_64",
                ),
            ),
            "raw_empty_token_fraction": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "raw_empty_token_fraction",
                ),
            ),
            "parent_zero_support_given_raw_nonempty_fraction": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "parent_zero_support_given_raw_nonempty_fraction",
                ),
            ),
            "child_zero_support_given_raw_nonempty_fraction": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "child_zero_support_given_raw_nonempty_fraction",
                ),
            ),
            "predicted_parent_prototype_supported_probability_mass_mean_per_token": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "predicted_parent_prototype_supported_probability_mass_mean_per_token",
                ),
            ),
            "predicted_child_prototype_supported_probability_mass_mean_per_token": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "predicted_child_prototype_supported_probability_mass_mean_per_token",
                ),
            ),
            "predicted_parent_zero_prototype_supported_mass_token_fraction": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "predicted_parent_zero_prototype_supported_mass_token_fraction",
                ),
            ),
            "predicted_child_zero_prototype_supported_mass_token_fraction": _mean(
                rows,
                (
                    "prototype_support",
                    "token_support_posterior_diagnostic",
                    "predicted_child_zero_prototype_supported_mass_token_fraction",
                ),
            ),
        }
    aggregate["total_physical_surface_area_m2"] = total_physical_surface_area_m2
    for level in ("parent", "child"):
        predicted_area = float(
            aggregate[f"{level}_predicted_surface_area_m2_at_64"]
        )
        selection_fraction = predicted_area / max(total_physical_surface_area_m2, 1e-12)
        exact_recall = float(aggregate[f"{level}_surface_exact_recall_at_64"])
        aggregate[f"{level}_map_surface_area_fraction_at_64"] = selection_fraction
        aggregate[f"{level}_exact_surface_recall_enrichment_at_64"] = (
            exact_recall / max(selection_fraction, 1e-12)
        )
    hierarchical_fraction = float(
        aggregate["hierarchical_child_predicted_surface_area_m2_parent16_x4"]
    ) / max(total_physical_surface_area_m2, 1e-12)
    aggregate[
        "hierarchical_child_map_surface_area_fraction_parent16_x4"
    ] = hierarchical_fraction
    aggregate[
        "hierarchical_child_exact_surface_recall_enrichment_parent16_x4"
    ] = float(
        aggregate["hierarchical_child_surface_exact_recall_parent16_x4"]
    ) / max(hierarchical_fraction, 1e-12)
    returned_fraction = float(
        aggregate["child_returned_set_predicted_surface_area_m2"]
    ) / max(total_physical_surface_area_m2, 1e-12)
    aggregate["child_returned_set_map_surface_area_fraction"] = returned_fraction
    aggregate["child_returned_set_exact_surface_recall_enrichment"] = float(
        aggregate["child_returned_set_exact_recall"]
    ) / max(returned_fraction, 1e-12)
    gates = {
        "parent_conditional_recall_at_32_ge_0.97": bool(
            aggregate["parent_conditional_recall_at_32"] >= 0.97
        ),
        "child_conditional_recall_at_64_ge_0.90": bool(
            aggregate["child_conditional_recall_at_64"] >= 0.90
        ),
        "parent_surface_tolerant_recall_0.5m_at_64_ge_0.95": bool(
            aggregate["parent_surface_tolerant_recall_0.5m_at_64"] >= 0.95
        ),
        "retrieved_set_geometric_non_degeneracy_at_64_ge_0.95": bool(
            aggregate["retrieved_set_geometric_non_degeneracy_at_64"] >= 0.95
        ),
    }
    null_diagnostic = {
        "in_map_query_ece": float(aggregate["out_of_map_ece"]),
        "in_map_query_brier": float(aggregate["out_of_map_brier"]),
        "out_of_map_detection_gate_evaluable": False,
        "reason": "the official 530-query set contains no declared out-of-map queries",
    }
    report = {
        "artifact_type": "goal_maplet_pure_retrieval_surface_evaluation_v1",
        "retrieval_summaries": [str(path.resolve()) for path in summary_paths],
        "retrieval_summary_sha256": [
            compute_file_sha256(path) for path in summary_paths
        ],
        "contributors": str(Path(args.contributors).resolve()),
        "physical_map_sha256": physical.content_sha256,
        "mapper_prototype_support_binding": mapper_bank_binding,
        "aggregate": aggregate,
        "sequence": _sequence_summary(rows),
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "provisional_retrieval_gates": gates,
        "all_provisional_gates_pass": bool(all(gates.values())),
        "null_diagnostic": null_diagnostic,
        "claim_scope": {
            "retrieval_only": True,
            "gt_opened_only_by_evaluator": True,
            "coordinate_correct_simple_radial_sampling": True,
            "not_pose_estimation": True,
            "not_localization_success": True,
            "not_real_world_geometry_truth": True,
            "out_of_map_detection_not_evaluated_without_ood_queries": True,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
