"""Run pose-free full-token RADIO retrieval over physical parent/child surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.child_retrieval import (
    retrieve_children_given_parents,
)
from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    COMPONENT_SEMANTICS as CONNECTED_FINE_SUPPORT_SEMANTICS,
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.feature_contract import (
    FieldFeatureContract,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    CHILD_PROBABILITY_SEMANTICS,
    SELECTION_SEMANTICS as FINE_SUPPORT_SELECTION_SEMANTICS,
    child_surface_area_m2,
    select_fine_supports_under_area_budget,
    total_map_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.multimodal_parent_retrieval import (
    PARENT_SCORE_ANONYMOUS_MODES,
    PARENT_SCORE_SINGLE_MEAN,
    build_anonymous_parent_mode_readout,
    score_anonymous_parent_modes,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PARENT_SCENE_RANK_RAW,
    PARENT_SCENE_RANK_SURFACE_DENSITY,
    SCENE_AGGREGATION,
    PureRadioPhysicalRetrieval,
    aggregate_sparse_token_evidence,
    all_radio_token_coordinates,
    rank_children_with_physical_iou_nms,
    rank_parent_regions,
)
from feature_extract.vfm.localization_goal_maplet.retrieval import (
    ValidityCalibration,
    retrieve_maplet_posterior_decomposed,
    retrieve_maplet_posterior_from_scores,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)
from feature_extract.vfm.tokens import TokenBankManifest, compute_file_sha256


POOL_SIZES = (1, 3, 5, 9)
POOL_WEIGHTS = (0.40, 0.30, 0.20, 0.10)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--validity_calibration", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_parent_candidates", type=int, default=64)
    parser.add_argument("--maximum_child_candidates", type=int, default=64)
    parser.add_argument("--maximum_scene_parents", type=int, default=64)
    parser.add_argument("--maximum_scene_children", type=int, default=64)
    parser.add_argument("--maximum_budgeted_scene_children", type=int, default=2048)
    parser.add_argument(
        "--target_candidate_posterior_mass_fraction", type=float, default=1.0
    )
    parser.add_argument(
        "--fine_support_area_fraction",
        type=float,
        default=0.0,
        help="positive value enables posterior-mass selection under this map-area budget",
    )
    parser.add_argument("--parent_temperature", type=float, default=0.07)
    parser.add_argument(
        "--parent_score_semantics",
        choices=(PARENT_SCORE_SINGLE_MEAN, PARENT_SCORE_ANONYMOUS_MODES),
        default=PARENT_SCORE_SINGLE_MEAN,
    )
    parser.add_argument("--parent_mode_temperature", type=float, default=0.03)
    parser.add_argument(
        "--parent_scene_ranking_semantics",
        choices=(PARENT_SCENE_RANK_RAW, PARENT_SCENE_RANK_SURFACE_DENSITY),
        default=PARENT_SCENE_RANK_RAW,
    )
    parser.add_argument("--child_temperature", type=float, default=0.07)
    parser.add_argument("--maximum_child_primitive_iou", type=float, default=0.50)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--include_trajectories", nargs="+", default=[])
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument(
        "--allow_legacy_coordinate_misaligned_control",
        action="store_true",
        help="diagnostic-only opt-in for old canonical fields without raw-RADIO coordinate alignment",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_field_coordinate_contract(
    field: CanonicalSurfaceField,
    *,
    allow_legacy_coordinate_misaligned_control: bool,
) -> tuple[bool, str]:
    coordinate_correct = bool(field.metadata.get("coordinate_correct", False))
    coordinate_contract = str(
        field.metadata.get("contributor_to_radio_coordinate_contract", "")
    )
    if not coordinate_correct and not bool(allow_legacy_coordinate_misaligned_control):
        raise ValueError(
            "pure retrieval requires a coordinate-correct canonical field; "
            "legacy fields are diagnostic controls only"
        )
    return coordinate_correct, coordinate_contract


def _atomic_save(result: PureRadioPhysicalRetrieval, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        temporary.unlink()
    result.save_npz(temporary)
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite pure retrieval summary")
    physical_path = Path(args.physical_map)
    field_path = Path(args.canonical_field)
    mapper_path = Path(args.surface_mapper)
    contract_path = Path(args.field_feature_contract)
    calibration_path = Path(args.validity_calibration)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    coordinate_correct, coordinate_contract = _validate_field_coordinate_contract(
        field,
        allow_legacy_coordinate_misaligned_control=bool(
            args.allow_legacy_coordinate_misaligned_control
        ),
    )
    contract = FieldFeatureContract.load_json(contract_path)
    if contract.query_readout_type != "surface_maplet_mapper":
        raise ValueError("pure retrieval requires the RADIO surface mapper")
    contract.validate(field, query_readout_path=mapper_path)
    calibration = ValidityCalibration.load_json(calibration_path)
    for name, expected in (
        ("physical_map_sha256", physical.content_sha256),
        ("canonical_field_sha256", field.content_sha256),
        ("pooling", "current_1x1_3x3_5x5_9x9"),
    ):
        if str(calibration.metadata.get(name, "")) != str(expected):
            raise ValueError(f"validity calibration {name} differs")
    calibration_parent_semantics = str(
        calibration.metadata.get("parent_score_semantics", PARENT_SCORE_SINGLE_MEAN)
    )
    if calibration_parent_semantics != str(args.parent_score_semantics):
        raise ValueError("validity calibration parent score semantics differ")
    readout = readout_canonical_field(field, physical)
    fine_child_area = (
        child_surface_area_m2(physical)
        if float(args.fine_support_area_fraction) > 0.0
        else None
    )
    fine_total_map_area = (
        total_map_surface_area_m2(physical)
        if float(args.fine_support_area_fraction) > 0.0
        else None
    )
    anonymous_parent_readout = (
        build_anonymous_parent_mode_readout(field, physical)
        if str(args.parent_score_semantics) == PARENT_SCORE_ANONYMOUS_MODES
        else None
    )
    mapper, mapper_metadata = load_surface_maplet_mapper(
        mapper_path, device=str(args.device)
    )
    query_manifest_path = Path(args.query_manifest)
    manifest = TokenBankManifest.from_json(query_manifest_path)
    manifest.validate()
    records = list(manifest.records)
    included_trajectories = {
        str(value) for value in args.include_trajectories
    }
    if included_trajectories:
        records = [
            record
            for record in records
            if str(record.image_id).split("/", 1)[0] in included_trajectories
        ]
    if (
        int(args.shard_count) <= 0
        or int(args.shard_index) < 0
        or int(args.shard_index) >= int(args.shard_count)
    ):
        raise ValueError("invalid retrieval shard_index/shard_count")
    records = records[int(args.shard_index) :: int(args.shard_count)]
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    if not records:
        raise ValueError("pure retrieval query manifest is empty")
    config = RadioFinalRegionConfig(
        pool_sizes=POOL_SIZES, pool_weights=POOL_WEIGHTS
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    for index, record in enumerate(records):
        destination = output_dir / (record.image_id.replace("/", "__") + ".npz")
        if destination.exists() and not bool(args.force):
            result = PureRadioPhysicalRetrieval.load_npz(destination)
            if result.image_id != record.image_id:
                raise ValueError("existing pure retrieval image identity differs")
            rows.append(
                {
                    "image_id": result.image_id,
                    "artifact": str(destination.resolve()),
                    "artifact_sha256": compute_file_sha256(destination),
                    "content_sha256": result.content_sha256,
                    "reused_verified": True,
                }
            )
            continue
        started = time.perf_counter()
        raw = _load_raw_final(Path(record.token_path), "radio_final")
        loaded_at = time.perf_counter()
        if raw.ndim != 3 or raw.shape[1:] != (36, 64):
            raise ValueError("pure retrieval requires complete RADIO 36x64 final grid")
        token_xy = all_radio_token_coordinates(36, 64)
        mapped = mapper.project(raw).measurement_context
        mapped_at = time.perf_counter()
        descriptor = encode_radio_final_regions(mapped, token_xy, config)
        encoded_at = time.perf_counter()
        if anonymous_parent_readout is None:
            parent = retrieve_maplet_posterior_decomposed(
                descriptor,
                readout.parent_descriptors,
                physical.maplet_ids,
                readout.parent_coverage > 0.0,
                maximum_candidates=int(args.maximum_parent_candidates),
                temperature=float(args.parent_temperature),
                null_similarity_center=float(calibration.center),
                null_similarity_scale=float(calibration.scale),
            )
        else:
            parent = retrieve_maplet_posterior_from_scores(
                score_anonymous_parent_modes(
                    descriptor,
                    anonymous_parent_readout,
                    mode_temperature=float(args.parent_mode_temperature),
                ),
                physical.maplet_ids,
                anonymous_parent_readout.parent_coverage > 0.0,
                maximum_candidates=int(args.maximum_parent_candidates),
                temperature=float(args.parent_temperature),
                null_similarity_center=float(calibration.center),
                null_similarity_scale=float(calibration.scale),
            )
        parent_at = time.perf_counter()
        child = retrieve_children_given_parents(
            descriptor,
            parent.candidate_ids,
            parent.candidate_probabilities,
            parent.unresolved_probabilities,
            readout.child_descriptors,
            readout.child_coverage,
            physical,
            maximum_child_candidates=int(args.maximum_child_candidates),
            temperature=float(args.child_temperature),
        )
        child_at = time.perf_counter()
        parent_rows = np.asarray(
            [
                [parent_row_by_id.get(int(value), -1) for value in values]
                for values in parent.candidate_ids.tolist()
            ],
            dtype=np.int64,
        )
        parent_scene_score = aggregate_sparse_token_evidence(
            token_xy,
            parent_rows,
            parent.candidate_probabilities,
            entity_count=physical.maplet_ids.size,
            token_height=36,
            token_width=64,
        )
        parent_order, ranked_parent_score = rank_parent_regions(
            parent_scene_score,
            physical,
            maximum_parents=int(args.maximum_scene_parents),
            semantics=str(args.parent_scene_ranking_semantics),
        )
        child_scene_score = aggregate_sparse_token_evidence(
            token_xy,
            child.candidate_child_rows,
            child.candidate_probabilities,
            entity_count=physical.child_parent_rows.size,
            token_height=36,
            token_width=64,
        )
        positive_child_count = int(np.sum(child_scene_score > 0.0))
        if float(args.fine_support_area_fraction) > 0.0:
            fine_selection = select_fine_supports_under_area_budget(
                child.candidate_child_rows,
                child.candidate_probabilities,
                physical.maplet_ids[parent_order],
                physical,
                maximum_area_fraction=float(args.fine_support_area_fraction),
                maximum_children=int(args.maximum_budgeted_scene_children),
                maximum_primitive_iou=float(args.maximum_child_primitive_iou),
                target_candidate_posterior_mass_fraction=float(
                    args.target_candidate_posterior_mass_fraction
                ),
                precomputed_child_surface_area_m2=fine_child_area,
                precomputed_total_map_surface_area_m2=fine_total_map_area,
            )
            scene_child_rows = fine_selection.child_rows
            scene_child_scores = fine_selection.posterior_mass
            suppressed = fine_selection.suppressed_duplicate_count
            connected_supports = connected_fine_support_components(
                scene_child_rows,
                physical,
                precomputed_child_surface_area_m2=fine_child_area,
            )
        else:
            fine_selection = None
            connected_supports = None
            scene_child_rows, scene_child_scores, suppressed = (
                rank_children_with_physical_iou_nms(
                    child_scene_score,
                    physical,
                    maximum_children=int(args.maximum_scene_children),
                    maximum_primitive_iou=float(args.maximum_child_primitive_iou),
                )
            )
        elapsed = float(time.perf_counter() - started)
        stage_seconds = {
            "radio_npz_load": float(loaded_at - started),
            "surface_mapper_projection": float(mapped_at - loaded_at),
            "full_radio_layout_encoding": float(encoded_at - mapped_at),
            "parent_multimode_scoring": float(parent_at - encoded_at),
            "child_joint_posterior_expansion": float(child_at - parent_at),
            "scene_aggregation_and_set_selection": float(
                started + elapsed - child_at
            ),
        }
        result = PureRadioPhysicalRetrieval(
            image_id=record.image_id,
            token_xy=token_xy,
            token_parent_ids=parent.candidate_ids,
            token_parent_probabilities=parent.candidate_probabilities,
            token_out_of_map_probabilities=parent.out_of_map_probabilities,
            token_in_map_tail_probabilities=parent.truncated_tail_probabilities,
            token_child_rows=child.candidate_child_rows,
            token_child_probabilities=child.candidate_probabilities,
            scene_parent_ids=physical.maplet_ids[parent_order],
            scene_parent_scores=ranked_parent_score,
            scene_child_rows=scene_child_rows,
            scene_child_scores=scene_child_scores,
            physical_map_sha256=physical.content_sha256,
            metadata={
                "artifact_type": "goal_maplet_pure_radio_physical_retrieval_v1",
                "representation": "full_radio_token_grid_to_physical_parent_then_child",
                "token_height": 36,
                "token_width": 64,
                "token_count": 2304,
                "vfm_layer": "radio_final",
                "pool_sizes": list(POOL_SIZES),
                "pool_weights": list(POOL_WEIGHTS),
                "scene_aggregation": SCENE_AGGREGATION,
                "maximum_parent_candidates": int(args.maximum_parent_candidates),
                "parent_score_semantics": str(args.parent_score_semantics),
                "parent_scene_ranking_semantics": str(
                    args.parent_scene_ranking_semantics
                ),
                "parent_mode_temperature": float(args.parent_mode_temperature),
                "anonymous_parent_mode_readout_sha256": (
                    anonymous_parent_readout.content_sha256
                    if anonymous_parent_readout is not None
                    else None
                ),
                "maximum_child_candidates": int(args.maximum_child_candidates),
                "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
                "child_probability_is_calibrated_credible_mass": False,
                "maximum_scene_parents": int(args.maximum_scene_parents),
                "maximum_scene_children": int(
                    args.maximum_budgeted_scene_children
                    if fine_selection is not None
                    else args.maximum_scene_children
                ),
                "maximum_budgeted_scene_children": int(
                    args.maximum_budgeted_scene_children
                ),
                "fine_support_selection_semantics": (
                    FINE_SUPPORT_SELECTION_SEMANTICS
                    if fine_selection is not None
                    else "fixed_topk_block_peak_score_v1"
                ),
                "maximum_fine_support_area_fraction": (
                    float(args.fine_support_area_fraction)
                    if fine_selection is not None
                    else None
                ),
                "selected_fine_support_area_m2": (
                    float(fine_selection.selected_surface_area_m2)
                    if fine_selection is not None
                    else None
                ),
                "target_candidate_posterior_mass_fraction": (
                    float(args.target_candidate_posterior_mass_fraction)
                    if fine_selection is not None
                    else None
                ),
                "achieved_candidate_posterior_mass_fraction": (
                    float(
                        fine_selection.posterior_mass.sum()
                        / max(fine_selection.eligible_posterior_mass, 1e-12)
                    )
                    if fine_selection is not None
                    else None
                ),
                "connected_fine_support_semantics": (
                    CONNECTED_FINE_SUPPORT_SEMANTICS
                    if connected_supports is not None
                    else None
                ),
                "connected_fine_support_count": (
                    int(connected_supports.component_count)
                    if connected_supports is not None
                    else None
                ),
                "maximum_child_primitive_iou": float(
                    args.maximum_child_primitive_iou
                ),
                "positive_scene_children_before_iou_nms": positive_child_count,
                "children_suppressed_by_primitive_iou_nms": suppressed,
                "elapsed_seconds": elapsed,
                "stage_seconds": stage_seconds,
                "radio_final_content_sha256": _array_sha256(raw),
                "physical_map_file_sha256": compute_file_sha256(physical_path),
                "canonical_field_file_sha256": compute_file_sha256(field_path),
                "canonical_field_coordinate_correct": coordinate_correct,
                "canonical_field_coordinate_contract": coordinate_contract,
                "legacy_coordinate_control": bool(not coordinate_correct),
                "surface_mapper_file_sha256": compute_file_sha256(mapper_path),
                "field_feature_contract_file_sha256": compute_file_sha256(
                    contract_path
                ),
                "validity_calibration_file_sha256": compute_file_sha256(
                    calibration_path
                ),
                "query_mapper_pool_sizes": list(
                    mapper_metadata.get("pool_sizes", ())
                ),
                "uses_query_pose": False,
                "uses_query_ground_truth": False,
                "uses_alike": False,
                "uses_pnp": False,
                "uses_sfm_points": False,
                "uses_sfm_tracks": False,
                "uses_mapping_rgb": False,
                "uses_image_retrieval": False,
            },
        )
        _atomic_save(result, destination)
        rows.append(
            {
                "image_id": record.image_id,
                "artifact": str(destination.resolve()),
                "artifact_sha256": compute_file_sha256(destination),
                "content_sha256": result.content_sha256,
                "elapsed_seconds": elapsed,
                "stage_seconds": stage_seconds,
                "reused_verified": False,
            }
        )
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "count": len(records),
                    "image_id": record.image_id,
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "query_count": len(rows),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "query_manifest": str(query_manifest_path.resolve()),
        "query_manifest_sha256": compute_file_sha256(query_manifest_path),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "canonical_field_coordinate_correct": coordinate_correct,
        "canonical_field_coordinate_contract": coordinate_contract,
        "legacy_coordinate_control": bool(not coordinate_correct),
        "field_feature_contract_sha256": contract.content_sha256,
        "validity_calibration_sha256": calibration.content_sha256,
        "method": "full_radio_36x64_parent_then_child_physical_retrieval",
        "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
        "child_probability_is_calibrated_credible_mass": False,
        "scene_aggregation": SCENE_AGGREGATION,
        "parent_score_semantics": str(args.parent_score_semantics),
        "parent_scene_ranking_semantics": str(args.parent_scene_ranking_semantics),
        "parent_mode_temperature": float(args.parent_mode_temperature),
        "anonymous_parent_mode_readout_sha256": (
            anonymous_parent_readout.content_sha256
            if anonymous_parent_readout is not None
            else None
        ),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
        "stage_runtime_seconds": {
            key: {
                "median": float(np.median([
                    float(row["stage_seconds"][key])
                    for row in rows if "stage_seconds" in row
                ])),
                "p90": float(np.quantile([
                    float(row["stage_seconds"][key])
                    for row in rows if "stage_seconds" in row
                ], 0.90)),
            }
            for key in (
                "radio_npz_load",
                "surface_mapper_projection",
                "full_radio_layout_encoding",
                "parent_multimode_scoring",
                "child_joint_posterior_expansion",
                "scene_aggregation_and_set_selection",
            )
            if any("stage_seconds" in row for row in rows)
        },
        "rows": rows,
    }
    _atomic_json(report, summary_path)
    print(
        json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
